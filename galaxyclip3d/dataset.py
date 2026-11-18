"""PyTorch access to the point-cloud dataset built by tools/build_selection.py, build_pointclouds.py and
finalize_pointclouds.py.

Each item is one galaxy at one snapshot with four clouds of `budget` points (stars, dm, gas, hi; see
pointclouds/sampling.py), a mask of which modalities are present, and the galaxy's identity (snapshot, redshift, halo,
subhalo, its z = 0 descendant) for pairing with the 2D maps or across snapshots. Training clouds are drawn fresh every
time; validation and test clouds come from each snapshot's frozen file, so evaluation is the same every run. Splits are
shared across snapshots (pointclouds/lineage.py): a galaxy's versions at different snapshots are always in one split.

Orientation (`rotation`): by default every modality gets its own random rotation, and with `mirror` a reflection half
of the time, so the four clouds cannot be matched by which way the galaxy points (pointclouds/sampling.py, orient).
Training draws a new one every time; frozen clouds get a fixed one per galaxy and modality, seeded by (dataset seed,
snapshot, subhalo, 11), so evaluation stays identical. "shared" (one rotation for all four, the old behaviour) and
"none" (frozen clouds as stored, in the simulation's frame) exist for controls.

For ablations: `budget` draws another number of points per cloud than the dataset's (frozen clouds then come from
frozen/<split>_r<radius>_p<budget>.h5, written by finalize_pointclouds.py --budget), and `common_radii` keeps only the
galaxies usable at every one of those spheres, with a modality on only if it is on at all of them, so runs at
different spheres train and validate on the same galaxies.

For scoring after training (linear probes need features of the training galaxies that do not change between runs):
`draw_seed` makes every fresh draw reproducible, one fixed draw and orientation per galaxy, seeded by (draw_seed,
snapshot, subhalo, 7). Training leaves it unset, so it draws anew every time.

Dark-matter sphere (`dm_aperture`, a control): by default every cloud is cut at `radius` stellar half-mass radii and
scaled by that radius, so the dark-matter cloud's shape carries the stars' size. "vmax" cuts the dark matter at the
galaxy's R_vmax instead (the radius where its circular velocity peaks, a property of the halo's dynamics) and scales it
by R_vmax; stars, gas and HI keep the stellar sphere. It needs r_vmax_kpc and n_dm_rvmax in snap_NNN/targets.h5
(tools/build_targets.py) and keeps only galaxies whose R_vmax lies inside the stored particles (extract_radius) with
at least min_particles dark-matter particles inside it. Frozen files hold stellar-sphere clouds, so validation and test
clouds are then drawn fresh, seeded by (dataset seed, snapshot, subhalo, 7), the same every time. `rvmax_subset` keeps
those same galaxies with any dark-matter sphere: a stellar-sphere run on them is the matched control for a vmax run
(its validation and test clouds stay the frozen ones).

Stage E settings: `min_particles` raises the particle count a galaxy needs inside the sphere above the dataset's own
(4,096 stars and 4,096 dark-matter particles; the gas modality needs as many gas cells, else it is masked). Lower values
are refused, since galaxies with fewer particles were never extracted. Validation and test galaxies keep their frozen
clouds, so the remaining ones are scored on exactly the clouds as before. `modalities` draws and returns only those
modalities; the mask keeps all four columns, the others always False. Frozen clouds of the kept modalities, rotations
included, are exactly those of a dataset with all four (orient draws a matrix for every modality).

    ds = PointCloudDataset("/cluster/projects/ska/illustris/tng50-1.pointclouds/v3", "train")             # all snapshots
    ds = PointCloudDataset("/cluster/projects/ska/illustris/tng50-1.pointclouds/v3", "val", snapshots=[99])
    loader = DataLoader(ds, batch_size=64, sampler=MassBalancedSampler(ds), num_workers=8)
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from .pointclouds.config import Config
from .pointclouds.layout import frozen_path, read_table, shard_path, snapshot_dir
from .pointclouds.sampling import MODALITIES, SOURCE, draw, orient, standardize
from .pointclouds.select import SPLITS
from .pointclouds.store import Shard


def galaxy_masks(index: dict[str, np.ndarray], cfg: Config, radius: float) -> tuple[np.ndarray, np.ndarray]:
    """(usable, modality mask (galaxies, 4)) at a sphere of `radius` stellar half-mass radii."""
    k = f"r{radius:g}"
    usable = (index[f"n_stars_{k}"] >= cfg.min_particles) & (index[f"n_dm_{k}"] >= cfg.min_particles)
    gas_on = index[f"n_gas_{k}"] >= cfg.min_particles
    hi_on = index[f"hi_eff_{k}"] >= cfg.hi_min_eff
    ones = np.ones(len(usable), dtype=bool)
    return usable, np.stack([ones, ones, gas_on, hi_on], axis=1)


DM_APERTURES = ("stellar", "vmax")


def dm_vmax_radius(root: Path, index: dict[str, np.ndarray], cfg: Config) -> np.ndarray:
    """Per galaxy (index order): R_vmax in kpc where a dark-matter sphere of that radius can be drawn (inside the
    stored particles, with at least cfg.min_particles dark-matter particles), else NaN. From snap_NNN/targets.h5."""
    out = np.full(len(index["subhalo"]), np.nan)
    for snap in np.unique(index["snapshot"]):
        path = snapshot_dir(root, int(snap)) / "targets.h5"
        t = read_table(path) if path.exists() else {}
        if "r_vmax_kpc" not in t:
            raise FileNotFoundError(f"{path} has no r_vmax_kpc: run tools/build_targets.py --out {root} --snap {snap}")
        where = {int(s): i for i, s in enumerate(t["subhalo"])}
        rows = np.flatnonzero(index["snapshot"] == snap)
        k = np.array([where.get(int(index["subhalo"][j]), -1) for j in rows], dtype=np.int64)
        r = np.where(k >= 0, t["r_vmax_kpc"][np.maximum(k, 0)], np.nan)
        n = np.where(k >= 0, t["n_dm_rvmax"][np.maximum(k, 0)], np.nan)
        ok = (r > 0) & (r <= cfg.extract_radius * index["rhalf_star_kpc"][rows]) & (n >= cfg.min_particles)
        out[rows] = np.where(ok, r, np.nan)
    return out


class PointCloudDataset(torch.utils.data.Dataset):
    def __init__(self, root: str | Path, split: str = "train", radius: float | None = None, snapshots=None,
                 gas_required: bool = False, frozen: bool | None = None, rotation: str = "independent",
                 mirror: bool = True, standardized: bool = True, budget: int | None = None, common_radii=None,
                 draw_seed: int | None = None, dm_aperture: str = "stellar", rvmax_subset: bool = False,
                 min_particles: int | None = None, modalities=None):
        self.root = Path(root)
        self.cfg = Config.load(self.root / "config.json")
        if min_particles is not None and int(min_particles) != self.cfg.min_particles:
            if int(min_particles) < self.cfg.min_particles:
                raise ValueError(f"min_particles must be at least the dataset's {self.cfg.min_particles} (galaxies with "
                                 f"fewer particles were never extracted), got {min_particles}")
            self.cfg = dataclasses.replace(self.cfg, min_particles=int(min_particles))
        given = list(MODALITIES) if modalities is None else list(modalities)
        if not given or set(given) - set(MODALITIES) or len(set(given)) != len(given):
            raise ValueError(f"modalities must be distinct names from {MODALITIES}, got {given}")
        self.modalities = tuple(m for m in MODALITIES if m in given)
        if dm_aperture not in DM_APERTURES:
            raise ValueError(f"dm_aperture must be one of {DM_APERTURES}, got {dm_aperture!r}")
        self.dm_aperture, self.rvmax_subset = dm_aperture, rvmax_subset
        self.radius = self.cfg.train_radius if radius is None else radius
        self.budget = self.cfg.budget if budget is None else int(budget)
        for r in [self.radius, *(common_radii or ())]:
            if r not in self.cfg.radii:
                raise ValueError(f"radius must be one of {self.cfg.radii}, got {r}")
        if not 1 <= self.budget <= self.cfg.min_particles:
            raise ValueError(f"budget must be between 1 and min_particles ({self.cfg.min_particles}), got {self.budget}")
        if gas_required and "gas" not in self.modalities:
            raise ValueError("gas_required needs the gas modality")
        self.index = read_table(self.root / "index.h5")
        self.snapshots = sorted({int(s) for s in self.index["snapshot"]}) if snapshots is None else \
            sorted(int(s) for s in snapshots)
        absent = set(self.snapshots) - {int(s) for s in self.index["snapshot"]}
        if absent:
            raise ValueError(f"snapshots {sorted(absent)} are not in {self.root}")
        usable, self.mask = galaxy_masks(self.index, self.cfg, self.radius)
        for r in common_radii or ():
            u, m = galaxy_masks(self.index, self.cfg, r)
            usable, self.mask = usable & u, self.mask & m
        self.mask[:, [k for k, m in enumerate(MODALITIES) if m not in self.modalities]] = False   # left out = absent
        if dm_aperture == "vmax" or rvmax_subset:
            self.r_dm = dm_vmax_radius(self.root, self.index, self.cfg)
            usable = usable & np.isfinite(self.r_dm)
        if dm_aperture == "vmax":
            if frozen:
                raise ValueError("frozen files hold stellar-sphere clouds; dm_aperture='vmax' draws its clouds")
            frozen = False
            if split != "train" and draw_seed is None:
                draw_seed = self.cfg.seed                   # fixed evaluation clouds, the same every run
        keep = usable & (self.index["split"] == SPLITS.index(split)) & np.isin(self.index["snapshot"], self.snapshots)
        if gas_required:
            keep &= self.mask[:, 2]
        self.rows = np.flatnonzero(keep)
        self.frozen = split != "train" if frozen is None else frozen
        self.rotation, self.mirror = rotation, mirror
        self.draw_seed = draw_seed
        self.stats = json.loads((self.root / "feature_stats.json").read_text()) if standardized else None
        self._shards: dict[tuple[int, int], Shard] = {}
        self._frozen: dict[int, h5py.File] = {}
        if self.frozen:
            self._frozen_paths, where = {}, {}
            other = self.budget if self.budget != self.cfg.budget else None
            for snap in self.snapshots:
                path = frozen_path(self.root, snap, split, self.radius, other)
                if not path.exists():
                    raise FileNotFoundError(f"{path} missing: run tools/finalize_pointclouds.py --snap {snap} "
                                            f"--radius {self.radius:g}" + (f" --budget {other}" if other else "") +
                                            " --frozen-only")
                self._frozen_paths[snap] = path
                with h5py.File(path, "r") as f:
                    if f["stars"].shape[1] != self.budget:
                        raise ValueError(f"{path} holds {f['stars'].shape[1]} points per cloud, not {self.budget}")
                    where.update({(snap, int(s)): i for i, s in enumerate(f["subhalo"][()])})
            self._frozen_row = np.array([where[(int(self.index["snapshot"][j]), int(self.index["subhalo"][j]))]
                                         for j in self.rows], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.rows)

    def clouds(self, i: int) -> dict[str, np.ndarray]:
        j = self.rows[i]
        snap = int(self.index["snapshot"][j])
        if self.frozen:
            if snap not in self._frozen:                    # opened lazily, inside each worker
                self._frozen[snap] = h5py.File(self._frozen_paths[snap], "r")
            clouds = {m: self._frozen[snap][m][self._frozen_row[i]] for m in self.modalities}
            rng = np.random.default_rng([self.cfg.seed, snap, int(self.index["subhalo"][j]), 11])
            return orient(clouds, rng, self.rotation, self.mirror)
        key = (snap, int(self.index["shard"][j]))
        if key not in self._shards:
            self._shards[key] = Shard(shard_path(self.root, *key))
        row = int(self.index["row"][j])
        parts = {name: self._shards[key].particles(row, name) for name in ("stars", "dm", "gas")
                 if name in {SOURCE[m] for m in self.modalities}}
        radius_kpc = self.radius * float(self.index["rhalf_star_kpc"][j])
        if self.dm_aperture == "vmax":
            radius_kpc = {**{m: radius_kpc for m in MODALITIES}, "dm": float(self.r_dm[j])}
        rng = (np.random.default_rng() if self.draw_seed is None else
               np.random.default_rng([self.draw_seed, snap, int(self.index["subhalo"][j]), 7]))
        clouds = draw(parts, radius_kpc, float(self.index["vmax"][j]), self.budget, rng, rotate=False,
                      modalities=self.modalities)
        return orient(clouds, rng, self.rotation, self.mirror)

    def __getitem__(self, i: int) -> dict:
        j = self.rows[i]
        clouds = self.clouds(i)
        if self.stats is not None:
            clouds = standardize(clouds, self.stats)
        item = {m: torch.from_numpy(np.ascontiguousarray(clouds[m])) for m in self.modalities}
        item["mask"] = torch.from_numpy(self.mask[j].copy())
        for k in ("snapshot", "redshift", "subhalo", "group", "halo_has_2d", "central", "z0_subhalo", "mstar"):
            item[k] = self.index[k][j].item()
        return item


class MassBalancedSampler(torch.utils.data.Sampler):
    """Indices drawn evenly across bins of log stellar mass (with replacement), len(dataset) per epoch."""

    def __init__(self, dataset: PointCloudDataset, bins: int = 6, seed: int = 0):
        lm = np.log10(np.maximum(dataset.index["mstar"][dataset.rows], 1.0))
        edges = np.linspace(lm.min(), lm.max(), bins + 1)[1:-1]
        which = np.digitize(lm, edges)
        self.members = [np.flatnonzero(which == b) for b in range(bins) if (which == b).any()]
        self.n, self.seed, self.epoch = len(lm), seed, 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.n

    def __iter__(self):
        rng = np.random.default_rng([self.seed, self.epoch])
        b = rng.integers(len(self.members), size=self.n)
        return iter(int(rng.choice(self.members[k])) for k in b)
