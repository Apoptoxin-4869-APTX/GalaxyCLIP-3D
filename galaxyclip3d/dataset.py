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

    ds = PointCloudDataset("/cluster/projects/ska/illustris/tng50-1.pointclouds/v3", "train")             # all snapshots
    ds = PointCloudDataset("/cluster/projects/ska/illustris/tng50-1.pointclouds/v3", "val", snapshots=[99])
    loader = DataLoader(ds, batch_size=64, sampler=MassBalancedSampler(ds), num_workers=8)
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch

from .pointclouds.config import Config
from .pointclouds.layout import frozen_path, read_table, shard_path
from .pointclouds.sampling import MODALITIES, draw, orient, standardize
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


class PointCloudDataset(torch.utils.data.Dataset):
    def __init__(self, root: str | Path, split: str = "train", radius: float | None = None, snapshots=None,
                 gas_required: bool = False, frozen: bool | None = None, rotation: str = "independent",
                 mirror: bool = True, standardized: bool = True, budget: int | None = None, common_radii=None,
                 draw_seed: int | None = None):
        self.root = Path(root)
        self.cfg = Config.load(self.root / "config.json")
        self.radius = self.cfg.train_radius if radius is None else radius
        self.budget = self.cfg.budget if budget is None else int(budget)
        for r in [self.radius, *(common_radii or ())]:
            if r not in self.cfg.radii:
                raise ValueError(f"radius must be one of {self.cfg.radii}, got {r}")
        if self.budget < 1:
            raise ValueError(f"budget must be positive, got {self.budget}")
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
            clouds = {m: self._frozen[snap][m][self._frozen_row[i]] for m in MODALITIES}
            rng = np.random.default_rng([self.cfg.seed, snap, int(self.index["subhalo"][j]), 11])
            return orient(clouds, rng, self.rotation, self.mirror)
        key = (snap, int(self.index["shard"][j]))
        if key not in self._shards:
            self._shards[key] = Shard(shard_path(self.root, *key))
        row = int(self.index["row"][j])
        parts = {name: self._shards[key].particles(row, name) for name in ("stars", "dm", "gas")}
        radius_kpc = self.radius * float(self.index["rhalf_star_kpc"][j])
        rng = (np.random.default_rng() if self.draw_seed is None else
               np.random.default_rng([self.draw_seed, snap, int(self.index["subhalo"][j]), 7]))
        clouds = draw(parts, radius_kpc, float(self.index["vmax"][j]), self.budget, rng, rotate=False)
        return orient(clouds, rng, self.rotation, self.mirror)

    def __getitem__(self, i: int) -> dict:
        j = self.rows[i]
        clouds = self.clouds(i)
        if self.stats is not None:
            clouds = standardize(clouds, self.stats)
        item = {m: torch.from_numpy(np.ascontiguousarray(clouds[m])) for m in MODALITIES}
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
