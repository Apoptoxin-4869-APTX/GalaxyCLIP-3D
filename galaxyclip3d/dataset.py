"""PyTorch access to the point-cloud dataset built by tools/build_pointclouds.py and tools/finalize_pointclouds.py.

Each item is one galaxy with four clouds of `budget` points (stars, dm, gas, hi; see pointclouds/sampling.py), a mask
of which modalities are present, and the galaxy's identity (snapshot, halo, subhalo) for pairing with the 2D maps.
Training clouds are drawn fresh every time; validation and test clouds come from the frozen file, so evaluation is the
same every run.

    ds = PointCloudDataset("/cluster/projects/ska/illustris/tng50-1.pointclouds/v1", "train")
    loader = DataLoader(ds, batch_size=64, sampler=MassBalancedSampler(ds), num_workers=8)
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch

from .pointclouds.config import Config
from .pointclouds.sampling import MODALITIES, draw, standardize
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
    def __init__(self, root: str | Path, split: str = "train", radius: float | None = None,
                 gas_required: bool = False, frozen: bool | None = None, rotate: bool | None = None,
                 standardized: bool = True):
        self.root = Path(root)
        self.cfg = Config.load(self.root / "config.json")
        self.radius = self.cfg.train_radius if radius is None else radius
        if self.radius not in self.cfg.radii:
            raise ValueError(f"radius must be one of {self.cfg.radii}")
        with h5py.File(self.root / "index.h5", "r") as f:
            self.index = {k: v[()] for k, v in f["galaxies"].items()}
        usable, self.mask = galaxy_masks(self.index, self.cfg, self.radius)
        keep = usable & (self.index["split"] == SPLITS.index(split))
        if gas_required:
            keep &= self.mask[:, 2]
        self.rows = np.flatnonzero(keep)
        self.frozen = split != "train" if frozen is None else frozen
        self.rotate = not self.frozen if rotate is None else rotate
        self.stats = json.loads((self.root / "feature_stats.json").read_text()) if standardized else None
        self._shards: dict[int, Shard] = {}
        self._frozen = None
        if self.frozen:
            path = self.root / "frozen" / f"{split}_r{self.radius:g}.h5"
            with h5py.File(path, "r") as f:
                pos = {s: i for i, s in enumerate(f["subhalo"][()])}
                self._frozen_row = np.array([pos[s] for s in self.index["subhalo"][self.rows]])
            self._frozen_path = path

    def __len__(self) -> int:
        return len(self.rows)

    def clouds(self, i: int) -> dict[str, np.ndarray]:
        j = self.rows[i]
        if self.frozen:
            if self._frozen is None:                        # opened lazily, inside each worker
                self._frozen = h5py.File(self._frozen_path, "r")
            return {m: self._frozen[m][self._frozen_row[i]] for m in MODALITIES}
        s = int(self.index["shard"][j])
        if s not in self._shards:
            self._shards[s] = Shard(self.root / "shards" / f"shard_{s:04d}.h5")
        row = int(self.index["row"][j])
        parts = {name: self._shards[s].particles(row, name) for name in ("stars", "dm", "gas")}
        radius_kpc = self.radius * float(self.index["rhalf_star_kpc"][j])
        return draw(parts, radius_kpc, float(self.index["vmax"][j]), self.cfg.budget, np.random.default_rng(),
                    rotate=self.rotate)

    def __getitem__(self, i: int) -> dict:
        j = self.rows[i]
        clouds = self.clouds(i)
        if self.stats is not None:
            clouds = standardize(clouds, self.stats)
        item = {m: torch.from_numpy(np.ascontiguousarray(clouds[m])) for m in MODALITIES}
        item["mask"] = torch.from_numpy(self.mask[j].copy())
        item["snapshot"] = self.cfg.snap
        for k in ("subhalo", "group", "halo_has_2d", "central"):
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
