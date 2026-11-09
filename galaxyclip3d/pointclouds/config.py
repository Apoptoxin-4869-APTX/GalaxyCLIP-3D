"""Settings of a point-cloud build. They are saved next to the output, so every dataset records how it was made.

A dataset can hold several snapshots, added one at a time. The dataset folder's config.json holds the settings all of
them share (snap is null there); each snapshot folder (snap_NNN/config.json) holds the full settings of its build.
Only the fields in PER_SNAPSHOT may differ between the snapshots of one dataset.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

PER_SNAPSHOT = ("root", "out", "snap", "chunk_rows")    # where the raw data is and how it is read; not the content


@dataclass(frozen=True)
class Config:
    root: str = "/cluster/projects/ska/illustris/tng50-1.new"      # snapshot copy laid out as snapshots/NNN, groupcats/NNN
    out: str = "/cluster/projects/ska/illustris/tng50-1.pointclouds/v3"
    snap: int | None = 99                   # snapshot of this build; null in a dataset's shared config.json
    budget: int = 2048                      # points per modality in a cloud
    min_particles: int = 4096               # stars and dark matter a galaxy needs (gas: else the gas mask is off)
    min_rhalf_star_kpc: float = 0.576       # two softening lengths of TNG50-1 at z <= 1 (0.288 physical kpc)
    radii: tuple = (2.0, 3.0, 5.0, 7.5, 10.0)   # in stellar half-mass radii; particle counts are stored at each
    extract_radius: float = 10.0            # stored sphere, in stellar half-mass radii (the largest of radii)
    train_radius: float = 5.0               # default sphere for training, in stellar half-mass radii (one of radii)
    hi_min_eff: float = 512.0               # HI modality on when the HI weights' effective number of cells reaches this
    cap: int = 262144                       # stored particles per type and galaxy; above it a weighted subset
    split: tuple = (0.8, 0.1, 0.1)          # train / val / test share of galaxies; whole split units go to one split
    seed: int = 4869
    shard_size: int = 64                    # galaxies per shard file
    chunk_rows: int = 8_000_000             # rows read at once from a snapshot block

    def __post_init__(self):
        if self.extract_radius != max(self.radii) or self.train_radius not in self.radii:
            raise ValueError("extract_radius must be the largest of radii and train_radius one of them")
        if self.min_particles < self.budget:
            raise ValueError("min_particles must be at least the budget")

    def shared(self) -> dict:
        """The settings every snapshot of one dataset must have in common."""
        return {k: v for k, v in dataclasses.asdict(self).items() if k not in PER_SNAPSHOT}

    def dataset(self) -> "Config":
        """The dataset-level config: these settings without a snapshot."""
        return dataclasses.replace(self, snap=None)

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json() + "\n")

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown settings: {sorted(unknown)}")
        return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.items()})

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        return cls.from_dict(json.loads(Path(path).read_text()))
