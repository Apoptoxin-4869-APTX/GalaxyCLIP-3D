"""Where the parts of a point-cloud dataset live. A dataset holds one or more snapshots, added one at a time:

    OUT/config.json              settings shared by all snapshots (snap is null here)
    OUT/index.h5                 every galaxy of every snapshot: the shards' galaxy tables plus the shard number
    OUT/snapshots.json           per snapshot: redshift, galaxy counts, when, from where and by which code it was built
    OUT/splits.h5                the split of every split unit, shared by all snapshots, and snapshot 99's SubhaloGrNr
                                 (see lineage.py)
    OUT/feature_stats.json       standardization; fixed when the first snapshot is finalized
    OUT/snap_NNN/config.json     full settings of that snapshot's build
    OUT/snap_NNN/selection.h5    its galaxies, splits and shard plan
    OUT/snap_NNN/lineage.h5      each galaxy's SubLink main branch from the TNG API
    OUT/snap_NNN/shards/shard_SSSS.h5
    OUT/snap_NNN/index.h5        its galaxies (the part of OUT/index.h5 from this snapshot)
    OUT/snap_NNN/frozen/{val,test}_r<radius>.h5
"""
from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np


def snapshot_dir(out: str | Path, snap: int) -> Path:
    return Path(out) / f"snap_{int(snap):03d}"


def shard_path(out: str | Path, snap: int, shard: int) -> Path:
    return snapshot_dir(out, snap) / "shards" / f"shard_{int(shard):04d}.h5"


def frozen_path(out: str | Path, snap: int, split: str, radius: float) -> Path:
    return snapshot_dir(out, snap) / "frozen" / f"{split}_r{radius:g}.h5"


def snapshots_in(out: str | Path, need: str = "selection.h5") -> list[int]:
    """Snapshots of the dataset whose folder holds `need`, in increasing order."""
    found = []
    for p in Path(out).glob("snap_[0-9][0-9][0-9]"):
        if p.is_dir() and (p / need).exists():
            found.append(int(p.name[5:]))
    return sorted(found)


def read_table(path: str | Path, group: str = "galaxies") -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as f:
        return {k: v[()] for k, v in f[group].items()}


def write_table(path: str | Path, table: dict[str, np.ndarray], attrs: dict, group: str | None = "galaxies") -> None:
    """Written to a .part file and renamed, so a killed job never leaves half a file under the real name."""
    path = Path(path)
    tmp = path.with_name(path.name + ".part")
    with h5py.File(tmp, "w") as f:
        g = f.create_group(group) if group else f
        for k, v in table.items():
            g.create_dataset(k, data=np.asarray(v))
        for k, v in attrs.items():
            f.attrs[k] = v
    os.replace(tmp, path)
