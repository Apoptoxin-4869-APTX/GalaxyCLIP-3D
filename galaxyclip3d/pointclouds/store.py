"""Shard files: the particles of a group of galaxies, concatenated per type with offsets, written atomically."""
from __future__ import annotations

import json
import os
from pathlib import Path

import h5py
import numpy as np

from .extract import FIELDS


def write_shard(path: str | Path, galaxies: dict[str, np.ndarray], particles: list[dict], attrs: dict) -> None:
    """galaxies: column -> array (one entry per galaxy); particles: extract() output per galaxy, same order."""
    path = Path(path)
    tmp = path.with_name(path.name + ".part")
    with h5py.File(tmp, "w") as f:
        g = f.create_group("galaxies")
        for k, v in galaxies.items():
            g.create_dataset(k, data=np.asarray(v))
        for name, fields in FIELDS.items():
            grp = f.create_group(name)
            sizes = [len(p[name]["pos"]) for p in particles]
            grp.create_dataset("offsets", data=np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64))
            for fld, unit in fields.items():
                arr = np.concatenate([p[name][fld] for p in particles])
                chunks = (min(65536, len(arr)),) + arr.shape[1:] if len(arr) else None
                grp.create_dataset(fld, data=arr, chunks=chunks).attrs["unit"] = unit
        for k, v in attrs.items():
            f.attrs[k] = v if isinstance(v, (str, int, float)) else json.dumps(v)
    os.replace(tmp, path)


class Shard:
    """Read access to one shard. The file opens on first use, so a Shard can be handed to forked loader workers."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._file = None

    @property
    def file(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.path, "r")
        return self._file

    def galaxies(self) -> dict[str, np.ndarray]:
        return {k: v[()] for k, v in self.file["galaxies"].items()}

    def particles(self, row: int, name: str, fields=None) -> dict[str, np.ndarray]:
        grp = self.file[name]
        a, b = grp["offsets"][row:row + 2]
        return {f: grp[f][a:b] for f in (fields or FIELDS[name])}

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
