"""Read a TNG snapshot copy laid out as
<root>/snapshots/NNN/snap_NNN.i.hdf5, <root>/groupcats/NNN/fof_subhalo_tab_NNN.i.hdf5, <root>/offsets/offsets_NNN.hdf5.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import h5py
import numpy as np

from .physics import Units


class Snapshot:
    def __init__(self, root: str | Path, snap: int):
        self.root, self.snap = Path(root), snap
        self.offsets_path = self.root / "offsets" / f"offsets_{snap:03d}.hdf5"
        with h5py.File(self.snap_file(0), "r") as f:
            self.header = dict(f["Header"].attrs)
        self.units = Units.from_header(self.header)
        self.redshift = float(self.header["Redshift"])
        with h5py.File(self.offsets_path, "r") as f:
            fo = f["FileOffsets/SnapByType"][()]
        self.file_offsets = fo.T if fo.shape[0] == 6 and fo.shape[1] != 6 else fo     # (files, 6); TNG stores (files, 6)

    def snap_file(self, i: int) -> Path:
        return self.root / "snapshots" / f"{self.snap:03d}" / f"snap_{self.snap:03d}.{i}.hdf5"

    def catalog(self, group_fields: list[str], subhalo_fields: list[str]) -> dict[str, np.ndarray]:
        """Requested Group/Subhalo fields, concatenated over all group-catalog files."""
        return read_catalog(self.root, self.snap, group_fields, subhalo_fields, expect_redshift=self.redshift)

    def subhalo_offsets(self, ids: np.ndarray) -> np.ndarray:
        """First snapshot row of each subhalo's particles, per type: (len(ids), 6)."""
        ids = np.asarray(ids, dtype=np.int64)
        order = np.argsort(ids)
        with h5py.File(self.offsets_path, "r") as f:
            rows = f["Subhalo/SnapByType"][ids[order]]                 # h5py needs increasing indices
        out = np.empty_like(rows)
        out[order] = rows
        return out

    def blocks(self, ptype: int, start: int, n: int, fields: list[str],
               chunk_rows: int = 8_000_000) -> Iterator[tuple[int, dict[str, np.ndarray]]]:
        """Rows [start, start + n) of one particle type, in chunks that may span files.

        Yields (offset of the chunk within the block, {field: array})."""
        done = 0
        j = int(np.searchsorted(self.file_offsets[:, ptype], start, side="right") - 1)
        off = start - int(self.file_offsets[j, ptype])
        while done < n:
            with h5py.File(self.snap_file(j), "r") as f:
                in_file = int(f["Header"].attrs["NumPart_ThisFile"][ptype])
                while done < n and off < in_file:
                    k = min(n - done, in_file - off, chunk_rows)
                    yield done, {fld: f[f"PartType{ptype}/{fld}"][off:off + k] for fld in fields}
                    done, off = done + k, off + k
            j, off = j + 1, 0


def read_catalog(root: str | Path, snap: int, group_fields: list[str], subhalo_fields: list[str],
                 expect_redshift: float | None = None) -> dict[str, np.ndarray]:
    """Group/Subhalo fields of a snapshot's group catalog, concatenated over its files; needs no particle files.
    With expect_redshift, refuses a catalog of another redshift (a catalog and particles from different snapshots)."""
    gdir = Path(root) / "groupcats" / f"{snap:03d}"
    paths = sorted(gdir.glob(f"fof_subhalo_tab_{snap:03d}.*.hdf5"), key=lambda p: int(p.name.split(".")[1]))
    if not paths:
        raise FileNotFoundError(f"no group catalog files in {gdir}")
    parts: dict[str, list[np.ndarray]] = {k: [] for k in group_fields + subhalo_fields}
    for i, p in enumerate(paths):
        with h5py.File(p, "r") as f:
            z = f["Header"].attrs.get("Redshift") if i == 0 and expect_redshift is not None else None
            if z is not None and abs(float(z) - expect_redshift) > 1e-6 * (1 + float(z)):
                raise ValueError(f"group catalog {p.name} is at z = {float(z):g} but the snapshot at z = "
                                 f"{expect_redshift:g}: they are not the same snapshot")
            for grp, keys, count in (("Group", group_fields, "Ngroups_ThisFile"),
                                     ("Subhalo", subhalo_fields, "Nsubgroups_ThisFile")):
                if f["Header"].attrs[count]:
                    for k in keys:
                        parts[k].append(f[grp][k][()])
    return {k: np.concatenate(v) for k, v in parts.items()}
