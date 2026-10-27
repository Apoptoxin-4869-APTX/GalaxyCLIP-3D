"""Check that a particle selection on the raw snapshot reproduces the GalaxyCLIP-2D maps' selection.

The 2D maps were made with skais-mapper (github.com/phdenzel/skais-mapper). For each halo it
keeps the FoF-group particles inside a cube centred on the central subhalo's SubhaloPos, with
half-width = the central's gas half-mass radius (the map's `extent`), without periodic
wrapping. This script counts the particles in that cube for the unrotated view (rotxy = 0, 0)
and compares with `num_particles` stored in the 2D map metadata, per domain (gas, star, dm).
Domains whose particles sit in unusable snapshot files are skipped.

Counts agree exactly or to within a few particles in 10^5-10^6: particles lying on a cube face
flip with floating-point rounding, and the exact float arithmetic of the July 2025 2D run is not
recoverable from the current skais-mapper code. Differences up to TOLERANCE of the count are
reported as rounding.

    python tools/check_2d_selection.py 100 500 1031 2913 765

Exit code 1 if any count differs by more than TOLERANCE.
"""
from __future__ import annotations

import argparse
import sys
from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np

ROOT = "/cluster/projects/ska/illustris/tng50-1"
ROOT_2D = "/cluster/projects/ska/illustris/tng50-1.2D"
DOMAINS = {"gas": 0, "dm": 1, "star": 4}
TOLERANCE = 1e-4                       # relative count difference accepted as boundary rounding


class Snapshot:
    """Minimal reader for one snapshot laid out as <root>/{snapshots,groupcats}/NNN and offsets/."""

    def __init__(self, root: str, snap: int):
        self.root, self.snap = Path(root), snap
        self.offsets_path = self.root / "offsets" / f"offsets_{snap:03d}.hdf5"
        with h5py.File(self.offsets_path, "r") as f:
            fo = f["FileOffsets/SnapByType"][()]
            # TNG docs say [6, Nc]; TNG files store (Nc, 6)
            self.file_offsets = fo.T if fo.shape[0] == 6 and fo.shape[1] != 6 else fo
            self.fo_group = f["FileOffsets/Group"][()]
            self.fo_subhalo = f["FileOffsets/Subhalo"][()]

    def snap_file(self, i: int) -> Path:
        return self.root / "snapshots" / f"{self.snap:03d}" / f"snap_{self.snap:03d}.{i}.hdf5"

    def catalog_row(self, kind: str, index: int, field: str):
        fo = self.fo_group if kind == "Group" else self.fo_subhalo
        j = int(np.searchsorted(fo, index, side="right") - 1)
        path = self.root / "groupcats" / f"{self.snap:03d}" / f"fof_subhalo_tab_{self.snap:03d}.{j}.hdf5"
        with h5py.File(path, "r") as f:
            return f[kind][field][index - fo[j]]

    def group_start(self, gid: int, ptype: int) -> int:
        with h5py.File(self.offsets_path, "r") as f:
            return int(f["Group/SnapByType"][gid, ptype])

    def load_block(self, ptype: int, start: int, n: int, field: str) -> np.ndarray | None:
        """Rows [start, start+n) of one particle field across file chunks; None if a chunk is unusable."""
        j = int(np.searchsorted(self.file_offsets[:, ptype], start, side="right") - 1)
        off, out = start - int(self.file_offsets[j, ptype]), []
        while n > 0:
            try:
                with h5py.File(self.snap_file(j), "r") as f:
                    k = min(n, int(f["Header"].attrs["NumPart_ThisFile"][ptype]) - off)
                    if k > 0:
                        out.append(f[f"PartType{ptype}/{field}"][off:off + k])
                        n -= k
            except (FileNotFoundError, OSError):
                return None
            j, off = j + 1, 0
        return np.concatenate(out) if out else np.zeros((0, 3))


def _chunk_lo(path: Path) -> int:
    """First gid of a 2D file, e.g. 250709_tng50-1.99.2D.gas.1000.2000.hdf5 -> 1000."""
    return int(path.name.split(".")[-3])


@lru_cache(maxsize=None)
def _unrotated_counts(path: Path, domain: str) -> dict[int, int]:
    """gid -> num_particles of the unrotated map, for one 2D file."""
    with h5py.File(path, "r") as f:
        md = f[f"{domain}/metadata"]
        return {int(md[k].attrs["gid"]): int(md[k].attrs["num_particles"])
                for k in md if tuple(md[k].attrs["rotxy"]) == (0, 0)}


def num_particles_2d(snap: int, domain: str, gid: int) -> int | None:
    """`num_particles` of the unrotated 2D map of one halo. Each 2D file holds the next 1000
    existing gids from its first gid, so search the files starting at or below `gid`."""
    files = sorted(Path(ROOT_2D).glob(f"*_tng50-1.{snap}.2D.{domain}.*.hdf5"), key=_chunk_lo, reverse=True)
    for p in files:
        if _chunk_lo(p) <= gid and gid in (counts := _unrotated_counts(p, domain)):
            return counts[gid]
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("gids", type=int, nargs="+")
    ap.add_argument("--snap", type=int, default=99)
    ap.add_argument("--root", default=ROOT)
    args = ap.parse_args()

    s = Snapshot(args.root, args.snap)
    ok = True
    print(f"{'gid':>5} {'domain':>6} {'2D num_particles':>16} {'cube count':>11}  result")
    for gid in args.gids:
        cen = int(s.catalog_row("Group", gid, "GroupFirstSub"))
        centre = s.catalog_row("Subhalo", cen, "SubhaloPos").astype(np.float64)
        hw = float(s.catalog_row("Subhalo", cen, "SubhaloHalfmassRadType")[0])   # ckpc/h, as positions
        lo, hi = centre - hw, centre + hw
        length = s.catalog_row("Group", gid, "GroupLenType")
        for domain, ptype in DOMAINS.items():
            target = num_particles_2d(args.snap, domain, gid)
            x = s.load_block(ptype, s.group_start(gid, ptype), int(length[ptype]), "Coordinates")
            if target is None or x is None:
                why = "no 2D map" if target is None else "particles in unusable snapshot files"
                print(f"{gid:5d} {domain:>6} {'-':>16} {'-':>11}  skipped ({why})")
                continue
            n = int(((x > lo) & (x < hi)).all(1).sum())
            diff = n - target
            within = abs(diff) <= TOLERANCE * target
            ok &= within
            result = "exact" if diff == 0 else f"{diff:+d} (rounding)" if within else f"{diff:+d} MISMATCH"
            print(f"{gid:5d} {domain:>6} {target:16,d} {n:11,d}  {result}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
