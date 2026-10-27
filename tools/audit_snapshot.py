"""Audit the raw TNG50-1 snapshot on disk before extracting 3D point clouds.

Finds snapshot files that are missing or unreadable (e.g. truncated downloads), maps them onto
halos through the offsets file (particles are stored in FoF-group order, so a missing file
removes contiguous blocks of specific halos), and reports which halos with 2D maps have every
particle on disk. Optionally writes a per-halo table for those halos.

    python tools/audit_snapshot.py --csv data/snap099_halo_availability.csv

Needs numpy + h5py. On TNG50-1 it reads the group catalog in ~1 min and needs ~3 GB of RAM.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path

import h5py
import numpy as np

ROOT = "/cluster/projects/ska/illustris/tng50-1"
BLOCKS_2D = Path(__file__).resolve().parents[1] / "data" / "block_to_gid_2d.json"
TYPES = {0: "gas", 1: "dm", 4: "stars", 5: "bh"}
# bytes per particle for the fields a cutout would request (positions are float64)
CUTOUT_BYTES = {0: 64, 1: 44, 4: 60, 5: 120}


def snapshot_files(root: str, snap: int) -> tuple[list[int], list[int], int, dict]:
    """Readable file numbers, unusable file numbers (missing or broken), total, header of a good file."""
    sdir = Path(root) / "snapshots" / f"{snap:03d}"
    present = sorted(int(m.group(1)) for f in os.listdir(sdir)
                     if (m := re.match(rf"snap_{snap:03d}\.(\d+)\.hdf5$", f)))
    good, broken, header = [], [], None
    for i in present:
        try:
            with h5py.File(sdir / f"snap_{snap:03d}.{i}.hdf5", "r") as f:
                n = f["Header"].attrs["NumPart_ThisFile"]
                for t in TYPES:
                    if n[t]:
                        f[f"PartType{t}/ParticleIDs"][-1]       # forces a read at the end of the file
                if header is None:
                    header = dict(f["Header"].attrs)
            good.append(i)
        except OSError as e:
            broken.append(i)
            print(f"  unreadable: snap_{snap:03d}.{i}.hdf5 ({str(e)[:90]})")
    nfiles = int(header["NumFilesPerSnapshot"])
    missing = sorted(set(range(nfiles)) - set(present))
    return good, sorted(missing + broken), nfiles, header


def load_groupcat(root: str, snap: int, fields: dict[str, list[str]]) -> dict[str, np.ndarray]:
    """Concatenate the requested Group/Subhalo fields over all group-catalog files."""
    gdir = Path(root) / "groupcats" / f"{snap:03d}"
    paths = sorted(gdir.glob(f"fof_subhalo_tab_{snap:03d}.*.hdf5"), key=lambda p: int(p.name.split(".")[1]))
    parts: dict[str, list[np.ndarray]] = {k: [] for ks in fields.values() for k in ks}
    for p in paths:
        with h5py.File(p, "r") as f:
            for grp, keys in fields.items():
                n = f["Header"].attrs["Ngroups_ThisFile" if grp == "Group" else "Nsubgroups_ThisFile"]
                if n:
                    for k in keys:
                        parts[k].append(f[grp][k][()])
    return {k: np.concatenate(v) for k, v in parts.items()}


def missing_counter(file_offsets: np.ndarray, ntot: np.ndarray, unusable: list[int]):
    """n_missing(t, start, length): particles of type t in [start, start+length) that sit in unusable files."""
    nfiles = file_offsets.shape[0]
    bad = np.zeros(nfiles, dtype=bool)
    bad[unusable] = True
    bounds, cum = {}, {}
    for t in TYPES:
        b = np.append(file_offsets[:, t].astype(np.int64), ntot[t])
        bounds[t], cum[t] = b, np.concatenate([[0], np.cumsum(np.diff(b) * bad)])

    def n_missing(t: int, start: np.ndarray, length: np.ndarray) -> np.ndarray:
        def below(x):
            j = np.clip(np.searchsorted(bounds[t], x, side="right") - 1, 0, nfiles - 1)
            return cum[t][j] + bad[j] * (x - bounds[t][j])
        return below(start + length) - below(start)

    return n_missing, {t: int(cum[t][-1]) for t in TYPES}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--snap", type=int, default=99)
    ap.add_argument("--csv", type=Path, help="write the per-halo table for the 2D halos here")
    args = ap.parse_args()

    good, unusable, nfiles, hd = snapshot_files(args.root, args.snap)
    ntot = hd["NumPart_Total"].astype(np.int64) + (hd["NumPart_Total_HighWord"].astype(np.int64) << 32)
    a, h, box = float(hd["Time"]), float(hd.get("HubbleParam", 0.6774)), float(hd["BoxSize"])
    print(f"snapshot {args.snap}: {len(good)} readable / {nfiles} files; unusable {len(unusable)}: {unusable}")

    with h5py.File(Path(args.root) / "offsets" / f"offsets_{args.snap:03d}.hdf5", "r") as f:
        fo = f["FileOffsets/SnapByType"][()]
        goff = f["Group/SnapByType"][()].astype(np.int64)
    if fo.shape[0] == 6 and fo.shape[1] != 6:       # TNG docs say [6, Nc]; TNG files store (Nc, 6)
        fo = fo.T
    n_missing, missing_tot = missing_counter(fo, ntot, unusable)
    for t, name in TYPES.items():
        print(f"  {name:5s}: {missing_tot[t]:>14,} of {ntot[t]:>14,} particles unreadable ({missing_tot[t] / ntot[t]:.1%})")

    cat = load_groupcat(args.root, args.snap, {
        "Group": ["GroupLenType", "GroupFirstSub"],
        "Subhalo": ["SubhaloHalfmassRadType", "SubhaloPos"]})
    glen = cat["GroupLenType"].astype(np.int64)
    gmiss = np.zeros_like(glen)
    for t in TYPES:
        gmiss[:, t] = n_missing(t, goff[:, t], glen[:, t])
    hit = (gmiss[:, list(TYPES)] > 0).any(1)
    print(f"FoF groups with unreadable particles: {hit.sum():,} of {len(glen):,}")

    # ---------------------------------------------------------------- halos with 2D maps
    gids = np.array(sorted({g for s, g in json.load(open(BLOCKS_2D))["blocks"] if s == args.snap}))
    if not gids.size:
        return
    cen = cat["GroupFirstSub"][gids]
    complete = (gmiss[gids][:, list(TYPES)] == 0).all(1)
    print(f"\nhalos with 2D maps at snapshot {args.snap}: {gids.size}; complete on disk: {complete.sum()}")
    for t, name in TYPES.items():
        print(f"  {name:5s}: {(gmiss[gids, t] > 0).sum():5d} halos incomplete")
    edges = [0, 100, 500, 1000, 2000, gids.max() + 1]
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (gids >= lo) & (gids < hi)
        print(f"  gid {lo:4d}-{hi - 1:4d}: {complete[s].sum():4d} / {s.sum():4d} complete")
    total_gb = 0.0
    print("to complete them with per-halo API cutouts (whole type for each incomplete halo):")
    for t, name in TYPES.items():
        inc = gmiss[gids, t] > 0
        n = glen[gids[inc], t].sum()
        total_gb += n * CUTOUT_BYTES[t] / 1e9
        print(f"  {name:5s}: {inc.sum():5d} halos, {n / 1e6:8.1f} M particles, ~{n * CUTOUT_BYTES[t] / 1e9:5.1f} GB")
    print(f"  total ~{total_gb:.0f} GB, vs ~{len(unusable) * 4.3:.0f} GB to re-download the unusable files")

    # 2D cube: half-width = central's gas half-mass radius, centred on its SubhaloPos, no periodic wrap
    hw = cat["SubhaloHalfmassRadType"][cen, 0]
    pos = cat["SubhaloPos"][cen]
    crosses = ((pos - hw[:, None] < 0) | (pos + hw[:, None] > box)).any(1)
    print(f"2D cubes that cross the periodic box edge (skais-mapper does not wrap): {crosses.sum()}")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["gid", "central_subhalo", "half_width_kpc", "complete", "crosses_box_edge"]
                       + [f"n_{n}" for n in TYPES.values()] + [f"missing_{n}" for n in TYPES.values()])
            for i, g in enumerate(gids):
                w.writerow([g, cen[i], f"{hw[i] * a / h:.3f}", int(complete[i]), int(crosses[i])]
                           + [glen[g, t] for t in TYPES] + [gmiss[g, t] for t in TYPES])
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
