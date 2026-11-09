"""Add one snapshot to a point-cloud dataset: its galaxies, their train/val/test split and the shard plan (step 1 of 3).

    PYTHONPATH=pylibs python3 -s tools/build_selection.py --config configs/pc_v3.json [--snap 91] [--root DIR]

The dataset is the config's `out` folder, created on first use. Writes OUT/config.json (settings shared by all
snapshots), OUT/snap_NNN/config.json, OUT/snap_NNN/selection.h5 (one entry per galaxy; see galaxyclip3d/pointclouds/
select.py) and OUT/snap_NNN/lineage.h5, and records new split units in OUT/splits.h5.

Splits follow the galaxies through time (galaxyclip3d/pointclouds/lineage.py): each galaxy's SubLink main branch is
fetched from the TNG API (key in ~/.tng_api_key; about 5,000 small requests, a few minutes) and the z = 0 halo it ends
in decides its split, so a galaxy's versions at different snapshots always share one. The first time, snapshot 99's
SubhaloGrNr is copied into OUT/splits.h5, so later snapshots can be linked after snapshot 99's raw data is deleted.

Only root, snap and chunk_rows may differ from the dataset's settings. Running it again for the same snapshot changes
nothing (branches are cached); a selection that would differ from an existing one is refused.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from galaxyclip3d.pointclouds import lineage                                   # noqa: E402
from galaxyclip3d.pointclouds.config import Config                             # noqa: E402
from galaxyclip3d.pointclouds.layout import read_table, snapshot_dir, write_table  # noqa: E402
from galaxyclip3d.pointclouds.select import SPLITS, select, shard_plan         # noqa: E402
from galaxyclip3d.pointclouds.snapshot import Snapshot, read_catalog           # noqa: E402


def differing(a: dict, b: dict) -> list[str]:
    return [k for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]


def load_branches_json(path: Path) -> dict[int, np.ndarray]:
    """{subhalo: [[snap, subhalo], ...]} written by hand or another tool, instead of asking the API."""
    return {int(k): lineage.parse_simple({"Main": v}) for k, v in json.loads(path.read_text()).items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--snap", type=int, help="snapshot to add (default: the config's)")
    ap.add_argument("--root", help="snapshot copy to read instead of the config's")
    ap.add_argument("--out", help="dataset folder instead of the config's")
    ap.add_argument("--z0-root", help="copy holding snapshot 99's group catalog, if OUT/splits.h5 lacks its SubhaloGrNr "
                                      "(default: --root)")
    ap.add_argument("--workers", type=int, default=8, help="parallel API requests")
    ap.add_argument("--branches", type=Path, help="JSON {subhalo: [[snap, subhalo], ...]} instead of the TNG API")
    args = ap.parse_args()
    cfg = Config.load(args.config)
    cfg = dataclasses.replace(cfg, **{k: v for k, v in (("snap", args.snap), ("root", args.root), ("out", args.out))
                                      if v is not None})
    if cfg.snap is None:
        sys.exit("no snapshot: give --snap")
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "config.json").exists():
        bad = differing(Config.load(out / "config.json").shared(), cfg.shared())
        if bad:
            sys.exit(f"{out} holds a dataset with different settings ({', '.join(bad)}); choose another --out")
    else:
        cfg.dataset().save(out / "config.json")
    sdir = snapshot_dir(out, cfg.snap)
    sdir.mkdir(exist_ok=True)

    snap = Snapshot(cfg.root, cfg.snap)
    if snap.redshift > 1.0 + 1e-6:
        print(f"note: z = {snap.redshift:.2f} > 1, where TNG50-1's softening shrinks below 0.288 kpc, so "
              f"min_rhalf_star_kpc = {cfg.min_rhalf_star_kpc} is stricter than two softening lengths")
    table = select(cfg, snap)
    if args.branches:
        branches = load_branches_json(args.branches)
    else:
        from galaxyclip3d.pointclouds.tng_api import ApiError, Client
        try:
            branches = lineage.fetch_branches(cfg.snap, table["subhalo"], sdir / "lineage.h5", Client(), args.workers)
        except ApiError as e:
            sys.exit(str(e))
    desc, has_branch = lineage.z0_descendants(cfg.snap, table["subhalo"], branches)
    missing = sum(int(s) not in branches for s in table["subhalo"])
    if missing and cfg.snap != lineage.Z0_SNAP:
        sys.exit(f"{missing:,} merger-tree branches could not be fetched; run again to fetch the rest (the "
                 f"{len(branches):,} found so far are cached in {sdir / 'lineage.h5'})")

    with lineage.dataset_lock(out):
        registry, z0_grnr = lineage.load_splits(out)
        if z0_grnr is None:
            try:
                z0_grnr = read_catalog(args.z0_root or cfg.root, lineage.Z0_SNAP, [], ["SubhaloGrNr"])["SubhaloGrNr"]
                z0_grnr = z0_grnr.astype(np.int32)
            except (OSError, ValueError) as e:
                if cfg.snap != lineage.Z0_SNAP:
                    sys.exit(f"OUT/splits.h5 has no snapshot-99 SubhaloGrNr and none could be read ({e}); "
                             f"give --z0-root with snapshot 99's group catalog")
                z0_grnr = None
        unit_snap, unit_group, z0_group = lineage.split_units(cfg.snap, table["group"], desc, z0_grnr)
        keys = lineage.unit_keys(unit_snap, unit_group)
        code, registry, n_new = lineage.assign(cfg.snap, keys, np.log10(np.maximum(table["mstar"], 1.0)), registry,
                                               cfg.split, cfg.seed)
        table.update(split=code, z0_subhalo=desc, z0_group=z0_group, unit_snap=unit_snap, unit_group=unit_group,
                     has_branch=has_branch)
        table["shard"] = shard_plan(table, cfg.shard_size)

        path = sdir / "selection.h5"
        if path.exists():
            old = read_table(path)
            same = set(old) == set(table) and all(np.array_equal(old[k], table[k]) for k in table)
            if not same:
                sys.exit(f"{path} exists and differs from this selection; delete {sdir} to build the snapshot again")
        cfg.save(sdir / "config.json")
        write_table(path, table, {"config": cfg.to_json(), "splits": ",".join(SPLITS), "snapshot": int(cfg.snap),
                                  "redshift": snap.redshift})
        lineage.save_splits(out, registry, z0_grnr, SPLITS)

    n = len(table["subhalo"])
    cen = table["central"]
    print(f"snapshot {cfg.snap} (z = {snap.redshift:.4f}): {n:,} galaxies ({cen.sum():,} centrals, {(~cen).sum():,} "
          f"satellites) in {len(np.unique(table['group'])):,} halos; {table['shard'].max() + 1} shards of up to "
          f"{cfg.shard_size}")
    for c, name in enumerate(SPLITS):
        s = code == c
        print(f"  {name:5s}: {s.sum():5,} galaxies ({s.mean():.1%}) in {len(np.unique(table['group'][s])):5,} halos")
    print(f"  split units: {len(np.unique(keys)):,}, of which {n_new:,} new (dataset total {len(registry):,}); "
          f"galaxies without a z = 0 descendant (own halo used): {(desc < 0).sum():,}")
    spread = sum(len(np.unique(code[table["group"] == g])) > 1 for g in np.unique(table["group"]))
    print(f"  halos at this snapshot whose galaxies sit in more than one split: {spread:,}")
    if cfg.snap == lineage.Z0_SNAP and has_branch.any():
        own = sum(int(branches[int(s)][branches[int(s)][:, 0] == lineage.Z0_SNAP, 1][0]) == int(s)
                  for s in table["subhalo"][has_branch])
        print(f"  branches: {has_branch.sum():,} fetched; {own:,} end in the galaxy itself at z = 0 (all should)")
    print(f"  bound gas cells >= {cfg.min_particles:,}: {(table['n_gas_bound'] >= cfg.min_particles).sum():,}; "
          f"halo has 2D maps: {table['halo_has_2d'].sum():,} ({(table['halo_has_2d'] & cen).sum():,} centrals)")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
