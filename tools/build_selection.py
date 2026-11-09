"""Write the galaxy selection, train/val/test split and shard plan of a point-cloud build (step 1 of 3).

    PYTHONPATH=pylibs python3 -s tools/build_selection.py --config configs/pc_v1.json

Writes OUT/config.json and OUT/selection.h5 (one entry per galaxy; see galaxyclip3d/pointclouds/select.py).
Refuses to overwrite a selection made with different settings.
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from galaxyclip3d.pointclouds.config import Config               # noqa: E402
from galaxyclip3d.pointclouds.select import SPLITS, select, shard_plan  # noqa: E402
from galaxyclip3d.pointclouds.snapshot import Snapshot           # noqa: E402


def write_table(path: Path, table: dict[str, np.ndarray], attrs: dict) -> None:
    tmp = path.with_name(path.name + ".part")
    with h5py.File(tmp, "w") as f:
        g = f.create_group("galaxies")
        for k, v in table.items():
            g.create_dataset(k, data=v)
        for k, v in attrs.items():
            f.attrs[k] = v
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", help="output folder instead of the one in the config")
    args = ap.parse_args()
    cfg = Config.load(args.config)
    if args.out:
        cfg = dataclasses.replace(cfg, out=args.out)
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "config.json").exists() and Config.load(out / "config.json") != cfg:
        sys.exit(f"{out} already holds a build with different settings; choose another --out")
    cfg.save(out / "config.json")

    snap = Snapshot(cfg.root, cfg.snap)
    table = select(cfg, snap)
    table["shard"] = shard_plan(table, cfg.shard_size)
    write_table(out / "selection.h5", table, {"config": cfg.to_json(), "splits": ",".join(SPLITS)})

    n = len(table["subhalo"])
    cen = table["central"]
    print(f"{n:,} galaxies ({cen.sum():,} centrals, {(~cen).sum():,} satellites) in "
          f"{len(np.unique(table['group'])):,} halos; {table['shard'].max() + 1} shards of up to {cfg.shard_size}")
    for code, name in enumerate(SPLITS):
        s = table["split"] == code
        print(f"  {name:5s}: {s.sum():5,} galaxies in {len(np.unique(table['group'][s])):5,} halos")
    print(f"  bound gas cells >= {cfg.min_particles:,}: {(table['n_gas_bound'] >= cfg.min_particles).sum():,}; "
          f"halo has 2D maps: {table['halo_has_2d'].sum():,} ({(table['halo_has_2d'] & cen).sum():,} centrals)")
    print(f"wrote {out / 'selection.h5'}")


if __name__ == "__main__":
    main()
