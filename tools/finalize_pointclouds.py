"""Finish one snapshot of a point-cloud dataset once all its shards exist (step 3 of 3): its galaxy index, the
dataset index over all snapshots, feature statistics (first snapshot only) and frozen validation and test clouds.

    PYTHONPATH=pylibs python3 -s tools/finalize_pointclouds.py --out DIR --snap 99 [--radius 5] [--stats-galaxies 1000]
                                                                                  [--recompute-stats]

Writes OUT/snap_NNN/index.h5, OUT/snap_NNN/frozen/{val,test}_r<radius>.h5, and rebuilds OUT/index.h5 and
OUT/snapshots.json from every finalized snapshot. OUT/feature_stats.json is computed from training galaxies when the
first snapshot is finalized and then kept, so standardization does not shift as snapshots are added; --recompute-stats
recomputes it from the training galaxies of all finalized snapshots (frozen clouds are stored unstandardized, so they
stay valid). Frozen clouds are drawn once, unrotated, with a seed tied to each galaxy (dataset seed, subhalo, 1), so
evaluation is identical in every run; the subhalo number repeats across snapshots, which only reuses a random stream
for a different set of particles. Needs only the dataset folder, not the raw snapshot.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from galaxyclip3d.pointclouds.config import Config                          # noqa: E402
from galaxyclip3d.pointclouds.layout import (frozen_path, read_table, shard_path, snapshot_dir,  # noqa: E402
                                             snapshots_in, write_table)
from galaxyclip3d.pointclouds.sampling import FEATURES, MODALITIES, draw    # noqa: E402
from galaxyclip3d.pointclouds.select import SPLITS                         # noqa: E402
from galaxyclip3d.pointclouds.store import Shard                           # noqa: E402


def masks(index: dict, cfg: Config, radius: float) -> tuple[np.ndarray, np.ndarray]:
    """Same rule as galaxyclip3d.dataset.galaxy_masks (kept torch-free here)."""
    k = f"r{radius:g}"
    usable = (index[f"n_stars_{k}"] >= cfg.min_particles) & (index[f"n_dm_{k}"] >= cfg.min_particles)
    ones = np.ones(len(usable), dtype=bool)
    return usable, np.stack([ones, ones, index[f"n_gas_{k}"] >= cfg.min_particles,
                             index[f"hi_eff_{k}"] >= cfg.hi_min_eff], axis=1)


def build_snapshot_index(out: Path, snap: int, n_shards: int) -> dict[str, np.ndarray]:
    parts: dict[str, list] = {}
    for s in range(n_shards):
        sh = Shard(shard_path(out, snap, s))
        g = sh.galaxies()
        if int(sh.file.attrs.get("snapshot", -1)) != snap or not (g["snapshot"] == snap).all():
            sys.exit(f"{sh.path} does not say it holds snapshot {snap}")
        sh.close()
        g["shard"] = np.full(len(g["subhalo"]), s, dtype=np.int32)
        for k, v in g.items():
            parts.setdefault(k, []).append(v)
    index = {k: np.concatenate(v) for k, v in parts.items()}
    order = np.argsort(index["subhalo"])
    return {k: v[order] for k, v in index.items()}


def build_dataset_index(out: Path) -> dict[str, np.ndarray]:
    """All finalized snapshots' indexes, by snapshot and then subhalo."""
    tables = [read_table(snapshot_dir(out, s) / "index.h5") for s in snapshots_in(out, "index.h5")]
    cols = set(tables[0])
    for t in tables[1:]:
        if set(t) != cols:
            sys.exit(f"snapshot indexes have different columns ({sorted(cols ^ set(t))}); finalize them again")
    return {k: np.concatenate([t[k] for t in tables]) for k in tables[0]}


def clouds_of(out: Path, index: dict, j: int, cfg: Config, radius: float, rng, rotate: bool, shards: dict):
    key = (int(index["snapshot"][j]), int(index["shard"][j]))
    if key not in shards:
        shards[key] = Shard(shard_path(out, *key))
    parts = {n: shards[key].particles(int(index["row"][j]), n) for n in ("stars", "dm", "gas")}
    return draw(parts, radius * float(index["rhalf_star_kpc"][j]), float(index["vmax"][j]), cfg.budget, rng, rotate)


def feature_stats(out: Path, index: dict, usable: np.ndarray, mask: np.ndarray, cfg: Config, radius: float,
                  n_galaxies: int, shards: dict) -> dict:
    rng = np.random.default_rng([cfg.seed, 2])
    train = np.flatnonzero(usable & (index["split"] == 0))
    if not len(train):
        return None
    pick = rng.choice(train, size=min(n_galaxies, len(train)), replace=False)
    sums = {m: np.zeros(len(FEATURES[m])) for m in MODALITIES}
    sq = {m: np.zeros(len(FEATURES[m])) for m in MODALITIES}
    n = {m: 0 for m in MODALITIES}
    for j in pick:
        c = clouds_of(out, index, j, cfg, radius, rng, True, shards)
        for col, m in enumerate(MODALITIES):
            if mask[j, col]:
                sums[m] += c[m].sum(0)
                sq[m] += (c[m].astype(np.float64) ** 2).sum(0)
                n[m] += len(c[m])
    empty = [m for m in MODALITIES if n[m] == 0]                # no galaxy had it: identity standardization
    stats = {m: {f: {"mean": sums[m][i] / n[m], "std": float(np.sqrt(max(sq[m][i] / n[m] - (sums[m][i] / n[m]) ** 2,
                                                                         1e-12)))} if n[m] else {"mean": 0.0, "std": 1.0}
                 for i, f in enumerate(FEATURES[m])} for m in MODALITIES}
    stats["_meta"] = {"radius": radius, "galaxies": int(len(pick)), "split": "train",
                      "snapshots": sorted({int(s) for s in index["snapshot"][pick]}), "empty_modalities": empty}
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True, help="dataset folder")
    ap.add_argument("--snap", type=int, required=True, help="snapshot to finalize")
    ap.add_argument("--radius", type=float, help="sphere for stats and frozen clouds (default: the config's)")
    ap.add_argument("--stats-galaxies", type=int, default=1000, help="training galaxies used for feature statistics")
    ap.add_argument("--recompute-stats", action="store_true", help="recompute feature statistics from all snapshots")
    args = ap.parse_args()
    out, snap = args.out, args.snap
    sdir = snapshot_dir(out, snap)
    cfg = Config.load(sdir / "config.json")
    radius = cfg.train_radius if args.radius is None else args.radius
    sel = read_table(sdir / "selection.h5")
    n_shards = int(sel["shard"].max()) + 1
    missing = [s for s in range(n_shards) if not shard_path(out, snap, s).exists()]
    if missing:
        sys.exit(f"{len(missing)} shards of snapshot {snap} missing, e.g. {missing[:10]}; run "
                 f"tools/build_pointclouds.py --snap {snap} first")

    index = build_snapshot_index(out, snap, n_shards)
    attrs = {"config": cfg.to_json(), "splits": ",".join(SPLITS), "snapshot": snap,
             "redshift": float(index["redshift"][0])}
    write_table(sdir / "index.h5", index, attrs)
    usable, mask = masks(index, cfg, radius)
    print(f"snapshot {snap}: {len(index['subhalo']):,} galaxies; usable at {radius:g} R1/2,*: {usable.sum():,}; "
          f"gas on {mask[usable, 2].sum():,}, HI on {mask[usable, 3].sum():,}", flush=True)

    every = build_dataset_index(out)
    write_table(out / "index.h5", every, {"config": Config.load(out / "config.json").to_json(),
                                          "splits": ",".join(SPLITS), "snapshots": json.dumps(snapshots_in(out, "index.h5"))})
    print(f"dataset index: {len(every['subhalo']):,} galaxies from snapshots {snapshots_in(out, 'index.h5')}", flush=True)

    shards: dict[tuple, Shard] = {}
    stats_path = out / "feature_stats.json"
    if args.recompute_stats or not stats_path.exists():
        u_all, m_all = masks(every, cfg, radius)
        stats = feature_stats(out, every, u_all, m_all, cfg, radius, args.stats_galaxies, shards)
        if stats is None:
            print("no usable training galaxies yet: feature statistics wait for a later finalize", flush=True)
        else:
            tmp = out / "feature_stats.json.part"
            tmp.write_text(json.dumps(stats, indent=1))
            os.replace(tmp, stats_path)
            print(f"feature statistics from {stats['_meta']['galaxies']} training galaxies of snapshots "
                  f"{stats['_meta']['snapshots']}", flush=True)
    else:
        print(f"feature statistics kept from {json.loads(stats_path.read_text())['_meta']}", flush=True)

    (sdir / "frozen").mkdir(exist_ok=True)
    for code, split in enumerate(SPLITS[1:], start=1):
        rows = np.flatnonzero(usable & (index["split"] == code))
        data = {m: np.zeros((len(rows), cfg.budget, len(FEATURES[m])), dtype=np.float32) for m in MODALITIES}
        for i, j in enumerate(rows):
            c = clouds_of(out, index, j, cfg, radius, np.random.default_rng([cfg.seed, int(index["subhalo"][j]), 1]),
                          False, shards)
            for m in MODALITIES:
                data[m][i] = c[m]
        data.update(snapshot=index["snapshot"][rows], subhalo=index["subhalo"][rows], mask=mask[rows])
        write_table(frozen_path(out, snap, split, radius), data,
                    {"radius": radius, "budget": cfg.budget, "features": json.dumps(FEATURES), "snapshot": snap},
                    group=None)
        print(f"frozen {split}: {len(rows)} galaxies", flush=True)

    summary, prev = {}, _previous(out)
    for s in snapshots_in(out, "index.h5"):
        t = read_table(snapshot_dir(out, s) / "index.h5")
        u, mk = masks(t, cfg, cfg.train_radius)
        sh = Shard(shard_path(out, s, 0))
        summary[str(s)] = {"redshift": float(t["redshift"][0]), "galaxies": int(len(t["subhalo"])),
                           f"usable_r{cfg.train_radius:g}": int(u.sum()),
                           "split_galaxies": {n: int((t["split"] == c).sum()) for c, n in enumerate(SPLITS)},
                           "built": sh.file.attrs.get("created", ""), "git_commit": sh.file.attrs.get("git_commit", ""),
                           "source": sh.file.attrs.get("source", ""),
                           "finalized": time.strftime("%Y-%m-%d %H:%M:%S") if s == snap else
                           prev.get(str(s), {}).get("finalized", "")}
        sh.close()
    tmp = out / "snapshots.json.part"
    tmp.write_text(json.dumps(summary, indent=1) + "\n")
    os.replace(tmp, out / "snapshots.json")


def _previous(out: Path) -> dict:
    p = out / "snapshots.json"
    return json.loads(p.read_text()) if p.exists() else {}


if __name__ == "__main__":
    main()
