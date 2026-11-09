"""Finish a point-cloud build once every shard exists (step 3 of 3): the galaxy index, feature statistics from the
training split, and frozen validation and test clouds.

    PYTHONPATH=pylibs python3 -s tools/finalize_pointclouds.py --out DIR [--radius 5] [--stats-galaxies 1000]

Writes OUT/index.h5, OUT/feature_stats.json and OUT/frozen/{val,test}_r<radius>.h5. Frozen clouds are drawn once,
unrotated and not standardized, with a seed tied to each galaxy, so evaluation is identical in every run.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from galaxyclip3d.pointclouds.config import Config                          # noqa: E402
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


def build_index(out: Path, n_shards: int) -> dict[str, np.ndarray]:
    parts: dict[str, list] = {}
    for s in range(n_shards):
        g = Shard(out / "shards" / f"shard_{s:04d}.h5").galaxies()
        g["shard"] = np.full(len(g["subhalo"]), s, dtype=np.int32)
        for k, v in g.items():
            parts.setdefault(k, []).append(v)
    index = {k: np.concatenate(v) for k, v in parts.items()}
    order = np.argsort(index["subhalo"])
    return {k: v[order] for k, v in index.items()}


def clouds_of(out: Path, index: dict, j: int, cfg: Config, radius: float, rng, rotate: bool, shards: dict):
    s = int(index["shard"][j])
    shards.setdefault(s, Shard(out / "shards" / f"shard_{s:04d}.h5"))
    parts = {n: shards[s].particles(int(index["row"][j]), n) for n in ("stars", "dm", "gas")}
    return draw(parts, radius * float(index["rhalf_star_kpc"][j]), float(index["vmax"][j]), cfg.budget, rng, rotate)


def atomic_h5(path: Path, data: dict[str, np.ndarray], attrs: dict, group: str | None = None) -> None:
    tmp = path.with_name(path.name + ".part")
    with h5py.File(tmp, "w") as f:
        g = f.create_group(group) if group else f
        for k, v in data.items():
            g.create_dataset(k, data=v)
        for k, v in attrs.items():
            f.attrs[k] = v
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--radius", type=float, help="sphere for stats and frozen clouds (default: the config's)")
    ap.add_argument("--stats-galaxies", type=int, default=1000, help="training galaxies used for feature statistics")
    args = ap.parse_args()
    out = args.out
    cfg = Config.load(out / "config.json")
    radius = cfg.train_radius if args.radius is None else args.radius
    with h5py.File(out / "selection.h5", "r") as f:
        n_shards = int(f["galaxies/shard"][()].max()) + 1
    missing = [s for s in range(n_shards) if not (out / "shards" / f"shard_{s:04d}.h5").exists()]
    if missing:
        sys.exit(f"{len(missing)} shards missing, e.g. {missing[:10]}; run tools/build_pointclouds.py first")

    index = build_index(out, n_shards)
    atomic_h5(out / "index.h5", index, {"config": cfg.to_json(), "splits": ",".join(SPLITS)}, group="galaxies")
    usable, mask = masks(index, cfg, radius)
    print(f"index: {len(index['subhalo']):,} galaxies; usable at {radius:g} R1/2,*: {usable.sum():,}; "
          f"gas on {mask[usable, 2].sum():,}, HI on {mask[usable, 3].sum():,}", flush=True)

    shards: dict[int, Shard] = {}
    rng = np.random.default_rng([cfg.seed, 2])
    train = np.flatnonzero(usable & (index["split"] == 0))
    pick = rng.choice(train, size=min(args.stats_galaxies, len(train)), replace=False)
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
    stats = {m: {f: {"mean": sums[m][i] / n[m], "std": float(np.sqrt(max(sq[m][i] / n[m] - (sums[m][i] / n[m]) ** 2,
                                                                         1e-12)))}
                 for i, f in enumerate(FEATURES[m])} for m in MODALITIES}
    stats["_meta"] = {"radius": radius, "galaxies": int(len(pick)), "split": "train"}
    tmp = out / "feature_stats.json.part"
    tmp.write_text(json.dumps(stats, indent=1))
    os.replace(tmp, out / "feature_stats.json")
    print(f"feature statistics from {len(pick)} training galaxies", flush=True)

    (out / "frozen").mkdir(exist_ok=True)
    for code, split in enumerate(SPLITS[1:], start=1):
        rows = np.flatnonzero(usable & (index["split"] == code))
        data = {m: np.zeros((len(rows), cfg.budget, len(FEATURES[m])), dtype=np.float32) for m in MODALITIES}
        for i, j in enumerate(rows):
            c = clouds_of(out, index, j, cfg, radius, np.random.default_rng([cfg.seed, int(index["subhalo"][j]), 1]),
                          False, shards)
            for m in MODALITIES:
                data[m][i] = c[m]
        data.update(subhalo=index["subhalo"][rows], mask=mask[rows])
        atomic_h5(out / "frozen" / f"{split}_r{radius:g}.h5", data,
                  {"radius": radius, "budget": cfg.budget, "features": json.dumps(FEATURES)})
        print(f"frozen {split}: {len(rows)} galaxies", flush=True)


if __name__ == "__main__":
    main()
