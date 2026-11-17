"""Retrieval baselines without a neural network, on the validation (or, with --split test, the test) galaxies, in the
same metric as training (full-gallery recall per modality pair, multimodal.masked_recall).

    PYTHONPATH=pylibs:. python -s tools/clip_baselines.py --out /cluster/projects/ska/illustris/tng50-1.pointclouds/v3 \
        --snap 99 --save outputs/clip_baselines [--workers 8] [--split test]

1. Orientation only: each cloud becomes the direction of its angular momentum (or its long shape axis), clouds are
   matched by direction. In the clouds' shared frame this works far above chance; with an independent rotation per
   modality (what training and evaluation use) it must fall to chance.
2. Simple statistics: each cloud becomes about 20 numbers that do not depend on its orientation (radii, shape, speeds,
   rotation, means and spreads of its other values). For each modality pair a canonical correlation analysis fitted on
   one draw of every training galaxy maps both into a shared space, where validation galaxies are matched. This is
   what a network has to beat to show it learns more than summary statistics.

Needs numpy, h5py, torch (for the recall function) and scikit-learn.
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from galaxyclip3d.multimodal import MODALITIES, NEAR_DUPLICATE, masked_recall           # noqa: E402
from galaxyclip3d.pointclouds.config import Config                                       # noqa: E402
from galaxyclip3d.pointclouds.layout import frozen_path, read_table, shard_path, snapshot_dir  # noqa: E402
from galaxyclip3d.pointclouds.sampling import FEATURES, draw, orient                     # noqa: E402
from galaxyclip3d.pointclouds.store import Shard                                         # noqa: E402


def statistics(c: np.ndarray) -> np.ndarray:
    """Orientation-free summary of one cloud (N, F): radii, shape, speeds, rotation, scalar means and spreads."""
    pos, vel = c[:, 0:3].astype(np.float64), c[:, 3:6].astype(np.float64)
    r = np.linalg.norm(pos, axis=1)
    rr = np.maximum(r, 1e-9)
    p = pos - pos.mean(0)
    ev = np.sqrt(np.maximum(np.linalg.eigvalsh(p.T @ p / len(p)), 1e-12))          # ascending
    vr = (pos * vel).sum(1) / rr
    vt = np.linalg.norm(np.cross(pos, vel), axis=1) / rr
    spin = np.linalg.norm(np.cross(pos, vel).sum(0)) / max((r * np.linalg.norm(vel, axis=1)).sum(), 1e-12)
    out = [r.mean(), np.median(r), np.percentile(r, 90), r.std(), ev[1] / ev[2], ev[0] / ev[2],
           np.linalg.norm(vel, axis=1).mean(), vel.std(0).mean(), vr.mean(), vr.std(), vt.mean(), vt.std(), spin]
    for j in range(6, c.shape[1]):
        out += [c[:, j].mean(), c[:, j].std()]
    return np.asarray(out)


def masks(index, cfg, radius):
    k = f"r{radius:g}"
    usable = (index[f"n_stars_{k}"] >= cfg.min_particles) & (index[f"n_dm_{k}"] >= cfg.min_particles)
    ones = np.ones(len(usable), dtype=bool)
    return usable, np.stack([ones, ones, index[f"n_gas_{k}"] >= cfg.min_particles,
                             index[f"hi_eff_{k}"] >= cfg.hi_min_eff], axis=1)


def train_stats(item):
    """Statistics of one seeded draw for each training galaxy of one shard."""
    path, rows, radii, vmax, positions, budget, seed = item
    sh = Shard(path)
    out = []
    for row, rad, vm, j in zip(rows, radii, vmax, positions):
        parts = {t: sh.particles(int(row), t) for t in ("stars", "dm", "gas")}
        c = draw(parts, float(rad), float(vm), budget, np.random.default_rng([seed, int(j), 3]), rotate=False)
        out.append({m: statistics(c[m]) for m in MODALITIES})
    sh.close()
    return positions, out


def unit(x: np.ndarray) -> torch.Tensor:
    x = torch.as_tensor(x, dtype=torch.float64)
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)


def spin_dir(c: np.ndarray) -> np.ndarray:
    return np.cross(c[:, :, 0:3], c[:, :, 3:6]).sum(1)


def axis_tensor(c: np.ndarray) -> np.ndarray:
    """The long shape axis as the flattened outer product a a^T, so that cosine similarity is (a.b)^2, sign-free."""
    p = c[:, :, 0:3] - c[:, :, 0:3].mean(1, keepdims=True)
    a = np.linalg.eigh(np.einsum("nki,nkj->nij", p, p))[1][:, :, -1]
    return np.einsum("ni,nj->nij", a, a).reshape(len(a), 9)


def summary_line(name: str, res: dict) -> str:
    dirs = [k[3:] for k in res if k.startswith("r1/") and "->" in k]
    return (f"{name:34s} R@1 {res['r1/mean']:6.1%}  without gas-HI {res['r1/mean_no_gas_hi']:6.1%}  chance "
            f"{res['chance/mean']:.2%}   " + ", ".join(f"{d} {res['r1/' + d]:.1%}" for d in dirs if d.startswith("stars")))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True, help="dataset folder")
    ap.add_argument("--snap", type=int, default=99)
    ap.add_argument("--save", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--components", type=int, default=8, help="canonical components per pair")
    ap.add_argument("--split", choices=("val", "test"), default="val", help="galaxies to score (fits use train)")
    args = ap.parse_args()
    from sklearn.cross_decomposition import CCA
    from sklearn.preprocessing import StandardScaler
    sdir = snapshot_dir(args.out, args.snap)
    cfg = Config.load(sdir / "config.json")
    index = read_table(sdir / "index.h5")
    usable, mask = masks(index, cfg, cfg.train_radius)
    args.save.mkdir(parents=True, exist_ok=True)
    results = {}

    with h5py.File(frozen_path(args.out, args.snap, args.split, cfg.train_radius), "r") as f:
        val = {m: f[m][()] for m in MODALITIES}
        vmask = torch.as_tensor(f["mask"][()])
        vsub = f["subhalo"][()]
    turned = {m: np.empty_like(val[m]) for m in MODALITIES}
    for i, s in enumerate(vsub):                 # the loader's fixed evaluation rotations
        o = orient({m: val[m][i] for m in MODALITIES}, np.random.default_rng([cfg.seed, args.snap, int(s), 11]))
        for m in MODALITIES:
            turned[m][i] = o[m]
    for frame, clouds in (("shared frame", val), ("independent rotations", turned)):
        for what, fn in (("spin direction", spin_dir), ("shape axis", axis_tensor)):
            res = masked_recall({m: unit(fn(clouds[m])) for m in MODALITIES}, vmask)
            results[f"orientation: {what}, {frame}"] = res
            print(summary_line(f"{what}, {frame}", res), flush=True)

    train = np.flatnonzero(usable & (index["split"] == 0))
    radius = cfg.train_radius * index["rhalf_star_kpc"]
    items = []
    for s in np.unique(index["shard"][train]):
        pos = train[index["shard"][train] == s]
        items.append((shard_path(args.out, args.snap, s), index["row"][pos], radius[pos], index["vmax"][pos], pos,
                      cfg.budget, cfg.seed))
    tstats = {}
    with Pool(args.workers) as pool:
        for pos, vals in pool.imap_unordered(train_stats, items):
            tstats.update(zip(pos.tolist(), vals))
    order = sorted(tstats)
    xtrain = {m: np.stack([tstats[j][m] for j in order]) for m in MODALITIES}
    tmask = mask[order]
    xval = {m: np.stack([statistics(turned[m][i]) for i in range(len(vsub))]) for m in MODALITIES}
    print(f"statistics: {len(order):,} training galaxies (one draw each), {len(vsub)} {args.split} galaxies, "
          + ", ".join(f"{m} {xval[m].shape[1]}" for m in MODALITIES) + " numbers per cloud", flush=True)

    out = {}
    for a, b in combinations(MODALITIES, 2):
        ia, ib = MODALITIES.index(a), MODALITIES.index(b)
        rows = tmask[:, ia] & tmask[:, ib]
        sa, sb = StandardScaler().fit(xtrain[a][rows]), StandardScaler().fit(xtrain[b][rows])
        k = min(args.components, xtrain[a].shape[1], xtrain[b].shape[1])
        cca = CCA(n_components=k, max_iter=2000).fit(sa.transform(xtrain[a][rows]), sb.transform(xtrain[b][rows]))
        ua, ub = cca.transform(sa.transform(xval[a]), sb.transform(xval[b]))
        pair = masked_recall({a: unit(ua), b: unit(ub)}, vmask[:, [ia, ib]], modalities=(a, b))
        out.update({k2: v for k2, v in pair.items() if "->" in k2})
    for kk in (1, 5, 10):
        dirs = [k2 for k2 in out if k2.startswith(f"r{kk}/")]
        out[f"r{kk}/mean"] = float(np.mean([out[d] for d in dirs]))
        plain = [d for d in dirs if tuple(sorted(d.split("/")[1].split("->"))) not in NEAR_DUPLICATE]
        out[f"r{kk}/mean_no_gas_hi"] = float(np.mean([out[d] for d in plain]))
    out["chance/mean"] = float(np.mean([v for k2, v in out.items() if k2.startswith("chance/")]))
    results["statistics + CCA"] = out
    print(summary_line("statistics + CCA", out), flush=True)
    name = "baselines.json" if args.split == "val" else f"baselines_{args.split}.json"
    (args.save / name).write_text(json.dumps(results, indent=1) + "\n")
    print(f"wrote {args.save / name}")


if __name__ == "__main__":
    main()
