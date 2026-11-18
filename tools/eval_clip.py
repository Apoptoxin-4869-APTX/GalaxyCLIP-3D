"""Retrieval scores of finished runs on one split (Phase D: the test set, scored once the final setup is fixed), in the
metric used during training: full-gallery recall per modality pair, each galaxy searched for among all galaxies that
have the other modality.

    PYTHONPATH=pylibs:. python -s tools/eval_clip.py --data /cluster/projects/ska/illustris/tng50-1.pointclouds/v3 \
        --split test --runs outputs/clip/phase_c/c21_combined outputs/clip/phase_c/c22_combined_seed2 \
        --save outputs/clip/phase_d/test_final.json [--rotation shared] [--device cuda]

Each run is scored from its best.pt with its own data settings (sphere, points per cloud, particle threshold,
modalities). --rotation replaces the runs' orientation mode, e.g. to score the shared-rotation control on
independently rotated clouds. --min-particles and --modalities score a run on fewer galaxies or modalities than it was
trained with, e.g. a four-modality run on the Stage E galaxies without HI (--min-particles 8192 --modalities stars dm
gas), for a like-for-like comparison. With several runs the mean and standard deviation over them are printed and
saved as well.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from galaxyclip3d.evaluation import embed, load_run, retrieval_scores, run_dataset   # noqa: E402
from galaxyclip3d.multimodal import MODALITIES                                        # noqa: E402

PAIRS = ("stars-dm", "stars-gas", "stars-hi", "dm-gas", "dm-hi", "gas-hi")


def pair_r1(res: dict, pair: str) -> float:
    a, b = pair.split("-")
    return (res[f"r1/{a}->{b}"] + res[f"r1/{b}->{a}"]) / 2


def scored_pairs(res: dict) -> list[str]:
    """The pairs a result has (fewer when modalities were left out)."""
    return [p for p in PAIRS if "r1/{}->{}".format(*p.split("-")) in res]


def summary(name: str, res: dict) -> str:
    return (f"{name:28s} R@1 {res['r1/mean']:6.1%}, without gas-HI {res['r1/mean_no_gas_hi']:6.1%} (R@5 "
            f"{res['r5/mean_no_gas_hi']:5.1%}, R@10 {res['r10/mean_no_gas_hi']:5.1%}; similar mass "
            f"{res.get('massbin_r1/mean_no_gas_hi', float('nan')):5.1%}) | "
            + " ".join(f"{p} {pair_r1(res, p):5.1%}" for p in scored_pairs(res)))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", required=True, help="the dataset's folder")
    ap.add_argument("--split", choices=("val", "test"), required=True)
    ap.add_argument("--runs", nargs="+", required=True, help="run folders (config.json + best.pt)")
    ap.add_argument("--rotation", choices=("independent", "shared", "none"), default=None,
                    help="orientation of the scored clouds (default: each run's own)")
    ap.add_argument("--min-particles", type=int, default=None,
                    help="score only galaxies with at least this many particles (default: each run's own)")
    ap.add_argument("--modalities", nargs="+", choices=MODALITIES, default=None,
                    help="score only these modalities, a subset of each run's (default: each run's own)")
    ap.add_argument("--save", type=Path, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None, help="score only N galaxies, spread over the split (checks)")
    args = ap.parse_args(argv)
    names = [Path(r).name for r in args.runs]
    if len(set(names)) != len(names):
        raise SystemExit(f"run folder names must differ: {names}")
    runs = {}
    for name, run in zip(names, args.runs):
        cfg, model, ck = load_run(run, data=args.data)
        if args.modalities and set(args.modalities) - set(cfg.modalities):
            raise SystemExit(f"{name} has no encoder for {sorted(set(args.modalities) - set(cfg.modalities))}")
        if args.min_particles is not None:
            cfg = dataclasses.replace(cfg, min_particles=args.min_particles)
        ds = run_dataset(cfg, args.split, rotation=args.rotation, modalities=args.modalities)
        out = embed({name: model}, ds, device=args.device, batch_size=args.batch, workers=args.workers,
                    limit=args.limit)
        res = retrieval_scores(out[name], out["mask"], out["mstar"], ks=tuple(cfg.ks), mass_bin_dex=cfg.mass_bin_dex)
        runs[name] = {"run_dir": str(run), "epoch": ck.get("epoch"), "rotation": args.rotation or cfg.rotation,
                      "min_particles": cfg.min_particles, "modalities": list(ds.modalities), "results": res}
        print(summary(f"{name} (epoch {ck.get('epoch')})", res), flush=True)
    saved = {"split": args.split, "rotation": args.rotation, "limit": args.limit, "min_particles": args.min_particles,
             "modalities": args.modalities, "runs": runs}
    if len(runs) > 1:
        keys = [k for k, v in next(iter(runs.values()))["results"].items()
                if isinstance(v, float) and all(k in r["results"] for r in runs.values())]
        table = {k: np.array([r["results"][k] for r in runs.values()]) for k in keys}
        saved["mean"] = {k: float(v.mean()) for k, v in table.items()}
        saved["std"] = {k: float(v.std(ddof=1)) for k, v in table.items()}
        m, s = saved["mean"], saved["std"]
        print(f"mean of {len(runs)} runs: R@1 {m['r1/mean']:.1%} +- {s['r1/mean']:.1%}, without gas-HI "
              f"{m['r1/mean_no_gas_hi']:.1%} +- {s['r1/mean_no_gas_hi']:.1%}, R@10 without gas-HI "
              f"{m['r10/mean_no_gas_hi']:.1%} +- {s['r10/mean_no_gas_hi']:.1%} | "
              + " ".join(f"{p} {np.mean([pair_r1(r['results'], p) for r in runs.values()]):.1%}"
                         for p in scored_pairs(m)))
    args.save.parent.mkdir(parents=True, exist_ok=True)
    args.save.write_text(json.dumps(saved, indent=1) + "\n")
    print(f"wrote {args.save}")


if __name__ == "__main__":
    main()
