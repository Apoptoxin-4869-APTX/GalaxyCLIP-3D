"""Linear probes on frozen embeddings (Phase D): how well each modality's embedding predicts the galaxy's physical
properties, next to the same probes on hand-made statistics and on an untrained network of the same shape.

    PYTHONPATH=pylibs:. python -s tools/probe_clip.py --data /cluster/projects/ska/illustris/tng50-1.pointclouds/v3 \
        --runs outputs/clip/phase_c/c21_combined outputs/clip/phase_c/c22_combined_seed2 \
        --untrained outputs/clip/phase_c/c21_combined --statistics --eval-split val \
        --cache outputs/clip/phase_d/embeddings --save outputs/clip/phase_d/probes_val.json [--device cuda]

Features per galaxy and modality: each run's best.pt embedding; --untrained: that run's architecture with its initial
random weights; --statistics: the ~20 orientation-free numbers per cloud of tools/clip_baselines.py (computed on the
standardized clouds the networks see, which only shifts and scales some of them; the probes standardize every
feature anyway). Training galaxies use one fixed draw each (draw_seed = the dataset seed), validation and test
galaxies their frozen clouds, all with independent rotations. Properties and the ridge probe: galaxyclip3d/probes.py.
The ridge strength is always chosen on validation; --eval-split test then scores the test galaxies with it (do that
once, after everything is fixed). Embeddings are cached per feature set and split in --cache.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
from clip_baselines import statistics                                                    # noqa: E402
from galaxyclip3d.evaluation import ITEM_KEYS, embed, load_run, run_dataset, untrained_model  # noqa: E402
from galaxyclip3d.multimodal import MODALITIES                                           # noqa: E402
from galaxyclip3d.pointclouds.config import Config                                       # noqa: E402
from galaxyclip3d.probes import PROPERTIES, load_targets, lookup, ridge_probe            # noqa: E402
from galaxyclip3d.train_clip import TrainConfig                                          # noqa: E402

DATA_KEYS = ("radius", "points", "common_radii", "snapshots", "gas_required", "dm_aperture", "rvmax_subset",
             "min_particles", "modalities")


def cache_path(cache: Path, name: str, split: str, limit) -> Path:
    return cache / f"{name}_{split}" f"{'' if limit is None else f'_spread{limit}'}.npz"


def save_features(path: Path, feats: dict, items: dict, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, meta=json.dumps(meta), **{f"emb_{m}": v for m, v in feats.items()}, **items)


def load_features(path: Path, meta: dict):
    """(features {modality: array}, items) from a cache file, or None if it is missing or was made otherwise."""
    if not path.exists():
        return None
    with np.load(path) as z:
        if json.loads(str(z["meta"])) != meta:
            return None
        return ({m: z[f"emb_{m}"] for m in MODALITIES if f"emb_{m}" in z},
                {k: z[k] for k in ITEM_KEYS if k in z})


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", required=True, help="the dataset's folder")
    ap.add_argument("--runs", nargs="*", default=[], help="run folders (config.json + best.pt)")
    ap.add_argument("--untrained", default=None,
                    help="a run folder whose architecture is probed untrained (its config.json is enough)")
    ap.add_argument("--statistics", action="store_true", help="also probe the hand-made per-cloud statistics")
    ap.add_argument("--eval-split", choices=("val", "test"), default="val")
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--save", type=Path, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None, help="only N galaxies, spread over each split (code checks)")
    args = ap.parse_args(argv)

    models, cfgs, metas = {}, {}, {}
    for run in args.runs:
        name = Path(run).name
        if name in models:
            raise SystemExit(f"run folder names must differ: {name}")
        cfgs[name], models[name], ck = load_run(run, data=args.data)
        metas[name] = {"run": str(Path(run).resolve()), "epoch": ck.get("epoch")}
    if args.untrained:                                  # only the run's config.json is needed, no checkpoint
        cfg = TrainConfig.from_file(str(Path(args.untrained) / "config.json"), [f"data={args.data}"])
        cfgs["untrained"], models["untrained"] = cfg, untrained_model(cfg)
        metas["untrained"] = {"run": str(Path(args.untrained).resolve()), "untrained_seed": cfg.seed}
    if not cfgs:
        raise SystemExit("give --runs and/or --untrained (the data settings come from them)")
    ref = next(iter(cfgs.values()))
    for name, cfg in cfgs.items():
        differ = [k for k in DATA_KEYS if getattr(cfg, k) != getattr(ref, k)]
        if differ:
            raise SystemExit(f"{name} uses other data settings ({differ}); probe it separately")
    names = list(models) + (["statistics"] if args.statistics else [])
    splits = ("train", "val") + (("test",) if args.eval_split == "test" else ())
    seed = Config.load(Path(args.data) / "config.json").seed           # the dataset seed (frozen clouds use it too)

    feats = {n: {} for n in names}
    items = {}
    for split in splits:
        common = {"split": split, "draw_seed": seed if split == "train" else None, "limit": args.limit,
                  "rotation": "independent"}
        if ref.dm_aperture != "stellar":                # (absent for the default, so older caches stay valid)
            common["dm_aperture"] = ref.dm_aperture
        if ref.rvmax_subset:
            common["rvmax_subset"] = True
        if ref.points:                                  # (each only when set, so older caches stay valid)
            common["points"] = ref.points
        if ref.min_particles:
            common["min_particles"] = ref.min_particles
        if ref.modalities != list(MODALITIES):
            common["modalities"] = ref.modalities
        todo = {}
        for n in names:
            meta = {**common, **metas.get(n, {"statistics": True})}
            got = load_features(cache_path(args.cache, n, split, args.limit), meta)
            if got is None:
                todo[n] = meta
            else:
                feats[n][split], it = got
                items.setdefault(split, it)
        if todo:
            ds = run_dataset(ref, split, rotation="independent", draw_seed=common["draw_seed"])
            out = embed({n: models[n] for n in todo if n != "statistics"}, ds, device=args.device,
                        batch_size=args.batch, workers=args.workers, limit=args.limit,
                        per_cloud=statistics if "statistics" in todo else None)
            it = {k: out[k] for k in ITEM_KEYS if k in out}
            if split in items and not np.array_equal(items[split]["subhalo"], it["subhalo"]):
                raise SystemExit(f"cached {split} features were made on other galaxies; clear {args.cache}")
            items[split] = it
            for n, meta in todo.items():
                feats[n][split] = out["per_cloud"] if n == "statistics" else out[n]
                save_features(cache_path(args.cache, n, split, args.limit), feats[n][split], it, meta)
            print(f"{split}: {len(it['subhalo']):,} galaxies embedded for {', '.join(todo)}", flush=True)
        else:
            print(f"{split}: {len(items[split]['subhalo']):,} galaxies, all features from the cache", flush=True)

    targets = load_targets(args.data, 99)
    for split, it in items.items():
        if not np.all(it["snapshot"] == 99):
            raise SystemExit("probes need snapshot 99 targets; other snapshots are not supported yet")
    y = {split: lookup(targets, it["subhalo"]) for split, it in items.items()}
    mods = [m for m in MODALITIES if all(m in feats[n][s] for n in names for s in splits)]
    results = {}
    for n in names:
        results[n] = {}
        for m in mods:
            mi = MODALITIES.index(m)
            results[n][m] = {}
            for prop, (_, _, score) in PROPERTIES.items():
                rows = {s: items[s]["mask"][:, mi] & np.isfinite(y[s][prop]) for s in splits}
                if min(int(r.sum()) for r in rows.values()) < 10:
                    continue
                x = {s: feats[n][s][m][rows[s]] for s in splits}
                t = {s: y[s][prop][rows[s]] for s in splits}
                results[n][m][prop] = ridge_probe(x["train"], t["train"], x["val"], t["val"], x.get("test"),
                                                  t.get("test"), score=score)

    ev = args.eval_split
    regress = [p for p, (_, _, s) in PROPERTIES.items() if s == "r2"]
    yes_no = [p for p, (_, _, s) in PROPERTIES.items() if s == "auc"]
    print(f"\nmean R^2 over {len(regress)} properties on {ev}, negative values counted as 0 (in brackets: AUC of "
          + ", ".join(yes_no) + ")")
    print(f"{'features':28s}" + "".join(f"{m:>22s}" for m in mods))
    for n in names:
        row = []
        for m in mods:
            v = [max(results[n][m][p][ev], 0.0) for p in regress if p in results[n][m]]
            c = "/".join(f"{results[n][m][p][ev]:.2f}" if p in results[n][m] else "-" for p in yes_no)
            row.append(f"{np.mean(v):6.3f} ({c})" if v else "-")
        print(f"{n:28s}" + "".join(f"{r:>22s}" for r in row))
    for m in mods:
        print(f"\n{m} embedding, R^2 on {ev} (" + ", ".join(yes_no) + ": AUC):")
        print(f"{'property':18s}" + "".join(f"{n[:14]:>15s}" for n in names))
        for p in PROPERTIES:
            print(f"{p:18s}" + "".join(f"{results[n][m][p][ev]:15.3f}" if p in results[n][m] else f"{'-':>15s}"
                                       for n in names))
    args.save.parent.mkdir(parents=True, exist_ok=True)
    args.save.write_text(json.dumps({"eval_split": ev, "limit": args.limit, "features": metas,
                                     "results": results}, indent=1) + "\n")
    print(f"\nwrote {args.save}")


if __name__ == "__main__":
    main()
