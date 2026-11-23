"""How well a galaxy property (default: the mean stellar age) can be read off frozen embeddings, next to simple
baselines. Probes are fitted on the training galaxies and scored on the validation galaxies.

    PYTHONPATH=pylibs:. python -s tools/age_probe.py \
        --runs outputs/clip/stage_e/e19_complete_shared outputs/clip/stage_e/e23_complete_shared_noage \
               outputs/clip/stage_e/e21_complete_patch_shared outputs/clip/stage_e/e24_complete_patch_shared_noage \
        --labels "e19 PointNet, age seen" "e23 PointNet, age hidden" "e21 patch, age seen" "e24 patch, age hidden" \
        --untrained outputs/clip/stage_e/e23_complete_shared_noage \
        --data /cluster/projects/ska/illustris/tng50-1.pointclouds/complete \
        --targets /cluster/projects/ska/illustris/tng50-1.pointclouds/v3 \
        --out outputs/clip/stage_e/age_probe [--property age_mean_gyr] [--device cuda]

Galaxies: snapshot 99, the only one with a targets.h5. age_mean_gyr there is the mass-weighted mean age of the stars
inside the training sphere, i.e. of the stars the clouds are drawn from. Training galaxies: one seeded draw each;
validation galaxies: their frozen clouds; every modality rotated on its own. The test set stays untouched. Every
setting of a probe (ridge strength, number of neighbours, MLP weight decay) is chosen by cross-validation on the
training galaxies; validation galaxies are only scored.

Features per modality:
  each run's best.pt embedding; --untrained: that run's architecture with its initial weights
  stats  the orientation-free numbers of clip_baselines.statistics (radii, shape, speeds, rotation, mean and spread of
         each physical value) of the cloud as the encoders get it; for the stars without the age column ("stats, no
         age"), split into its metallicity and its structure + kinematics numbers, and with the age column ("stats with
         age": a ceiling, as the mean age of the drawn stars estimates the target directly)
  mass   the stellar mass alone (log M*)
Probes: ridge regression (linear), k nearest neighbours (cosine on embeddings), an MLP with one hidden layer of 128
(5 seeds averaged). Scores on validation: R², RMSE and MAE with 95% bootstrap intervals over galaxies; for the linear
probe also the R² of what is left after the stellar-mass trend ("at fixed mass").

Writes OUT/results.json, OUT/cache/ (embeddings and statistics) and the figures OUT/age_scatter, OUT/age_r2 (PDF and
PNG). The scatter figure puts the runs two per row in the order given (e.g. with and without age side by side).
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
from clip_baselines import statistics                                                       # noqa: E402
from clip_report import AQUA, AXIS, BLUE, INK, INK2, MUTED, NEUTRAL, ORANGE, axis_labels, save, style  # noqa: E402
from galaxyclip3d.evaluation import embed, load_run, run_dataset, untrained_model          # noqa: E402
from galaxyclip3d.multimodal import MODALITIES                                              # noqa: E402
from galaxyclip3d.pointclouds.config import Config                                          # noqa: E402
from galaxyclip3d.probes import load_targets, lookup                                        # noqa: E402
from galaxyclip3d.train_clip import TrainConfig                                             # noqa: E402

import matplotlib.pyplot as plt                                                              # noqa: E402
from matplotlib.lines import Line2D                                                          # noqa: E402
from sklearn.exceptions import ConvergenceWarning                                            # noqa: E402
from sklearn.linear_model import Ridge                                                       # noqa: E402
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict                  # noqa: E402
from sklearn.neighbors import KNeighborsRegressor                                            # noqa: E402
from sklearn.neural_network import MLPRegressor                                              # noqa: E402
from sklearn.pipeline import make_pipeline                                                   # noqa: E402
from sklearn.preprocessing import StandardScaler                                             # noqa: E402

MODS = ("stars", "dm", "gas")
MOD_NAME = {"stars": "stars", "dm": "dark matter", "gas": "gas"}
DATA_KEYS = ("radius", "points", "common_radii", "gas_required", "dm_aperture", "rvmax_subset", "min_particles",
             "modalities")
PROBES = ("linear", "knn", "mlp")
PROBE_NAME = {"linear": "linear (ridge)", "knn": "nearest neighbours", "mlp": "MLP"}
FEW = ("stellar mass", "stats: metallicity only")      # one or two numbers: no MLP
PROBE_COLOUR = {"linear": BLUE, "knn": ORANGE, "mlp": AQUA}      # categorical slots 1-3 in their fixed order
GRIDS = {"linear": {"ridge__alpha": list(np.logspace(-3, 5, 17))},
         "knn": {"kneighborsregressor__n_neighbors": [5, 10, 20, 40, 80]},
         "mlp": {"mlpregressor__alpha": [1e-3, 1e-1, 1e1]}}
N_STRUCT = 13          # clip_baselines.statistics: 13 structure + kinematics numbers, then mean and spread per value
STAR_AGE, STAR_Z = [13, 14], [15, 16]                       # the stars' age and metallicity numbers
STAT_ROWS = ("stats, no age", "stats: metallicity only", "stats: structure + kinematics", "stats with age")


# ------------------------------------------------------------------------------------------------------- features
def features(args, runs: dict, untrained) -> tuple[dict, dict]:
    """({split: {feature name: {modality: array}}}, {split: per-galaxy items}), embedded once and cached."""
    cfgs = {name: cfg for name, (cfg, _, _) in runs.items()}
    ref = next(iter(cfgs.values()))
    for name, cfg in cfgs.items():
        differ = [k for k in DATA_KEYS if getattr(cfg, k) != getattr(ref, k)]
        if differ:
            raise SystemExit(f"{name} uses other data settings ({differ}); probe it separately")
    seed = Config.load(args.data / "config.json").seed           # the dataset seed (the frozen clouds use it too)
    models = {name: model for name, (_, model, _) in runs.items()}
    if untrained is not None:
        models["untrained"] = untrained
    meta_runs = {n: {"run": str(Path(r).resolve()), "epoch": runs[n][2].get("epoch")} for n, r in zip(runs, args.runs)}
    feats, items = {}, {}
    for split in ("train", "val"):
        meta = {"split": split, "draw_seed": seed if split == "train" else None, "limit": args.limit,
                "rotation": "independent", "runs": meta_runs, "untrained": str(args.untrained)}
        cache = args.out / "cache" / f"{split}{'' if args.limit is None else f'_spread{args.limit}'}.npz"
        if cache.exists():
            with np.load(cache, allow_pickle=False) as z:
                if json.loads(str(z["meta"])) == meta:
                    feats[split] = {n: {m: z[f"{n}__{m}"] for m in MODS} for n in [*models, "stats"]}
                    items[split] = {k: z[k] for k in ("mask", "subhalo", "snapshot")}
                    print(f"{split}: {len(items[split]['subhalo']):,} galaxies from {cache}", flush=True)
                    continue
        ds = run_dataset(ref, split, rotation="independent", draw_seed=meta["draw_seed"])
        out = embed(models, ds, device=args.device, batch_size=args.batch, workers=args.workers, limit=args.limit,
                    per_cloud=statistics)
        feats[split] = {n: {m: out[n][m] for m in MODS} for n in models}
        feats[split]["stats"] = {m: out["per_cloud"][m] for m in MODS}
        items[split] = {k: out[k] for k in ("mask", "subhalo", "snapshot")}
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, meta=json.dumps(meta), **items[split],
                 **{f"{n}__{m}": v for n, d in feats[split].items() for m, v in d.items()})
        print(f"{split}: {len(items[split]['subhalo']):,} galaxies embedded", flush=True)
    return feats, items


def feature_sets(feats: dict, m: str, run_names: list[str], has_untrained: bool) -> dict:
    """{row name: (array, is_embedding)} for one modality of one split."""
    rows = {n: (feats[n][m], True) for n in run_names}
    if has_untrained:
        rows["untrained"] = (feats["untrained"][m], True)
    s = feats["stats"][m]
    if m == "stars":
        rows["stats, no age"] = (s[:, [*range(N_STRUCT), *STAR_Z]], False)
        rows["stats: metallicity only"] = (s[:, STAR_Z], False)
        rows["stats: structure + kinematics"] = (s[:, :N_STRUCT], False)
        rows["stats with age"] = (s, False)
    else:
        rows["stats, no age"] = (s, False)          # dark matter and gas clouds carry no age column anyway
        rows["stats: structure + kinematics"] = (s[:, :N_STRUCT], False)
    return rows


# ---------------------------------------------------------------------------------------------------------- probes
def pipeline(kind: str, embedding: bool, seed: int = 0):
    if kind == "linear":
        return make_pipeline(StandardScaler(), Ridge())
    if kind == "knn":
        if embedding:                              # the encoders compare embeddings by cosine similarity
            return make_pipeline(KNeighborsRegressor(metric="cosine", algorithm="brute"))
        return make_pipeline(StandardScaler(), KNeighborsRegressor())
    return make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(128,), early_stopping=True,
                                                        n_iter_no_change=20, max_iter=2000, random_state=seed))


def probe(kind: str, embedding: bool, xtr, ytr, xva, seeds: int = 5) -> tuple[np.ndarray, float, dict]:
    """(validation predictions, cross-validated R² on the training galaxies, the chosen setting)."""
    search = GridSearchCV(pipeline(kind, embedding), GRIDS[kind], scoring="r2",
                          cv=KFold(3 if kind == "mlp" else 5, shuffle=True, random_state=0))
    search.fit(xtr, ytr)
    chosen = {k.split("__")[1]: float(v) for k, v in search.best_params_.items()}
    if kind == "mlp":                              # several seeds, averaged: one MLP fit is noisy
        pred = np.mean([pipeline(kind, embedding, s).set_params(**search.best_params_).fit(xtr, ytr).predict(xva)
                        for s in range(seeds)], axis=0)
    else:
        pred = search.predict(xva)
    return pred, float(search.best_score_), chosen


def r2(y, p) -> float:
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def scores(y: np.ndarray, p: np.ndarray, rng: np.random.Generator, n_boot: int = 2000) -> dict:
    idx = rng.integers(0, len(y), size=(n_boot, len(y)))
    r2s = np.array([r2(y[i], p[i]) for i in idx])
    rms = np.sqrt(((y[idx] - p[idx]) ** 2).mean(axis=1))
    return {"r2": r2(y, p), "r2_ci": np.percentile(r2s, [2.5, 97.5]).tolist(),
            "rmse": float(np.sqrt(((y - p) ** 2).mean())), "rmse_ci": np.percentile(rms, [2.5, 97.5]).tolist(),
            "mae": float(np.abs(y - p).mean())}


# ---------------------------------------------------------------------------------------------------------- figures
def fig_scatter(out: Path, panels: list, prop_label: str, unit: str) -> None:
    """True against predicted values on validation, one panel per feature set (two per row)."""
    rows = (len(panels) + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(6.4, 3.0 * rows), squeeze=False,
                             gridspec_kw={"wspace": 0.28, "hspace": 0.42})
    allv = np.concatenate([np.concatenate([p["y"], p["pred"]]) for p in panels])
    lo, hi = np.percentile(allv, [0.5, 99.5])
    pad = 0.05 * (hi - lo)
    lim = (lo - pad, hi + pad)
    for ax, p in zip(axes.flat, panels):
        ax.plot(lim, lim, color=NEUTRAL, lw=1.0, ls=(0, (3, 2)), zorder=1)
        ax.scatter(p["y"], p["pred"], s=6, color=BLUE, edgecolors="white", linewidths=0.2, zorder=2)
        s = p["score"]
        ax.text(0.03, 0.97, f"R² {s['r2']:.2f} [{s['r2_ci'][0]:.2f}, {s['r2_ci'][1]:.2f}]\n"
                f"RMSE {s['rmse']:.2f} {unit} [{s['rmse_ci'][0]:.2f}, {s['rmse_ci'][1]:.2f}]\n{PROBE_NAME[p['probe']]}",
                transform=ax.transAxes, ha="left", va="top", fontsize=6.8, color=INK2, linespacing=1.35)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal")
        style(ax, grid="both")
        ax.set_title(p["title"], fontsize=8, color=INK, loc="left")
        axis_labels(ax, f"true {prop_label} ({unit})", f"predicted ({unit})")
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    save(fig, out, "age_scatter")


def fig_r2(out: Path, results: dict, row_names: list[str], labels: dict) -> None:
    """Validation R² with 95% intervals for every feature set, probe and modality."""
    fig, axes = plt.subplots(1, len(MODS), figsize=(7.6, 0.32 * len(row_names) + 1.2), sharey=True,
                             gridspec_kw={"wspace": 0.08})
    ypos = {n: i for i, n in enumerate(row_names)}
    offset = {"linear": -0.22, "knn": 0.0, "mlp": 0.22}
    for ax, m in zip(axes, MODS):
        for n in row_names:
            res = results[m].get(n)
            if res is None:
                continue
            for k in (k for k in PROBES if k in res):
                s = res[k]["score"]
                y = ypos[n] + offset[k]
                ax.plot(s["r2_ci"], [y, y], color=PROBE_COLOUR[k], lw=1.0, solid_capstyle="butt")
                ax.plot([s["r2"]], [y], ls="", marker="o", ms=3.8, color=PROBE_COLOUR[k], mec="white", mew=0.4)
        ax.axvline(0, color=AXIS, lw=0.8)
        style(ax, grid="x")
        ax.set_xlim(-0.1, 1.04)                    # room for a dot at R² 1
        ax.set_title(MOD_NAME[m], fontsize=8.5, color=INK, loc="left")
        axis_labels(ax, "R² on validation")
    axes[0].set_yticks(range(len(row_names)), [labels.get(n, n) for n in row_names], fontsize=7.2, color=INK2)
    axes[0].invert_yaxis()
    handles = [Line2D([], [], color=PROBE_COLOUR[k], lw=1.0, marker="o", ms=3.8, mec="white", mew=0.4,
                      label=PROBE_NAME[k]) for k in PROBES]
    leg = fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.55, 0.0), ncol=3, frameon=False,
                     fontsize=7.5, handletextpad=0.3, columnspacing=1.4)
    for t in leg.get_texts():
        t.set_color(INK2)
    save(fig, out, "age_r2")


# ------------------------------------------------------------------------------------------------------------- main
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", nargs="+", type=Path, required=True, help="run folders (config.json + best.pt)")
    ap.add_argument("--labels", nargs="+", default=None, help="a short label per run for tables and figures")
    ap.add_argument("--untrained", type=Path, default=None, help="probe this run's architecture untrained too")
    ap.add_argument("--data", type=Path, required=True, help="the dataset's permanent folder")
    ap.add_argument("--targets", type=Path, required=True,
                    help="dataset folder whose snap_099/targets.h5 holds the property")
    ap.add_argument("--property", default="age_mean_gyr")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None, help="only N galaxies, spread over each split (code checks)")
    args = ap.parse_args(argv)
    if args.labels is not None and len(args.labels) != len(args.runs):
        raise SystemExit("give one label per run")
    args.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"pdf.fonttype": 42, "font.family": "DejaVu Sans", "axes.unicode_minus": True})
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    runs = {}
    for run in args.runs:
        if run.name in runs:
            raise SystemExit(f"run folder names must differ: {run.name}")
        runs[run.name] = load_run(run, data=args.data, overrides=["snapshots=[99]"])
    untrained = None
    if args.untrained is not None:
        cfg = TrainConfig.from_file(str(args.untrained / "config.json"), [f"data={args.data}", "snapshots=[99]"])
        untrained = untrained_model(cfg)
    labels = dict(zip(runs, args.labels or list(runs)))
    labels["untrained"] = "untrained network"
    feats, items = features(args, runs, untrained)

    targets = load_targets(args.targets, 99)
    if args.property not in targets:
        raise SystemExit(f"{args.property} is not in targets.h5 ({sorted(targets)})")
    rng = np.random.default_rng(0)
    results: dict = {}
    scatter_panels = []
    for m in MODS:
        mi = MODALITIES.index(m)
        prop = {s: lookup(targets, items[s]["subhalo"]) for s in ("train", "val")}
        keep = {s: items[s]["mask"][:, mi].astype(bool) & np.isfinite(prop[s][args.property])
                & np.isfinite(prop[s]["log_mstar_total"]) for s in ("train", "val")}
        y = {s: prop[s][args.property][keep[s]] for s in ("train", "val")}
        mass = {s: prop[s]["log_mstar_total"][keep[s]][:, None] for s in ("train", "val")}
        sets = {s: feature_sets(feats[s], m, list(runs), untrained is not None) for s in ("train", "val")}
        sets = {s: {n: (x[keep[s]], e) for n, (x, e) in d.items()} for s, d in sets.items()}
        sets["train"]["stellar mass"], sets["val"]["stellar mass"] = (mass["train"], False), (mass["val"], False)
        # the stellar-mass trend, for "at fixed mass": nearest neighbours in log M* (k chosen on the training set)
        trend = GridSearchCV(pipeline("knn", False), GRIDS["knn"], scoring="r2",
                             cv=KFold(5, shuffle=True, random_state=0)).fit(mass["train"], y["train"])
        res_tr = y["train"] - cross_val_predict(trend.best_estimator_, mass["train"], y["train"],
                                                cv=KFold(5, shuffle=True, random_state=1))
        res_va = y["val"] - trend.predict(mass["val"])
        results[m] = {"galaxies": {"train": int(keep["train"].sum()), "val": int(keep["val"].sum())},
                      "spread_val": float(y["val"].std())}
        for n, (xtr, emb) in sets["train"].items():
            xva = sets["val"][n][0]
            row = {}
            for k in PROBES:
                if k == "mlp" and n in FEW:
                    continue
                pred, cv, chosen = probe(k, emb, xtr, y["train"], xva)
                row[k] = {"score": scores(y["val"], pred, rng), "cv_r2_train": cv, "chosen": chosen,
                          "pred": pred}
            if n != "stellar mass":
                pred, cv, _ = probe("linear", emb, xtr, res_tr, xva)
                row["linear_at_fixed_mass"] = {"r2": r2(res_va, pred), "cv_r2_train": cv}
            results[m][n] = row
            print(f"{m:5s} {labels.get(n, n):34s} " + "  ".join(
                f"{k} R² {row[k]['score']['r2']:5.2f} RMSE {row[k]['score']['rmse']:.2f}" for k in PROBES if k in row)
                + (f"  | at fixed mass {row['linear_at_fixed_mass']['r2']:5.2f}" if n != "stellar mass" else ""),
                flush=True)
        if m == "stars":
            for n in [*runs, "stats with age", "stats, no age"]:
                best = max((k for k in PROBES if k in results[m][n]),
                           key=lambda k: results[m][n][k]["cv_r2_train"])               # chosen on training only
                scatter_panels.append({"title": labels.get(n, n), "y": y["val"], "pred": results[m][n][best]["pred"],
                                       "score": results[m][n][best]["score"], "probe": best})

    unit = "Gyr" if args.property == "age_mean_gyr" else ""
    prop_label = "mean stellar age" if args.property == "age_mean_gyr" else args.property
    fig_scatter(args.out, scatter_panels, prop_label, unit)
    row_names = [*runs, *(["untrained"] if untrained is not None else []), *STAT_ROWS, "stellar mass"]
    fig_r2(args.out, results, row_names, labels)
    clean = {m: {n: ({k: {kk: vv for kk, vv in v.items() if kk != "pred"} if isinstance(v, dict) else v
                      for k, v in row.items()} if isinstance(row, dict) else row)
                 for n, row in res.items()} for m, res in results.items()}
    (args.out / "results.json").write_text(json.dumps(
        {"property": args.property, "labels": labels, "runs": {n: str(r) for n, r in zip(runs, args.runs)},
         "results": clean}, indent=1) + "\n")
    print(f"wrote {args.out / 'results.json'}, age_scatter and age_r2")


if __name__ == "__main__":
    main()
