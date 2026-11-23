"""Pictures of a trained model's latent space: where the validation galaxies' embeddings lie.

    PYTHONPATH=pylibs:. python -s tools/latent_space.py --run outputs/clip/stage_e/e18_complete_independent \
        --data /cluster/projects/ska/illustris/tng50-1.pointclouds/complete \
        --targets /cluster/projects/ska/illustris/tng50-1.pointclouds/v3 \
        --out outputs/clip/stage_e/e18_complete_independent/latent [--device cuda] [--figures properties ...]

Only validation galaxies (their frozen clouds, in the run's rotation mode); the test set stays untouched. The run's
best.pt embeds every validation galaxy of every snapshot once; the embeddings are cached in OUT/embeddings.npz
(--recompute redoes them). A galaxy with too few gas cells has no gas embedding (the dataset's mask); every figure
uses only the embeddings that exist. 2-d maps are t-SNE on cosine distance (fixed seed): nearby points are similar,
but distances between far-apart groups and the axes mean nothing. Figures (PDF + PNG, the style of clip_report.py):

  latent_modalities   one map of snapshot 99's stars, dark-matter and gas embeddings together: as they are (each
                      modality keeps to its own region, the "modality gap" of contrastive models), and with each
                      modality's mean embedding removed, by modality and by stellar mass
  latent_properties   one map per modality, coloured by physical properties (snapshot 99's targets.h5), with two R²
                      of predicting each property from the 512-d embedding: from a galaxy's 10 nearest neighbours
                      (leave-one-out; what the map's neighbourhoods show) and from a linear fit (ridge, 5-fold
                      cross-validation; whether the property can be read off along some direction)
  latent_similarity   cosine similarity between two modalities' embeddings for every pair of galaxies, sorted by
                      stellar mass, and its distribution for the same galaxy against all other pairs
  latent_redshift     the stellar embeddings of the validation galaxies of all snapshots, by redshift, and how far a
                      galaxy's embedding at each redshift lies from its own z = 0 embedding (SubLink main branch)

OUT/summary.json keeps the numbers: both R² per modality and property, similarity statistics, retrieval R@1 per snapshot
and pair, and the redshift drift; for a run trained with shared rotations also snapshot 99's R@1 with every modality
rotated on its own (embeddings_independent_z0.npz), which removes the orientation shortcut.
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
from clip_report import AQUA, AXIS, BLUE, INK, INK2, MUTED, NEUTRAL, ORANGE, axis_labels, save, style  # noqa: E402
from galaxyclip3d.evaluation import embed, load_run, run_dataset                                    # noqa: E402
from galaxyclip3d.multimodal import MODALITIES                                                      # noqa: E402
from galaxyclip3d.pointclouds.lineage import read_branches, z0_descendants                          # noqa: E402
from galaxyclip3d.probes import load_targets, lookup                                                # noqa: E402

import matplotlib.pyplot as plt                                                                      # noqa: E402
from matplotlib.colors import LinearSegmentedColormap                                                # noqa: E402
from matplotlib.lines import Line2D                                                                  # noqa: E402
from matplotlib.ticker import MaxNLocator                                                            # noqa: E402
from sklearn.linear_model import RidgeCV                                                             # noqa: E402
from sklearn.manifold import TSNE                                                                    # noqa: E402
from sklearn.model_selection import KFold, cross_val_predict                                         # noqa: E402

MODS = ("stars", "dm", "gas")
PAIRS = (("stars", "dm"), ("stars", "gas"), ("dm", "gas"))
MOD_NAME = {"stars": "stars", "dm": "dark matter", "gas": "gas"}
MOD_COLOUR = {"stars": BLUE, "dm": ORANGE, "gas": AQUA}           # categorical slots 1-3 in their fixed order
# the sequential blue ramp of clip_report.py without its two lightest steps, so every point stays visible on white
SEQ = LinearSegmentedColormap.from_list("blue", ["#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
MISSING = "#d9d8d2"
PROPS = (("log_mstar_total", "stellar mass", "log M☉"), ("log_m200c", "halo mass M200c", "log M☉"),
         ("log_ssfr", "specific SFR", "log yr⁻¹ (grey: none)"), ("age_mean_gyr", "mean stellar age", "Gyr"),
         ("rhalf_star_kpc", "half-mass radius", "log kpc"), ("gas_fraction", "gas fraction", "of all mass"))
FIGURES = ("modalities", "properties", "similarity", "redshift")
Z0 = 99


def tsne(x: np.ndarray, seed: int = 0) -> np.ndarray:
    perplexity = min(30.0, (len(x) - 1) / 3)
    return TSNE(2, metric="cosine", init="pca", perplexity=perplexity, random_state=seed).fit_transform(x)


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def knn_r2(x: np.ndarray, y: np.ndarray, k: int = 10) -> float:
    """Leave-one-out R² of predicting y from the mean of each galaxy's k nearest neighbours (cosine, in x)."""
    ok = np.isfinite(y)
    x, y = x[ok], y[ok]
    sim = x @ x.T
    np.fill_diagonal(sim, -np.inf)
    near = np.argsort(-sim, axis=1)[:, :k]
    pred = y[near].mean(axis=1)
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def linear_r2(x: np.ndarray, y: np.ndarray, seed: int = 0) -> float:
    """R² of a ridge fit (strength chosen inside each training fold) predicting y, 5-fold cross-validation."""
    ok = np.isfinite(y)
    pred = cross_val_predict(RidgeCV(alphas=np.logspace(-3, 4, 15)), x[ok], y[ok],
                             cv=KFold(5, shuffle=True, random_state=seed))
    return float(1 - ((y[ok] - pred) ** 2).sum() / ((y[ok] - y[ok].mean()) ** 2).sum())


def r1(a: np.ndarray, b: np.ndarray) -> float:
    """Retrieval R@1 in % between two modalities' embeddings of the same galaxies, mean of both directions."""
    sim = a @ b.T
    n = len(sim)
    return float(100 * ((sim.argmax(1) == np.arange(n)).mean() + (sim.argmax(0) == np.arange(n)).mean()) / 2)


def bare(ax) -> None:
    """A map panel: t-SNE axes carry no units, so no ticks; a light frame keeps the panels apart."""
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(AXIS)
        s.set_linewidth(0.6)


def title(ax, text: str, size: float = 8.5) -> None:
    ax.set_title(text, fontsize=size, color=INK, loc="left")


def colourbar(fig, mappable, ax, label: str, **kw):
    cb = fig.colorbar(mappable, ax=ax, **kw)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6.5, colors=MUTED, labelcolor=INK2, length=2)
    cb.set_label(label, fontsize=7, color=INK2)
    return cb


def modality_handles(counts: dict) -> list:
    return [Line2D([], [], ls="", marker="o", ms=4.5, mfc=MOD_COLOUR[m], mec="white", mew=0.3,
                   label=f"{MOD_NAME[m]} ({counts[m]})") for m in MODS]


def fig_legend(fig, handles, ncol: int, y: float = -0.02) -> None:
    leg = fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol, frameon=False,
                     fontsize=7.5, handletextpad=0.3, columnspacing=1.2)
    for t in leg.get_texts():
        t.set_color(INK2)


# ------------------------------------------------------------------------------------------------------- embeddings
def embeddings(args) -> dict:
    cache = args.out / "embeddings.npz"
    if cache.exists() and not args.recompute:
        print(f"embeddings from {cache}")
        return dict(np.load(cache))
    cfg, model, ck = load_run(args.run, data=args.data)
    ds = run_dataset(cfg, "val")
    print(f"{len(ds)} validation galaxies, checkpoint of epoch {ck.get('epoch')}, embedding on {args.device}")
    torch.set_num_threads(max(1, args.threads))
    res = embed({"run": model}, ds, device=args.device, batch_size=args.batch, workers=args.workers)
    out = {m: res["run"][m] for m in MODS}
    out.update(subhalo=res["subhalo"], snapshot=res["snapshot"], mstar=res["mstar"], mask=res["mask"],
               epoch=np.array(ck.get("epoch", -1)))
    np.savez(cache, **out)
    return out


def independent_check(args) -> dict | None:
    """For a run trained with shared rotations: R@1 of snapshot 99's galaxies when each modality is rotated on its
    own, next to the run's own mode. Shared rotations let a model match galaxies by their orientation (the shortcut);
    independent rotations remove it. None for a run that already uses independent rotations."""
    if json.loads((args.run / "config.json").read_text())["rotation"] == "independent":
        return None
    cache = args.out / "embeddings_independent_z0.npz"
    if cache.exists() and not args.recompute:
        e = dict(np.load(cache))
    else:
        cfg, model, _ = load_run(args.run, data=args.data, overrides=["snapshots=[99]"])
        res = embed({"run": model}, run_dataset(cfg, "val", rotation="independent"), device=args.device,
                    batch_size=args.batch, workers=args.workers)
        e = {m: res["run"][m] for m in MODS}
        e.update(mask=res["mask"], subhalo=res["subhalo"])
        np.savez(cache, **e)
    has = {m: e["mask"][:, MODALITIES.index(m)].astype(bool) for m in MODS}
    return {f"{a}-{b}": {"galaxies": int((has[a] & has[b]).sum()),
                         "r1": r1(e[a][has[a] & has[b]], e[b][has[a] & has[b]])} for a, b in PAIRS}


def z0_of(data: Path, snap: np.ndarray, subhalo: np.ndarray) -> np.ndarray:
    """Each galaxy's subhalo at snapshot 99 along its SubLink main branch (-1 if the branch does not reach it)."""
    desc = np.full(len(snap), -1, dtype=np.int64)
    for s in np.unique(snap):
        rows = np.flatnonzero(snap == s)
        branches = read_branches(data / f"snap_{int(s):03d}" / "lineage.h5")
        desc[rows] = z0_descendants(int(s), subhalo[rows], branches)[0]
    return desc


# ---------------------------------------------------------------------------------------------------------- figures
def joint_map(E: dict, rows: np.ndarray, sel: dict, seed: int, centre: bool) -> dict:
    """t-SNE of every existing embedding of `rows`, all modalities together; with `centre`, each modality's mean
    embedding is removed first (and the rest normalized again). Returns {modality: (len(rows), 2), NaN if missing}."""
    parts = []
    for m in MODS:
        x = E[m][rows][sel[m]]
        parts.append(unit(x - x.mean(axis=0)) if centre else x)
    xy_all = tsne(np.concatenate(parts), seed)
    xy, start = {}, 0
    for m in MODS:
        xy[m] = np.full((len(rows), 2), np.nan)
        xy[m][sel[m]] = xy_all[start:start + sel[m].sum()]
        start += sel[m].sum()
    return xy


def fig_modalities(out: Path, E: dict, rows: np.ndarray, mstar: np.ndarray, has: dict, seed: int) -> dict:
    sel = {m: has[m][rows] for m in MODS}
    raw = joint_map(E, rows, sel, seed, centre=False)
    cen = joint_map(E, rows, sel, seed, centre=True)
    full = np.flatnonzero(np.logical_and.reduce([sel[m] for m in MODS]))
    linked = np.random.default_rng(seed).choice(full, size=min(10, len(full)), replace=False)
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.95), gridspec_kw={"wspace": 0.05})
    for ax, xy, text in ((axes[0], raw, "(a) as they are"), (axes[1], cen, "(b) modality means removed")):
        if xy is cen:
            for i in linked:                               # one galaxy's three embeddings, joined
                tri = np.stack([xy[m][i] for m in (*MODS, MODS[0])])
                ax.plot(tri[:, 0], tri[:, 1], color=INK2, lw=0.6, alpha=0.6, zorder=3)
        for m in MODS:
            ax.scatter(*xy[m][sel[m]].T, s=5, color=MOD_COLOUR[m], edgecolors="white", linewidths=0.2, zorder=2)
        bare(ax)
        title(ax, text)
    axes[1].text(0.99, 0.01, f"lines: {len(linked)} galaxies", transform=axes[1].transAxes, ha="right", va="bottom",
                 fontsize=6.8, color=MUTED)
    ax = axes[2]
    allxy = np.concatenate([cen[m][sel[m]] for m in MODS])
    c = np.concatenate([mstar[sel[m]] for m in MODS])
    order = np.argsort(c)                                  # heaviest on top
    sc = ax.scatter(*allxy[order].T, c=c[order], cmap=SEQ, s=5, edgecolors="white", linewidths=0.2)
    bare(ax)
    title(ax, "(c) (b) by stellar mass")
    colourbar(fig, sc, ax, "stellar mass (log M☉)", fraction=0.05, pad=0.02, aspect=26)
    fig_legend(fig, modality_handles({m: int(sel[m].sum()) for m in MODS}), ncol=3, y=0.08)
    save(fig, out, "latent_modalities")
    # the gap in numbers: mean cosine similarity of different galaxies within one modality, and across modalities
    within = {m: float(((s := E[m][rows][sel[m]] @ E[m][rows][sel[m]].T).sum() - np.trace(s)) / (len(s) ** 2 - len(s)))
              for m in MODS}
    return {"within_modality_other_galaxies": within}


def fig_properties(out: Path, E: dict, rows: np.ndarray, props: dict, has: dict, seed: int) -> tuple[dict, dict]:
    fig, axes = plt.subplots(len(MODS), len(PROPS), figsize=(7.6, 4.6),
                             gridspec_kw={"wspace": 0.05, "hspace": 0.08})
    scores: dict = {}
    linear: dict = {}
    for r, m in enumerate(MODS):
        sel = has[m][rows]
        x = E[m][rows][sel]
        xy = tsne(x, seed)
        for c, (key, label, unit_) in enumerate(PROPS):
            ax = axes[r, c]
            y = props[key][sel]
            ok = np.isfinite(y)
            ax.scatter(*xy[~ok].T, s=4, color=MISSING, linewidths=0)
            order = np.flatnonzero(ok)[np.argsort(y[ok])]
            vmin, vmax = np.nanpercentile(y, [2, 98])
            sc = ax.scatter(*xy[order].T, c=y[order], cmap=SEQ, vmin=vmin, vmax=vmax, s=5, edgecolors="white",
                            linewidths=0.2)
            score, lin = knn_r2(x, y), linear_r2(x, y, seed)
            scores.setdefault(m, {})[key] = score
            linear.setdefault(m, {})[key] = lin
            ax.text(0.03, 0.97, f"R² {score:.2f} / {lin:.2f}", transform=ax.transAxes, ha="left", va="top",
                    fontsize=6.3, color=INK2, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
            bare(ax)
            if r == 0:
                title(ax, label, size=7.5)
            if c == 0:
                ax.set_ylabel(f"{MOD_NAME[m]} ({sel.sum()})", fontsize=8.5, color=INK)
            if r == len(MODS) - 1:
                cb = colourbar(fig, sc, axes[:, c], unit_, orientation="horizontal", fraction=0.04, pad=0.025,
                               aspect=16, shrink=0.8)
                cb.ax.xaxis.set_major_locator(MaxNLocator(2, steps=[1, 2, 2.5, 5, 10]))   # neighbouring bars sit close
    fig.text(0.125, 0.035, "R² in each panel, in the full 512-d space: from the 10 nearest neighbours / from a linear "
             "fit (5-fold cross-validation)", ha="left", va="top", fontsize=7, color=MUTED)
    save(fig, out, "latent_properties")
    return scores, linear


def fig_similarity(out: Path, E: dict, rows: np.ndarray, mstar: np.ndarray, has: dict) -> dict:
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.0), gridspec_kw={"height_ratios": [2.3, 1], "hspace": 0.38,
                                                                    "wspace": 0.18})
    stats: dict = {}
    for c, (a, b) in enumerate(PAIRS):
        sel = has[a][rows] & has[b][rows]                  # galaxies with both modalities, by stellar mass
        order = rows[sel][np.argsort(mstar[sel])]
        sim = E[a][order] @ E[b][order].T
        n = len(sim)
        true, other = np.diag(sim), sim[~np.eye(n, dtype=bool)]
        stats[f"{a}-{b}"] = {"galaxies": int(n), "r1": r1(E[a][order], E[b][order]),
                             "same_galaxy_mean": float(true.mean()), "other_pairs_mean": float(other.mean())}
        ax = axes[0, c]
        im = ax.imshow(sim, cmap=SEQ, vmin=np.percentile(other, 2), vmax=np.percentile(true, 95),
                       interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        title(ax, f"{MOD_NAME[a]} × {MOD_NAME[b]}\n{n} galaxies, R@1 {stats[f'{a}-{b}']['r1']:.1f}%", size=8)
        ax.set_xlabel(f"{MOD_NAME[b]}, by stellar mass →", fontsize=7, color=MUTED, labelpad=2)
        ax.set_ylabel(f"← {MOD_NAME[a]}, by stellar mass", fontsize=7, color=MUTED, labelpad=2)
        colourbar(fig, im, ax, "cosine similarity", orientation="horizontal", fraction=0.05, pad=0.08, aspect=22)
        ax = axes[1, c]
        bins = np.linspace(-0.5, 1, 61)
        mid = (bins[1:] + bins[:-1]) / 2
        for vals, colour in ((other, NEUTRAL), (true, BLUE)):
            h, _ = np.histogram(vals, bins, density=True)
            ax.plot(mid, h, color=colour, lw=1.6)
        ax.text(0.02, 0.97, f"means\nsame {true.mean():.2f}\nothers {other.mean():.2f}", transform=ax.transAxes,
                ha="left", va="top", fontsize=6.8, color=INK2, linespacing=1.3)
        style(ax, grid=None)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        axis_labels(ax, "cosine similarity")
    fig_legend(fig, [Line2D([], [], color=BLUE, lw=1.6, label="the same galaxy"),
                     Line2D([], [], color=NEUTRAL, lw=1.6, label="all other pairs of galaxies")], ncol=2, y=0.03)
    save(fig, out, "latent_similarity")
    return stats


def fig_redshift(out: Path, E: dict, has: dict, snap: np.ndarray, desc: np.ndarray, redshift: dict,
                 seed: int) -> dict:
    snaps = sorted(redshift, key=lambda s: -redshift[s])          # z = 1 first, z = 0 last
    shades = dict(zip(snaps, SEQ(np.linspace(0.0, 1.0, len(snaps)))))   # light = early, dark = today
    xy = tsne(E["stars"], seed)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3), gridspec_kw={"wspace": 0.22, "width_ratios": [1, 1.05]})
    ax = axes[0]
    for s in snaps:
        rows = snap == s
        ax.scatter(*xy[rows].T, s=4, color=shades[s], edgecolors="white", linewidths=0.15)
    bare(ax)
    title(ax, f"(a) stellar embeddings by redshift ({len(snap)})")
    handles = [Line2D([], [], ls="", marker="o", ms=4.5, mfc=shades[s], mec="white", mew=0.3,
                      label=f"z = {redshift[s]:.1f}") for s in snaps]
    leg = ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=3, frameon=False,
                    fontsize=6.8, handletextpad=0.1, columnspacing=0.8)
    for t in leg.get_texts():
        t.set_color(INK2)
    # drift: cosine similarity between a galaxy's embedding at each snapshot and its own z = 0 embedding
    ax = axes[1]
    z0_row = {int(d): i for i, (d, s) in enumerate(zip(desc, snap)) if s == Z0}
    rng = np.random.default_rng(seed)
    drift: dict = {}
    zs = [redshift[s] for s in snaps]
    for m in MODS:
        med, lo, hi, base = [], [], [], []
        for s in snaps:
            prog = [i for i in np.flatnonzero((snap == s) & has[m]) if int(desc[i]) in z0_row
                    and has[m][z0_row[int(desc[i])]]]
            mine = np.array([z0_row[int(desc[i])] for i in prog])
            v = np.einsum("ij,ij->i", E[m][prog], E[m][mine])
            other = rng.permutation(mine)                       # the same z = 0 galaxies, shuffled partners
            keep = other != mine
            b = np.einsum("ij,ij->i", E[m][prog][keep], E[m][other[keep]])
            med.append(np.median(v))
            lo.append(np.percentile(v, 16))
            hi.append(np.percentile(v, 84))
            base.append(np.median(b))
            drift.setdefault(m, {})[f"z={redshift[s]:.2f}"] = {"galaxies": len(prog), "median": float(np.median(v)),
                                                                 "random_median": float(np.median(b))}
        ax.fill_between(zs, lo, hi, color=MOD_COLOUR[m], alpha=0.14, linewidth=0)
        ax.plot(zs, med, color=MOD_COLOUR[m], lw=1.8, label=MOD_NAME[m])
        ax.text(zs[0] + 0.02, med[0], MOD_NAME[m], ha="left", va="center", fontsize=7, color=INK2, clip_on=False)
        ax.plot(zs, base, color=MOD_COLOUR[m], lw=1.0, ls=(0, (3, 2)))
    style(ax, grid="y")
    axis_labels(ax, "redshift z", "similarity to its own z = 0 embedding")
    ax.set_xlim(0, max(zs))
    ax.set_ylim(min(0, ax.get_ylim()[0]), 1.02)
    title(ax, "(b) drift from the z = 0 embedding")
    ax.text(0.99, 0.99, "solid: median, band: 16-84% of galaxies\ndashed: a random other galaxy (median)",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.5, color=MUTED, linespacing=1.4)
    save(fig, out, "latent_redshift")
    return drift


def retrieval_by_snapshot(E: dict, has: dict, snap: np.ndarray, redshift: dict) -> dict:
    out = {}
    for s in sorted(redshift, key=lambda s: redshift[s]):
        row = {}
        for a, b in PAIRS:
            sel = (snap == s) & has[a] & has[b]
            row[f"{a}-{b}"] = {"galaxies": int(sel.sum()), "r1": r1(E[a][sel], E[b][sel])}
        out[f"{s} (z={redshift[s]:.2f})"] = row
    return out


# ------------------------------------------------------------------------------------------------------------- main
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", type=Path, required=True, help="run folder (config.json + best.pt)")
    ap.add_argument("--data", type=Path, required=True, help="the dataset's permanent folder")
    ap.add_argument("--targets", type=Path, required=True,
                    help="dataset folder whose snap_099/targets.h5 holds the physical properties")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--figures", nargs="+", choices=FIGURES, default=list(FIGURES),
                    help="redraw only these figures (summary.json keeps the other figures' numbers)")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"pdf.fonttype": 42, "font.family": "DejaVu Sans", "axes.unicode_minus": True})

    E = embeddings(args)
    has = {m: E["mask"][:, MODALITIES.index(m)].astype(bool) for m in MODS}   # gas needs enough gas cells
    snap, sub = E["snapshot"].astype(int), E["subhalo"].astype(np.int64)
    meta = json.loads((args.data / "snapshots.json").read_text())
    redshift = {int(s): float(meta[str(s)]["redshift"]) for s in np.unique(snap)}
    z0 = np.flatnonzero(snap == Z0)
    props = lookup(load_targets(args.targets, Z0), sub[z0])
    mstar = props["log_mstar_total"]
    print(f"snapshot 99: {len(z0)} galaxies, {has['gas'][z0].sum()} with gas; all snapshots: {len(snap)} "
          f"({', '.join(f'{s}: {(snap == s).sum()}' for s in sorted(redshift))})")

    path = args.out / "summary.json"
    summary = json.loads(path.read_text()) if path.exists() else {}
    summary.update(run=str(args.run), epoch=int(E["epoch"]), galaxies_z0=int(len(z0)), galaxies_all=int(len(snap)),
                   retrieval_by_snapshot=retrieval_by_snapshot(E, has, snap, redshift))
    if "modalities" in args.figures:
        summary["gap"] = fig_modalities(args.out, E, z0, mstar, has, args.seed)
    if "properties" in args.figures:
        summary["knn_r2"], summary["linear_r2"] = fig_properties(args.out, E, z0, props, has, args.seed)
    if "similarity" in args.figures:
        summary["similarity"] = fig_similarity(args.out, E, z0, mstar, has)
    if "redshift" in args.figures:
        desc = z0_of(args.data, snap, sub)
        summary["drift"] = fig_redshift(args.out, E, has, snap, desc, redshift, args.seed)
    check = independent_check(args)
    if check is not None:
        summary["r1_z0_independent_rotation"] = check
    path.write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
