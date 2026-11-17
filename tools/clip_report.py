"""Thesis figures and tables for the GalaxyCLIP-3D alignment study (Phases B-D), built from saved files only: the runs'
history.json, the Phase D scoring files in outputs/clip/phase_d (tools/eval_clip.py, tools/probe_clip.py,
tools/clip_baselines.py --split test), the validation baselines (outputs/clip_baselines/baselines.json) and the cached
test embeddings of tools/probe_clip.py (for bootstrap intervals). Nothing is trained or embedded again.

    PYTHONPATH=pylibs:. python -s tools/clip_report.py [--clip outputs/clip] [--out outputs/clip/report]

Writes figures/*.pdf + .png, tables/*.tex (booktabs) and report.md (every table, the figures and suggested captions).
Retrieval scores are full-gallery R@1; "10 directions" leaves out gas <-> HI, whose two clouds come from the same gas
cells. Phase C numbers are the best validation score logged during training (GPU, bf16); Phase D numbers are re-scored
on CPU in float32 (they match the logs within 0.2 point). Bootstrap intervals resample the test galaxies (queries) 4,000 times
with the galleries fixed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
import numpy as np                                                 # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgb      # noqa: E402
from matplotlib.lines import Line2D                                # noqa: E402

MODS = ("stars", "dm", "gas", "hi")
MOD_NAME = {"stars": "stars", "dm": "dark matter", "gas": "gas", "hi": "HI"}
PAIRS = ("stars-dm", "stars-gas", "stars-hi", "dm-gas", "dm-hi", "gas-hi")
PAIR_NAME = {"stars-dm": "stars – dark matter", "stars-gas": "stars – gas", "stars-hi": "stars – HI",
             "dm-gas": "dark matter – gas", "dm-hi": "dark matter – HI", "gas-hi": "gas – HI"}
FINAL = ("phase_c/c21_combined", "phase_c/c22_combined_seed2", "phase_c/c28_combined_seed3")
FINAL_NAMES = tuple(Path(r).name for r in FINAL)
WITH_INVARIANTS = ("phase_c/c25_final", "phase_c/c26_final_seed2", "phase_c/c27_final_seed3")

# colours: the dataviz reference palette, categorical slots 1-5 in their fixed order (validated as an adjacent set);
# each model keeps its colour in every figure
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#e34948"
INK, INK2, MUTED, GRID, AXIS, BAND, NEUTRAL = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#f0efec", "#b3b2aa"
FEATURES = (("final", "final setup (3 seeds)", BLUE), ("pointnet_baseline", "Phase B PointNet", ORANGE),
            ("pointnet_shared_rotation", "shared-rotation control", AQUA),
            ("statistics", "hand-made statistics", YELLOW), ("untrained", "untrained network", MAGENTA))
PROPS = (("log_mstar_total", "stellar mass"), ("log_mdm_total", "dark-matter mass"), ("log_mgas_total", "gas mass"),
         ("log_mbh", "black-hole mass"), ("log_m200c", "host halo mass M200c"), ("gas_fraction", "gas fraction"),
         ("log_mgas_sphere", "gas mass in the sphere"), ("log_mhi_sphere", "HI mass in the sphere"),
         ("log_sfr", "star-formation rate"), ("log_ssfr", "specific star-formation rate"),
         ("age_mean_gyr", "mean stellar age"), ("g_minus_r", "g – r colour"), ("log_z_stars", "stellar metallicity"),
         ("log_z_gas", "gas metallicity"), ("veldisp", "velocity dispersion"), ("spin", "spin"),
         ("rhalf_star_kpc", "stellar half-mass radius"), ("vmax", "maximum circular velocity"))
NOISE = 2.0          # Phase C rule: a change counts if it moves the best validation score by more than 2 points

# Phase C rows: (group, label, runs; several runs = mean over seeds). Same 520 validation galaxies as the base.
ABLATIONS = (
    ("reference", "Phase B PointNet", ("pointnet_baseline",)),
    ("reference", "Phase B PointNet, second seed", ("phase_c/c01_seed2",)),
    ("training", "batch 128 (instead of 256)", ("phase_c/c02_batch128",)),
    ("training", "batch 512", ("phase_c/c03_batch512",)),
    ("training", "learned temperature", ("phase_c/c04_learned_temp",)),
    ("training", "600 epochs (instead of 300)", ("phase_c/c08_600ep",)),
    ("training", "weight averaging, EMA 0.99", ("phase_c/c07b_ema099",)),
    ("training", "weight averaging, EMA 0.999", ("phase_c/c07_ema",)),
    ("encoder", "narrow PointNet (3.3 M parameters)", ("phase_c/c05_narrow",)),
    ("encoder", "wide PointNet (9.1 M; base 4.9 M)", ("phase_c/c06_wide",)),
    ("encoder", "one encoder shared by all modalities", ("phase_c/c17_shared_encoder",)),
    ("loss", "only the 3 pairs with stars", ("phase_c/c16_stars_anchor",)),
    ("inputs", "1,024 points per cloud (instead of 2,048)", ("phase_c/c09_points1024",)),
    ("inputs", "4,096 points per cloud", ("phase_c/c10_points4096",)),
    ("inputs", "no velocities", ("phase_c/c14_no_velocities",)),
    ("inputs", "no per-point physical values", ("phase_c/c15_no_scalars",)),
    ("inputs", "+ rotation-free point features", ("phase_c/c23_invariants",)),
    ("inputs", "no physical values, + rotation-free features", ("phase_c/c24_invariants_no_scalars",)),
    ("combined", "learned temperature + wide + 600 epochs (3 seeds)", FINAL),
    ("combined", "the same + rotation-free features (3 seeds)", WITH_INVARIANTS),
)
# table-only rows with their own reference: another galaxy set (sphere sizes) or another encoder (patch transformer)
EXTRA = (
    ("sphere size (474 galaxies usable at every size)", "5 half-mass radii (reference)", ("phase_c/c12_r5_common",)),
    ("sphere size (474 galaxies usable at every size)", "3 half-mass radii", ("phase_c/c11_r3_common",)),
    ("sphere size (474 galaxies usable at every size)", "7.5 half-mass radii", ("phase_c/c13_r7p5_common",)),
    ("patch transformer", "Phase B patch transformer (reference)", ("patch_main",)),
    ("patch transformer", "smaller (width 192, depth 4)", ("phase_c/c18_patch_small",)),
    ("patch transformer", "dropout 0.1 + weight decay 0.1", ("phase_c/c19_patch_regularised",)),
    ("patch transformer", "smaller + radially spread patch centres", ("phase_c/c20_patch_radial",)),
)


# ----------------------------------------------------------------------------------------------------------- helpers
def pair(res: dict, p: str, k: str = "r1") -> float:
    a, b = p.split("-")
    return 100 * (res[f"{k}/{a}->{b}"] + res[f"{k}/{b}->{a}"]) / 2


def plain_chance(res: dict) -> float:
    dirs = [k for k in res if k.startswith("chance/") and "->" in k and "gas->hi" not in k and "hi->gas" not in k]
    return 100 * float(np.mean([res[k] for k in dirs]))


def best_of(clip: Path, run: str) -> dict:
    """Best validation R@1 (10 directions) during training, its epoch and stars-DM, the last evaluation, galaxies."""
    h = json.loads((clip / run / "history.json").read_text())
    vals = [v for v in h["val"] if v["epoch"] > 0]
    i = max(range(len(vals)), key=lambda k: vals[k]["r1/mean_no_gas_hi"])
    v = vals[i]
    return {"r1": 100 * v["r1/mean_no_gas_hi"], "epoch": v["epoch"], "stars_dm": pair(v, "stars-dm"),
            "final": 100 * vals[-1]["r1/mean_no_gas_hi"], "last_epoch": vals[-1]["epoch"],
            "galaxies": int(v["gallery/stars->dm"])}


def row_of(clip: Path, runs) -> dict:
    b = [best_of(clip, r) for r in runs]
    out = {k: float(np.mean([x[k] for x in b])) for k in ("r1", "epoch", "stars_dm", "final")}
    out.update(lo=min(x["r1"] for x in b), hi=max(x["r1"] for x in b), n=len(b), galaxies=b[0]["galaxies"],
               last_epoch=b[0]["last_epoch"])
    return out


def scored(d: Path, name: str) -> dict:
    """{run name: results} of a tools/eval_clip.py file."""
    return {k: v["results"] for k, v in json.loads((d / name).read_text())["runs"].items()}


def style(ax, grid: str | None = "x") -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelcolor=INK2, labelsize=8, length=3, width=0.8)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def axis_labels(ax, xlabel: str | None = None, ylabel: str | None = None) -> None:
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8.5, color=INK2)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8.5, color=INK2)


def legend(ax, **kw):
    leg = ax.legend(frameon=False, fontsize=7.5, **kw)
    for t in leg.get_texts():
        t.set_color(INK2)
    return leg


def save(fig, out: Path, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def dark(colour) -> bool:
    r, g, b = to_rgb(colour)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.45


def tex(s: str) -> str:
    s = str(s)
    indent = s.startswith("   ")
    for a, b in (("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("±", r"$\pm$"), ("–", "--"), ("−", "$-$"),
                 ("R²", "$R^2$"), ("≥", r"$\ge$"), ("→", r"$\to$"), ("M200c", r"$M_\mathrm{200c}$")):
        s = s.replace(a, b)
    return r"\quad " + s.strip() if indent else s


def signed(x: float, reference: bool = False) -> str:
    return "–" if reference else f"{x:+.1f}".replace("-", "−")


def md_cell(s: str) -> str:
    s = str(s)
    return "&nbsp;&nbsp;&nbsp;" + s.strip() if s.startswith("   ") else s


class Tables:
    """Collects tables: one .tex file each (booktabs) plus a Markdown section for report.md. A row "---" is a rule,
    a row ("#group", text) a group heading; a first cell starting with three spaces is indented."""

    def __init__(self, out: Path):
        self.out, self.md = out, []

    def add(self, name: str, caption: str, header: list, rows: list, note: str = "", align: str | None = None):
        align = align or "l" + "r" * (len(header) - 1)
        wide = len(header) >= 7                                    # scaled to the text width (needs graphicx)
        lines = [r"% generated by tools/clip_report.py; needs \usepackage{booktabs}" + (" and graphicx" if wide else ""),
                 r"\begin{table}[t]", r"\centering", r"\small"] + ([r"\resizebox{\linewidth}{!}{%"] if wide else []) + \
                [rf"\begin{{tabular}}{{{align}}}", r"\toprule", " & ".join(tex(h) for h in header) + r" \\", r"\midrule"]
        md = [f"**{caption}**", "", "| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
        for r in rows:
            if r == "---":
                lines.append(r"\midrule")
            elif isinstance(r, tuple) and r[0] == "#group":
                lines.append(rf"\multicolumn{{{len(header)}}}{{l}}{{\textit{{{tex(r[1])}}}}} \\")
                md.append(f"| *{r[1]}* |" + " |" * (len(header) - 1))
            else:
                lines.append(" & ".join(tex(c) for c in r) + r" \\")
                md.append("| " + " | ".join(md_cell(c) for c in r) + " |")
        lines += [r"\bottomrule", r"\end{tabular}" + ("}" if wide else ""),
                  rf"\caption{{{tex(caption)}{' ' + tex(note) if note else ''}}}", rf"\label{{tab:{name}}}",
                  r"\end{table}", ""]
        (self.out / f"{name}.tex").write_text("\n".join(lines))
        self.md += md + (["", f"*{note}*"] if note else []) + ["", f"(`tables/{name}.tex`)", ""]


# --------------------------------------------------------------------------------------------------------- bootstrap
def hits(npz: Path):
    """Per pair: (rows, hits a->b, hits b->a) over the galaxies with both modalities, from cached embeddings."""
    z = np.load(npz)
    emb, mask = {m: z[f"emb_{m}"].astype(np.float64) for m in MODS}, z["mask"].astype(bool)
    out = {}
    for i, a in enumerate(MODS):
        for j, b in enumerate(MODS):
            if j <= i:
                continue
            rows = np.flatnonzero(mask[:, i] & mask[:, j])
            za, zb = emb[a][rows], emb[b][rows]
            hh = []
            for q, g in ((za, zb), (zb, za)):
                sim = q @ g.T
                hh.append((((sim >= sim.diagonal()[:, None]).sum(1) - 1) < 1).astype(np.float64))
            out[f"{a}-{b}"] = (rows, hh[0], hh[1])
    return out, len(mask)


def bootstrap(d: Path, names, n_boot: int = 4000, seed: int = 0) -> dict:
    """{name: {pair or "mean": (score, resampled scores)}} on test; "mean" = the 10 directions without gas-HI."""
    sets = {n: hits(d / "embeddings" / f"{n}_test.npz") for n in names}
    ngal = next(iter(sets.values()))[1]
    rng = np.random.default_rng(seed)
    w = np.stack([np.bincount(rng.integers(0, ngal, ngal), minlength=ngal) for _ in range(n_boot)]).astype(float)
    out = {}
    for n, (h, _) in sets.items():
        res = {}
        for p, (rows, h1, h2) in h.items():
            ww = w[:, rows]
            per = 50 * ((ww * h1).sum(1) + (ww * h2).sum(1)) / ww.sum(1)
            res[p] = (50 * (h1.mean() + h2.mean()), per)
        plain = [p for p in PAIRS if p != "gas-hi"]
        res["mean"] = (float(np.mean([res[p][0] for p in plain])), np.mean([res[p][1] for p in plain], axis=0))
        out[n] = res
    return out


def interval(x: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))


# ----------------------------------------------------------------------------------------------------------- figures
def fig_curves(clip: Path, out: Path, stats_val: float) -> None:
    fig, ax = plt.subplots(figsize=(6.3, 3.1))
    style(ax, grid="y")
    for i, run in enumerate(FINAL):
        h = json.loads((clip / run / "history.json").read_text())
        ax.plot([v["epoch"] for v in h["val"]], [100 * v["r1/mean_no_gas_hi"] for v in h["val"]], color=BLUE,
                lw=1.2, alpha=0.9, label="final setup (3 seeds)" if i == 0 else None, zorder=3)
    ends = {}
    for run, colour, name in (("pointnet_baseline", ORANGE, "Phase B PointNet"),
                              ("pointnet_shared_rotation", AQUA, "shared-rotation control")):
        h = json.loads((clip / run / "history.json").read_text())
        ep, r = [v["epoch"] for v in h["val"]], [100 * v["r1/mean_no_gas_hi"] for v in h["val"]]
        ax.plot(ep, r, color=colour, lw=1.5, label=name, zorder=2)
        ends[run] = (ep[-1], r[-1])
    ax.axhline(stats_val, color=MUTED, lw=1, ls=(0, (4, 3)), label="statistics + CCA (no network)", zorder=1)
    kw = dict(fontsize=7.5, color=INK2, va="center")
    ax.text(596, 41.5, "final setup, 3 seeds", ha="right", **kw)
    ax.text(ends["pointnet_baseline"][0] + 6, ends["pointnet_baseline"][1] - 4.5, "Phase B PointNet", ha="left", **kw)
    ax.text(ends["pointnet_shared_rotation"][0] + 6, ends["pointnet_shared_rotation"][1], "shared-rotation control",
            ha="left", **kw)
    ax.text(596, stats_val + 2.6, "statistics + CCA, no network", ha="right", **kw)
    ax.set_xlim(0, 600)
    ax.set_ylim(0, 80)
    axis_labels(ax, "epoch", "validation R@1, 10 directions (%)")
    legend(ax, loc="upper right", bbox_to_anchor=(1.0, 0.86), handlelength=1.8)
    save(fig, out, "fig1_training_curves")


def fig_ablations(clip: Path, out: Path, base: float) -> list:
    rows = [(g, lab, row_of(clip, runs)) for g, lab, runs in ABLATIONS]
    groups = []
    for g, _, _ in rows:
        if g not in groups:
            groups.append(g)
    y, ticks, labels, heads = 0.0, [], [], []
    ypos = []
    for g in groups:
        heads.append((y, g))
        y += 1.0
        for gg, lab, r in rows:
            if gg == g:
                ypos.append((y, lab, r))
                y += 1.0
        y += 0.35
    fig, ax = plt.subplots(figsize=(6.3, 0.2 * y + 0.7))
    style(ax, grid="x")
    ax.axvspan(base - NOISE, base + NOISE, color=BAND, zorder=0, lw=0)
    ax.axvline(base, color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=1)
    for yy, lab, r in ypos:
        delta = r["r1"] - base
        colour = BLUE if delta > NOISE else RED if delta < -NOISE else NEUTRAL
        ax.barh(yy, r["r1"], height=0.72, color=colour, lw=0, zorder=2)
        text = f"{r['r1']:.1f}" + (f"  ({r['lo']:.1f}–{r['hi']:.1f})" if r["n"] > 1 else "")
        ax.text(r["r1"] + 0.6, yy, text, va="center", ha="left", fontsize=7, color=INK2)
        ticks.append(yy)
        labels.append(lab)
    for yy, g in heads:
        ax.text(-0.01, yy, g, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=8,
                fontweight="bold", color=INK)
    ax.set_yticks(ticks, labels)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(y - 0.2, -0.8)
    ax.set_xlim(0, 46)
    axis_labels(ax, "best validation R@1, 10 directions (%)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (BLUE, NEUTRAL, RED)]
    leg = ax.legend(handles, [f"better than Phase B by more than {NOISE:.0f} points", f"within ±{NOISE:.0f} points",
                              f"worse by more than {NOISE:.0f} points"], frameon=False, fontsize=7,
                    loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=3, handlelength=1.2, columnspacing=1.2)
    for t in leg.get_texts():
        t.set_color(INK2)
    save(fig, out, "fig2_phase_c_ablations")
    return rows


def fig_pairs(out: Path, stats: dict, boot: dict) -> None:
    rows = [("mean", "mean of 10 directions\n(without gas – HI)")] + [(p, PAIR_NAME[p]) for p in PAIRS[:-1]] + \
           [("gas-hi", "gas – HI\n(same gas cells)")]
    fig, ax = plt.subplots(figsize=(6.3, 3.6))
    style(ax, grid="x")
    h = 0.36
    for i, (key, _) in enumerate(rows):
        y = i + (0.45 if i else 0) + (0.45 if key == "gas-hi" else 0)
        fm = np.mean([boot[n][key][1] for n in FINAL_NAMES], axis=0)
        fv = float(np.mean([boot[n][key][0] for n in FINAL_NAMES]))
        bv, bper = boot["pointnet_baseline"][key]
        for yy, v, per, colour in ((y - h / 2, fv, fm, BLUE), (y + h / 2, bv, bper, ORANGE)):
            lo, hi = interval(per)
            ax.barh(yy, v, height=h * 0.92, color=colour, lw=0, zorder=2)
            ax.plot([lo, hi], [yy, yy], color=INK, lw=0.9, zorder=3, solid_capstyle="butt")
            ax.text(hi + 1.0, yy, f"{v:.1f}", va="center", ha="left", fontsize=7, color=INK2)
        s = stats[key]
        ax.plot([s, s], [y - h - 0.05, y + h + 0.05], color=INK, lw=1.6, zorder=4, solid_capstyle="butt")
        rows[i] = (y, rows[i][1])
    ax.set_yticks([r[0] for r in rows], [r[1] for r in rows])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(rows[-1][0] + 0.7, -0.7)
    ax.set_xlim(0, 105)
    axis_labels(ax, "test R@1 (%), 499 galaxies; whiskers: 95% bootstrap interval")
    handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=ORANGE),
               Line2D([], [], color=INK, lw=1.6)]
    leg = ax.legend(handles, ["final setup (mean of 3 seeds)", "Phase B PointNet", "statistics + CCA (no network)"],
                    frameon=False, fontsize=7.5, loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=3,
                    handlelength=1.4, columnspacing=1.4)
    for t in leg.get_texts():
        t.set_color(INK2)
    save(fig, out, "fig3_test_retrieval_pairs")


def probe_means(P: dict, n: str, key: str) -> list:
    return [float(np.mean([max(P[n][m][p][key], 0.0) for p, _ in PROPS if p in P[n][m]])) for m in MODS]


def feature_means(P: dict, key: str) -> dict:
    out = {}
    for k, _, _ in FEATURES:
        if k == "final":
            per = np.array([probe_means(P, n, key) for n in FINAL_NAMES])
            out[k] = (per.mean(0), per.std(0, ddof=1))
        else:
            out[k] = (np.array(probe_means(P, k, key)), None)
    return out


def fig_probe_means(out: Path, P: dict) -> None:
    means = feature_means(P, "test")
    fig, ax = plt.subplots(figsize=(6.3, 2.9))
    style(ax, grid="y")
    w = 0.16
    for j, (k, name, colour) in enumerate(FEATURES):
        x = np.arange(len(MODS)) + (j - 2) * w
        v = means[k][0]
        ax.bar(x, v, width=w * 0.9, color=colour, lw=0, label=name, zorder=2)
        if k in ("final", "statistics"):
            for xx, vv in zip(x, v):
                ax.text(xx, vv + 0.015, f"{vv:.2f}", ha="center", va="bottom", fontsize=6.5, color=INK2)
    ax.set_xticks(np.arange(len(MODS)), [f"{MOD_NAME[m]} embedding" for m in MODS])
    ax.tick_params(axis="x", length=0, labelsize=8)
    ax.set_ylim(0, 1.0)
    axis_labels(ax, None, "mean test R², 18 properties")
    legend(ax, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, handlelength=1.2, columnspacing=1.2)
    save(fig, out, "fig4_probe_means")


def fig_probe_heatmaps(out: Path, P: dict) -> None:
    final = np.array([[np.mean([P[n][m][p]["test"] for n in FINAL_NAMES]) for m in MODS] for p, _ in PROPS])
    stats = np.array([[P["statistics"][m][p]["test"] for m in MODS] for p, _ in PROPS])
    seq = LinearSegmentedColormap.from_list("blue", ["#f7fafe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
                                                     "#256abf", "#184f95", "#0d366b"])
    div = LinearSegmentedColormap.from_list("div", ["#a32d2c", RED, "#f2a8a2", BAND, "#9ec5f4", BLUE, "#184f95"])
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 5.6), gridspec_kw={"wspace": 0.12})
    for ax, data, cmap, lim, title, fmt in (
            (axes[0], final, seq, (0, 1), "(a) R² of the final setup", "{:.2f}"),
            (axes[1], final - stats, div, (-0.6, 0.6), "(b) gain over hand-made statistics", "{:+.2f}")):
        im = ax.imshow(data, cmap=cmap, vmin=lim[0], vmax=lim[1], aspect="auto")
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                c = im.cmap(im.norm(data[i, j]))
                v = data[i, j]
                ax.text(j, i, "0.00" if abs(v) < 0.005 else fmt.format(v).replace("-", "−"), ha="center",
                        va="center", fontsize=6.8, color="white" if dark(c) else INK)
        ax.set_xticks(range(len(MODS)), [MOD_NAME[m] for m in MODS], fontsize=7.5, color=INK2)
        ax.xaxis.tick_top()
        ax.set_yticks(range(len(PROPS)), [lab for _, lab in PROPS], fontsize=7.5, color=INK2)
        ax.tick_params(length=0, labelleft=ax is axes[0])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(title, fontsize=8.5, color=INK, loc="left", pad=18)
        cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.035, pad=0.02, aspect=30, shrink=0.8)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=7, colors=MUTED, labelcolor=INK2, length=2)
    save(fig, out, "fig5_probe_properties")


# ------------------------------------------------------------------------------------------------------------- main
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--clip", type=Path, default=Path("outputs/clip"), help="folder holding the runs and phase_d/")
    ap.add_argument("--baselines", type=Path, default=Path("outputs/clip_baselines/baselines.json"))
    ap.add_argument("--out", type=Path, default=Path("outputs/clip/report"))
    args = ap.parse_args(argv)
    plt.rcParams.update({"pdf.fonttype": 42, "font.family": "DejaVu Sans", "axes.unicode_minus": True})
    clip, d = args.clip, args.clip / "phase_d"
    figs, tabs = args.out / "figures", args.out / "tables"
    figs.mkdir(parents=True, exist_ok=True)
    tabs.mkdir(parents=True, exist_ok=True)

    val, test = json.loads((d / "val_final.json").read_text()), json.loads((d / "test_final.json").read_text())
    vb, tb = scored(d, "val_base.json")["pointnet_baseline"], scored(d, "test_base.json")["pointnet_baseline"]
    vs_s = scored(d, "val_shared_control_shared.json")["pointnet_shared_rotation"]
    vs_i = scored(d, "val_shared_control_independent.json")["pointnet_shared_rotation"]
    ts_s = scored(d, "test_shared_control_shared.json")["pointnet_shared_rotation"]
    ts_i = scored(d, "test_shared_control_independent.json")["pointnet_shared_rotation"]
    vb_s = scored(d, "val_base_shared.json")["pointnet_baseline"]
    bl_val = json.loads(args.baselines.read_text())["statistics + CCA"]
    bl_test = json.loads((d / "baselines_test.json").read_text())["statistics + CCA"]
    P = json.loads((d / "probes_test.json").read_text())["results"]
    boot = bootstrap(d, FINAL_NAMES + ("pointnet_baseline", "pointnet_shared_rotation"))
    base_hist = best_of(clip, "pointnet_baseline")["r1"]

    fig_curves(clip, figs, 100 * bl_val["r1/mean_no_gas_hi"])
    abl = fig_ablations(clip, figs, base_hist)
    stats_pairs = {p: pair(bl_test, p) for p in PAIRS}
    stats_pairs["mean"] = 100 * bl_test["r1/mean_no_gas_hi"]
    fig_pairs(figs, stats_pairs, boot)
    fig_probe_means(figs, P)
    fig_probe_heatmaps(figs, P)

    T = Tables(tabs)
    # -- main retrieval results
    fm_boot = np.mean([boot[n]["mean"][1] for n in FINAL_NAMES], axis=0)
    diff = fm_boot - boot["pointnet_baseline"]["mean"][1]
    m, s, vm, vsd = test["mean"], test["std"], val["mean"], val["std"]
    pc = lambda x: f"{100 * x:.1f}"                                                       # noqa: E731
    lo, hi = interval(fm_boot)
    blo, bhi = interval(boot["pointnet_baseline"]["mean"][1])
    T.add("retrieval_main", "Retrieval of a galaxy's other modalities (R@1, %).",
          ["model", "val, 10 dir.", "test, 10 dir.", "test 95% interval", "test, 12 dir.", "test R@10, 10 dir.",
           "test R@1 similar mass"],
          [["final setup (3 seeds, mean ± sd)", f"{pc(vm['r1/mean_no_gas_hi'])} ± {pc(vsd['r1/mean_no_gas_hi'])}",
            f"{pc(m['r1/mean_no_gas_hi'])} ± {pc(s['r1/mean_no_gas_hi'])}", f"{lo:.1f}–{hi:.1f}",
            f"{pc(m['r1/mean'])} ± {pc(s['r1/mean'])}", f"{pc(m['r10/mean_no_gas_hi'])} ± {pc(s['r10/mean_no_gas_hi'])}",
            f"{pc(m['massbin_r1/mean_no_gas_hi'])} ± {pc(s['massbin_r1/mean_no_gas_hi'])}"],
           ["Phase B PointNet", pc(vb["r1/mean_no_gas_hi"]), pc(tb["r1/mean_no_gas_hi"]), f"{blo:.1f}–{bhi:.1f}",
            pc(tb["r1/mean"]), pc(tb["r10/mean_no_gas_hi"]), pc(tb["massbin_r1/mean_no_gas_hi"])],
           ["statistics + CCA (no network)", pc(bl_val["r1/mean_no_gas_hi"]), pc(bl_test["r1/mean_no_gas_hi"]), "",
            pc(bl_test["r1/mean"]), pc(bl_test["r10/mean_no_gas_hi"]), "–"],
           ["chance", f"{plain_chance(vb):.2f}", f"{plain_chance(tb):.2f}", "", f"{100 * tb['chance/mean']:.2f}",
            f"{10 * plain_chance(tb):.1f}", f"{100 * tb['massbin_chance/mean']:.1f}"],
           "---",
           ["shared-rotation control, aligned clouds", pc(vs_s["r1/mean_no_gas_hi"]), pc(ts_s["r1/mean_no_gas_hi"]),
            "", pc(ts_s["r1/mean"]), pc(ts_s["r10/mean_no_gas_hi"]), pc(ts_s["massbin_r1/mean_no_gas_hi"])],
           ["shared-rotation control, independently rotated", pc(vs_i["r1/mean_no_gas_hi"]),
            pc(ts_i["r1/mean_no_gas_hi"]), "", pc(ts_i["r1/mean"]), pc(ts_i["r10/mean_no_gas_hi"]),
            pc(ts_i["massbin_r1/mean_no_gas_hi"])]],
          note=(f"Validation: 520 galaxies, test: 499. 10 dir. = mean over the 10 search directions without gas–HI; "
                f"12 dir. includes it. Similar mass: each galaxy is searched for only among galaxies within the same "
                f"0.5 dex stellar-mass bin. Interval: 95% bootstrap over test galaxies. Final minus Phase B on test: "
                f"{float(np.mean([boot[n]['mean'][0] for n in FINAL_NAMES])) - boot['pointnet_baseline']['mean'][0]:+.1f} "
                f"points (95% interval {interval(diff)[0]:+.1f} to {interval(diff)[1]:+.1f}). The shared-rotation "
                f"control was trained and is scored with one rotation shared by a galaxy's four clouds."))
    pr = [["final setup (3 seeds)"] + [f"{np.mean([pair(test['runs'][n]['results'], p) for n in FINAL_NAMES]):.1f}"
                                       for p in PAIRS],
          ["Phase B PointNet"] + [f"{pair(tb, p):.1f}" for p in PAIRS],
          ["statistics + CCA"] + [f"{pair(bl_test, p):.1f}" for p in PAIRS],
          ["chance"] + [f"{100 / tb[f'gallery/{p.replace(chr(45), chr(45) + chr(62))}']:.2f}" for p in PAIRS],
          ["galleries (galaxies)"] + [f"{tb[f'gallery/{p.replace(chr(45), chr(45) + chr(62))}']:.0f}" for p in PAIRS]]
    T.add("retrieval_pairs", "Test R@1 (%) per modality pair, mean of both search directions.",
          ["model"] + [PAIR_NAME[p] for p in PAIRS], pr,
          note="Gas and HI clouds are drawn from the same gas cells, so gas–HI is close to a duplicate match.")
    # -- orientation shortcut
    T.add("orientation_shortcut", "The orientation shortcut: R@1 (%, 10 directions) by how the clouds are rotated.",
          ["trained on", "val, aligned", "val, independent", "test, aligned", "test, independent"],
          [["independently rotated clouds (Phase B PointNet)", pc(vb_s["r1/mean_no_gas_hi"]),
            pc(vb["r1/mean_no_gas_hi"]), "–", pc(tb["r1/mean_no_gas_hi"])],
           ["one rotation shared by a galaxy's clouds (control)", pc(vs_s["r1/mean_no_gas_hi"]),
            pc(vs_i["r1/mean_no_gas_hi"]), pc(ts_s["r1/mean_no_gas_hi"]), pc(ts_i["r1/mean_no_gas_hi"])]],
          note="Aligned: a galaxy's four clouds share one random rotation, so their orientations match. "
               "The Phase B model was not scored on aligned test clouds (the test set was scored once, as planned).")
    # -- Phase C
    rows = []
    for g in dict.fromkeys(r[0] for r in abl):
        rows.append(("#group", g))
        for gg, lab, r in abl:
            if gg == g:
                rows.append([f"   {lab}", f"{r['r1']:.1f}" + (f" ({r['lo']:.1f}–{r['hi']:.1f})" if r["n"] > 1 else ""),
                             signed(r["r1"] - base_hist, lab == ABLATIONS[0][1]), f"{r['stars_dm']:.1f}",
                             f"{r['epoch']:.0f}", f"{r['final']:.1f}"])
    for g in dict.fromkeys(e[0] for e in EXTRA):
        rows.append(("#group", g))
        ref = None
        for gg, lab, runs in EXTRA:
            if gg == g:
                r = row_of(clip, runs)
                rows.append([f"   {lab}", f"{r['r1']:.1f}", signed(r["r1"] - (ref or r["r1"]), ref is None),
                             f"{r['stars_dm']:.1f}", f"{r['epoch']:.0f}", f"{r['final']:.1f}"])
                ref = r["r1"] if ref is None else ref
    T.add("phase_c_ablations", "Phase C: one change at a time against the Phase B PointNet (validation R@1, %).",
          ["change", "best, 10 dir.", "vs reference", "stars–DM", "best epoch", "last epoch"], rows,
          note=f"Best validation score during training (520 galaxies), several seeds as mean (range). A change counts "
               f"if it moves the score by more than {NOISE:.0f} points (two seeds of the base differ by 0.3). Sphere "
               f"sizes use the 474 galaxies usable at every size and are compared with their own 5-radii run; the "
               f"patch transformer is compared with its Phase B run. Last epoch: score at the end of training. EMA 0.999 "
               f"was not a fair test: with 15 updates per epoch it averages over about 67 of the 300 epochs and lags "
               f"behind (0.99 is the fair setting). Without per-point physical values training never started (loss "
               f"flat at its initial value); adding rotation-free features made that input set learnable.")
    # -- probes
    tm, vmn = feature_means(P, "test"), feature_means(P, "val")
    rows = []
    for k, name, _ in FEATURES:
        cells = []
        for i in range(len(MODS)):
            sd = f" ± {tm[k][1][i]:.3f}" if tm[k][1] is not None else ""
            cells.append(f"{tm[k][0][i]:.3f}{sd} ({vmn[k][0][i]:.3f})")
        rows.append([name] + cells)
    T.add("probes_summary", "Linear probes: mean R² over 18 galaxy properties (test; validation in brackets).",
          ["features"] + [f"{MOD_NAME[m]} embedding" for m in MODS], rows,
          note="Ridge regression fitted on the 4,052 training galaxies, its strength chosen on validation; negative "
               "R² counted as 0. Statistics: about 20 orientation-free numbers per cloud. Untrained: the final "
               "architecture with its initial random weights.")
    rows = []
    for k, name, _ in FEATURES:
        src = FINAL_NAMES if k == "final" else (k,)
        rows.append([name] + [f"{np.mean([P[n][m]['central']['test'] for n in src]):.2f}" for m in MODS]
                    + [f"{np.mean([P[n][m]['quenched']['test'] for n in src]):.2f}" for m in ("stars", "dm")])
    T.add("probes_auc", "Yes/no properties on test: area under the ROC curve.",
          ["features"] + [f"central, {MOD_NAME[m]}" for m in MODS] + ["quenched, stars", "quenched, dark matter"],
          rows, note="0.5 = guessing, 1 = perfect. Quenched (no star formation): 61 of 499 test galaxies; it is not "
                     "scored for gas and HI clouds, which belong to only 0 and 1 quenched test galaxies.")
    rows = []
    for p, lab in PROPS:
        r = [lab]
        for mm in MODS:
            r += [f"{np.mean([P[n][mm][p]['test'] for n in FINAL_NAMES]):.2f}", f"{P['statistics'][mm][p]['test']:.2f}"]
        rows.append(r)
    T.add("probes_properties", "Test R² per property: final setup vs hand-made statistics, for each modality's "
          "embedding.", ["property"] + [f"{MOD_NAME[m]}: {w}" for m in MODS for w in ("final", "stat.")], rows,
          note="Masses, rates and metallicities as log10; for gas mass, black-hole mass, gas and HI mass in the sphere "
               "and the star-formation rates only galaxies that have some (above the floor of the property table).")

    # -- report.md
    tb_final = float(np.mean([boot[n]["mean"][0] for n in FINAL_NAMES]))
    md = ["# GalaxyCLIP-3D alignment: thesis figures and tables", "",
          f"Built by `tools/clip_report.py` from saved files. Final setup = PointNet per modality (widths "
          f"128-256-512-1024, 9.1 M parameters), learned temperature, 600 epochs, seeds 4869, 2 and 3. Test R@1 "
          f"(10 directions) {tb_final:.1f}% (95% interval {lo:.1f}–{hi:.1f}), Phase B {boot['pointnet_baseline']['mean'][0]:.1f}%.",
          "", "## Figures", ""]
    captions = {
        "fig1_training_curves": "Validation R@1 (10 directions, 520 galaxies) during training, every 5 epochs. Blue: "
                                "the final setup, three seeds. Orange: the Phase B PointNet (300 epochs). Green: the "
                                "control trained on clouds that share one rotation per galaxy; it matches clouds by "
                                "their orientation and falls to about 5% on independently rotated clouds. Dashed: "
                                "statistics + CCA, no network.",
        "fig2_phase_c_ablations": "Phase C ablations: best validation R@1 for one change at a time. Shaded band: the "
                                  "Phase B PointNet ± 2 points, the threshold set before the runs (two seeds of the "
                                  "base differ by 0.3). The three gains under the threshold (learned temperature, "
                                  "wide encoder, 600 epochs) add up when combined. EMA 0.999 averaged over about 67 "
                                  "epochs, too long for a 300-epoch run, so it is not evidence against weight "
                                  "averaging; without per-point physical values training never started.",
        "fig3_test_retrieval_pairs": "Test R@1 per modality pair, mean of both search directions. Whiskers: 95% "
                                     "bootstrap intervals over the 499 test galaxies. Black tick: statistics + CCA "
                                     "without a network. Gas and HI come from the same gas cells.",
        "fig4_probe_means": "Linear probes on frozen embeddings: mean test R² over 18 galaxy properties for each "
                            "modality's embedding (negative R² counted as 0).",
        "fig5_probe_properties": "Test R² per property for the final setup (a) and its gain over hand-made statistics "
                                 "of the same clouds (b). Each column uses only that modality's cloud: e.g. the dark-"
                                 "matter embedding predicts stellar mass with R² 0.80, statistics of the same "
                                 "dark-matter cloud with 0.29.",
    }
    for name, cap in captions.items():
        md += [f"### {name}", "", f"![{name}](figures/{name}.png)", "", cap, "",
               f"(`figures/{name}.pdf`)", ""]
    md += ["## Tables", ""] + T.md
    (args.out / "report.md").write_text("\n".join(md) + "\n")
    summary = {"test_r1_final": tb_final, "test_interval": [lo, hi], "test_r1_base": boot["pointnet_baseline"]["mean"][0],
               "final_minus_base": tb_final - boot["pointnet_baseline"]["mean"][0], "diff_interval": list(interval(diff))}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    print(f"wrote {args.out}/report.md, {len(list(figs.glob('*.pdf')))} figures, {len(list(tabs.glob('*.tex')))} tables")


if __name__ == "__main__":
    main()
