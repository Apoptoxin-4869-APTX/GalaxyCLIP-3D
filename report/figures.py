"""Figures for the project report and the supervisor deck: the thesis figures as 300-dpi PNGs, plus new ones (data
pipeline, model, dataset overview, dark-matter controls). Numbers come from outputs/project_report/numbers.json.

    cd GalaxyCLIP_3D && PYTHONPATH=~/tools/report_pylibs:pylibs:. python -s report/figures.py
    # -> outputs/project_report/figures/*.png (+ .pdf for the new ones)
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402
import numpy as np                                                       # noqa: E402
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch   # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from galaxyclip3d.dataset import galaxy_masks                           # noqa: E402
from galaxyclip3d.pointclouds.config import Config                      # noqa: E402
from galaxyclip3d.pointclouds.layout import read_table                  # noqa: E402

OUT = REPO / "outputs/project_report"
FIG = OUT / "figures"
V3 = Path("/cluster/projects/ska/illustris/tng50-1.pointclouds/v3")

# dataviz reference palette (as tools/clip_report.py); sets below were run through validate_palette.js
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
                                                           "#008300", "#6250d6", "#e34948")
INK, INK2, MUTED, GRID, AXIS, BAND = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#f0efec"
MOD_COL = {"stars": ORANGE, "dm": VIOLET, "gas": AQUA, "hi": BLUE}      # passes as an adjacent set (labels carry it)
MOD_NAME = {"stars": "stars", "dm": "dark matter", "gas": "gas", "hi": "HI"}
# dark-matter control figure, bar order c29 | C21 | c30 (red, blue, green passes as an adjacent set)
RUN_COL = {"c29-Rvmax": RED, "C21-stellar": BLUE, "c30-subset": GREEN}
RUN_NAME = {"c29-Rvmax": "halo frame (c29)", "C21-stellar": "stellar frame, final setup (C21)",
            "c30-subset": "stellar frame, same galaxies (c30)"}

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "legend.fontsize": 7, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.titlecolor": INK,
                     "font.family": "DejaVu Sans", "pdf.fonttype": 42, "svg.fonttype": "none"})


def style(ax, grid="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    ax.tick_params(length=2.5, width=0.6)


def save(fig, name: str) -> None:
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)


def thesis_figures() -> None:
    """The five thesis figures (tools/clip_report.py) at 300 dpi, rendered from their PDFs."""
    for pdf in sorted((REPO / "outputs/clip/report/figures").glob("fig*.pdf")):
        subprocess.run(["pdftoppm", "-png", "-r", "300", "-singlefile", str(pdf), str(FIG / f"thesis_{pdf.stem}")],
                       check=True)
    shutil.copy(REPO / "outputs/pointcloud_examples/galaxy_402557.png", FIG / "example_galaxy.png")
    print("wrote thesis figures and the example galaxy")


def box(ax, x, y, w, h, title, lines=(), upi=1.0, fc="white", ec=AXIS, tc=INK, size=6.6, title_size=7.4, lw=0.8):
    """Rounded box with a bold title and lines below it, the block centred vertically; upi = y data units per inch."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.012", fc=fc, ec=ec, lw=lw))
    step = [title_size * 1.35 / 72 * upi] + [size * 1.3 / 72 * upi] * len(lines)
    top = y + h / 2 + sum(step) / 2
    ax.text(x + w / 2, top, title, ha="center", va="top", fontsize=title_size, weight="bold", color=tc)
    for i, s in enumerate(lines):
        ax.text(x + w / 2, top - sum(step[:i + 1]), s, ha="center", va="top", fontsize=size, color=INK2)


def arrow(ax, x0, y0, x1, y1, color=INK2):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8, lw=0.9, color=color))


def fig_pipeline(n: dict) -> None:
    d, f = n["dataset"], n["facts"]
    sp = d["splits"]
    steps = [
        ("TNG50-1", ["snapshot 99 (z = 0)", f"{f['raw_snapshot']['size_tb']} TB raw data", "complete fresh copy"]),
        ("Selection", [f"{d['selected']:,} galaxies", f"in {d['halos']:,} halos", "≥ 4,096 particles"]),
        ("Extraction", ["particles ≤ 10 R½", f"{f['storage']['particles_billion']} billion stored",
                        f"{f['storage']['shards_gb']} GB on disk"]),
        ("Splits by halo", ["80 / 10 / 10", f"{sp['train']['galaxies']:,} / {sp['val']['galaxies']} / "
                            f"{sp['test']['galaxies']}", "usable at 5 R½"]),
        ("Loader", ["4 clouds per galaxy", "2,048 points each", "fresh draw per epoch"]),
    ]
    fig, ax = plt.subplots(figsize=(6.6, 1.2))
    ax.set_xlim(-0.005, 1.005)
    ax.set_ylim(0, 1.2)
    ax.axis("off")
    w, gap = 0.18, 0.025
    for i, (t, ls) in enumerate(steps):
        x = i * (w + gap)
        box(ax, x, 0.03, w, 1.14, t, ls, upi=1.0, fc="#f7f7f5", size=6.0, title_size=7.0)
        if i:
            arrow(ax, x - gap + 0.003, 0.6, x - 0.003, 0.6)
    save(fig, "pipeline")


def fig_model() -> None:
    fig_h, y_top = 2.9, 0.5
    fig, ax = plt.subplots(figsize=(6.6, fig_h))
    ax.set_xlim(-0.005, 1.005)
    ax.set_ylim(0, y_top)
    ax.axis("off")
    upi = y_top / fig_h
    feats = {"stars": "x, v, age, Z", "dm": "x, v", "gas": "x, v, nH, T, Z, B, xHI, SF", "hi": "x, v, xHI"}
    h, ys = 0.082, [0.33, 0.225, 0.12, 0.015]
    for (m, ft), y in zip(feats.items(), ys):
        c = MOD_COL[m]
        box(ax, 0.0, y, 0.215, h, f"{MOD_NAME[m]} cloud", ["2,048 points with", ft], upi=upi, ec=c, lw=1.4,
            size=6.0, title_size=7.2)
        arrow(ax, 0.219, y + h / 2, 0.291, y + h / 2)
        box(ax, 0.295, y, 0.235, h, "PointNet encoder", ["per-point MLP,", "max-pool, projection"],
            upi=upi, fc="#f7f7f5", ec=c, lw=1.4, size=5.8, title_size=7.2)
        arrow(ax, 0.534, y + h / 2, 0.632, 0.225)
    ax.text(0.255, 0.452, "own random\nrotation + mirror", ha="center", va="center", fontsize=6.2, color=MUTED)
    ax.text(0.58, 0.452, "512-d vector,\nunit length", ha="center", va="center", fontsize=6.2, color=MUTED)
    cx, cy, r = 0.79, 0.225, 0.13
    sq = (y_top / fig_h) / (1.01 / 6.6)                   # y units per x unit on paper, so circles stay round

    def Circle(xy, rad, **kw):
        return Ellipse(xy, 2 * rad, 2 * rad * sq, **kw)

    ax.add_patch(Circle((cx, cy), r, fc="#f7f7f5", ec=AXIS, lw=0.9))
    rng = np.random.default_rng(4)
    for _ in range(9):
        a, rr = rng.uniform(0, 2 * np.pi), rng.uniform(0.4, 0.8) * r
        ox, oy = cx + rr * np.cos(a), cy + rr * np.sin(a) * sq
        for j, m in enumerate(feats):
            ax.add_patch(Circle((ox + 0.011 * np.cos(j * 1.6 + a), oy + 0.011 * sq * np.sin(j * 1.6 + a)), 0.005,
                                fc=MOD_COL[m], ec="none", alpha=0.3))
    for j, m in enumerate(feats):
        ax.add_patch(Circle((cx + 0.015 * np.cos(j * 1.57 + 0.4), cy + 0.015 * sq * np.sin(j * 1.57 + 0.4)), 0.008,
                            fc=MOD_COL[m], ec="white", lw=0.6))
    ax.text(cx, cy + r * sq + 0.012, "shared embedding space", ha="center", va="bottom", fontsize=7.4, weight="bold")
    ax.text(cx, cy - r * sq - 0.012, "contrastive loss over all 6 modality pairs:\none galaxy's 4 clouds pulled "
            "together,\nthe batch's other galaxies pushed apart", ha="center", va="top", fontsize=6.2, color=INK2)
    save(fig, "model")


def fig_dataset(n: dict) -> None:
    cfg = Config.load(V3 / "config.json")
    idx = read_table(V3 / "snap_099/index.h5")
    usable, mask = galaxy_masks(idx, cfg, 5.0)
    logm = np.log10(idx["mstar"])
    fig, axs = plt.subplots(1, 3, figsize=(6.6, 2.15), gridspec_kw={"wspace": 0.42})
    bins = np.arange(7.75, 12.01, 0.25)
    for i, (sp, c) in enumerate(zip(("train", "val", "test"), (BLUE, ORANGE, AQUA))):
        k = usable & (idx["split"] == i)
        h, e = np.histogram(logm[k], bins=bins)
        axs[0].stairs(h / h.sum(), e, color=c, lw=1.4, label=f"{sp} ({k.sum():,})")
    axs[0].set_xlabel("log10 stellar mass [M☉]")
    axs[0].set_ylabel("fraction of galaxies")
    axs[0].set_title("(a) stellar mass by split", loc="left")
    axs[0].legend(frameon=False, loc="upper right", handlelength=1.4)
    centres = 0.5 * (bins[1:] + bins[:-1])
    for j, (m, lab) in enumerate((("gas", "gas cloud on"), ("hi", "HI cloud on")), start=2):
        frac = [mask[usable & (logm >= a) & (logm < b), j].mean() if (usable & (logm >= a) & (logm < b)).sum() > 9
                else np.nan for a, b in zip(bins[:-1], bins[1:])]
        axs[1].plot(centres, 100 * np.array(frac), color=MOD_COL[m], lw=1.6, label=lab)
    axs[1].set_ylim(0, 100)
    axs[1].set_xlabel("log10 stellar mass [M☉]")
    axs[1].set_ylabel("% of usable galaxies")
    axs[1].set_title("(b) galaxies with enough gas", loc="left")
    axs[1].legend(frameon=False, loc="lower right", handlelength=1.4)
    for m in ("stars", "dm", "gas"):
        x = np.sort(idx[f"n_{m}_r5"][usable])
        x = x[x > 0]
        axs[2].plot(x, 100 * (1 - np.arange(len(x)) / len(x)), color=MOD_COL[m], lw=1.6, label=MOD_NAME[m])
    axs[2].set_xscale("log")
    axs[2].set_xlim(3e2, 3e7)
    axs[2].set_ylim(0, 118)
    axs[2].set_yticks([0, 20, 40, 60, 80, 100])
    for v, lab, ha in ((2048, "2,048 drawn", "right"), (4096, "4,096 needed", "left")):
        axs[2].axvline(v, color=MUTED, lw=0.8, ls="--" if v == 2048 else ":")
        axs[2].text(v * (0.85 if ha == "right" else 1.18), 110, lab, fontsize=5.8, color=MUTED, ha=ha, va="center")
    axs[2].set_xlabel("particles inside 5 R½")
    axs[2].set_ylabel("% of galaxies with more")
    axs[2].set_title("(c) particles per galaxy", loc="left")
    axs[2].legend(frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.86), handlelength=1.4)
    for a in axs:
        style(a, grid="y")
    save(fig, "dataset")


def fig_dm_control(n: dict) -> None:
    dm, c30 = n["dm_control"], n["c30"]
    src = c30 or dm
    runs = [r for r in ("c29-Rvmax", "C21-stellar", "c30-subset") if r in src["retrieval"]]
    pairs = [("stars-dm", "stars – DM"), ("dm-gas", "DM – gas"), ("dm-hi", "DM – HI"), ("stars-gas", "stars – gas"),
             ("stars-hi", "stars – HI")]
    props = [("log_mstar_total", "stellar\nmass"), ("rhalf_star_kpc", "stellar half-\nmass radius"),
             ("gas_fraction", "gas\nfraction"), ("log_mdm_total", "DM\nmass"), ("log_mbh", "black-hole\nmass"),
             ("log_z_stars", "stellar\nmetallicity"), ("vmax", "max. circular\nvelocity")]
    fig, axs = plt.subplots(2, 1, figsize=(6.6, 4.6), gridspec_kw={"hspace": 0.62})
    w = 0.8 / len(runs)
    for k, r in enumerate(runs):
        vals = [src["retrieval"][r]["pairs"][p] for p, _ in pairs] + [src["retrieval"][r]["r1_no_gas_hi"]]
        x = np.arange(len(vals)) + (k - (len(runs) - 1) / 2) * w
        axs[0].bar(x, vals, w * 0.92, color=RUN_COL[r], label=RUN_NAME[r])
        for xi, v in zip(x, vals):
            axs[0].text(xi, v + 1.2, f"{v:.1f}", ha="center", va="bottom", fontsize=5.6, color=INK2)
    axs[0].set_xticks(range(len(pairs) + 1), [p for _, p in pairs] + ["mean, 10\ndirections"])
    axs[0].set_ylabel("R@1 (%)")
    axs[0].set_ylim(0, 100)
    axs[0].set_title(f"(a) retrieval on the same {src['shared']['val']} validation galaxies", loc="left")
    axs[0].legend(frameon=False, ncol=1, loc="upper left", bbox_to_anchor=(0, 1.0), handlelength=1.2)
    stats = {"stats-stellar": ("statistics, stellar frame", "-"), "stats-Rvmax": ("statistics, halo frame", "--")}
    for k, r in enumerate(runs):
        vals = [src["probes"][r]["dm"]["props"][p] for p, _ in props] + [src["probes"][r]["dm"]["mean_r2"]]
        x = np.arange(len(vals)) + (k - (len(runs) - 1) / 2) * w
        axs[1].bar(x, vals, w * 0.92, color=RUN_COL[r])  # colours named by the legend of panel (a)
    for s, (lab, ls) in stats.items():
        vals = [src["probes"][s]["dm"]["props"][p] for p, _ in props] + [src["probes"][s]["dm"]["mean_r2"]]
        for i, v in enumerate(vals):
            axs[1].plot([i - 0.43, i + 0.43], [v, v], color=INK, lw=1.1, ls=ls, label=lab if i == 0 else None)
    axs[1].set_xticks(range(len(props) + 1), [p for _, p in props] + ["mean of\n18"])
    axs[1].set_ylabel("validation R²")
    axs[1].set_ylim(0, 1)
    axs[1].set_title("(b) what the dark-matter embedding predicts (linear probes)", loc="left")
    axs[1].legend(frameon=False, ncol=2, loc="upper left", bbox_to_anchor=(0, 1.02), handlelength=1.6,
                  columnspacing=1.4)
    axs[1].set_ylim(0, 1.1)
    axs[1].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    for a in axs:
        style(a, grid="y")
    save(fig, "dm_control")


def fig_control_curves(n: dict) -> None:
    """Validation R@1 during training: the final setup (C21), the halo-frame control (c29) and, once it exists, the
    matched control (c30). Line charts without markers."""
    runs = [("c29-Rvmax", REPO / "outputs/clip/phase_d/c29_dm_rvmax/history.json"),
            ("C21-stellar", REPO / "outputs/clip/phase_c/c21_combined/history.json"),
            ("c30-subset", REPO / "outputs/clip/phase_d/c30_subset_stellar/history.json")]
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.3), gridspec_kw={"wspace": 0.3})
    for key, path in runs:
        if not path.exists():
            continue
        h = json.loads(path.read_text())
        vals = [v for v in h["val"] if v]
        e = [v["epoch"] for v in vals]
        axs[0].plot(e, [100 * v["r1/mean_no_gas_hi"] for v in vals], color=RUN_COL[key], lw=1.4, label=RUN_NAME[key])
        axs[1].plot(e, [50 * (v["r1/stars->dm"] + v["r1/dm->stars"]) for v in vals], color=RUN_COL[key], lw=1.4)
    axs[0].set_title("(a) mean R@1, 10 directions", loc="left")
    axs[1].set_title("(b) stars – dark matter R@1", loc="left")
    for a in axs:
        a.set_xlabel("epoch")
        a.set_ylabel("validation R@1 (%)")
        a.set_xlim(0, 600)
        a.set_ylim(0, None)
        style(a, grid="y")
    axs[0].legend(frameon=False, loc="lower right", handlelength=1.4)
    save(fig, "control_curves")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    n = json.loads((OUT / "numbers.json").read_text())
    thesis_figures()
    fig_pipeline(n)
    fig_model()
    fig_dataset(n)
    fig_dm_control(n)
    fig_control_curves(n)


if __name__ == "__main__":
    main()
