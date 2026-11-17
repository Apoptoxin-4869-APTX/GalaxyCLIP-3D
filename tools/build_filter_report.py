"""Build a PDF of suggestions for filtering a 3D-only point-cloud dataset (no 2D maps), with charts.

Galaxy counts come from the group catalog of the snapshot copy at --root and use every particle bound to
a galaxy; particle-level charts use the two samples written by tools/collect_histogram_data.py.

    python tools/build_filter_report.py --root /cluster/projects/ska/illustris/tng50-1.new \
        --hist2d outputs/data_audit_complete/histogram_data.npz \
        --histall outputs/data_audit_complete/histogram_data_all.npz \
        --out outputs/filter_suggestions_3d/filter_suggestions_3d.pdf

Needs numpy, h5py, matplotlib and reportlab (see tools/build_report.py).
"""
from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg                                         # noqa: E402
import matplotlib.pyplot as plt                                          # noqa: E402
import numpy as np                                                        # noqa: E402
from reportlab.lib import colors                                          # noqa: E402
from reportlab.lib.pagesizes import A4                                    # noqa: E402
from reportlab.lib.styles import ParagraphStyle                           # noqa: E402
from reportlab.lib.units import mm                                        # noqa: E402
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer,  # noqa: E402
                                Table, TableStyle)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_snapshot import load_groupcat                                  # noqa: E402
from build_report import IMG_W, INK, INK2, STYLES                         # noqa: E402  (registers the fonts)
from plot_histograms import (BLUE, GRID, INK2 as INK2_HEX, ORANGE, SURFACE, legend, load, steps,  # noqa: E402
                             style, threshold)

SOFTENING = 0.288       # TNG50-1 stellar and dark-matter softening length at z = 0, physical kpc
BUDGET = 2048           # points per modality in a cloud
THRESH = 2 * BUDGET     # particles a galaxy needs, so each epoch can draw a different set of points
OLD_SET = 2924          # 2D halos kept by the recommended filters of the data-distribution report


# ---------------------------------------------------------------- data
def galaxies(root: str) -> dict[str, np.ndarray]:
    """One entry per Subfind subhalo: bound particle counts, flags and sizes (physical units)."""
    with h5py.File(Path(root) / "snapshots" / "099" / "snap_099.0.hdf5", "r") as f:
        hd = dict(f["Header"].attrs)
    h, a = float(hd["HubbleParam"]), float(hd["Time"])
    cat = load_groupcat(root, 99, {"Group": ["GroupFirstSub"],
                                   "Subhalo": ["SubhaloLenType", "SubhaloFlag", "SubhaloHalfmassRadType",
                                               "SubhaloMassInRadType", "SubhaloSFRinRad", "SubhaloGrNr"]})
    n = cat["SubhaloLenType"]
    cen = np.zeros(len(n), bool)
    fs = cat["GroupFirstSub"]
    cen[fs[fs >= 0]] = True
    return dict(n_gas=n[:, 0], n_dm=n[:, 1], n_star=n[:, 4], flag=cat["SubhaloFlag"] == 1, cen=cen,
                rstar=cat["SubhaloHalfmassRadType"][:, 4] * a / h,
                mstar=cat["SubhaloMassInRadType"][:, 4] * 1e10 / h, sfr=cat["SubhaloSFRinRad"],
                group=cat["SubhaloGrNr"])


def funnel(g, t=THRESH):
    """[(label, mask)], each mask applying all filters up to and including that step."""
    out = [("≥ 2,048 star particles", g["n_star"] >= 2048)]
    for label, cond in (("SubhaloFlag = 1 (real galaxy)", g["flag"]),
                        (f"≥ {t:,} star particles", g["n_star"] >= t),
                        (f"≥ {t:,} dark-matter particles", g["n_dm"] >= t),
                        (f"stellar half-mass radius ≥ {2 * SOFTENING:.2f} kpc", g["rstar"] >= 2 * SOFTENING),
                        (f"≥ {t:,} gas cells", g["n_gas"] >= t)):
        out.append((label, out[-1][1] & cond))
    return out


def select(g, t, gas):
    m = g["flag"] & (g["n_star"] >= t) & (g["n_dm"] >= t) & (g["rstar"] >= 2 * SOFTENING)
    return m & (g["n_gas"] >= t) if gas else m


# ---------------------------------------------------------------- charts
def hgrid(ax):
    ax.grid(False)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def chart_cube(da):
    hw, r200, rs = da["avail_half_width_kpc"], da["halo_r200c_kpc"], da["halo_rhalf_star_kpc"]
    rows = sorted([("cube wider than the halo (R200c)", int((hw / r200 > 1).sum())),
                   ("stellar body larger than the cube", int((rs / hw > 1).sum())),
                   ("cube crosses the box edge", int((da["avail_crosses_box_edge"] == 1).sum())),
                   ("cube smaller than 1 kpc", int((hw < 1).sum()))], key=lambda r: -r[1])

    def draw(ax):
        y = np.arange(len(rows))[::-1]
        ax.barh(y, [v for _, v in rows], height=0.6, color=BLUE, linewidth=0)
        for yi, (_, v) in zip(y, rows):
            ax.text(v, yi, f"  {v:,}", va="center", ha="left", fontsize=8, color=INK2_HEX)
        ax.set_yticks(y, [lab for lab, _ in rows])
        ax.set_xlim(0, max(v for _, v in rows) * 1.15)
        style(ax, f"Halos where the 2D cube rule breaks down (of {len(hw):,})", "halos")
        hgrid(ax)
    return draw, rows


def chart_funnel(g, steps_):
    def draw(ax):
        y = np.arange(len(steps_))[::-1]
        c = np.array([(m & g["cen"]).sum() for _, m in steps_])
        s = np.array([(m & ~g["cen"]).sum() for _, m in steps_])
        ax.barh(y, c, height=0.62, color=BLUE, edgecolor=SURFACE, linewidth=1.5, label="centrals")
        ax.barh(y, s, left=c, height=0.62, color=ORANGE, edgecolor=SURFACE, linewidth=1.5, label="satellites")
        for yi, tot in zip(y, c + s):
            ax.text(tot, yi, f"  {tot:,}", va="center", ha="left", fontsize=8, color=INK2_HEX)
        ax.set_yticks(y, [lab for lab, _ in steps_])
        ax.set_xlim(0, (c + s).max() * 1.15)
        legend(ax, "lower right")
        style(ax, "Galaxies left after each filter (cumulative)", "galaxies")
        hgrid(ax)
    return draw


def chart_hist_log(values_cen, values_sat, edges, title, xlabel, lines=(), note=None, vertical=True, headroom=1.0):
    def draw(ax):
        for v, color, label in ((values_cen, BLUE, f"centrals ({len(values_cen):,})"),
                                (values_sat, ORANGE, f"satellites ({len(values_sat):,})")):
            steps(ax, np.histogram(v, edges)[0], edges, color, label, baseline=0.5)
        ax.set_yscale("log")
        ax.set_ylim(0.5, ax.get_ylim()[1] * headroom)       # headroom keeps line labels clear of the curves
        for x, lab in lines:
            threshold(ax, x, lab, vertical=vertical)
        legend(ax, "upper right")
        style(ax, title, xlabel, "galaxies (log scale)", note=note, note_y=0.80)
    return draw


def chart_wind(d2, da):
    def draw(ax):
        e = np.arange(0, 6.51, 0.25)
        for d, color, label in ((d2, BLUE, f"halos with 2D maps ({len(d2['sample_gid'])})"),
                                (da, ORANGE, f"halos of all masses ({len(da['sample_gid'])})")):
            steps(ax, np.histogram(100 * d["sample_wind_mass_frac"], e)[0], e, color, label)
        legend(ax, "upper right")
        style(ax, "Wind cells among the star particles, per halo", "wind mass / PartType4 mass in the cube [%]",
              "halos")
    return draw


def chart_sfr(g, m):
    rows = [("centrals", m & g["cen"]), ("satellites", m & ~g["cen"])]

    def draw(ax):
        y = np.arange(len(rows))[::-1]
        on = np.array([(r & (g["sfr"] > 0)).sum() for _, r in rows])
        off = np.array([(r & (g["sfr"] <= 0)).sum() for _, r in rows])
        ax.barh(y, on, height=0.55, color=BLUE, edgecolor=SURFACE, linewidth=1.5, label="forming stars")
        ax.barh(y, off, left=on, height=0.55, color=ORANGE, edgecolor=SURFACE, linewidth=1.5,
                label="no star formation (SFR = 0)")
        for yi, a, b in zip(y, on, off):
            ax.text(a + b, yi, f"  {b:,} of {a + b:,} without star formation", va="center", ha="left",
                    fontsize=8, color=INK2_HEX)
        ax.set_yticks(y, [lab for lab, _ in rows])
        ax.set_xlim(0, (on + off).max() * 2.3)
        legend(ax, "center right")
        style(ax, f"Star formation in the {m.sum():,} galaxies before the gas filter", "galaxies")
        hgrid(ax)
    return draw


def chart_temperature(d):
    total = d["hist_logT_sf"].sum() + d["hist_logT_nonsf"].sum()
    e = d["bins_logT"]
    c = 0.5 * (e[:-1] + e[1:])
    mode = float(c[np.argmax(d["hist_logT_sf"])])

    def draw(ax):
        steps(ax, 100 * d["hist_logT_nonsf"] / total, e, BLUE, "not star-forming")
        steps(ax, 100 * d["hist_logT_sf"] / total, e, ORANGE, "star-forming (SFR > 0)")
        threshold(ax, mode + 0.08, f"sub-grid ISM temperature, 10^{mode:.1f} K")
        ax.set_xlim(2.5, 8)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
        legend(ax, "upper right")
        style(ax, f"Gas temperature in the cubes of {len(d['sample_gid'])} halos of all masses (mass-weighted)",
              "log10 T [K]", "% of gas mass per bin")
    return draw, mode


def chart_mstar(g, m):
    lm = np.log10(g["mstar"][m])
    e = np.arange(np.floor(lm.min() * 2) / 2, np.ceil(lm.max() * 2) / 2 + 0.01, 0.5)
    counts = np.histogram(lm, e)[0]

    def draw(ax):
        ax.bar(e[:-1], counts, width=0.44, align="edge", color=BLUE, linewidth=0)
        for k in (int(np.argmax(counts)), len(counts) - 1):
            ax.text(e[k] + 0.22, counts[k], f"{counts[k]:,}", ha="center", va="bottom", fontsize=8, color=INK2_HEX)
        style(ax, f"Stellar mass of the {m.sum():,} selected galaxies", "log10 M* within 2 R½,* [M☉]", "galaxies")
    return draw, e, counts


def chart_groups(g, m):
    sizes = np.bincount(g["group"][m])
    sizes = sizes[sizes > 0]
    cats = [("1", 1, 1), ("2", 2, 2), ("3–4", 3, 4), ("5–9", 5, 9), ("10–49", 10, 49), ("50+", 50, 10 ** 9)]
    cats = [c for c in cats if c[1] <= sizes.max()]
    counts = [int(((sizes >= lo) & (sizes <= hi)).sum()) for _, lo, hi in cats]
    held = [int(sizes[(sizes >= lo) & (sizes <= hi)].sum()) for _, lo, hi in cats]

    def draw(ax):
        x = np.arange(len(cats))
        ax.bar(x, counts, width=0.7, color=BLUE, linewidth=0)
        for xi, v in zip(x, counts):
            ax.text(xi, v, f"{v:,}", ha="center", va="bottom", fontsize=8, color=INK2_HEX)
        ax.set_xticks(x, [c for c, _, _ in cats])
        ax.set_yscale("log")
        ax.set_ylim(0.7, max(counts) * 3)
        style(ax, "Halos by how many selected galaxies they hold", "selected galaxies in the halo",
              "halos (log scale)", note=f"largest: {sizes.max():,} galaxies in one halo")
    return draw, sizes, cats, counts, held


def render(draw, path: Path, figsize=(7.4, 3.4)):
    fig, ax = plt.subplots(figsize=figsize, facecolor=SURFACE)
    draw(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return path


# ---------------------------------------------------------------- layout
def table(rows, widths, header=True, size=8.5):
    cell = ParagraphStyle("cell", parent=STYLES["body"], alignment=0, spaceAfter=0, fontSize=size, leading=size * 1.3)
    rows = [[Paragraph(f"<b>{c}</b>" if (header and i == 0) else c, cell) for c in r] for i, r in enumerate(rows)]
    t = Table(rows, colWidths=widths)
    t.setStyle(TableStyle([
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8 if header else 0.3, colors.HexColor("#c3c2b7")),
        ("LINEBELOW", (0, 1), (-1, -1), 0.3, colors.HexColor("#e1e0d9")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def tiles(items):
    big = ParagraphStyle("big", parent=STYLES["title"], fontSize=22, leading=26, spaceAfter=2)
    small = ParagraphStyle("tile", parent=STYLES["small"], fontSize=8.5, leading=11)
    t = Table([[Paragraph(v, big) for v, _ in items], [Paragraph(lab, small) for _, lab in items]],
              colWidths=[IMG_W / len(items)] * len(items))
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                           ("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.HexColor("#c3c2b7")),
                           ("TOPPADDING", (0, 0), (-1, 0), 8)]))
    return t


def figure(story, heading, img, text):
    w, h = mpimg.imread(img).shape[1::-1]
    story.append(KeepTogether([Paragraph(heading, STYLES["h2"]), Image(str(img), width=IMG_W, height=IMG_W * h / w),
                               Paragraph(text, STYLES["body"]), Spacer(1, 4)]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", required=True, help="complete snapshot copy (snapshots/, groupcats/, offsets/)")
    ap.add_argument("--hist2d", type=Path, required=True)
    ap.add_argument("--histall", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    g = galaxies(args.root)
    d2, da = load(args.hist2d), load(args.histall)
    panels = args.out.parent / "panels"
    panels.mkdir(parents=True, exist_ok=True)

    steps_ = funnel(g)
    final, masked = select(g, THRESH, True), select(g, THRESH, False)
    low, low_masked = select(g, BUDGET, True), select(g, BUDGET, False)
    before_dm = steps_[2][1]                                     # flag = 1 and >= THRESH stars
    before_r = steps_[3][1]
    before_gas = steps_[4][1]
    base = steps_[1][1]                                          # flag = 1 and >= 2,048 stars
    sel_stars = g["flag"] & (g["n_star"] >= 100)

    draw_cube, cube_rows = chart_cube(da)
    draw_temp, sf_mode = chart_temperature(da)
    draw_ms, ms_edges, ms_counts = chart_mstar(g, final)
    draw_groups, sizes, cats, gcounts, gheld = chart_groups(g, final)
    imgs = {
        "cube": render(draw_cube, panels / "cube_rule.png", (7.4, 2.6)),
        "funnel": render(chart_funnel(g, steps_), panels / "funnel.png", (7.4, 3.3)),
        "stars": render(chart_hist_log(np.log10(g["n_star"][sel_stars & g["cen"]]),
                                       np.log10(g["n_star"][sel_stars & ~g["cen"]]), np.arange(2.0, 8.11, 0.1),
                                       "Star particles per galaxy (galaxies with at least 100)",
                                       "log10 bound star particles",
                                       lines=[(np.log10(BUDGET), f"{BUDGET:,}"), (np.log10(THRESH), f"{THRESH:,}")],
                                       note=f"dashed: the {BUDGET:,}-point budget and twice that", headroom=12),
                        panels / "stars.png"),
        "rstar": render(chart_hist_log(np.log10(g["rstar"][before_r & g["cen"]]),
                                       np.log10(g["rstar"][before_r & ~g["cen"]]), np.arange(-1.0, 1.81, 0.05),
                                       "Stellar half-mass radius", "log10 R½,* [kpc]",
                                       lines=[(np.log10(2 * SOFTENING), f"{2 * SOFTENING:.2f} kpc")],
                                       note=f"below: {(before_r & (g['rstar'] < 2 * SOFTENING)).sum():,}",
                                       vertical=False),
                        panels / "rstar.png"),
        "gas": render(chart_hist_log(np.log10(g["n_gas"][before_gas & g["cen"] & (g["n_gas"] > 0)]),
                                     np.log10(g["n_gas"][before_gas & ~g["cen"] & (g["n_gas"] > 0)]),
                                     np.arange(0.0, 9.01, 0.1), "Gas cells per galaxy (galaxies with gas)",
                                     "log10 bound gas cells", lines=[(np.log10(THRESH), f"{THRESH:,}")],
                                     note=f"no gas at all: {(before_gas & (g['n_gas'] == 0)).sum():,}",
                                     vertical=False),
                      panels / "gas.png"),
        "wind": render(chart_wind(d2, da), panels / "wind.png"),
        "sfr": render(chart_sfr(g, before_gas), panels / "sfr.png", (7.4, 2.2)),
        "temp": render(draw_temp, panels / "temperature.png"),
        "mstar": render(draw_ms, panels / "mstar.png"),
        "groups": render(draw_groups, panels / "groups.png"),
    }

    n_cen, n_sat = int((final & g["cen"]).sum()), int((final & ~g["cen"]).sum())
    groups_final = len(np.unique(g["group"][final]))
    lm = np.log10(g["mstar"][final])
    doc = SimpleDocTemplate(str(args.out), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm,
                            title="GalaxyCLIP-3D: filtering the 3D point-cloud dataset", author="GalaxyCLIP-3D")
    body, h1 = STYLES["body"], STYLES["h1"]
    story = [
        Paragraph("Filtering the 3D point-cloud dataset for CLIP", STYLES["title"]),
        Paragraph(f"Suggestions without the 2D maps · GalaxyCLIP-3D · {datetime.date.today():%Y-%m-%d} · TNG50-1 "
                  "snapshot 99, complete copy · generated by tools/build_filter_report.py", STYLES["subtitle"]),
        Paragraph(
            "These suggestions are for a dataset of 3D point clouds alone: the 2D maps and the rule that chose their "
            "halos are set aside. Counts come from the group catalog of the complete snapshot copy and use every "
            "particle bound to a galaxy, so they will shrink once a region size is chosen and should be recounted "
            "then. Particle-level charts use the two samples of the data-distribution report: "
            f"{len(d2['sample_gid'])} halos with 2D maps and {len(da['sample_gid'])} halos spread over all masses.",
            body),
        Spacer(1, 4),
        tiles([(f"{final.sum():,}", f"galaxies in the starting configuration ({n_cen:,} centrals, {n_sat:,} "
                                    "satellites)"),
               (f"{masked.sum():,}", "if galaxies without enough gas stay in, with their gas modality masked"),
               (f"{OLD_SET:,}", "halos in the 2D-matched set, for comparison")]),
        Spacer(1, 10),
        Paragraph("The suggestions in short", STYLES["h2"]),
        Paragraph(
            "1. Make each galaxy the object, not the 2D cube: one Subfind subhalo, satellites included, with one "
            "region shared by all modalities and coordinates wrapped across the box edge.<br/>"
            f"2. Apply four hard filters: real galaxies only (SubhaloFlag = 1), at least twice the point budget of "
            f"particles in each modality ({THRESH:,} for {BUDGET:,} points), a stellar half-mass radius of at least "
            f"two softening lengths ({2 * SOFTENING:.2f} kpc), and enough gas cells or a masked gas modality. Drop "
            "wind cells and black holes from the clouds.<br/>"
            "3. Keep quenched and gas-poor galaxies, star-forming gas, mergers and massive groups, and handle them "
            "with flags, masks and sampling instead.<br/>"
            "4. Balance batches by stellar mass, split train, validation and test by halo, and compare filter "
            "variants on one fixed validation set.<br/>"
            "5. Start from the configuration in section 5 and test the variants listed there.", body),
        PageBreak(),
    ]

    # 1. object
    story += [Paragraph("1. The object: galaxies instead of the 2D cube", h1), Paragraph(
        "The 2D maps show the particles of a friends-of-friends (FoF) halo inside a cube around the central galaxy, "
        "with half-width equal to the central's gas half-mass radius. That rule existed to match the maps. Applied "
        "to every halo it misbehaves, and it ignores satellites, which are galaxies in their own right.", body)]
    figure(story, "Where the cube rule breaks down", imgs["cube"],
           f"Among all {len(da['avail_gid']):,} halos whose central galaxy has gas and stars: "
           + "; ".join(f"{lab} for {v:,}" for lab, v in cube_rows) + ". Without the 2D maps there is no reason to "
           "accept any of this.")
    story.append(Paragraph(
        "<b>Suggestion.</b> Use Subfind subhalos as the objects: each galaxy with the particles bound to it, centrals "
        "and satellites alike. Centre each cloud on SubhaloPos and cut one spherical region that every modality "
        "shares, scaled to the galaxy; a few stellar half-mass radii is a reasonable first choice, compared against "
        "a larger region because gas and neutral hydrogen extend beyond the stars. Wrap coordinates across the "
        "periodic box instead of dropping galaxies near the edge. Note that a central's bound particles include the "
        "whole diffuse halo, so the region, not the subhalo, decides how much circumgalactic gas a cloud shows.",
        body))
    story.append(PageBreak())

    # 2. hard filters
    counts = {lab: int(m.sum()) for lab, m in steps_}
    labels = [lab for lab, _ in steps_]
    story += [Paragraph("2. Hard filters", h1), Paragraph(
        f"Four filters remove galaxies that would hurt training. The chart starts from the "
        f"{counts[labels[0]]:,} subhalos with at least 2,048 star particles and applies them in turn.", body)]
    figure(story, "From all galaxies to the starting configuration", imgs["funnel"],
           f"SubhaloFlag removes {counts[labels[0]] - counts[labels[1]]:,} subhalos, the particle thresholds "
           f"{counts[labels[1]] - counts[labels[3]]:,}, the size filter {counts[labels[3]] - counts[labels[4]]:,} and "
           f"the gas filter {counts[labels[4]] - counts[labels[5]]:,}, of which "
           f"{(steps_[4][1] & ~steps_[5][1] & ~g['cen']).sum():,} are satellites. "
           f"{final.sum():,} galaxies remain ({n_cen:,} centrals, {n_sat:,} satellites) in {groups_final:,} halos.")
    figure(story, f"Particles: at least twice the point budget ({THRESH:,} for {BUDGET:,} points)", imgs["stars"],
           f"With fewer particles than points, a cloud repeats points, and with exactly as many it is the same cloud "
           f"every epoch, so fresh sampling no longer works as augmentation. The encoder could also learn the amount "
           f"of repetition as a stand-in for galaxy mass. Of the galaxies with SubhaloFlag = 1, "
           f"{(base & (g['n_star'] < THRESH)).sum():,} have between 2,048 and {THRESH:,} star particles and drop out "
           f"here. The same threshold applied to dark matter removes "
           f"{counts[labels[2]] - counts[labels[3]]:,} more.")
    figure(story, f"Size: stellar half-mass radius of at least {2 * SOFTENING:.2f} kpc", imgs["rstar"],
           f"Gravity in the simulation is smoothed over {SOFTENING} kpc for stars and dark matter. Below about two "
           f"smoothing lengths a galaxy's shape is set by this numerical smoothing rather than by physics, so a point "
           f"cloud of it carries no real structure. "
           f"{(before_r & (g['rstar'] < 2 * SOFTENING)).sum():,} of the galaxies that pass the particle thresholds are "
           f"smaller than that.")
    no_gas = before_gas & (g["n_gas"] == 0)
    biggest = np.bincount(g["group"][base]).argmax()
    n_big = [int((m & (g["group"] == biggest)).sum()) for m in (base, final, masked)]
    figure(story, f"Gas: at least {THRESH:,} gas cells, or a masked gas modality", imgs["gas"],
           f"CLIP needs both sides of every pair, and a galaxy without gas cannot supply a gas cloud. "
           f"{no_gas.sum():,} galaxies that pass everything else have no gas at all, "
           f"{(no_gas & ~g['cen']).sum():,} of them satellites, which typically lose their gas to the host halo, and "
           f"{(before_gas & (g['n_gas'] > 0) & (g['n_gas'] < THRESH)).sum():,} more have fewer than {THRESH:,} cells. "
           f"Either drop them ({final.sum():,} galaxies left) or keep them with the gas modality masked in the loss "
           f"({masked.sum():,}). Dropping them thins out dense environments: the largest halo holds {n_big[0]:,} "
           f"galaxies with SubhaloFlag = 1 and at least 2,048 star particles, of which {n_big[1]:,} survive with gas "
           f"required and {n_big[2]:,} with gas masked. If environment matters for what the model should learn, "
           "masking is the better choice.")
    figure(story, "Inside the clouds: drop wind cells and black holes", imgs["wind"],
           "Wind cells are gas in transit that the simulation stores among the star particles; they are not stars. "
           f"They are a small share of the star-particle mass (median {np.median(d2['sample_wind_mass_frac']):.2%} "
           f"for halos with 2D maps, {np.median(da['sample_wind_mass_frac']):.2%} across all masses, at most "
           f"{max(d2['sample_wind_mass_frac'].max(), da['sample_wind_mass_frac'].max()):.1%}), so removing them "
           "keeps the star clouds physically clean at almost no cost. Black holes (4,055 particles in the whole box) "
           "are too few to form a modality.")

    # 3. keep
    sf0 = before_gas & (g["sfr"] <= 0)
    story += [Paragraph("3. Keep, but handle differently", h1), Paragraph(
        "Some groups of galaxies look like candidates for removal but are real physics that the model should see. "
        "They need handling, not filtering.", body)]
    figure(story, "Galaxies without star formation", imgs["sfr"],
           f"{sf0.sum():,} of the {before_gas.sum():,} galaxies that reach the gas filter form no stars "
           f"({(sf0 & ~g['cen']).sum():,} satellites, {(sf0 & g['cen']).sum():,} centrals). They are part of the "
           "diversity the model should learn, so keep them. If neutral hydrogen is a modality of its own, give it a "
           "minimum number of cells with neutral gas, or mask it like gas.")
    figure(story, "Star-forming gas: keep it, with a flag", imgs["temp"],
           f"Star-forming gas sits at about 10<super>{sf_mode:.1f}</super> K because its temperature and neutral "
           "fraction come from the simplified model the simulation uses for the interstellar medium, not from "
           "resolved physics. A per-point star-forming flag lets the model tell these points apart, instead of "
           "learning the model's fixed temperature as if it were a real feature.")
    story.append(Paragraph(
        "<b>Also keep.</b> Mergers: with subhalos as objects, each galaxy of a merging pair is its own object, so no "
        "filter is needed. Massive groups and clusters: keep them, sample points in proportion to mass and scale "
        "positions by the region's radius. Cell mass and metallicity: no filter is needed, because 99.8% of gas "
        "cells lie within a factor of two of the target mass and no sampled cell has zero metallicity.", body))
    story.append(PageBreak())

    # 4. balance and splits
    k = int(np.argmax(ms_counts))
    story += [Paragraph("4. Balance and splits", h1), Paragraph(
        "These choices do not remove galaxies, but they matter as much as the filters for contrastive training.",
        body)]
    figure(story, "Stellar mass: balance the batches", imgs["mstar"],
           f"Small galaxies dominate: {ms_counts[k]:,} of the {final.sum():,} have log M* between {ms_edges[k]:.1f} "
           f"and {ms_edges[k + 1]:.1f}, and only {(lm >= 11).sum()} are above 10<super>11</super> M☉. Random batches "
           "would be full of similar-looking dwarfs, which makes the contrastive task easy at the massive end and "
           "noisy at the low end. Sample batches evenly across stellar-mass bins, or weight them, and report "
           "retrieval per mass bin.")
    figure(story, "Halos: split train, validation and test by halo", imgs["groups"],
           f"The {final.sum():,} galaxies live in {groups_final:,} halos. {gcounts[0]:,} halos hold a single "
           f"selected galaxy, while {sum(gcounts[4:]):,} hold ten or more ({sum(gheld[4:]):,} galaxies) and the "
           f"largest holds {sizes.max():,}. Satellites share their host's environment and gas, so a satellite in "
           "training and its central in validation would leak information. Assign whole halos to one split.")
    story.append(Paragraph(
        "<b>Compare variants on one fixed validation set.</b> Retrieval accuracy (R@1) depends on how many candidates "
        "the search runs over: a filter that keeps more galaxies makes retrieval harder by itself. When testing "
        "thresholds, evaluate every variant on the same held-out galaxies, for example those that pass the strictest "
        "variant. <b>More snapshots later.</b> The same galaxy at neighbouring snapshots is almost a duplicate: it "
        "would appear as a false negative inside a batch and leak across splits. Split by merger-tree lineage and "
        "keep a galaxy's snapshots out of the same batch.", body))
    story.append(PageBreak())

    # 5. configuration
    story += [Paragraph("5. Starting configuration and variants", h1), Paragraph(
        "A concrete setup to start from; each line can be tested against the variants below.", body)]
    story.append(table([
        ["Choice", "Starting configuration"],
        ["Object", "Subfind subhalo with SubhaloFlag = 1, centrals and satellites"],
        ["Region", "one sphere around SubhaloPos shared by all modalities, a few stellar half-mass radii; coordinates "
                   "wrapped across the box edge; positions scaled by the region's radius"],
        ["Particles", f"at least {THRESH:,} star particles, dark-matter particles and gas cells ({BUDGET:,} points "
                      "per modality, drawn in proportion to mass, fresh each epoch)"],
        ["Size", f"stellar half-mass radius at least {2 * SOFTENING:.2f} kpc"],
        ["Clouds", "wind cells and black holes removed; star-forming gas flagged per point"],
        ["Batches", "balanced across stellar-mass bins"],
        ["Splits", "by halo; one fixed validation set for comparing variants"],
    ], [32 * mm, IMG_W - 32 * mm]))
    story.append(Spacer(1, 10))
    story.append(Paragraph("Variants: how many galaxies each one keeps", STYLES["h2"]))
    rows = [["Particles per modality", "Gas cells required", "Gas modality masked when short"]]
    for t, req, msk in ((THRESH, final, masked), (BUDGET, low, low_masked)):
        rows.append([f"≥ {t:,}" + (" (starting configuration)" if t == THRESH else f" (= budget of {BUDGET:,})"),
                     f"{req.sum():,} ({(req & g['cen']).sum():,} centrals, {(req & ~g['cen']).sum():,} satellites)",
                     f"{msk.sum():,} ({(msk & g['cen']).sum():,} centrals, {(msk & ~g['cen']).sum():,} satellites)"])
    story.append(table(rows, [52 * mm, 61 * mm, 61 * mm]))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        f"Using {BUDGET:,} particles per modality keeps more galaxies but leaves no headroom for fresh sampling at "
        "the low end. For comparison, the 2D-matched set has "
        f"{OLD_SET:,} halos. <b>Caveat.</b> All counts use the particles bound to each galaxy; once the region is "
        "chosen, recount from the particle data, because the region will cut some of them, most of all the gas around "
        "satellites. The particle-level statements rest on samples of 60 and 51 halos.", body))

    def footer(canvas, doc_):
        canvas.saveState()
        canvas.setFont("DejaVu", 7.5)
        canvas.setFillColor(INK2)
        canvas.drawRightString(A4[0] - 18 * mm, 9 * mm, f"{doc_.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"wrote {args.out}")
    print(f"funnel: {[(lab, int(m.sum())) for lab, m in steps_]}")
    print(f"final {final.sum():,} ({n_cen:,} cen / {n_sat:,} sat, {groups_final:,} halos); masked {masked.sum():,}; "
          f"budget-level {low.sum():,} / {low_masked.sum():,}")


if __name__ == "__main__":
    main()
