"""Plot the filter-candidate histograms collected by tools/collect_histogram_data.py.

Writes three PNG grids (halo level, gas cells, star particles + per-halo fractions) and prints
the numbers behind them (percentiles and counts past candidate thresholds). Every panel is a
separate function so tools/build_report.py can render each one on its own.

    python tools/plot_histograms.py outputs/data_audit/histogram_data.npz
    python tools/plot_histograms.py outputs/data_audit/histogram_data_all.npz
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                         # noqa: E402
import numpy as np                                                       # noqa: E402

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"                     # categorical slots 1-3
POINT_BUDGETS = (2048, 8192, 16384)


def style(ax, title, xlabel, ylabel=None, note=None, note_left=False, note_y=0.97):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelcolor=INK2, labelsize=8, length=3)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=10, color=INK, pad=6)
    ax.set_xlabel(xlabel, fontsize=8, color=INK2)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8, color=INK2)
    if note:
        ax.text(0.01 if note_left else 0.99, note_y, note, transform=ax.transAxes, va="top",
                ha="left" if note_left else "right", fontsize=7.5, color=INK2, linespacing=1.4)


def bars(ax, values, bins, color=BLUE):
    counts, edges = np.histogram(values, bins)
    ax.bar(edges[:-1], counts, width=np.diff(edges) * 0.88, align="edge", color=color, linewidth=0)
    return counts


def steps(ax, counts, edges, color, label, baseline=0.0):
    ax.stairs(counts, edges, color=color, linewidth=2, label=label, baseline=baseline)


def threshold(ax, x, label, vertical=False):
    ax.axvline(x, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    if vertical:            # several close lines: label each along its own line
        ax.text(x, 0.97, f"{label} ", transform=ax.get_xaxis_transform(), fontsize=7, color=INK2,
                va="top", ha="right", rotation=90)
    else:
        ax.text(x, 0.98, f" {label}", transform=ax.get_xaxis_transform(), fontsize=7, color=INK2,
                va="top", ha="left")


def legend(ax, loc="upper left"):
    leg = ax.legend(frameon=False, fontsize=7.5, loc=loc)
    for t in leg.get_texts():
        t.set_color(INK2)


def pct(x, qs=(5, 25, 50, 75, 95)):
    x = x[np.isfinite(x)]
    return " / ".join(f"{v:.3g}" for v in np.percentile(x, qs))


def log(x):
    return np.log10(np.where(x > 0, x, np.nan))


def stellar_share(d):
    return d["halo_mstar_cen"] / np.where(d["halo_mstar_fof"] > 0, d["halo_mstar_fof"], np.nan)


def gas_fraction(d):
    mg, ms = d["halo_mgas_cen_rad"], d["halo_mstar_cen_rad"]
    return mg / np.where(mg + ms > 0, mg + ms, np.nan)


# ---------------------------------------------------------------- panels: halos with 2D maps
def halo_panels(d):
    """[(key, draw(ax)), ...] for the 2D-map population."""
    hi, ms, sfr = d["halo_hi_mass"], d["halo_mstar_cen_rad"], d["halo_sfr_cen_rad"]
    hw, rs = d["avail_half_width_kpc"], d["halo_rhalf_star_kpc"]
    comp = d["avail_complete"] == 1

    def points(ax):
        edges = np.arange(2.0, 9.01, 0.1)
        for key, color, label in (("gas", BLUE, "gas cells"), ("dm", ORANGE, "dark-matter particles"),
                                  ("star", AQUA, "star particles")):
            v = d[f"halo_npts_{key}"]
            steps(ax, np.histogram(np.log10(v[v > 0]), edges)[0], edges, color, label)
        for b in POINT_BUDGETS:
            threshold(ax, np.log10(b), f"{b:,}", vertical=True)
        legend(ax, "upper right")
        style(ax, "Points inside the 2D cube", "log10 points (unrotated view)", "halos")

    def hi_mass(ax):
        bars(ax, np.log10(hi[hi > 0]), np.arange(3.0, 12.01, 0.1))
        style(ax, "HI mass inside the cube", "log10 M_HI [M☉]", "halos",
              note=f"zero or missing: {int((~(hi > 0)).sum())}")

    def halo_mass(ax):
        m200, e = np.log10(d["halo_m200c"]), np.arange(10.4, 14.41, 0.1)
        steps(ax, np.histogram(m200[comp], e)[0], e, BLUE, f"complete on disk ({comp.sum()})")
        steps(ax, np.histogram(m200[~comp], e)[0], e, ORANGE, f"incomplete ({(~comp).sum()})")
        legend(ax, "upper right")
        style(ax, "Halo mass", "log10 M200c [M☉]", "halos")

    def stellar_mass(ax):
        bars(ax, np.log10(ms[ms > 0]), np.arange(6.0, 12.51, 0.1))
        style(ax, "Central galaxy stellar mass", "log10 M* within 2 R½,* [M☉]", "halos",
              note=f"M* = 0: {int((ms <= 0).sum())}")

    def ssfr(ax):
        ok = (sfr > 0) & (ms > 0)
        bars(ax, np.log10(sfr[ok] / ms[ok]), np.arange(-14.0, -7.99, 0.1))
        threshold(ax, -11, "quenched below")
        style(ax, "Central specific star-formation rate", "log10 sSFR within 2 R½,* [yr⁻¹]", "halos",
              note=f"SFR = 0: {int((sfr <= 0).sum())}")

    def half_width(ax):
        bars(ax, np.log10(hw), np.arange(0.8, 2.86, 0.025))
        style(ax, "Cube half-width (central gas half-mass radius)", "log10 half-width [kpc]", "halos")

    def hw_r200(ax):
        bars(ax, hw / d["halo_r200c_kpc"], np.arange(0.0, 1.21, 0.02))
        style(ax, "Cube half-width / R200c", "ratio", "halos")

    def rstar_hw(ax):
        bars(ax, np.log10(rs[rs > 0] / hw[rs > 0]), np.arange(-2.5, 0.51, 0.05))
        style(ax, "Stellar half-mass radius / cube half-width", "log10 ratio", "halos")

    def share(ax):
        bars(ax, stellar_share(d), np.arange(0.0, 1.001, 0.02))
        style(ax, "Central's share of the halo's stellar mass", "M*(central) / M*(FoF)", "halos")

    def nsubs(ax):
        bars(ax, np.log10(d["halo_nsubs"]), np.arange(0.0, 4.01, 0.1))
        style(ax, "Subhalos per halo", "log10 number of subhalos", "halos")

    def gas_frac(ax):
        bars(ax, gas_fraction(d), np.arange(0.0, 1.001, 0.02))
        style(ax, "Central gas fraction within 2 R½,*", "M_gas / (M_gas + M*)", "halos")

    def hi_to_stars(ax):
        okh = (hi > 0) & (ms > 0)
        bars(ax, np.log10(hi[okh] / ms[okh]), np.arange(-5.0, 2.01, 0.1))
        style(ax, "HI in the cube relative to the central's stars", "log10 M_HI(cube) / M*", "halos")

    return [("points", points), ("hi_mass", hi_mass), ("halo_mass", halo_mass), ("stellar_mass", stellar_mass),
            ("ssfr", ssfr), ("half_width", half_width), ("hw_r200", hw_r200), ("rstar_hw", rstar_hw),
            ("share", share), ("nsubs", nsubs), ("gas_frac", gas_frac), ("hi_to_stars", hi_to_stars)]


# ---------------------------------------------------------------- panels: all halos vs 2D
def all_halo_panels(d):
    """[(key, draw(ax)), ...] for every halo whose central galaxy has gas and stars."""
    n = len(d["avail_gid"])
    in2d = d["avail_in_2d"] == 1
    all_lbl, d2_lbl = f"all ({n:,})", f"with 2D maps ({in2d.sum():,})"
    ms, sfr, hw, rs = d["halo_mstar_cen_rad"], d["halo_sfr_cen_rad"], d["avail_half_width_kpc"], d["halo_rhalf_star_kpc"]

    def pair(ax, values, edges, title, xlabel, note=None, loc="upper right", note_left=None, note_y=0.97):
        ok = np.isfinite(values)
        steps(ax, np.histogram(values[ok], edges)[0], edges, BLUE, all_lbl, baseline=0.5)
        steps(ax, np.histogram(values[ok & in2d], edges)[0], edges, ORANGE, d2_lbl, baseline=0.5)
        ax.set_yscale("log")
        ax.set_ylim(0.5, None)
        legend(ax, loc)
        style(ax, title, xlabel, "halos (log scale)", note=note,
              note_left=(loc == "upper right") if note_left is None else note_left, note_y=note_y)

    def central_stars(ax):
        ns = d["halo_ncen_star"]
        pair(ax, log(ns), np.arange(0.0, 8.51, 0.1), "Star particles in the central galaxy",
             "log10 bound star particles", note_left=False, note_y=0.84,
             note="≥ 2,048: " + f"{(ns >= 2048).sum():,} all / {(ns[in2d] >= 2048).sum():,} 2D")
        for b in POINT_BUDGETS:
            threshold(ax, np.log10(b), f"{b:,}", vertical=True)

    def fof_particles(ax):
        e = np.arange(0.0, 9.01, 0.1)
        for key, color, label in (("n_gas", BLUE, "gas cells"), ("n_dm", ORANGE, "dark-matter particles"),
                                  ("n_stars", AQUA, "star particles")):
            steps(ax, np.histogram(log(d[f"avail_{key}"].astype(float)), e)[0], e, color, label, baseline=0.5)
        ax.set_yscale("log")
        ax.set_ylim(0.5, None)
        legend(ax, "upper right")
        style(ax, "Particles in the whole FoF group (all halos)", "log10 particles (upper bound for the cube)",
              "halos (log scale)")

    def halo_mass(ax):
        pair(ax, log(d["halo_m200c"]), np.arange(8.0, 14.51, 0.1), "Halo mass", "log10 M200c [M☉]")

    def stellar_mass(ax):
        pair(ax, log(ms), np.arange(4.0, 12.51, 0.1), "Central galaxy stellar mass",
             "log10 M* within 2 R½,* [M☉]", note=f"M* = 0 within 2 R½,*: {(ms <= 0).sum():,}",
             note_left=False, note_y=0.84)

    def ssfr(ax):
        pair(ax, log(sfr / np.where(ms > 0, ms, np.nan)), np.arange(-14.0, -7.99, 0.1),
             "Central specific star-formation rate", "log10 sSFR within 2 R½,* [yr⁻¹]", loc="upper left",
             note_left=True, note_y=0.80, note=f"SFR = 0: {(sfr <= 0).sum():,} all / {(sfr[in2d] <= 0).sum():,} 2D")
        threshold(ax, -11, "quenched below")

    def half_width(ax):
        pair(ax, log(hw), np.arange(-1.0, 3.01, 0.05), "Cube half-width (central gas half-mass radius)",
             "log10 half-width [kpc]")

    def hw_r200(ax):
        pair(ax, hw / d["halo_r200c_kpc"], np.arange(0.0, 3.01, 0.05), "Cube half-width / R200c", "ratio")

    def rstar_hw(ax):
        pair(ax, log(rs / hw), np.arange(-3.0, 1.01, 0.05), "Stellar half-mass radius / cube half-width",
             "log10 ratio")

    def share(ax):
        pair(ax, stellar_share(d), np.arange(0.0, 1.001, 0.02), "Central's share of the halo's stellar mass",
             "M*(central) / M*(FoF)", loc="upper left")

    def nsubs(ax):
        pair(ax, log(d["halo_nsubs"]), np.arange(0.0, 4.01, 0.1), "Subhalos per halo", "log10 number of subhalos")

    def gas_frac(ax):
        pair(ax, gas_fraction(d), np.arange(0.0, 1.001, 0.02), "Central gas fraction within 2 R½,*",
             "M_gas / (M_gas + M*)", loc="upper center")

    def complete(ax):
        m, e = log(d["halo_m200c"]), np.arange(8.0, 14.51, 0.25)
        idx = np.digitize(m, e) - 1
        frac = np.array([np.mean(d["avail_complete"][idx == k]) if (idx == k).sum() >= 5 else np.nan
                         for k in range(len(e) - 1)])
        ax.plot(0.5 * (e[:-1] + e[1:]), 100 * frac, color=BLUE, linewidth=2)
        ax.set_ylim(0, 105)
        style(ax, "Halos complete on disk", "log10 M200c [M☉]", "% of halos (bins with ≥ 5)",
              note=f"complete: {(d['avail_complete'] == 1).sum():,} of {n:,}\n"
                   f"cube crosses box edge: {(d['avail_crosses_box_edge'] == 1).sum():,}", note_left=True)

    return [("central_stars", central_stars), ("fof_particles", fof_particles), ("halo_mass", halo_mass),
            ("stellar_mass", stellar_mass), ("ssfr", ssfr), ("half_width", half_width), ("hw_r200", hw_r200),
            ("rstar_hw", rstar_hw), ("share", share), ("nsubs", nsubs), ("gas_frac", gas_frac),
            ("complete", complete)]


# ---------------------------------------------------------------- panels: particles of the sample
def gas_panels(d):
    total = sum(d[f"hist_logT_{ph}"].sum() for ph in ("sf", "nonsf"))
    sf_share = d["hist_logT_sf"].sum() / total
    specs = [("logT", "Temperature", "log10 T [K]", "upper right"),
             ("lognH", "Hydrogen number density", "log10 n_H [cm⁻³]", "upper left"),
             ("logB_uG", "Magnetic field strength", "log10 |B| [µG]", "upper left"),
             ("xHI", "Neutral hydrogen fraction", "x_HI", "upper center"),
             ("logZ", "Gas metallicity", "log10 Z/Z☉ (floored at −4)", "upper left"),
             ("logr_kpc", "Cell size (equal-volume sphere radius)", "log10 r_cell [kpc]", "upper right")]

    def mass_weighted(k, title, xl, loc):
        def draw(ax):
            e = d[f"bins_{k}"]
            steps(ax, 100 * d[f"hist_{k}_nonsf"] / total, e, BLUE, f"not star-forming ({1 - sf_share:.0%} of mass)")
            steps(ax, 100 * d[f"hist_{k}_sf"] / total, e, ORANGE, f"star-forming, SFR > 0 ({sf_share:.0%})")
            ax.set_ylim(0, ax.get_ylim()[1] * 1.35)          # headroom so the legend sits above the curves
            legend(ax, loc)
            style(ax, title, xl, "% of gas mass per bin")
        return draw

    def cell_mass(ax):
        e, c = d["bins_m_over_target"], d["hist_m_over_target"]
        ax.bar(e[:-1], 100 * c / c.sum(), width=np.diff(e) * 0.88, align="edge", color=BLUE, linewidth=0)
        threshold(ax, 0.5, "0.5×")
        threshold(ax, 2.0, "2×")
        style(ax, "Cell mass / target gas mass (8.5e4 M☉)", "ratio", "% of cells per bin",
              note=f"{int(c.sum()):,} cells")

    return [(k, mass_weighted(k, t, xl, loc)) for k, t, xl, loc in specs] + [("cell_mass", cell_mass)]


def star_panels(d):
    def distribution(k, title, xl, yl):
        def draw(ax):
            e, c = d[f"bins_{k}"], d[f"hist_{k}"]
            ax.bar(e[:-1], 100 * c / c.sum(), width=np.diff(e) * 0.88, align="edge", color=BLUE, linewidth=0)
            style(ax, title, xl, yl)
        return draw

    def per_halo(k, title, xl):
        def draw(ax):
            v = d[f"sample_{k}"]
            hi = max(0.02, float(np.nanmax(v)) * 1.05)
            bars(ax, v, np.linspace(0, hi, 31))
            style(ax, title, xl, "halos", note=f"median {np.nanmedian(v):.3g}", note_left=np.nanmedian(v) > 0.5 * hi)
        return draw

    return [
        ("age", distribution("age_Gyr", "Stellar age (mass-weighted)", "age [Gyr]", "% of stellar mass per bin")),
        ("star_Z", distribution("star_logZ", "Stellar metallicity (mass-weighted)", "log10 Z/Z☉ (floored at −4)",
                                "% of stellar mass per bin")),
        ("mass_loss", distribution("m_over_init", "Current / birth mass of star particles", "M / M_birth",
                                   "% of star particles per bin")),
        ("wind", per_halo("wind_mass_frac", "Wind cells in PartType4", "wind mass / PartType4 mass in cube")),
        ("sf_frac", per_halo("sf_mass_frac", "Star-forming gas", "star-forming gas mass / gas mass in cube")),
        ("hi_frac", per_halo("hi_mass_frac", "Neutral hydrogen", "neutral H mass / gas mass in cube")),
        ("cold_frac", per_halo("cold_mass_frac", "Cold gas (T < 10^4.5 K)", "cold gas mass / gas mass in cube")),
    ]


# ---------------------------------------------------------------- rendering
def grid(panels, nrows, ncols, suptitle, out: Path, figsize):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor=SURFACE)
    for ax, (_, draw) in zip(axes.flat, panels):
        draw(ax)
    for ax in list(axes.flat)[len(panels):]:
        ax.axis("off")
    fig.suptitle(suptitle, x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96 if nrows > 2 else 0.94))
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def render_each(panels, outdir: Path, prefix: str, figsize=(7.4, 3.5), dpi=200) -> dict[str, Path]:
    """One PNG per panel; returns key -> path."""
    outdir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for key, draw in panels:
        fig, ax = plt.subplots(figsize=figsize, facecolor=SURFACE)
        draw(ax)
        fig.tight_layout()
        paths[key] = outdir / f"{prefix}_{key}.png"
        fig.savefig(paths[key], dpi=dpi, facecolor=SURFACE)
        plt.close(fig)
    return paths


def sample_label(d):
    return "(every complete halo has 2D maps)" if str(d["population"]) == "all" else "with 2D maps"


def report(d):
    print("== halo level (2,991 halos) — percentiles 5/25/50/75/95 ==")
    for key, name in (("gas", "gas cells"), ("dm", "DM particles"), ("star", "star particles")):
        v = d[f"halo_npts_{key}"]
        below = ", ".join(f"<{b:,}: {int((v < b).sum())}" for b in POINT_BUDGETS)
        print(f"points in cube, {name:15s}: {pct(v)} | {below}")
    hi, ms, sfr = d["halo_hi_mass"], d["halo_mstar_cen_rad"], d["halo_sfr_cen_rad"]
    print(f"HI mass in cube [Msun]          : {pct(hi)} | <1e6: {int((hi < 1e6).sum())}, "
          f"<1e7: {int((hi < 1e7).sum())}, <1e8: {int((hi < 1e8).sum())}")
    print(f"central M* (2R1/2) [Msun]       : {pct(ms)} | <1e8: {int((ms < 1e8).sum())}, <1e9: {int((ms < 1e9).sum())}")
    ssfr = np.where(ms > 0, sfr / np.where(ms > 0, ms, 1), np.nan)
    print(f"central sSFR [1/yr]             : SFR=0: {int((sfr <= 0).sum())}, sSFR<1e-11: "
          f"{int(((ssfr < 1e-11) | (sfr <= 0)).sum())}")
    hw = d["avail_half_width_kpc"]
    print(f"cube half-width [kpc]           : {pct(hw)} | >100: {int((hw > 100).sum())}, >200: {int((hw > 200).sum())}")
    print(f"half-width / R200c              : {pct(hw / d['halo_r200c_kpc'])}")
    print(f"R1/2,* / half-width             : {pct(d['halo_rhalf_star_kpc'] / hw)}")
    share = stellar_share(d)
    print(f"central share of FoF stars      : {pct(share)} | <0.5: {int((share < 0.5).sum())}")
    print(f"subhalos per halo               : {pct(d['halo_nsubs'])}")
    comp = d["avail_complete"] == 1
    m = np.log10(d["halo_m200c"])
    print(f"median log M200c complete/incomplete: {np.median(m[comp]):.2f} / {np.median(m[~comp]):.2f}")
    print(f"central SubhaloFlag == 0        : {int((d['halo_flag_cen'] == 0).sum())}")
    print("== gas cells (sample) ==")
    sf = d["hist_logT_sf"].sum()
    print(f"star-forming share of gas mass  : {sf / (sf + d['hist_logT_nonsf'].sum()):.1%}")
    c, e = d["hist_m_over_target"], d["bins_m_over_target"]
    print(f"cells outside 0.5-2x target mass: {c[(e[:-1] < 0.5) | (e[:-1] >= 2.0)].sum() / c.sum():.2%} of {int(c.sum()):,}")
    for k in ("wind_mass_frac", "sf_mass_frac", "hi_mass_frac", "cold_mass_frac"):
        print(f"per-halo {k:15s}: {pct(d[f'sample_{k}'])}")
    print(f"gas cells with Z = 0 (sample)   : {int(d['sample_zero_Z_gas'].sum())}")


def report_all(d):
    n = len(d["avail_gid"])
    in2d = d["avail_in_2d"] == 1
    print(f"== all halos whose central has gas and stars: {n:,} ({in2d.sum():,} with 2D maps) ==")
    ns = d["halo_ncen_star"]
    for b in (100, 1000) + POINT_BUDGETS:
        print(f"central star particles >= {b:6,}: {(ns >= b).sum():6,} all | {(ns[in2d] >= b).sum():5,} 2D | "
              f"{((ns >= b) & ~in2d).sum():5,} without 2D maps")
    ms = d["halo_mstar_cen_rad"]
    for m in (1e7, 1e8, 1e9, 1e10):
        print(f"central M*(2R1/2) >= {m:.0e}: {(ms >= m).sum():6,} all | {(ms[in2d] >= m).sum():5,} 2D")
    comp = d["avail_complete"] == 1
    print(f"complete on disk: {comp.sum():,} all | {(comp & in2d).sum():,} 2D | "
          f"{(comp & ~in2d & (ns >= 2048)).sum():,} without 2D maps and >= 2,048 central stars")
    print(f"cube crosses box edge: {(d['avail_crosses_box_edge'] == 1).sum():,}; central SubhaloFlag == 0: "
          f"{(d['halo_flag_cen'] == 0).sum()}")
    sfr = d["halo_sfr_cen_rad"]
    print(f"SFR = 0: {(sfr <= 0).sum():,} all | {(sfr[in2d] <= 0).sum()} 2D")
    print(f"half-width [kpc] percentiles 5/25/50/75/95: {pct(d['avail_half_width_kpc'])}")
    print("== gas cells / stars (sample) ==")
    sf = d["hist_logT_sf"].sum()
    print(f"star-forming share of gas mass  : {sf / (sf + d['hist_logT_nonsf'].sum()):.1%}")
    for k in ("wind_mass_frac", "sf_mass_frac", "hi_mass_frac", "cold_mass_frac"):
        print(f"per-halo {k:15s}: {pct(d[f'sample_{k}'])}")
    print(f"sample gids: {d['sample_gid'].min()}..{d['sample_gid'].max()}, "
          f"{(np.isin(d['sample_gid'], d['avail_gid'][in2d])).sum()} of {len(d['sample_gid'])} have 2D maps")


def load(path: Path) -> dict:
    d = dict(np.load(path, allow_pickle=False))
    d.setdefault("population", np.array("2d"))
    return d


def main():
    src = Path(sys.argv[1])
    d = load(src)
    out = src.parent
    ng = len(d["sample_gid"])
    pre = "hist_all" if str(d["population"]) == "all" else "hist"
    if str(d["population"]) == "all":
        grid(all_halo_panels(d), 3, 4, f"Halo level — all {len(d['avail_gid']):,} halos whose central galaxy has gas "
             f"and stars (snapshot 99), vs the {(d['avail_in_2d'] == 1).sum():,} with 2D maps",
             out / f"{pre}_halos.png", (17, 11))
    else:
        grid(halo_panels(d), 3, 4, f"Halo level — all {len(d['avail_gid']):,} halos with 2D maps (snapshot 99)",
             out / f"{pre}_halos.png", (17, 11))
    grid(gas_panels(d), 2, 4, f"Gas cells inside the cube — {ng} complete halos {sample_label(d)}, mass-weighted",
         out / f"{pre}_gas.png", (17, 7.6))
    grid(star_panels(d), 2, 4, f"Star particles and per-halo fractions — {ng} complete halos {sample_label(d)}",
         out / f"{pre}_stars.png", (17, 7.6))
    (report_all if str(d["population"]) == "all" else report)(d)
    print(f"figures in {out}")


if __name__ == "__main__":
    main()
