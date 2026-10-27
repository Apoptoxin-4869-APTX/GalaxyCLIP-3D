"""Build a PDF that shows every histogram panel on its own with a short explanation.

Reads the two outputs of tools/collect_histogram_data.py, renders each panel of
tools/plot_histograms.py as its own image, fills the explanations with numbers computed from
the same data, and lays everything out with reportlab.

    python tools/build_report.py outputs/data_audit/histogram_data.npz \
        outputs/data_audit/histogram_data_all.npz outputs/data_audit/data_distributions.pdf

Needs reportlab (pip install --target <dir> reportlab, then put <dir> on PYTHONPATH).
"""
from __future__ import annotations

import datetime
import sys
from pathlib import Path

import matplotlib
import matplotlib.image as mpimg
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_histograms import (all_halo_panels, gas_fraction, gas_panels, halo_panels, load, render_each,  # noqa: E402
                             star_panels, stellar_share)

FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
pdfmetrics.registerFont(TTFont("DejaVu", str(FONT_DIR / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DejaVu-Bold", str(FONT_DIR / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold", italic="DejaVu", boldItalic="DejaVu-Bold")
INK, INK2 = colors.HexColor("#0b0b0b"), colors.HexColor("#52514e")
STYLES = {
    "title": ParagraphStyle("title", fontName="DejaVu-Bold", fontSize=18, leading=23, textColor=INK, spaceAfter=4),
    "subtitle": ParagraphStyle("subtitle", fontName="DejaVu", fontSize=10, leading=14, textColor=INK2, spaceAfter=10),
    "h1": ParagraphStyle("h1", fontName="DejaVu-Bold", fontSize=13.5, leading=18, textColor=INK, spaceBefore=4,
                         spaceAfter=6),
    "h2": ParagraphStyle("h2", fontName="DejaVu-Bold", fontSize=10.5, leading=14, textColor=INK, spaceAfter=2),
    "body": ParagraphStyle("body", fontName="DejaVu", fontSize=9, leading=12.6, textColor=INK, alignment=TA_JUSTIFY,
                           spaceAfter=6),
    "small": ParagraphStyle("small", fontName="DejaVu", fontSize=8, leading=11, textColor=INK2),
}
IMG_W = 174 * mm


# ---------------------------------------------------------------- number helpers
def q(x, p):
    x = np.asarray(x, dtype=float)
    return float(np.percentile(x[np.isfinite(x)], p))


def sci(x, digits=2):
    """1.35e9 -> '1.35×10<super>9</super>' (reportlab markup)."""
    e = int(np.floor(np.log10(abs(x))))
    m = x / 10 ** e
    return f"{m:.{digits - 1}f}×10<super>{e}</super>" if round(m, digits - 1) != 1 else f"10<super>{e}</super>"


def centres(e):
    return 0.5 * (e[:-1] + e[1:])


def wquantile(hist, edges, p):
    c = np.cumsum(hist) / hist.sum()
    return float(centres(edges)[np.searchsorted(c, p)])


def wmode(hist, edges, lo=-np.inf, hi=np.inf):
    m = (centres(edges) >= lo) & (centres(edges) < hi)
    return float(centres(edges)[m][np.argmax(hist[m])])


def wfrac(hist, edges, lo=-np.inf, hi=np.inf, total=None):
    m = (edges[:-1] >= lo) & (edges[:-1] < hi)
    return float(hist[m].sum() / (hist.sum() if total is None else total))


# ---------------------------------------------------------------- explanations
def texts_2d(d):
    hi, ms, sfr = d["halo_hi_mass"], d["halo_mstar_cen_rad"], d["halo_sfr_cen_rad"]
    hw, rs = d["avail_half_width_kpc"], d["halo_rhalf_star_kpc"]
    comp = d["avail_complete"] == 1
    m200 = np.log10(d["halo_m200c"])
    pts = {k: d[f"halo_npts_{k}"] for k in ("gas", "dm", "star")}
    ssfr = np.where(ms > 0, sfr / np.where(ms > 0, ms, 1), np.nan)
    quenched = int(((ssfr < 1e-11) | (sfr <= 0)).sum())
    share, gf = stellar_share(d), gas_fraction(d)
    okh = (hi > 0) & (ms > 0)
    hs = np.log10(hi[okh] / ms[okh])
    return {
        "points": (
            "Points inside the 2D cube",
            f"How many particles of each type the 2D maps contain, read from the maps' metadata (unrotated view), "
            f"so the numbers are exact even for halos whose snapshot files are missing locally. The median halo has "
            f"{q(pts['gas'], 50):,.0f} gas cells, {q(pts['dm'], 50):,.0f} dark-matter particles and "
            f"{q(pts['star'], 50):,.0f} star particles in its cube. Stars are the binding constraint for a "
            f"fixed-size point cloud: {(pts['star'] < 2048).sum()} halos have fewer than 2,048 star particles, "
            f"{(pts['star'] < 4096).sum()} fewer than 4,096, {(pts['star'] < 8192).sum()} fewer than 8,192 and "
            f"{(pts['star'] < 16384).sum():,} fewer than 16,384. Gas and dark matter are not limiting: only "
            f"{(pts['gas'] < 8192).sum()} halos have fewer than 8,192 gas cells and {(pts['dm'] < 16384).sum()} "
            f"fewer than 16,384 dark-matter particles."),
        "hi_mass": (
            "HI mass inside the cube",
            f"Total neutral atomic hydrogen in the cube, obtained by summing the unrotated 2D HI map. Every halo "
            f"has HI: the median is {sci(q(hi, 50))} M☉ and 90% of halos lie between {sci(q(hi, 5))} and "
            f"{sci(q(hi, 95))} M☉. Only {(hi < 1e8).sum()} halos have less than 10<super>8</super> M☉ and "
            f"{(hi < 1e7).sum()} less than 10<super>7</super> M☉; for those, sampling points in proportion to HI "
            f"mass will draw repeatedly from a few cells. The HI modality is usable for essentially every 2D halo."),
        "halo_mass": (
            "Halo mass, complete versus incomplete on disk",
            f"Halo mass M200c for halos whose particles are all on disk (blue, {comp.sum()}) and halos missing at "
            f"least one particle type (orange, {(~comp).sum():,}). The complete subset is strongly biased towards "
            f"massive halos (median log M200c {np.median(m200[comp]):.2f} versus {np.median(m200[~comp]):.2f}), and "
            f"none of the {(d['avail_gid'] >= 2000).sum():,} halos with gid ≥ 2000 is complete. Training only on "
            f"the complete halos would skew the sample to high masses; filling in the missing data avoids this."),
        "stellar_mass": (
            "Central galaxy stellar mass",
            f"Stellar mass of the central galaxy within twice its stellar half-mass radius. 90% of centrals lie "
            f"between {sci(q(ms, 5))} and {sci(q(ms, 95))} M☉ (median {sci(q(ms, 50))} M☉): mostly dwarf to "
            f"Milky Way-mass galaxies plus a tail of massive ellipticals. {(ms < 1e8).sum()} centrals are below "
            f"10<super>8</super> M☉, roughly 1,000 star particles, where star point clouds become sparse."),
        "ssfr": (
            "Central specific star-formation rate",
            f"Star-formation rate per unit stellar mass of the central, both measured within 2 R<sub>1/2,*</sub>. "
            f"Most centrals sit on the star-forming sequence around 10<super>-10</super> yr<super>-1</super>. "
            f"{(sfr <= 0).sum()} have no star-forming gas at all (SFR = 0, not shown) and {quenched} in total are "
            f"quenched (sSFR &lt; 10<super>-11</super> yr<super>-1</super>). The SFR = 0 centrals still hold a "
            f"median of {sci(q(hi[sfr <= 0], 50))} M☉ of HI in their cube, so the cube's HI is not tied to the "
            f"central's star formation; being quenched is not a reason to drop a halo."),
        "half_width": (
            "Cube half-width",
            f"Half-width of the 2D cube, which is the central galaxy's gas half-mass radius. 90% of halos lie "
            f"between {q(hw, 5):.0f} and {q(hw, 95):.0f} kpc (median {q(hw, 50):.0f} kpc), with a long tail: "
            f"{(hw > 100).sum()} cubes are wider than 100 kpc and {(hw > 200).sum()} wider than 200 kpc, up to "
            f"{hw.max():.0f} kpc for the most massive cluster. Scaling positions by the half-width (as the 2D maps "
            f"implicitly do) puts all halos on a common scale, but then one scaled unit corresponds to physical "
            f"sizes that differ by more than a factor of ten."),
        "hw_r200": (
            "Cube half-width relative to R200c",
            f"The cube half-width divided by the halo's virial radius R200c. The cube covers {q(hw / d['halo_r200c_kpc'], 5):.2f}"
            f"–{q(hw / d['halo_r200c_kpc'], 95):.2f} R200c (median {q(hw / d['halo_r200c_kpc'], 50):.2f}), i.e. the "
            f"inner third of the halo: the central galaxy and the inner circumgalactic gas, not the whole halo. "
            f"Satellites in the outer halo mostly fall outside the cube."),
        "rstar_hw": (
            "Stellar half-mass radius relative to the cube",
            f"How large the stellar body is compared with the cube. The stellar half-mass radius is only "
            f"{q(rs / hw, 5):.1%}–{q(rs / hw, 95):.1%} of the cube half-width (median {q(rs / hw, 50):.1%}), so "
            f"most stars sit in a small central region while gas fills the cube. If positions are scaled by the "
            f"cube, star clouds will be very compact; the star modality may need its own scaling (for example by "
            f"the stellar half-mass radius) or a log-radius transform."),
        "share": (
            "Central's share of the halo's stellar mass",
            f"Fraction of the halo's stellar mass that belongs to the central galaxy. Usually the central "
            f"dominates completely (median {q(share, 50):.3f}), but {(share < 0.5).sum()} halos have a central that "
            f"holds less than half of the stars: likely mergers or halos with two comparable galaxies. These are "
            f"candidates for removal, because their cube is centred on a galaxy that is not the main one."),
        "nsubs": (
            "Subhalos per halo",
            f"Number of Subfind subhalos in each halo, including tiny ones without stars. The median is "
            f"{q(d['halo_nsubs'], 50):.0f} (90% between {q(d['halo_nsubs'], 5):.0f} and {q(d['halo_nsubs'], 95):.0f}), "
            f"rising to {d['halo_nsubs'].max():,.0f} in the cluster. Most of these subhalos are small or outside "
            f"the cube; the panel mainly shows how crowded the most massive halos are."),
        "gas_frac": (
            "Central gas fraction",
            f"Gas mass over gas plus stellar mass of the central, within 2 R<sub>1/2,*</sub>. The distribution is "
            f"broad (median {q(gf, 50):.2f}), from gas-rich star-forming discs to gas-poor centrals; "
            f"{(gf < 0.01).sum()} centrals have almost no gas there (&lt; 1%). These still have gas elsewhere in "
            f"their cube, which is set by the gas half-mass radius of the whole subhalo, so this is not a reason "
            f"to drop them."),
        "hi_to_stars": (
            "HI in the cube relative to the central's stellar mass",
            f"HI mass in the cube divided by the central's stellar mass. The median log ratio is {np.median(hs):.2f}, "
            f"with 90% between {q(hs, 5):.2f} and {q(hs, 95):.2f}: HI-rich dwarfs to the right, HI-poor massive "
            f"galaxies to the left. This spread is physical and part of what the model should learn; it does not "
            f"suggest a filter."),
    }


def texts_all(d, d2):
    n = len(d["avail_gid"])
    in2d = d["avail_in_2d"] == 1
    ns, ms, sfr = d["halo_ncen_star"], d["halo_mstar_cen_rad"], d["halo_sfr_cen_rad"]
    hw, rs = d["avail_half_width_kpc"], d["halo_rhalf_star_kpc"]
    m200 = np.log10(np.where(d["halo_m200c"] > 0, d["halo_m200c"], np.nan))
    comp = d["avail_complete"] == 1
    share, gf = stellar_share(d), gas_fraction(d)
    cube_frac = d2["halo_npts_gas"] / d2["avail_n_gas"]
    first_complete = np.nanmin(m200[comp])
    return {
        "central_stars": (
            "Star particles in the central galaxy",
            f"Bound star particles of the central galaxy, from the group catalog, for all {n:,} halos whose "
            f"central has gas and stars (blue) and the {in2d.sum():,} with 2D maps (orange). The extra halos are "
            f"overwhelmingly tiny: {(ns < 100).sum():,} have fewer than 100 star particles. Halos without 2D maps "
            f"add {((ns >= 2048) & ~in2d).sum():,} galaxies with at least 2,048 star particles, "
            f"{((ns >= 8192) & ~in2d).sum()} with at least 8,192 and only {((ns >= 16384) & ~in2d).sum()} with at "
            f"least 16,384: the 2D set already contains nearly every galaxy large enough for a dense star cloud."),
        "fof_particles": (
            "Particles in the whole FoF group",
            f"Particle counts of the entire FoF group for all halos. These are upper bounds for the cube: in the 2D "
            f"halos the cube held {q(cube_frac, 5):.0%}–{q(cube_frac, 95):.0%} of the group's gas. The median halo "
            f"has {q(d['avail_n_gas'], 50):,.0f} gas cells, {q(d['avail_n_dm'], 50):,.0f} dark-matter particles "
            f"and {q(d['avail_n_stars'], 50):,.0f} star particles in its group, far fewer than the 2D halos. The "
            f"spike at one to a few star particles is the large population of barely resolved galaxies."),
        "halo_mass": (
            "Halo mass",
            f"Halo mass of all halos with a galaxy (median log M200c {np.nanmedian(m200):.2f}, down to "
            f"{np.nanmin(m200):.1f}) versus the 2D set (median {np.nanmedian(m200[in2d]):.2f}, from "
            f"{np.nanmin(m200[in2d]):.1f}). The 2D maps cover the top ~3,000 halos by mass; below about "
            f"10<super>10.5</super> M☉ halos host galaxies of at most a few hundred star particles."),
        "stellar_mass": (
            "Central galaxy stellar mass",
            f"Central stellar mass within 2 R<sub>1/2,*</sub>. {(ms <= 0).sum():,} centrals have no stars inside "
            f"that radius (they consist of one or two star particles), and the rest peak around "
            f"10<super>5</super>–10<super>6</super> M☉. Above 10<super>9</super> M☉ almost every galaxy already has "
            f"2D maps ({(ms[in2d] >= 1e9).sum():,} of {(ms >= 1e9).sum():,}); below 10<super>8</super> M☉ almost "
            f"none do ({(in2d & (ms < 1e8)).sum()} of {(ms < 1e8).sum():,})."),
        "ssfr": (
            "Central specific star-formation rate",
            f"{(sfr <= 0).sum():,} of {n:,} centrals ({(sfr <= 0).mean():.0%}) have no star-forming gas, versus "
            f"{(sfr[in2d] <= 0).sum()} of {in2d.sum():,} in the 2D set: most small galaxies in TNG50 are not forming "
            f"stars at z = 0. Those that are have specific star-formation rates similar to the 2D centrals."),
        "half_width": (
            "Cube half-width",
            f"The 2D cube rule applied to every halo. The median cube shrinks from {np.median(hw[in2d]):.0f} kpc "
            f"(2D set) to {np.median(hw):.0f} kpc (all halos), and {(hw < 1).sum()} cubes are smaller than 1 kpc, "
            f"comparable to a single gas cell. Extending the dataset would need a minimum half-width or a minimum "
            f"number of cells per cube."),
        "hw_r200": (
            "Cube half-width relative to R200c",
            f"For small halos the gas half-mass radius is a larger fraction of the virial radius, and "
            f"{(hw / d['halo_r200c_kpc'] > 1).sum():,} halos have a cube wider than R200c. Their cube would include "
            f"gas outside the halo itself, so the 2D rule does not transfer cleanly to the smallest halos."),
        "rstar_hw": (
            "Stellar half-mass radius relative to the cube",
            f"The relation is broader than in the 2D set and {(rs / hw > 1).sum():,} halos have a stellar body "
            f"larger than the cube. The extra halos are structurally different, not just smaller versions of the "
            f"2D halos."),
        "share": (
            "Central's share of the halo's stellar mass",
            f"As in the 2D set, the central holds almost all the stars in nearly every halo. {(share < 0.5).sum():,} "
            f"halos have a central with less than half of the stars ({(share[in2d] < 0.5).sum()} of them in the 2D "
            f"set)."),
        "nsubs": (
            "Subhalos per halo",
            f"Small halos contain very few subhalos: {(d['halo_nsubs'] <= 3).sum():,} halos have three or fewer, "
            f"while every 2D halo has at least {d['halo_nsubs'][in2d].min():.0f}. The 2D halos are the crowded, "
            f"massive end of the population."),
        "gas_frac": (
            "Central gas fraction",
            f"Compared with the 2D set, all halos add two populations: centrals made almost entirely of gas "
            f"({(gf > 0.9).sum():,} with a gas fraction above 0.9, the gas-rich dwarfs) and centrals with no gas "
            f"inside 2 R<sub>1/2,*</sub> ({(gf < 0.01).sum():,})."),
        "complete": (
            "Halos complete on disk",
            f"Share of halos whose particles are all on disk, versus halo mass. No halo below log M200c "
            f"{first_complete:.1f} is complete, and all {comp.sum()} complete halos have 2D maps: the stars of "
            f"every other halo sit in snapshot files 621–678 and their dark matter in files 241–356, where most of "
            f"the missing files are. The curve is noisy at high mass because there are only a few halos per bin. "
            f"{(d['avail_crosses_box_edge'] == 1).sum()} cubes cross the box edge "
            f"({(d['avail_crosses_box_edge'][in2d] == 1).sum()} in the 2D set)."),
    }


def texts_gas(d):
    tot = d["hist_logT_sf"].sum() + d["hist_logT_nonsf"].sum()
    h = lambda k, ph: d[f"hist_{k}_{ph}"]                                     # noqa: E731
    e = lambda k: d[f"bins_{k}"]                                              # noqa: E731
    both = lambda k: h(k, "sf") + h(k, "nonsf")                               # noqa: E731
    r_sf_pc = 1000 * 10 ** wquantile(h("logr_kpc", "sf"), e("logr_kpc"), 0.5)
    r_lo, r_hi = (10 ** wquantile(h("logr_kpc", "nonsf"), e("logr_kpc"), p) for p in (0.05, 0.95))
    c, em = d["hist_m_over_target"], d["bins_m_over_target"]
    in_range = c[(em[:-1] >= 0.5) & (em[:-1] < 2.0)].sum() / c.sum()
    return {
        "logT": (
            "Gas temperature",
            f"Mass-weighted gas temperature, split into star-forming cells (SFR &gt; 0, orange) and the rest "
            f"(blue). Non-star-forming gas piles up at 10<super>4</super> K, the floor set by cooling against the "
            f"UV background, with a second, hot component peaking near "
            f"10<super>{wmode(h('logT', 'nonsf'), e('logT'), 5.0):.1f}</super> K "
            f"({wfrac(both('logT'), e('logT'), 5.0):.0%} of all gas mass is above 10<super>5</super> K). "
            f"Star-forming gas ({h('logT', 'sf').sum() / tot:.0%} of the mass) sits at about "
            f"10<super>{wmode(h('logT', 'sf'), e('logT')):.1f}</super> K: the effective temperature of TNG's "
            f"sub-grid ISM model, not a physical temperature. The 2D temp maps contain the same values."),
        "lognH": (
            "Hydrogen number density",
            f"Star-forming gas begins sharply at n<sub>H</sub> ≈ 0.1 cm<super>-3</super>, TNG's star-formation "
            f"threshold. Non-star-forming gas has two peaks: diffuse halo gas near "
            f"10<super>{wmode(h('lognH', 'nonsf'), e('lognH'), hi=-3):.1f}</super> cm<super>-3</super> and denser "
            f"gas just below the threshold near 10<super>{wmode(h('lognH', 'nonsf'), e('lognH'), lo=-3):.1f}</super> "
            f"cm<super>-3</super>. The cube therefore contains both the galaxy's interstellar medium and a large "
            f"amount of circumgalactic gas."),
        "logB_uG": (
            "Magnetic field strength",
            f"Mass-weighted median field strengths are "
            f"{10 ** wquantile(h('logB_uG', 'nonsf'), e('logB_uG'), 0.5):.1f} µG for non-star-forming gas and "
            f"{10 ** wquantile(h('logB_uG', 'sf'), e('logB_uG'), 0.5):.1f} µG for star-forming gas, with a tail of "
            f"weakly magnetised diffuse gas down to about 10<super>-2</super> µG. The field spans roughly four "
            f"decades, so |B| should enter the model on a log scale."),
        "xHI": (
            "Neutral hydrogen fraction",
            f"Non-star-forming gas is close to all-or-nothing: {wfrac(both('xHI'), e('xHI'), hi=0.1):.0%} of all "
            f"gas mass is almost fully ionised (x<sub>HI</sub> &lt; 0.1) and {wfrac(both('xHI'), e('xHI'), lo=0.9):.0%} "
            f"almost fully neutral (&gt; 0.9). Star-forming gas is never fully neutral: its x<sub>HI</sub> spreads "
            f"between 0 and about 0.7 (mass-weighted median {wquantile(h('xHI', 'sf'), e('xHI'), 0.5):.2f}; "
            f"{wfrac(h('xHI', 'sf'), e('xHI'), hi=0.1):.0%} of it below 0.1), set by the sub-grid model rather "
            f"than resolved physics. The HI modality inherits this approximation, as do the 2D HI maps."),
        "logZ": (
            "Gas metallicity",
            f"Star-forming gas is the most enriched (mass-weighted median "
            f"{10 ** wquantile(h('logZ', 'sf'), e('logZ'), 0.5):.2f} Z☉) and the surrounding gas less so "
            f"(median {10 ** wquantile(h('logZ', 'nonsf'), e('logZ'), 0.5):.2f} Z☉), with a tail of nearly pristine "
            f"gas. No cell in the sample has zero metallicity, so a log transform is safe."),
        "logr_kpc": (
            "Gas cell size",
            f"Radius of a sphere with the cell's volume. Star-forming cells have a median radius of "
            f"{r_sf_pc:.0f} pc, while non-star-forming cells range from about {r_lo:.2f} to {r_hi:.1f} kpc (5–95%, "
            f"mass-weighted). Resolution is adaptive: dense gas is sampled much more finely than halo gas, so point "
            f"density reflects both mass and resolution. Sampling in proportion to mass handles this; picking cells "
            f"uniformly would over-represent the dense centre."),
        "cell_mass": (
            "Gas cell mass relative to the target mass",
            f"Cell mass divided by the simulation's target gas mass (8.5×10<super>4</super> M☉). TNG's refinement "
            f"keeps cells within a factor of two of the target, and {in_range:.2%} of the {int(c.sum()):,} cells "
            f"lie between 0.5× and 2×. Gas cells are close to equal-mass, so sampling by mass and by cell are "
            f"nearly equivalent; no filter is needed."),
    }


def texts_stars(d):
    ea, ha = d["bins_age_Gyr"], d["hist_age_Gyr"]
    ez, hz = d["bins_star_logZ"], d["hist_star_logZ"]
    em, hm = d["bins_m_over_init"], d["hist_m_over_init"]
    s = {k: d[f"sample_{k}"] for k in ("wind_mass_frac", "sf_mass_frac", "hi_mass_frac", "cold_mass_frac")}
    return {
        "age": (
            "Stellar age",
            f"Mass-weighted age of the star particles in the cubes. All ages are present: the median is "
            f"{wquantile(ha, ea, 0.5):.1f} Gyr, {wfrac(ha, ea, hi=1.0):.0%} of the stellar mass is younger than "
            f"1 Gyr and {wfrac(ha, ea, lo=10.0):.0%} older than 10 Gyr. Age is a natural per-point feature for the "
            f"star modality."),
        "star_Z": (
            "Stellar metallicity",
            f"Mass-weighted stellar metallicity. It peaks near solar (median {10 ** wquantile(hz, ez, 0.5):.2f} Z☉) "
            f"with a tail of metal-poor, old stars. The small spike at the floor ({wfrac(hz, ez, hi=-3.95):.2%} of the "
            f"stellar mass) are stars with essentially no metals, formed early from nearly pristine gas."),
        "mass_loss": (
            "Current versus birth mass of star particles",
            f"Current mass of each star particle relative to its mass at birth. Most particles have lost about 40% "
            f"(peak at {wmode(hm, em):.2f}) through stellar winds and supernovae; young particles have lost less. "
            f"Sampling stars by current mass therefore gives old stars slightly less weight than sampling by birth "
            f"mass would."),
        "wind": (
            "Wind cells among PartType4",
            f"Mass in wind-phase cells (stored with the stars but not stars) relative to all PartType4 mass in each "
            f"halo's cube. The median is {np.median(s['wind_mass_frac']):.2%} and the largest "
            f"{s['wind_mass_frac'].max():.1%}, so dropping wind cells barely changes the star clouds while keeping "
            f"them physically clean. The 2D star maps include them; the difference is negligible."),
        "sf_frac": (
            "Star-forming gas per halo",
            f"Fraction of each cube's gas mass that is star-forming. The median is {np.median(s['sf_mass_frac']):.0%} "
            f"and 90% of halos lie between {q(s['sf_mass_frac'], 5):.0%} and {q(s['sf_mass_frac'], 95):.0%}. These "
            f"are the points whose temperature and neutral fraction come from the sub-grid model; a per-point "
            f"star-forming flag would let the model tell them apart."),
        "hi_frac": (
            "Neutral hydrogen per halo",
            f"Neutral hydrogen mass as a fraction of each cube's gas mass. The median is "
            f"{np.median(s['hi_mass_frac']):.0%}, with 90% between {q(s['hi_mass_frac'], 5):.0%} and "
            f"{q(s['hi_mass_frac'], 95):.0%}. Halos at the low end will give sparse clouds when points are drawn in "
            f"proportion to HI mass."),
        "cold_frac": (
            "Cold gas per halo",
            f"Mass fraction of gas below 10<super>4.5</super> K in each cube (star-forming gas counts as cold "
            f"here). The median is {np.median(s['cold_mass_frac']):.0%}, ranging from {q(s['cold_mass_frac'], 5):.0%} "
            f"to {q(s['cold_mass_frac'], 95):.0%}: most cubes are dominated by gas at about 10<super>4</super> K, "
            f"while systems dominated by a hot halo sit at the low end."),
    }


# ---------------------------------------------------------------- layout
def section(story, title, intro, panels, texts, images, prefix):
    story.append(Paragraph(title, STYLES["h1"]))
    story.append(Paragraph(intro, STYLES["body"]))
    for i, (key, _) in enumerate(panels, 1):
        heading, text = texts[key]
        img = images[key]
        w, h = mpimg.imread(img).shape[1::-1]
        story.append(KeepTogether([
            Paragraph(f"{prefix}{i}. {heading}", STYLES["h2"]),
            Image(str(img), width=IMG_W, height=IMG_W * h / w),
            Paragraph(text, STYLES["body"]),
            Spacer(1, 4),
        ]))
    story.append(PageBreak())


def sf_share(d):
    sf = d["hist_logT_sf"].sum()
    return sf / (sf + d["hist_logT_nonsf"].sum())


def summary(story, d, da):
    hw = d["avail_half_width_kpc"]
    pts = d["halo_npts_star"]
    share = stellar_share(d)
    flag0, edge, few = d["halo_flag_cen"] == 0, d["avail_crosses_box_edge"] == 1, pts < 2048
    base = flag0 | edge | few
    comp = d["avail_complete"] == 1
    sfr, ms = d["halo_sfr_cen_rad"], d["halo_mstar_cen_rad"]
    ssfr = np.where(ms > 0, sfr / np.where(ms > 0, ms, 1), np.nan)
    cell = ParagraphStyle("cell", parent=STYLES["body"], alignment=0, spaceAfter=0, fontSize=8.5, leading=11)
    rows = [
        ["Candidate filter", "Halos affected (of 2,991)", "Suggestion"],
        ["Central has SubhaloFlag = 0", f"{flag0.sum()}", "Drop"],
        ["Cube crosses the box edge (2D map truncated)", f"{edge.sum()}", "Drop, at least for 2D–3D pairing"],
        ["Fewer than 2,048 star particles in the cube", f"{few.sum()}", "Drop; use about 4,096 star points"],
        ["Central holds &lt; 50% of the halo's stars", f"{(share < 0.5).sum()}", "Optional (likely mergers)"],
        ["SFR = 0 / quenched centrals", f"{(sfr <= 0).sum()} / {((ssfr < 1e-11) | (sfr <= 0)).sum()}", "Keep"],
        ["HI in the cube &lt; 10<super>8</super> M☉", f"{(d['halo_hi_mass'] < 1e8).sum()}", "Keep"],
        ["Cube half-width &gt; 200 kpc (groups, clusters)", f"{(hw > 200).sum()}", "Keep; scale by half-width"],
    ]
    rows = [[Paragraph(c if i else f"<b>{c}</b>", cell) for c in r] for i, r in enumerate(rows)]
    t = Table(rows, colWidths=[78 * mm, 40 * mm, 56 * mm])
    t.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, -1), "DejaVu", 8.5),
        ("FONT", (0, 0), (-1, 0), "DejaVu-Bold", 8.5),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.HexColor("#c3c2b7")),
        ("LINEBELOW", (0, 1), (-1, -1), 0.3, colors.HexColor("#e1e0d9")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(Paragraph("E. Summary: candidate filters", STYLES["h1"]))
    story.append(Paragraph(
        "Halo-level candidates for the 2,991 halos with 2D maps, from sections A and B. The first three rows are "
        "the recommended set.", STYLES["body"]))
    story.append(t)
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        f"The recommended set removes {base.sum()} halos and keeps {(~base).sum():,}, of which "
        f"{(comp & ~base).sum()} are complete on disk today. Adding the merger filter keeps "
        f"{(~(base | (share < 0.5))).sum():,} ({(comp & ~(base | (share < 0.5))).sum()} complete). At particle level "
        f"(sections C and D): drop wind cells; keep star-forming gas but add a per-point star-forming flag; no "
        f"filter is needed on cell mass or metallicity. Star clouds probably need their own spatial scaling "
        f"because stars occupy only the central few per cent of the cube.", STYLES["body"]))
    extra = (da["avail_in_2d"] == 0) & (da["halo_ncen_star"] >= 2048)
    gb = (da["avail_n_stars"][extra].sum() * 60 + da["avail_n_dm"][extra].sum() * 44) / 1e9
    story.append(Paragraph(
        f"Outside the 2D set, {extra.sum():,} halos have at least 2,048 star particles in their central galaxy. "
        f"They have no 2D maps, so they could only be used for 3D-only training, and none is complete on disk; "
        f"filling them in with API cutouts of stars and dark matter would take about {gb:.1f} GB.", STYLES["body"]))


def main():
    p2d, pall, out = (Path(x) for x in sys.argv[1:4])
    d2, da = load(p2d), load(pall)
    img_dir = out.parent / "report_panels"
    panels = {"A": halo_panels(d2), "B": all_halo_panels(da), "C": gas_panels(d2), "D": star_panels(d2)}
    images = {s: render_each(p, img_dir, s) for s, p in panels.items()}

    doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm,
                            title="GalaxyCLIP-3D: TNG50-1 snapshot 99 data distributions", author="GalaxyCLIP-3D")
    story = [
        Paragraph("TNG50-1 snapshot 99: data distributions for choosing filters", STYLES["title"]),
        Paragraph(f"GalaxyCLIP-3D · {datetime.date.today():%Y-%m-%d} · generated by tools/build_report.py",
                  STYLES["subtitle"]),
        Paragraph(
            "This document shows every histogram used to decide how to filter the 3D point-cloud dataset, each with "
            "a short explanation. The data are TNG50-1 at redshift zero. The local copy of the snapshot is "
            f"incomplete (108 of its 680 files were never downloaded and one is truncated), so only "
            f"{(d2['avail_complete'] == 1).sum()} of the {len(d2['avail_gid']):,} halos with 2D maps have all their "
            f"particles on disk.", STYLES["body"]),
        Paragraph(
            "<b>The cube.</b> Each 2D map shows the halo's friends-of-friends (FoF) particles inside a cube centred "
            "on the central galaxy, with half-width equal to the central's gas half-mass radius. The 3D point clouds "
            "are meant to cover the same region, so most panels refer to this cube.", STYLES["body"]),
        Paragraph(
            "<b>Populations.</b> Section A covers the 2,991 halos with 2D maps; points and HI in the cube come from "
            "the 2D map metadata and are exact for every halo. Section B compares them with all 36,057 halos whose "
            "central galaxy contains gas and stars; there, particle counts come from the group catalog because "
            "there are no maps. Sections C and D use the particles of 60 halos that are complete on disk, chosen "
            "evenly across their mass range; every complete halo has 2D maps, so these sections describe the "
            f"massive end only. A second, independently drawn sample gave the same picture (star-forming share of "
            f"the gas mass {sf_share(da):.1%} versus {sf_share(d2):.1%}).", STYLES["body"]),
        Paragraph(
            "<b>Reading the graphs.</b> Blue is the main population and orange the comparison (2D set, incomplete "
            "halos or star-forming gas, as labelled). Section B uses logarithmic counts so both populations are "
            "visible. Gas histograms in section C are weighted by mass, so they show where the gas mass is, which "
            "is what mass-weighted point sampling would draw from.", STYLES["body"]),
        Spacer(1, 6),
    ]
    section(story, "A. Halos with 2D maps (2,991)",
            "Halo-level properties of the 2,991 halos at snapshot 99 that have 2D maps in GalaxyCLIP-2D.",
            panels["A"], texts_2d(d2), images["A"], "A")
    section(story, "B. All halos with a galaxy (36,057) compared with the 2D set",
            "The same properties for every halo whose central galaxy contains gas and at least one star particle "
            "(blue), with the 2D set overlaid (orange). Counts are on a log scale.",
            panels["B"], texts_all(da, d2), images["B"], "B")
    section(story, "C. Gas cells inside the cube (60 complete halos)",
            "Mass-weighted distributions of gas-cell properties, summed over the cubes of 60 halos that are "
            "complete on disk, split into star-forming and non-star-forming gas.",
            panels["C"], texts_gas(d2), images["C"], "C")
    section(story, "D. Star particles and per-halo fractions (60 complete halos)",
            "Star-particle distributions for the same 60 halos, and per-halo fractions that describe how each "
            "cube's gas and PartType4 content is composed.",
            panels["D"], texts_stars(d2), images["D"], "D")
    summary(story, d2, da)

    def footer(canvas, doc_):
        canvas.saveState()
        canvas.setFont("DejaVu", 7.5)
        canvas.setFillColor(INK2)
        canvas.drawRightString(A4[0] - 18 * mm, 9 * mm, f"{doc_.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
