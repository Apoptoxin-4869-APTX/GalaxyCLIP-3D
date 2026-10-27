"""Collect the quantities used to choose dataset filters at snapshot 99.

Two populations (--population):
  2d   the 2,991 halos with 2D maps. Halo level from the group catalog + 2D map metadata (so
       independent of the missing snapshot files): points per modality inside the 2D cube
       (num_particles of the unrotated gas/dm/star maps) and the HI mass inside the cube (sum
       of the unrotated 2D hi map), plus the catalog quantities below.
  all  every FoF halo whose central galaxy has gas and at least one star particle (~36k).
       There are no 2D maps for most of them, so particle counts come from the catalog (FoF
       group and central galaxy) instead of the cube, and there is no HI mass.

Catalog quantities (both): M200c, R200c, central stellar mass and SFR within 2 R_1/2,*,
stellar half-mass radius, subhalos per halo, the central's share of the halo's stellar mass,
completeness on disk and the box-edge flag of the 2D cube rule.

Particle level (both): a mass-stratified sample of halos that are complete on disk, do not
cross the box edge and have <= --max-cells FoF gas cells and star particles. Mass-weighted
histograms of gas temperature, hydrogen density, |B|, neutral fraction, metallicity and cell
size (split into star-forming and non-star-forming gas), the cell-mass distribution, star-
particle age, metallicity and current/initial mass, plus per-halo summaries (wind-cell and
star-forming fractions). Cube = the 2D rule (central SubhaloPos, half-width = its gas
half-mass radius). For `all`, strata are equal-width in log M200c so low masses are covered.

    python tools/collect_histogram_data.py --population 2d  --out outputs/data_audit/histogram_data.npz
    python tools/collect_histogram_data.py --population all --out outputs/data_audit/histogram_data_all.npz

A few minutes each, < 4 GB of RAM.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_snapshot import BLOCKS_2D, load_groupcat, missing_counter, snapshot_files   # noqa: E402
from check_2d_selection import ROOT, ROOT_2D, Snapshot                                    # noqa: E402

AVAIL = Path(__file__).resolve().parents[1] / "data" / "snap099_halo_availability.csv"
M_P, K_B, MSUN, KPC = 1.67262192e-24, 1.380649e-16, 1.98847e33, 3.0856776e21   # cgs
X_H, Z_SUN, B_UNIT = 0.76, 0.0127, 2.60e-6         # H mass fraction, solar Z, code B -> Gauss
BINS = {                                            # shared bin edges, stored with the output
    "logT": np.arange(1.0, 9.0001, 0.05),
    "lognH": np.arange(-7.0, 4.0001, 0.05),
    "logB_uG": np.arange(-7.0, 3.0001, 0.05),
    "xHI": np.linspace(0.0, 1.0, 101),
    "logZ": np.arange(-4.0, 1.0001, 0.05),
    "logr_kpc": np.arange(-2.5, 1.5001, 0.025),
    "m_over_target": np.arange(0.0, 3.0001, 0.02),
    "age_Gyr": np.arange(0.0, 14.0001, 0.1),
    "star_logZ": np.arange(-4.0, 1.0001, 0.05),
    "m_over_init": np.arange(0.3, 1.0501, 0.005),
}
GROUP_FIELDS = ["Group_M_Crit200", "Group_R_Crit200", "GroupNsubs", "GroupMassType", "GroupFirstSub",
                "GroupLenType"]
SUBHALO_FIELDS = ["SubhaloMassInRadType", "SubhaloMassType", "SubhaloSFRinRad", "SubhaloHalfmassRadType",
                  "SubhaloFlag", "SubhaloPos", "SubhaloLenType"]


def maps_2d(domain: str, snap: int = 99):
    """2D files of one domain at one snapshot (excluding hi-21cm when domain='hi')."""
    pat = re.compile(rf"_tng50-1\.{snap}\.2D\.{re.escape(domain)}\.(\d{{4}})\.(\d{{4}})\.hdf5$")
    return sorted(p for p in Path(ROOT_2D).iterdir() if pat.search(p.name))


def unrotated_entries(f: h5py.File, domain: str):
    """Image index, gid, attrs of every unrotated map (views are stored in blocks of 4)."""
    md = f[f"{domain}/metadata"]
    for i in range(0, len(md), 4):
        at = md[f"{i:04d}"].attrs
        assert tuple(at["rotxy"]) == (0, 0), (f.filename, i)
        yield i, int(at["gid"]), at


def map_quantities(gids: np.ndarray) -> dict[str, np.ndarray]:
    """Points per modality and HI mass inside the cube, from the unrotated 2D maps."""
    pos = {g: k for k, g in enumerate(gids)}
    out = {k: np.full(len(gids), np.nan) for k in ("npts_gas", "npts_dm", "npts_star", "hi_mass")}
    for domain in ("gas", "dm", "star"):
        for p in maps_2d(domain):
            with h5py.File(p, "r") as f:
                for _, g, at in unrotated_entries(f, domain):
                    if g in pos and np.isnan(out[f"npts_{domain}"][pos[g]]):
                        out[f"npts_{domain}"][pos[g]] = int(at["num_particles"])
        print(f"  2D {domain} counts read", flush=True)
    for p in maps_2d("hi"):
        with h5py.File(p, "r") as f:
            entries = list(unrotated_entries(f, "hi"))
            img = f["hi/images"]
            for k in range(0, len(entries), 25):
                chunk = entries[k:k + 25]
                sums = img[chunk[0][0]:chunk[-1][0] + 1:4].sum(axis=(1, 2))      # Msun/kpc^2 summed
                for (_, g, at), s in zip(chunk, sums):
                    pix = 2 * float(at["extent"][1]) / img.shape[-1]              # kpc per pixel
                    if g in pos and np.isnan(out["hi_mass"][pos[g]]):
                        out["hi_mass"][pos[g]] = s * pix ** 2
        print(f"  2D hi masses read from {p.name}", flush=True)
    return out


def catalog_quantities(cat, gids, cen, h, a) -> dict[str, np.ndarray]:
    msun = 1e10 / h
    return dict(
        m200c=cat["Group_M_Crit200"][gids] * msun,
        r200c_kpc=cat["Group_R_Crit200"][gids] * a / h,
        nsubs=cat["GroupNsubs"][gids].astype(float),
        mstar_fof=cat["GroupMassType"][gids, 4] * msun,
        mstar_cen_rad=cat["SubhaloMassInRadType"][cen, 4] * msun,
        mstar_cen=cat["SubhaloMassType"][cen, 4] * msun,
        mgas_cen_rad=cat["SubhaloMassInRadType"][cen, 0] * msun,
        sfr_cen_rad=cat["SubhaloSFRinRad"][cen].astype(float),
        rhalf_star_kpc=cat["SubhaloHalfmassRadType"][cen, 4] * a / h,
        flag_cen=cat["SubhaloFlag"][cen].astype(float),
        ncen_gas=cat["SubhaloLenType"][cen, 0].astype(float),
        ncen_dm=cat["SubhaloLenType"][cen, 1].astype(float),
        ncen_star=cat["SubhaloLenType"][cen, 4].astype(float),
    )


def all_halos(cat, hd) -> dict[str, np.ndarray]:
    """Availability table (same columns as data/snap099_halo_availability.csv) for every FoF halo
    whose central galaxy has gas and at least one star particle."""
    h, a, box = float(hd["HubbleParam"]), float(hd["Time"]), float(hd["BoxSize"])
    fs = cat["GroupFirstSub"]
    has = fs >= 0
    sel = np.zeros(len(fs), dtype=bool)
    c = fs[has]
    sel[has] = (cat["SubhaloHalfmassRadType"][c, 0] > 0) & (cat["SubhaloLenType"][c, 4] > 0)
    gids = np.flatnonzero(sel)
    cen = fs[gids]
    _, unusable, _, _ = snapshot_files(ROOT, 99)
    ntot = hd["NumPart_Total"].astype(np.int64) + (hd["NumPart_Total_HighWord"].astype(np.int64) << 32)
    with h5py.File(Path(ROOT) / "offsets" / "offsets_099.hdf5", "r") as f:
        fo = f["FileOffsets/SnapByType"][()]
        goff = f["Group/SnapByType"][()][gids].astype(np.int64)
    n_missing, _ = missing_counter(fo if fo.shape[1] == 6 else fo.T, ntot, unusable)
    glen = cat["GroupLenType"][gids].astype(np.int64)
    miss = np.stack([n_missing(t, goff[:, t], glen[:, t]) for t in (0, 1, 4, 5)], 1)
    hw = cat["SubhaloHalfmassRadType"][cen, 0]
    pos = cat["SubhaloPos"][cen]
    crosses = ((pos - hw[:, None] < 0) | (pos + hw[:, None] > box)).any(1)
    in_2d = np.isin(gids, [g for s, g in json.load(open(BLOCKS_2D))["blocks"] if s == 99])
    return dict(gid=gids, central_subhalo=cen, half_width_kpc=hw * a / h,
                complete=(miss == 0).all(1).astype(int), crosses_box_edge=crosses.astype(int),
                n_gas=glen[:, 0], n_dm=glen[:, 1], n_stars=glen[:, 4], n_bh=glen[:, 5], in_2d=in_2d.astype(int))


def cosmic_time(a, h, om, ol):
    """Age of a flat LCDM universe at scale factor a, in Gyr."""
    return 2.0 / (3.0 * np.sqrt(ol)) * (9.7779 / h) * np.arcsinh(np.sqrt(ol / om) * np.power(a, 1.5))


def load_fields(s: Snapshot, ptype: int, start: int, n: int, fields: list[str]) -> dict:
    """Several fields of rows [start, start+n) of one particle type, across file chunks."""
    j = int(np.searchsorted(s.file_offsets[:, ptype], start, side="right") - 1)
    off, out = start - int(s.file_offsets[j, ptype]), {k: [] for k in fields}
    while n > 0:
        with h5py.File(s.snap_file(j), "r") as f:
            k = min(n, int(f["Header"].attrs["NumPart_ThisFile"][ptype]) - off)
            if k > 0:
                for fld in fields:
                    out[fld].append(f[f"PartType{ptype}/{fld}"][off:off + k])
                n -= k
        j, off = j + 1, 0
    return {k: np.concatenate(v) for k, v in out.items()}


def pick_sample(avail, cat, h, n_per_bin, n_bins, max_cells, seed, equal_width, min_mstar):
    """Mass-stratified sample of halos that are complete on disk and fully inside the box."""
    ok = ((avail["complete"] == 1) & (avail["crosses_box_edge"] == 0)
          & (avail["n_gas"] <= max_cells) & (avail["n_stars"] <= max_cells)
          & (cat["SubhaloMassInRadType"][avail["central_subhalo"], 4] * 1e10 / h >= min_mstar))
    cand = np.flatnonzero(ok)
    logm = np.log10(cat["Group_M_Crit200"][avail["gid"][cand]] * 1e10 / h)
    edges = (np.linspace(logm.min(), logm.max(), n_bins + 1) if equal_width
             else np.quantile(logm, np.linspace(0, 1, n_bins + 1)))
    rng = np.random.default_rng(seed)
    pick = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        inb = cand[(logm >= lo) & (logm <= hi)]
        pick += list(rng.choice(inb, size=min(n_per_bin, inb.size), replace=False))
    return np.array(sorted(set(pick)))


def particle_level(s, avail, cat, hd, pick):
    h, a, box = float(hd["HubbleParam"]), float(hd["Time"]), float(hd["BoxSize"])
    om, ol = float(hd["Omega0"]), float(hd["OmegaLambda"])
    m_target = float(hd["MassTable"][3]) * 1e10 / h          # tracer mass = target gas-cell mass
    t_now = cosmic_time(a, h, om, ol)

    hist = {f"{k}_{ph}": np.zeros(len(BINS[k]) - 1) for k in
            ("logT", "lognH", "logB_uG", "xHI", "logZ", "logr_kpc") for ph in ("sf", "nonsf")}
    hist.update(m_over_target=np.zeros(len(BINS["m_over_target"]) - 1),
                age_Gyr=np.zeros(len(BINS["age_Gyr"]) - 1),
                star_logZ=np.zeros(len(BINS["star_logZ"]) - 1),
                m_over_init=np.zeros(len(BINS["m_over_init"]) - 1))
    per_halo = {k: [] for k in ("gid", "n_gas_cube", "n_star_cube", "n_wind_cube", "sf_mass_frac",
                                "wind_mass_frac", "hi_mass_frac", "cold_mass_frac", "zero_Z_gas")}
    with h5py.File(s.offsets_path, "r") as f:
        gstart = {int(avail["gid"][i]): f["Group/SnapByType"][int(avail["gid"][i])] for i in pick}
    for n_done, i in enumerate(pick, 1):
        g, cen = int(avail["gid"][i]), int(avail["central_subhalo"][i])
        c = cat["SubhaloPos"][cen].astype(np.float64)
        hw = float(cat["SubhaloHalfmassRadType"][cen, 0])

        def in_cube(x):
            d = x - c
            d -= box * np.round(d / box)
            return (np.abs(d) < hw).all(1)

        gas = load_fields(s, 0, int(gstart[g][0]), int(avail["n_gas"][i]),
                          ["Coordinates", "Masses", "Density", "InternalEnergy", "ElectronAbundance",
                           "MagneticField", "NeutralHydrogenAbundance", "StarFormationRate",
                           "GFM_Metallicity"])
        sel = in_cube(gas.pop("Coordinates"))
        gas = {k: v[sel] for k, v in gas.items()}
        m = gas["Masses"].astype(np.float64) * 1e10 / h                         # Msun
        rho = gas["Density"].astype(np.float64) * 1e10 * h ** 2 / a ** 3         # Msun / kpc^3
        mu = 4.0 / (1.0 + 3.0 * X_H + 4.0 * X_H * gas["ElectronAbundance"])
        temp = (5.0 / 3.0 - 1.0) * gas["InternalEnergy"] * 1e10 * mu * M_P / K_B
        q = {"logT": np.log10(temp),
             "lognH": np.log10(X_H * rho * MSUN / KPC ** 3 / M_P),
             "logB_uG": np.log10(np.linalg.norm(gas["MagneticField"], axis=1) * h / a ** 2 * B_UNIT * 1e6 + 1e-30),
             "xHI": gas["NeutralHydrogenAbundance"],
             "logZ": np.log10(np.maximum(gas["GFM_Metallicity"] / Z_SUN, 1e-4)),
             "logr_kpc": np.log10((3.0 * m / (4.0 * np.pi * rho)) ** (1.0 / 3.0))}
        sf = gas["StarFormationRate"] > 0
        for k, v in q.items():
            for ph, msk in (("sf", sf), ("nonsf", ~sf)):
                hist[f"{k}_{ph}"] += np.histogram(np.clip(v[msk], BINS[k][0], BINS[k][-1] - 1e-9),
                                                  BINS[k], weights=m[msk])[0]
        hist["m_over_target"] += np.histogram(np.clip(m / m_target, 0, BINS["m_over_target"][-1] - 1e-9),
                                              BINS["m_over_target"])[0]

        st = load_fields(s, 4, int(gstart[g][4]), int(avail["n_stars"][i]),
                         ["Coordinates", "Masses", "GFM_InitialMass", "GFM_Metallicity",
                          "GFM_StellarFormationTime"])
        sel = in_cube(st.pop("Coordinates"))
        st = {k: v[sel] for k, v in st.items()}
        wind = st["GFM_StellarFormationTime"] <= 0
        real = ~wind
        ms = st["Masses"].astype(np.float64) * 1e10 / h
        age = t_now - cosmic_time(st["GFM_StellarFormationTime"][real].astype(np.float64), h, om, ol)
        hist["age_Gyr"] += np.histogram(np.clip(age, 0, BINS["age_Gyr"][-1] - 1e-9),
                                        BINS["age_Gyr"], weights=ms[real])[0]
        zs = np.log10(np.maximum(st["GFM_Metallicity"][real] / Z_SUN, 1e-4))
        hist["star_logZ"] += np.histogram(np.clip(zs, BINS["star_logZ"][0], BINS["star_logZ"][-1] - 1e-9),
                                          BINS["star_logZ"], weights=ms[real])[0]
        ratio = st["Masses"][real] / st["GFM_InitialMass"][real]
        hist["m_over_init"] += np.histogram(np.clip(ratio, BINS["m_over_init"][0], BINS["m_over_init"][-1] - 1e-9),
                                            BINS["m_over_init"])[0]

        per_halo["gid"].append(g)
        per_halo["n_gas_cube"].append(int(m.size))
        per_halo["n_star_cube"].append(int(real.sum() + wind.sum()))
        per_halo["n_wind_cube"].append(int(wind.sum()))
        per_halo["sf_mass_frac"].append(m[sf].sum() / max(m.sum(), 1e-30))
        per_halo["wind_mass_frac"].append(ms[wind].sum() / max(ms.sum(), 1e-30))
        per_halo["hi_mass_frac"].append((X_H * gas["NeutralHydrogenAbundance"] * m).sum() / max(m.sum(), 1e-30))
        per_halo["cold_mass_frac"].append(m[temp < 10 ** 4.5].sum() / max(m.sum(), 1e-30))
        per_halo["zero_Z_gas"].append(int((gas["GFM_Metallicity"] <= 0).sum()))
        print(f"  [{n_done:3d}/{len(pick)}] gid {g}: {m.size:,} gas cells, {real.size:,} star particles "
              f"({wind.sum():,} wind) in the cube", flush=True)
    return hist, {k: np.array(v) for k, v in per_halo.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--population", choices=["2d", "all"], default="2d")
    ap.add_argument("--out", type=Path, default=Path("outputs/data_audit/histogram_data.npz"))
    ap.add_argument("--per-bin", type=int, default=10, help="halos per M200c bin in the particle sample")
    ap.add_argument("--bins", type=int, default=6, help="M200c bins in the particle sample")
    ap.add_argument("--max-cells", type=int, default=5_000_000, help="skip halos with more FoF gas/stars")
    ap.add_argument("--min-mstar", type=float, default=1e7, help="central M* (2 R1/2) floor for the sample")
    ap.add_argument("--seed", type=int, default=4869)
    args = ap.parse_args()

    s = Snapshot(ROOT, 99)
    with h5py.File(s.snap_file(0), "r") as f:
        hd = dict(f["Header"].attrs)
    h, a = float(hd["HubbleParam"]), float(hd["Time"])
    print("loading group catalog", flush=True)
    cat = load_groupcat(ROOT, 99, {"Group": GROUP_FIELDS, "Subhalo": SUBHALO_FIELDS})

    if args.population == "2d":
        avail = np.genfromtxt(AVAIL, delimiter=",", names=True, dtype=None, encoding=None)
        avail = {k: np.asarray(avail[k]) for k in avail.dtype.names}
        print("halo level (2D maps)", flush=True)
        halo = map_quantities(avail["gid"])
    else:
        print("halo level (all halos with a galaxy)", flush=True)
        avail = all_halos(cat, hd)
        halo = {}
    halo.update(catalog_quantities(cat, avail["gid"], avail["central_subhalo"], h, a))

    print("particle level", flush=True)
    pick = pick_sample(avail, cat, h, args.per_bin, args.bins, args.max_cells, args.seed,
                       equal_width=args.population == "all", min_mstar=args.min_mstar)
    hist, per_halo = particle_level(s, avail, cat, hd, pick)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out, population=args.population,
        **{f"avail_{k}": v for k, v in avail.items()}, **{f"halo_{k}": v for k, v in halo.items()},
        **{f"hist_{k}": v for k, v in hist.items()}, **{f"bins_{k}": v for k, v in BINS.items()},
        **{f"sample_{k}": v for k, v in per_halo.items()})
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
