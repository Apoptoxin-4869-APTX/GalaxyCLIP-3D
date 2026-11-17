"""Physical properties of every galaxy of one snapshot, to test what the learned embeddings know (linear probes).

    PYTHONPATH=pylibs python3 -s tools/build_targets.py --out /cluster/projects/ska/illustris/tng50-1.pointclouds/v3 \
        --snap 99 [--root RAW_COPY] [--workers 8]

Writes OUT/snap_NNN/targets.h5: one row per galaxy, in the order of snap_NNN/index.h5 (subhalo order), with
  from the group catalog (needs groupcats/NNN of the raw copy; particles are not needed):
    log_m200c, r200c_kpc        the host halo's M200c [Msun] and R200c [kpc]; satellites get their host's
    log_mstar_total, log_mgas_total, log_mdm_total, log_mbh   bound masses [Msun] (log10, floored at 1)
    gas_fraction                M_gas / (M_gas + M_stars), bound
    log_sfr, log_ssfr           star-formation rate [Msun/yr] and per stellar mass [1/yr], floored at 1e-5 / 1e-15
    log_z_stars, log_z_gas      mass-weighted metallicities [log10 Z/Zsun], NaN without gas
    veldisp, spin               velocity dispersion [km/s] and |specific angular momentum| [kpc km/s]
    g_minus_r                   colour from SubhaloStellarPhotometrics (magnitudes g - r)
  from the stored particles, inside the training sphere (train_radius stellar half-mass radii):
    age_mean_gyr                mass-weighted mean stellar age
    log_mhi_sphere, log_mgas_sphere   HI and gas mass in the sphere [Msun], from mass x wcorr
  copied from the index: central, rhalf_star_kpc, vmax, mstar.
"""
from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from galaxyclip3d.pointclouds.config import Config                               # noqa: E402
from galaxyclip3d.pointclouds.layout import read_table, shard_path, snapshot_dir, write_table  # noqa: E402
from galaxyclip3d.pointclouds.snapshot import read_catalog                       # noqa: E402
from galaxyclip3d.pointclouds.store import Shard                                 # noqa: E402

Z_SUN = 0.0127
G_BAND, R_BAND = 4, 5           # SubhaloStellarPhotometrics: U B V K g r i z


def log10_floor(x, floor):
    return np.log10(np.maximum(np.asarray(x, dtype=np.float64), floor))


def sphere_values(item):
    """(index positions, [(age_mean, m_hi, m_gas)]) inside the sphere for one shard's galaxies."""
    path, rows, radii, positions = item
    sh = Shard(path)
    out = []
    for row, rad in zip(rows, radii):
        st = sh.particles(int(row), "stars", ["pos", "mass", "age", "wcorr"])
        w = st["mass"].astype(np.float64) * st["wcorr"]
        inside = np.linalg.norm(st["pos"], axis=1) <= rad
        age = float((w[inside] * st["age"][inside]).sum() / w[inside].sum()) if inside.any() else np.nan
        gas = sh.particles(int(row), "gas", ["pos", "mass", "m_hi", "wcorr"])
        g_in = np.linalg.norm(gas["pos"], axis=1) <= rad
        out.append((age, float((gas["m_hi"][g_in] * gas["wcorr"][g_in]).sum()),
                    float((gas["mass"][g_in] * gas["wcorr"][g_in]).sum())))
    sh.close()
    return positions, out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True, help="dataset folder")
    ap.add_argument("--snap", type=int, required=True)
    ap.add_argument("--root", help="raw snapshot copy (default: the snapshot's build root)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    sdir = snapshot_dir(args.out, args.snap)
    cfg = Config.load(sdir / "config.json")
    root = args.root or cfg.root
    index = read_table(sdir / "index.h5")
    sub, grp = index["subhalo"].astype(np.int64), index["group"].astype(np.int64)

    gpath = sorted((Path(root) / "groupcats" / f"{args.snap:03d}").glob("fof_subhalo_tab_*.0.hdf5"))[0]
    with h5py.File(gpath, "r") as f:
        h, a = float(f["Header"].attrs["HubbleParam"]), float(f["Header"].attrs["Time"])
    cat = read_catalog(root, args.snap, ["Group_M_Crit200", "Group_R_Crit200"],
                       ["SubhaloMassType", "SubhaloSFR", "SubhaloStarMetallicity", "SubhaloGasMetallicity",
                        "SubhaloVelDisp", "SubhaloSpin", "SubhaloStellarPhotometrics", "SubhaloBHMass"])
    msun = 1e10 / h
    mt = cat["SubhaloMassType"][sub].astype(np.float64) * msun
    mstar_t, mgas_t = mt[:, 4], mt[:, 0]
    sfr = cat["SubhaloSFR"][sub].astype(np.float64)
    zgas = cat["SubhaloGasMetallicity"][sub].astype(np.float64)
    phot = cat["SubhaloStellarPhotometrics"][sub].astype(np.float64)
    t = {
        "subhalo": sub, "group": grp, "central": index["central"], "rhalf_star_kpc": index["rhalf_star_kpc"],
        "vmax": index["vmax"], "mstar": index["mstar"],
        "log_m200c": log10_floor(cat["Group_M_Crit200"][grp] * msun, 1.0),
        "r200c_kpc": cat["Group_R_Crit200"][grp].astype(np.float64) * a / h,
        "log_mstar_total": log10_floor(mstar_t, 1.0), "log_mgas_total": log10_floor(mgas_t, 1.0),
        "log_mdm_total": log10_floor(mt[:, 1], 1.0), "log_mbh": log10_floor(cat["SubhaloBHMass"][sub] * msun, 1.0),
        "gas_fraction": mgas_t / np.maximum(mgas_t + mstar_t, 1.0),
        "log_sfr": log10_floor(sfr, 1e-5), "log_ssfr": log10_floor(sfr / np.maximum(mstar_t, 1.0), 1e-15),
        "log_z_stars": log10_floor(cat["SubhaloStarMetallicity"][sub] / Z_SUN, 1e-4),
        "log_z_gas": np.where(zgas > 0, log10_floor(zgas / Z_SUN, 1e-4), np.nan),
        "veldisp": cat["SubhaloVelDisp"][sub].astype(np.float64),
        "spin": np.linalg.norm(cat["SubhaloSpin"][sub].astype(np.float64), axis=1) * a / h,
        "g_minus_r": phot[:, G_BAND] - phot[:, R_BAND],
    }

    radius = cfg.train_radius * index["rhalf_star_kpc"]
    items = []
    for s in np.unique(index["shard"]):
        pos = np.flatnonzero(index["shard"] == s)
        items.append((shard_path(args.out, args.snap, s), index["row"][pos], radius[pos], pos))
    age = np.full(len(sub), np.nan)
    mhi, mgas = np.zeros(len(sub)), np.zeros(len(sub))
    with Pool(args.workers) as pool:
        for pos, vals in pool.imap_unordered(sphere_values, items):
            for j, (ag, hi, gm) in zip(pos, vals):
                age[j], mhi[j], mgas[j] = ag, hi, gm
    t.update(age_mean_gyr=age, log_mhi_sphere=log10_floor(mhi, 1.0), log_mgas_sphere=log10_floor(mgas, 1.0))
    write_table(sdir / "targets.h5", t, {"snapshot": args.snap, "sphere_radii": cfg.train_radius,
                                         "about": __doc__.split("Writes ")[1]})
    print(f"wrote {sdir / 'targets.h5'}: {len(sub):,} galaxies, {len(t) - 2} quantities")
    for k in ("log_m200c", "log_mstar_total", "gas_fraction", "log_sfr", "age_mean_gyr", "log_mhi_sphere", "g_minus_r"):
        v = t[k]
        print(f"  {k:16s} median {np.nanmedian(v):8.3f}  5-95%: {np.nanpercentile(v, 5):8.3f} .. "
              f"{np.nanpercentile(v, 95):8.3f}  NaN {np.isnan(v).sum()}")


if __name__ == "__main__":
    main()
