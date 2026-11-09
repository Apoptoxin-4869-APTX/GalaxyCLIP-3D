"""One galaxy's particles within a sphere, in physical units, ready for a shard file.

Particles are those bound to the subhalo (one contiguous block per type in the snapshot), centred on SubhaloPos
with the periodic box wrapped, cut to a sphere of cfg.extract_radius stellar half-mass radii. Wind cells are removed
from the stars. Above cfg.cap particles per type a random subset is kept, each particle with inclusion probability
pi = min(1, c q) (Poisson sampling): q follows mass for stars, is uniform for dark matter, and for gas is half mass,
half HI mass, so that mass- and HI-weighted draws both stay unbiased once weighted by wcorr = 1 / pi.
"""
from __future__ import annotations

import numpy as np

from . import physics as ph
from .config import Config
from .snapshot import Snapshot

TYPES = {"gas": 0, "dm": 1, "stars": 4}
READ = {
    "gas": ["Coordinates", "Velocities", "Masses", "Density", "InternalEnergy", "ElectronAbundance",
            "NeutralHydrogenAbundance", "StarFormationRate", "GFM_Metallicity", "MagneticField"],
    "dm": ["Coordinates", "Velocities"],
    "stars": ["Coordinates", "Velocities", "Masses", "GFM_InitialMass", "GFM_Metallicity", "GFM_StellarFormationTime"],
}
FIRST_PASS = {"gas": ["Coordinates", "Masses", "NeutralHydrogenAbundance"], "dm": ["Coordinates"],
              "stars": ["Coordinates", "Masses", "GFM_StellarFormationTime"]}
FIELDS = {      # stored per particle: name -> unit
    "gas": {"pos": "kpc", "vel": "km/s", "mass": "Msun", "m_hi": "Msun", "log_nh": "log10 cm^-3", "log_t": "log10 K",
            "log_z": "log10 Z/Zsun", "log_b": "log10 uG", "x_hi": "neutral H fraction", "sf": "1 if SFR > 0",
            "r_cell": "kpc", "wcorr": "1 / inclusion probability"},
    "dm": {"pos": "kpc", "vel": "km/s", "wcorr": "1 / inclusion probability"},
    "stars": {"pos": "kpc", "vel": "km/s", "mass": "Msun", "mass_init": "Msun", "age": "Gyr", "log_z": "log10 Z/Zsun",
              "wcorr": "1 / inclusion probability"},
}
B_FLOOR_UG = 1e-6


def inclusion_probabilities(q: np.ndarray, cap: int) -> np.ndarray:
    """pi_i = min(1, c q_i) with sum(pi) = cap."""
    q = np.asarray(q, dtype=np.float64)
    q = q / q.sum()
    lo, hi = float(cap), float(cap) / q[q > 0].min()
    for _ in range(200):
        mid = np.sqrt(lo * hi)
        if np.minimum(1.0, mid * q).sum() < cap:
            lo = mid
        else:
            hi = mid
    return np.minimum(1.0, hi * q)


def _cat(parts: list[np.ndarray], dtype, tail=()) -> np.ndarray:
    return np.concatenate(parts) if parts else np.empty((0,) + tail, dtype=dtype)


def _gather(snap: Snapshot, t: int, start: int, n: int, fields: list[str], rows: np.ndarray, chunk_rows: int) -> dict:
    """The given rows (sorted, relative to start) of a block."""
    out = {f: [] for f in fields}
    for row0, ch in snap.blocks(t, start, n, fields, chunk_rows):
        k = len(next(iter(ch.values())))
        a, b = np.searchsorted(rows, [row0, row0 + k])
        if b > a:
            local = rows[a:b] - row0
            for f in fields:
                out[f].append(ch[f][local])
    return out


def convert(name: str, d: dict, galaxy: dict, units: ph.Units, wcorr: np.ndarray) -> dict[str, np.ndarray]:
    """Raw snapshot fields -> stored fields (physical units, float32)."""
    n = len(wcorr)
    if n == 0:
        return {f: np.empty((0, 3) if f in ("pos", "vel") else (0,), dtype=np.uint8 if f == "sf" else np.float32)
                for f in FIELDS[name]}
    out = {"pos": units.kpc(ph.relative_positions(d["Coordinates"], galaxy["centre"], units.box)),
           "vel": units.kms(d["Velocities"]) - np.asarray(galaxy["bulk_velocity"])}
    if name == "gas":
        m = units.msun(d["Masses"])
        rho = units.density_msun_kpc3(d["Density"])
        x_hi = np.asarray(d["NeutralHydrogenAbundance"], dtype=np.float64)
        out.update(mass=m, m_hi=ph.X_H * x_hi * m, log_nh=np.log10(ph.hydrogen_density_cm3(rho)),
                   log_t=np.log10(ph.temperature_k(d["InternalEnergy"], d["ElectronAbundance"])),
                   log_z=ph.log_metallicity(d["GFM_Metallicity"]),
                   log_b=np.log10(np.maximum(ph.field_strength_ug(d["MagneticField"], units), B_FLOOR_UG)),
                   x_hi=x_hi, sf=(np.asarray(d["StarFormationRate"]) > 0).astype(np.uint8),
                   r_cell=ph.cell_radius_kpc(m, rho))
    elif name == "stars":
        out.update(mass=units.msun(d["Masses"]), mass_init=units.msun(d["GFM_InitialMass"]),
                   age=ph.stellar_age_gyr(d["GFM_StellarFormationTime"], units),
                   log_z=ph.log_metallicity(d["GFM_Metallicity"]))
    out["wcorr"] = wcorr
    return {k: v if k == "sf" else np.asarray(v, dtype=np.float32) for k, v in out.items()}


def extract(snap: Snapshot, cfg: Config, galaxy: dict) -> dict:
    """{'gas'|'dm'|'stars': {field: array}, 'counts': {column: value}} for one row of the selection table."""
    u = snap.units
    rh = float(galaxy["rhalf_star_kpc"])
    radii_kpc = np.asarray(cfg.radii) * rh
    r_ext = cfg.extract_radius * rh
    out: dict = {"counts": {}}
    for name, t in TYPES.items():
        start, n = int(galaxy["offsets"][t]), int(galaxy[f"n_{name}_bound"])
        two_pass = n > cfg.cap                      # the sphere may still need capping: decide after a light pass
        fields = FIRST_PASS[name] if two_pass else READ[name]
        rows, radius, mass, hi = [], [], [], []
        raw = {f: [] for f in fields}
        for row0, ch in snap.blocks(t, start, n, fields, cfg.chunk_rows):
            r = np.linalg.norm(u.kpc(ph.relative_positions(ch["Coordinates"], galaxy["centre"], u.box)), axis=1)
            inside = r <= r_ext
            if name == "stars":
                inside &= ch["GFM_StellarFormationTime"] > 0        # wind cells are not stars
            sel = np.flatnonzero(inside)
            rows.append(row0 + sel)
            radius.append(r[sel])
            if name in ("gas", "stars"):
                mass.append(np.asarray(ch["Masses"][sel], dtype=np.float64))
            if name == "gas":
                hi.append(mass[-1] * ch["NeutralHydrogenAbundance"][sel])
            if not two_pass:
                for f in fields:
                    raw[f].append(ch[f][sel])
        rows, radius = _cat(rows, np.int64), _cat(radius, np.float64)
        for k, rk in zip(cfg.radii, radii_kpc):
            out["counts"][f"n_{name}_r{k:g}"] = int((radius <= rk).sum())
        if name == "gas":
            hi_all = _cat(hi, np.float64)
            for k, rk in zip(cfg.radii, radii_kpc):
                w = hi_all[radius <= rk]
                out["counts"][f"hi_eff_r{k:g}"] = float(w.sum() ** 2 / (w ** 2).sum()) if w.sum() > 0 else 0.0
        if len(rows) > cfg.cap:
            if name == "dm":
                q = np.ones(len(rows))
            else:
                m = _cat(mass, np.float64)
                q = m / m.sum()
                if name == "gas" and hi_all.sum() > 0:
                    q = 0.5 * q + 0.5 * hi_all / hi_all.sum()
            pi = inclusion_probabilities(q, cfg.cap)
            rng = np.random.default_rng([cfg.seed, int(galaxy["subhalo"]), t])
            take = rng.random(len(rows)) < pi
            wcorr = 1.0 / pi[take]
        else:
            take, wcorr = np.ones(len(rows), dtype=bool), np.ones(len(rows))
        if two_pass:
            raw = _gather(snap, t, start, n, READ[name], rows[take], cfg.chunk_rows)
            data = {f: _cat(v, np.float64) for f, v in raw.items()}
        else:
            data = {f: _cat(v, np.float64)[take] for f, v in raw.items()}
        out[name] = convert(name, data, galaxy, u, wcorr)
        out["counts"][f"n_{name}_stored"] = int(take.sum())
    return out
