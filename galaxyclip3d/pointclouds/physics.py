"""Unit conversions and derived particle quantities for TNG snapshots, in physical units at any redshift.

Code units (TNG data specifications): lengths ckpc/h, masses 1e10 Msun/h, particle velocities km sqrt(a)/s,
Density (1e10 Msun/h)/(ckpc/h)^3, InternalEnergy (km/s)^2, MagneticField (h/a^2) (UnitPressure)^(1/2).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

M_P, K_B, MSUN_G, KPC_CM = 1.67262192e-24, 1.380649e-16, 1.98847e33, 3.0856776e21   # cgs
X_H, Z_SUN, GAMMA = 0.76, 0.0127, 5.0 / 3.0    # hydrogen mass fraction, solar metallicity, adiabatic index
B_GAUSS = 2.60e-6                              # code magnetic field (times h/a^2) to Gauss
Z_FLOOR = 1e-4                                 # metallicity floor before taking logs, in solar units


@dataclass(frozen=True)
class Units:
    h: float
    a: float
    box: float          # code units (ckpc/h)
    omega_m: float
    omega_l: float

    @classmethod
    def from_header(cls, hd: dict) -> "Units":
        return cls(float(hd["HubbleParam"]), float(hd["Time"]), float(hd["BoxSize"]), float(hd["Omega0"]),
                   float(hd["OmegaLambda"]))

    def kpc(self, x):
        return np.asarray(x, dtype=np.float64) * self.a / self.h

    def msun(self, m):
        return np.asarray(m, dtype=np.float64) * 1e10 / self.h

    def kms(self, v):
        return np.asarray(v, dtype=np.float64) * np.sqrt(self.a)

    def density_msun_kpc3(self, rho):
        return np.asarray(rho, dtype=np.float64) * 1e10 * self.h ** 2 / self.a ** 3


def relative_positions(coords: np.ndarray, centre: np.ndarray, box: float) -> np.ndarray:
    """Coordinates minus centre with the periodic box wrapped (nearest image), in code units."""
    d = np.asarray(coords, dtype=np.float64) - np.asarray(centre, dtype=np.float64)
    d -= box * np.round(d / box)
    return d


def temperature_k(internal_energy, electron_abundance) -> np.ndarray:
    mu = 4.0 / (1.0 + 3.0 * X_H + 4.0 * X_H * np.asarray(electron_abundance, dtype=np.float64))
    return (GAMMA - 1.0) * np.asarray(internal_energy, dtype=np.float64) * 1e10 * mu * M_P / K_B


def hydrogen_density_cm3(rho_msun_kpc3) -> np.ndarray:
    return X_H * np.asarray(rho_msun_kpc3) * MSUN_G / KPC_CM ** 3 / M_P


def log_metallicity(z) -> np.ndarray:
    """log10 Z/Zsun, floored at Z_FLOOR."""
    return np.log10(np.maximum(np.asarray(z, dtype=np.float64) / Z_SUN, Z_FLOOR))


def field_strength_ug(b, units: Units) -> np.ndarray:
    return np.linalg.norm(np.asarray(b, dtype=np.float64), axis=1) * units.h / units.a ** 2 * B_GAUSS * 1e6


def cell_radius_kpc(m_msun, rho_msun_kpc3) -> np.ndarray:
    """Radius of the sphere with the cell's volume."""
    return (3.0 * np.asarray(m_msun) / (4.0 * np.pi * np.asarray(rho_msun_kpc3))) ** (1.0 / 3.0)


def cosmic_time_gyr(a, units: Units) -> np.ndarray:
    """Age of a flat LCDM universe at scale factor a."""
    a = np.asarray(a, dtype=np.float64)
    return (2.0 / (3.0 * np.sqrt(units.omega_l)) * (9.7779 / units.h)
            * np.arcsinh(np.sqrt(units.omega_l / units.omega_m) * a ** 1.5))


def stellar_age_gyr(formation_a, units: Units) -> np.ndarray:
    """Clipped at 0: stars born at the snapshot time come out a rounding error below it."""
    return np.maximum(cosmic_time_gyr(units.a, units) - cosmic_time_gyr(formation_a, units), 0.0)
