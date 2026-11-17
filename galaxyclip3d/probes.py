"""Linear probes: how well a per-galaxy feature vector (an embedding, or hand-made statistics) predicts the galaxy's
physical properties.

Ridge regression on standardized features, fitted on the training galaxies, with its strength chosen on the validation
galaxies and scored on the validation or test galaxies: R^2 for continuous properties, the area under the ROC curve
(AUC) for yes/no properties (central vs satellite, quenched or not). Properties come from snap_NNN/targets.h5
(tools/build_targets.py). Transforms: "log" takes log10 of a positive value; "above_floor" drops the values that
build_targets.py floored (a mass of 0 means the galaxy has none of that component, a star-formation rate at its floor
means none at all), so those fits measure how much, given there is some; "at_floor" turns the floor into a yes/no
property. A galaxy without a usable value (NaN) is left out of that probe.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

FLOORS = {"log_mgas_total": 0.0, "log_mbh": 0.0, "log_mgas_sphere": 0.0, "log_mhi_sphere": 0.0, "log_sfr": -5.0,
          "log_ssfr": -15.0}                      # floors of tools/build_targets.py (masses floored at 1 Msun)
# name: (column of targets.h5, transform, score). Masses, rates and metallicities are stored as log10 already.
PROPERTIES = {
    "log_mstar_total": ("log_mstar_total", None, "r2"), "log_mdm_total": ("log_mdm_total", None, "r2"),
    "log_mgas_total": ("log_mgas_total", "above_floor", "r2"), "log_mbh": ("log_mbh", "above_floor", "r2"),
    "log_m200c": ("log_m200c", None, "r2"),          # (R200c follows from M200c, so it is not probed separately)
    "gas_fraction": ("gas_fraction", None, "r2"), "log_mgas_sphere": ("log_mgas_sphere", "above_floor", "r2"),
    "log_mhi_sphere": ("log_mhi_sphere", "above_floor", "r2"), "log_sfr": ("log_sfr", "above_floor", "r2"),
    "log_ssfr": ("log_ssfr", "above_floor", "r2"), "age_mean_gyr": ("age_mean_gyr", None, "r2"),
    "g_minus_r": ("g_minus_r", None, "r2"), "log_z_stars": ("log_z_stars", None, "r2"),
    "log_z_gas": ("log_z_gas", None, "r2"), "veldisp": ("veldisp", "log", "r2"), "spin": ("spin", "log", "r2"),
    "rhalf_star_kpc": ("rhalf_star_kpc", "log", "r2"), "vmax": ("vmax", "log", "r2"),
    "central": ("central", None, "auc"), "quenched": ("log_ssfr", "at_floor", "auc"),
}
ALPHAS = np.logspace(-2, 6, 17)


def load_targets(root: str | Path, snap: int = 99) -> dict[str, np.ndarray]:
    """{property: float array} in the row order of snap_NNN/targets.h5, transformed as in PROPERTIES, plus "subhalo"."""
    path = Path(root) / f"snap_{snap:03d}" / "targets.h5"
    with h5py.File(path, "r") as f:
        g = f["galaxies"]
        out = {"subhalo": g["subhalo"][()].astype(np.int64)}
        for name, (column, transform, _) in PROPERTIES.items():
            v = g[column][()].astype(np.float64)
            if transform == "log":
                v = np.where(v > 0, np.log10(np.where(v > 0, v, 1.0)), np.nan)
            elif transform == "above_floor":
                v = np.where(v > FLOORS[column] + 1e-9, v, np.nan)
            elif transform == "at_floor":
                v = np.where(np.isfinite(v), (v <= FLOORS[column] + 1e-9).astype(np.float64), np.nan)
            out[name] = v
    return out


def lookup(targets: dict[str, np.ndarray], subhalo) -> dict[str, np.ndarray]:
    """Each property for the given subhalos (NaN for a subhalo the table lacks)."""
    where = {int(s): i for i, s in enumerate(targets["subhalo"])}
    rows = np.array([where.get(int(s), -1) for s in np.asarray(subhalo)])
    return {k: np.where(rows >= 0, v[np.maximum(rows, 0)], np.nan) for k, v in targets.items() if k != "subhalo"}


def r2(y: np.ndarray, pred: np.ndarray) -> float:
    """Coefficient of determination against the scored set's own mean (0 = no better than the mean, 1 = exact)."""
    ss = float(((y - y.mean()) ** 2).sum())
    return 1.0 - float(((y - pred) ** 2).sum()) / ss if ss > 0 else float("nan")


def auc(y: np.ndarray, score: np.ndarray) -> float:
    """Area under the ROC curve for binary labels y: the chance that a random positive outranks a random negative
    (ties count half)."""
    y = np.asarray(y).astype(bool)
    npos, nneg = int(y.sum()), int((~y).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    s = np.asarray(score)[order]
    i = 0
    while i < len(s):                                        # average ranks over ties
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def ridge_probe(xtr, ytr, xva, yva, xte=None, yte=None, score: str = "r2", alphas=ALPHAS) -> dict:
    """Fit ridge regression on (xtr, ytr) for each strength in `alphas`, keep the one scoring best on the validation
    set, and report its validation (and test) score. Features are standardized with the training mean and spread."""
    fn = r2 if score == "r2" else auc
    xtr, xva = np.asarray(xtr, np.float64), np.asarray(xva, np.float64)
    mu, sd = xtr.mean(0), xtr.std(0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    ztr, zva = (xtr - mu) / sd, (xva - mu) / sd
    ym = float(np.mean(ytr))
    u, s, vt = np.linalg.svd(ztr, full_matrices=False)
    uty = u.T @ (np.asarray(ytr, np.float64) - ym)
    best = None
    for a in alphas:
        w = vt.T @ (s / (s ** 2 + a) * uty)
        val = fn(np.asarray(yva), zva @ w + ym)
        if best is None or val > best[1]:
            best = (float(a), val, w)
    out = {"alpha": best[0], "val": best[1], "n_train": len(ytr), "n_val": len(yva)}
    if xte is not None:
        zte = (np.asarray(xte, np.float64) - mu) / sd
        out.update(test=fn(np.asarray(yte), zte @ best[2] + ym), n_test=len(yte))
    return out
