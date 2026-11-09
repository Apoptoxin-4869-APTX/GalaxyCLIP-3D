"""Point clouds drawn from a galaxy's stored particles: the four modalities of GalaxyCLIP-3D.

    stars  star particles, drawn by mass             x y z vx vy vz age log_z
    dm     dark-matter particles, drawn uniformly    x y z vx vy vz
    gas    gas cells, drawn by mass                  x y z vx vy vz log_nh log_t log_z log_b x_hi sf
    hi     gas cells, drawn by neutral-H mass        x y z vx vy vz x_hi

Positions are divided by the sphere radius and velocities by the galaxy's maximum circular velocity. Every draw is
weighted by wcorr, which undoes the capping in extract.py. Gas and HI points are moved to a random place inside their
cell's sphere, so the two clouds never share exact coordinates. HI is drawn with replacement, because a few cells can
hold most of the neutral gas; the others without. One random rotation is applied to all modalities of a galaxy.
"""
from __future__ import annotations

import numpy as np

MODALITIES = ("stars", "dm", "gas", "hi")
SOURCE = {"stars": "stars", "dm": "dm", "gas": "gas", "hi": "gas"}
SCALARS = {"stars": ("age", "log_z"), "dm": (), "gas": ("log_nh", "log_t", "log_z", "log_b", "x_hi", "sf"),
           "hi": ("x_hi",)}
FEATURES = {m: ("x", "y", "z", "vx", "vy", "vz") + SCALARS[m] for m in MODALITIES}
STANDARDIZE = {"stars": ("age", "log_z"), "gas": ("log_nh", "log_t", "log_z", "log_b")}


def weights(modality: str, p: dict[str, np.ndarray]) -> np.ndarray:
    if modality == "dm":
        return p["wcorr"].astype(np.float64)
    key = "m_hi" if modality == "hi" else "mass"
    return p[key].astype(np.float64) * p["wcorr"]


def weighted_sample(rng: np.random.Generator, w: np.ndarray, k: int, replace: bool) -> np.ndarray:
    """k indices drawn with probability proportional to w; without replacement uses Efraimidis-Spirakis keys."""
    if replace or (w > 0).sum() < k:
        return rng.choice(len(w), size=k, p=w / w.sum())
    with np.errstate(divide="ignore"):
        keys = np.where(w > 0, np.log(rng.random(len(w))) / w, -np.inf)
    return np.argpartition(-keys, k - 1)[:k]


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Uniform random rotation matrix (from a random unit quaternion)."""
    a, b, c, d = (q := rng.normal(size=4)) / np.linalg.norm(q)
    return np.array([[a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
                     [2 * (b * c + a * d), a * a - b * b + c * c - d * d, 2 * (c * d - a * b)],
                     [2 * (b * d - a * c), 2 * (c * d + a * b), a * a - b * b - c * c + d * d]])


def in_ball(rng: np.random.Generator, n: int) -> np.ndarray:
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True) * rng.random((n, 1)) ** (1.0 / 3.0)


def draw(particles: dict[str, dict[str, np.ndarray]], radius_kpc: float, vmax: float, budget: int,
         rng: np.random.Generator, rotate: bool = True) -> dict[str, np.ndarray]:
    """{modality: (budget, len(FEATURES[modality])) float32}; all zeros when a modality has nothing to draw from."""
    rot = random_rotation(rng) if rotate else np.eye(3)
    vscale = vmax if vmax > 0 else 1.0
    out = {}
    for mod in MODALITIES:
        p = particles[SOURCE[mod]]
        inside = np.flatnonzero(np.linalg.norm(p["pos"], axis=1) <= radius_kpc)
        w = weights(mod, {k: v[inside] for k, v in p.items() if k in ("mass", "m_hi", "wcorr")})
        if len(inside) == 0 or w.sum() <= 0:
            out[mod] = np.zeros((budget, len(FEATURES[mod])), dtype=np.float32)
            continue
        pick = inside[weighted_sample(rng, w, budget, replace=(mod == "hi"))]
        pos = p["pos"][pick].astype(np.float64)
        if mod in ("gas", "hi"):
            pos += in_ball(rng, len(pick)) * p["r_cell"][pick, None]
        cols = [pos @ rot.T / radius_kpc, p["vel"][pick].astype(np.float64) @ rot.T / vscale]
        cols += [p[f][pick, None].astype(np.float64) for f in SCALARS[mod]]
        out[mod] = np.concatenate(cols, axis=1).astype(np.float32)
    return out


def standardize(clouds: dict[str, np.ndarray], stats: dict) -> dict[str, np.ndarray]:
    """Subtract the training mean and divide by the training spread for the features in STANDARDIZE."""
    out = {}
    for mod, arr in clouds.items():
        arr = arr.copy()
        for f in STANDARDIZE.get(mod, ()):
            j = FEATURES[mod].index(f)
            arr[:, j] = (arr[:, j] - stats[mod][f]["mean"]) / stats[mod][f]["std"]
        out[mod] = arr
    return out
