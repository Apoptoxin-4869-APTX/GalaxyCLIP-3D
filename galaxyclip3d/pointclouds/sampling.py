"""Point clouds drawn from a galaxy's stored particles: the four modalities of GalaxyCLIP-3D.

    stars  star particles, drawn by mass             x y z vx vy vz age log_z
    dm     dark-matter particles, drawn uniformly    x y z vx vy vz
    gas    gas cells, drawn by mass                  x y z vx vy vz log_nh log_t log_z log_b x_hi sf
    hi     gas cells, drawn by neutral-H mass        x y z vx vy vz x_hi

Positions are divided by the sphere radius and velocities by the galaxy's maximum circular velocity. Every draw is
weighted by wcorr, which undoes the capping in extract.py. Gas and HI points are moved to a random place inside their
cell's sphere, so the two clouds never share exact coordinates. HI is drawn with replacement, because a few cells can
hold most of the neutral gas; the others without.

Orientation: draw() can turn all four clouds by one random rotation, but then they share an orientation and can be
matched by it instead of by physics (matching only the direction of the stars' and the gas's angular momentum finds
the right galaxy among 394 for 35% of them). orient() gives every modality its own random rotation, and with mirror
also a reflection half of the time; that is what training uses.
"""
from __future__ import annotations

import numpy as np

MODALITIES = ("stars", "dm", "gas", "hi")
SOURCE = {"stars": "stars", "dm": "dm", "gas": "gas", "hi": "gas"}
SCALARS = {"stars": ("age", "log_z"), "dm": (), "gas": ("log_nh", "log_t", "log_z", "log_b", "x_hi", "sf"),
           "hi": ("x_hi",)}
FEATURES = {m: ("x", "y", "z", "vx", "vy", "vz") + SCALARS[m] for m in MODALITIES}
STANDARDIZE = {"stars": ("age", "log_z"), "gas": ("log_nh", "log_t", "log_z", "log_b")}
INPUTS = ("all", "no_velocities", "no_scalars")


def input_columns(modality: str, inputs: str = "all") -> list[int]:
    """Columns of FEATURES[modality] an encoder sees, for input ablations: everything, positions and the physical
    values (no_velocities), or positions and velocities (no_scalars). Positions stay first, where the patch encoder
    expects them."""
    n = len(FEATURES[modality])
    if inputs == "all":
        return list(range(n))
    if inputs == "no_velocities":
        return [0, 1, 2, *range(6, n)]
    if inputs == "no_scalars":
        return list(range(6))
    raise ValueError(f"inputs must be one of {INPUTS}, got {inputs!r}")


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


ORIENTATIONS = ("independent", "shared", "none")


def random_orthogonal(rng: np.random.Generator, mirror: bool = True) -> np.ndarray:
    """A uniform random rotation; with mirror, also a reflection (determinant -1) half of the time. Gravity and gas
    dynamics look the same in a mirror, and galaxies turn either way, so a mirrored galaxy is still a valid one."""
    r = random_rotation(rng)
    if mirror and rng.random() < 0.5:
        r = r.copy()
        r[:, 0] = -r[:, 0]
    return r


def orient(clouds: dict[str, np.ndarray], rng: np.random.Generator, mode: str = "independent",
           mirror: bool = True) -> dict[str, np.ndarray]:
    """Turn positions and velocities (columns 0-2 and 3-5) of each cloud: every modality by its own random matrix
    ("independent"), all by one ("shared"), or not at all ("none"). A matrix is drawn for every modality in
    MODALITIES order, whether its cloud is empty or left out of `clouds`, so a seeded rng gives a modality the same
    matrix every time, whichever other modalities are there."""
    if mode not in ORIENTATIONS:
        raise ValueError(f"orientation must be one of {ORIENTATIONS}, not {mode!r}")
    if mode == "none":
        return dict(clouds)
    shared = random_orthogonal(rng, mirror) if mode == "shared" else None
    out = {}
    for mod in MODALITIES:
        r = shared if shared is not None else random_orthogonal(rng, mirror)
        if mod not in clouds:
            continue
        c = np.array(clouds[mod], dtype=np.float32, copy=True)
        c[:, 0:3] = c[:, 0:3] @ r.T
        c[:, 3:6] = c[:, 3:6] @ r.T
        out[mod] = c
    return out


def in_ball(rng: np.random.Generator, n: int) -> np.ndarray:
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True) * rng.random((n, 1)) ** (1.0 / 3.0)


def draw(particles: dict[str, dict[str, np.ndarray]], radius_kpc: float | dict[str, float], vmax: float, budget: int,
         rng: np.random.Generator, rotate: bool = True, modalities=MODALITIES) -> dict[str, np.ndarray]:
    """{modality: (budget, len(FEATURES[modality])) float32} for each of `modalities` (the others are not drawn); all
    zeros when a modality has nothing to draw from. radius_kpc is one sphere for every modality, or {modality: radius}
    (e.g. a dark-matter sphere of its own); each cloud's positions are divided by its own sphere's radius."""
    rot = random_rotation(rng) if rotate else np.eye(3)
    vscale = vmax if vmax > 0 else 1.0
    out = {}
    for mod in MODALITIES:
        if mod not in modalities:
            continue
        p = particles[SOURCE[mod]]
        r_mod = float(radius_kpc[mod]) if isinstance(radius_kpc, dict) else radius_kpc
        inside = np.flatnonzero(np.linalg.norm(p["pos"], axis=1) <= r_mod)
        w = weights(mod, {k: v[inside] for k, v in p.items() if k in ("mass", "m_hi", "wcorr")})
        if len(inside) == 0 or w.sum() <= 0:
            out[mod] = np.zeros((budget, len(FEATURES[mod])), dtype=np.float32)
            continue
        pick = inside[weighted_sample(rng, w, budget, replace=(mod == "hi"))]
        pos = p["pos"][pick].astype(np.float64)
        if mod in ("gas", "hi"):
            pos += in_ball(rng, len(pick)) * p["r_cell"][pick, None]
        cols = [pos @ rot.T / r_mod, p["vel"][pick].astype(np.float64) @ rot.T / vscale]
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
