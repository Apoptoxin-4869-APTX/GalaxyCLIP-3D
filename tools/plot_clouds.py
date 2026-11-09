"""Pictures of galaxies' four point clouds (stars, dark matter, gas, HI) in 3D, each point coloured by a property.

    python tools/plot_clouds.py --out /cluster/projects/ska/illustris/tng50-1.pointclouds/v1 \
        --save outputs/pointcloud_examples [--subhalo 0 143885] [--n 6]

Clouds are drawn the way the training loader draws them (sphere of the build's training radius, `budget` points each,
features not standardized), then turned so that the stars' angular momentum points up (+z); the turn is for display
only. Stars are coloured by age, dark matter by speed, gas by temperature and HI by rotation velocity around that axis
(positive = turning with the stars). Without --subhalo, n galaxies spread over stellar mass with all four modalities
present, plus one galaxy whose gas and HI are masked. Needs matplotlib.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                             # noqa: E402
import numpy as np                                                           # noqa: E402
from matplotlib.colors import LinearSegmentedColormap                       # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from galaxyclip3d.pointclouds.config import Config                          # noqa: E402
from galaxyclip3d.pointclouds.sampling import FEATURES, draw                # noqa: E402
from galaxyclip3d.pointclouds.store import Shard                            # noqa: E402

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
# one-hue ramps from the categorical slots (blue, aqua, orange), started dark enough to stay visible on the surface,
# and a blue-red diverging ramp around a neutral grey
RAMPS = {
    "blue": LinearSegmentedColormap.from_list("blue", ["#86b6ef", "#3987e5", "#256abf", "#0d366b"]),
    "aqua": LinearSegmentedColormap.from_list("aqua", ["#8fdcbf", "#1baf7a", "#148a60", "#0b5139"]),
    "orange": LinearSegmentedColormap.from_list("orange", ["#f6b496", "#eb6834", "#c4501f", "#7a2e0f"]),
    "diverging": LinearSegmentedColormap.from_list("div", ["#184f95", "#5598e7", "#a9a8a2", "#ec7b7a", "#a32a29"]),
}
PANELS = [  # modality, title, colour label, ramp, value range
    ("stars", "stars", "age [Gyr]", "blue", (0.0, 13.8)),
    ("dm", "dark matter", "speed [km/s]", "aqua", None),
    ("gas", "gas", "log10 temperature [K]", "orange", (3.8, 7.2)),
    ("hi", "HI", "rotation velocity / Vmax", "diverging", (-1.2, 1.2)),
]
MASKED = {"gas": "masked: fewer than {n:,} gas cells", "hi": "masked: too little neutral gas"}


def spin_frame(stars: dict, radius_kpc: float) -> np.ndarray:
    """Rotation matrix whose rows are x, y, z of a frame with z along the stars' angular momentum."""
    inside = np.linalg.norm(stars["pos"], axis=1) <= radius_kpc
    m = (stars["mass"] * stars["wcorr"])[inside, None].astype(np.float64)
    L = (m * np.cross(stars["pos"][inside].astype(np.float64), stars["vel"][inside].astype(np.float64))).sum(0)
    z = L / np.linalg.norm(L)
    x = np.cross([0.0, 0.0, 1.0] if abs(z[2]) < 0.9 else [1.0, 0.0, 0.0], z)
    x /= np.linalg.norm(x)
    return np.vstack([x, np.cross(z, x), z])


def colour_values(mod: str, cloud: np.ndarray, vmax: float) -> np.ndarray:
    f = FEATURES[mod]
    if mod == "stars":
        return cloud[:, f.index("age")]
    if mod == "dm":
        return np.linalg.norm(cloud[:, 3:6], axis=1) * vmax
    if mod == "gas":
        return cloud[:, f.index("log_t")]
    x, y, vx, vy = cloud[:, 0], cloud[:, 1], cloud[:, 3], cloud[:, 4]
    return (x * vy - y * vx) / np.maximum(np.hypot(x, y), 1e-6)


def figure(index: dict, j: int, out: Path, cfg: Config, mask: np.ndarray, path: Path) -> None:
    sh = Shard(out / "shards" / f"shard_{int(index['shard'][j]):04d}.h5")
    row = int(index["row"][j])
    parts = {t: sh.particles(row, t) for t in ("stars", "dm", "gas")}
    radius = cfg.train_radius * float(index["rhalf_star_kpc"][j])
    vmax = float(index["vmax"][j])
    clouds = draw(parts, radius, vmax, cfg.budget, np.random.default_rng(int(index["subhalo"][j])), rotate=False)
    frame = spin_frame(parts["stars"], radius)
    fig = plt.figure(figsize=(16, 4.9), facecolor=SURFACE)
    for k, (mod, title, label, ramp, rng) in enumerate(PANELS):
        ax = fig.add_subplot(1, 4, k + 1, projection="3d")
        ax.set_facecolor(SURFACE)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_pane_color((0.97, 0.97, 0.96, 1.0))
            axis.line.set_color("#c3c2b7")
            axis._axinfo["grid"]["color"] = "#e1e0d9"
            axis.set_ticklabels([])
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=22, azim=-58)
        on = mask[j, ["stars", "dm", "gas", "hi"].index(mod)]
        if not on:
            ax.text2D(0.5, 0.5, MASKED[mod].format(n=cfg.min_particles), transform=ax.transAxes, ha="center",
                      va="center", fontsize=9, color=INK2)
            ax.set_title(title, loc="left", fontsize=11, color=INK)
            placeholder = fig.colorbar(matplotlib.cm.ScalarMappable(cmap=RAMPS[ramp]), ax=ax,
                                       orientation="horizontal", shrink=0.62, pad=0.02, aspect=28)
            placeholder.ax.set_visible(False)              # keeps the panel the same size as the others
            continue
        c = clouds[mod].astype(np.float64)
        c[:, :3] = c[:, :3] @ frame.T
        c[:, 3:6] = c[:, 3:6] @ frame.T
        val = colour_values(mod, c, vmax)
        lo, hi = rng if rng else (np.percentile(val, 1), np.percentile(val, 99))
        sc = ax.scatter(c[:, 0], c[:, 1], c[:, 2], c=val, cmap=RAMPS[ramp], vmin=lo, vmax=hi, s=1.6, linewidths=0,
                        alpha=0.9, depthshade=False)
        cb = fig.colorbar(sc, ax=ax, orientation="horizontal", shrink=0.62, pad=0.02, aspect=28)
        cb.set_label(label, fontsize=8, color=INK2)
        cb.ax.tick_params(labelsize=7, colors=MUTED, labelcolor=INK2)
        cb.outline.set_edgecolor("#c3c2b7")
        ax.set_title(title, loc="left", fontsize=11, color=INK)
    kind = "central" if index["central"][j] else "satellite"
    fig.suptitle(f"subhalo {index['subhalo'][j]}  ·  {kind}  ·  log M* {np.log10(index['mstar'][j]):.2f}  ·  "
                 f"sphere {radius:.1f} kpc  ·  Vmax {vmax:.0f} km/s  ·  {cfg.budget:,} points per cloud, "
                 f"turned so the stars' spin points up", x=0.01, ha="left", fontsize=11, color=INK)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.04, wspace=0.02)
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def pick(index: dict, usable: np.ndarray, mask: np.ndarray, n: int) -> list[int]:
    lm = np.log10(index["mstar"])
    full = np.flatnonzero(usable & mask[:, 2] & mask[:, 3])
    targets = np.linspace(np.percentile(lm[full], 3), np.percentile(lm[full], 99.5), n)
    rows: list[int] = []
    for i, x in enumerate(targets):
        pool = full[index["central"][full] == (i % 2 == 0)]
        pool = np.setdiff1d(pool, rows)
        rows.append(int(pool[np.argmin(np.abs(lm[pool] - x))]))
    gasless = np.flatnonzero(usable & ~mask[:, 2] & ~index["central"])
    if len(gasless):
        rows.append(int(gasless[np.argmin(np.abs(lm[gasless] - np.median(lm[full])))]))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True, help="the dataset folder")
    ap.add_argument("--save", type=Path, required=True, help="folder for the pictures")
    ap.add_argument("--subhalo", type=int, nargs="*")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--combine", type=Path, help="also stack all pictures into this one PNG")
    args = ap.parse_args()
    cfg = Config.load(args.out / "config.json")
    with h5py.File(args.out / "index.h5", "r") as f:
        index = {k: v[()] for k, v in f["galaxies"].items()}
    k = f"r{cfg.train_radius:g}"
    usable = (index[f"n_stars_{k}"] >= cfg.min_particles) & (index[f"n_dm_{k}"] >= cfg.min_particles)
    ones = np.ones(len(usable), dtype=bool)
    mask = np.stack([ones, ones, index[f"n_gas_{k}"] >= cfg.min_particles, index[f"hi_eff_{k}"] >= cfg.hi_min_eff], 1)
    rows = ([int(np.flatnonzero(index["subhalo"] == s)[0]) for s in args.subhalo] if args.subhalo
            else pick(index, usable, mask, args.n))
    args.save.mkdir(parents=True, exist_ok=True)
    paths = []
    for j in sorted(rows, key=lambda r: (mask[r].sum() < 4, index["mstar"][r])):
        paths.append(args.save / f"galaxy_{index['subhalo'][j]}.png")
        figure(index, j, args.out, cfg, mask, paths[-1])
        print(f"wrote {paths[-1]}")
    if args.combine:
        plt.imsave(args.combine, np.vstack([plt.imread(p)[..., :3] for p in paths]))
        print(f"wrote {args.combine}")


if __name__ == "__main__":
    main()
