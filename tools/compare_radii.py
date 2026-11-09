"""Compare point clouds drawn inside two sphere sizes (default 5 and 10 stellar half-mass radii).

    python tools/compare_radii.py --out /cluster/projects/ska/illustris/tng50-1.pointclouds/v1 \
        --save outputs/radius_comparison [--radii 5 10] [--n 4] [--subhalo 167393]

For example galaxies: the four clouds drawn at each sphere size, face-on in the frame of the stars' spin, on the same
physical scale, with the smaller sphere marked. For all usable galaxies: the share of the larger sphere's points that
falls inside the smaller one, from the stored particles and the draw weights. Needs matplotlib.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from galaxyclip3d.pointclouds.config import Config                          # noqa: E402
from galaxyclip3d.pointclouds.sampling import MODALITIES, SOURCE, draw, weights  # noqa: E402
from galaxyclip3d.pointclouds.store import Shard                            # noqa: E402
from plot_clouds import INK, INK2, MUTED, SURFACE, pick, spin_frame         # noqa: E402

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"     # categorical slots 1-4, in order
GRID, AXIS = "#e1e0d9", "#c3c2b7"
TITLES = {"stars": "stars", "dm": "dark matter", "gas": "gas", "hi": "HI"}


def load_index(out: Path, cfg: Config, radius: float):
    with h5py.File(out / "index.h5", "r") as f:
        index = {k: v[()] for k, v in f["galaxies"].items()}
    k = f"r{radius:g}"
    usable = (index[f"n_stars_{k}"] >= cfg.min_particles) & (index[f"n_dm_{k}"] >= cfg.min_particles)
    ones = np.ones(len(usable), dtype=bool)
    mask = np.stack([ones, ones, index[f"n_gas_{k}"] >= cfg.min_particles, index[f"hi_eff_{k}"] >= cfg.hi_min_eff], 1)
    return index, usable, mask


def galaxy_figure(index, j, out: Path, cfg: Config, radii, path: Path) -> None:
    sh = Shard(out / "shards" / f"shard_{int(index['shard'][j]):04d}.h5")
    parts = {t: sh.particles(int(index["row"][j]), t) for t in ("stars", "dm", "gas")}
    rh = float(index["rhalf_star_kpc"][j])
    small, big = radii
    frame = spin_frame(parts["stars"], small * rh)
    lim = big * rh
    fig, axes = plt.subplots(2, 4, figsize=(13, 7.0), facecolor=SURFACE)
    for row, k in enumerate(radii):
        clouds = draw(parts, k * rh, float(index["vmax"][j]), cfg.budget,
                      np.random.default_rng([int(index["subhalo"][j]), int(k * 10)]), rotate=False)
        for col, mod in enumerate(MODALITIES):
            ax = axes[row, col]
            ax.set_facecolor(SURFACE)
            xyz = clouds[mod][:, :3].astype(np.float64) * k * rh @ frame.T          # back to kpc, face-on
            if np.any(clouds[mod]):
                ax.scatter(xyz[:, 0], xyz[:, 1], s=0.7, color=BLUE, linewidths=0, alpha=0.75, rasterized=True)
                inside = (np.linalg.norm(xyz, axis=1) <= small * rh).mean()
                note = f"{inside:.0%} of points inside {small:g} radii" if k != small else ""
            else:
                note = "masked at this sphere"
            for r, style in ((small * rh, (0, (4, 3))), (big * rh, "-")):
                ax.add_patch(plt.Circle((0, 0), r, fill=False, color=MUTED if r == small * rh else AXIS,
                                        linewidth=0.9, linestyle=style))
            ax.set_xlim(-lim * 1.05, lim * 1.05)
            ax.set_ylim(-lim * 1.05, lim * 1.05)
            ax.set_aspect("equal")
            ax.tick_params(labelsize=7, colors=MUTED, labelcolor=INK2, length=2)
            for s in ax.spines.values():
                s.set_color(AXIS)
            ax.set_title(f"{TITLES[mod]}, {k:g} radii", loc="left", fontsize=10, color=INK)
            if note:
                ax.text(0.02, 0.02, note, transform=ax.transAxes, fontsize=8, color=INK2, va="bottom")
            if col == 0:
                ax.set_ylabel("kpc", fontsize=8, color=INK2)
            if row == 1:
                ax.set_xlabel("kpc", fontsize=8, color=INK2)
            else:
                ax.tick_params(labelbottom=False)            # same scale below; keeps the lower titles clear
    kind = "central" if index["central"][j] else "satellite"
    fig.suptitle(f"subhalo {index['subhalo'][j]}  ·  {kind}  ·  log M* {np.log10(index['mstar'][j]):.2f}  ·  "
                 f"stellar half-mass radius {rh:.1f} kpc  ·  {cfg.budget:,} points per cloud  ·  face-on; dashed circle "
                 f"= {small:g} radii ({small * rh:.0f} kpc), solid = {big:g} radii ({big * rh:.0f} kpc)",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def inside_shares(index, rows, out: Path, radii) -> dict[str, np.ndarray]:
    """Expected share of a draw at the larger sphere that lands inside the smaller one, per galaxy and modality."""
    small, big = radii
    shares = {m: np.full(len(rows), np.nan) for m in MODALITIES}
    shards: dict[int, Shard] = {}
    for i, j in enumerate(rows):
        s = int(index["shard"][j])
        shards.setdefault(s, Shard(out / "shards" / f"shard_{s:04d}.h5"))
        rh = float(index["rhalf_star_kpc"][j])
        for mod in MODALITIES:
            p = shards[s].particles(int(index["row"][j]), SOURCE[mod],
                                    ["pos", "wcorr"] + (["mass"] if mod != "dm" else []) + (["m_hi"] if mod == "hi" else []))
            r = np.linalg.norm(p["pos"], axis=1)
            w = weights(mod, p)
            tot = w[r <= big * rh].sum()
            if tot > 0:
                shares[mod][i] = w[r <= small * rh].sum() / tot
        if (i + 1) % 1000 == 0:
            print(f"  shares: {i + 1:,} of {len(rows):,} galaxies", flush=True)
    return shares


def summary_figure(shares: dict[str, np.ndarray], radii, n: int, path: Path) -> None:
    small, big = radii
    edges = np.linspace(0, 1, 41)
    fig, ax = plt.subplots(figsize=(8.6, 4.4), facecolor=SURFACE)
    for mod, color in (("stars", BLUE), ("dm", ORANGE), ("gas", AQUA), ("hi", YELLOW)):
        v = shares[mod][np.isfinite(shares[mod])]
        ax.stairs(100 * np.histogram(v, edges)[0] / len(v), edges, color=color, linewidth=2,
                  label=f"{TITLES[mod]}: median {np.median(v):.0%} ({len(v):,} galaxies)")
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelcolor=INK2, labelsize=8)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel(f"share of the {big:g}-radii cloud's points inside {small:g} radii", fontsize=9, color=INK2)
    ax.set_ylabel("% of galaxies per bin", fontsize=9, color=INK2)
    ax.set_title(f"How much of a {big:g}-radii cloud still covers the {small:g}-radii region ({n:,} galaxies usable "
                 f"at {small:g} radii)", loc="left", fontsize=10, color=INK)
    leg = ax.legend(frameon=False, fontsize=8, loc="upper left")
    for t in leg.get_texts():
        t.set_color(INK2)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True, help="the dataset folder")
    ap.add_argument("--save", type=Path, required=True, help="folder for the pictures")
    ap.add_argument("--radii", type=float, nargs=2, default=(5.0, 10.0))
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--subhalo", type=int, nargs="*")
    args = ap.parse_args()
    cfg = Config.load(args.out / "config.json")
    radii = tuple(sorted(args.radii))
    if any(r not in cfg.radii for r in radii):
        sys.exit(f"radii must be among {cfg.radii}")
    index, usable, mask = load_index(args.out, cfg, radii[0])
    rows = ([int(np.flatnonzero(index["subhalo"] == s)[0]) for s in args.subhalo] if args.subhalo
            else pick(index, usable, mask, args.n)[:args.n])
    args.save.mkdir(parents=True, exist_ok=True)
    paths = []
    for j in sorted(rows, key=lambda r: index["mstar"][r]):
        paths.append(args.save / f"radii_{index['subhalo'][j]}.png")
        galaxy_figure(index, j, args.out, cfg, radii, paths[-1])
        print(f"wrote {paths[-1]}", flush=True)
    plt.imsave(args.save / "radii_examples.png", np.vstack([plt.imread(p)[..., :3] for p in paths]))
    every = np.flatnonzero(usable)
    shares = inside_shares(index, every, args.out, radii)
    summary_figure(shares, radii, len(every), args.save / "radii_summary.png")
    for mod in MODALITIES:
        v = shares[mod][np.isfinite(shares[mod])]
        print(f"{mod:5s}: share inside {radii[0]:g} radii when drawing at {radii[1]:g}: median {np.median(v):.1%}, "
              f"25-75% {np.percentile(v, 25):.1%}-{np.percentile(v, 75):.1%}, over {len(v):,} galaxies")
    print(f"wrote {args.save / 'radii_examples.png'} and {args.save / 'radii_summary.png'}")


if __name__ == "__main__":
    main()
