"""Compare point clouds drawn inside several sphere sizes (default 5 and 10 stellar half-mass radii).

    python tools/compare_radii.py --out /cluster/projects/ska/illustris/tng50-1.pointclouds/v3 [--snap 99] \
        --save outputs/radius_comparison [--radii 5 10 15] [--n 4] [--subhalo 167393] [--seed 1]

For example galaxies: the four clouds drawn at each sphere size, face-on in the frame of the stars' spin, on the same
physical scale, with every sphere marked. Each galaxy's draws are seeded by its subhalo number, so reruns repeat
them; --seed gives another set. Radii beyond the dataset's stored sphere are read from the raw snapshot (--root,
default the build's), for the example galaxies only, with the dataset's own extraction rules. For all usable galaxies,
when every radius is inside the stored sphere: the share of the largest sphere's points that falls inside the smallest
one, from the stored particles and the draw weights. Needs matplotlib.
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                             # noqa: E402
import numpy as np                                                           # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from galaxyclip3d.pointclouds.config import Config                          # noqa: E402
from galaxyclip3d.pointclouds.extract import extract                        # noqa: E402
from galaxyclip3d.pointclouds.layout import read_table, shard_path, snapshot_dir  # noqa: E402
from galaxyclip3d.pointclouds.sampling import MODALITIES, SOURCE, draw, weights  # noqa: E402
from galaxyclip3d.pointclouds.snapshot import Snapshot                      # noqa: E402
from galaxyclip3d.pointclouds.store import Shard                            # noqa: E402
from plot_clouds import INK, INK2, MUTED, SURFACE, pick, spin_frame         # noqa: E402

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"     # categorical slots 1-4, in order
GRID, AXIS = "#e1e0d9", "#c3c2b7"
TITLES = {"stars": "stars", "dm": "dark matter", "gas": "gas", "hi": "HI"}
LINES = [((0, (4, 3)), "dashed"), ((0, (1, 2)), "dotted"), ((0, (6, 2, 1, 2)), "dash-dot")]   # inner circles


def load_index(out: Path, cfg: Config, radius: float):
    index = read_table(snapshot_dir(out, cfg.snap) / "index.h5")
    k = f"r{radius:g}"
    usable = (index[f"n_stars_{k}"] >= cfg.min_particles) & (index[f"n_dm_{k}"] >= cfg.min_particles)
    ones = np.ones(len(usable), dtype=bool)
    mask = np.stack([ones, ones, index[f"n_gas_{k}"] >= cfg.min_particles, index[f"hi_eff_{k}"] >= cfg.hi_min_eff], 1)
    return index, usable, mask


def galaxy_figure(index, j, out: Path, cfg: Config, radii, path: Path, seed: int | None = None,
                  raw: dict | None = None) -> None:
    """One row of four clouds per radius. `raw`: the galaxy's particles out to the largest radius, read from the raw
    snapshot, used for radii beyond the stored sphere."""
    sh = Shard(shard_path(out, int(index["snapshot"][j]), int(index["shard"][j])))
    stored = {t: sh.particles(int(index["row"][j]), t) for t in ("stars", "dm", "gas")}
    rh = float(index["rhalf_star_kpc"][j])
    frame = spin_frame(stored["stars"], radii[0] * rh)
    lim = radii[-1] * rh
    circles = [(k, "-", "solid", AXIS) if k == radii[-1] else (k, *LINES[i % len(LINES)], MUTED)
               for i, k in enumerate(radii)]
    fig, axes = plt.subplots(len(radii), 4, figsize=(13, 3.5 * len(radii) + 0.3), facecolor=SURFACE, squeeze=False)
    for row, k in enumerate(radii):
        key = [int(index["subhalo"][j]), int(k * 10)]
        clouds = draw(stored if k <= cfg.extract_radius else raw, k * rh, float(index["vmax"][j]), cfg.budget,
                      np.random.default_rng(key if seed is None else [seed] + key), rotate=False)
        for col, mod in enumerate(MODALITIES):
            ax = axes[row, col]
            ax.set_facecolor(SURFACE)
            xyz = clouds[mod][:, :3].astype(np.float64) * k * rh @ frame.T          # back to kpc, face-on
            if np.any(clouds[mod]):
                ax.scatter(xyz[:, 0], xyz[:, 1], s=0.7, color=BLUE, linewidths=0, alpha=0.75, rasterized=True)
                r = np.linalg.norm(xyz, axis=1)
                inner = [s for s in radii if s < k]
                note = ("points inside " + "  ·  ".join(f"{s:g} radii: {(r <= s * rh).mean():.0%}" for s in inner)
                        if inner else "")
            else:
                note = "masked at this sphere"
            for s, style, _, color in circles:
                ax.add_patch(plt.Circle((0, 0), s * rh, fill=False, color=color, linewidth=0.9, linestyle=style))
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
            if row == len(radii) - 1:
                ax.set_xlabel("kpc", fontsize=8, color=INK2)
            else:
                ax.tick_params(labelbottom=False)            # same scale below; keeps the lower titles clear
    kind = "central" if index["central"][j] else "satellite"
    marks = ", ".join(f"{name} = {k:g} radii ({k * rh:.0f} kpc)" for k, _, name, _ in circles)
    beyond = [f"{k:g}" for k in radii if k > cfg.extract_radius]
    from_raw = (f";  {', '.join(beyond)} radii read from the raw snapshot (the dataset stores {cfg.extract_radius:g})"
                if beyond else "")
    fig.suptitle(f"subhalo {index['subhalo'][j]}  ·  {kind}  ·  log M* {np.log10(index['mstar'][j]):.2f}  ·  "
                 f"stellar half-mass radius {rh:.1f} kpc  ·  {cfg.budget:,} points per cloud\n"
                 f"face-on;  circles: {marks}{from_raw}", x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.62 / fig.get_figheight()))
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def inside_shares(index, rows, out: Path, radii) -> dict[str, np.ndarray]:
    """Expected share of a draw at the larger sphere that lands inside the smaller one, per galaxy and modality."""
    small, big = radii
    shares = {m: np.full(len(rows), np.nan) for m in MODALITIES}
    shards: dict[int, Shard] = {}
    for i, j in enumerate(rows):
        s = int(index["shard"][j])
        shards.setdefault(s, Shard(shard_path(out, int(index["snapshot"][j]), s)))
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
    ap.add_argument("--snap", type=int, default=99, help="snapshot of the dataset to draw from")
    ap.add_argument("--save", type=Path, required=True, help="folder for the pictures")
    ap.add_argument("--radii", type=float, nargs="+", default=(5.0, 10.0),
                    help="two or more sphere sizes in stellar half-mass radii; the smallest must be one of the build's")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--subhalo", type=int, nargs="*")
    ap.add_argument("--seed", type=int, help="another set of point draws (default: seeded by subhalo, so reruns repeat)")
    ap.add_argument("--root", type=Path, help="raw snapshot folder for radii beyond the stored sphere (default: the build's)")
    args = ap.parse_args()
    cfg = Config.load(snapshot_dir(args.out, args.snap) / "config.json")
    radii = tuple(sorted(set(args.radii)))
    if len(radii) < 2 or radii[0] not in cfg.radii:
        sys.exit(f"give two or more radii, the smallest among {cfg.radii}")
    index, usable, mask = load_index(args.out, cfg, radii[0])
    rows = ([int(np.flatnonzero(index["subhalo"] == s)[0]) for s in args.subhalo] if args.subhalo
            else pick(index, usable, mask, args.n)[:args.n])
    beyond = radii[-1] > cfg.extract_radius
    if beyond:                                       # the same extraction as the build, to a larger sphere
        table = read_table(snapshot_dir(args.out, args.snap) / "selection.h5")
        snap = Snapshot(args.root or cfg.root, cfg.snap)
        wide = replace(cfg, radii=tuple(cfg.radii) + (radii[-1],), extract_radius=radii[-1])
    args.save.mkdir(parents=True, exist_ok=True)
    paths = []
    for j in sorted(rows, key=lambda r: index["mstar"][r]):
        raw = None
        if beyond:
            t0 = time.time()
            s = int(np.flatnonzero(table["subhalo"] == index["subhalo"][j])[0])
            p = extract(snap, wide, {k: v[s] for k, v in table.items()})
            raw = {t: p[t] for t in ("stars", "dm", "gas")}
            print(f"subhalo {index['subhalo'][j]}: read to {radii[-1]:g} radii from the raw snapshot in "
                  f"{time.time() - t0:.0f} s; stored " + ", ".join(f"{t} {len(raw[t]['pos']):,}" for t in raw), flush=True)
        paths.append(args.save / f"radii_{index['subhalo'][j]}.png")
        galaxy_figure(index, j, args.out, cfg, radii, paths[-1], args.seed, raw)
        print(f"wrote {paths[-1]}", flush=True)
    plt.imsave(args.save / "radii_examples.png", np.vstack([plt.imread(p)[..., :3] for p in paths]))
    print(f"wrote {args.save / 'radii_examples.png'}")
    if beyond:
        print(f"no summary: it needs every galaxy's particles out to {radii[-1]:g} radii; the dataset stores "
              f"{cfg.extract_radius:g}")
        return
    pair = (radii[0], radii[-1])
    every = np.flatnonzero(usable)
    shares = inside_shares(index, every, args.out, pair)
    summary_figure(shares, pair, len(every), args.save / "radii_summary.png")
    for mod in MODALITIES:
        v = shares[mod][np.isfinite(shares[mod])]
        print(f"{mod:5s}: share inside {pair[0]:g} radii when drawing at {pair[1]:g}: median {np.median(v):.1%}, "
              f"25-75% {np.percentile(v, 25):.1%}-{np.percentile(v, 75):.1%}, over {len(v):,} galaxies")
    print(f"wrote {args.save / 'radii_summary.png'}")


if __name__ == "__main__":
    main()
