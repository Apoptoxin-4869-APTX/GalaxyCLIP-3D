"""Quality checks of one snapshot of a point-cloud dataset, and a PDF of example clouds to look at before training.

    python tools/check_pointclouds.py --out DIR --snap 99 [--file DIR/snap_099/pilot.h5] [--pdf PATH] [--galaxies 12]

Checks every shard of the snapshot (or one --file): it says it holds this snapshot, stored counts against the
selection, no NaN or infinite values, plausible value ranges, gas and HI draws sharing no coordinates, and the splits:
every split unit (the z = 0 halo a galaxy ends in) in one split, the same one OUT/splits.h5 records for all snapshots.
Halos of this snapshot whose galaxies end in different z = 0 halos, and so may sit in different splits, are counted.
The PDF shows, for galaxies across stellar mass, one drawn cloud per modality projected on x-y; only it needs
matplotlib.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from galaxyclip3d.pointclouds.config import Config                          # noqa: E402
from galaxyclip3d.pointclouds.extract import FIELDS                         # noqa: E402
from galaxyclip3d.pointclouds.layout import read_table, snapshot_dir         # noqa: E402
from galaxyclip3d.pointclouds.lineage import load_splits, unit_keys         # noqa: E402
from galaxyclip3d.pointclouds.sampling import MODALITIES, draw              # noqa: E402
from galaxyclip3d.pointclouds.store import Shard                            # noqa: E402

RANGES = {"log_nh": (-9, 6), "log_t": (1, 10), "log_z": (-4.01, 2), "log_b": (-6.01, 4), "x_hi": (0, 1), "age": (0, 14),
          "r_cell": (0, 100), "mass": (0, 1e8), "m_hi": (0, 1e8), "mass_init": (0, 1e8)}
SURFACE, INK, INK2, BLUE = "#fcfcfb", "#0b0b0b", "#52514e", "#2a78d6"


def check_shard(sh: Shard, cfg: Config, problems: list[str]) -> dict:
    g = sh.galaxies()
    name = sh.path.name
    if int(sh.file.attrs.get("snapshot", -1)) != cfg.snap or not (g["snapshot"] == cfg.snap).all():
        problems.append(f"{name}: does not say it holds snapshot {cfg.snap}")
    for typ, fields in FIELDS.items():
        off = sh.file[typ]["offsets"][()]
        stored = np.diff(off)
        if not np.array_equal(stored, g[f"n_{typ}_stored"]):
            problems.append(f"{name}: {typ} stored counts differ from the galaxy table")
        if (stored > g[f"n_{typ}_r{cfg.extract_radius:g}"]).any():
            problems.append(f"{name}: more {typ} stored than inside the sphere")
        for f in fields:
            a = sh.file[typ][f][()]
            if a.dtype.kind == "f" and not np.isfinite(a).all():
                problems.append(f"{name}: {typ}/{f} has NaN or infinite values")
            if f in RANGES and len(a) and (a.min() < RANGES[f][0] or a.max() > RANGES[f][1]):
                problems.append(f"{name}: {typ}/{f} outside {RANGES[f]}: {a.min():.3g}..{a.max():.3g}")
            if f == "pos" and len(a):
                r = np.linalg.norm(a, axis=1)
                lim = np.repeat(cfg.extract_radius * g["rhalf_star_kpc"], stored) * 1.0001
                if (r > lim).any():
                    problems.append(f"{name}: {typ} particles outside the extraction sphere")
            if f == "wcorr" and len(a) and a.min() < 1 - 1e-6:
                problems.append(f"{name}: {typ}/wcorr below 1")
    return g


def shared_coordinates(sh: Shard, row: int, radius_kpc: float, vmax: float, budget: int) -> int:
    parts = {n: sh.particles(row, n) for n in ("stars", "dm", "gas")}
    c = draw(parts, radius_kpc, vmax, budget, np.random.default_rng(row), rotate=False)
    if not c["gas"].any() or not c["hi"].any():          # a masked modality is all zeros
        return 0
    a = {tuple(x) for x in np.round(c["gas"][:, :3], 7)}
    return sum(tuple(x) in a for x in np.round(c["hi"][:, :3], 7))


def pdf(files: list[Path], cfg: Config, n: int, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    rows = []
    for f in files:
        sh = Shard(f)
        g = sh.galaxies()
        for r in range(len(g["subhalo"])):
            rows.append((np.log10(g["mstar"][r]), f, r))
    rows.sort()
    pick = [rows[int(round(i))] for i in np.linspace(0, len(rows) - 1, min(n, len(rows)))]
    with PdfPages(path) as pp:
        for k in range(0, len(pick), 3):
            fig, axes = plt.subplots(3, 4, figsize=(11, 8.6), facecolor=SURFACE)
            for ax_row, (lm, f, r) in zip(axes, pick[k:k + 3]):
                sh = Shard(f)
                g = sh.galaxies()
                parts = {t: sh.particles(r, t) for t in ("stars", "dm", "gas")}
                radius = cfg.train_radius * g["rhalf_star_kpc"][r]
                c = draw(parts, radius, g["vmax"][r], cfg.budget, np.random.default_rng(r), rotate=False)
                kind = "central" if g["central"][r] else "satellite"
                for ax, m in zip(ax_row, MODALITIES):
                    pts = c[m]
                    empty = not np.any(pts)
                    if not empty:
                        ax.scatter(pts[:, 0], pts[:, 1], s=0.6, color=BLUE, linewidths=0, rasterized=True)
                    ax.set_xlim(-1, 1)
                    ax.set_ylim(-1, 1)
                    ax.set_aspect("equal")
                    ax.set_xticks([])
                    ax.set_yticks([])
                    for s in ax.spines.values():
                        s.set_color("#c3c2b7")
                    ax.set_facecolor(SURFACE)
                    ax.set_title(f"{m}" + (" (nothing to draw)" if empty else ""), fontsize=9, color=INK, loc="left")
                ax_row[0].set_ylabel(f"subhalo {g['subhalo'][r]}\n{kind}, log M* {lm:.2f}\n"
                                     f"sphere {radius:.1f} kpc", fontsize=8, color=INK2)
            for ax_row in axes[len(pick[k:k + 3]):]:
                for ax in ax_row:
                    ax.axis("off")
            fig.suptitle(f"Snapshot {cfg.snap}: drawn clouds ({cfg.budget:,} points each), x-y projection, sphere of "
                         f"{cfg.train_radius:g} stellar half-mass radii", x=0.01, ha="left", fontsize=11, color=INK)
            fig.tight_layout(rect=(0, 0, 1, 0.96))
            pp.savefig(fig, facecolor=SURFACE)
            plt.close(fig)
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True, help="dataset folder")
    ap.add_argument("--snap", type=int, required=True)
    ap.add_argument("--file", type=Path, help="check this one shard-format file instead of the snapshot's shards")
    ap.add_argument("--pdf", type=Path)
    ap.add_argument("--galaxies", type=int, default=12)
    args = ap.parse_args()
    sdir = snapshot_dir(args.out, args.snap)
    cfg = Config.load(sdir / "config.json")
    files = [args.file] if args.file else sorted((sdir / "shards").glob("shard_*.h5"))
    problems: list[str] = []
    if not args.file:
        n_shards = int(read_table(sdir / "selection.h5")["shard"].max()) + 1
        if len(files) != n_shards:
            problems.append(f"{n_shards - len(files)} of {n_shards} shards missing")
    registry, _ = load_splits(args.out)
    units_split: dict[int, set] = {}
    groups_split: dict[int, set] = {}
    shared = 0
    n_gal = 0
    for f in files:
        sh = Shard(f)
        g = check_shard(sh, cfg, problems)
        n_gal += len(g["subhalo"])
        for grp, key, sp in zip(g["group"], unit_keys(g["unit_snap"], g["unit_group"]), g["split"]):
            groups_split.setdefault(int(grp), set()).add(int(sp))
            units_split.setdefault(int(key), set()).add(int(sp))
            if int(key) not in registry or registry[int(key)][0] != int(sp):
                problems.append(f"{f.name}: a galaxy's split differs from OUT/splits.h5 for its split unit")
        for r in range(0, len(g["subhalo"]), max(1, len(g["subhalo"]) // 4)):
            radius = cfg.train_radius * g["rhalf_star_kpc"][r]
            shared += shared_coordinates(sh, r, radius, g["vmax"][r], cfg.budget)
    split_units = sum(len(v) > 1 for v in units_split.values())
    if split_units:
        problems.append(f"{split_units} split units have galaxies in more than one split")
    split_halos = sum(len(v) > 1 for v in groups_split.values())
    if shared:
        problems.append(f"{shared} HI points share exact coordinates with gas points")
    print(f"snapshot {args.snap}: checked {len(files)} file(s), {n_gal:,} galaxies: " +
          ("no problems" if not problems else f"{len(problems)} problems"))
    print(f"  halos of this snapshot with galaxies in more than one split (their galaxies end in different z = 0 "
          f"halos): {split_halos}")
    for p in problems[:50]:
        print("  " + p)
    if args.pdf:
        pdf(files, cfg, args.galaxies, args.pdf)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
