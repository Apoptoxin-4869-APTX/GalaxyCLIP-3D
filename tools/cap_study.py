"""Was the storage cap a good choice? Particles per galaxy, what each cap would cost, and how far a capped galaxy's
stored particles are from all of its particles.

    python tools/cap_study.py --out /cluster/projects/ska/illustris/tng50-1.pointclouds/v3 --save outputs/cap_study \
        [--snap 99] [--per-bin 20] [--draws 16] [--seeds 3]

The cap (extract.py) keeps at most `cap` particles per type and galaxy inside the stored sphere, each with inclusion
probability proportional to its draw weight, and `wcorr` = 1/probability makes draws unbiased. Writes:

  particle_counts.png  particles per galaxy and type inside the stored sphere and the training sphere, caps marked
  cap_cost.png         dataset size and share of galaxies capped for caps 2^12 ... 2^23, from the index counts and the
                       measured bytes per stored particle
  cap_accuracy.png     for capped galaxies read in full from the raw snapshot (`per-bin` from each of three size bins):
                       statistics of the particles kept at each cap (half-mass radius, velocity dispersion, rotation,
                       mean age or temperature) against the same statistics of all particles, in units of how much they
                       vary between clouds of `budget` points, inside 2, 5 and 10 stellar half-mass radii
  cap_accuracy.json    the per-galaxy numbers behind cap_accuracy.png (reused unless --recompute)

The reading and statistics need numpy and h5py; the pictures need matplotlib.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from galaxyclip3d.pointclouds.config import Config                          # noqa: E402
from galaxyclip3d.pointclouds.extract import extract, inclusion_probabilities  # noqa: E402
from galaxyclip3d.pointclouds.layout import read_table, snapshot_dir          # noqa: E402
from galaxyclip3d.pointclouds.sampling import MODALITIES, SOURCE, weighted_sample  # noqa: E402
from galaxyclip3d.pointclouds.snapshot import Snapshot                      # noqa: E402

TYPES = ("stars", "dm", "gas")
CAPS = [2 ** k for k in range(12, 24)]
RADII = (2.0, 5.0, 10.0)
SCALAR = {"stars": "age", "gas": "log_t"}
STATS = {m: ["half-mass radius", "velocity dispersion", "rotation velocity"] +
         ([f"mean {SCALAR[m]}"] if m in SCALAR else []) for m in MODALITIES}
BINS = [(2 ** 18, 2 ** 19), (2 ** 19, 2 ** 20), (2 ** 20, 2 ** 22 + 1)]   # by the largest type count in the sphere


def load_index(out: Path, snap: int) -> dict[str, np.ndarray]:
    return read_table(snapshot_dir(out, snap) / "index.h5")


# ---------------------------------------------------------------- the cap, exactly as extract.py applies it
def inclusion_q(name: str, p: dict) -> np.ndarray:
    if name == "dm":
        return np.ones(len(p["pos"]))
    q = p["mass"].astype(np.float64)
    q = q / q.sum()
    if name == "gas" and p["m_hi"].sum() > 0:
        h = p["m_hi"].astype(np.float64)
        q = 0.5 * q + 0.5 * h / h.sum()
    return q


class CapProbabilities:
    """pi = min(1, c q) with sum(pi) = cap, solved in closed form on the sorted weights (same result as
    extract.inclusion_probabilities, much faster when many caps are needed)."""

    def __init__(self, q: np.ndarray):
        self.q = q / q.sum()
        self.desc = np.sort(self.q)[::-1]
        self.tail = np.concatenate([np.cumsum(self.desc[::-1])[::-1], [0.0]])     # tail[j] = sum of desc[j:]

    def __call__(self, cap: int) -> np.ndarray:
        j = np.arange(min(cap, len(self.desc)))                 # j particles clipped at 1, the rest scaled by c
        c = (cap - j) / self.tail[j]
        first = int(np.flatnonzero(c * self.desc[j] < 1.0)[0])  # the first particle that is not clipped
        return np.minimum(1.0, c[first] * self.q)


def draw_weights(mod: str, p: dict, sel: np.ndarray, wcorr: np.ndarray) -> np.ndarray:
    if mod == "dm":
        return wcorr
    return p["m_hi" if mod == "hi" else "mass"][sel].astype(np.float64) * wcorr


# ---------------------------------------------------------------- statistics of a weighted set of particles
def statistics(pos, vel, w, scal, radius_kpc: float, vmax: float, axis: np.ndarray) -> np.ndarray:
    """Weighted: half-mass radius / sphere radius, velocity dispersion / Vmax, mean rotation velocity around the
    stars' spin axis / Vmax, and the mean of `scal` when given. All are averages or quantiles, so a capped set with
    wcorr weights estimates them without a bias that depends on the number of particles."""
    pos, vel, w = pos.astype(np.float64), vel.astype(np.float64), w.astype(np.float64)
    r = np.linalg.norm(pos, axis=1)
    order = np.argsort(r)
    cw = np.cumsum(w[order])
    r50 = r[order][min(np.searchsorted(cw, 0.5 * cw[-1]), len(r) - 1)] / radius_kpc
    W = w.sum()
    v = vel - (w[:, None] * vel).sum(0) / W
    sigma = np.sqrt((w * (v ** 2).sum(1)).sum() / W) / vmax
    rr = pos - (pos @ axis)[:, None] * axis                   # position in the plane perpendicular to the axis
    ephi = np.cross(axis, rr) / np.maximum(np.linalg.norm(rr, axis=1), 1e-12)[:, None]
    vphi = (w * (vel * ephi).sum(1)).sum() / W / vmax
    out = [r50, sigma, vphi]
    if scal is not None:
        out.append((w * scal.astype(np.float64)).sum() / W)
    return np.array(out)


def study_galaxy(full: dict, rh: float, vmax: float, subhalo: int, budget: int, draws: int, seeds: int,
                 min_particles: int, hi_min_eff: float) -> dict:
    """{mod: {radius: {'n': particles in the sphere, 'sd': [...], 'err': {cap: [rms over seeds per statistic]}}}}
    for the modalities the loader would use at each radius (same masks as galaxyclip3d/dataset.py)."""
    res: dict = {m: {} for m in MODALITIES}
    st = full["stars"]
    r_st = np.linalg.norm(st["pos"], axis=1)
    spin = {}
    for k in RADII:                                           # one axis per sphere, from all its stars, for all
        m = r_st <= k * rh
        L = (st["mass"][m, None].astype(np.float64) *
             np.cross(st["pos"][m].astype(np.float64), st["vel"][m].astype(np.float64))).sum(0)
        spin[k] = L / np.linalg.norm(L)
    for name in TYPES:
        p = full[name]
        n = len(p["pos"])
        r_all = np.linalg.norm(p["pos"], axis=1)
        mods = [m for m in MODALITIES if SOURCE[m] == name]
        truth, sd, ok = {}, {}, {}
        for mod in mods:                                          # all particles: the truth and one cloud's spread
            scal = p[SCALAR[mod]] if mod in SCALAR else None
            w_all = draw_weights(mod, p, np.arange(n), np.ones(n))
            for k in RADII:
                inside = np.flatnonzero(r_all <= k * rh)
                w = w_all[inside]
                if w.sum() <= 0:
                    continue
                if mod == "hi" and w.sum() ** 2 / (w ** 2).sum() < hi_min_eff:
                    continue                                  # the loader masks HI here
                if mod != "hi" and len(inside) < min_particles:
                    continue                                  # too few particles: masked, or the galaxy unusable
                key = (mod, k)
                truth[key] = statistics(p["pos"][inside], p["vel"][inside], w,
                                        None if scal is None else scal[inside], k * rh, vmax, spin[k])
                rng = np.random.default_rng([4869, subhalo, int(k * 10), MODALITIES.index(mod)])
                S = []
                for _ in range(draws):
                    pick = inside[weighted_sample(rng, w, budget, replace=(mod == "hi"))]
                    S.append(statistics(p["pos"][pick], p["vel"][pick], np.ones(budget),
                                        None if scal is None else scal[pick], k * rh, vmax, spin[k]))
                sd[key] = np.std(S, axis=0, ddof=1)
                ok[key] = True
                res[mod][k] = {"n": int(len(inside)), "sd": sd[key].tolist(), "err": {}}
        if not ok:
            continue
        probs = CapProbabilities(inclusion_q(name, p))
        for cap in CAPS:
            if n <= cap:
                break
            pi = probs(cap)
            errs = {key: [] for key in ok}
            for s in range(seeds):
                take = np.flatnonzero(np.random.default_rng([4869, subhalo, cap, s, TYPES.index(name)]).random(n) < pi)
                wcorr = 1.0 / pi[take]
                for (mod, k) in ok:
                    ins = r_all[take] <= k * rh
                    sel = take[ins]
                    w = draw_weights(mod, p, sel, wcorr[ins])
                    if w.sum() <= 0 or len(sel) < 2:
                        continue
                    scal = p[SCALAR[mod]][sel] if mod in SCALAR else None
                    S = statistics(p["pos"][sel], p["vel"][sel], w, scal, k * rh, vmax, spin[k])
                    errs[(mod, k)].append((S - truth[(mod, k)]) / np.maximum(sd[(mod, k)], 1e-12))
            for (mod, k), e in errs.items():
                if e:
                    res[mod][k]["err"][str(cap)] = np.sqrt(np.mean(np.square(e), axis=0)).tolist()
    return res


_WORKER: dict = {}


def _start_worker(root: Path, cfg: Config, draws: int, seeds: int) -> None:
    _WORKER.update(snap=Snapshot(root, cfg.snap), cfg=replace(cfg, cap=2 ** 40), draws=draws, seeds=seeds)


def _one_galaxy(item: tuple) -> dict:
    """Read one galaxy's particles in full (the build's extraction, never capped) and compare caps against them."""
    galaxy, rh, vmax = item
    t = time.time()
    full = extract(_WORKER["snap"], _WORKER["cfg"], galaxy)
    cfg = _WORKER["cfg"]
    res = study_galaxy(full, rh, vmax, int(galaxy["subhalo"]), cfg.budget, _WORKER["draws"], _WORKER["seeds"],
                       cfg.min_particles, cfg.hi_min_eff)
    return {"subhalo": int(galaxy["subhalo"]), "counts": {t_: int(len(full[t_]["pos"])) for t_ in TYPES},
            "seconds": time.time() - t, "results": res}


def accuracy(out: Path, cfg: Config, index: dict, per_bin: int, draws: int, seeds: int, root: Path | None,
             workers: int) -> dict:
    from multiprocessing import Pool

    table = read_table(snapshot_dir(out, cfg.snap) / "selection.h5")
    biggest = np.max([index[f"n_{t}_r{cfg.extract_radius:g}"] for t in TYPES], axis=0)
    rng = np.random.default_rng(4869)
    rows = []
    for lo, hi in BINS:
        pool = np.flatnonzero((biggest > lo if lo == 2 ** 18 else biggest >= lo) & (biggest < hi))
        rows += sorted(rng.choice(pool, size=min(per_bin, len(pool)), replace=False).tolist())
    q = inclusion_probabilities(np.array([1.0, 2.0, 3.0, 4.0]), 3)   # closed form agrees with extract.py's solver
    assert np.allclose(CapProbabilities(np.array([1.0, 2.0, 3.0, 4.0]))(3), q, rtol=1e-6)
    items = []
    for j in rows:
        s = int(np.flatnonzero(table["subhalo"] == index["subhalo"][j])[0])
        items.append(({k: v[s] for k, v in table.items()}, float(index["rhalf_star_kpc"][j]), float(index["vmax"][j])))
    galaxies = []
    t0 = time.time()
    with Pool(workers, initializer=_start_worker, initargs=(root or Path(cfg.root), cfg, draws, seeds)) as pool:
        for i, g in enumerate(pool.imap_unordered(_one_galaxy, items), 1):
            galaxies.append(g)
            print(f"[{i:3d}/{len(items)}] subhalo {g['subhalo']:>7d}  " +
                  "  ".join(f"{t_} {g['counts'][t_]:>10,}" for t_ in TYPES) +
                  f"  {g['seconds']:5.1f} s  (total {time.time() - t0:6.0f} s)", flush=True)
    galaxies.sort(key=lambda g: g["subhalo"])
    return {"budget": cfg.budget, "draws": draws, "seeds": seeds, "stats": STATS, "galaxies": galaxies}


def summarize(acc: dict, min_galaxies: int = 5) -> dict:
    """{mod: {radius: {cap: (median, p90, galaxies)}}} of the per-galaxy error (rms over statistics)."""
    out: dict = {}
    for mod in MODALITIES:
        for k in RADII:
            for cap in CAPS:
                e = [np.sqrt(np.mean(np.square(g["results"][mod][str(k)]["err"][str(cap)])))
                     for g in acc["galaxies"]
                     if str(k) in g["results"][mod] and str(cap) in g["results"][mod][str(k)]["err"]]
                if len(e) >= min_galaxies:
                    out.setdefault(mod, {}).setdefault(k, {})[cap] = (float(np.median(e)),
                                                                      float(np.percentile(e, 90)), len(e))
    return out


# ---------------------------------------------------------------- pictures
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
COLOR = {"stars": "#2a78d6", "dm": "#eb6834", "gas": "#1baf7a", "hi": "#eda100"}       # categorical slots 1-4
TITLE = {"stars": "stars", "dm": "dark matter", "gas": "gas", "hi": "HI"}


def style(ax, plt):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelcolor=INK2, labelsize=8)


def cap_lines(ax, cfg, ymax_frac=0.97, labels=True):
    """The v1 and v2 caps as vertical lines, labelled on the outer side of each so the labels never collide."""
    for cap, name, color, left in ((2 ** 17, "2¹⁷ (v1)", MUTED, True), (cfg.cap, "2¹⁸ (v2, now)", INK, False)):
        ax.axvline(cap, color=color, linewidth=1.0, linestyle=(0, (4, 3)) if cap != cfg.cap else "-")
        if labels:
            ax.text(cap / 1.08 if left else cap * 1.08, ymax_frac, name, transform=ax.get_xaxis_transform(),
                    fontsize=7.5, color=INK2, va="top", ha="right" if left else "left")


def plot_counts(index, cfg, save: Path, plt) -> None:
    n = len(index["subhalo"])
    edges = np.logspace(2, 8.5, 66)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), facecolor=SURFACE, sharey=True)
    for ax, t in zip(axes, TYPES):
        style(ax, plt)
        for k, color, label in ((cfg.extract_radius, COLOR["stars"], "inside 10 radii (stored sphere; the cap acts here)"),
                                (cfg.train_radius, COLOR["dm"], "inside 5 radii (training sphere)")):
            v = index[f"n_{t}_r{k:g}"]
            ax.stairs(np.histogram(np.maximum(v, 1), edges)[0], edges, color=color, linewidth=2, label=label)
        ax.set_xscale("log")
        ax.axvline(cfg.budget, color=MUTED, linewidth=0.9, linestyle=(0, (1, 2)))
        ax.text(cfg.budget / 1.08, 0.97, f"{cfg.budget:,} points\nper cloud", transform=ax.get_xaxis_transform(),
                fontsize=7.5, color=INK2, va="top", ha="right")
        cap_lines(ax, cfg)
        v10 = index[f"n_{t}_r{cfg.extract_radius:g}"]
        above = int((v10 > cfg.cap).sum())
        kept = np.minimum(v10, cfg.cap).sum() / v10.sum()
        noun = {"stars": "star particles", "dm": "dark-matter particles", "gas": "gas cells"}[t]
        ax.text(0.98, 0.62, f"{above:,} of {n:,} galaxies ({above / n:.0%})\nhave more than the cap;\n{kept:.0%} of all "
                f"{noun}\ninside 10 radii are kept", transform=ax.transAxes, ha="right", va="top", fontsize=8,
                color=INK2)
        ax.set_title(TITLE[t], loc="left", fontsize=11, color=INK)
        ax.set_xlabel("particles in the sphere", fontsize=9, color=INK2)
    axes[0].set_ylabel("galaxies per bin", fontsize=9, color=INK2)
    handles, labels = axes[0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(0.01, 0.92),
                     ncol=2)
    for tx in leg.get_texts():
        tx.set_color(INK2)
    fig.suptitle(f"Particles per galaxy ({n:,} galaxies, gas only where present)", x=0.01, ha="left",
                 fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(save / "particle_counts.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_cost(index, cfg, bytes_per: dict, measured: dict, save: Path, plt) -> dict:
    caps = np.array(CAPS, dtype=float)
    r = f"r{cfg.extract_radius:g}"
    gb = {t: np.array([np.minimum(index[f"n_{t}_{r}"], c).sum() for c in caps]) * bytes_per[t] / 1e9 for t in TYPES}
    total = sum(gb.values())
    nocap = sum(index[f"n_{t}_{r}"].sum() * bytes_per[t] for t in TYPES) / 1e9
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.4), facecolor=SURFACE)
    for ax in (a, b):
        style(ax, plt)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("cap (particles kept per type and galaxy)", fontsize=9, color=INK2)
        cap_lines(ax, cfg, labels=ax is b)
    a.plot(caps, total, color=INK, linewidth=2, label="all types")
    for t in TYPES:
        a.plot(caps, gb[t], color=COLOR[t], linewidth=2, label=TITLE[t])
    a.axhline(nocap, color=MUTED, linewidth=0.9, linestyle=(0, (4, 3)))
    a.text(caps[-1], nocap * 1.015, f"no cap: {nocap:.0f} GB", fontsize=8, color=INK2, va="bottom", ha="right")
    for cap, name in ((2 ** 17, "v1"), (cfg.cap, "v2")):
        y = float(np.interp(np.log2(cap), np.log2(caps), total))
        a.annotate(f"{name}: {measured[name]:.1f} GB measured", xy=(cap, y), xytext=(-8, 4), textcoords="offset points",
                   ha="right", va="bottom", fontsize=7.5, color=INK2)
    a.set_ylabel("shards [GB]", fontsize=9, color=INK2)
    a.set_title("Dataset size", loc="left", fontsize=11, color=INK)
    for t in TYPES:
        share = [100 * (index[f"n_{t}_{r}"] > c).mean() for c in caps]
        b.plot(caps, share, color=COLOR[t], linewidth=2, label=TITLE[t])
    b.set_ylabel("% of galaxies above the cap", fontsize=9, color=INK2)
    b.set_title("Galaxies whose particles are capped", loc="left", fontsize=11, color=INK)
    for ax in (a, b):
        leg = ax.legend(frameon=False, fontsize=8, loc="upper right" if ax is b else "upper left",
                        bbox_to_anchor=None if ax is b else (0.0, 0.88))
        for tx in leg.get_texts():
            tx.set_color(INK2)
    fig.tight_layout()
    fig.savefig(save / "cap_cost.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return {"caps": CAPS, "total_gb": total.tolist(), "no_cap_gb": nocap, **{f"{t}_gb": gb[t].tolist() for t in TYPES}}


def plot_accuracy(summary: dict, acc: dict, cfg, save: Path, plt) -> None:
    fig, axes = plt.subplots(1, len(RADII), figsize=(14, 4.5), facecolor=SURFACE, sharey=True)
    for ax, k in zip(axes, RADII):
        style(ax, plt)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        for mod in MODALITIES:
            pts = summary.get(mod, {}).get(k, {})
            if pts:
                caps = sorted(pts)
                ax.plot(caps, [pts[c][0] for c in caps], color=COLOR[mod], linewidth=2, label=TITLE[mod])
        ax.axhline(0.1, color=MUTED, linewidth=0.9, linestyle=(0, (1, 2)))
        ax.text(2 ** 12, 0.1 * 0.93, "0.1: adds 1% to one cloud's noise", fontsize=7.5, color=INK2, va="top")
        cap_lines(ax, cfg, ymax_frac=0.97)
        ax.set_title(f"inside {k:g} radii", loc="left", fontsize=11, color=INK)
        ax.set_xlabel("cap (particles kept per type and galaxy)", fontsize=9, color=INK2)
    axes[0].set_ylabel("error of the kept particles\n÷ spread between clouds of 2,048 points", fontsize=9, color=INK2)
    leg = axes[-1].legend(frameon=False, fontsize=8, loc="lower left")
    for tx in leg.get_texts():
        tx.set_color(INK2)
    fig.suptitle(f"How far the capped particles are from all particles: median over {len(acc['galaxies'])} capped "
                 f"galaxies read in full from the raw snapshot\neach statistic in units of how much it varies between "
                 f"clouds of {acc['budget']:,} points; a galaxy counts at a cap only if it has more particles than the cap",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(save / "cap_accuracy.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True, help="the dataset folder")
    ap.add_argument("--snap", type=int, default=99, help="snapshot of the dataset to study")
    ap.add_argument("--save", type=Path, required=True, help="folder for the pictures and numbers")
    ap.add_argument("--per-bin", type=int, default=20, help="capped galaxies read in full per size bin")
    ap.add_argument("--draws", type=int, default=16, help="clouds per galaxy and sphere for the spread")
    ap.add_argument("--seeds", type=int, default=3, help="capped samples per galaxy and cap")
    ap.add_argument("--root", type=Path, help="raw snapshot folder (default: the build's)")
    ap.add_argument("--workers", type=int, default=6, help="galaxies read in parallel")
    ap.add_argument("--recompute", action="store_true", help="read the galaxies again even if cap_accuracy.json exists")
    args = ap.parse_args()
    cfg = Config.load(snapshot_dir(args.out, args.snap) / "config.json")
    index = load_index(args.out, args.snap)
    args.save.mkdir(parents=True, exist_ok=True)
    path = args.save / "cap_accuracy.json"
    if args.recompute or not path.exists():
        acc = accuracy(args.out, cfg, index, args.per_bin, args.draws, args.seeds, args.root, args.workers)
        path.write_text(json.dumps(acc))
    acc = json.loads(path.read_text())
    summary = summarize(acc)

    stored = {t: index[f"n_{t}_stored"].sum() for t in TYPES}
    size = {t: 0 for t in TYPES}
    for shard in sorted(glob.glob(str(snapshot_dir(args.out, args.snap) / "shards" / "shard_*.h5"))):
        with h5py.File(shard, "r") as f:
            for t in TYPES:
                size[t] += sum(d.id.get_storage_size() for d in f[t].values())
    bytes_per = {t: size[t] / stored[t] for t in TYPES}
    measured = {"v2": sum(size.values()) / 1e9, "v1": 34.2}    # 2^18 (v2, rebuilt as v3) and 2^17 (v1, measured
                                                                 # before it was deleted)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plot_counts(index, cfg, args.save, plt)
    cost = plot_cost(index, cfg, bytes_per, measured, args.save, plt)
    plot_accuracy(summary, acc, cfg, args.save, plt)

    print("bytes per stored particle: " + ", ".join(f"{t} {bytes_per[t]:.1f}" for t in TYPES))
    print("cap          size GB   galaxies above the cap (stars / dm / gas)")
    for c, gbv in zip(CAPS, cost["total_gb"]):
        print(f"2^{int(np.log2(c)):<2} {c:>10,}  {gbv:7.1f}   " +
              " / ".join(f"{(index[f'n_{t}_r{cfg.extract_radius:g}'] > c).sum():,}" for t in TYPES))
    print(f"no cap        {cost['no_cap_gb']:7.1f}")
    print("error of the kept particles / spread between clouds (median, 90th percentile, galaxies):")
    for k in RADII:
        for mod in MODALITIES:
            pts = summary.get(mod, {}).get(k, {})
            print(f"  {k:>4g} radii {mod:5s} " + "  ".join(f"2^{int(np.log2(c))}: {pts[c][0]:.3f}/{pts[c][1]:.3f} ({pts[c][2]})"
                                                        for c in sorted(pts) if c >= 2 ** 14))
    print(f"wrote {args.save}/particle_counts.png, cap_cost.png, cap_accuracy.png")


if __name__ == "__main__":
    main()
