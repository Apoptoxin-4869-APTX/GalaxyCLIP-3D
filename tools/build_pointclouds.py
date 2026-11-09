"""Extract one snapshot's selected galaxies into shard files (step 2 of 3). Finished shards are skipped, so a build
can simply be run again after an interruption.

    PYTHONPATH=pylibs python3 -s tools/build_pointclouds.py --out DIR --snap 99 --shard 7           # one shard
    PYTHONPATH=pylibs python3 -s tools/build_pointclouds.py --out DIR --snap 99 --task 3 --tasks 42 # shards 3, 45, ...
                                                                                                   # (array task 3/42)
    PYTHONPATH=pylibs python3 -s tools/build_pointclouds.py --out DIR --snap 99 --all --workers 8   # every missing one
    PYTHONPATH=pylibs python3 -s tools/build_pointclouds.py --out DIR --snap 99 --pilot 20          # 20 galaxies across
                                                                         # stellar mass, into DIR/snap_099/pilot.h5

Needs OUT/snap_NNN/config.json and selection.h5 from tools/build_selection.py; writes OUT/snap_NNN/shards. Every shard
records its snapshot and redshift (file attributes and the `snapshot`/`redshift` columns of its galaxy table).
"""
from __future__ import annotations

import argparse
import dataclasses
import socket
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from galaxyclip3d.pointclouds.config import Config        # noqa: E402
from galaxyclip3d.pointclouds.extract import extract      # noqa: E402
from galaxyclip3d.pointclouds.layout import read_table, shard_path, snapshot_dir  # noqa: E402
from galaxyclip3d.pointclouds.snapshot import Snapshot    # noqa: E402
from galaxyclip3d.pointclouds.store import write_shard    # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def git_commit() -> str:
    """Short commit of the code that built a shard, with +dirty when the working tree had uncommitted changes."""
    try:
        head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return (head or "unknown") + ("+dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def load_snapshot_build(out: Path, snap: int, root: str | None = None) -> tuple[Config, dict[str, np.ndarray]]:
    """The snapshot's build settings (with another raw-data root if given) and its selection table."""
    sdir = snapshot_dir(out, snap)
    if not (sdir / "selection.h5").exists():
        sys.exit(f"no selection for snapshot {snap} in {out}; run tools/build_selection.py --snap {snap} first")
    cfg = Config.load(sdir / "config.json")
    return (dataclasses.replace(cfg, root=root) if root else cfg), read_table(sdir / "selection.h5")


def build(cfg: Config, table: dict[str, np.ndarray], rows: np.ndarray, path: Path, label: str) -> None:
    """Extract the galaxies at `rows` of the selection table and write them to one shard file."""
    snap = Snapshot(cfg.root, cfg.snap)
    if abs(snap.redshift - float(table["redshift"][0])) > 1e-6:
        sys.exit(f"{cfg.root} holds snapshot {cfg.snap} at z = {snap.redshift:g}, the selection says "
                 f"z = {float(table['redshift'][0]):g}")
    t0 = time.time()
    particles, counts = [], {}
    for i, r in enumerate(rows, 1):
        galaxy = {k: v[r] for k, v in table.items()}
        t = time.time()
        p = extract(snap, cfg, galaxy)
        particles.append(p)
        for k, v in p["counts"].items():
            counts.setdefault(k, []).append(v)
        print(f"{label} [{i:3d}/{len(rows)}] subhalo {galaxy['subhalo']:>7d}  log M* {np.log10(galaxy['mstar']):5.2f}  "
              f"stored stars {p['counts']['n_stars_stored']:>7,} dm {p['counts']['n_dm_stored']:>7,} "
              f"gas {p['counts']['n_gas_stored']:>7,}  {time.time() - t:6.1f} s", flush=True)
    galaxies = {k: v[rows] for k, v in table.items()}
    galaxies.update({k: np.asarray(v) for k, v in counts.items()})
    galaxies["row"] = np.arange(len(rows), dtype=np.int32)
    write_shard(path, galaxies, particles, {"config": cfg.to_json(), "snapshot": int(cfg.snap),
                                             "redshift": snap.redshift, "source": str(cfg.root),
                                             "git_commit": git_commit(), "host": socket.gethostname(),
                                             "created": time.strftime("%Y-%m-%d %H:%M:%S")})
    print(f"{label} wrote {path.name}: {len(rows)} galaxies in {time.time() - t0:.0f} s", flush=True)


def shard_task(args) -> int:
    out, snap, root, s = args
    path = shard_path(out, snap, s)
    if path.exists():
        print(f"shard {s:04d} already done", flush=True)
        return s
    cfg, table = load_snapshot_build(out, snap, root)
    build(cfg, table, np.flatnonzero(table["shard"] == s), path, f"snap {snap} shard {s:04d}")
    return s


def pilot_rows(table: dict[str, np.ndarray], n: int) -> np.ndarray:
    """n galaxies spread evenly over log stellar mass, alternating centrals and satellites where possible."""
    lm = np.log10(table["mstar"])
    targets = np.linspace(np.percentile(lm, 1), lm.max(), n)
    rows = []
    for i, x in enumerate(targets):
        want_central = i % 2 == 0
        pool = np.flatnonzero(table["central"] == want_central) if i < n - 1 else np.arange(len(lm))
        pool = np.setdiff1d(pool, rows)
        rows.append(int(pool[np.argmin(np.abs(lm[pool] - x))]))
    return np.array(sorted(rows))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True, help="dataset folder")
    ap.add_argument("--snap", type=int, required=True, help="snapshot whose shards to build")
    ap.add_argument("--root", help="read the raw snapshot from here instead of the build's root")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--shard", type=int)
    g.add_argument("--task", type=int, help="with --tasks: build the shards s with s % tasks == task")
    g.add_argument("--all", action="store_true")
    g.add_argument("--pilot", type=int, metavar="N")
    ap.add_argument("--tasks", type=int, help="number of array tasks sharing the shards (with --task)")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    if (args.task is None) != (args.tasks is None):
        ap.error("--task and --tasks go together")
    out, snap = args.out, args.snap
    cfg, table = load_snapshot_build(out, snap, args.root)
    (snapshot_dir(out, snap) / "shards").mkdir(exist_ok=True)
    if args.pilot:
        build(cfg, table, pilot_rows(table, args.pilot), snapshot_dir(out, snap) / "pilot.h5", "pilot")
    elif args.shard is not None:
        shard_task((out, snap, args.root, args.shard))
    elif args.task is not None:
        for s in range(args.task, int(table["shard"].max()) + 1, args.tasks):
            shard_task((out, snap, args.root, s))
    else:
        todo = [s for s in range(int(table["shard"].max()) + 1) if not shard_path(out, snap, s).exists()]
        print(f"{len(todo)} shards to build with {args.workers} workers", flush=True)
        with Pool(args.workers) as pool:
            for s in pool.imap_unordered(shard_task, [(out, snap, args.root, s) for s in todo]):
                pass


if __name__ == "__main__":
    main()
