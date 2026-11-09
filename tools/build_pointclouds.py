"""Extract the selected galaxies' particles into shard files (step 2 of 3). Finished shards are skipped, so a build
can simply be run again after an interruption.

    PYTHONPATH=pylibs python3 -s tools/build_pointclouds.py --out DIR --shard 7           # one shard
    PYTHONPATH=pylibs python3 -s tools/build_pointclouds.py --out DIR --task 3 --tasks 42 # shards 3, 45, ... (one
                                                                                         # SLURM array task of 42)
    PYTHONPATH=pylibs python3 -s tools/build_pointclouds.py --out DIR --all --workers 8   # every missing shard
    PYTHONPATH=pylibs python3 -s tools/build_pointclouds.py --out DIR --pilot 20          # 20 galaxies across stellar
                                                                                         # mass, into DIR/pilot.h5

Needs OUT/config.json and OUT/selection.h5 from tools/build_selection.py.
"""
from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from galaxyclip3d.pointclouds.config import Config        # noqa: E402
from galaxyclip3d.pointclouds.extract import extract      # noqa: E402
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


def load_selection(out: Path) -> dict[str, np.ndarray]:
    with h5py.File(out / "selection.h5", "r") as f:
        return {k: v[()] for k, v in f["galaxies"].items()}


def build(out: Path, cfg: Config, table: dict[str, np.ndarray], rows: np.ndarray, path: Path, label: str) -> None:
    """Extract the galaxies at `rows` of the selection table and write them to one shard file."""
    snap = Snapshot(cfg.root, cfg.snap)
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
    write_shard(path, galaxies, particles, {"config": cfg.to_json(), "git_commit": git_commit(),
                                             "host": socket.gethostname(), "created": time.strftime("%Y-%m-%d %H:%M:%S")})
    print(f"{label} wrote {path.name}: {len(rows)} galaxies in {time.time() - t0:.0f} s", flush=True)


def shard_task(args) -> int:
    out, s = args
    cfg = Config.load(out / "config.json")
    table = load_selection(out)
    path = out / "shards" / f"shard_{s:04d}.h5"
    if path.exists():
        print(f"shard {s:04d} already done", flush=True)
        return s
    build(out, cfg, table, np.flatnonzero(table["shard"] == s), path, f"shard {s:04d}")
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
    ap.add_argument("--out", type=Path, required=True)
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
    out = args.out
    cfg = Config.load(out / "config.json")
    table = load_selection(out)
    (out / "shards").mkdir(exist_ok=True)
    if args.pilot:
        build(out, cfg, table, pilot_rows(table, args.pilot), out / "pilot.h5", "pilot")
    elif args.shard is not None:
        shard_task((out, args.shard))
    elif args.task is not None:
        for s in range(args.task, int(table["shard"].max()) + 1, args.tasks):
            shard_task((out, s))
    else:
        todo = [s for s in range(int(table["shard"].max()) + 1) if not (out / "shards" / f"shard_{s:04d}.h5").exists()]
        print(f"{len(todo)} shards to build with {args.workers} workers", flush=True)
        with Pool(args.workers) as pool:
            for s in pool.imap_unordered(shard_task, [(out, s) for s in todo]):
                pass


if __name__ == "__main__":
    main()
