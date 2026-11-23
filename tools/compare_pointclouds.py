"""Compare one snapshot of two point-cloud datasets array by array (file attributes such as build times, hosts and
paths are ignored), e.g. a streamed build against the classic one:

    PYTHONPATH=pylibs python3 -s tools/compare_pointclouds.py --a .../tng50-1.pointclouds/v3 \\
        --b .../tng50-1.pointclouds/complete --snap 99

Compares the shared settings, snap_NNN/{selection,index,lineage}.h5, every shard and frozen set, and the dataset's
splits.h5, feature_stats.json and index.h5 (rows of that snapshot). Exit code 0 when everything is identical.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from galaxyclip3d.pointclouds.config import Config        # noqa: E402
from galaxyclip3d.pointclouds.layout import snapshot_dir  # noqa: E402

SKIP = {"pilot.h5", "targets.h5"}        # not part of a build: a pilot run, tools/build_targets.py


def arrays(path: Path) -> dict[str, np.ndarray]:
    out = {}
    with h5py.File(path, "r") as f:
        f.visititems(lambda k, v: out.__setitem__(k, v[()]) if isinstance(v, h5py.Dataset) else None)
    return out


def same_arrays(a: dict, b: dict) -> list[str]:
    bad = [f"only in one: {k}" for k in sorted(set(a) ^ set(b))]
    for k in sorted(set(a) & set(b)):
        x, y = a[k], b[k]
        if x.shape != y.shape or x.dtype != y.dtype or not np.array_equal(x, y, equal_nan=x.dtype.kind == "f"):
            bad.append(f"differs: {k} ({x.dtype}{x.shape} vs {y.dtype}{y.shape})")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--a", type=Path, required=True)
    ap.add_argument("--b", type=Path, required=True)
    ap.add_argument("--snap", type=int, required=True)
    ap.add_argument("--frozen-warn-only", action="store_true",
                    help="report differing frozen sets but do not fail on them (they are drawn from the shards by "
                         "finalize_pointclouds.py, whose code may have changed since the reference was drawn)")
    args = ap.parse_args()
    problems: list[str] = []
    warnings: list[str] = []
    ca, cb = Config.load(args.a / "config.json").shared(), Config.load(args.b / "config.json").shared()
    if ca != cb:
        problems.append(f"shared settings differ: {sorted(k for k in ca if ca[k] != cb.get(k))}")
    sa, sb = snapshot_dir(args.a, args.snap), snapshot_dir(args.b, args.snap)
    files = sorted(p.relative_to(sa) for p in sa.rglob("*.h5")
                   if "staging" not in p.parts and not p.name.endswith(".part") and p.name not in SKIP)
    n_compared = 0
    for rel in files:
        if not (sb / rel).exists():
            problems.append(f"missing in b: {rel}")
            continue
        found = [f"{rel}: {m}" for m in same_arrays(arrays(sa / rel), arrays(sb / rel))]
        (warnings if args.frozen_warn_only and rel.parts[0] == "frozen" else problems).extend(found)
        n_compared += 1
    extra = sorted(str(p.relative_to(sb)) for p in sb.rglob("*.h5")
                   if "staging" not in p.parts and not p.name.endswith(".part") and p.name not in SKIP
                   and not (sa / p.relative_to(sb)).exists())
    problems += [f"only in b: {p}" for p in extra]
    problems += [f"splits.h5: {m}" for m in same_arrays(arrays(args.a / "splits.h5"), arrays(args.b / "splits.h5"))]
    if json.loads((args.a / "feature_stats.json").read_text()) != json.loads((args.b / "feature_stats.json").read_text()):
        problems.append("feature_stats.json differs")
    ia, ib = arrays(args.a / "index.h5"), arrays(args.b / "index.h5")

    def rows_of(ix):
        key = next((k for k in ix if k.endswith("snapshot")), None)
        sel = ix[key] == args.snap if key else slice(None)
        return {k: v[sel] for k, v in ix.items() if len(v) == len(ix[key] if key else v)}
    problems += [f"index.h5: {m}" for m in same_arrays(rows_of(ia), rows_of(ib))]
    print(f"snapshot {args.snap}: compared {n_compared} files of {args.a} and {args.b}, plus splits.h5, "
          f"feature_stats.json and index.h5")
    for p in problems[:50]:
        print("  " + p)
    for w in warnings[:50]:
        print("  warning (frozen set): " + w)
    print(("IDENTICAL" if not problems else f"{len(problems)} DIFFERENCES")
          + (f" ({len(warnings)} differences in frozen sets, not counted)" if warnings else ""))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
