"""Which galaxies at different snapshots are the same object, and the train/val/test split that follows from that.

A galaxy appears at every snapshot of the dataset: a progenitor at snapshot 91 and its descendant at snapshot 99 look
alike. If they sat in different splits, the test set would hold galaxies the model has in effect seen in training. So
splits are given to *split units* shared by all snapshots: the FoF halo at snapshot 99 (z = 0) in which a galaxy's
SubLink main descendant branch ends. At snapshot 99 a galaxy is its own descendant, so the unit is simply its halo,
the rule of the single-snapshot builds. A galaxy whose branch does not reach snapshot 99 falls back to its own halo at
its own snapshot (unit (snap, group)).

Branches come from the TNG API (sublink/simple.json, the main branch: descendants and progenitors as (snapshot,
subhalo) pairs), so linking needs no raw data of other snapshots. OUT/splits.h5 keeps every unit's split and snapshot
99's SubhaloGrNr, which turns a descendant into its halo after snapshot 99's raw data is gone. Units seen before keep
their split; new units are dealt like the first snapshot's halos (select.assign_splits): with the dataset seed for the
first snapshot of a dataset, with (seed, snapshot) for every later one.
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path

import h5py
import numpy as np

from .select import assign_splits

Z0_SNAP = 99                       # TNG50-1's last snapshot, z = 0
KEY_SCALE = 10 ** 10               # unit key = unit_snap * KEY_SCALE + unit_group (TNG50-1 has about 1e7 groups)


# ---------------------------------------------------------------- main branches (cache: OUT/snap_NNN/lineage.h5)
def read_branches(path: Path) -> dict[int, np.ndarray]:
    """subhalo -> (k, 2) int64 array of (snapshot, subhalo) along its main branch, increasing snapshot."""
    if not Path(path).exists():
        return {}
    with h5py.File(path, "r") as f:
        sub, off = f["subhalo"][()], f["offsets"][()]
        pairs = np.stack([f["branch_snap"][()].astype(np.int64), f["branch_subhalo"][()]], axis=1)
    return {int(s): pairs[off[i]:off[i + 1]] for i, s in enumerate(sub)}


def write_branches(path: Path, branches: dict[int, np.ndarray], attrs: dict) -> None:
    subs = np.array(sorted(branches), dtype=np.int64)
    parts = [branches[int(s)] for s in subs]
    pairs = np.concatenate(parts) if parts else np.empty((0, 2), np.int64)
    off = np.concatenate([[0], np.cumsum([len(p) for p in parts])]).astype(np.int64)
    tmp = Path(path).with_name(Path(path).name + ".part")
    with h5py.File(tmp, "w") as f:
        f.create_dataset("subhalo", data=subs)
        f.create_dataset("offsets", data=off)
        f.create_dataset("branch_snap", data=pairs[:, 0].astype(np.int16))
        f.create_dataset("branch_subhalo", data=pairs[:, 1].astype(np.int64))
        for k, v in attrs.items():
            f.attrs[k] = v
    os.replace(tmp, path)


def parse_simple(doc: dict) -> np.ndarray:
    """TNG's sublink/simple.json {'Main': [[snap, subhalo], ...], 'Mergers': ...} -> main branch, increasing snapshot."""
    main = np.asarray(doc.get("Main", []), dtype=np.int64).reshape(-1, 2)
    return main[np.argsort(main[:, 0], kind="stable")]


def fetch_branches(snap: int, subhalos, cache: Path, client, workers: int = 8, log=print) -> dict[int, np.ndarray]:
    """Main branches of the given subhalos at `snap`, from the cache or else the API; the cache is updated as results
    come in, so an interrupted run loses little. Subhalos whose requests all failed are missing from the result (and
    asked again next time); a subhalo the API has no tree for (HTTP 404) gets an empty branch."""
    branches = read_branches(cache)
    todo = [int(s) for s in subhalos if int(s) not in branches]
    attrs = {"snapshot": int(snap), "source": f"{client.api_base}/snapshots/{snap}/subhalos/<id>/sublink/simple.json",
             "fields": "Main branch: (snapshot, subhalo), descendants and progenitors"}
    if not todo:
        return branches
    log(f"fetching {len(todo):,} merger-tree branches from the TNG API with {workers} parallel requests "
        f"({len(branches):,} cached)")
    t0, done, failed = time.time(), 0, []

    def one(s: int):
        try:
            return s, parse_simple(client.get_json(f"snapshots/{snap}/subhalos/{s}/sublink/simple.json"))
        except Exception as e:                          # noqa: BLE001
            if getattr(e, "status", None) == 404:
                return s, np.empty((0, 2), dtype=np.int64)
            raise

    with ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(one, s) for s in todo]
        for fut in as_completed(futures):
            try:
                s, b = fut.result()
                branches[s] = b
            except Exception as e:                      # noqa: BLE001  (reported, retried on the next run)
                if getattr(e, "fatal", False):
                    for f in futures:
                        f.cancel()
                    raise
                failed.append(str(e))
            done += 1
            if done % 500 == 0 or done == len(todo):
                write_branches(cache, branches, {**attrs, "updated": time.strftime("%Y-%m-%d %H:%M:%S")})
                log(f"  {done:,}/{len(todo):,} in {time.time() - t0:.0f} s, {len(failed)} failed")
    if failed:
        log(f"{len(failed)} requests failed, e.g. {failed[0]}")
    return branches


def z0_descendants(snap: int, subhalos: np.ndarray, branches: dict[int, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """(subhalo at snapshot 99 or -1, whether the galaxy has a branch that contains itself) per galaxy."""
    desc = np.full(len(subhalos), -1, dtype=np.int64)
    ok = np.zeros(len(subhalos), dtype=bool)
    for i, s in enumerate(np.asarray(subhalos, dtype=np.int64)):
        b = branches.get(int(s))
        if b is None or not ((b[:, 0] == snap) & (b[:, 1] == s)).any():
            continue
        ok[i] = True
        at = b[b[:, 0] == Z0_SNAP, 1]
        if len(at):
            desc[i] = at[0]
    if snap == Z0_SNAP:                             # every galaxy at z = 0 is its own descendant, branch or not
        desc = np.asarray(subhalos, dtype=np.int64).copy()
    return desc, ok


# ---------------------------------------------------------------- split units and their splits (OUT/splits.h5)
def unit_keys(unit_snap: np.ndarray, unit_group: np.ndarray) -> np.ndarray:
    return np.asarray(unit_snap, dtype=np.int64) * KEY_SCALE + np.asarray(unit_group, dtype=np.int64)


def split_units(snap: int, group: np.ndarray, z0_subhalo: np.ndarray, z0_grnr: np.ndarray | None):
    """(unit_snap, unit_group, z0_group) per galaxy: the z = 0 halo of its descendant, else its own halo."""
    group = np.asarray(group, dtype=np.int64)
    if snap == Z0_SNAP:
        return np.full(len(group), Z0_SNAP, np.int16), group.copy(), group.copy()
    has = z0_subhalo >= 0
    z0_group = np.full(len(group), -1, dtype=np.int64)
    if has.any():
        if z0_grnr is None:
            raise ValueError("snapshot 99's SubhaloGrNr is needed to turn descendants into halos")
        z0_group[has] = z0_grnr[z0_subhalo[has]]
    unit_snap = np.where(has, Z0_SNAP, snap).astype(np.int16)
    unit_group = np.where(has, z0_group, group)
    return unit_snap, unit_group, z0_group


def load_splits(out: Path) -> tuple[dict[int, tuple[int, int]], np.ndarray | None]:
    """({unit key: (split code, snapshot that assigned it)}, snapshot 99's SubhaloGrNr or None)."""
    path = Path(out) / "splits.h5"
    if not path.exists():
        return {}, None
    with h5py.File(path, "r") as f:
        u = f["units"]
        keys = unit_keys(u["unit_snap"][()], u["unit_group"][()])
        reg = {int(k): (int(c), int(a)) for k, c, a in zip(keys, u["split"][()], u["assigned_at"][()])}
        z0 = f["z0/SubhaloGrNr"][()] if "z0/SubhaloGrNr" in f else None
    return reg, z0


def save_splits(out: Path, registry: dict[int, tuple[int, int]], z0_grnr: np.ndarray | None, splits_names) -> None:
    keys = np.array(sorted(registry), dtype=np.int64)
    tmp = Path(out) / "splits.h5.part"
    with h5py.File(tmp, "w") as f:
        u = f.create_group("units")
        u.create_dataset("unit_snap", data=(keys // KEY_SCALE).astype(np.int16))
        u.create_dataset("unit_group", data=keys % KEY_SCALE)
        u.create_dataset("split", data=np.array([registry[int(k)][0] for k in keys], dtype=np.int8))
        u.create_dataset("assigned_at", data=np.array([registry[int(k)][1] for k in keys], dtype=np.int16))
        if z0_grnr is not None:
            f.create_dataset("z0/SubhaloGrNr", data=np.asarray(z0_grnr, dtype=np.int32), compression="gzip")
        f.attrs["splits"] = ",".join(splits_names)
        f.attrs["about"] = ("split per split unit = FoF halo (unit_group) at snapshot unit_snap: the z = 0 halo a "
                            "galaxy's SubLink main descendant branch ends in, else its own halo; z0/SubhaloGrNr = "
                            "snapshot 99's subhalo -> halo")
    os.replace(tmp, Path(out) / "splits.h5")


def assign(snap: int, keys: np.ndarray, log_mstar: np.ndarray, registry: dict[int, tuple[int, int]], fractions,
           seed: int) -> tuple[np.ndarray, dict[int, tuple[int, int]], int]:
    """Split code per galaxy, the registry with this snapshot's new units added, and how many units were new."""
    first = not registry
    code = np.zeros(len(keys), dtype=np.int8)
    known = np.array([int(k) in registry for k in keys], dtype=bool)
    code[known] = [registry[int(k)][0] for k in keys[known]]
    new = ~known
    registry = dict(registry)
    n_new = 0
    if new.any():
        uniq, first_row, inverse = np.unique(keys[new], return_index=True, return_inverse=True)
        dealt = assign_splits(inverse.ravel(), log_mstar[new], fractions, seed if first else [seed, int(snap)])
        code[new] = dealt
        for k, r in zip(uniq, first_row):
            registry[int(k)] = (int(dealt[r]), int(snap))
        n_new = len(uniq)
    return code, registry, n_new


@contextmanager
def dataset_lock(out: Path):
    """One snapshot at a time may change OUT/splits.h5."""
    path = Path(out) / ".splits.lock"
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise SystemExit(f"{path} exists: another build_selection is running, or one was killed (then delete it)")
    try:
        os.write(fd, f"{os.uname().nodename} {os.getpid()} {time.strftime('%Y-%m-%d %H:%M:%S')}\n".encode())
        os.close(fd)
        yield
    finally:
        path.unlink(missing_ok=True)
