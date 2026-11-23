"""Build one snapshot's shards while its raw files stream through a small local folder (tools/stream_snapshot.py).

The builder reads each galaxy's bound particles: one contiguous block per particle type (extract.py). TNG splits every
particle type over the snapshot files on its own, so a galaxy's gas, dark matter and stars usually sit in files far
apart (snapshot 99: about 500 of its 680 files; 679 files hold selected particles). Blocks are therefore planned and
extracted per (galaxy, type): a block is extracted as soon as every file it spans is on disk, its result is saved in
OUT/snap_NNN/staging/, and a file is deleted once every block that spans it is saved. Fetching the files in order then
keeps only a few dozen of them on disk at a time (snapshot 99: at most 49 for the largest block, subhalo 0's stars).
After the last block the shards are written exactly as tools/build_pointclouds.py writes them, from the staged blocks.

File 0 is fetched first and kept until the end: Snapshot() reads the header from it.
"""
from __future__ import annotations

import json
import multiprocessing
import os
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from .config import Config
from .extract import FIELDS, READ, TYPES, extract_type
from .layout import shard_path, snapshot_dir
from .snapshot import Snapshot
from .store import write_shard


def subset_query(read: dict[str, list[str]] = READ) -> str:
    """The TNG API query that asks a snapshot file for only the fields the builder reads, e.g.
    'gas=Coordinates,Velocities,...&dm=Coordinates,Velocities&stars=...' (API particle names gas, dm, stars)."""
    return "&".join(f"{name}={','.join(fields)}" for name, fields in read.items())


@dataclass(frozen=True)
class Block:
    row: int            # row of the selection table
    name: str           # 'gas', 'dm' or 'stars'
    first: int          # first snapshot file the block spans
    last: int           # last snapshot file (first - 1 when the block is empty)


def plan_blocks(table: dict[str, np.ndarray], file_offsets: np.ndarray) -> list[Block]:
    """Every (galaxy, type) block of the selection and the snapshot files it spans (FileOffsets/SnapByType, (files, 6)).
    Snapshot.blocks opens every file from the first to the last, so a block needs that whole range."""
    blocks = []
    for name, t in TYPES.items():
        start = table["offsets"][:, t].astype(np.int64)
        n = table[f"n_{name}_bound"].astype(np.int64)
        first = np.searchsorted(file_offsets[:, t], start, side="right") - 1
        last = np.searchsorted(file_offsets[:, t], start + np.maximum(n, 1) - 1, side="right") - 1
        for r in range(len(start)):
            blocks.append(Block(r, name, int(first[r]), int(last[r]) if n[r] > 0 else int(first[r]) - 1))
    return blocks


def staging_dir(out: str | Path, snap: int) -> Path:
    return snapshot_dir(out, snap) / "staging"


def staging_path(out: str | Path, snap: int, subhalo: int, name: str) -> Path:
    return staging_dir(out, snap) / f"{int(subhalo):08d}_{name}.h5"


def write_block(path: Path, arrays: dict[str, np.ndarray], counts: dict) -> None:
    """One extracted block, written to a .part file and renamed, so a killed job never leaves half a block."""
    tmp = path.with_name(path.name + ".part")
    with h5py.File(tmp, "w") as f:
        for k, v in arrays.items():
            f.create_dataset(k, data=v)
        f.attrs["counts"] = json.dumps(counts)
    os.replace(tmp, path)


def read_block(path: Path, name: str) -> tuple[dict[str, np.ndarray], dict]:
    with h5py.File(path, "r") as f:
        arrays = {k: f[k][()] for k in FIELDS[name]}
        counts = json.loads(f.attrs["counts"])
    return arrays, counts


_SNAP: dict = {}


def _extract_block(job: tuple) -> tuple[int, str, float]:
    """Worker process: extract one block and stage it. One Snapshot per process (it only reads the header and
    offsets once)."""
    root, snap_num, cfg_json, galaxy, name, path = job
    key = (root, snap_num)
    if key not in _SNAP:
        _SNAP[key] = Snapshot(root, snap_num)
    t0 = time.time()
    cfg = Config.from_dict(json.loads(cfg_json))
    arrays, counts = extract_type(_SNAP[key], cfg, galaxy, name)
    write_block(Path(path), arrays, counts)
    return galaxy["row_"], name, time.time() - t0


class Streamer:
    """Fetch the files a snapshot's pending blocks need, in order, at most `max_raw_bytes` on disk; extract each block
    once its files are there; delete each file once its blocks are staged.

    fetch(j) -> bool puts file j into the raw folder (True on success); remove(j) deletes it (file 0 stays until the
    end); file_size(j) is the disk space file j takes once fetched (0 for a link), est_file_bytes the space a file
    being fetched is assumed to take. `stop` (threading.Event) ends the run early: running extractions finish,
    nothing new starts. `tune()` (optional), called every `poll_s` seconds with the progress update, returns how many
    files to fetch in parallel: more threads start at once, threads above a lower number stop after their current
    file."""

    def __init__(self, cfg: Config, table: dict[str, np.ndarray], file_offsets: np.ndarray, fetch, remove,
                 file_size, est_file_bytes: float, max_raw_bytes: float, fetch_workers: int, extract_workers: int,
                 stop: threading.Event, log=print, progress_path: Path | None = None,
                 max_running_rows: int = 120_000_000, tune=None, poll_s: float = 30.0):
        self.cfg, self.table, self.fo = cfg, table, file_offsets
        self.fetch, self.remove, self.file_size, self.est = fetch, remove, file_size, float(est_file_bytes)
        self.max_raw, self.n_fetch, self.n_extract = max_raw_bytes, int(fetch_workers), extract_workers
        self.stop, self.log, self.progress_path = stop, log, progress_path
        self.tune, self.poll_s = tune, float(poll_s)
        # memory: blocks running at once may read at most this many bound particles together (a larger block runs
        # alone); the largest galaxies' blocks are hundreds of millions of rows and sit in the first files
        self.max_running_rows = int(max_running_rows)
        out, snap = Path(cfg.out), int(cfg.snap)
        staging_dir(out, snap).mkdir(parents=True, exist_ok=True)
        blocks = plan_blocks(table, file_offsets)
        self.todo = [b for b in blocks
                     if not staging_path(out, snap, table["subhalo"][b.row], b.name).exists()]
        self.n_blocks, self.n_done0 = len(blocks), len(blocks) - len(self.todo)
        n_files = len(file_offsets)
        self.users: list[set] = [set() for _ in range(n_files)]     # pending blocks spanning each file
        for i, b in enumerate(self.todo):
            for j in range(b.first, b.last + 1):
                self.users[j].add(i)
        self.widest = max((b.last - b.first + 1 for b in self.todo), default=0)
        self.missing = [b.last - b.first + 1 for b in self.todo]
        self.need = sorted(j for j in range(n_files) if self.users[j]) if self.todo else []
        self.failed: str | None = None
        self.stats = {"fetched": 0, "fetched_bytes": 0, "deleted": 0, "extracted": 0, "t0": time.time()}

    # ------------------------------------------------------------------ bookkeeping shared by threads
    def write_progress(self, on_disk: set, inflight: set, state: str) -> None:
        if self.progress_path is None:
            return
        el = max(time.time() - self.stats["t0"], 1e-6)
        done = self.n_done0 + self.stats["extracted"]
        doc = {"snap": int(self.cfg.snap), "state": state, "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
               "blocks_total": self.n_blocks, "blocks_done": done, "files_needed": len(self.need),
               "files_fetched": self.stats["fetched"], "files_deleted": self.stats["deleted"],
               "files_on_disk": len(on_disk), "files_in_flight": len(inflight), "fetch_workers": self.n_fetch,
               "raw_bytes_on_disk": int(sum(self.file_size(j) for j in on_disk)),
               "fetched_bytes": int(self.stats["fetched_bytes"]), "elapsed_s": round(el),
               "speed_bytes_s": self.stats["fetched_bytes"] / el}
        tmp = self.progress_path.with_name(self.progress_path.name + ".tmp")
        tmp.write_text(json.dumps(doc, indent=1))
        os.replace(tmp, self.progress_path)

    def run(self) -> bool:
        """True when every block is staged; False when stopped or failed (self.failed says why)."""
        if not self.todo:
            return True
        out, snap = Path(self.cfg.out), int(self.cfg.snap)
        cfg_json = self.cfg.to_json()
        lock = threading.Lock()
        on_disk: set[int] = set()
        inflight: set[int] = set()
        queue_ = list(self.need)                         # files still to fetch, in order
        ready: list[int] = [i for i, m in enumerate(self.missing) if m == 0]   # blocks with no files (empty)
        cond = threading.Condition(lock)
        est = self.est
        if (self.widest + 1) * est > self.max_raw:      # +1: file 0 stays on disk throughout
            self.failed = (f"the widest block spans {self.widest} files (about {self.widest * est / 1e9:.0f} GB), "
                           f"more than the {self.max_raw / 1e9:.0f} GB allowed on disk: raise --max-raw-gb")
            return False

        def raw_bytes() -> float:
            return sum(self.file_size(j) for j in on_disk) + est * len(inflight)

        n_alive = [0]                                    # fetcher threads running; self.n_fetch is the target

        def fetcher():
            counted = True                               # this thread is still in n_alive
            try:
                while True:
                    with cond:
                        while (queue_ and not self.stop.is_set() and self.failed is None
                               and n_alive[0] <= self.n_fetch
                               and raw_bytes() + est > self.max_raw and (on_disk or inflight)):
                            cond.wait(timeout=5)
                        if (not queue_ or self.stop.is_set() or self.failed is not None
                                or n_alive[0] > self.n_fetch):    # nothing left, stopping, or one thread too many
                            n_alive[0] -= 1
                            counted = False
                            return
                        j = queue_.pop(0)
                        inflight.add(j)
                    try:
                        ok = self.fetch(j)
                    except Exception as e:               # a bug or an unexpected error: stop rather than hang
                        self.log(f"fetching file {j} raised {type(e).__name__}: {e}")
                        ok = False
                    with cond:
                        inflight.discard(j)
                        if not ok:
                            if not self.stop.is_set():
                                self.failed = self.failed or f"file {j} could not be fetched"
                            n_alive[0] -= 1
                            counted = False
                            cond.notify_all()
                            return
                        on_disk.add(j)
                        self.stats["fetched"] += 1
                        self.stats["fetched_bytes"] += self.file_size(j)
                        for i in self.users[j]:
                            self.missing[i] -= 1
                            if self.missing[i] == 0:
                                ready.append(i)
                        cond.notify_all()
            finally:
                if counted:                              # ended by an unexpected error
                    with cond:
                        n_alive[0] -= 1
                        cond.notify_all()

        threads: list[threading.Thread] = []

        def add_fetchers() -> None:
            """Start threads up to the target self.n_fetch (excess threads retire on their own)."""
            with cond:
                k = max(0, self.n_fetch - n_alive[0])
                n_alive[0] += k
            for _ in range(k):
                t = threading.Thread(target=fetcher, daemon=True)
                t.start()
                threads.append(t)

        add_fetchers()
        tick = min(5.0, self.poll_s)
        running = {}
        weight = [max(1, int(self.table[f"n_{b.name}_bound"][b.row])) for b in self.todo]
        running_w = 0
        remaining = len(self.todo)
        last_progress = 0.0
        # spawn, not fork: the fetcher threads are already running
        with ProcessPoolExecutor(self.n_extract, mp_context=multiprocessing.get_context("spawn")) as pool:
            while remaining:
                with cond:
                    k = 0
                    while k < len(ready) and len(running) < 2 * self.n_extract and not self.stop.is_set() \
                            and self.failed is None:
                        i = ready[k]
                        if running_w and running_w + weight[i] > self.max_running_rows:
                            k += 1                      # too heavy next to what runs: try a lighter one
                            continue
                        ready.pop(k)
                        running_w += weight[i]
                        b = self.todo[i]
                        galaxy = {k: v[b.row] for k, v in self.table.items()}
                        galaxy["row_"] = i
                        path = staging_path(out, snap, galaxy["subhalo"], b.name)
                        running[pool.submit(_extract_block, (self.cfg.root, snap, cfg_json, galaxy, b.name,
                                                             str(path)))] = i
                    stopping = self.stop.is_set() or self.failed is not None
                if stopping and not running:
                    break
                if not running:
                    with cond:
                        if not ready and not any(t.is_alive() for t in threads) and self.failed is None \
                                and not self.stop.is_set():
                            self.failed = f"{remaining} blocks left but no file is coming (internal error)"
                            continue
                        cond.wait(timeout=tick)
                else:
                    done, _ = wait(list(running), timeout=tick, return_when=FIRST_COMPLETED)
                    for fut in done:
                        i = running.pop(fut)
                        running_w -= weight[i]
                        try:
                            fut.result()
                        except Exception as e:          # a bad file or a bug: stop; the block is redone next run
                            b = self.todo[i]
                            self.failed = self.failed or (f"extracting subhalo {self.table['subhalo'][b.row]} "
                                                          f"{b.name} failed: {type(e).__name__}: {e}")
                            continue
                        remaining -= 1
                        self.stats["extracted"] += 1
                        b = self.todo[i]
                        with cond:
                            for j in range(b.first, b.last + 1):
                                self.users[j].discard(i)
                                if not self.users[j] and j != 0 and j in on_disk:
                                    self.remove(j)
                                    on_disk.discard(j)
                                    self.stats["deleted"] += 1
                            cond.notify_all()
                if time.time() - last_progress > self.poll_s:
                    last_progress = time.time()
                    if self.tune is not None:
                        try:
                            want = int(self.tune())
                        except Exception as e:           # a bad settings file must not end the run
                            self.log(f"reading new settings failed ({type(e).__name__}: {e}); keeping the old ones")
                            want = self.n_fetch
                        if want >= 1 and want != self.n_fetch:
                            self.log(f"parallel files {self.n_fetch} -> {want}")
                            with cond:
                                self.n_fetch = want
                                cond.notify_all()       # threads above the new number waiting for disk space retire
                            add_fetchers()
                    with cond:
                        self.write_progress(on_disk, inflight, "streaming")
                        self.log(f"blocks {self.n_done0 + self.stats['extracted']:,}/{self.n_blocks:,}  files fetched "
                                 f"{self.stats['fetched']}/{len(self.need)}  on disk {len(on_disk)} "
                                 f"({raw_bytes() / 1e9:.0f} GB)  deleted {self.stats['deleted']}  "
                                 f"fetching threads {n_alive[0]}")
        with cond:
            cond.notify_all()
        for t in threads:
            t.join(timeout=1)
        with cond:
            self.write_progress(on_disk, inflight, "staged" if remaining == 0 else "stopped")
        return remaining == 0


def assemble(cfg: Config, table: dict[str, np.ndarray], attrs: dict, log=print) -> int:
    """Write every missing shard of the selection's shard plan from the staged blocks, as
    tools/build_pointclouds.build() writes it. Returns the number of shards written."""
    out, snap = Path(cfg.out), int(cfg.snap)
    (snapshot_dir(out, snap) / "shards").mkdir(exist_ok=True)
    written = 0
    for s in range(int(table["shard"].max()) + 1):
        path = shard_path(out, snap, s)
        if path.exists():
            continue
        rows = np.flatnonzero(table["shard"] == s)
        particles, counts = [], {}
        for r in rows:
            p: dict = {"counts": {}}
            for name in TYPES:
                p[name], c = read_block(staging_path(out, snap, table["subhalo"][r], name), name)
                p["counts"].update(c)
            particles.append(p)
            for k, v in p["counts"].items():
                counts.setdefault(k, []).append(v)
        galaxies = {k: v[rows] for k, v in table.items()}
        galaxies.update({k: np.asarray(v) for k, v in counts.items()})
        galaxies["row"] = np.arange(len(rows), dtype=np.int32)
        write_shard(path, galaxies, particles, attrs)
        written += 1
    log(f"wrote {written} shards")
    return written
