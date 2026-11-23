"""Add snapshots to a point-cloud dataset without ever storing them: each snapshot's files stream through the job's
local /scratch, are turned into point clouds as they arrive, and are deleted again (galaxyclip3d/pointclouds/stream.py).

    PYTHONPATH=pylibs python3 -s tools/stream_snapshot.py run --config configs/pc_complete.json --snap 91 84 78 67 50
    PYTHONPATH=pylibs python3 -s tools/stream_snapshot.py run --config configs/pc_complete.json --snap 99 \\
        --source local:/cluster/projects/ska/illustris/tng50-1.new --lineage-from .../tng50-1.pointclouds/v3
    python3 tools/stream_snapshot.py status --out DIR --snap 91          # progress, from any node
    PYTHONPATH=pylibs python3 -s tools/stream_snapshot.py probe --snap 99 --files 0 300 679   # API subsets vs local

Per snapshot (resumable: run it again after any interruption):
  1. quota check of /cluster/projects/ska (Ceph quota attributes; df shows the whole filesystem instead);
  2. group catalog and offsets into --catalog-root (default tng50-1.new) with download_snapshot.py --catalogs-only,
     kept afterwards (7.4 GB per snapshot);
  3. raw folder RAW/snapshots/NNN plus links to the catalog and offsets, so Snapshot() reads it unchanged;
  4. file 0 first (the header), then the selection with tools/build_selection.py;
  5. streaming: files fetched in order (--source api: only the fields the builder reads, about 37% of each file;
     --source local:DIR: links to a full local copy), each (galaxy, type) block extracted as soon as its files are
     there, each file deleted once its blocks are staged in OUT/snap_NNN/staging/;
  6. shards written from the staged blocks (same plan and content as tools/build_pointclouds.py), then the staging
     folder and the raw files are removed; tools/finalize_pointclouds.py and tools/check_pointclouds.py.

Download settings of a running job can be changed without a restart in OUT/stream_overrides.json, e.g.
{"fetch_workers": 50, "max_backoff": 120, "retries": 90}; they replace the command line's (see Overrides).

Exit codes: 0 done, 1 failed (run again), 2 cannot run (quota, settings, key), 3 stopped by a signal.
The API key comes from ~/.tng_api_key (mode 600) and is never printed; signed redirect URLs are not logged.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "pylibs"))   # numpy/h5py without PYTHONPATH
import h5py
import numpy as np

TOOLS = Path(__file__).resolve().parent
REPO = TOOLS.parent
sys.path[:0] = [str(REPO), str(TOOLS)]
from galaxyclip3d.pointclouds.config import Config                          # noqa: E402
from galaxyclip3d.pointclouds.extract import READ, TYPES                   # noqa: E402
from galaxyclip3d.pointclouds.layout import read_table, shard_path, snapshot_dir  # noqa: E402
from galaxyclip3d.pointclouds.stream import Streamer, assemble, staging_dir, subset_query  # noqa: E402

SKA = Path("/cluster/projects/ska")
CATALOG_ROOT = SKA / "illustris" / "tng50-1.new"
PROTECTED = [SKA / "illustris" / "tng50-1", SKA / "illustris" / "tng50-1.new"]   # raw folders never go here
EST_SUBSET_BYTES = 1.7e9          # a snapshot file with only the builder's fields (snapshot 99: 1.09 TB / 680)
OVERRIDES = "stream_overrides.json"
# settings OVERRIDES may change, with their type and allowed range
TUNABLE = {"fetch_workers": (int, 1, 100), "max_backoff": (float, 15, 3600), "retries": (int, 1, 1000),
           "timeout": (float, 30, 3600)}


def log(msg: str) -> None:
    sys.stdout.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}\n")
    sys.stdout.flush()


class Overrides:
    """Download settings in OUT/stream_overrides.json that replace the command line's, e.g.
    {"fetch_workers": 50, "max_backoff": 120, "retries": 90}. Read when a snapshot starts and every 30 s while it
    streams, so a running job takes a change without a restart: max_backoff and timeout count from the next attempt,
    retries from the next file, fetch_workers within a minute (threads above a lower number stop after their current
    file). A setting left out, or the whole file removed, means the command line's value again; an unreadable file
    or a value out of range is reported and ignored."""

    def __init__(self, path: Path, args):
        self.path, self.args = Path(path), args
        self.base = {k: getattr(args, k) for k in TUNABLE}      # the command line's values
        self.seen: int | None = -1                             # mtime of the file last applied; None: no file

    def read(self) -> dict | None:
        """The valid settings in the file ({} if there is none), or None if it cannot be read."""
        try:
            doc = json.loads(self.path.read_text())
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as e:
            log(f"{self.path.name}: cannot read it ({e}); settings unchanged")
            return None
        if not isinstance(doc, dict):
            log(f"{self.path.name}: not a JSON object; settings unchanged")
            return None
        values = {}
        for k, v in doc.items():
            if k not in TUNABLE:
                log(f"{self.path.name}: unknown setting {k!r} ignored (known: {', '.join(TUNABLE)})")
                continue
            kind, lo, hi = TUNABLE[k]
            if (isinstance(v, bool) or not isinstance(v, (int, float))
                    or (kind is int and not float(v).is_integer()) or not lo <= v <= hi):
                log(f"{self.path.name}: {k} = {v!r} ignored (needs a{'n integer' if kind is int else ' number'} "
                    f"from {lo:g} to {hi:g})")
                continue
            values[k] = kind(v)
        return values

    def poll(self) -> None:
        """Apply the file if it changed since the last look."""
        try:
            mtime: int | None = self.path.stat().st_mtime_ns
        except FileNotFoundError:
            mtime = None
        except OSError as e:
            log(f"{self.path.name}: cannot look at it ({e}); settings unchanged")
            return
        if mtime == self.seen:
            return
        values = self.read() if mtime is not None else {}
        self.seen = mtime
        if values is None:
            return
        changes = []
        for k in TUNABLE:
            new, old = values.get(k, self.base[k]), getattr(self.args, k)
            if new != old:
                setattr(self.args, k, new)
                changes.append(f"{k} {old:g} -> {new:g}")
        if changes:
            log((f"settings from {self.path}" if mtime is not None else f"{self.path} removed: command-line settings")
                + ": " + ", ".join(changes))

    def tune(self) -> int:
        """For Streamer(tune=...): apply any change; the number of files to fetch in parallel."""
        self.poll()
        return int(self.args.fetch_workers)


def quota_free(path: Path = SKA) -> float | None:
    """Free bytes under the Ceph directory quota of `path`, or None where there is none."""
    try:
        return float(int(os.getxattr(path, "ceph.quota.max_bytes")) - int(os.getxattr(path, "ceph.dir.rbytes")))
    except (OSError, ValueError):
        return None


def catalogs_complete(root: Path, snap: int) -> bool:
    gdir = root / "groupcats" / f"{snap:03d}"
    if not (root / "offsets" / f"offsets_{snap:03d}.hdf5").exists() or not gdir.is_dir():
        return False
    files = list(gdir.glob(f"fof_subhalo_tab_{snap:03d}.*.hdf5"))
    if not files or any(gdir.glob("*.part")):
        return False
    try:
        with h5py.File(gdir / f"fof_subhalo_tab_{snap:03d}.0.hdf5", "r") as f:
            n = f["Header"].attrs.get("NumFiles")
    except OSError:
        return False
    return n is None or len(files) == int(n)


def inside(path: Path, parent: Path) -> bool:
    path, parent = path.resolve(), parent.resolve()
    return path == parent or parent in path.parents


def prepare_raw(raw: Path, catalog_root: Path, snap: int) -> Path:
    """RAW/snapshots/NNN (real folder) with RAW/groupcats and RAW/offsets linking to the kept catalogs."""
    for p in PROTECTED:
        if inside(raw, p):
            sys.exit(f"refusing raw folder {raw}: it lies inside {p}, which holds raw data that is kept")
    sdir = raw / "snapshots" / f"{snap:03d}"
    sdir.mkdir(parents=True, exist_ok=True)
    for sub in ("groupcats", "offsets"):
        link, target = raw / sub, catalog_root / sub
        if link.is_symlink() or link.exists():
            if not link.is_symlink() or link.resolve() != target.resolve():
                sys.exit(f"{link} exists and is not a link to {target}")
        else:
            link.symlink_to(target)
    return sdir


class LocalSource:
    """Links to the files of a full local copy; removing a file removes only the link."""

    def __init__(self, src_root: Path, sdir: Path, snap: int):
        self.src = src_root / "snapshots" / f"{snap:03d}"
        self.sdir, self.snap, self.est = sdir, snap, 0.0
        self.describe = str(src_root)

    def path(self, j: int) -> Path:
        return self.sdir / f"snap_{self.snap:03d}.{j}.hdf5"

    def fetch(self, j: int) -> bool:
        link, target = self.path(j), self.src / f"snap_{self.snap:03d}.{j}.hdf5"
        if not target.exists():
            log(f"missing in the local copy: {target}")
            return False
        if not link.is_symlink():
            link.symlink_to(target)
        return True

    def size(self, j: int) -> float:
        return 0.0

    def remove(self, j: int) -> None:
        p = self.path(j)
        if p.is_symlink():                      # never the file it points to
            p.unlink()


class ApiSource:
    """Snapshot files from the TNG API with only the builder's fields, through download_snapshot.py's retrying
    downloader; every file is checked (particle counts per type against the offsets file, every field present)
    before it gets its final name."""

    def __init__(self, args, sdir: Path, raw: Path, snap: int, file_offsets: np.ndarray, redshift: float):
        import download_snapshot as dl
        self.dl, self.sdir, self.snap, self.fo, self.redshift = dl, sdir, snap, file_offsets, redshift
        self.est = EST_SUBSET_BYTES
        base = args.api_base.rstrip("/")
        self.query = subset_query()
        self.describe = f"{base}/files/snapshot-{snap}.<file>.hdf5?{self.query}"
        self.items = [dl.Item("snapshot", f"snap_{snap:03d}.{j}.hdf5", f"{base}/files/snapshot-{snap}.{j}.hdf5?"
                              f"{self.query}", str(sdir / f"snap_{snap:03d}.{j}.hdf5"), None)
                      for j in range(len(file_offsets))]
        dl.state_dir(raw, snap).mkdir(parents=True, exist_ok=True)
        self.progress = dl.Progress(raw, snap, self.items, args.fetch_workers)
        self.download = dl.Download(args, dl.load_key(), self.items, self.progress, check=self.check, resume=False)
        self.totals: np.ndarray | None = None

    def check(self, item, path: Path) -> None:
        j = int(item.name.split(".")[1])
        p0 = Path(self.items[0].path)
        if self.totals is None and j != 0 and p0.exists():     # file 0 fetched by an earlier pass of this job
            with h5py.File(p0, "r") as f0:
                h0 = f0["Header"].attrs
                self.totals = (h0["NumPart_Total"].astype(np.int64)
                               + (h0["NumPart_Total_HighWord"].astype(np.int64) << 32))
        with h5py.File(path, "r") as f:
            h = f["Header"].attrs
            if abs(float(h["Redshift"]) - self.redshift) > 1e-6 * (1 + self.redshift):
                raise ValueError(f"redshift {float(h['Redshift']):g}, expected {self.redshift:g}")
            if j == 0:
                self.totals = h["NumPart_Total"].astype(np.int64) + (h["NumPart_Total_HighWord"].astype(np.int64) << 32)
            npart = h["NumPart_ThisFile"]
            for name, t in TYPES.items():
                if j + 1 < len(self.fo):
                    expect = int(self.fo[j + 1, t] - self.fo[j, t])
                elif self.totals is not None:
                    expect = int(self.totals[t] - self.fo[j, t])
                else:
                    expect = int(npart[t])
                if int(npart[t]) != expect:
                    raise ValueError(f"{name}: {int(npart[t])} particles, the offsets file says {expect}")
                for fld in READ[name] if expect else ():
                    ds = f.get(f"PartType{t}/{fld}")
                    if ds is None or ds.shape[0] != expect:
                        raise ValueError(f"{name}/{fld} missing or of the wrong length")

    def path(self, j: int) -> Path:
        return Path(self.items[j].path)

    def fetch(self, j: int) -> bool:
        if self.path(j).exists():               # fetched and checked earlier in this job
            return True
        return self.download.get(self.items[j])

    def size(self, j: int) -> float:
        p = self.path(j)
        return float(p.stat().st_size) if p.exists() else 0.0

    def remove(self, j: int) -> None:
        p = self.path(j)
        if p.parent.resolve() != self.sdir.resolve():
            raise RuntimeError(f"refusing to delete {p}: not in {self.sdir}")
        p.unlink(missing_ok=True)
        Path(str(p) + ".part").unlink(missing_ok=True)


def clean_raw(sdir: Path, raw: Path, snap: int, local: bool) -> None:
    for p in sdir.iterdir():
        if p.is_symlink():
            p.unlink()
        elif local:
            raise RuntimeError(f"{p} is a real file in a local-mode raw folder; left alone")
        else:
            p.unlink()
    sdir.rmdir()
    shutil.rmtree(raw / ".download" / f"{snap:03d}", ignore_errors=True)


def run_tool(*cmd) -> int:
    log("running " + " ".join(str(c) for c in cmd))
    return subprocess.run([sys.executable, "-s", "-u", *map(str, cmd)]).returncode   # -u: progress shows at once


def stream_one(args, snap: int, stop: threading.Event, overrides: Overrides | None = None) -> int:
    from build_pointclouds import git_commit, load_snapshot_build
    cfg0 = Config.load(args.config)
    out = Path(args.out or cfg0.out)
    sdir_out = snapshot_dir(out, snap)
    log(f"=== snapshot {snap} -> {out}")
    if (sdir_out / "stream_done.json").exists():
        log(f"snapshot {snap} is already finished in {out}")
        return 0
    if overrides is not None:
        overrides.poll()
    free = quota_free()
    if free is not None:
        log(f"ska quota: {free / 1e9:,.0f} GB free")
        if free < args.min_free_gb * 1e9 and not (sdir_out / "shards").exists():
            log(f"less than {args.min_free_gb} GB free under the ska quota: not starting snapshot {snap}")
            return 2
    local = args.source.startswith("local:")
    catalog_root = Path(args.source[6:]) if local and not args.catalog_root else Path(args.catalog_root or CATALOG_ROOT)
    if not catalogs_complete(catalog_root, snap):
        if local:
            log(f"no complete group catalog and offsets for snapshot {snap} in {catalog_root}")
            return 2
        rc = subprocess.run([sys.executable, "-I", str(TOOLS / "download_snapshot.py"), "get", "--dest",
                             str(catalog_root), "--snap", str(snap), "--catalogs-only", "--workers",
                             str(args.catalog_workers)]).returncode
        if rc != 0 or not catalogs_complete(catalog_root, snap):
            log(f"group catalog download for snapshot {snap} ended with code {rc}")
            return 3 if rc == 3 else 1
    raw = Path(args.raw)
    sdir = prepare_raw(raw, catalog_root, snap)
    with h5py.File(raw / "offsets" / f"offsets_{snap:03d}.hdf5", "r") as f:
        fo = f["FileOffsets/SnapByType"][()]
    fo = fo.T if fo.shape[0] == 6 and fo.shape[1] != 6 else fo
    with h5py.File(raw / "groupcats" / f"{snap:03d}" / f"fof_subhalo_tab_{snap:03d}.0.hdf5", "r") as f:
        redshift = float(f["Header"].attrs["Redshift"])
    source = (LocalSource(Path(args.source[6:]), sdir, snap) if local
              else ApiSource(args, sdir, raw, snap, fo, redshift))
    if not local:
        source.download.stop = stop             # a signal also ends the running downloads
    if not source.fetch(0):
        log("file 0 could not be fetched")
        return 3 if stop.is_set() else 1

    if not (sdir_out / "selection.h5").exists():
        if args.lineage_from:
            src = snapshot_dir(args.lineage_from, snap) / "lineage.h5"
            if src.exists() and not (sdir_out / "lineage.h5").exists():
                sdir_out.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, sdir_out / "lineage.h5")
                log(f"merger-tree branches copied from {src}")
        rc = run_tool(TOOLS / "build_selection.py", "--config", args.config, "--snap", snap, "--root", raw,
                      "--out", out, *(["--branches", args.branches] if args.branches else []))
        if rc != 0:
            return 1
    cfg, table = load_snapshot_build(out, snap, str(raw))
    t0 = time.time()
    shards_done = all(shard_path(out, snap, s).exists() for s in range(int(table["shard"].max()) + 1))
    if not shards_done:
        if overrides is not None:
            overrides.poll()
        if not local:
            log(f"downloads: {args.fetch_workers} files in parallel, waits of up to {args.max_backoff:g} s between "
                f"attempts, {args.retries} attempts per file, {args.timeout:g} s timeout"
                + (f" (change them in {overrides.path})" if overrides is not None else ""))
        streamer = Streamer(cfg, table, fo, source.fetch, source.remove, source.size, source.est or 1.0,
                            args.max_raw_gb * 1e9, args.fetch_workers, args.extract_workers, stop, log,
                            sdir_out / "stream_progress.json",
                            tune=overrides.tune if overrides is not None else None)
        log(f"{streamer.n_blocks:,} blocks ({streamer.n_done0:,} staged before), {len(streamer.need)} files needed, "
            f"widest block {streamer.widest} files")
        if not streamer.run():
            log(f"streaming stopped: {streamer.failed or 'signal'}")
            return 3 if stop.is_set() and not streamer.failed else 1
        log(f"all blocks staged in {(time.time() - t0) / 3600:.2f} h; {streamer.stats['fetched']} files fetched "
            f"({streamer.stats['fetched_bytes'] / 1e9:,.0f} GB on disk), {streamer.stats['deleted']} deleted on the way")
        attrs = {"config": cfg.to_json(), "snapshot": int(snap), "redshift": redshift, "source": source.describe,
                 "git_commit": git_commit(), "host": socket.gethostname(),
                 "created": time.strftime("%Y-%m-%d %H:%M:%S"), "built_by": "tools/stream_snapshot.py"}
        assemble(cfg, table, attrs, log)
    n_rows = 0
    for s in range(int(table["shard"].max()) + 1):
        with h5py.File(shard_path(out, snap, s), "r") as f:
            n_rows += len(f["galaxies"]["subhalo"])
    if n_rows != len(table["subhalo"]):
        log(f"the shards hold {n_rows} galaxies, the selection {len(table['subhalo'])}: staging kept")
        return 1
    shutil.rmtree(staging_dir(out, snap), ignore_errors=True)
    clean_raw(sdir, raw, snap, local)
    log(f"shards complete ({n_rows:,} galaxies); staging and raw files removed")
    if not (sdir_out / "index.h5").exists():
        if run_tool(TOOLS / "finalize_pointclouds.py", "--out", out, "--snap", snap) != 0:
            return 1
    try:
        import matplotlib  # noqa: F401
        pdf = ["--pdf", sdir_out / "qa_clouds.pdf"]
    except ImportError:
        pdf = []
        log("no matplotlib here: quality checks without the PDF (draw it later with tools/check_pointclouds.py --pdf)")
    rc = run_tool(TOOLS / "check_pointclouds.py", "--out", out, "--snap", snap, *pdf)
    if rc != 0:
        log(f"check_pointclouds.py ended with code {rc}; see above")
        return 1
    (sdir_out / "stream_done.json").write_text(json.dumps({
        "snapshot": int(snap), "redshift": redshift, "galaxies": int(n_rows), "source": source.describe,
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"), "host": socket.gethostname()}, indent=1) + "\n")
    log(f"=== snapshot {snap} finished")
    return 0


def cmd_run(args) -> int:
    stop = threading.Event()

    def on_signal(signum, _frame):
        log(f"signal {signal.Signals(signum).name}: stopping after the running extractions")
        stop.set()

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(sig, on_signal)
    overrides = Overrides(Path(args.out or Config.load(args.config).out) / OVERRIDES, args)
    for snap in args.snap:
        rc = stream_one(args, snap, stop, overrides)
        if rc != 0:
            return rc
    return 0


def cmd_status(args) -> int:
    for snap in args.snap:
        sd = snapshot_dir(args.out, snap)
        p = sd / "stream_progress.json"
        n_shards = len(list((sd / "shards").glob("shard_*.h5"))) if (sd / "shards").exists() else 0
        done = (sd / "stream_done.json").exists()
        if not p.exists():
            print(f"snapshot {snap}: {'finished' if done else 'no streaming progress yet'}; {n_shards} shards")
            continue
        d = json.loads(p.read_text())
        left = d["files_needed"] - d["files_fetched"]
        speed = d["speed_bytes_s"]
        eta = left * EST_SUBSET_BYTES / speed / 3600 if speed > 0 else float("nan")
        print(f"snapshot {snap}: {d['state']} (updated {d['updated']}); blocks {d['blocks_done']:,}/{d['blocks_total']:,}; "
              f"files fetched {d['files_fetched']}/{d['files_needed']}"
              + (f" ({d['fetch_workers']} in parallel)" if "fetch_workers" in d else "")
              + f", on disk {d['files_on_disk']} "
              f"({d['raw_bytes_on_disk'] / 1e9:.0f} GB), deleted {d['files_deleted']}; "
              f"{speed / 1e6:.0f} MB/s; about {eta:.1f} h left; {n_shards} shards"
              + ("; finished" if done else ""))
    return 0


def cmd_probe(args) -> int:
    """Fetch a few files with only the builder's fields and compare every array with a full local copy."""
    snap, root = args.snap[0], Path(args.catalog_root or CATALOG_ROOT)
    raw = Path(args.raw) / "probe"
    sdir = prepare_raw(raw, root, snap)
    with h5py.File(root / "offsets" / f"offsets_{snap:03d}.hdf5", "r") as f:
        fo = f["FileOffsets/SnapByType"][()]
    fo = fo.T if fo.shape[0] == 6 and fo.shape[1] != 6 else fo
    with h5py.File(root / "snapshots" / f"{snap:03d}" / f"snap_{snap:03d}.0.hdf5", "r") as f:
        redshift = float(f["Header"].attrs["Redshift"])
    src = ApiSource(args, sdir, raw, snap, fo, redshift)
    ok_all = True
    for j in sorted(set([0] + list(args.files))):  # file 0 first: the check of the last file needs its totals
        t = time.time()
        ok = src.fetch(j)
        took = time.time() - t
        if not ok:
            log(f"file {j}: download failed")
            ok_all = False
            continue
        size = src.size(j)
        same = True
        with h5py.File(src.path(j), "r") as a, h5py.File(root / "snapshots" / f"{snap:03d}" /
                                                          f"snap_{snap:03d}.{j}.hdf5", "r") as b:
            for name, t_ in TYPES.items():
                for fld in READ[name]:
                    k = f"PartType{t_}/{fld}"
                    if (k in a) != (k in b) or (k in a and not np.array_equal(a[k][()], b[k][()])):
                        same = False
                        log(f"file {j}: {k} differs")
        full = (root / "snapshots" / f"{snap:03d}" / f"snap_{snap:03d}.{j}.hdf5").stat().st_size
        log(f"file {j}: {size / 1e9:.2f} GB ({size / full:.0%} of the full file) in {took:.0f} s "
            f"({size / max(took, 1e-6) / 1e6:.0f} MB/s incl. waiting); arrays identical: {same}")
        ok_all &= same
        src.remove(j)
    shutil.rmtree(raw, ignore_errors=True)
    log("probe " + ("passed" if ok_all else "FAILED"))
    return 0 if ok_all else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=["run", "status", "probe"])
    ap.add_argument("--config", help="dataset settings (run), e.g. configs/pc_complete.json")
    ap.add_argument("--out", help="dataset folder instead of the config's")
    ap.add_argument("--snap", type=int, nargs="+", required=True, help="snapshots, processed in this order")
    ap.add_argument("--source", default="api", help="'api' (TNG, only the needed fields) or 'local:DIR' (full copy)")
    ap.add_argument("--catalog-root", help=f"where group catalogs and offsets are kept (default {CATALOG_ROOT}, "
                                           "or the local copy with --source local:DIR)")
    ap.add_argument("--raw", default=f"/scratch/{os.environ.get('USER', 'user')}/tng-stream",
                    help="local folder the raw files pass through (job-private /scratch)")
    ap.add_argument("--lineage-from", help="dataset whose cached merger-tree branches (snap_NNN/lineage.h5) to reuse")
    ap.add_argument("--branches", help="JSON {subhalo: [[snap, subhalo], ...]} for build_selection.py instead of the "
                                       "TNG API (tests)")
    ap.add_argument("--files", type=int, nargs="+", default=[0, 300, 679], help="probe: files to compare")
    ap.add_argument("--fetch-workers", type=int, default=25, help="files downloaded in parallel")
    ap.add_argument("--catalog-workers", type=int, default=50, help="group-catalog files downloaded in parallel")
    ap.add_argument("--extract-workers", type=int, default=8, help="processes extracting blocks")
    ap.add_argument("--max-raw-gb", type=float, default=300, help="most raw data on local disk at once")
    ap.add_argument("--min-free-gb", type=float, default=120, help="free ska quota needed to start a snapshot")
    ap.add_argument("--retries", type=int, default=20, help="attempts per file")
    ap.add_argument("--max-backoff", type=float, default=900, help="longest wait between attempts, seconds")
    ap.add_argument("--timeout", type=float, default=600, help="seconds without data before an attempt fails (the "
                                                              "server builds reduced files on request)")
    ap.add_argument("--api-base", default="https://www.tng-project.org/api/TNG50-1")
    args = ap.parse_args()
    if args.command == "run":
        if not args.config:
            ap.error("run needs --config")
        return cmd_run(args)
    if args.command == "status":
        if not args.out:
            ap.error("status needs --out")
        return cmd_status(args)
    return cmd_probe(args)


if __name__ == "__main__":
    sys.exit(main())
