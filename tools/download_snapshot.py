"""Download a fresh, independent copy of a TNG snapshot, resumably and with a progress view.

    python3 tools/download_snapshot.py check  --dest DEST            # test the key and the server
    python3 tools/download_snapshot.py get    --dest DEST            # download; safe to restart any time
    python3 tools/download_snapshot.py status --dest DEST --watch 10 # progress, from any node, no key needed
    python3 tools/download_snapshot.py verify --dest DEST            # re-hash the finished files

A file gets its final name only after its size and sha256 have been checked. Until then it is
`<name>.part`, and a restart continues it, so an interrupted job can simply be run again. (The
existing copy's truncated snap_099.657 came from a downloader without these checks.)

The data servers often answer 403 or 504 to a request that works a moment later, with the same key.
These are retried, each time with a fresh redirect from the API; only the API refusing the key stops
the job.

Files go to DEST/snapshots/099/, the same layout as the existing copy, so tools/ can read it with
--root DEST once the group catalog and offsets are there too (--with-catalogs). DEST may not be
inside the existing copy, which is only ever read: its checksums.txt files are copied from it.

The API key comes from ~/.tng_api_key (mode 600) or the TNG_API_KEY environment variable, never
from the command line, and is not written to logs or status files. It is only sent to the API's
own hosts. Needs nothing beyond the standard library, so /usr/bin/python3 works on any node.
"""
from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
import http.client
import json
import os
import queue
import random
import re
import shutil
import signal
import socket
import stat
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlsplit

API_BASE = "https://www.tng-project.org/api/TNG50-1"
EXISTING = Path("/cluster/projects/ska/illustris/tng50-1")      # another copy: only ever read
KEY_FILE = Path.home() / ".tng_api_key"
CHUNK = 4 * 1024 ** 2
TYPICAL_SIZE = {"snapshot": 4.3e9, "groupcat": 9.3e6, "offsets": 1.1e9}   # bytes, until the real size is known
FATAL_DISK = {errno.ENOSPC, errno.EDQUOT, errno.EACCES, errno.EROFS, errno.EPERM}
HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"
USER_AGENT = "GalaxyCLIP-3D download_snapshot"


class Fatal(Exception):
    """Stops the whole download: retrying cannot help (rejected key, full disk)."""


class Retry(Exception):
    """This attempt failed; try the file again."""

    def __init__(self, message, delay=None):
        super().__init__(message)
        self.delay = delay


class Permanent(Exception):
    """This file cannot be downloaded; carry on with the others."""


class Stopped(Exception):
    """The job was told to stop; the .part file stays for the next run."""


@dataclass
class Item:
    kind: str
    name: str
    url: str
    path: str
    sha256: str | None

    @property
    def part(self) -> Path:
        return Path(self.path + ".part")


# ---------------------------------------------------------------- files and state
def state_dir(dest: Path) -> Path:
    return dest / ".download"


def check_dest(dest: Path, snap: int) -> None:
    """Refuse any destination that would write into the existing copy."""
    existing = EXISTING.resolve()
    for target in (dest / "snapshots" / f"{snap:03d}", dest / "groupcats" / f"{snap:03d}", dest / "offsets"):
        target = target.resolve()
        if target == existing or existing in target.parents:
            sys.exit(f"refusing to write inside the existing copy {existing}; choose another --dest")


def read_checksums(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            digest, name = line.split()
            out[name.lstrip("*")] = digest
    return out


def checksums_for(subdir: Path, source: Path, no_checksums: bool) -> dict[str, str]:
    """The copy's own checksums.txt, created from `source` (read-only) on the first run."""
    own = subdir / "checksums.txt"
    if no_checksums:
        return {}
    if not own.exists():
        if not source.exists():
            sys.exit(f"no checksum list at {source}; pass --no-checksums to rely on file sizes only")
        subdir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, own)
    return read_checksums(own)


def build_items(args, meta: dict) -> list[Item]:
    dest, snap, base = args.dest, args.snap, args.api_base.rstrip("/")
    snap_dir, gc_dir = dest / "snapshots" / f"{snap:03d}", dest / "groupcats" / f"{snap:03d}"
    items: list[Item] = []

    def add_chunks(kind, api_name, disk_name, subdir, sums, count):
        if sums and count and len(sums) != count:
            sys.exit(f"{kind}: checksum list has {len(sums)} files but the API says {count}")
        if not (sums or count):
            sys.exit(f"{kind}: unknown number of files (no checksum list, and the API gave no count)")
        for i in range(count or len(sums)):
            name = f"{disk_name}_{snap:03d}.{i}.hdf5"
            digest = sums.get(name)
            if sums and digest is None:
                sys.exit(f"{name} is not in the {kind} checksum list")
            items.append(Item(kind, name, f"{base}/files/{api_name}-{snap}.{i}.hdf5", str(subdir / name), digest))

    sums = checksums_for(snap_dir, Path(args.checksums), args.no_checksums)
    add_chunks("snapshot", "snapshot", "snap", snap_dir, sums, meta.get("num_files_snapshot"))
    if args.with_catalogs:      # last: 681 small files, whose speed says nothing about the time left for the snapshot
        sums = checksums_for(gc_dir, Path(args.checksums).parents[2] / "groupcats" / f"{snap:03d}" / "checksums.txt",
                             args.no_checksums)
        add_chunks("groupcat", "groupcat", "fof_subhalo_tab", gc_dir, sums, meta.get("num_files_groupcat"))
        name = f"offsets_{snap:03d}.hdf5"
        items.append(Item("offsets", name, f"{base}/files/offsets.{snap}.hdf5", str(dest / "offsets" / name), None))
    return items


def write_json(path: Path, obj) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


def scan(items: list[Item], sizes: dict[str, int]) -> tuple[int, float, float]:
    """(finished files, bytes on disk, estimated total bytes), from the files themselves."""
    done, on_disk, total = 0, 0.0, 0.0
    for it in items:
        try:
            size = os.stat(it.path).st_size
            done += 1
            on_disk += size
            total += size
            continue
        except FileNotFoundError:
            pass
        try:
            on_disk += os.stat(it.part).st_size
        except FileNotFoundError:
            pass
        total += sizes.get(it.name, TYPICAL_SIZE[it.kind])
    return done, on_disk, total


def human_bytes(n: float) -> str:
    for unit, scale in (("TB", 1e12), ("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if abs(n) >= scale:
            return f"{n / scale:.2f} {unit}"
    return f"{n:.0f} B"


def human_time(seconds: float) -> str:
    seconds = int(seconds)
    if seconds >= 86400:
        return f"{seconds // 86400} d {seconds % 86400 // 3600} h"
    if seconds >= 3600:
        return f"{seconds // 3600} h {seconds % 3600 // 60} min"
    if seconds >= 60:
        return f"{seconds // 60} min {seconds % 60} s"
    return f"{seconds} s"


# ---------------------------------------------------------------- API key and HTTP
def load_key() -> str:
    env = os.environ.get("TNG_API_KEY", "").strip()
    if env:
        return env
    if not KEY_FILE.exists():
        sys.exit(f"no API key: put it in {KEY_FILE} (chmod 600) or set TNG_API_KEY")
    if stat.S_IMODE(KEY_FILE.stat().st_mode) & 0o077:
        sys.exit(f"{KEY_FILE} is readable by others; run: chmod 600 {KEY_FILE}")
    key = KEY_FILE.read_text().strip()
    if not key:
        sys.exit(f"{KEY_FILE} is empty")
    return key


class KeepKeyOnApiHosts(urllib.request.HTTPRedirectHandler):
    """The API redirects downloads to its data servers; the key must not follow to anyone else."""

    def __init__(self, api_base: str):
        self.api_host = urlsplit(api_base).hostname or ""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        host = urlsplit(newurl).hostname or ""
        if new is not None and host != self.api_host and not host.endswith(".tng-project.org"):
            new.remove_header("Api-key")
        return new


def make_opener(api_base: str):
    return urllib.request.build_opener(KeepKeyOnApiHosts(api_base))


def public(url: str) -> str:
    """A URL without its query string, which holds the data server's signature."""
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}{parts.path}"


def quick_backoff(attempt: int, args) -> float:
    """Wait between the few attempts of a single start-up request, seconds."""
    return min(args.max_backoff, 60, 5 * 2 ** (attempt - 1))


def get_json(opener, url: str, key: str, args, tries: int = 6) -> dict:
    """A short outage must not end an unattended job at the moment SLURM restarts it, so retry a few times."""
    req = urllib.request.Request(url, headers={"api-key": key, "Accept": "application/json", "User-Agent": USER_AGENT})
    for attempt in range(1, tries + 1):
        try:
            with opener.open(req, timeout=args.timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                sys.exit(f"the API rejected the key (HTTP {e.code} from {public(url)})")
            if e.code < 500:
                sys.exit(f"HTTP {e.code} from {public(url)}")
            error = f"HTTP {e.code}"
        except (OSError, http.client.HTTPException, ValueError) as e:
            error = str(e) or type(e).__name__
        if attempt < tries:
            print(f"{public(url)}: {error}; attempt {attempt} of {tries}", flush=True)
            time.sleep(quick_backoff(attempt, args))
    sys.exit(f"could not read {public(url)} after {tries} attempts: {error}")


def open_range(opener, url: str, key: str, first: int, last: int, args, tries: int = 8):
    """Open a byte range for `check`, retrying what `get` retries: (response, seconds until the answer)."""
    headers = {"api-key": key, "User-Agent": USER_AGENT, "Range": f"bytes={first}-{last}"}
    for attempt in range(1, tries + 1):
        t0 = time.time()
        try:
            return opener.open(urllib.request.Request(url, headers=headers), timeout=args.timeout), time.time() - t0
        except urllib.error.HTTPError as e:
            if e.code in (401, 403) and "/api/" in urlsplit(e.url or url).path:
                sys.exit(f"the API refused the file request (HTTP {e.code} from {public(e.url or url)})")
            error = f"HTTP {e.code} from {urlsplit(e.url or url).hostname}"
        except (OSError, http.client.HTTPException) as e:
            error = str(e) or type(e).__name__
        print(f"  attempt {attempt} of {tries} refused after {time.time() - t0:.0f} s: {error}", flush=True)
        if attempt < tries:
            time.sleep(quick_backoff(attempt, args))
    sys.exit(f"the server refused all {tries} attempts; last answer: {error}")


def expected_total(resp, offset: int) -> int | None:
    """Full file size from a 200 or 206 response; complains if a 206 starts at the wrong place."""
    if resp.status == 206:
        m = re.match(r"bytes (\d+)-\d+/(\d+|\*)", resp.headers.get("Content-Range", ""))
        if not m or int(m.group(1)) != offset:
            raise Retry(f"server answered with the wrong range: {resp.headers.get('Content-Range')!r}")
        return None if m.group(2) == "*" else int(m.group(2))
    length = resp.headers.get("Content-Length")
    return int(length) if length else None


# ---------------------------------------------------------------- progress
class Progress:
    """Thread-safe counters, written to DEST/.download/progress.json for `status`."""

    def __init__(self, dest: Path, items: list[Item], workers: int):
        self.path = state_dir(dest) / "progress.json"
        self.items = items
        self.lock = threading.Lock()
        self.active: dict[str, dict] = {}
        self.failed: dict[str, str] = {}
        self.sizes: dict[str, int] = {}
        self.retries = 0
        self.received = 0
        self.samples = deque([(time.time(), 0)])
        self.info = {"started": time.time(), "host": socket.gethostname(), "pid": os.getpid(), "workers": workers,
                     "job": os.environ.get("SLURM_JOB_ID")}
        if self.path.exists():              # sizes learned by earlier runs make the total more exact
            try:
                self.sizes.update(json.loads(self.path.read_text()).get("sizes", {}))
            except (OSError, ValueError):
                pass

    def begin(self, item: Item, state: str, done: int = 0) -> None:
        with self.lock:
            self.active[item.name] = {"state": state, "done": done, "total": self.sizes.get(item.name), "attempt": 1,
                                      "t0": time.time(), "got": 0, "error": None, "resume_at": None}

    def set(self, item: Item, **fields) -> None:
        with self.lock:
            if fields.get("total") is None:
                fields.pop("total", None)
            else:
                self.sizes[item.name] = fields["total"]
            self.active[item.name].update(fields)

    def add(self, item: Item, n: int) -> None:
        with self.lock:
            a = self.active[item.name]
            a["done"] += n
            a["got"] += n
            self.received += n

    def retry(self, item: Item, attempt: int, error: str, delay: float) -> None:
        with self.lock:
            self.retries += 1
            self.active[item.name].update(state="waiting to retry", attempt=attempt + 1, error=error,
                                          resume_at=time.time() + delay, t0=time.time() + delay, got=0)

    def finish(self, item: Item) -> dict:
        with self.lock:
            return self.active.pop(item.name)

    def fail(self, item: Item, reason: str) -> None:
        with self.lock:
            self.active.pop(item.name, None)
            self.failed[item.name] = reason

    def speed(self, window: float = 300.0) -> float:
        """Bytes per second received over the last `window` seconds."""
        with self.lock:
            now = time.time()
            self.samples.append((now, self.received))
            while len(self.samples) > 2 and now - self.samples[1][0] > window:
                self.samples.popleft()
            t0, r0 = self.samples[0]
            return (self.received - r0) / max(now - t0, 1e-6)

    def write(self, finished: bool = False, outcome: str | None = None) -> None:
        speed = self.speed()
        with self.lock:
            now = time.time()
            active = [{"file": n, "state": a["state"], "done": a["done"], "total": a["total"], "attempt": a["attempt"],
                       "error": a["error"], "speed": a["got"] / max(now - a["t0"], 1e-6) if now > a["t0"] else 0.0,
                       "retry_in": max(a["resume_at"] - now, 0) if a["state"] == "waiting to retry" else None}
                      for n, a in sorted(self.active.items())]
            write_json(self.path, {**self.info, "updated": now, "speed": speed, "active": active, "sizes": self.sizes,
                                   "failed": self.failed, "retries": self.retries, "finished": finished,
                                   "outcome": outcome})


# ---------------------------------------------------------------- download
class Download:
    def __init__(self, args, key: str, items: list[Item], progress: Progress):
        self.args, self.key, self.items, self.progress = args, key, items, progress
        self.opener = make_opener(args.api_base)
        self.todo: queue.Queue[Item] = queue.Queue()
        self.stop = threading.Event()
        self.fatal: str | None = None

    def hash_partial(self, item: Item, digest) -> None:
        """Hash what is already on disk: it may have been written by a job that was killed."""
        with open(item.part, "rb") as f:
            while block := f.read(CHUNK):
                if self.stop.is_set():
                    raise Stopped
                digest.update(block)

    def finalise(self, item: Item, digest, total: int | None) -> None:
        size = item.part.stat().st_size
        if total is not None and size != total:
            raise Retry(f"incomplete: {size} of {total} bytes")
        if item.sha256 and digest.hexdigest() != item.sha256:
            item.part.unlink()
            raise Retry("sha256 mismatch; starting this file again")
        if not item.sha256:                 # no checksum to catch an error page served as the file
            with open(item.part, "rb") as f:
                head = f.read(len(HDF5_SIGNATURE))
            if head != HDF5_SIGNATURE:
                item.part.unlink()
                raise Retry("not an HDF5 file; starting this file again")
        os.replace(item.part, item.path)

    def fetch(self, item: Item) -> None:
        offset = item.part.stat().st_size if item.part.exists() else 0
        known = self.progress.sizes.get(item.name)
        digest = hashlib.sha256()
        if offset:
            self.progress.set(item, state="checking partial file", done=offset)
            if item.sha256:
                self.hash_partial(item, digest)
            if known == offset:             # killed between the last byte and the rename
                self.finalise(item, digest, known)
                return

        headers = {"api-key": self.key, "User-Agent": USER_AGENT}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        req = urllib.request.Request(item.url, headers=headers)
        try:
            resp = self.opener.open(req, timeout=self.args.timeout)
        except urllib.error.HTTPError as e:
            if e.code == 416 and offset:    # the partial file is complete, or longer than the file
                m = re.match(r"bytes \*/(\d+)", e.headers.get("Content-Range", ""))
                if m and int(m.group(1)) == offset:
                    self.finalise(item, digest, offset)
                    return
                item.part.unlink()
                raise Retry("partial file does not fit the server's file; starting again") from e
            if e.code in (401, 403) and "/api/" in urlsplit(e.url or item.url).path:
                raise Fatal(f"the API rejected the key (HTTP {e.code})") from e
            if e.code == 404:
                raise Permanent("not on the server (HTTP 404)") from e
            after = e.headers.get("Retry-After", "")
            host = urlsplit(e.url or item.url).hostname
            raise Retry(f"HTTP {e.code} from {host}", delay=float(after) if after.isdigit() else None) from e

        with resp:
            if offset and resp.status == 200:   # the server ignored Range: all over again
                offset, digest = 0, hashlib.sha256()
            total = expected_total(resp, offset)
            self.progress.set(item, state="downloading", done=offset, total=total, got=0, t0=time.time(), error=None)
            with open(item.part, "r+b" if offset else "wb") as f:
                f.seek(offset)
                while True:
                    if self.stop.is_set():
                        f.flush()
                        raise Stopped
                    block = resp.read(CHUNK)
                    if not block:
                        break
                    f.write(block)
                    digest.update(block)
                    self.progress.add(item, len(block))
                f.flush()
                os.fsync(f.fileno())
        self.finalise(item, digest, total)

    def work(self) -> None:
        while not self.stop.is_set():
            try:
                item = self.todo.get_nowait()
            except queue.Empty:
                return
            self.progress.begin(item, "starting")
            for attempt in range(1, self.args.retries + 1):
                try:
                    self.fetch(item)
                    a = self.progress.finish(item)
                    took = max(time.time() - a["t0"], 1e-6)
                    log(f"finished {item.name}  {human_bytes(a['done'])}  {human_bytes(a['got'] / took)}/s  "
                        f"attempts {attempt}")
                    break
                except Stopped:
                    return
                except Fatal as e:
                    self.fatal = str(e)
                    self.stop.set()
                    return
                except Permanent as e:
                    self.progress.fail(item, str(e))
                    log(f"FAILED {item.name}: {e}")
                    break
                except (Retry, OSError, http.client.HTTPException, ValueError) as e:
                    if isinstance(e, OSError) and e.errno in FATAL_DISK:
                        self.fatal = f"cannot write {item.part}: {e}"
                        self.stop.set()
                        return
                    delay = getattr(e, "delay", None)
                    if delay is None:
                        delay = min(self.args.max_backoff, 15 * 2 ** (attempt - 1)) * random.uniform(0.7, 1.3)
                    error = str(e) or type(e).__name__
                    self.progress.retry(item, attempt, error, delay)
                    log(f"{item.name}: attempt {attempt} failed ({error}); retrying in {delay:.0f} s")
                    if self.stop.wait(delay):
                        return
            else:
                self.progress.fail(item, f"gave up after {self.args.retries} attempts: {error}")
                log(f"FAILED {item.name}: gave up after {self.args.retries} attempts")


def stamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    """A single write per line, so that lines from different workers cannot run together."""
    sys.stdout.write(f"{stamp()}  {message}\n")
    sys.stdout.flush()


def cmd_get(args) -> int:
    check_dest(args.dest, args.snap)
    key = load_key()
    meta = get_json(make_opener(args.api_base), args.api_base.rstrip("/") + "/", key, args)
    args.dest.mkdir(parents=True, exist_ok=True)
    state_dir(args.dest).mkdir(exist_ok=True)

    lock = open(state_dir(args.dest) / "lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit(f"another download is already writing to {args.dest}")
    except OSError as e:
        print(f"WARNING: cannot lock {args.dest} ({e}); make sure only one download runs there", flush=True)

    items = build_items(args, meta)
    if not items:
        sys.exit("no files to download: the API gave no file count and there is no checksum list")
    write_json(state_dir(args.dest) / "manifest.json", {"snap": args.snap, "items": [asdict(i) for i in items]})
    for it in items:
        Path(it.path).parent.mkdir(parents=True, exist_ok=True)

    progress = Progress(args.dest, items, args.workers)
    done, on_disk, total = scan(items, progress.sizes)
    free = shutil.disk_usage(args.dest).free
    if total - on_disk > free:
        sys.exit(f"not enough space: about {human_bytes(total - on_disk)} still to download, "
                 f"{human_bytes(free)} free at {args.dest}")

    job = Download(args, key, items, progress)
    pending = [it for it in items if not os.path.exists(it.path)]
    for it in pending[:args.limit] if args.limit else pending:
        job.todo.put(it)
    log(f"{done} of {len(items)} files already finished; {job.todo.qsize()} to fetch with {args.workers} workers; "
        f"about {human_bytes(total - on_disk)} left; {human_bytes(free)} free")

    def on_signal(signum, _frame):
        log(f"signal {signal.Signals(signum).name}: stopping; partial files stay for the next run")
        job.stop.set()

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(sig, on_signal)

    threads = [threading.Thread(target=job.work, daemon=True) for _ in range(args.workers)]
    for t in threads:
        t.start()
    last_line = 0.0
    while any(t.is_alive() for t in threads):
        for t in threads:
            t.join(timeout=args.state_every / len(threads))
        progress.write()
        if time.time() - last_line >= args.log_every:
            last_line = time.time()
            done, on_disk, total = scan(items, progress.sizes)
            speed = progress.speed()
            eta = human_time((total - on_disk) / speed) if speed > 0 else "unknown"
            log(f"{done}/{len(items)} files  {human_bytes(on_disk)} of about {human_bytes(total)} "
                f"({on_disk / total:.1%})  {human_bytes(speed)}/s  left {eta}  retries {progress.retries}  "
                f"failed {len(progress.failed)}")

    done, on_disk, total = scan(items, progress.sizes)
    if job.fatal:
        outcome, code = f"stopped: {job.fatal}", 2
    elif job.stop.is_set():
        outcome, code = "stopped by a signal; run again to continue", 3
    elif progress.failed:
        outcome, code = f"{len(progress.failed)} files failed; run again to retry them", 1
    elif args.limit and done < len(items):
        outcome, code = f"test run finished ({args.limit} files requested)", 0
    else:
        outcome, code = f"all {len(items)} files finished and checked", 0
    progress.write(finished=True, outcome=outcome)
    log(f"{outcome}  ({done}/{len(items)} files, {human_bytes(on_disk)})")
    for name, reason in progress.failed.items():
        print(f"  {name}: {reason}", flush=True)
    return code


# ---------------------------------------------------------------- check / status / verify
def cmd_check(args) -> int:
    """Everything that can go wrong before the real download, without writing anything."""
    check_dest(args.dest, args.snap)
    key = load_key()
    print("API key: found")
    base = args.api_base.rstrip("/")
    opener = make_opener(base)
    meta = get_json(opener, base + "/", key, args)
    print(f"API: key accepted; {meta.get('name', '?')}, {meta.get('num_files_snapshot', '?')} snapshot files, "
          f"{meta.get('num_files_groupcat', '?')} group-catalog files")

    listing = get_json(opener, f"{base}/files/snapshot-{args.snap}/", key, args)
    files = listing.get("files", listing) if isinstance(listing, dict) else listing
    print(f"file list: {len(files) if hasattr(files, '__len__') else '?'} entries")

    source = args.dest / "snapshots" / f"{args.snap:03d}" / "checksums.txt"
    source = source if source.exists() else Path(args.checksums)
    sums = read_checksums(source) if source.exists() else {}
    print(f"checksums: {len(sums)} entries in {source}" if sums else f"checksums: none found at {source}")
    if sums and meta.get("num_files_snapshot") not in (None, len(sums)):
        print(f"WARNING: the API says {meta['num_files_snapshot']} files, the checksum list {len(sums)}")

    probe = args.probe_mib * 1024 ** 2
    url = f"{base}/files/snapshot-{args.snap}.0.hdf5"
    r, wait = open_range(opener, url, key, 0, probe - 1, args)
    t0 = time.time()
    with r:
        head, got = b"", 0
        while got < probe and (block := r.read(1024 ** 2)):
            head = head or block[:8]
            got += len(block)
        final, status, content_range = r.geturl(), r.status, r.headers.get("Content-Range")
    speed = got / max(time.time() - t0, 1e-6)
    print(f"first file: served from {urlsplit(final).hostname}, HTTP {status}, {content_range or 'no range header'}")
    print(f"format: {'HDF5 signature OK' if head == HDF5_SIGNATURE else 'NOT an HDF5 file: ' + repr(head)}")
    print(f"speed: {human_bytes(speed)}/s once data flows, after {wait:.0f} s until the server answered "
          f"({human_bytes(got)} read, not saved)")

    r, _ = open_range(opener, url, key, 1048576, 1048587, args)
    with r:
        resumable = r.status == 206 and (r.headers.get("Content-Range") or "").startswith("bytes 1048576-")
    print(f"resume: {'supported' if resumable else 'NOT supported, an interrupted file restarts from zero'}")

    n = int(meta.get("num_files_snapshot") or len(sums) or 0)
    total = n * TYPICAL_SIZE["snapshot"]
    probe_path = args.dest
    while not probe_path.exists():
        probe_path = probe_path.parent
    free = shutil.disk_usage(probe_path).free
    print(f"space: about {human_bytes(total)} needed, {human_bytes(free)} free at {probe_path}")
    if speed > 0:
        one = total / speed + n * wait      # one sample of each; refused attempts are not counted
        print(f"time: about {human_time(one)} on one connection; "
              f"with {args.workers} workers at best {human_time(one / args.workers)}")
    print("OK" if resumable and head == HDF5_SIGNATURE else "PROBLEMS: see above")
    return 0 if resumable and head == HDF5_SIGNATURE else 1


def load_state(dest: Path) -> tuple[list[Item], dict]:
    manifest = state_dir(dest) / "manifest.json"
    if not manifest.exists():
        sys.exit(f"no download has started at {dest} (no {manifest})")
    items = [Item(**i) for i in json.loads(manifest.read_text())["items"]]
    path = state_dir(dest) / "progress.json"
    try:
        return items, json.loads(path.read_text())
    except (OSError, ValueError):
        return items, {}


def render_status(dest: Path) -> str:
    items, st = load_state(dest)
    sizes = st.get("sizes", {})
    done, on_disk, total = scan(items, sizes)
    frac = on_disk / total if total else 0.0
    bar = "#" * int(40 * frac) + "." * (40 - int(40 * frac))
    age = time.time() - st.get("updated", 0)
    running = bool(st) and not st.get("finished") and age < 120
    speed = st.get("speed", 0.0) if running else 0.0
    lines = [f"Download to {dest}",
             f"[{bar}] {frac:.1%}",
             f"files    {done} of {len(items)} finished and checked    retries so far {st.get('retries', 0)}    "
             f"failed {len(st.get('failed', {}))}",
             f"data     {human_bytes(on_disk)} of about {human_bytes(total)}"]
    if running:
        left = human_time((total - on_disk) / speed) if speed > 0 else "unknown"
        lines.append(f"speed    {human_bytes(speed)}/s over the last 5 min    time left about {left}")
        job = f"SLURM job {st['job']} on {st['host']}" if st.get("job") else f"process {st.get('pid')} on {st.get('host')}"
        lines.append(f"state    running: {job} (updated {age:.0f} s ago)")
    elif st.get("finished"):
        lines.append(f"state    not running: {st.get('outcome')}")
    elif st:
        lines.append(f"state    not running: no update for {human_time(age)}; the job died or was killed. "
                     f"Run `get` again to continue")
    else:
        lines.append("state    no progress recorded yet")
    if running and st.get("active"):
        lines.append("in progress")
        for a in st["active"]:
            size = f"{human_bytes(a['done'])} of {human_bytes(a['total'])} ({a['done'] / a['total']:.0%})" \
                if a.get("total") else human_bytes(a["done"])
            if a["state"] == "waiting to retry":
                what = f"attempt {a['attempt']} in {a['retry_in']:.0f} s, after: {a['error']}"
            elif a["state"] == "downloading":
                what = f"{human_bytes(a['speed'])}/s"
            else:
                what = a["state"]
            lines.append(f"  {a['file']:<28} {size:<32} {what}")
    if st.get("failed"):
        lines.append("failed")
        lines += [f"  {name}: {why}" for name, why in st["failed"].items()]
    return "\n".join(lines)


def cmd_status(args) -> int:
    while True:
        text = render_status(args.dest)
        if args.watch and sys.stdout.isatty():
            print("\033[2J\033[H" + stamp() + "\n" + text, flush=True)
        else:
            print(text, flush=True)
        if not args.watch:
            return 0
        try:
            time.sleep(args.watch)
        except KeyboardInterrupt:
            return 0


def cmd_verify(args) -> int:
    """Recompute the sha256 of every finished file; reads only, deletes nothing."""
    items, _ = load_state(args.dest)
    todo = queue.Queue()
    for it in items:
        if os.path.exists(it.path):
            todo.put(it)
    n, results, lock = todo.qsize(), {"ok": 0, "bad": [], "unchecked": 0}, threading.Lock()

    def work():
        while True:
            try:
                it = todo.get_nowait()
            except queue.Empty:
                return
            if not it.sha256:
                with lock:
                    results["unchecked"] += 1
                continue
            digest = hashlib.sha256()
            with open(it.path, "rb") as f:
                while block := f.read(CHUNK):
                    digest.update(block)
            with lock:
                if digest.hexdigest() == it.sha256:
                    results["ok"] += 1
                else:
                    results["bad"].append(it.name)
                    print(f"MISMATCH {it.name}", flush=True)
                checked = results["ok"] + len(results["bad"]) + results["unchecked"]
                if checked % 20 == 0:
                    print(f"{stamp()}  checked {checked} of {n}", flush=True)

    threads = [threading.Thread(target=work) for _ in range(args.workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    missing = len(items) - n
    print(f"{results['ok']} match, {len(results['bad'])} mismatch, {results['unchecked']} have no checksum, "
          f"{missing} not downloaded yet")
    return 1 if results["bad"] else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=["check", "get", "status", "verify"])
    ap.add_argument("--dest", type=Path, required=True, help="root of the new copy; files go to DEST/snapshots/NNN/")
    ap.add_argument("--snap", type=int, default=99)
    ap.add_argument("--workers", type=int, default=4, help="files downloaded in parallel")
    ap.add_argument("--with-catalogs", action="store_true", help="also fetch the group catalog and offsets (7.4 GB)")
    ap.add_argument("--limit", type=int, help="fetch only this many files (for a test run)")
    ap.add_argument("--retries", type=int, default=20, help="attempts per file before giving up on it")
    ap.add_argument("--max-backoff", type=float, default=900, help="longest wait between attempts, seconds")
    ap.add_argument("--timeout", type=float, default=120, help="seconds without data before an attempt fails")
    ap.add_argument("--log-every", type=float, default=60, help="seconds between progress lines in the log")
    ap.add_argument("--state-every", type=float, default=10, help="seconds between progress.json updates")
    ap.add_argument("--watch", type=float, help="status: refresh every this many seconds")
    ap.add_argument("--probe-mib", type=int, default=64, help="check: MiB to read when measuring the speed")
    ap.add_argument("--checksums", default=str(EXISTING / "snapshots" / "099" / "checksums.txt"),
                    help="checksum list to copy into the new copy (read-only)")
    ap.add_argument("--no-checksums", action="store_true", help="check file sizes only")
    ap.add_argument("--api-base", default=API_BASE)
    args = ap.parse_args()
    return {"check": cmd_check, "get": cmd_get, "status": cmd_status, "verify": cmd_verify}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
