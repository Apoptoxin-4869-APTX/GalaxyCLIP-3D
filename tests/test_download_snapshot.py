"""Tests for tools/download_snapshot.py against a local fake of the TNG API, with injected faults.
Needs only the standard library. Run: python3 tests/test_download_snapshot.py
"""
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "download_snapshot.py"
KEY, SECRET = "test-key-12345", "signed-url-secret"
N_FILES, SIZE = 6, 3 * 1024 ** 2 + 123
CONTENT = {i: (b"\x89HDF\r\n\x1a\n" + os.urandom(SIZE))[:SIZE] for i in range(N_FILES)}
N_CATALOG = 2
CONTENT.update({100 + i: os.urandom(200_000) for i in range(N_CATALOG)})   # group-catalog chunks
CONTENT[200] = b"\x89HDF\r\n\x1a\n" + os.urandom(300_000)                  # offsets file, which has no checksum


class Fake:
    """Server state: what to break, and what was asked for."""
    faults: dict = {}           # chunk -> list of one-shot faults: 'drop', '503', '403', '504', 'corrupt', 'norange',
                                # 'stall', 'html'
    api_faults: list = []       # one-shot HTTP codes from the API's own metadata page
    always_corrupt: set = set()
    missing: set = set()
    slow: float = 0.0           # seconds per 256 KiB, to give the signal test time
    ranges: dict = {}           # chunk -> list of Range headers received on the data server
    keys_on_data: list = []     # api-key header seen on the data server
    foreign_redirect = False
    requests = 0


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_):
        pass

    def send_json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_text(self, text):
        body = text.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        Fake.requests += 1
        path = self.path.split("?")[0]
        if path.startswith("/api/"):
            if self.headers.get("api-key") != KEY:
                return self.send_json(403, {"detail": "Failed session auth (and no API-Key sent)."})
            if path == "/api/SIM/":
                if Fake.api_faults:
                    return self.send_json(Fake.api_faults.pop(0), {"detail": "busy"})
                return self.send_json(200, {"name": "SIM", "num_files_snapshot": N_FILES,
                                            "num_files_groupcat": N_CATALOG})
            if re.fullmatch(r"/api/SIM/files/snapshot-\d+/", path):           # every snapshot has the same files
                return self.send_json(200, {"files": [f"x{i}" for i in range(N_FILES)]})
            m = re.fullmatch(r"/api/SIM/checksums/(snapshot|groupcat)-(\d+)/", path)
            if m:
                snap = int(m.group(2))
                rows = ([(CONTENT[i], f"snap_{snap:03d}.{i}.hdf5") for i in range(N_FILES)] if m.group(1) == "snapshot"
                        else [(CONTENT[100 + i], f"fof_subhalo_tab_{snap:03d}.{i}.hdf5") for i in range(N_CATALOG)])
                return self.send_text("".join(f"{hashlib.sha256(c).hexdigest()}  {n}\n" for c, n in rows))
            m = re.fullmatch(r"/api/SIM/files/(snapshot|groupcat)-\d+\.(\d+)\.hdf5", path)
            if m:
                chunk = int(m.group(2)) + (100 if m.group(1) == "groupcat" else 0)
            elif re.fullmatch(r"/api/SIM/files/offsets\.\d+\.hdf5", path):
                chunk = 200
            else:
                return self.send_json(404, {"detail": "not found"})
            if chunk in Fake.missing:
                return self.send_json(404, {"detail": "not found"})
            host = "localhost" if Fake.foreign_redirect else "127.0.0.1"
            self.send_response(302)
            self.send_header("Location", f"http://{host}:{self.server.server_port}/data/{chunk}?mac={SECRET}")
            self.send_header("Content-Length", "0")
            return self.end_headers()
        if path.startswith("/data/"):
            return self.serve_file(int(path.rsplit("/", 1)[1]))
        return self.send_json(404, {})

    def serve_file(self, chunk):
        Fake.keys_on_data.append(self.headers.get("api-key"))
        rng = self.headers.get("Range")
        Fake.ranges.setdefault(chunk, []).append(rng)
        faults = Fake.faults.get(chunk, [])
        fault = faults.pop(0) if faults else None
        if fault == "503":
            self.send_response(503)
            self.send_header("Retry-After", "1")
            self.send_header("Content-Length", "0")
            return self.end_headers()
        if fault in ("403", "504"):     # what the real data server does now and then, with a valid signature
            body = f"<h1>{fault}</h1>".encode()
            self.send_response(int(fault))
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            return self.wfile.write(body)
        data = CONTENT[chunk]
        if fault == "corrupt" or chunk in Fake.always_corrupt:
            data = data[:1000] + b"X" + data[1001:]
        if fault == "html":             # an error page sent with status 200
            data = b"<h1>busy</h1>"
        start = 0
        if rng and fault not in ("norange", "html"):
            start = int(re.match(r"bytes=(\d+)-", rng).group(1))
            if start >= len(data):
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{len(data)}")
                self.send_header("Content-Length", "0")
                return self.end_headers()
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
        else:
            self.send_response(200)
        self.send_header("Content-Length", str(len(data) - start))
        self.end_headers()
        body = data[start:]
        if fault == "drop":             # the connection dies part-way: the announced length never arrives
            self.wfile.write(body[:len(body) // 2])
            self.wfile.flush()
            self.close_connection = True
            return None
        for i in range(0, len(body), 256 * 1024):
            self.wfile.write(body[i:i + 256 * 1024])
            if Fake.slow:
                time.sleep(Fake.slow)
        return None


def reset():
    Fake.faults, Fake.always_corrupt, Fake.missing, Fake.ranges, Fake.keys_on_data = {}, set(), set(), {}, []
    Fake.slow, Fake.foreign_redirect, Fake.api_faults = 0.0, False, []


class QuietServer(ThreadingHTTPServer):
    def handle_error(self, request, client_address):
        pass                    # clients that are killed or stop mid-transfer are part of the tests


server = QuietServer(("127.0.0.1", 0), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
API = f"http://127.0.0.1:{server.server_port}/api/SIM"
TMP = Path(tempfile.mkdtemp(prefix="dl_test_"))
SUMS = TMP / "existing" / "snapshots" / "099" / "checksums.txt"
SUMS.parent.mkdir(parents=True)
SUMS.write_text("".join(f"{hashlib.sha256(CONTENT[i]).hexdigest()}  snap_099.{i}.hdf5\n" for i in range(N_FILES)))
GC_SUMS = TMP / "existing" / "groupcats" / "099" / "checksums.txt"
GC_SUMS.parent.mkdir(parents=True)
GC_SUMS.write_text("".join(f"{hashlib.sha256(CONTENT[100 + i]).hexdigest()}  fof_subhalo_tab_099.{i}.hdf5\n"
                           for i in range(N_CATALOG)))
ENV = {**os.environ, "TNG_API_KEY": KEY}


def run(command, dest, *extra, env=ENV, check=None, wait=True):
    cmd = [sys.executable, "-I", str(SCRIPT), command, "--dest", str(dest), "--api-base", API,
           "--checksums", str(SUMS), "--state-every", "0.2", "--log-every", "0.5", "--max-backoff", "1", *extra]
    if not wait:
        return subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=180)
    if check is not None:
        assert p.returncode == check, (p.returncode, p.stdout, p.stderr)
    return p


def all_good(dest, n=N_FILES):
    d = Path(dest) / "snapshots" / "099"
    for i in range(n):
        assert (d / f"snap_099.{i}.hdf5").read_bytes() == CONTENT[i], i
    assert not list(d.glob("*.part")), list(d.glob("*.part"))


def no_secrets(*texts):
    for t in texts:
        assert KEY not in t and SECRET not in t, "a secret reached the output"


# ---------------------------------------------------------------- happy path + status
dest = TMP / "ok"
p = run("get", dest, check=0)
all_good(dest)
assert all(r == [None] for r in Fake.ranges.values())
st = run("status", dest, check=0).stdout
assert f"{N_FILES} of {N_FILES} finished" in st and "100.0%" in st and "not running: all" in st, st
no_secrets(p.stdout, p.stderr, st, *(f.read_text() for f in (dest / ".download").rglob("*.json")))
print("PASS download: all files correct, no .part left, status reports 100%, no key or signed URL in any output")

p = run("get", dest, check=0)
assert "6 of 6 files already finished; 0 to fetch" in p.stdout, p.stdout
print("PASS rerun: finished files are not downloaded again")

v = run("verify", dest, check=0).stdout
assert "6 match, 0 mismatch" in v, v
target = dest / "snapshots" / "099" / "snap_099.2.hdf5"
blob = bytearray(target.read_bytes())
blob[5000] ^= 1
target.write_bytes(blob)
v = run("verify", dest, check=1).stdout
assert "MISMATCH snap_099.2.hdf5" in v and "5 match, 1 mismatch" in v, v
print("PASS verify: re-hashing catches a flipped byte in a finished file")

# ---------------------------------------------------------------- faults
reset()
dest = TMP / "faults"
Fake.faults = {0: ["drop", "drop"], 1: ["503", "503"], 2: ["corrupt"], 3: ["drop", "norange"]}
p = run("get", dest, check=0)
all_good(dest)
assert Fake.ranges[0][0] is None and Fake.ranges[0][1].startswith("bytes=") and Fake.ranges[0][2].startswith("bytes=")
assert "incomplete" in p.stdout and "HTTP 503" in p.stdout and "sha256 mismatch" in p.stdout, p.stdout
print("PASS faults: dropped connections resume with Range; 503 waits and retries; a corrupt file is refetched; "
      "a server that ignores Range restarts the file")

reset()
dest = TMP / "refused"
Fake.faults = {0: ["403", "504", "403"], 1: ["504"], 200: ["html", "403"]}
Fake.api_faults = [504, 503]
p = run("get", dest, "--with-catalogs", check=0)
all_good(dest)
assert (dest / "offsets" / "offsets_099.hdf5").read_bytes() == CONTENT[200]
assert "HTTP 403 from 127.0.0.1" in p.stdout and "HTTP 504 from 127.0.0.1" in p.stdout, p.stdout
assert "not an HDF5 file" in p.stdout and "HTTP 504; attempt 1 of 6" in p.stdout, p.stdout
no_secrets(p.stdout, p.stderr)
print("PASS busy server: data-server 403 and 504 are retried and do not stop the job, an API outage at start-up "
      "is waited out, and an error page is not accepted as the offsets file")

reset()
dest = TMP / "partial"
d = dest / "snapshots" / "099"
d.mkdir(parents=True)
(d / "snap_099.0.hdf5.part").write_bytes(CONTENT[0][:1234567])
(d / "snap_099.1.hdf5.part").write_bytes(CONTENT[1])                     # complete, but killed before the rename
(d / "snap_099.2.hdf5.part").write_bytes(CONTENT[2] + b"extra")          # longer than the real file
(d / "snap_099.3.hdf5.part").write_bytes(b"garbage" + CONTENT[3][7:100000])
run("get", dest, check=0)
all_good(dest)
assert Fake.ranges[0] == ["bytes=1234567-"], Fake.ranges[0]
assert Fake.ranges[1] == [f"bytes={SIZE}-"], Fake.ranges[1]
assert len(Fake.ranges[3]) == 2 and Fake.ranges[3][1] is None, Fake.ranges[3]
print("PASS leftovers from a killed job: continued, finished without refetching, or replaced if wrong")

reset()
dest = TMP / "perm"
Fake.always_corrupt, Fake.missing = {1}, {2}
p = run("get", dest, "--retries", "2", check=1)
d = dest / "snapshots" / "099"
assert not (d / "snap_099.1.hdf5").exists() and not (d / "snap_099.2.hdf5").exists()
assert (d / "snap_099.0.hdf5").exists() and (d / "snap_099.5.hdf5").exists()
st = run("status", dest, check=0).stdout
assert "failed 2" in st and "snap_099.2.hdf5: not on the server" in st and "gave up after 2" in st, st
print("PASS failures: a file that stays corrupt never gets its final name; a 404 is reported; the others still finish")
reset()
run("get", dest, check=0)
all_good(dest)
print("PASS failed files are fetched by simply running again")

reset()
dest = TMP / "catalogs"
Fake.faults = {100: ["corrupt"]}
run("get", dest, "--with-catalogs", check=0)
all_good(dest)
for i in range(N_CATALOG):
    assert (dest / "groupcats" / "099" / f"fof_subhalo_tab_099.{i}.hdf5").read_bytes() == CONTENT[100 + i]
assert (dest / "offsets" / "offsets_099.hdf5").read_bytes() == CONTENT[200]
assert (dest / "snapshots" / "099" / "checksums.txt").read_text() == SUMS.read_text()
assert (dest / "groupcats" / "099" / "checksums.txt").read_text() == GC_SUMS.read_text()
st = run("status", dest, check=0).stdout
assert f"{N_FILES + N_CATALOG + 1} of {N_FILES + N_CATALOG + 1} finished" in st, st
print("PASS --with-catalogs: group catalog (checked against its own list) and offsets land in the same layout "
      "as the existing copy, and the checksum lists are copied along")

reset()
multi = TMP / "multi"
nowhere = TMP / "nowhere" / "snapshots" / "099" / "checksums.txt"          # no local list: fetched from the API
n_all = N_FILES + N_CATALOG + 1
for snap in (99, 91):
    run("get", multi, "--snap", str(snap), "--with-catalogs", "--checksums", str(nowhere), check=0)
for snap in (99, 91):
    d = multi / "snapshots" / f"{snap:03d}"
    assert all((d / f"snap_{snap:03d}.{i}.hdf5").read_bytes() == CONTENT[i] for i in range(N_FILES)), snap
    assert (multi / "groupcats" / f"{snap:03d}" / "checksums.txt").exists()
    assert (multi / "offsets" / f"offsets_{snap:03d}.hdf5").read_bytes() == CONTENT[200]
    assert (multi / ".download" / f"{snap:03d}" / "manifest.json").exists()
manifest = json.loads((multi / ".download" / "091" / "manifest.json").read_text())
assert not any(os.path.isabs(i["path"]) for i in manifest["items"]), "manifest paths must be relative"
renamed = TMP / "multi_renamed"
os.rename(multi, renamed)
for snap in (99, 91):
    st = run("status", renamed, "--snap", str(snap), check=0).stdout
    assert f"{n_all} of {n_all} finished" in st and f"snapshot {snap}" in st, st
v = run("verify", renamed, "--snap", "91", check=0).stdout
assert f"{N_FILES + N_CATALOG} match, 0 mismatch" in v, v
print("PASS several snapshots in one folder: separate state per snapshot, checksum lists fetched from the API "
      "when there is no local one, and status and verify still work after the folder is renamed")

# ---------------------------------------------------------------- key handling
reset()
dest = TMP / "badkey"
n0 = Fake.requests
p = run("get", dest, env={**os.environ, "TNG_API_KEY": "wrong"}, check=1)
assert "rejected the key" in p.stderr and Fake.requests - n0 == 1, (p.stderr, Fake.requests - n0)
print("PASS wrong key: stops at the first request instead of retrying")

reset()
dest = TMP / "foreign"
Fake.foreign_redirect = True
run("get", dest, "--limit", "1", "--workers", "1", check=0)
assert Fake.keys_on_data == [None], Fake.keys_on_data
reset()
dest2 = TMP / "own"
run("get", dest2, "--limit", "1", "--workers", "1", check=0)
assert Fake.keys_on_data == [KEY], Fake.keys_on_data
print("PASS the key follows redirects to the API's own host but not to any other host")

env = {k: v for k, v in os.environ.items() if k != "TNG_API_KEY"}
home = TMP / "home"
home.mkdir()
keyfile = home / ".tng_api_key"
keyfile.write_text(KEY + "\n")
keyfile.chmod(0o644)
p = run("get", TMP / "kf", env={**env, "HOME": str(home)}, check=1)
assert "readable by others" in p.stderr, p.stderr
keyfile.chmod(0o600)
run("get", TMP / "kf", "--limit", "1", env={**env, "HOME": str(home)}, check=0)
print("PASS key file: refused while group/world readable, used once it is mode 600")

# ---------------------------------------------------------------- the existing copy is protected
for bad in ("/cluster/projects/ska/illustris/tng50-1", "/cluster/projects/ska/illustris/tng50-1/snapshots",
            "/cluster/projects/ska/illustris/tng50-1/new"):
    p = run("get", bad, check=1)
    assert "refusing to write inside the existing copy" in p.stderr, p.stderr
link = TMP / "linked"
link.mkdir()
(link / "snapshots").symlink_to("/cluster/projects/ska/illustris/tng50-1/snapshots")
p = run("get", link, check=1)
assert "refusing to write inside the existing copy" in p.stderr, p.stderr
print("PASS guard: refuses the existing copy, its subfolders, and symlinks pointing into it")

# ---------------------------------------------------------------- signals, locking, live status
reset()
dest = TMP / "sig"
Fake.slow = 0.05
proc = run("get", dest, "--workers", "2", wait=False)
time.sleep(1.5)
live = run("status", dest, check=0).stdout
assert "state    running" in live and "in progress" in live and "/s" in live, live
if os.environ.get("SHOW_STATUS"):
    print(live)
second = run("get", dest, check=1)
assert "already writing" in second.stderr, second.stderr
print("PASS while running: status shows the job, active files and speeds; a second download is refused")
proc.send_signal(signal.SIGUSR1)
out, _ = proc.communicate(timeout=60)
assert proc.returncode == 3 and "stopping; partial files stay" in out, (proc.returncode, out)
d = dest / "snapshots" / "099"
assert list(d.glob("*.part")), "partial files should remain after a signal"
st = run("status", dest, check=0).stdout
assert "not running: stopped by a signal" in st, st
Fake.slow = 0.0
run("get", dest, check=0)
all_good(dest)
print("PASS signal: stops cleanly, keeps partial files, and the next run finishes the job")

reset()
dest = TMP / "dead"
Fake.slow = 0.05
proc = run("get", dest, "--workers", "2", wait=False)
time.sleep(1.5)
proc.kill()
proc.communicate()
Fake.slow = 0.0
run("get", dest, check=0)
all_good(dest)
print("PASS kill -9: no signal handler runs, and the next run still finishes with correct files")

# ---------------------------------------------------------------- check command
reset()
p = run("check", TMP / "not-created" / "x", "--probe-mib", "1", check=0)
assert "key accepted" in p.stdout and "HDF5 signature OK" in p.stdout and "resume: supported" in p.stdout, p.stdout
assert not (TMP / "not-created").exists()
no_secrets(p.stdout, p.stderr)
print("PASS check: reports the key, format, resume support and speed, and writes nothing")

reset()
Fake.faults = {0: ["403", "504", "403"]}
p = run("check", TMP / "not-created" / "x", "--probe-mib", "1", check=0)
assert "attempt 1 of 8 refused" in p.stdout and "HTTP 403 from 127.0.0.1" in p.stdout and p.stdout.endswith("OK\n"), p.stdout
no_secrets(p.stdout, p.stderr)
reset()
Fake.faults = {0: ["504"] * 8}
p = run("check", TMP / "not-created" / "x", "--probe-mib", "1", check=1)
assert "refused all 8 attempts" in p.stderr and "HTTP 504" in p.stderr, p.stderr
no_secrets(p.stdout, p.stderr)
print("PASS check on a busy server: says what was refused and by whom, retries, and fails only if nothing gets through")

server.shutdown()
shutil.rmtree(TMP, ignore_errors=True)
print("ALL TESTS PASSED")
