"""A small client for the TNG web API (https://www.tng-project.org/data/docs/api/), standard library only.

The key comes from the TNG_API_KEY environment variable or ~/.tng_api_key (mode 600). It is never printed or logged,
and it is sent only to tng-project.org hosts. The API answers from www.tng-project.org and redirects files to its data
servers, which often refuse a valid request (403 after about 40 s, 504 after about 50 s; measured 2026-11-02) and serve
the same request moments later. So only a 401/403 from the API host itself counts as a bad key; everything else that
looks temporary is retried with growing waits.
"""
from __future__ import annotations

import gzip
import http.client
import json
import os
import random
import socket
import stat
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

API_BASE = "https://www.tng-project.org/api/TNG50-1"
KEY_FILE = Path.home() / ".tng_api_key"
USER_AGENT = "GalaxyCLIP-3D pointclouds"


class ApiError(RuntimeError):
    """status: the HTTP status if the server answered; fatal: no other request can succeed either (the key)."""

    def __init__(self, message: str, status: int | None = None, fatal: bool = False):
        super().__init__(message)
        self.status, self.fatal = status, fatal


def load_key() -> str:
    env = os.environ.get("TNG_API_KEY", "").strip()
    if env:
        return env
    if not KEY_FILE.exists():
        raise ApiError(f"no API key: put it in {KEY_FILE} (chmod 600) or set TNG_API_KEY")
    if stat.S_IMODE(KEY_FILE.stat().st_mode) & 0o077:
        raise ApiError(f"{KEY_FILE} is readable by others; run: chmod 600 {KEY_FILE}")
    key = KEY_FILE.read_text().strip()
    if not key:
        raise ApiError(f"{KEY_FILE} is empty")
    return key


def public(url: str) -> str:
    """A URL without its query string, which holds the data server's signature."""
    p = urlsplit(url)
    return f"{p.scheme}://{p.netloc}{p.path}"


class _KeepKeyOnTngHosts(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        host = urlsplit(newurl).hostname or ""
        if new is not None and host != "tng-project.org" and not host.endswith(".tng-project.org"):
            new.remove_header("Api-key")
        return new


class Client:
    def __init__(self, key: str | None = None, api_base: str = API_BASE, timeout: float = 120, tries: int = 10,
                 max_wait: float = 120):
        self._key = key or load_key()
        self.api_base, self.timeout, self.tries, self.max_wait = api_base.rstrip("/"), timeout, tries, max_wait
        self.api_host = urlsplit(self.api_base).hostname
        self._local = threading.local()             # one opener per thread

    def _opener(self):
        if not hasattr(self._local, "opener"):
            self._local.opener = urllib.request.build_opener(_KeepKeyOnTngHosts())
        return self._local.opener

    def get(self, path: str) -> bytes:
        """Body of `path` (relative to the API base, or a full URL), gunzipped if the server sent it compressed."""
        url = path if path.startswith("http") else f"{self.api_base}/{path.lstrip('/')}"
        req = urllib.request.Request(url, headers={"api-key": self._key, "User-Agent": USER_AGENT})
        error = ""
        for attempt in range(1, self.tries + 1):
            try:
                with self._opener().open(req, timeout=self.timeout) as r:
                    body = r.read()
                return gzip.decompress(body) if body[:2] == b"\x1f\x8b" else body
            except urllib.error.HTTPError as e:
                host = urlsplit(e.url or url).hostname
                if e.code in (401, 403) and host == self.api_host:
                    raise ApiError(f"the API rejected the key (HTTP {e.code} from {public(url)})", e.code,
                                   fatal=True) from None
                if e.code in (400, 404, 410):
                    raise ApiError(f"HTTP {e.code} from {public(e.url or url)}", e.code) from None
                error = f"HTTP {e.code} from {host}"
            except (urllib.error.URLError, http.client.HTTPException, socket.timeout, OSError, EOFError) as e:
                error = str(e) or type(e).__name__
            if attempt < self.tries:
                time.sleep(min(self.max_wait, 5 * 2 ** (attempt - 1)) * random.uniform(0.7, 1.3))
        raise ApiError(f"gave up on {public(url)} after {self.tries} attempts; last: {error}")

    def get_json(self, path: str):
        return json.loads(self.get(path))
