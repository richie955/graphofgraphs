"""Disk-backed HTTP cache + rate limiter.

Colab sessions die. Every external response lands in data/cache/ keyed by URL
hash, so a re-run costs nothing and the whole build is reproducible offline.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any

import requests

from .config import cfg


class RateLimiter:
    def __init__(self, min_interval: float) -> None:
        self.min_interval = min_interval
        self._last = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            delta = time.monotonic() - self._last
            if delta < self.min_interval:
                time.sleep(self.min_interval - delta)
            self._last = time.monotonic()


class JsonCache:
    """One subdirectory per service, sharded two hex chars deep."""

    def __init__(self, namespace: str) -> None:
        self.root = cfg()["paths"]["cache_dir"] / namespace
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode()).hexdigest()
        return self.root / digest[:2] / f"{digest}.json"

    def get(self, key: str) -> Any | None:
        p = self._path(key)
        if not p.exists():
            return None
        try:
            with open(p) as fh:
                return json.load(fh)["value"]
        except (json.JSONDecodeError, KeyError):
            return None

    def put(self, key: str, value: Any) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        with open(tmp, "w") as fh:
            json.dump({"key": key, "value": value, "fetched_at": time.time()}, fh)
        tmp.replace(p)


class CachedSession:
    """GET-only JSON client. A 404 is cached as None: a missing record is a fact."""

    def __init__(self, namespace: str, min_interval: float) -> None:
        self.cache = JsonCache(namespace)
        self.limiter = RateLimiter(min_interval)
        self.session = requests.Session()
        self.session.headers["Accept"] = "application/json"
        # Generic client name on purpose: no user or project identifiers leave the machine.
        self.session.headers["User-Agent"] = "gog-pipeline/0.1 (research; python-requests)"
        http = cfg()["http"]
        self.timeout = http["timeout"]
        self.max_retries = http["max_retries"]

    def get_text(self, url: str, *, force: bool = False) -> str | None:
        """For endpoints that serve text/plain and 404 under a JSON Accept header."""
        value = self.get_json(url, force=force, accept="text/plain")
        if value is None:
            return None
        if isinstance(value, dict) and "_text" in value:
            return value["_text"]
        return value if isinstance(value, str) else None

    def get_json(self, url: str, *, force: bool = False,
                 accept: str = "application/json") -> Any | None:
        key = url if accept == "application/json" else f"{url}#accept={accept}"
        if not force:
            hit = self.cache.get(key)
            if hit is not None or self.cache._path(key).exists():
                return hit

        backoff = 1.0
        for attempt in range(self.max_retries):
            self.limiter.wait()
            try:
                r = self.session.get(url, timeout=self.timeout, headers={"Accept": accept})
            except requests.RequestException:
                time.sleep(backoff)
                backoff *= 2
                continue

            if r.status_code == 404:
                self.cache.put(key, None)
                return None
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(backoff)
                backoff *= 2
                continue
            if not r.ok:
                self.cache.put(key, None)
                return None
            if accept == "application/json":
                try:
                    value = r.json()
                except ValueError:
                    value = {"_text": r.text}
            else:
                value = r.text
            self.cache.put(key, value)
            return value

        raise RuntimeError(f"giving up after {self.max_retries} attempts: {url}")
