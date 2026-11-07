"""One writer per Lance root.

Lance commits are atomic per table, but the committer's own steps are not: append() reads
which recordings are committed, then writes; cleanup() lists unreferenced MP4s, then deletes
them. Two writers interleaving those steps store a recording's frames twice, or delete videos
a commit has just placed. So every write to a root (egomimic.rldb.lance.commit, and the
committer for a whole run) holds `writer_lock(root)`. Readers never take it.

On R2 the lock is the object `<root>/_writer.lock`, created with `If-None-Match: *` and
rewritten only with `If-Match: <its ETag>`, so of two writers racing for it exactly one
wins. The holder rewrites it every ttl/4 to push `expires_at` forward; a holder that dies
stops doing so and the lock can be taken over once it expires. R2 ignores `If-Match` on
DELETE, so the object is never deleted: releasing it is a conditional rewrite marking it
`released`. Before each write, `check` makes sure the lock is still ours, renewing it if the
last renewal is getting old, and raises `LockLost` otherwise.

A local root uses `fcntl.flock` on the same file instead: the kernel drops it when the
holder exits, so it needs no expiry. It only excludes writers on one machine.

The lock is re-entrant within a process: nested `writer_lock(root)` calls (the committer's
run, then commit_batch, then append) share one hold.
"""

from __future__ import annotations

import contextlib
import fcntl
import getpass
import json
import logging
import os
import socket
import threading
import time
import uuid
from pathlib import Path

from egomimic.rldb.lance.dataset import _is_remote, _s3_client

log = logging.getLogger(__name__)

LOCK_NAME = "_writer.lock"
# A dead holder blocks other writers for at most this long.
DEFAULT_TTL_S = 600.0
# How long a writer waits for the lock before giving up.
DEFAULT_WAIT_S = 1800.0
POLL_S = 5.0


class LockHeld(RuntimeError):
    """Another writer held the lock for longer than we were willing to wait."""


class LockLost(RuntimeError):
    """Our hold on the lock lapsed and another writer took it over."""


def _env_seconds(name: str, default: float) -> float:
    value = os.environ.get(name)
    return float(value) if value else default


def _key(root) -> str:
    return str(root).rstrip("/")


def lock_uri(root) -> str:
    return f"{_key(root)}/{LOCK_NAME}"


def _describe(info: dict | None) -> str:
    if not info:
        return "unknown holder"
    out = f"{info.get('holder')} on {info.get('host')} (pid {info.get('pid')})"
    if info.get("expires_at"):
        out += f", expires in {info['expires_at'] - time.time():.0f}s"
    return out


class _Lock:
    depth = 0
    lost = False

    def __init__(self, root: str, ttl: float, holder: str):
        self.root, self.ttl, self.holder = root, ttl, holder
        self.token = uuid.uuid4().hex

    def info(self, released: bool = False) -> dict:
        now = time.time()
        return {
            "holder": self.holder,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "token": self.token,
            "acquired_at": getattr(self, "acquired_at", now),
            "expires_at": 0.0 if released else now + self.ttl,
            "released": released,
        }

    def check(self) -> None:
        if self.lost:
            raise LockLost(f"lost the writer lock on {self.root}")


class _FileLock(_Lock):
    def acquire(self, wait: float) -> None:
        path = Path(self.root) / LOCK_NAME
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "a+")
        deadline = time.monotonic() + wait
        while True:
            try:
                fcntl.flock(self.f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    self.f.seek(0)
                    try:
                        other = json.loads(self.f.read() or "null")
                    except ValueError:
                        other = None
                    self.f.close()
                    raise LockHeld(
                        f"{self.root} is locked by {_describe(other)}"
                    ) from None
                time.sleep(min(0.2, max(0.0, deadline - time.monotonic())))
        self.acquired_at = time.time()
        self.f.seek(0)
        self.f.truncate()
        self.f.write(json.dumps(self.info()))
        self.f.flush()

    def release(self) -> None:
        fcntl.flock(self.f, fcntl.LOCK_UN)
        self.f.close()


class _S3Lock(_Lock):
    def __init__(self, root: str, ttl: float, holder: str, client=None):
        super().__init__(root, ttl, holder)
        self.bucket, self.key = lock_uri(root)[len("s3://") :].split("/", 1)
        self.client = client or _s3_client()
        self.etag: str | None = None
        self.renewed = 0.0
        self._io = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _put(self, body: dict, **cond) -> str | None:
        """PUT `body` under `cond` (IfNoneMatch / IfMatch); its ETag, or None on 412."""
        try:
            resp = self.client.put_object(
                Bucket=self.bucket, Key=self.key, Body=json.dumps(body).encode(), **cond
            )
        except Exception as e:
            if _status(e) in (409, 412):
                return None
            raise
        return resp["ETag"]

    def _read(self) -> tuple[str, dict] | None:
        try:
            resp = self.client.get_object(Bucket=self.bucket, Key=self.key)
        except Exception as e:
            if _status(e) == 404:
                return None
            raise
        try:
            info = json.loads(resp["Body"].read())
        except ValueError:
            info = {}
        return resp["ETag"], info

    def acquire(self, wait: float) -> None:
        deadline = time.monotonic() + wait
        while True:
            self.acquired_at = time.time()
            seen = self._read()
            if seen is None:
                etag = self._put(self.info(), IfNoneMatch="*")
            else:
                etag_seen, other = seen
                free = other.get("released") or other.get("expires_at", 0) < time.time()
                etag = self._put(self.info(), IfMatch=etag_seen) if free else None
                if not free and time.monotonic() >= deadline:
                    raise LockHeld(f"{self.root} is locked by {_describe(other)}")
                if not free:
                    time.sleep(min(POLL_S, max(0.0, deadline - time.monotonic())))
                    continue
            if etag is not None:
                self.etag, self.renewed = etag, time.monotonic()
                break
            # Lost a race for a free lock: look again at once.
            if time.monotonic() >= deadline:
                raise LockHeld(f"{self.root}: lost the race for a free lock")
        self._thread = threading.Thread(
            target=self._beat, name="lance-writer-lock", daemon=True
        )
        self._thread.start()

    def renew(self) -> None:
        with self._io:
            if self.lost:
                return
            etag = self._put(self.info(), IfMatch=self.etag)
            if etag is None:
                self.lost = True
                log.error("lost the writer lock on %s", self.root)
            else:
                self.etag, self.renewed = etag, time.monotonic()

    def _beat(self) -> None:
        while not self._stop.wait(self.ttl / 4):
            try:
                self.renew()
            except Exception:  # transient: the next beat, or check(), tries again
                log.warning(
                    "renewing the writer lock on %s failed", self.root, exc_info=True
                )
            if self.lost:
                return

    def check(self) -> None:
        # A holder paused past its ttl must not write: renew now unless the last renewal
        # is recent, so a takeover in the meantime surfaces as a failed renewal.
        if not self.lost and time.monotonic() - self.renewed > self.ttl / 4:
            self.renew()
        super().check()

    def release(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        with self._io:
            if (
                not self.lost
                and self._put(self.info(released=True), IfMatch=self.etag) is None
            ):
                log.warning(
                    "writer lock on %s was taken over before release", self.root
                )


def _status(e: Exception) -> int | None:
    resp = getattr(e, "response", None) or {}
    return resp.get("ResponseMetadata", {}).get("HTTPStatusCode")


_held: dict[str, _Lock] = {}
_guard = threading.Lock()


@contextlib.contextmanager
def writer_lock(root, wait: float | None = None, ttl: float | None = None):
    """Hold the writer lock on `root` for the block; re-entrant within the process.

    Waits up to `wait` seconds (default $EGOVERSE_LANCE_LOCK_WAIT or 30 min) for another
    writer, then raises LockHeld. `ttl` (default $EGOVERSE_LANCE_LOCK_TTL or 10 min) bounds
    how long a dead holder blocks others on R2."""
    key = _key(root)
    with _guard:
        lk = _held.get(key)
        if lk is not None:
            lk.depth += 1
    if lk is None:
        wait = (
            _env_seconds("EGOVERSE_LANCE_LOCK_WAIT", DEFAULT_WAIT_S)
            if wait is None
            else wait
        )
        ttl = (
            _env_seconds("EGOVERSE_LANCE_LOCK_TTL", DEFAULT_TTL_S)
            if ttl is None
            else ttl
        )
        holder = getpass.getuser()
        lk = (
            _S3Lock(key, ttl, holder)
            if _is_remote(key)
            else _FileLock(key, ttl, holder)
        )
        lk.acquire(wait)
        with _guard:
            lk.depth = 1
            _held[key] = lk
    try:
        lk.check()
        yield lk
    finally:
        with _guard:
            lk.depth -= 1
            last = lk.depth == 0
            if last:
                _held.pop(key, None)
        if last:
            lk.release()


def check(uri) -> None:
    """Raise LockLost if the hold covering `uri` (a root or a table under it) has lapsed.
    A write outside any hold is let through: only writers that took the lock are checked."""
    uri = _key(uri)
    with _guard:
        held = [lk for k, lk in _held.items() if uri == k or uri.startswith(k + "/")]
    for lk in held:
        lk.check()
