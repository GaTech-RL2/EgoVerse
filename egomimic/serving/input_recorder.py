"""
Opt-in recording of policy-server inputs (and outputs) for offline analysis.

Why server-side: the hardware rollout client never persists its observation
dict, so after a rollout there is no record of what the policy actually saw
(operator decision 2026-08-26 after the dp3c_dual session left no obs log).
Recording at the server catches every client (hardware, sim, replay) with zero
client changes.

Design constraints (hardware inference must stay robust):
  * OFF (the default) is a single ``is not None`` check in the handler —
    identical code path, zero cost.
  * ON never blocks and never raises into serving: the inference thread only
    copies the obs arrays and does a non-blocking put onto a bounded queue;
    all disk I/O happens in a daemon writer thread. Queue full -> the record
    is dropped and counted (warned once). Any writer error -> counted, logged
    once, serving continues.
  * Crash-safe files: uncompressed ``np.savez`` per chunk (fast), written to a
    temp name then ``os.replace`` (atomic). ``meta.json`` is written at start
    so even a crashed session has identity; counters are refreshed on close.
  * Bounded: chunk by count or time, bounded queue, optional ``every_n``
    thinning and a ``max_gb`` cap (stop recording, warn once, keep serving).
  * Rollout-attributable: each websocket connection gets an id; open/close
    events go to ``connections.jsonl`` and every record carries ``conn_id``.

Layout of a session directory::

    <session>/meta.json            # ckpt, port, server metadata, counters
    <session>/connections.jsonl    # {"conn_id", "event", "t_wall", "remote"}
    <session>/chunk_0000.npz ...   # stacked per-key arrays + t_wall/conn_id/
                                   # infer_ms/idx; ragged keys -> object arrays

``load_session(dir)`` reads it all back as one dict of concatenated arrays.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import queue
import threading
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_META_NAME = "meta.json"
_CONN_NAME = "connections.jsonl"
_CHUNK_FMT = "chunk_{:04d}.npz"
_ALWAYS_KEYS = ("idx", "t_wall", "conn_id", "infer_ms")


def _copy_value(v):
    """Detach a wire value from the request buffer. Arrays copied, scalars kept."""
    if isinstance(v, np.ndarray):
        return np.array(v, copy=True)
    if isinstance(v, (list, tuple)):
        try:
            return np.array(v)
        except Exception:  # noqa: BLE001
            return np.array(v, dtype=object)
    return v


class InputRecorder:
    def __init__(
        self,
        session_dir: str | os.PathLike,
        session_meta: dict | None = None,
        *,
        chunk_size: int = 100,
        flush_s: float = 5.0,
        queue_max: int = 2000,
        every_n: int = 1,
        max_gb: float | None = None,
    ) -> None:
        self.dir = Path(session_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = max(1, int(chunk_size))
        self.flush_s = float(flush_s)
        self.every_n = max(1, int(every_n))
        self.max_bytes = None if max_gb is None else float(max_gb) * (1024 ** 3)

        self._q: queue.Queue = queue.Queue(maxsize=int(queue_max))
        self._stop = threading.Event()
        self._capped = threading.Event()
        self._lock = threading.Lock()
        # counters (read by close()/status; written by both threads under lock
        # or by a single owner)
        self.n_seen = 0          # record() calls (inference thread)
        self.n_recorded = 0      # enqueued
        self.n_dropped = 0       # queue full / capped
        self.n_written = 0       # records on disk (writer thread)
        self.n_chunks = 0
        self.bytes_written = 0
        self.n_errors = 0
        self._warned_full = False
        self._warned_err = False
        self._warned_cap = False
        self._t0 = time.time()

        self.meta = dict(session_meta or {})
        self.meta.update({
            "started_at": self._t0,
            "started_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self._t0)),
            "chunk_size": self.chunk_size,
            "every_n": self.every_n,
            "max_gb": max_gb,
        })
        self._write_meta()

        self._thread = threading.Thread(
            target=self._writer, name="input-recorder", daemon=True)
        self._thread.start()
        atexit.register(self.close)
        logger.info("[recorder] session %s (chunk %d / %.1fs, every_n=%d, max_gb=%s)",
                    self.dir, self.chunk_size, self.flush_s, self.every_n, max_gb)

    # ------------------------------------------------------------------ hot path
    def record(self, obs, action, *, conn_id: int = -1, infer_ms: float = float("nan")) -> None:
        """Enqueue one inference (or one per element for batch lists). Never raises."""
        try:
            if isinstance(obs, list):
                acts = action if isinstance(action, list) else [action] * len(obs)
                for o, a in zip(obs, acts):
                    self._record_one(o, a, conn_id, infer_ms)
            else:
                self._record_one(obs, action, conn_id, infer_ms)
        except Exception as e:  # noqa: BLE001 — recording must never break serving
            self.n_errors += 1
            if not self._warned_err:
                self._warned_err = True
                logger.warning("[recorder] record() error (suppressed hereafter): %r", e)

    def _record_one(self, obs, action, conn_id, infer_ms) -> None:
        self.n_seen += 1
        if self._capped.is_set() or (self.n_seen - 1) % self.every_n:
            return
        rec = {"idx": self.n_seen - 1, "t_wall": time.time(),
               "conn_id": int(conn_id), "infer_ms": float(infer_ms)}
        if isinstance(obs, dict):
            for k, v in obs.items():
                rec[f"obs/{k}"] = _copy_value(v)
        if isinstance(action, dict):
            for k, v in action.items():
                if k == "server_timing":
                    continue
                rec[f"act/{k}"] = _copy_value(v)
        elif action is not None:
            rec["act/actions"] = _copy_value(action)
        try:
            self._q.put_nowait(rec)
            self.n_recorded += 1
        except queue.Full:
            self.n_dropped += 1
            if not self._warned_full:
                self._warned_full = True
                logger.warning("[recorder] queue full — dropping records (disk too slow?)")

    def on_connection(self, conn_id: int, event: str, remote=None) -> None:
        """Append a connection open/close event (rare; tiny; direct write)."""
        try:
            with open(self.dir / _CONN_NAME, "a") as f:
                f.write(json.dumps({"conn_id": int(conn_id), "event": event,
                                    "t_wall": time.time(),
                                    "remote": str(remote)}) + "\n")
        except Exception as e:  # noqa: BLE001
            self.n_errors += 1
            logger.warning("[recorder] connections.jsonl write failed: %r", e)

    # ------------------------------------------------------------------ writer
    def _writer(self) -> None:
        buf: list[dict] = []
        last_flush = time.time()
        while True:
            timeout = max(0.05, self.flush_s - (time.time() - last_flush))
            try:
                buf.append(self._q.get(timeout=timeout))
            except queue.Empty:
                pass
            due = len(buf) >= self.chunk_size or (buf and time.time() - last_flush >= self.flush_s)
            if due:
                self._flush(buf)
                buf = []
                last_flush = time.time()
            if self._stop.is_set() and self._q.empty():
                if buf:
                    self._flush(buf)
                return

    def _flush(self, buf: list[dict]) -> None:
        if not buf:
            return
        if self._capped.is_set():
            # cap means "stop writing", including records already queued
            self.n_dropped += len(buf)
            return
        try:
            arrays = _stack_records(buf)
            path = self.dir / _CHUNK_FMT.format(self.n_chunks)
            tmp = path.with_suffix(".npz.tmp")
            with open(tmp, "wb") as f:
                np.savez(f, **arrays)
            os.replace(tmp, path)
            with self._lock:
                self.n_chunks += 1
                self.n_written += len(buf)
                self.bytes_written += path.stat().st_size
            if self.max_bytes is not None and self.bytes_written >= self.max_bytes \
                    and not self._capped.is_set():
                self._capped.set()
                logger.warning("[recorder] max_gb reached (%.2f GB) — recording stopped, "
                               "serving continues", self.bytes_written / 1024 ** 3)
            self._write_meta()   # live counters survive a hard kill
            if self.n_chunks % 10 == 1:
                dt = max(1e-6, time.time() - self._t0)
                logger.info("[recorder] %d records, %d chunks, %.1f MB (%.2f MB/s)",
                            self.n_written, self.n_chunks,
                            self.bytes_written / 1e6, self.bytes_written / 1e6 / dt)
        except Exception as e:  # noqa: BLE001
            self.n_errors += 1
            self.n_dropped += len(buf)
            if not self._warned_err:
                self._warned_err = True
                logger.warning("[recorder] chunk write failed (suppressed hereafter): %r", e)

    def _write_meta(self) -> None:
        try:
            m = dict(self.meta)
            m.update(records_seen=self.n_seen, records_recorded=self.n_recorded,
                     records_written=self.n_written, records_dropped=self.n_dropped,
                     chunks=self.n_chunks, bytes_written=self.bytes_written,
                     errors=self.n_errors)
            tmp = self.dir / (_META_NAME + ".tmp")
            tmp.write_text(json.dumps(m, indent=2, default=str))
            os.replace(tmp, self.dir / _META_NAME)
        except Exception as e:  # noqa: BLE001
            logger.warning("[recorder] meta.json write failed: %r", e)

    def close(self, timeout: float = 10.0) -> None:
        if self._stop.is_set():
            return
        self._stop.set()
        self._thread.join(timeout=timeout)
        self._write_meta()
        logger.info("[recorder] closed: %d written / %d dropped / %d chunks / %.1f MB -> %s",
                    self.n_written, self.n_dropped, self.n_chunks,
                    self.bytes_written / 1e6, self.dir)

    def status(self) -> dict:
        return {"seen": self.n_seen, "recorded": self.n_recorded, "written": self.n_written,
                "dropped": self.n_dropped, "chunks": self.n_chunks,
                "bytes": self.bytes_written, "errors": self.n_errors}


# ---------------------------------------------------------------------- helpers
def _stack_records(buf: list[dict]) -> dict[str, np.ndarray]:
    """Stack per-key across records; ragged/missing keys become object arrays."""
    keys: list[str] = []
    for r in buf:
        for k in r:
            if k not in keys:
                keys.append(k)
    out: dict[str, np.ndarray] = {}
    for k in keys:
        vals = [r.get(k) for r in buf]
        present = [v for v in vals if v is not None]
        if len(present) == len(vals) and all(isinstance(v, np.ndarray) for v in vals):
            s0, d0 = vals[0].shape, vals[0].dtype
            if all(v.shape == s0 and v.dtype == d0 for v in vals) and d0 != object:
                out[k] = np.stack(vals)
                continue
        if len(present) == len(vals) and all(np.isscalar(v) for v in vals):
            out[k] = np.asarray(vals)
            continue
        arr = np.empty(len(vals), dtype=object)
        for i, v in enumerate(vals):
            arr[i] = v
        out[k] = arr
    return out


def load_session(session_dir: str | os.PathLike) -> dict:
    """All chunks concatenated per key (+ 'meta'). Object keys stay object."""
    d = Path(session_dir)
    chunks = sorted(d.glob("chunk_*.npz"))
    if not chunks:
        raise FileNotFoundError(f"no chunk_*.npz in {d}")
    parts: dict[str, list[np.ndarray]] = {}
    for p in chunks:
        with np.load(p, allow_pickle=True) as z:
            for k in z.files:
                parts.setdefault(k, []).append(z[k])
    out: dict = {}
    for k, lst in parts.items():
        try:
            out[k] = np.concatenate(lst, axis=0)
        except Exception:  # noqa: BLE001 — mixed shapes across chunks
            arr = np.empty(sum(len(a) for a in lst), dtype=object)
            i = 0
            for a in lst:
                for v in a:
                    arr[i] = v
                    i += 1
            out[k] = arr
    meta_p = d / _META_NAME
    out["meta"] = json.loads(meta_p.read_text()) if meta_p.is_file() else {}
    conn_p = d / _CONN_NAME
    out["connections"] = ([json.loads(l) for l in conn_p.read_text().splitlines() if l.strip()]
                          if conn_p.is_file() else [])
    return out


def make_session_dir(root: str | os.PathLike, ckpt: str, port: int) -> Path:
    """<root>/<run_tag>_<port>_<YYYYmmdd_HHMMSS>  (run_tag from the ckpt path)."""
    parts = Path(ckpt).parts
    tag = parts[-4] if len(parts) >= 4 else Path(ckpt).stem      # RBY1_<name>
    tag = str(tag).replace("RBY1_", "")
    return Path(root) / f"{tag}_{port}_{time.strftime('%Y%m%d_%H%M%S')}"
