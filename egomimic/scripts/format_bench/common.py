"""Loader timing shared by the zarr/Lance and LeRobot benches. Imports nothing from egomimic,
because the LeRobot bench runs in its own venv."""

import functools
import json
import multiprocessing as mp
import os
import socket
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import psutil
import torch
from torch.utils.data import DataLoader, RandomSampler, SubsetRandomSampler


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).parent,
            text=True,
        ).strip()
    except Exception:
        return ""


def emit(rec: dict, out: str | None):
    rec = {
        "node": socket.gethostname().split(".")[0],
        "git": git_sha(),
        "t": time.time(),
        **rec,
    }
    line = json.dumps(rec)
    print(line, flush=True)
    if out:
        with open(out, "a") as f:
            f.write(line + "\n")


def evict(*roots) -> int:
    """Drop our files from the page cache (no root needed). Returns bytes evicted."""
    total = 0
    for root in roots:
        root = Path(root)
        files = (
            [root] if root.is_file() else (p for p in root.rglob("*") if p.is_file())
        )
        for p in files:
            try:
                fd = os.open(p, os.O_RDONLY)
            except OSError:
                continue
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                total += os.fstat(fd).st_size
            finally:
                os.close(fd)
    return total


def tree_rss_mb() -> float:
    proc = psutil.Process(os.getpid())
    total = proc.memory_info().rss
    for c in proc.children(recursive=True):
        try:
            total += c.memory_info().rss
        except psutil.NoSuchProcess:
            pass
    return total / 2**20


class PeakRSS:
    """Samples the process tree's RSS every `period` seconds on a thread."""

    def __init__(self, period: float = 0.5):
        self.peak, self.period, self._stop = 0.0, period, threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, tree_rss_mb())
            self._stop.wait(self.period)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()
        self.peak = max(self.peak, tree_rss_mb())


def collate(batch):
    out = {}
    for k in batch[0]:
        vals = [b[k] for b in batch]
        out[k] = torch.stack(vals) if isinstance(vals[0], torch.Tensor) else vals
    return out


def _worker_init(worker_id, ready, counters, hook):
    """Marks the worker ready (dataset unpickled) and installs the decoder-cache counter."""
    ready[worker_id] = time.time()
    if hook:
        hook(counters, worker_id)


def run_loader(
    ds,
    *,
    workers: int,
    context: str | None,
    batch_size: int,
    batches: int,
    warmup: int,
    hook=None,
    seed: int = 0,
    indices: list[int] | None = None,
    collate_fn=collate,
) -> dict:
    """Phase timings and steady-state throughput of a shuffled DataLoader over `ds`. `hook` is a
    module-level function from `hooks.py`, called in each worker to count decoder-cache hits."""
    ctx = mp.get_context(context) if workers else mp.get_context()
    ready = ctx.Array("d", max(workers, 1), lock=False)
    # [hits, misses] per worker
    counters = ctx.Array("q", 2 * max(workers, 1), lock=False)
    g = torch.Generator().manual_seed(seed)
    if isinstance(ds, torch.utils.data.IterableDataset):
        # Streaming datasets shuffle themselves (shuffle buffer); DataLoader takes no sampler.
        sampler = None
    elif indices is None:
        sampler = RandomSampler(ds, generator=g)
    else:
        sampler = SubsetRandomSampler(indices, generator=g)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=workers,
        collate_fn=collate_fn,
        prefetch_factor=2 if workers else None,
        persistent_workers=workers > 0,
        multiprocessing_context=ctx if workers else None,
        worker_init_fn=functools.partial(
            _worker_init, ready=ready, counters=counters, hook=hook
        )
        if workers
        else None,
    )
    if not workers and hook:
        hook(counters, 0)
    with PeakRSS() as rss:
        t0 = time.time()
        it = iter(loader)
        t_iter = time.time()
        next(it)
        t_first = time.time()
        for _ in range(warmup):
            next(it)
        waits = []
        t1 = time.perf_counter()
        for _ in range(batches):
            tb = time.perf_counter()
            next(it)
            waits.append(time.perf_counter() - tb)
        dt = max(time.perf_counter() - t1, 1e-9)
    ready_t = [r for r in ready[:workers] if r > 0]
    c = np.asarray(counters[:]).reshape(-1, 2)
    hits, misses = int(c[:, 0].sum()), int(c[:, 1].sum())
    del it, loader
    return dict(
        iter_s=round(t_iter - t0, 3),
        worker_spawn_s=round(max(ready_t) - t0, 3) if ready_t else 0.0,
        first_batch_s=round(t_first - t0, 3),
        samples_per_s=round(batches * batch_size / dt, 1) if batches else None,
        batch_wait_p50_ms=round(float(np.percentile(waits, 50)) * 1e3, 1)
        if waits
        else None,
        batch_wait_p95_ms=round(float(np.percentile(waits, 95)) * 1e3, 1)
        if waits
        else None,
        peak_rss_mb=round(rss.peak),
        decoder_hits=hits,
        decoder_misses=misses,
        decoder_hit_rate=round(hits / (hits + misses), 3) if hits + misses else None,
    )
