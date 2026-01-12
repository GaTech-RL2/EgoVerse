import os
import psutil
import threading
import time
import re
from contextlib import contextmanager

_root = psutil.Process(os.getpid())

def _proc_rss_mb(p: psutil.Process) -> float:
    return p.memory_info().rss / (1024 ** 2)

def cgroup_memory_peak_mb() -> float | None:
    # cgroup v2
    candidates = [
        "/sys/fs/cgroup/memory.peak",
        "/sys/fs/cgroup/memory.max_usage_in_bytes",  # older v1
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                with open(p, "r") as f:
                    return int(f.read().strip()) / (1024 ** 2)
            except (OSError, ValueError):
                pass
    return None


def _read_smaps_rollup_kb(pid: int) -> dict[str, int]:
    out = {}
    path = f"/proc/{pid}/smaps_rollup"
    with open(path, "r") as f:
        for line in f:
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            v = v.strip().split()
            if len(v) >= 2 and v[1] == "kB":
                out[k] = int(v[0])
    return out

def tree_pss_mb() -> float:
    procs = [_root]
    try:
        procs += _root.children(recursive=True)
    except psutil.Error:
        pass

    total_kb = 0
    for p in procs:
        try:
            d = _read_smaps_rollup_kb(p.pid)
            if "Pss" in d:
                total_kb += d["Pss"]
            else:
                # fallback
                total_kb += p.memory_info().rss // 1024
        except Exception:
            pass
    return total_kb / 1024.0

def tree_mem_mb(include_children: bool = True, use_uss: bool = True) -> float:
    root = psutil.Process(os.getpid())
    procs = [root]
    if include_children:
        try:
            procs += root.children(recursive=True)
        except Exception:
            pass

    total = 0
    for p in procs:
        try:
            if use_uss and hasattr(p, "memory_full_info"):
                total += p.memory_full_info().uss
            else:
                total += p.memory_info().rss
        except Exception:
            pass
    return total / (1024 ** 2)

class _Sampler:
    def __init__(self, interval_s: float = 0.025):
        self.interval_s = interval_s
        self.ts = []
        self.mbs = []
        self._stop = threading.Event()
        self._t = None
        self._errored = False

    def start(self):
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self):
        t0 = time.time()
        while not self._stop.is_set():
            t = time.time() - t0
            try:
                mb = tree_pss_mb()
            except Exception:
                self._errored = True
                time.sleep(self.interval_s)
                continue
            self.ts.append(t)
            self.mbs.append(mb)
            time.sleep(self.interval_s)

    def stop(self):
        self._stop.set()
        if self._t is not None:
            self._t.join()


@contextmanager
def mem_section(name: str, sample_interval_s: float = 0.2, plot: bool = True, enabled: bool = False):
    if not enabled:
        yield
        return

    start = tree_pss_mb()
    sampler = _Sampler(interval_s=sample_interval_s)
    sampler.start()
    t0 = time.time()
    try:
        yield
    finally:
        sampler.stop()
        end = tree_pss_mb()
        dt = time.time() - t0

        peak = max(sampler.mbs) if sampler.mbs else end
        print(f"[{name}] end={end:.2f} MB  delta={end-start:+.2f} MB  peak={peak:.2f} MB  time={dt:.2f}s")

        if plot and sampler.mbs and sampler.ts:
            import matplotlib.pyplot as plt
            n = min(len(sampler.ts), len(sampler.mbs))
            if n > 1:
                plt.plot(sampler.ts[:n], sampler.mbs[:n])
                plt.xlabel("time (s)")
                plt.ylabel("tree RSS (MB)")
                plt.tight_layout()
                plt.savefig(f"{_safe_name(name)}.png", dpi=150)
                plt.close()

def _safe_name(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s).strip("_")