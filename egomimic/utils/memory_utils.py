import os, time, platform, resource, torch, psutil

def _fmt_bytes(n):
    for unit in ["B","KB","MB","GB"]:
        if abs(n) < 1024.0:
            return f"{n:6.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}TB"

def _rss_bytes():
    kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return kb if platform.system()=="Darwin" else kb*1024

def _cuda_bytes():
    if torch.cuda.is_available():
        dev = torch.cuda.current_device()
        alloc = torch.cuda.memory_allocated(dev)
        res = torch.cuda.memory_reserved(dev)
        peak = torch.cuda.max_memory_allocated(dev)
        return alloc, res, peak
    return 0, 0, 0

def _print_mem(label, t0):
    rss = _rss_bytes()
    alloc, res, peak = _cuda_bytes()
    print(f"[{label:20s}]  Δt={time.time()-t0:6.2f}s | "
          f"RSS={_fmt_bytes(rss)} | "
          f"CUDA alloc={_fmt_bytes(alloc)}, res={_fmt_bytes(res)}, peak={_fmt_bytes(peak)}")
