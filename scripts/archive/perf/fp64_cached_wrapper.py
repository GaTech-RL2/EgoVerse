"""Combined: global cache dict + fp64 auto-cast."""
import torch
import numpy as np
import time
from pathlib import Path

# === Patch 1: Auto-cast fp32→fp64 ===
_orig_module_call = torch.nn.Module.__call__
def _auto_cast_call(self, *args, **kwargs):
    try:
        p = next(self.parameters())
        if p.dtype == torch.float64:
            args = tuple(a.double() if isinstance(a, torch.Tensor) and a.is_floating_point() else a for a in args)
            kwargs = {k: v.double() if isinstance(v, torch.Tensor) and v.is_floating_point() else v for k, v in kwargs.items()}
    except StopIteration:
        pass
    return _orig_module_call(self, *args, **kwargs)
torch.nn.Module.__call__ = _auto_cast_call

# === Patch 2: Global cache for ZarrEpisode ===
import egomimic.rldb.zarr.zarr_dataset_multi as zdm

_CACHE = {}

_orig_ep_init = zdm.ZarrEpisode.__init__
def _cached_ep_init(self, path):
    _orig_ep_init(self, path)
    key = str(path)
    if key in _CACHE:
        return
    t0 = time.time()
    store = self._get_store()
    cache = {}
    for k in store.keys():
        try:
            cache[k] = np.asarray(store[k][:])
        except Exception:
            pass
    _CACHE[key] = cache
    dt = time.time() - t0
    mb = sum(a.nbytes for a in cache.values() if hasattr(a, 'nbytes')) / 1e6
    print(f"  [cache] {Path(path).name}: {len(cache)} keys, {mb:.1f}MB, {dt:.2f}s", flush=True)

_orig_read = zdm.ZarrEpisode.read
def _cached_read(self, keys_with_ranges):
    cache = _CACHE.get(str(self._path))
    if not cache:
        return _orig_read(self, keys_with_ranges)
    result = {}
    for key, (start, end) in keys_with_ranges.items():
        if key in cache:
            arr = cache[key]
            if end is not None:
                result[key] = arr[start:end]
            else:
                result[key] = arr[start]
        else:
            store = self._get_store()
            arr = store[key]
            if end is not None:
                result[key] = np.asarray(arr[start:end])
            else:
                result[key] = np.asarray(arr[start])
    return result

zdm.ZarrEpisode.__init__ = _cached_ep_init
zdm.ZarrEpisode.read = _cached_read
print("[fp64_cached_wrapper] Global cache + auto-cast patched", flush=True)

import runpy
runpy.run_module("egomimic.trainHydra", run_name="__main__")
