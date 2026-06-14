"""Prove InMemoryZarrDataset == ZarrDataset, sample-for-sample.

Run on a skynet compute node inside the EgoVerse2 env:
    python test_inmem_equiv.py /coc/flash7/paphiwetsa3/datasets/new_circle_3
"""
import sys
import time

import numpy as np
import torch

from egomimic.rldb.embodiment.pushshapes import get_keymap
from egomimic.rldb.zarr.zarr_dataset_multi import ZarrDataset
from egomimic.rldb.zarr.zarr_dataset_inmem import InMemoryZarrDataset


def episode_dirs(root):
    import os
    out = []
    for n in sorted(os.listdir(root)):
        p = os.path.join(root, n)
        if os.path.isdir(p) and n.endswith(".zarr"):
            out.append(p)
    return out


def compare(root, n_eps=3):
    km = get_keymap(action_horizon=32)
    eps = episode_dirs(root)[:n_eps]
    assert eps, f"no .zarr episodes under {root}"

    total_checks = 0
    for ep in eps:
        base = ZarrDataset(ep, key_map=km)
        t0 = time.time()
        mem = InMemoryZarrDataset(ep, key_map=km)
        preload_s = time.time() - t0
        T = base.total_frames
        assert mem.total_frames == T, f"len mismatch {mem.total_frames} != {T}"

        # Check a spread of indices incl. the last frames (padding path).
        idxs = sorted(set([0, 1, T // 2, T - 32, T - 2, T - 1]))
        idxs = [i for i in idxs if 0 <= i < T]
        for i in idxs:
            a = base[i]
            b = mem[i]
            assert set(a.keys()) == set(b.keys()), (
                f"key mismatch ep={ep} idx={i}: {set(a.keys())} vs {set(b.keys())}"
            )
            for k in a:
                va, vb = a[k], b[k]
                if isinstance(va, torch.Tensor):
                    assert isinstance(vb, torch.Tensor), f"{k} type mismatch"
                    assert va.shape == vb.shape, (
                        f"{k} shape {va.shape} vs {vb.shape} ep={ep} idx={i}"
                    )
                    assert va.dtype == vb.dtype, f"{k} dtype {va.dtype} vs {vb.dtype}"
                    maxdiff = (va - vb).abs().max().item() if va.numel() else 0.0
                    assert torch.allclose(va, vb, atol=1e-6, rtol=1e-5), (
                        f"{k} VALUE mismatch ep={ep} idx={i} maxdiff={maxdiff}"
                    )
                else:
                    assert va == vb, f"{k} scalar mismatch {va} vs {vb}"
                total_checks += 1
        print(
            f"OK  {ep.split('/')[-1]}  T={T}  preload={preload_s:.2f}s  "
            f"checked idxs={idxs}"
        )

    print(f"\nALL EQUAL — {total_checks} tensor/scalar comparisons passed.")


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "/coc/flash7/paphiwetsa3/datasets/new_circle_3"
    compare(root)
