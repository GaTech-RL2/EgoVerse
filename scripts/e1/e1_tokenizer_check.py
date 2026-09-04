#!/usr/bin/env python3
"""Check the E1 vectorized tokenizer against the lab's TokenizeBimanualArcLengthCartesian
on real chunks (same D / M / dt, chord-norm, mean mode): max |diff| of the (M+1, 14) token
and of detokenize(token, H), plus per-sample timings of both."""
import sys
import time

import numpy as np
import zarr
from scipy.spatial.transform import Rotation as R

from egomimic.rldb.zarr.arc_length_tokenizer import TokenizeBimanualArcLengthCartesian
from egomimic.rldb.zarr.e1_arc_tokenizer import TokenizeBimanualArcLengthE1

root, n_eps, per_ep = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
hashes = [l.strip() for l in open(sys.argv[4])][:n_eps]
D, M, dt, W, H = 0.40, 100, 1 / 30, 200, 40
parent = TokenizeBimanualArcLengthCartesian(min_distance_unit=D, resampled_vector_length=M, dt=dt)
fast = TokenizeBimanualArcLengthE1(min_distance_unit=D, resampled_vector_length=M, dt=dt, velocity_norm="chord", velocity_mode="mean")
rng = np.random.default_rng(0)
d_tok, d_det, t_par, t_fast, n = [], [], 0.0, 0.0, 0
for h in hashes:
    g = zarr.open_group(f"{root}/{h}", mode="r")
    cols = []
    for side in ("left", "right"):
        x = np.asarray(g[f"{side}.obs_ee_pose"][:], dtype=np.float64)
        q = x[:, 3:7] / np.linalg.norm(x[:, 3:7], axis=1, keepdims=True)
        ypr = R.from_quat(q[:, [1, 2, 3, 0]]).as_euler("ZYX")
        cols += [x[:, :3], ypr, np.zeros((len(x), 1))]
    ep = np.concatenate(cols, axis=1)
    for s in rng.integers(0, len(ep) - W, size=per_ep):
        chunk = ep[s : s + W]
        t0 = time.perf_counter(); a = parent.transform({"actions_cartesian": chunk.copy()})["actions_cartesian"]; t_par += time.perf_counter() - t0
        t0 = time.perf_counter(); b = fast.transform({"actions_cartesian": chunk.copy()})["actions_cartesian"]; t_fast += time.perf_counter() - t0
        d_tok.append(np.abs(a - b).max())
        da = parent.detokenize(a, action_horizon=H); db = fast.detokenize(b, action_horizon=H)
        d_det.append(np.abs(da - db).max())
        n += 1
print(f"{n} chunks: token max|diff| = {max(d_tok):.3e} (median {np.median(d_tok):.3e}); detok max|diff| = {max(d_det):.3e}")
print(f"per-sample: parent {1e3 * t_par / n:.1f} ms, fast {1e3 * t_fast / n:.2f} ms (x{t_par / max(t_fast, 1e-9):.0f})")
ok = max(d_tok) < 1e-6 and max(d_det) < 1e-6
print("TOKENIZER_EQUIV_OK" if ok else "TOKENIZER_MISMATCH")
sys.exit(0 if ok else 1)
