#!/usr/bin/env python3
"""CPU verification of the ported duration codec (velocity_mode="dur") against
the rows it is meant to replace, on real chunks.

usage: e1_duration_check.py <episode_root> <n_eps> <chunks_per_ep>

Three reads, mirroring the verification section of Ryan's commit (EgoVerse-graph
6d1b5f93) with the parts that transfer to our bimanual codec:

1. EQUIVALENCE. `dur` and `logdur` are built from the same per-interval `seg`,
   so on GROUND-TRUTH tokens they must decode to the same actions. Ryan's
   analogue was "the duration decoder reproduces the velocity decoder's native
   actions to 1e-15". Any residual here is logdur's LOGDUR_CLIP truncating a
   segment, which is exactly the hold case he calls out, so the clip rate is
   reported alongside.
2. ROUND-TRIP. Codec-only E_time (no model) for every row, so the port cannot
   silently cost fidelity.
3. NOISE. His "duration decodes ~4x more accurately than velocity under matched
   noise" measurement. Noise is applied in each channel's OWN scale (the corpus
   std of that channel), which is what "matched in normalized token space"
   means for a model that trains on quantile-normalised tokens.
"""
import os, sys

import numpy as np
import zarr
from scipy.spatial.transform import Rotation as R

from egomimic.rldb.zarr.e1_arc_tokenizer import (
    ARM_LAYOUT, LOGDUR_CLIP, TokenizeBimanualArcLengthE1,
)

root, n_eps, per_ep = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
D, M, dt, W, H = 0.40, 100, 1 / 30, 200, 40
mk = lambda mode: TokenizeBimanualArcLengthE1(
    min_distance_unit=D, resampled_vector_length=M, dt=dt,
    velocity_norm="path", velocity_mode=mode)
toks = {m: mk(m) for m in ("profile", "logdur", "dur")}

eps = sorted(d for d in os.listdir(root) if os.path.isdir(f"{root}/{d}"))[:n_eps]
rng = np.random.default_rng(0)
chunks = []
for h in eps:
    g = zarr.open_group(f"{root}/{h}", mode="r")
    cols = []
    for side in ("left", "right"):
        x = np.asarray(g[f"{side}.obs_ee_pose"][:], dtype=np.float64)
        nq = np.linalg.norm(x[:, 3:7], axis=1, keepdims=True)
        q = np.where(nq > 1e-9, x[:, 3:7] / np.where(nq > 1e-9, nq, 1.0), np.array([1.0, 0, 0, 0]))
        ypr = R.from_quat(q[:, [1, 2, 3, 0]]).as_euler("ZYX")
        grip = np.asarray(g[f"{side}.obs_gripper"][:], dtype=np.float64).reshape(-1, 1)
        cols += [x[:, :3], ypr, grip]
    ep = np.concatenate(cols, axis=1)
    if len(ep) <= W:
        continue
    for s in rng.integers(0, len(ep) - W, size=per_ep):
        chunks.append(ep[s : s + W])
print(f"{len(chunks)} chunks from {len(eps)} episodes of {root}")

tokens = {m: [t.transform({"actions_cartesian": c.copy()})["actions_cartesian"] for c in chunks]
          for m, t in toks.items()}

# ---- 1. equivalence: dur vs logdur on true tokens --------------------------
d_dec, clip_frac, n_arm = [], 0, 0
for i, c in enumerate(chunks):
    a = toks["dur"].detokenize(tokens["dur"][i], action_horizon=H)
    b = toks["logdur"].detokenize(tokens["logdur"][i], action_horizon=H)
    d_dec.append(np.abs(a - b).max())
    for k in range(2):
        col = tokens["logdur"][i][:, 14 + k]
        clip_frac += int(np.count_nonzero(np.abs(col[1:]) >= LOGDUR_CLIP - 1e-9))
        n_arm += M - 1
d_dec = np.array(d_dec)
print(f"\n1. EQUIVALENCE dur vs logdur on ground-truth tokens (metres, both arms, {H} frames)")
print(f"   max |diff| = {d_dec.max():.3e}   median = {np.median(d_dec):.3e}   "
      f"p99 = {np.quantile(d_dec, 0.99):.3e}")
print(f"   logdur segments at the clip: {clip_frac}/{n_arm} = {clip_frac / max(n_arm,1):.4%} "
      f"(where the two rows are ALLOWED to differ)")

# ---- 2. round-trip codec fidelity -----------------------------------------
print(f"\n2. ROUND-TRIP codec E_time over {H} frames (no model), metres")
base = {}
for m in toks:
    errs = []
    for i, c in enumerate(chunks):
        dec = toks[m].detokenize(tokens[m][i], action_horizon=H)
        for xo, _, _, _ in ARM_LAYOUT:
            errs.append(np.sqrt(np.mean(np.sum((dec[:, xo:xo+3] - c[:H, xo:xo+3]) ** 2, axis=1))))
    base[m] = float(np.mean(errs))
    print(f"   {m:8s} E_time = {base[m]:.5f}   p90 = {np.quantile(errs, 0.9):.5f}")

# ---- 3. matched-noise decode robustness -----------------------------------
scale = {m: float(np.std(np.concatenate([t[:, 14:16].ravel() for t in tokens[m]]))) for m in toks}
print(f"\n3. NOISE on the timing channel only, sigma = f x that channel's corpus std")
print(f"   channel std: " + "  ".join(f"{m}={scale[m]:.4g}" for m in toks))
print(f"   {'f':>6} " + "".join(f"{m:>12}" for m in toks) + "   (E_time, m)")
for f in (0.05, 0.1, 0.25, 0.5):
    line = f"   {f:>6.2f} "
    for m in toks:
        rng2 = np.random.default_rng(1234)
        errs = []
        for i, c in enumerate(chunks):
            tk = tokens[m][i].copy()
            tk[:, 14:16] += rng2.normal(0.0, f * scale[m], size=(M, 2))
            dec = toks[m].detokenize(tk, action_horizon=H)
            for xo, _, _, _ in ARM_LAYOUT:
                errs.append(np.sqrt(np.mean(np.sum((dec[:, xo:xo+3] - c[:H, xo:xo+3]) ** 2, axis=1))))
        line += f"{np.mean(errs):>12.5f}"
    print(line)
print("\nDURATION_CHECK_DONE")
