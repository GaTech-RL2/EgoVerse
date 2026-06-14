"""Do the proposed FPL preference axes carry independent signal on real data?

Computes each axis over a dataset of episodes, then reports:
  1. distributions      -- is there any spread to prefer over?
  2. correlation matrix -- are the axes measuring different things?
  3. disagreement rate  -- how often do axes actually conflict on a pair?

(3) is the one that decides whether multi-axis beats single-preference. If the
axes almost never disagree, every axis-conditioned reward learns the same
function and FPL degenerates to binary preference learning.
"""

import glob
import json
import re
import sys

import numpy as np
import zarr

D = sys.argv[1]
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 0

paths = sorted(glob.glob(f"{D}/*.zarr"))
if LIMIT:
    paths = paths[:LIMIT]
print(f"episodes: {len(paths)}", flush=True)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def lowpass(p, w=9):
    if len(p) < w:
        return p
    k = np.ones(w) / w
    return np.stack([np.convolve(p[:, i], k, "valid") for i in range(2)], axis=1)


rows, row_tags, bad = [], [], 0
for i, p in enumerate(paths):
    try:
        g = zarr.open(p, mode="r")
        at = dict(g.attrs)
        n = int(at["total_frames"])
        if n < 12:
            bad += 1
            continue
        st = np.asarray(g["observations.state"][:n], float)
        pusher, obj, th = st[:, 0:2], st[:, 2:4], st[:, 4]
        cov = np.asarray(g["reward"][:n], float).reshape(-1)
        goal = np.asarray(g["goal_pose"][:n], float)

        # 1 translational alignment (lower better)
        trans = float(np.linalg.norm(obj[-1] - goal[-1, :2]))
        # 2 rotational alignment, degrees (lower better)
        rot = float(np.degrees(abs(wrap(th[-1] - goal[-1, 2]))))
        # 3 speed = duration in frames (lower better)
        dur = float(n)
        # 4 directness: net coverage gain / total variation (higher better)
        d = np.diff(cov)
        d = d[np.abs(d) > 1e-4]
        tv = np.abs(d).sum()
        direct = float((cov[-1] - cov[0]) / tv) if tv > 1e-12 else 1.0
        # 5 gentleness: peak filtered object speed (lower better)
        fo = lowpass(obj)
        gentle = float(np.linalg.norm(np.diff(fo, axis=0), axis=1).max()) if len(fo) > 1 else 0.0
        # 6 smoothness: filtered pusher reversals (lower better)
        fp = lowpass(pusher)
        v = np.diff(fp, axis=0)
        v = v[np.linalg.norm(v, axis=1) > 2.0]
        if len(v) < 2:
            rev = 0.0
        else:
            u = v / np.linalg.norm(v, axis=1, keepdims=True)
            rev = float((np.sum(u[1:] * u[:-1], axis=1) < 0).sum())

        rows.append([trans, rot, dur, direct, gentle, rev, cov[-1]])
        _m = re.match(r'^episode_.+?_obs\d+_([A-Za-z0-9]+)_\d+\.zarr$', p.split('/')[-1])
        row_tags.append(_m.group(1) if _m else 'untagged')
    except Exception as e:
        bad += 1
        if bad < 3:
            print("  skip", p.split("/")[-1], repr(e)[:90], flush=True)
    if (i + 1) % 500 == 0:
        print(f"  {i+1}/{len(paths)}", flush=True)

M = np.array(rows)
print(f"\nparsed {len(M)} episodes ({bad} skipped)\n")

NAMES = ["translation", "rotation", "duration", "directness", "gentleness", "smoothness"]
# True = lower is better
LOWER = [True, True, True, False, True, True]

print("=" * 78)
print("1. DISTRIBUTIONS  (is there spread to prefer over?)")
print("=" * 78)
print(f"{'axis':13}{'min':>10}{'p25':>10}{'median':>10}{'p75':>10}{'max':>10}{'CV':>9}")
for j, nm in enumerate(NAMES):
    c = M[:, j]
    cv = c.std() / abs(c.mean()) if abs(c.mean()) > 1e-9 else 0.0
    print(f"{nm:13}{c.min():10.3f}{np.percentile(c,25):10.3f}{np.median(c):10.3f}"
          f"{np.percentile(c,75):10.3f}{c.max():10.3f}{cv:9.3f}")
print(f"\nfinal coverage: median={np.median(M[:,6]):.3f}  "
      f"frac >=0.95: {(M[:,6]>=0.95).mean():.3f}")

print("\n" + "=" * 78)
print("2. CORRELATION  (|r|>0.8 => redundant axis)")
print("=" * 78)
C = np.corrcoef(M[:, :6].T)
print(f"{'':13}" + "".join(f"{n[:9]:>11}" for n in NAMES))
for j, nm in enumerate(NAMES):
    print(f"{nm:13}" + "".join(f"{C[j,k]:11.2f}" for k in range(6)))

flag = [(NAMES[a], NAMES[b], C[a, b]) for a in range(6) for b in range(a + 1, 6)
        if abs(C[a, b]) > 0.8]
print("\nredundant pairs:", flag if flag else "none above |r|=0.8")

print("\n" + "=" * 78)
print("3. DISAGREEMENT  (do axes actually conflict on a pair?)")
print("=" * 78)
rng = np.random.default_rng(0)
NP_ = 4000
ii = rng.integers(0, len(M), NP_)
jj = rng.integers(0, len(M), NP_)
keep = ii != jj
ii, jj = ii[keep], jj[keep]

# deadband per axis: 10% of that axis's std -> below this, call it a tie
eps = 0.10 * M[:, :6].std(axis=0)

pref = np.zeros((len(ii), 6), dtype=int)  # +1 A better, -1 B better, 0 tie
for k in range(6):
    da = M[ii, k] - M[jj, k]
    better_a = (da < 0) if LOWER[k] else (da > 0)
    pref[:, k] = np.where(np.abs(da) < eps[k], 0, np.where(better_a, 1, -1))

nz = [(pref[:, k] != 0).mean() for k in range(6)]
print("fraction of pairs where each axis gives a (non-tie) label:")
for k, nm in enumerate(NAMES):
    print(f"  {nm:13}{nz[k]:6.3f}")

pos = (pref > 0).sum(1)
neg = (pref < 0).sum(1)
disagree = ((pos > 0) & (neg > 0)).mean()
unanimous = (((pos > 0) & (neg == 0)) | ((neg > 0) & (pos == 0))).mean()
print(f"\npairs where axes CONFLICT (A wins some, B wins others): {disagree:.3f}")
print(f"pairs where all labelled axes AGREE:                    {unanimous:.3f}")

print("\nsame, for your current 4 axes only (trans, rot, duration, directness):")
p4 = pref[:, :4]
d4 = (((p4 > 0).sum(1) > 0) & ((p4 < 0).sum(1) > 0)).mean()
print(f"  conflict rate: {d4:.3f}")


# ------------------------------------------------------------ per-mode table
tags = np.array(row_tags)

if len(set(tags)) > 1:
    print("\n" + "=" * 78)
    print("4. PER-MODE PROFILE  (median per axis; does each mode hit its target?)")
    print("=" * 78)
    print(f"{'mode':14}{'n':>5}" + "".join(f"{n[:10]:>12}" for n in NAMES) + f"{'coverage':>11}")
    for tg in sorted(set(tags)):
        sel = M[tags == tg]
        print(f"{tg:14}{len(sel):5d}" +
              "".join(f"{np.median(sel[:, k]):12.2f}" for k in range(6)) +
              f"{np.median(sel[:, 6]):11.3f}")
