"""Verify minmax-normalized action targets lie within [-1, 1].

Scans every episode's raw `actions` in the nc3 zarr store, applies the exact
minmax normalization used by norm_mode='minmax'
(x_norm = 2*(x - min)/(max - min + 1e-6) - 1) with the SAME stored min/max the
training run loads, and reports the global per-dim normalized min/max.
"""
import json
import glob
import os
import numpy as np
import zarr

NC3 = "/coc/flash7/paphiwetsa3/datasets/new_circle_3"
NORM = "/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/hnet_smoke/fused_nochunk_nc3_5ep_2026-05-30_23-34-47/norm_stats/norm_stats.json"

stats = json.load(open(NORM))
a = stats["stats"]["15"]["actions"]
mn = np.asarray(a["min"], dtype=np.float64)
mx = np.asarray(a["max"], dtype=np.float64)
print("stored action min:", mn.tolist())
print("stored action max:", mx.tolist())

# Find zarr episode stores under NC3
cands = sorted(glob.glob(os.path.join(NC3, "**", "actions"), recursive=True))
# Each `actions` is an array within a zarr group; open parent group.
group_paths = sorted(set(os.path.dirname(p) for p in cands))
print("num episode stores found:", len(group_paths))

gmin = np.array([np.inf, np.inf])
gmax = np.array([-np.inf, -np.inf])
raw_min = np.array([np.inf, np.inf])
raw_max = np.array([-np.inf, -np.inf])
n_frames = 0
n_out = 0
for gp in group_paths:
    try:
        g = zarr.open(gp, mode="r")
        acts = np.asarray(g["actions"])  # (T, 2)
    except Exception as e:
        print("skip", gp, e)
        continue
    if acts.ndim != 2 or acts.shape[-1] != mn.shape[0]:
        # try squeeze
        acts = acts.reshape(-1, mn.shape[0])
    raw_min = np.minimum(raw_min, acts.min(0))
    raw_max = np.maximum(raw_max, acts.max(0))
    norm = 2.0 * ((acts - mn) / (mx - mn + 1e-6)) - 1.0
    gmin = np.minimum(gmin, norm.min(0))
    gmax = np.maximum(gmax, norm.max(0))
    n_frames += acts.shape[0]
    n_out += int(((norm < -1.0) | (norm > 1.0)).any(axis=1).sum())

print("total frames scanned:", n_frames)
print("raw action range:  min", raw_min.tolist(), "max", raw_max.tolist())
print("NORMALIZED action range (minmax): min", gmin.tolist(), "max", gmax.tolist())
print("frames with ANY dim outside [-1,1]:", n_out,
      f"({100.0*n_out/max(n_frames,1):.4f}%)")
within = bool((gmin >= -1.0 - 1e-6).all() and (gmax <= 1.0 + 1e-6).all())
print("ALL TARGETS WITHIN [-1,1]:", within)
