#!/usr/bin/env python
"""Print the fresh minmax norm stats (per-dim min/max) over the bal200 subset,
for the proprio key state_agent_obj (observations.state, 5-d) and the action
key actions (2-d) -- exactly what trainHydra computes with norm_mode=minmax.

Reads only valid (non-padded) frames per episode using total_frames from attrs.
"""
import json
from pathlib import Path
import numpy as np
import zarr

DST = Path("/coc/flash7/paphiwetsa3/datasets/new_circle_3_bal200")

eps = sorted(p for p in DST.iterdir() if p.is_symlink() and p.name.endswith(".zarr"))
print(f"episodes: {len(eps)}", flush=True)

state_chunks = []
action_chunks = []
for ep in eps:
    g = zarr.open_group(str(ep.resolve()), mode="r")
    tf = int(dict(g.attrs).get("total_frames"))
    s = np.asarray(g["observations.state"][:tf])  # (tf, 5)
    a = np.asarray(g["actions"][:tf])              # (tf, 2)
    state_chunks.append(s)
    action_chunks.append(a)

S = np.concatenate(state_chunks, axis=0)
A = np.concatenate(action_chunks, axis=0)
print(f"total valid frames: state={S.shape}, actions={A.shape}", flush=True)

np.set_printoptions(precision=4, suppress=True)
print("\n=== FRESH MINMAX NORM STATS over bal200 subset ===")
print("state_agent_obj (observations.state) = [agent_x, agent_y, obj_x, obj_y, obj_theta]")
print(f"  min = {np.min(S, axis=0).tolist()}")
print(f"  max = {np.max(S, axis=0).tolist()}")
print("actions = [act_x, act_y]")
print(f"  min = {np.min(A, axis=0).tolist()}")
print(f"  max = {np.max(A, axis=0).tolist()}")
