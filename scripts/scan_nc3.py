#!/usr/bin/env python
"""Scan all new_circle_3 episodes: extract T-start (obj) pose at frame 0,
goal pose at frame 0, and obstacle_level from attrs. Dump to JSON.

State layout CONFIRMED from egomimic/eval/eval_sim.py:42-46:
    state[0:2] = agent (pusher) pos
    state[2:5] = object (T) pose (x, y, theta)
goal_pose = (x, y, theta) -> goal (x,y) = goal_pose[0:2].
"""
import json
import sys
from pathlib import Path

import numpy as np
import zarr

DATA = Path("/coc/flash7/paphiwetsa3/datasets/new_circle_3")
OUT = Path("/tmp/nc3_scan.json")


def parse_obstacle_level(attrs) -> int:
    """obstacle_level lives inside the task_description JSON string."""
    td = attrs.get("task_description", None)
    if td is None:
        return None
    try:
        d = json.loads(td)
        return int(d["env_args"]["obstacle_level"])
    except Exception:
        return None


def main():
    eps = sorted(DATA.glob("episode_*.zarr"))
    print(f"Found {len(eps)} episodes", flush=True)
    records = []
    bad = []
    obstacle_levels = {}
    filename_obs_tokens = {}
    for i, ep in enumerate(eps):
        name = ep.name
        try:
            g = zarr.open_group(str(ep), mode="r")
            attrs = dict(g.attrs)
            state = g["observations.state"]
            goal = g["goal_pose"]
            # frame 0
            s0 = np.asarray(state[0])  # (5,)
            g0 = np.asarray(goal[0])   # (3,)
            tstart_x, tstart_y = float(s0[2]), float(s0[3])
            tstart_theta = float(s0[4])
            goal_x, goal_y = float(g0[0]), float(g0[1])
            obs_level = parse_obstacle_level(attrs)
            obstacle_levels[obs_level] = obstacle_levels.get(obs_level, 0) + 1
            # filename token: episode_T_circle_obs0_000000.zarr -> obs0
            tok = name.split("_")[3] if len(name.split("_")) > 3 else "?"
            filename_obs_tokens[tok] = filename_obs_tokens.get(tok, 0) + 1
            total_frames = attrs.get("total_frames", None)
            records.append({
                "name": name,
                "tstart_x": tstart_x,
                "tstart_y": tstart_y,
                "tstart_theta": tstart_theta,
                "goal_x": goal_x,
                "goal_y": goal_y,
                "goal_theta": float(g0[2]),
                "obstacle_level": obs_level,
                "fname_obs_token": tok,
                "total_frames": total_frames,
            })
        except Exception as e:
            bad.append((name, str(e)))
        if (i + 1) % 100 == 0:
            print(f"  scanned {i+1}/{len(eps)}", flush=True)

    print(f"\nobstacle_level (from attrs.task_description) counts: {obstacle_levels}", flush=True)
    print(f"filename obs token counts: {filename_obs_tokens}", flush=True)
    print(f"bad episodes: {len(bad)}", flush=True)
    for n, e in bad[:10]:
        print(f"  BAD {n}: {e}", flush=True)

    OUT.write_text(json.dumps({
        "n_episodes": len(records),
        "obstacle_level_counts": {str(k): v for k, v in obstacle_levels.items()},
        "filename_obs_token_counts": filename_obs_tokens,
        "n_bad": len(bad),
        "records": records,
    }, indent=2))
    print(f"\nWrote {OUT} with {len(records)} records", flush=True)


if __name__ == "__main__":
    main()
