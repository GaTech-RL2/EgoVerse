"""Rebuild a dataset cell-by-cell: dedupe, optionally regenerate, rewrite.

Usage:
    python -m Tsimulation.sim_v2.generate.rebuild_dataset \
        --src   /local/path/to/<version>            \
        --out   /local/path/to/<new-version>        \
        --min-distance 60 --attempts 1500

DESIGN NOTE, measured rather than assumed. On
control_gap_all6x13_1000_per_cell_simv2_20260826, cell ideal/gripper/T:

    original 1000 episodes   medNN  10.8   dup<20 69%   uniq@60 326
    deduped   326 episodes   medNN  82.7   dup<20  0%   uniq@60 326
    regenerated from those   2600 attempts -> 250 kept, uniq@60 250

Deduping is nearly free and loses NOTHING -- both sets contain the same 326
distinct behaviours, so ~67% of the cell is pure repetition. Regeneration is
expensive (9.6% acceptance) and on its own yields fewer distinct behaviours
than the dedupe it started from. So dedupe is the default and regeneration is
opt-in via --attempts, to be justified by whether it ADDS to the deduped set
rather than by its own count.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from Tsimulation.sim_v2.collect.zarr_writer import ZarrDemoWriter
from Tsimulation.sim_v2.generate.diversity import trajectory_signature
from Tsimulation.sim_v2.generate.mimicgen import generate as mg_generate
from Tsimulation.sim_v2.generate.reseed import (
    distinct_subset,
    load_cell,
    to_sources,
)
from Tsimulation.sim_v2.pushshapes.env import PushShapesEnv


def find_cells(root: str) -> list[Path]:
    """Every directory holding episode zarrs, at any depth."""
    root = Path(root)
    out = set()
    for p in root.rglob("episode_*.zarr"):
        out.add(p.parent)
    return sorted(out)


def replay_and_write(writer, agent, records, image_size):
    """Re-run each kept episode with rendering ON to capture frames."""
    kept = 0
    for r in records:
        ini = r["init"]
        env = PushShapesEnv(object_shape=ini["object_shape"],
                            pusher_shape=ini["pusher_shape"],
                            obstacle_level=int(ini.get("obstacle_level", 0)),
                            image_size=image_size)
        env.reset(seed=0)
        env.set_state(object_pose=tuple(ini["object_pose"]),
                      goal_pose=tuple(ini["goal_pose"]),
                      agent_pos=tuple(ini["agent_pos"]),
                      agent_angle=float(ini.get("agent_angle", 0.0)))
        writer.start_episode(init_state=env.get_episode_init())
        ok = False
        for a in np.asarray(r["actions"], dtype=np.float64):
            obs, rew, term, _t, _i = env.step(a)
            px, py = env.agent_pos
            ox, oy, oth = env.object_pose
            writer.add_step(image=obs["image"],
                            pusher_obs_pose=np.array([px, py, env.pusher_angle]),
                            object_obs_pose=np.array([ox, oy, oth]),
                            pusher_cmd_pose=np.array([a[0], a[1],
                                                      a[2] if len(a) > 2 else 0.0]),
                            action=a, reward=rew,
                            goal_pose=np.array(env.goal_pose))
            if term:
                ok = True
                break
        # Only keep episodes that still SUCCEED under current physics. The
        # gripper fix in PR586 invalidated grasps that latched without both
        # jaws touching, so a straight copy would carry those forward.
        if ok and writer.steps_in_episode > 0:
            writer.commit_episode()
            kept += 1
        else:
            writer.abort_episode()
    return kept


def process_cell(cell: Path, src_root: Path, out_root: Path, *,
                 min_distance: float, attempts: int, image_size: int,
                 novelty: float) -> dict:
    rel = cell.relative_to(src_root)
    recs = load_cell(str(cell))
    if not recs:
        return {"cell": str(rel), "read": 0, "kept": 0}
    dis = distinct_subset(recs, min_distance)

    extra = []
    if attempts > 0:
        srcs = to_sources(dis)
        res = mg_generate(srcs, attempts, seed=17, perturb=True,
                          min_novelty=novelty)
        # Keep a generated demo only if it is genuinely new relative to the
        # deduped seeds -- generation that merely rediscovers its own sources
        # is the redundancy we are removing, reintroduced.
        Sd = np.asarray([trajectory_signature(r["object_xy"]) for r in dis])
        for d in res.demos:
            sig = trajectory_signature(np.asarray(d.actions)[:, :2])
            if np.linalg.norm(Sd - sig[None, :], axis=1).min() >= min_distance:
                extra.append({"actions": np.asarray(d.actions),
                              "object_xy": np.zeros((2, 2)),
                              "init": {"object_pose": list(d.object_pose),
                                       "goal_pose": list(d.goal_pose),
                                       "agent_pos": list(d.agent_pos),
                                       "agent_angle": 0.0,
                                       "object_shape": d.object_shape,
                                       "pusher_shape": d.agent,
                                       "obstacle_level": d.obstacle_level}})

    agent = recs[0]["init"]["pusher_shape"]
    dest = out_root / rel
    dest.mkdir(parents=True, exist_ok=True)
    w = ZarrDemoWriter(path=dest,
                       env_args={"object_shape": recs[0]["init"]["object_shape"],
                                 "pusher_shape": agent,
                                 "obstacle_level": int(recs[0]["init"].get("obstacle_level", 0))},
                       image_size=image_size)
    kept = replay_and_write(w, agent, dis + extra, image_size)
    w.close()
    return {"cell": str(rel), "read": len(recs), "distinct": len(dis),
            "generated_new": len(extra), "kept": kept}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-distance", type=float, default=60.0)
    ap.add_argument("--attempts", type=int, default=0,
                    help="MimicGen attempts per cell; 0 = dedupe only")
    ap.add_argument("--novelty", type=float, default=40.0)
    ap.add_argument("--image-size", type=int, default=96)
    ap.add_argument("--limit-cells", type=int, default=0)
    a = ap.parse_args()

    src, out = Path(a.src), Path(a.out)
    cells = find_cells(src)
    if a.limit_cells:
        cells = cells[:a.limit_cells]
    print(f"{len(cells)} cells under {src}", flush=True)
    tot_read = tot_kept = tot_new = 0
    for i, c in enumerate(cells, 1):
        t0 = time.time()
        r = process_cell(c, src, out, min_distance=a.min_distance,
                         attempts=a.attempts, image_size=a.image_size,
                         novelty=a.novelty)
        tot_read += r["read"]; tot_kept += r["kept"]; tot_new += r.get("generated_new", 0)
        print(f"[{i}/{len(cells)}] {r['cell']:<34} read={r['read']:>4} "
              f"distinct={r.get('distinct',0):>4} new={r.get('generated_new',0):>4} "
              f"written={r['kept']:>4}  {time.time()-t0:5.0f}s", flush=True)
    print(f"\nTOTAL read={tot_read} written={tot_kept} generated_new={tot_new}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
