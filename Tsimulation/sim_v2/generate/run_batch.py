"""Generate demos at scale: plan seeds, expand with MimicGen, write zarr."""
from __future__ import annotations
import argparse, sys, time
import numpy as np
from pathlib import Path
from Tsimulation.sim_v2.generate.planner import generate as plan_gen
from Tsimulation.sim_v2.generate.mimicgen import (
    SourceDemo,
    apply_source_control_gap,
    generate as mg_gen,
)
from Tsimulation.sim_v2.pushshapes.env import PushShapesEnv
from Tsimulation.sim_v2.collect.zarr_writer import ZarrDemoWriter


def seeds_for(agent, n_seed, max_steps):
    plans = plan_gen(agent, n_seed, max_steps=max_steps)
    return [SourceDemo(agent=agent, actions=np.array(p.actions),
                       object_pose=tuple(p.init["object_pose"]),
                       goal_pose=tuple(p.init["goal_pose"]),
                       agent_pos=tuple(p.init["agent_pos"]),
                       agent_angle=float(p.init.get("agent_angle", 0.0)))
            for p in plans]


def write(demos, out_root, agent, image_size=96):
    """Replay each generated demo ONCE with rendering on, to capture frames.

    Generation runs headless for speed, so the images have to be produced in a
    second pass -- the alternative is paying 3.3x the step cost on every
    attempt including the ~80% that fail.
    """
    if not demos:
        return 0
    d = Path(out_root) / agent / demos[0].object_shape
    d.mkdir(parents=True, exist_ok=True)
    w = ZarrDemoWriter(path=d, env_args={"object_shape": demos[0].object_shape,
                                         "pusher_shape": agent,
                                         "obstacle_level": 0},
                       image_size=image_size)
    kept = 0
    for dm in demos:
        env = PushShapesEnv(object_shape=dm.object_shape, pusher_shape=agent,
                            obstacle_level=0, image_size=image_size)
        env.reset(seed=0)
        apply_source_control_gap(env, dm)
        env.set_state(object_pose=dm.object_pose, goal_pose=dm.goal_pose,
                      agent_pos=(float(dm.agent_pos[0]), float(dm.agent_pos[1])),
                      agent_angle=float(dm.agent_angle))
        init_state = env.get_episode_init()
        if dm.control_gap_mode is not None:
            init_state["control_gap_mode"] = dm.control_gap_mode
        w.start_episode(init_state=init_state)
        ok = False
        for a in dm.actions:
            obs, r, term, _t, info = env.step(np.asarray(a, dtype=np.float64))
            px, py = env.agent_pos
            ox, oy, oth = env.object_pose
            w.add_step(image=obs["image"],
                       pusher_obs_pose=np.array([px, py, env.pusher_angle]),
                       object_obs_pose=np.array([ox, oy, oth]),
                       pusher_cmd_pose=np.array([a[0], a[1],
                                                 a[2] if len(a) > 2 else 0.0]),
                       action=np.asarray(a), reward=r,
                       goal_pose=np.array(env.goal_pose))
            if term:
                ok = True
                break
        if ok and w.steps_in_episode > 0:
            w.commit_episode(); kept += 1
        else:
            w.abort_episode()
    w.close()
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", default="umi,gripper,suction")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--attempts", type=int, default=400)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-steps", type=int, default=900)
    a = ap.parse_args()
    total = 0
    for agent in [x.strip() for x in a.agents.split(",") if x.strip()]:
        t0 = time.time()
        srcs = seeds_for(agent, a.seeds, a.max_steps)
        if not srcs:
            print(f"{agent:<9} no seed demos -- planner could not solve it", flush=True)
            continue
        res = mg_gen(srcs, a.attempts, seed=7)
        kept = write(res.demos, a.out, agent) if a.out else len(res.demos)
        total += kept
        print(f"{agent:<9} seeds {len(srcs):>2}  generated {len(res.demos):>4}"
              f"/{res.attempts} ({res.rate:5.1%})  written {kept:>4}"
              f"  {time.time()-t0:6.0f}s", flush=True)
    print(f"TOTAL {total}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
