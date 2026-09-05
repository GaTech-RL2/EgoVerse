"""Collect demonstrations with the SE(2) pose controller.

Two phases per attempt, because rendering costs ~3.3x:

  search  run the controller with obs rendering OFF and keep only the actions
  record  on success, reset to the SAME seed and replay those actions with
          rendering ON, writing the episode through ZarrDemoWriter

so image observations are paid for only on episodes that are actually kept.
Failed attempts are discarded, which is what makes a low success rate usable:
what matters for collection is successes per cpu-hour, not success RATE.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from Tsimulation.pushshapes import PushShapesEnv
from Tsimulation.sim_v2.collect.pose_controller import PosePushController
from Tsimulation.sim_v2.collect.zarr_writer import ZarrDemoWriter


def _pusher_pose(obs) -> np.ndarray:
    return np.concatenate([np.asarray(obs["agent_pos"], dtype=np.float64),
                           np.asarray(obs["agent_angle"], dtype=np.float64)])


def search(emb, seed, gap, max_steps, obstacle_level=0):
    """Headless attempt. Returns the action list if it succeeded, else None."""
    env = PushShapesEnv(object_shape="T", pusher_shape=emb,
                        obstacle_level=obstacle_level)
    env.reset(seed=seed)
    env._skip_obs_render = True
    if gap and gap != "ideal":
        from Tsimulation.sim_v2.pushshapes.agents import CONTROL_GAPS
        env.agent.control_gap = CONTROL_GAPS[gap]
    ctl = PosePushController(env.WORLD_SIZE, rng=np.random.default_rng(seed))
    aw = env.action_space.shape[0]
    acts = []
    for _ in range(max_steps):
        o = env._get_obs()
        xy = ctl(o["agent_pos"], o["object_pose"], o["goal_pose"])
        a = np.zeros(aw, dtype=np.float64)
        a[:2] = xy
        if aw >= 3:
            a[2] = float(o["agent_angle"][0])
        _o, _r, term, _tr, _i = env.step(a)
        acts.append(a.copy())
        if term:
            return acts
    return None


def record(writer, emb, seed, gap, acts, obstacle_level=0):
    """Replay a successful action list with rendering on and write it."""
    env = PushShapesEnv(object_shape="T", pusher_shape=emb,
                        obstacle_level=obstacle_level, image_size=writer.image_size)
    env.reset(seed=seed)
    if gap and gap != "ideal":
        from Tsimulation.sim_v2.pushshapes.agents import CONTROL_GAPS
        env.agent.control_gap = CONTROL_GAPS[gap]
    writer.start_episode(init_state=env.get_episode_init())
    obs = env._get_obs()
    final = 0.0
    for a in acts:
        nxt, reward, term, _tr, info = env.step(np.asarray(a, dtype=np.float64))
        writer.add_step(image=obs["image"], pusher_obs_pose=_pusher_pose(obs),
                        object_obs_pose=obs["object_pose"], pusher_cmd_pose=a[:3]
                        if len(a) >= 3 else np.concatenate([a[:2], [0.0]]),
                        action=a, reward=float(reward),
                        goal_pose=obs["goal_pose"])
        obs = nxt
        final = float(info.get("coverage", 0.0))
        if term:
            break
    if final < 0.95:           # replay must reproduce the search
        writer.abort_episode()
        return False
    writer.commit_episode()
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--embodiment", default="L")
    ap.add_argument("--gap", default="ideal")
    ap.add_argument("--target", type=int, default=100)
    ap.add_argument("--max-attempts", type=int, default=100000)
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--image-size", type=int, default=96)
    a = ap.parse_args()

    writer = ZarrDemoWriter(
        Path(a.out), env_args={"object_shape": "T", "pusher_shape": a.embodiment,
                               "obstacle_level": 0},
        image_size=a.image_size, embodiment=f"pushshapes_sim_{a.embodiment}")
    t0 = time.time()
    kept = attempts = drift = 0
    while kept < a.target and attempts < a.max_attempts:
        seed = a.seed0 + attempts
        attempts += 1
        acts = search(a.embodiment, seed, a.gap, a.max_steps)
        if acts is None:
            continue
        if record(writer, a.embodiment, seed, a.gap, acts):
            kept += 1
        else:
            drift += 1
        if kept and kept % 10 == 0:
            el = time.time() - t0
            print(f"  kept={kept:>5} attempts={attempts:>7} "
                  f"rate={kept/attempts:.4f} drift={drift} "
                  f"{el:.0f}s {el/kept:.1f}s/ep", flush=True)
    writer.close()
    el = time.time() - t0
    print(f"DONE kept={kept} attempts={attempts} rate={kept/max(attempts,1):.4f} "
          f"replay_drift={drift} {el:.0f}s", flush=True)


if __name__ == "__main__":
    main()
