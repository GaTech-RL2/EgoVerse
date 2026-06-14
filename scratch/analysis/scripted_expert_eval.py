"""Standalone: does the CURRENT scripted_action controller actually SOLVE
PushShapes new_circle_3 (T object, circle pusher, obstacles=0)?

For N episodes initialized the SAME way HNetSimEval / SimRolloutEval does
(init_mode='replay': env.reset(seed=0) then env.set_state(frame-0 demo state
+ goal) read directly from the new_circle_3 zarr dataset), roll out the
current scripted_action controller for max_steps and record per-episode final
coverage (= max info['coverage'] over the episode, the same metric the eval
reports). Render one episode to mp4.

No policy / model involved — just the sim + the expert.
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import torch
import torchvision.io as tvio
import zarr

# Repo on path
sys.path.insert(0, "/coc/flash7/paphiwetsa3/projects/EgoVerse2")

from Tsimulation.pushshapes.env import PushShapesEnv
from Tsimulation.collect.scripted_collect import scripted_action  # CURRENT expert
from egomimic.eval.eval_sim import _state_to_init  # exact same split eval uses

DATA_DIR = "/coc/flash7/paphiwetsa3/datasets/new_circle_3"
_REPLAY_RESET_SEED = 0  # matches eval_sim._REPLAY_RESET_SEED


def load_ep_init(ep_path: str):
    """Read frame-0 state (5,) and goal_pose (3,) from one episode zarr,
    mirroring what the eval's replay-init reads from the packed sample:
      state_agent_obj <- observations.state   (frame 0)
      goal_pose       <- goal_pose            (frame 0)
    """
    g = zarr.open(ep_path, mode="r")
    state0 = np.asarray(g["observations.state"][0], dtype=np.float64).reshape(-1)
    goal0 = np.asarray(g["goal_pose"][0], dtype=np.float64).reshape(-1)
    return state0, goal0


def rollout(env: PushShapesEnv, state0, goal0, max_steps: int, jitter: float,
            rng, render: bool):
    """One scripted_action rollout from a replay-init state. Returns
    (final_coverage, frames_or_None). final_coverage = max coverage over ep."""
    # ---- init EXACTLY like SimRolloutEval._init_env (replay) ----
    agent_pos, object_pose = _state_to_init(state0)
    goal_pose = tuple(float(x) for x in goal0[:3])
    env.reset(seed=_REPLAY_RESET_SEED)
    env.set_state(agent_pos=agent_pos, object_pose=object_pose, goal_pose=goal_pose)
    obs = env._get_obs()

    last_coverage = float(env._coverage())
    frames = []
    for _ in range(max_steps):
        agent_xy = np.asarray(obs["agent_pos"], dtype=np.float64)
        object_xy = np.asarray(obs["object_pose"][:2], dtype=np.float64)
        goal_xy = np.asarray(obs["goal_pose"][:2], dtype=np.float64)
        action = scripted_action(
            agent_xy=agent_xy,
            object_xy=object_xy,
            goal_xy=goal_xy,
            world_size=float(env.WORLD_SIZE),
        )
        if jitter > 0.0:
            action = np.clip(action + rng.normal(0.0, jitter, size=(2,)),
                             0.0, float(env.WORLD_SIZE))
        obs, reward, terminated, truncated, info = env.step(action)
        last_coverage = max(last_coverage, float(info.get("coverage", 0.0)))
        if render:
            frames.append(env.render())  # (512,512,3) uint8
        # eval stops at coverage_threshold; here we let it run to capture the
        # true max, but a terminated episode (>=0.95) we can stop early.
        if terminated:
            if render:
                # pad a few extra frames so the success is visible in the mp4
                for _ in range(10):
                    frames.append(env.render())
            break
    return last_coverage, (frames if render else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-episodes", type=int, default=18)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--jitter", type=float, default=0.0,
                    help="action noise stddev (world coords). 0 = clean expert.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="/tmp/scripted_eval_out")
    ap.add_argument("--render-ep", type=int, default=0,
                    help="which episode index to render to mp4")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    eps = sorted(
        d for d in os.listdir(DATA_DIR)
        if d.startswith("episode_") and d.endswith(".zarr")
    )[: args.num_episodes]
    print(f"[INFO] {len(eps)} episodes from {DATA_DIR}", flush=True)

    env = PushShapesEnv(
        object_shape="T", pusher_shape="circle", obstacle_level=0,
        render_mode="rgb_array", image_size=96,
    )

    covs = []
    mp4_path = None
    for i, ep in enumerate(eps):
        ep_path = os.path.join(DATA_DIR, ep)
        state0, goal0 = load_ep_init(ep_path)
        render = (i == args.render_ep)
        cov, frames = rollout(env, state0, goal0, args.max_steps, args.jitter,
                              rng, render)
        covs.append(cov)
        print(f"  ep {i:02d} {ep:42s} final_coverage={cov:.4f}", flush=True)
        if render and frames:
            vid = torch.from_numpy(np.stack(frames)).to(torch.uint8)  # (T,H,W,3)
            mp4_path = os.path.join(args.out_dir, f"scripted_rollout_ep{i:02d}.mp4")
            tvio.write_video(mp4_path, vid, fps=30, video_codec="h264")
            print(f"  [VIDEO] ep {i} ({len(frames)} frames) -> {mp4_path}", flush=True)

    env.close()
    covs = np.asarray(covs)
    print("\n========== SCRIPTED_ACTION EXPERT RESULT ==========", flush=True)
    print(f"  episodes        : {len(covs)}", flush=True)
    print(f"  mean coverage   : {covs.mean():.4f}", flush=True)
    print(f"  median coverage : {np.median(covs):.4f}", flush=True)
    print(f"  min / max       : {covs.min():.4f} / {covs.max():.4f}", flush=True)
    print(f"  >=0.95 (success): {int((covs >= 0.95).sum())}/{len(covs)}", flush=True)
    print(f"  >=0.70 (evalthr): {int((covs >= 0.70).sum())}/{len(covs)}", flush=True)
    print(f"  per-episode     : {np.round(covs, 3).tolist()}", flush=True)
    if mp4_path:
        print(f"  sample mp4      : {mp4_path}", flush=True)


if __name__ == "__main__":
    main()
