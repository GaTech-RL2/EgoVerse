"""Standalone: do the new_circle_3 DEMONSTRATIONS themselves solve PushShapes
(T object, circle pusher, obstacles=0) when their RECORDED actions are replayed
in the sim?

Same init as scripted_expert_eval.py / SimRolloutEval (init_mode='replay':
env.reset(seed=0) then env.set_state(frame-0 demo state + goal) read straight
from the new_circle_3 zarr). But instead of querying a scripted controller each
step, we STEP THE ENV WITH THE DEMO'S OWN RECORDED ACTIONS
(== observations.pusher_cmd_pose == `actions`, absolute world-coord cursor
targets). No policy / model involved.

Per-episode metric = max info['coverage'] over the episode (same as the eval).

Demos have a zero-padded action tail (trailing (0,0) actions after the object
stops moving). We truncate replay at the last nonzero action so we don't yank
the cursor to the world corner. We report the truncation per episode.
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

sys.path.insert(0, "/coc/flash7/paphiwetsa3/projects/EgoVerse2")

from Tsimulation.pushshapes.env import PushShapesEnv
from egomimic.eval.eval_sim import _state_to_init, _REPLAY_RESET_SEED

DATA_DIR = "/coc/flash7/paphiwetsa3/datasets/new_circle_3"


def load_ep(ep_path: str):
    """Read frame-0 state (5,), goal_pose (3,), and the full recorded action
    sequence (T,2) from one episode zarr. Confirm actions == pusher_cmd_pose."""
    g = zarr.open(ep_path, mode="r")
    state0 = np.asarray(g["observations.state"][0], dtype=np.float64).reshape(-1)
    goal0 = np.asarray(g["goal_pose"][0], dtype=np.float64).reshape(-1)
    actions = np.asarray(g["actions"][:], dtype=np.float64)
    pcp = np.asarray(g["observations.pusher_cmd_pose"][:], dtype=np.float64)
    gap = float(np.abs(actions - pcp).max())
    return state0, goal0, actions, gap


def replay(env, state0, goal0, actions, max_steps, render):
    """Replay recorded demo actions from a replay-init state. Returns
    (final_coverage, n_steps_replayed, frames_or_None). final = max coverage."""
    agent_pos, object_pose = _state_to_init(state0)
    goal_pose = tuple(float(x) for x in goal0[:3])
    env.reset(seed=_REPLAY_RESET_SEED)
    env.set_state(agent_pos=agent_pos, object_pose=object_pose, goal_pose=goal_pose)

    # Truncate the zero-padded action tail: replay up to (and including) the
    # last nonzero action. Trailing (0,0) actions would drive the cursor to the
    # world corner and are padding, not real expert commands.
    nz = np.where(np.abs(actions).sum(axis=1) > 0)[0]
    last = int(nz.max()) + 1 if len(nz) else 0
    seq = actions[:last]
    seq = seq[:max_steps]

    last_coverage = float(env._coverage())
    frames = []
    if render:
        frames.append(env.render())
    n = 0
    for t in range(seq.shape[0]):
        action = seq[t]  # raw absolute world coords, NOT normalized
        obs, reward, terminated, truncated, info = env.step(action)
        last_coverage = max(last_coverage, float(info.get("coverage", 0.0)))
        n += 1
        if render:
            frames.append(env.render())
        if terminated:
            if render:
                for _ in range(10):
                    frames.append(env.render())
            break
    return last_coverage, n, (frames if render else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-episodes", type=int, default=18)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--out-dir", default="/tmp/replay_demo_out")
    ap.add_argument("--render-ep", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

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
    max_gap = 0.0
    for i, ep in enumerate(eps):
        ep_path = os.path.join(DATA_DIR, ep)
        state0, goal0, actions, gap = load_ep(ep_path)
        max_gap = max(max_gap, gap)
        render = (i == args.render_ep)
        cov, nsteps, frames = replay(env, state0, goal0, actions, args.max_steps, render)
        covs.append(cov)
        print(f"  ep {i:02d} {ep:30s} stored_len={actions.shape[0]:4d} "
              f"replayed={nsteps:4d} final_coverage={cov:.4f}", flush=True)
        if render and frames:
            vid = torch.from_numpy(np.stack(frames)).to(torch.uint8)
            mp4_path = os.path.join(args.out_dir, f"replay_demo_ep{i:02d}.mp4")
            tvio.write_video(mp4_path, vid, fps=30, video_codec="h264")
            print(f"  [VIDEO] ep {i} ({len(frames)} frames) -> {mp4_path}", flush=True)

    env.close()
    covs = np.asarray(covs)
    print("\n========== REPLAYED-DEMO RESULT (new_circle_3) ==========", flush=True)
    print(f"  actions vs pusher_cmd_pose max abs gap : {max_gap:.6f}", flush=True)
    print(f"  episodes        : {len(covs)}", flush=True)
    print(f"  mean coverage   : {covs.mean():.4f}", flush=True)
    print(f"  median coverage : {np.median(covs):.4f}", flush=True)
    print(f"  min / max       : {covs.min():.4f} / {covs.max():.4f}", flush=True)
    print(f"  >=0.95 (success): {int((covs >= 0.95).sum())}/{len(covs)}", flush=True)
    print(f"  >=0.70 (evalthr): {int((covs >= 0.70).sum())}/{len(covs)}", flush=True)
    print(f"  per-episode     : {np.round(covs, 4).tolist()}", flush=True)
    if mp4_path:
        print(f"  sample mp4      : {mp4_path}", flush=True)


if __name__ == "__main__":
    main()
