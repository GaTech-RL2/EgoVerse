"""Replay a recorded episode's actions in a fresh env and verify determinism.

Usage::

    python -m Tsimulation.examples.replay_zarr \
        --dataset data/pushshapes_demos --episode 0

    python -m Tsimulation.examples.replay_zarr \
        --dataset data/pushshapes_demos --all
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import zarr

from Tsimulation.collect.zarr_writer import ACTION_KEY, GOAL_KEY, STATE_KEY
from Tsimulation.pushshapes.env import PushShapesEnv

_NEW_EPISODE_RE = re.compile(r"^episode_[A-Za-z0-9]+_[A-Za-z0-9]+_obs\d+_(\d+)\.zarr$")
_OLD_EPISODE_RE = re.compile(r"^episode_(\d+)\.zarr$")


def _resolve_episode_path(dataset: Path, episode: int) -> Path:
    for entry in sorted(dataset.iterdir()):
        for regex in (_NEW_EPISODE_RE, _OLD_EPISODE_RE):
            m = regex.match(entry.name)
            if m and int(m.group(1)) == episode:
                return entry
    raise FileNotFoundError(f"no episode with index {episode} in {dataset}")


def _all_episode_paths(dataset: Path) -> list[Path]:
    eps = []
    for entry in sorted(dataset.iterdir()):
        if not entry.is_dir():
            continue
        for regex in (_NEW_EPISODE_RE, _OLD_EPISODE_RE):
            if regex.match(entry.name):
                eps.append(entry)
                break
    return eps


def replay_one(episode_path: Path, tol: float) -> dict:
    store = zarr.open_group(str(episode_path), mode="r")
    attrs = dict(store.attrs)
    total_frames = attrs.get("total_frames", None)
    actions = np.asarray(store[ACTION_KEY][:])
    states = np.asarray(store[STATE_KEY][:])
    goal_pose = np.asarray(store[GOAL_KEY][0])
    reward = np.asarray(store["reward"][:]).squeeze()
    env_args = json.loads(attrs["task_description"])["env_args"]

    if total_frames is not None and total_frames < len(actions):
        actions = actions[:total_frames]
        states = states[:total_frames]
        reward = reward[:total_frames]

    ep_init = json.loads(attrs["episode_init"]) if "episode_init" in attrs else None

    env = PushShapesEnv(
        object_shape=env_args["object_shape"],
        pusher_shape=env_args["pusher_shape"],
        obstacle_level=env_args.get("obstacle_level", 0),
        image_size=env_args.get("image_size", 96),
    )
    reset_seed = ep_init.get("reset_seed") if ep_init else None
    env.reset(seed=reset_seed)

    if ep_init is not None:
        ap = tuple(ep_init["agent_pos"])
        op = tuple(ep_init["object_pose"])
        gp = tuple(ep_init["goal_pose"])
    else:
        s0 = states[0]
        ap = (float(s0[0]), float(s0[1]))
        op = (float(s0[2]), float(s0[3]), float(s0[4]))
        gp = (float(goal_pose[0]), float(goal_pose[1]), float(goal_pose[2]))
    env.set_state(agent_pos=ap, object_pose=op, goal_pose=gp)

    drift = []
    max_cov = 0.0
    for i in range(len(actions)):
        obs, _, term, _, info = env.step(actions[i])
        max_cov = max(max_cov, info["coverage"])
        if i + 1 < len(states):
            live = np.concatenate([obs["agent_pos"], obs["object_pose"]])
            drift.append(float(np.linalg.norm(states[i + 1] - live)))
        if term:
            break

    env.close()
    drift = np.asarray(drift) if drift else np.zeros(1)
    stored_max = float(reward.max())

    return {
        "name": episode_path.name,
        "T": len(actions),
        "stored_cov": stored_max,
        "replay_cov": max_cov,
        "drift_mean": float(drift.mean()),
        "drift_max": float(drift.max()),
        "ok": max_cov >= stored_max - tol,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, help="directory containing episode_*.zarr")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--all", action="store_true", help="replay every episode in the dataset")
    p.add_argument("--tol", type=float, default=0.05)
    args = p.parse_args()

    dataset = Path(args.dataset)

    if args.all:
        episodes = _all_episode_paths(dataset)
        print(f"Replaying {len(episodes)} episodes from {dataset}\n")
        results = []
        for ep in episodes:
            r = replay_one(ep, args.tol)
            results.append(r)
            status = "OK" if r["ok"] else "FAIL"
            print(f"  {r['name']}: T={r['T']:4d} stored={r['stored_cov']:.3f} "
                  f"replay={r['replay_cov']:.3f} drift_max={r['drift_max']:.4f} {status}")
        n_ok = sum(1 for r in results if r["ok"])
        print(f"\n{n_ok}/{len(results)} episodes replayed within tolerance ({args.tol})")
        return 0 if n_ok == len(results) else 1
    else:
        ep_path = _resolve_episode_path(dataset, args.episode)
        r = replay_one(ep_path, args.tol)
        status = "OK" if r["ok"] else "FAIL"
        print(f"{r['name']}: T={r['T']} stored={r['stored_cov']:.3f} "
              f"replay={r['replay_cov']:.3f} drift_mean={r['drift_mean']:.4f} "
              f"drift_max={r['drift_max']:.4f} {status}")
        return 0 if r["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
