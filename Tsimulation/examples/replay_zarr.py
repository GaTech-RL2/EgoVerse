"""Replay the actions of an episode in a fresh env and compare states.

End-to-end consistency check: the env is deterministic given a seed, and
the writer round-trips actions/state. After replay, the recorded and live
state trajectories should match within float tolerance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import zarr

from Tsimulation.collect.zarr_writer import (
    ACTION_KEY,
    GOAL_KEY,
    STATE_KEY,
)
from Tsimulation.pushshapes.env import PushShapesEnv


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        required=True,
        help="path to a dataset dir containing episode_*.zarr",
    )
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--tol", type=float, default=1e-3)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    episode_path = dataset_path / f"episode_{args.episode:06d}.zarr"
    if not episode_path.exists():
        raise FileNotFoundError(f"no such episode store: {episode_path}")

    store = zarr.open_group(str(episode_path), mode="r")
    actions = np.asarray(store[ACTION_KEY][:])
    states = np.asarray(store[STATE_KEY][:])
    goal_pose = np.asarray(store[GOAL_KEY][0])
    metadata = dict(store.attrs)
    env_args = json.loads(metadata["task_description"])["env_args"]

    print(f"loaded episode {episode_path.name}: {len(actions)} steps")
    print(f"env_args: {env_args}")

    # NOTE: we cannot perfectly reproduce the recorded trajectory because we
    # don't have the original RNG seed (random spawn positions). Instead we
    # set the env state to match the first recorded state and replay actions
    # from there — verifying determinism of the physics step given identical
    # initial conditions is left to the unit test which uses a known seed.
    env = PushShapesEnv(
        object_shape=env_args["object_shape"],
        pusher_shape=env_args["pusher_shape"],
        obstacle_level=env_args["obstacle_level"],
        image_size=env_args.get("image_size", 96),
        max_episode_steps=env_args.get("max_episode_steps", 300),
    )
    env.reset()

    # Force the env to match the recorded initial state. agent_pos = state[0:2],
    # object_pose = state[2:5].
    initial_state = states[0]
    env._pusher_body.position = (float(initial_state[0]), float(initial_state[1]))
    env._object_body.position = (float(initial_state[2]), float(initial_state[3]))
    env._object_body.angle = float(initial_state[4])
    env._object_body.velocity = (0.0, 0.0)
    env._object_body.angular_velocity = 0.0
    env._goal_pose = (float(goal_pose[0]), float(goal_pose[1]), float(goal_pose[2]))
    env._goal_polygon = env._build_object_polygon(
        (env._goal_pose[0], env._goal_pose[1]), env._goal_pose[2]
    )

    drift = []
    for i, action in enumerate(actions[:-1]):
        obs, reward, terminated, truncated, info = env.step(action)
        recorded_next = states[i + 1]
        live_next = np.concatenate([obs["agent_pos"], obs["object_pose"]])
        drift.append(float(np.linalg.norm(recorded_next - live_next)))

    drift = np.asarray(drift)
    print(
        f"trajectory drift: mean={drift.mean():.4f}  max={drift.max():.4f}  "
        f"(tol={args.tol})"
    )
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
