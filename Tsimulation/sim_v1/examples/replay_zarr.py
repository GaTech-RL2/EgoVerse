"""Replay a recorded episode's actions in a fresh env and report state drift.

End-to-end consistency check: physics are deterministic given a seed and the
writer round-trips actions/state. We can't re-derive the random spawn (no
seed is saved), so we instead force the env to the first recorded state via
set_state() and replay action-by-action, comparing recorded vs live state.

Usage::

    python -m Tsimulation.examples.replay_zarr \\
        --dataset data/pushshapes_demos --episode 0
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import zarr

from Tsimulation.sim_v1.collect.zarr_writer import ACTION_KEY, GOAL_KEY, STATE_KEY
from Tsimulation.sim_v1.pushshapes.env import PushShapesEnv

# Match both new-style `episode_<obj>_<pusher>_obs<N>_<idx>.zarr` and the
# older flat `episode_NNNNNN.zarr` naming.
_NEW_EPISODE_RE = re.compile(r"^episode_[A-Za-z0-9]+_[A-Za-z0-9]+_obs\d+_(\d+)\.zarr$")
_OLD_EPISODE_RE = re.compile(r"^episode_(\d+)\.zarr$")


def _resolve_episode_path(dataset: Path, episode: int) -> Path:
    for entry in sorted(dataset.iterdir()):
        for regex in (_NEW_EPISODE_RE, _OLD_EPISODE_RE):
            m = regex.match(entry.name)
            if m and int(m.group(1)) == episode:
                return entry
    raise FileNotFoundError(f"no episode with index {episode} in {dataset}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset",
        required=True,
        help="directory containing episode_*.zarr stores",
    )
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--tol", type=float, default=1e-3)
    args = p.parse_args()

    episode_path = _resolve_episode_path(Path(args.dataset), args.episode)
    store = zarr.open_group(str(episode_path), mode="r")
    actions = np.asarray(store[ACTION_KEY][:])
    states = np.asarray(store[STATE_KEY][:])
    goal_pose = np.asarray(store[GOAL_KEY][0])
    env_args = json.loads(dict(store.attrs)["task_description"])["env_args"]

    print(f"loaded {episode_path.name}: {len(actions)} steps")
    print(f"env_args: {env_args}")

    env = PushShapesEnv(
        object_shape=env_args["object_shape"],
        pusher_shape=env_args["pusher_shape"],
        obstacle_level=env_args["obstacle_level"],
        image_size=env_args.get("image_size", 96),
    )
    env.reset()

    # Force env to the recorded initial state: agent_pos = state[0:2],
    # object_pose = state[2:5], goal_pose from the per-frame goal array.
    s0 = states[0]
    env.set_state(
        agent_pos=(float(s0[0]), float(s0[1])),
        object_pose=(float(s0[2]), float(s0[3]), float(s0[4])),
        goal_pose=(float(goal_pose[0]), float(goal_pose[1]), float(goal_pose[2])),
    )

    drift = []
    for i, action in enumerate(actions[:-1]):
        obs, _, _, _, _ = env.step(action)
        live_next = np.concatenate([obs["agent_pos"], obs["object_pose"]])
        drift.append(float(np.linalg.norm(states[i + 1] - live_next)))

    drift = np.asarray(drift)
    print(
        f"trajectory drift: mean={drift.mean():.4f}  max={drift.max():.4f}  "
        f"(tol={args.tol})"
    )
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
