"""Headless scripted demonstration collection for PushShapesEnv.

Uses a simple two-phase heuristic so we can bootstrap a demo set without a
mouse — handy while the physics is being tuned and human mouse demos are
finicky to record.

Heuristic per step:

1. Compute the unit vector from the object's center toward the goal center.
2. Pick an "approach point" a fixed offset behind the object along the
   anti-push direction.
3. If the pusher is far from that approach point, the action target is the
   approach point (move into contact).
4. Otherwise the action target is past the goal along the push direction
   (push the object toward the goal).

Episodes that don't reach a configurable coverage threshold are discarded
so we don't pollute the dataset with failed attempts. Obstacle-heavy
levels will see a lower success rate.

Usage::

    python -m Tsimulation.collect.scripted_collect \\
        --output data/pushshapes_scripted \\
        --object T --pusher circle --obstacles 0 \\
        --num-episodes 50
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from Tsimulation.collect.zarr_writer import ZarrDemoWriter
from Tsimulation.pushshapes.env import PushShapesEnv

APPROACH_OFFSET = 60.0
PUSH_LOOKAHEAD = 80.0
CONTACT_RADIUS = 25.0


def scripted_action(
    *,
    agent_xy: np.ndarray,
    object_xy: np.ndarray,
    goal_xy: np.ndarray,
    world_size: float,
) -> np.ndarray:
    """Return a (2,) float32 action target in world coords."""
    push_vec = goal_xy - object_xy
    push_dist = float(np.linalg.norm(push_vec))
    if push_dist < 1e-3:
        return object_xy.astype(np.float32)
    push_dir = push_vec / push_dist
    approach_point = object_xy - push_dir * APPROACH_OFFSET
    if float(np.linalg.norm(agent_xy - approach_point)) > CONTACT_RADIUS:
        target = approach_point
    else:
        target = object_xy + push_dir * PUSH_LOOKAHEAD
    return np.clip(target, 0.0, world_size).astype(np.float32)


def run_episode(
    env: PushShapesEnv,
    writer: ZarrDemoWriter,
    *,
    max_steps: int,
    success_threshold: float,
    rng: np.random.Generator,
    jitter: float,
    seed: int | None,
) -> tuple[bool, float, int]:
    """Roll out one scripted episode. Returns (committed, final_coverage, steps)."""
    obs, info = env.reset(seed=seed)
    writer.start_episode()
    coverage = float(info.get("coverage", 0.0))
    final_coverage = coverage
    steps = 0
    for _ in range(max_steps):
        agent_xy = np.asarray(obs["agent_pos"], dtype=np.float32)
        object_xy = np.asarray(obs["object_pose"][:2], dtype=np.float32)
        goal_xy = np.asarray(obs["goal_pose"][:2], dtype=np.float32)
        action = scripted_action(
            agent_xy=agent_xy,
            object_xy=object_xy,
            goal_xy=goal_xy,
            world_size=float(env.WORLD_SIZE),
        )
        if jitter > 0.0:
            action = action + rng.normal(0.0, jitter, size=(2,)).astype(np.float32)
            action = np.clip(action, 0.0, float(env.WORLD_SIZE)).astype(np.float32)

        next_obs, reward, terminated, truncated, info = env.step(action)
        coverage = float(info.get("coverage", 0.0))

        writer.add_step(
            image=obs["image"],
            pusher_obs_pose=obs["agent_pos"],
            object_obs_pose=obs["object_pose"],
            pusher_cmd_pose=action,
            action=action,
            reward=float(reward),
            goal_pose=obs["goal_pose"],
        )

        obs = next_obs
        steps += 1
        final_coverage = coverage
        if terminated or truncated:
            break

    committed = False
    if final_coverage >= success_threshold and writer.steps_in_episode > 0:
        idx = writer.commit_episode()
        committed = idx >= 0
    else:
        writer.abort_episode()
    return committed, final_coverage, steps


def run(args: argparse.Namespace) -> int:
    if args.headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    env = PushShapesEnv(
        object_shape=args.object,
        pusher_shape=args.pusher,
        obstacle_level=args.obstacles,
        image_size=args.image_size,
        max_episode_steps=args.max_steps,
        seed=args.seed,
    )
    env_args = {
        "object_shape": args.object,
        "pusher_shape": args.pusher,
        "obstacle_level": args.obstacles,
        "image_size": args.image_size,
        "fps": args.fps,
        "max_episode_steps": args.max_steps,
        "collector": "scripted",
    }
    writer = ZarrDemoWriter(
        path=args.output,
        env_args=env_args,
        image_size=args.image_size,
        fps=args.fps,
    )
    rng = np.random.default_rng(args.seed)

    saved = 0
    attempts = 0
    max_attempts = args.num_episodes * args.max_tries
    while saved < args.num_episodes and attempts < max_attempts:
        attempts += 1
        ep_seed = (args.seed + attempts) if args.seed is not None else None
        committed, coverage, steps = run_episode(
            env,
            writer,
            max_steps=args.max_steps,
            success_threshold=args.success_threshold,
            rng=rng,
            jitter=args.jitter,
            seed=ep_seed,
        )
        if committed:
            saved += 1
            print(
                f"saved ep {saved:03d}/{args.num_episodes}  "
                f"steps={steps:3d}  coverage={coverage:.3f}  (attempt {attempts})"
            )
        elif args.verbose:
            print(
                f"discard       attempt {attempts:3d}  "
                f"steps={steps:3d}  coverage={coverage:.3f}"
            )

    writer.close()
    env.close()
    print(
        f"done. saved {saved}/{args.num_episodes} in {attempts} attempts "
        f"-> {args.output}"
    )
    return 0 if saved == args.num_episodes else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", required=True, help="directory for episode_*.zarr stores"
    )
    parser.add_argument("--object", default="T", choices=["T", "U", "Z"])
    parser.add_argument("--pusher", default="circle", choices=["circle", "stick"])
    parser.add_argument("--obstacles", type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument("--num-episodes", type=int, default=20)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=0.6,
        help="minimum final coverage to count an episode as a success",
    )
    parser.add_argument(
        "--max-tries",
        type=int,
        default=4,
        help="give up after num_episodes * max_tries attempts",
    )
    parser.add_argument(
        "--jitter",
        type=float,
        default=4.0,
        help="zero-mean Gaussian noise (stddev, in world coords) added to actions",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=True,
        help="force SDL dummy driver so no display is required (default: True)",
    )
    parser.add_argument(
        "--no-headless",
        dest="headless",
        action="store_false",
        help="render to a real display (requires a windowed environment)",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
