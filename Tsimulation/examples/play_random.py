"""Smoke check: run N random steps in PushShapesEnv and print final coverage.

Useful for confirming the env starts cleanly after a dependency bump or
config change. No window opens unless --render-mode is set.

Usage::

    python -m Tsimulation.examples.play_random --object T --steps 100
"""

from __future__ import annotations

import argparse

import numpy as np

from Tsimulation.pushshapes.env import PushShapesEnv


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--object", default="T", choices=["T", "U", "Z"])
    p.add_argument(
        "--pusher",
        default="circle",
        choices=["circle", "circle_small", "stick"],
    )
    p.add_argument("--obstacles", type=int, default=0, choices=[0, 1, 2, 3])
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    env = PushShapesEnv(
        object_shape=args.object,
        pusher_shape=args.pusher,
        obstacle_level=args.obstacles,
        seed=args.seed,
    )
    obs, info = env.reset(seed=args.seed)
    rng = np.random.default_rng(args.seed)

    coverage = info["coverage"]
    steps_taken = 0
    for i in range(args.steps):
        steps_taken = i + 1
        action = rng.uniform(0.0, env.WORLD_SIZE, size=(2,)).astype(np.float32)
        _, _, terminated, truncated, info = env.step(action)
        coverage = info["coverage"]
        if terminated or truncated:
            break

    print(
        f"object={args.object} pusher={args.pusher} obstacles={args.obstacles} "
        f"steps={steps_taken} final_coverage={coverage:.3f}"
    )
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
