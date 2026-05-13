"""Smoke check: run 100 random steps and print final coverage."""

from __future__ import annotations

import argparse

import numpy as np

from Tsimulation.pushshapes.env import PushShapesEnv


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--object", default="T", choices=["T", "U", "Z"])
    parser.add_argument("--pusher", default="circle", choices=["circle", "stick"])
    parser.add_argument("--obstacles", type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = PushShapesEnv(
        object_shape=args.object,
        pusher_shape=args.pusher,
        obstacle_level=args.obstacles,
        seed=args.seed,
    )
    obs, info = env.reset(seed=args.seed)
    rng = np.random.default_rng(args.seed)

    coverage = info["coverage"]
    for i in range(args.steps):
        action = rng.uniform(0.0, env.WORLD_SIZE, size=(2,)).astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        coverage = info["coverage"]
        if terminated or truncated:
            break

    print(
        f"object={args.object} pusher={args.pusher} obstacles={args.obstacles} "
        f"steps={i + 1} final_coverage={coverage:.3f}"
    )
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
