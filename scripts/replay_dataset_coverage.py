"""Batch replay: dataset actions through PushShapesEnv, coverage statistics.

Replays N sampled episodes of a zarr dataset (recorded frame-0 state + goal
via env.set_state, then the RECORDED actions verbatim) and reports the
final/max coverage distribution. This is the data's achievable-coverage
reference under replay (NOTE: ~40% of episodes are known to diverge under
replay from physics chaos at contact points — recorded (obs,action) pairs are
still correct; see the 2026-05-24 data-quality notes).

Run (EgoVerse2, CPU is fine):
    python scripts/replay_dataset_coverage.py \
        --dataset /coc/flash7/paphiwetsa3/datasets/new_circle_3 --n 60
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import zarr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--object-shape", default="T")
    p.add_argument("--pusher-shape", default="circle")
    p.add_argument("--coverage-threshold", type=float, default=0.8)
    args = p.parse_args()

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    from Tsimulation.pushshapes import PushShapesEnv

    eps = sorted(glob.glob(os.path.join(args.dataset, "*.zarr")))
    rng = np.random.default_rng(args.seed)
    if len(eps) > args.n:
        eps = [eps[i] for i in rng.choice(len(eps), args.n, replace=False)]
    print(f"replaying {len(eps)} episodes from {args.dataset}")

    env = PushShapesEnv(
        object_shape=args.object_shape,
        pusher_shape=args.pusher_shape,
        obstacle_level=0,
        render_mode="rgb_array",
    )

    finals, maxes, hit = [], [], 0
    for ep in eps:
        z = zarr.open(ep, mode="r")
        state = np.asarray(z["observations.state"])
        actions = np.asarray(z["actions"])
        goal_pose = np.asarray(z["goal_pose"])
        if goal_pose.ndim == 1:
            goal_pose = goal_pose[None].repeat(state.shape[0], axis=0)
        nz = np.flatnonzero(np.any(state != 0, axis=1))
        if nz.size == 0:
            continue
        T = int(nz.max() + 1)
        env.reset(seed=0)
        env.set_state(
            agent_pos=(float(state[0, 0]), float(state[0, 1])),
            object_pose=(
                float(state[0, 2]), float(state[0, 3]), float(state[0, 4])
            ),
            goal_pose=tuple(float(x) for x in goal_pose[0, :3]),
        )
        covs = [0.0]
        for t in range(T):
            _, _, terminated, _, info = env.step(actions[t].astype(np.float32))
            covs.append(float(info["coverage"]))
            if terminated or covs[-1] >= args.coverage_threshold:
                break
        finals.append(covs[-1])
        maxes.append(max(covs))
        hit += int(max(covs) >= args.coverage_threshold)

    f, m = np.array(finals), np.array(maxes)
    print(f"\n=== replay coverage ({len(f)} episodes, thr={args.coverage_threshold}) ===")
    print(f"final coverage: mean={f.mean():.3f} median={np.median(f):.3f} "
          f"p25={np.percentile(f,25):.3f} p75={np.percentile(f,75):.3f}")
    print(f"max coverage:   mean={m.mean():.3f} median={np.median(m):.3f}")
    print(f"episodes reaching >= {args.coverage_threshold}: {hit}/{len(f)} "
          f"({hit/len(f):.1%})")


if __name__ == "__main__":
    main()
