from __future__ import annotations

import argparse
import glob
import os
from multiprocessing import Pool

import numpy as np
import zarr


_ENV = None
_ARGS = None


def _init_worker(args_dict: dict):
    global _ENV, _ARGS
    _ARGS = args_dict
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    from Tsimulation.pushshapes import PushShapesEnv

    _ENV = PushShapesEnv(
        object_shape=_ARGS["object_shape"],
        pusher_shape=_ARGS["pusher_shape"],
        obstacle_level=0,
        render_mode="rgb_array",
    )


def _replay_one(ep: str) -> tuple[str, float, float, int, bool]:
    global _ENV, _ARGS
    z = zarr.open(ep, mode="r")
    state = np.asarray(z["observations.state"])
    actions = np.asarray(z["actions"])
    goal_pose = np.asarray(z["goal_pose"])
    if goal_pose.ndim == 1:
        goal_pose = goal_pose[None].repeat(state.shape[0], axis=0)
    nz = np.flatnonzero(np.any(state != 0, axis=1))
    if nz.size == 0:
        return ep, float("nan"), float("nan"), 0, False
    t_eff = int(nz.max() + 1)
    _ENV.reset(seed=0)
    _ENV.set_state(
        agent_pos=(float(state[0, 0]), float(state[0, 1])),
        object_pose=(float(state[0, 2]), float(state[0, 3]), float(state[0, 4])),
        goal_pose=tuple(float(x) for x in goal_pose[0, :3]),
    )
    cov = 0.0
    max_cov = 0.0
    steps = 0
    for t in range(t_eff):
        _, _, terminated, _, info = _ENV.step(actions[t].astype(np.float32))
        cov = float(info["coverage"])
        max_cov = max(max_cov, cov)
        steps += 1
        if terminated or max_cov >= float(_ARGS["coverage_threshold"]):
            break
    return ep, cov, max_cov, steps, bool(max_cov >= float(_ARGS["coverage_threshold"]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--object-shape", default="T")
    p.add_argument("--pusher-shape", default="circle")
    p.add_argument("--coverage-threshold", type=float, default=0.7)
    args = p.parse_args()

    eps = sorted(glob.glob(os.path.join(args.dataset, "*.zarr")))
    print(f"parallel replaying {len(eps)} episodes from {args.dataset}", flush=True)
    args_dict = {
        "object_shape": args.object_shape,
        "pusher_shape": args.pusher_shape,
        "coverage_threshold": args.coverage_threshold,
    }
    rows = []
    with Pool(processes=args.workers, initializer=_init_worker, initargs=(args_dict,)) as pool:
        for i, row in enumerate(pool.imap_unordered(_replay_one, eps, chunksize=8), 1):
            rows.append(row)
            if i % 100 == 0 or i == len(eps):
                vals = np.asarray([r[2] for r in rows if np.isfinite(r[2])], dtype=np.float64)
                print(
                    f"progress {i}/{len(eps)} max_cov_mean={vals.mean():.4f} "
                    f"max_cov_median={np.median(vals):.4f}",
                    flush=True,
                )

    finals = np.asarray([r[1] for r in rows if np.isfinite(r[1])], dtype=np.float64)
    maxes = np.asarray([r[2] for r in rows if np.isfinite(r[2])], dtype=np.float64)
    hits = int(sum(r[4] for r in rows))
    print(f"\n=== parallel replay coverage ({len(finals)} episodes, thr={args.coverage_threshold}) ===", flush=True)
    print(
        f"final coverage: mean={finals.mean():.6f} median={np.median(finals):.6f} "
        f"p25={np.percentile(finals,25):.6f} p75={np.percentile(finals,75):.6f}",
        flush=True,
    )
    print(
        f"max coverage:   mean={maxes.mean():.6f} median={np.median(maxes):.6f} "
        f"p25={np.percentile(maxes,25):.6f} p75={np.percentile(maxes,75):.6f}",
        flush=True,
    )
    print(
        f"episodes reaching >= {args.coverage_threshold}: {hits}/{len(finals)} "
        f"({hits/len(finals):.2%})",
        flush=True,
    )


if __name__ == "__main__":
    main()
