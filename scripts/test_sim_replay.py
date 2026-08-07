"""Sanity test: replay GT actions through the sim env across a whole dataset.

Usage:
    python test_sim_replay.py [DATA_DIR] [N_EPISODES]
        DATA_DIR    default /coc/flash7/paphiwetsa3/datasets/circle/basic
        N_EPISODES  default 0 (= all)
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import zarr

from Tsimulation.pushshapes import PushShapesEnv

DEFAULT_DIR = Path("/coc/flash7/paphiwetsa3/datasets/circle/basic")
ENV_KWARGS = dict(
    object_shape="T",
    pusher_shape="circle",
    obstacle_level=0,
    image_size=96,
)


def replay(episode_dir: Path) -> tuple[float, int]:
    z = zarr.open_group(str(episode_dir), mode="r")
    state = np.asarray(z["observations.state"][:])
    actions = np.asarray(z["actions"][:])
    goal = np.asarray(z["goal_pose"][:])
    T = actions.shape[0]

    env = PushShapesEnv(**ENV_KWARGS)
    env.reset(seed=0)

    frame0 = state[0]
    agent_pos = (float(frame0[0]), float(frame0[1]))
    object_pose = (float(frame0[2]), float(frame0[3]), float(frame0[4]))
    goal_pose = tuple(float(x) for x in goal[0].reshape(-1)[:3])
    env.set_state(agent_pos=agent_pos, object_pose=object_pose, goal_pose=goal_pose)

    last_coverage = float(env.step(actions[0])[4].get("coverage", 0.0))
    for t in range(1, T):
        _, _, terminated, _, info = env.step(actions[t])
        last_coverage = float(info.get("coverage", 0.0))
        if terminated:
            break
    return last_coverage, T


def main() -> None:
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DIR
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 0  # 0 = all
    all_episodes = sorted(p for p in data_dir.iterdir() if p.name.endswith(".zarr"))
    episodes = all_episodes[:n] if n > 0 else all_episodes
    print(f"Replaying {len(episodes)} episodes from {data_dir}\n", flush=True)
    out_csv = Path(f"/tmp/replay_coverage_{data_dir.name}.csv")
    rows = []
    coverages = []
    for i, ep in enumerate(episodes):
        try:
            cov, T = replay(ep)
            coverages.append(cov)
            rows.append((ep.name, T, cov, ""))
            if (i + 1) % 20 == 0 or i < 5:
                print(
                    f"  [{i + 1:3d}/{len(episodes)}] {ep.name}: T={T:4d} cov={cov:.3f}",
                    flush=True,
                )
        except Exception as e:
            rows.append((ep.name, 0, 0.0, f"{type(e).__name__}: {e}"))
            print(
                f"  [{i + 1:3d}/{len(episodes)}] {ep.name}: ERROR {type(e).__name__}: {e}",
                flush=True,
            )

    with out_csv.open("w") as f:
        w = csv.writer(f)
        w.writerow(["episode", "T", "coverage", "error"])
        w.writerows(rows)
    print(f"\nSaved {len(rows)} rows to {out_csv}", flush=True)

    if coverages:
        arr = np.asarray(coverages)
        n_success = int((arr >= 0.7).sum())
        print(
            f"\nMean coverage: {arr.mean():.3f}  median: {np.median(arr):.3f}  min={arr.min():.3f}  max={arr.max():.3f}"
        )
        print(
            f"Success (>=0.7): {n_success}/{len(arr)} ({100 * n_success / len(arr):.1f}%)"
        )
        print(f"Cov >=0.9: {int((arr >= 0.9).sum())}/{len(arr)}")
        print(f"Cov ==0.0: {int((arr == 0.0).sum())}/{len(arr)}")
        # Coverage histogram in deciles.
        bins = np.arange(0.0, 1.01, 0.1)
        hist, _ = np.histogram(arr, bins=bins)
        print("\nDecile histogram:")
        for lo, hi, c in zip(bins[:-1], bins[1:], hist):
            print(f"  [{lo:.1f}, {hi:.1f}): {'#' * c} ({c})")


if __name__ == "__main__":
    main()
