"""Sanity test: replay GT actions through the sim env across a whole dataset.

Usage:
    python test_sim_replay.py [DATA_DIR] [N_EPISODES] [WORKERS]
        DATA_DIR    default /coc/flash7/paphiwetsa3/datasets/circle/basic
        N_EPISODES  default 0 (= all)
        WORKERS     default 1
"""

from __future__ import annotations

import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import zarr
from Tsimulation.collect.replay_init import reset_to_init
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
    attrs = dict(z.attrs)
    T = int(attrs.get("total_frames", z["actions"].shape[0]))
    actions = np.asarray(z["actions"][:T])
    raw_init = attrs.get("episode_init")
    episode_init = json.loads(raw_init) if isinstance(raw_init, str) else raw_init

    if episode_init is not None:
        env_kwargs = {
            "object_shape": episode_init.get("object_shape", "T"),
            "pusher_shape": episode_init.get("pusher_shape", "circle"),
            "obstacle_level": int(episode_init.get("obstacle_level", 0)),
            "image_size": 96,
        }
    else:
        env_kwargs = ENV_KWARGS
    env = PushShapesEnv(**env_kwargs)
    env._skip_obs_render = True
    speed = float(attrs.get("speed_factor", 1.0))
    env.PUSHER_SPEED = type(env).PUSHER_SPEED * speed
    env.STICK_TURN_RATE = type(env).STICK_TURN_RATE * speed
    try:
        if episode_init is not None:
            reset_to_init(env, episode_init)
        else:
            state = np.asarray(z["observations.state"][:T])
            goal = np.asarray(z["goal_pose"][:T])
            env.reset(seed=0)
            frame0 = state[0]
            env.set_state(
                agent_pos=(float(frame0[0]), float(frame0[1])),
                object_pose=(float(frame0[2]), float(frame0[3]), float(frame0[4])),
                goal_pose=tuple(float(x) for x in goal[0].reshape(-1)[:3]),
            )

        last_coverage = 0.0
        for action in actions:
            _, _, _, _, info = env.step(action)
            last_coverage = float(info.get("coverage", 0.0))
        return last_coverage, T
    finally:
        env.close()


def replay_row(episode_dir: Path) -> tuple[str, int, float, str]:
    try:
        coverage, frames = replay(episode_dir)
        return episode_dir.name, frames, coverage, ""
    except Exception as exc:
        return episode_dir.name, 0, 0.0, f"{type(exc).__name__}: {exc}"


def main() -> None:
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DIR
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 0  # 0 = all
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    if workers < 1:
        raise ValueError("WORKERS must be at least 1")
    all_episodes = sorted(p for p in data_dir.iterdir() if p.name.endswith(".zarr"))
    episodes = all_episodes[:n] if n > 0 else all_episodes
    print(f"Replaying {len(episodes)} episodes from {data_dir}\n", flush=True)
    out_csv = Path(f"/tmp/replay_coverage_{data_dir.name}.csv")
    rows = []
    coverages = []
    if workers == 1:
        completed_rows = map(replay_row, episodes)
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        futures = [executor.submit(replay_row, episode) for episode in episodes]
        completed_rows = (future.result() for future in as_completed(futures))
    try:
        for i, (episode_name, T, cov, error) in enumerate(completed_rows):
            if not error:
                coverages.append(cov)
            rows.append((episode_name, T, cov, error))
            if (i + 1) % 20 == 0 or i < 5 or error:
                detail = f"ERROR {error}" if error else f"T={T:4d} cov={cov:.3f}"
                print(f"  [{i + 1:4d}/{len(episodes)}] {episode_name}: {detail}", flush=True)
    finally:
        if workers > 1:
            executor.shutdown()

    with out_csv.open("w") as f:
        w = csv.writer(f)
        w.writerow(["episode", "T", "coverage", "error"])
        w.writerows(rows)
    print(f"\nSaved {len(rows)} rows to {out_csv}", flush=True)

    if coverages:
        arr = np.asarray(coverages)
        n_success = int((arr > 0.95).sum())
        print(
            f"\nMean coverage: {arr.mean():.3f}  median: {np.median(arr):.3f}  min={arr.min():.3f}  max={arr.max():.3f}"
        )
        print(
            f"Coverage (>0.95): {n_success}/{len(arr)} ({100 * n_success / len(arr):.1f}%)"
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
