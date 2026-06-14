"""Replay episodes from a zarr dataset by feeding recorded actions into the sim.
Outputs mp4 videos for visual inspection.

Usage:
    python replay_dataset.py <zarr_dir> <output_dir> [--n_episodes 50]
"""
import sys
import json
import numpy as np
import zarr
from pathlib import Path
import random
import imageio
import traceback

sys.path.insert(0, "/coc/flash7/paphiwetsa3/projects/EgoVerse2")
from Tsimulation.pushshapes import PushShapesEnv


def replay_episode(zarr_path, env):
    """Replay one episode, return frames and final coverage."""
    z = zarr.open_group(str(zarr_path), mode="r")

    # Get initial state from episode_init attr
    if "episode_init" not in z.attrs:
        # Fallback: reconstruct from observations.state first frame
        state = np.asarray(z["observations.state"][0])
        agent_pos = (float(state[0]), float(state[1]))
        object_pose = (float(state[2]), float(state[3]), float(state[4]))
        goal = np.asarray(z["goal_pose"][0])
        goal_pose = (float(goal[0]), float(goal[1]), float(goal[2]))
    else:
        init = json.loads(z.attrs["episode_init"])
        agent_pos = tuple(init["agent_pos"])
        object_pose = tuple(init["object_pose"])
        goal_pose = tuple(init["goal_pose"])

    actions = np.asarray(z["actions"][:])

    # Reset and set initial state
    env.reset()
    env.set_state(agent_pos=agent_pos, object_pose=object_pose, goal_pose=goal_pose)

    frames = []
    # Render initial frame
    frame = env.render()
    if frame is not None:
        frames.append(frame)

    final_cov = 0.0
    for t in range(actions.shape[0]):
        action = actions[t]
        # Skip zero actions (idle tail)
        if np.allclose(action, 0.0):
            continue
        obs, rew, term, trunc, info = env.step(action)
        final_cov = info.get("coverage", 0.0)
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        if term:
            break

    return frames, final_cov


def main():
    if len(sys.argv) < 3:
        print("Usage: python replay_dataset.py <zarr_dir> <output_dir> [--n_episodes 50]")
        sys.exit(1)

    zarr_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    n_episodes = 50

    if "--n_episodes" in sys.argv:
        idx = sys.argv.index("--n_episodes")
        n_episodes = int(sys.argv[idx + 1])

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all zarr episodes
    episodes = sorted([p for p in zarr_dir.iterdir() if p.name.endswith(".zarr")])
    print(f"Found {len(episodes)} episodes in {zarr_dir}")

    if len(episodes) == 0:
        print("ERROR: No zarr episodes found!")
        sys.exit(1)

    # Sample random subset
    random.seed(42)
    if n_episodes < len(episodes):
        selected = random.sample(episodes, n_episodes)
        selected.sort()
    else:
        selected = episodes

    env = PushShapesEnv(
        object_shape="T",
        pusher_shape="circle",
        obstacle_level=0,
        image_size=96,
        render_mode="rgb_array",
    )

    coverages = []
    errors = []
    for i, ep_path in enumerate(selected):
        try:
            frames, cov = replay_episode(ep_path, env)
            coverages.append(cov)

            # Write video
            out_path = output_dir / f"replay_{ep_path.stem}.mp4"
            if len(frames) > 0:
                imageio.mimsave(str(out_path), frames, fps=30)

            status = "OK" if cov > 0.5 else "LOW"
            print(f"[{i+1}/{len(selected)}] {ep_path.name}: cov={cov:.3f} frames={len(frames)} [{status}]")
        except Exception as e:
            errors.append((ep_path.name, str(e)))
            print(f"[{i+1}/{len(selected)}] {ep_path.name}: ERROR - {e}")

    # Summary
    coverages = np.array(coverages) if coverages else np.array([])
    print(f"\n=== SUMMARY ===")
    print(f"Episodes replayed: {len(coverages)}")
    print(f"Errors: {len(errors)}")
    if len(coverages) > 0:
        print(f"Mean coverage: {coverages.mean():.3f}")
        print(f"Median coverage: {np.median(coverages):.3f}")
        print(f"Min coverage: {coverages.min():.3f}")
        print(f"Max coverage: {coverages.max():.3f}")
        print(f">50% coverage: {(coverages > 0.5).sum()}/{len(coverages)}")
        print(f">90% coverage: {(coverages > 0.9).sum()}/{len(coverages)}")
    if errors:
        print(f"\nFailed episodes:")
        for name, err in errors:
            print(f"  {name}: {err}")
    print(f"\nVideos saved to: {output_dir}")


if __name__ == "__main__":
    main()
