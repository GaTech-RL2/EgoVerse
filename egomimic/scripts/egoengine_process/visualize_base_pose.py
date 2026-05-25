"""
Verify obs.base_pose_integrated for a given episode by plotting:
  - top-down (x, y) trajectory in world frame
  - periodic body-frame triads (red = body x-axis, green = body y-axis,
    blue = body z-axis pointing up) so orientation is visible

Run AFTER egoengine_lerobot_extract_base_pose.py.

Usage:
  python egomimic/scripts/egoengine_process/visualize_base_pose.py \
      datasets/0524_ye_push_cart_new_ours_basepose --episode-idx 0 \
      --output base_pose_ep0.png

  # Sanity-check integration by recomputing from raw action deltas on the fly:
  python egomimic/scripts/egoengine_process/visualize_base_pose.py \
      datasets/0524_ye_push_cart_new_ours_basepose --episode-idx 0 --overlay-recomputed

  # Dump every episode to a folder (one PNG per episode):
  python egomimic/scripts/egoengine_process/visualize_base_pose.py \
      datasets/0524_ye_push_cart_new_ours_basepose --all-episodes \
      --output-dir base_pose_plots/0524_ye_push_cart_new_ours
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

# Keep imports stable across reorg: pull the integrator from the sibling extractor.
from egomimic.scripts.egoengine_process.egoengine_lerobot_extract_base_pose import (
    integrate_base_deltas_base_frame,
    SOURCE_ACTION_KEY,
    FALLBACK_ACTION_KEY,
    BASE_DELTA_START,
    BASE_DELTA_END,
    FALLBACK_BASE_START,
    FALLBACK_BASE_END,
    NEW_OBS_KEY,
    _resolve_lerobot_root,
    _episode_index_scalar,
)


def _episode_rows(dataset, episode_idx: int):
    """Return rows belonging to `episode_idx`, sorted by frame_index."""
    indices = [
        i for i, ep in enumerate(dataset["episode_index"])
        if _episode_index_scalar(ep) == episode_idx
    ]
    if not indices:
        raise ValueError(f"episode_idx {episode_idx} not found in dataset")
    subset = dataset.select(indices)
    fi = np.array([_episode_index_scalar(x) for x in subset["frame_index"]])
    order = np.argsort(fi)
    return subset.select(order)


def _pose_from_dataset(subset) -> np.ndarray:
    if NEW_OBS_KEY not in subset.column_names:
        raise KeyError(
            f"{NEW_OBS_KEY!r} missing from dataset. Did you run "
            f"egoengine_lerobot_extract_base_pose.py first?"
        )
    return np.stack(
        [np.asarray(x, dtype=np.float32) for x in subset[NEW_OBS_KEY]], axis=0
    )


def _pose_from_actions(subset) -> np.ndarray:
    """Independently recompute the trajectory from the action deltas for sanity check."""
    if SOURCE_ACTION_KEY in subset.column_names:
        action_key, s, e = SOURCE_ACTION_KEY, BASE_DELTA_START, BASE_DELTA_END
    else:
        action_key, s, e = FALLBACK_ACTION_KEY, FALLBACK_BASE_START, FALLBACK_BASE_END
    actions = np.stack(
        [np.asarray(x, dtype=np.float32) for x in subset[action_key]], axis=0
    )
    return integrate_base_deltas_base_frame(actions[:, s:e])


def _add_triads_2d(ax, pose: np.ndarray, n_triads: int, length: float):
    """Draw body x/y axis arrows at evenly spaced points along the trajectory."""
    T = pose.shape[0]
    if T == 0:
        return
    step = max(1, T // n_triads)
    for i in range(0, T, step):
        x, y, th = pose[i]
        c, s = np.cos(th), np.sin(th)
        # body x in red
        ax.arrow(x, y, length * c, length * s,
                 head_width=length * 0.25, head_length=length * 0.25,
                 fc="red", ec="red", length_includes_head=True, alpha=0.9)
        # body y in green
        ax.arrow(x, y, -length * s, length * c,
                 head_width=length * 0.25, head_length=length * 0.25,
                 fc="green", ec="green", length_includes_head=True, alpha=0.9)


def _add_triads_3d(ax, pose: np.ndarray, n_triads: int, length: float):
    """Draw body x/y/z axes (z = world up) at evenly spaced points."""
    T = pose.shape[0]
    if T == 0:
        return
    step = max(1, T // n_triads)
    for i in range(0, T, step):
        x, y, th = pose[i]
        c, s = np.cos(th), np.sin(th)
        z = 0.0
        # body x (red), y (green), z (blue)
        ax.quiver(x, y, z, length * c, length * s, 0, color="red", linewidth=1.5)
        ax.quiver(x, y, z, -length * s, length * c, 0, color="green", linewidth=1.5)
        ax.quiver(x, y, z, 0, 0, length, color="blue", linewidth=1.5)


def _plot_one_episode(subset, ep_idx: int, out_path: Path,
                      n_triads: int, triad_length_frac: float,
                      overlay_recomputed: bool):
    pose = _pose_from_dataset(subset)
    recomputed = _pose_from_actions(subset) if overlay_recomputed else None

    T = pose.shape[0]
    bbox = np.ptp(pose[:, :2], axis=0).max() if T > 1 else 1.0
    triad_len = max(1e-3, triad_length_frac * (bbox if bbox > 0 else 1.0))

    fig = plt.figure(figsize=(13, 6))
    ax2 = fig.add_subplot(1, 2, 1)
    ax2.plot(pose[:, 0], pose[:, 1], "-", color="black", alpha=0.5, label="integrated")
    ax2.scatter(pose[0, 0], pose[0, 1], color="black", s=60, zorder=5, label="start (0,0)")
    ax2.scatter(pose[-1, 0], pose[-1, 1], color="black", marker="x", s=60, zorder=5, label="end")
    if recomputed is not None:
        ax2.plot(recomputed[:, 0], recomputed[:, 1], "--", color="orange", alpha=0.7,
                 label="recomputed from actions")
    _add_triads_2d(ax2, pose, n_triads, triad_len)
    ax2.set_xlabel("x_world [m]")
    ax2.set_ylabel("y_world [m]")
    ax2.set_aspect("equal", adjustable="datalim")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best", fontsize=9)
    ax2.set_title(
        f"ep {ep_idx}, T={T} frames | "
        f"final pose (x,y,θ°) = "
        f"({pose[-1,0]:+.3f}, {pose[-1,1]:+.3f}, {np.degrees(pose[-1,2]):+.1f}°)"
    )

    ax3 = fig.add_subplot(1, 2, 2, projection="3d")
    z = np.zeros(T)
    ax3.plot(pose[:, 0], pose[:, 1], z, "-", color="black", alpha=0.5)
    _add_triads_3d(ax3, pose, n_triads, triad_len)
    ax3.set_xlabel("x [m]")
    ax3.set_ylabel("y [m]")
    ax3.set_zlabel("z [m]")
    ax3.set_title("Body frames in world (red=x, green=y, blue=z↑)")
    span = max(np.ptp(pose[:, 0]), np.ptp(pose[:, 1]), 1e-2) * 0.6
    cx, cy = pose[:, 0].mean(), pose[:, 1].mean()
    ax3.set_xlim(cx - span, cx + span)
    ax3.set_ylim(cy - span, cy + span)
    ax3.set_zlim(-span / 2, span / 2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return pose


def main():
    p = argparse.ArgumentParser(
        description="Plot obs.base_pose_integrated for an episode (or all)."
    )
    p.add_argument("dataset_path", type=str, help="LeRobot dataset path (with the new key).")
    p.add_argument("--episode-idx", type=int, default=0,
                   help="Episode to visualize. Ignored if --all-episodes is set.")
    p.add_argument("--all-episodes", action="store_true",
                   help="Plot every episode found in the dataset.")
    p.add_argument("--output", type=str, default=None,
                   help="Output figure path (single-episode mode). Default: base_pose_ep{N}.png in CWD.")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Output directory for per-episode PNGs (required with --all-episodes; "
                        "default for that mode: base_pose_plots/<dataset_name>).")
    p.add_argument("--n-triads", type=int, default=20,
                   help="Number of body-frame triads to draw along the path.")
    p.add_argument("--triad-length-frac", type=float, default=0.05,
                   help="Triad axis length as a fraction of the trajectory bounding box.")
    p.add_argument("--overlay-recomputed", action="store_true",
                   help="Also recompute trajectory from raw action deltas as a sanity check.")
    args = p.parse_args()

    root = _resolve_lerobot_root(Path(args.dataset_path).resolve())
    ds = load_dataset("parquet", data_dir=str(root / "data"), split="train")

    if args.all_episodes:
        out_dir = Path(args.output_dir or f"base_pose_plots/{root.name}").resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        ep_ids = sorted({_episode_index_scalar(v) for v in ds["episode_index"]})
        print(f"Plotting {len(ep_ids)} episodes -> {out_dir}")
        for ep in ep_ids:
            subset = _episode_rows(ds, ep)
            out_path = out_dir / f"base_pose_ep{ep:03d}.png"
            pose = _plot_one_episode(subset, ep, out_path,
                                     args.n_triads, args.triad_length_frac,
                                     args.overlay_recomputed)
            print(f"  ep {ep:>3d}: T={pose.shape[0]:>4d}, "
                  f"final (x,y,θ°) = ({pose[-1,0]:+.3f}, {pose[-1,1]:+.3f}, "
                  f"{np.degrees(pose[-1,2]):+.1f}°)  -> {out_path.name}")
        print(f"Done. {len(ep_ids)} figures in {out_dir}")
        return

    subset = _episode_rows(ds, args.episode_idx)
    out_path = Path(args.output or f"base_pose_ep{args.episode_idx}.png").resolve()
    pose = _plot_one_episode(subset, args.episode_idx, out_path,
                             args.n_triads, args.triad_length_frac,
                             args.overlay_recomputed)
    T = pose.shape[0]
    print(f"Saved figure: {out_path}")
    print(f"Episode {args.episode_idx}: T={T} | "
          f"x∈[{pose[:,0].min():+.3f},{pose[:,0].max():+.3f}], "
          f"y∈[{pose[:,1].min():+.3f},{pose[:,1].max():+.3f}], "
          f"θ°∈[{np.degrees(pose[:,2].min()):+.1f},{np.degrees(pose[:,2].max()):+.1f}]")


if __name__ == "__main__":
    main()
