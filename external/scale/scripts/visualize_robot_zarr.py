#!/usr/bin/env python3
"""
Visualize Scale ALOHA robot Zarr episodes.

Produces:
  1. A 2x2 camera grid video (mp4) with EE trajectory dots on the front cam
  2. Joint trajectory plots (png)
  3. Cartesian EE trajectory plots (png)

Usage:
  python visualize_robot_zarr.py <zarr_episode_path> [--out-dir ./viz_out]
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import simplejpeg
import zarr


# cam_high intrinsics from MCAP CameraInfo at native 1280x720:
#   fx=648.79  fy=648.09  cx=649.58  cy=360.48
# Scaled for non-uniform resize to 640x480 (sx=0.5, sy=0.667):
CAM_INTRINSICS_640x480 = np.array(
    [[324.40, 0.0, 324.79, 0.0], [0.0, 432.06, 240.32, 0.0], [0.0, 0.0, 1.0, 0.0]]
)


def load_episode(zarr_path: str) -> dict:
    root = zarr.open_group(zarr_path, mode="r")
    data = {}
    for key in [
        "observations.state.joint_positions",
        "observations.state.ee_pose",
        "actions_joints",
        "actions_cartesian",
    ]:
        data[key] = root[key][:]

    data["images"] = {}
    for key in [
        "observations.images.front_img_1",
        "observations.images.cam_low",
        "observations.images.left_wrist_img",
        "observations.images.right_wrist_img",
    ]:
        raw = root[key][:]
        data["images"][key] = raw

    data["attrs"] = dict(root.attrs)
    return data


def decode_jpeg(raw) -> np.ndarray:
    if isinstance(raw, np.ndarray) and raw.ndim == 0:
        raw = raw.item()
    if isinstance(raw, bytes):
        return simplejpeg.decode_jpeg(raw, colorspace="RGB")
    return raw


def project_to_pixels(xyz_cam: np.ndarray, K: np.ndarray) -> tuple[int, int]:
    """Project a 3D point in camera frame to 2D pixel coordinates."""
    if xyz_cam[2] <= 0.01:
        return None
    pts = np.array([xyz_cam[0], xyz_cam[1], xyz_cam[2], 1.0])
    px = K @ pts
    px = px / px[2]
    return int(px[0]), int(px[1])


def draw_ee_dots(
    img: np.ndarray,
    ee_pose: np.ndarray,
    actions_cart: np.ndarray,
    K: np.ndarray,
    trail: list[np.ndarray] = None,
):
    """Draw EE position dots and optional trail on the front camera image."""
    h, w = img.shape[:2]

    if trail:
        for i, past_ee in enumerate(trail):
            alpha = 0.3 + 0.7 * (i / max(len(trail), 1))
            for xyz, color in [(past_ee[:3], (0, int(180 * alpha), 0)),
                               (past_ee[7:10], (0, int(120 * alpha), 0))]:
                pt = project_to_pixels(xyz, K)
                if pt and 0 <= pt[0] < w and 0 <= pt[1] < h:
                    cv2.circle(img, pt, 3, color, -1)

    for xyz, color, label in [
        (ee_pose[:3], (0, 255, 0), "L_obs"),
        (ee_pose[7:10], (0, 200, 50), "R_obs"),
        (actions_cart[:3], (255, 80, 80), "L_act"),
        (actions_cart[7:10], (200, 50, 50), "R_act"),
    ]:
        pt = project_to_pixels(xyz, K)
        if pt and 0 <= pt[0] < w and 0 <= pt[1] < h:
            cv2.circle(img, pt, 7, color, -1)
            cv2.circle(img, pt, 7, (255, 255, 255), 1)
            cv2.putText(img, label, (pt[0] + 10, pt[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    return img


def make_grid_frame(
    images: dict,
    t: int,
    ee_pose: np.ndarray,
    actions_cart: np.ndarray,
    K: np.ndarray,
    trail: list[np.ndarray] = None,
) -> np.ndarray:
    """Build a 2x2 camera grid for timestep t."""
    cam_map = {
        "front_img_1": "observations.images.front_img_1",
        "cam_low": "observations.images.cam_low",
        "left_wrist": "observations.images.left_wrist_img",
        "right_wrist": "observations.images.right_wrist_img",
    }

    decoded = {}
    for short, full in cam_map.items():
        decoded[short] = decode_jpeg(images[full][t])

    front = decoded["front_img_1"].copy()
    draw_ee_dots(front, ee_pose, actions_cart, K, trail=trail)

    labels = {
        "front_img_1": "FRONT (cam_high) + EE",
        "cam_low": "CAM LOW",
        "left_wrist": "LEFT WRIST",
        "right_wrist": "RIGHT WRIST",
    }
    for name, img in decoded.items():
        if name == "front_img_1":
            img = front
            decoded[name] = img
        label = labels[name]
        cv2.putText(img, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)
        cv2.putText(img, f"t={t}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 200, 200), 1)

    top = np.concatenate([decoded["front_img_1"], decoded["cam_low"]], axis=1)
    bot = np.concatenate([decoded["left_wrist"], decoded["right_wrist"]], axis=1)
    return np.concatenate([top, bot], axis=0)


def plot_joint_trajectories(data: dict, out_path: str):
    """Plot joint position trajectories for both arms."""
    obs = data["observations.state.joint_positions"]
    act = data["actions_joints"]
    T = obs.shape[0]
    t = np.arange(T)

    joint_labels = [
        "shoulder_pan", "shoulder_lift", "elbow_flex",
        "wrist_angle", "wrist_rotate", "wrist_roll", "gripper",
    ]

    fig, axes = plt.subplots(7, 2, figsize=(16, 20), sharex=True)
    fig.suptitle("Joint Trajectories: Follower (obs) vs Leader (act)", fontsize=14)

    for i in range(7):
        for col, arm, offset in [(0, "Left", 0), (1, "Right", 7)]:
            ax = axes[i, col]
            ax.plot(t, obs[:, offset + i], label="follower", color="tab:blue", alpha=0.8)
            ax.plot(t, act[:, offset + i], label="leader", color="tab:orange",
                    alpha=0.8, linestyle="--")
            ax.set_ylabel(joint_labels[i], fontsize=9)
            if i == 0:
                ax.set_title(f"{arm} Arm")
                ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel("Frame")
    axes[-1, 1].set_xlabel("Frame")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved joint trajectories -> {out_path}")


def plot_cartesian_trajectories(data: dict, out_path: str):
    """Plot Cartesian EE pose trajectories."""
    ee = data["observations.state.ee_pose"]
    ac = data["actions_cartesian"]
    T = ee.shape[0]
    t = np.arange(T)

    dim_labels = ["x", "y", "z", "yaw", "pitch", "roll", "gripper"]

    fig, axes = plt.subplots(7, 2, figsize=(16, 20), sharex=True)
    fig.suptitle("Cartesian EE Pose: ee_pose (obs) vs actions_cartesian (act)", fontsize=14)

    for i in range(7):
        for col, arm, offset in [(0, "Left", 0), (1, "Right", 7)]:
            ax = axes[i, col]
            ax.plot(t, ee[:, offset + i], label="ee_pose", color="tab:green", alpha=0.8)
            ax.plot(t, ac[:, offset + i], label="actions_cart", color="tab:red",
                    alpha=0.8, linestyle="--")
            ax.set_ylabel(dim_labels[i], fontsize=9)
            if i == 0:
                ax.set_title(f"{arm} Arm")
                ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel("Frame")
    axes[-1, 1].set_xlabel("Frame")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved Cartesian trajectories -> {out_path}")


def plot_3d_trajectories(data: dict, out_path: str):
    """3D scatter of EE positions in camera frame."""
    ee = data["observations.state.ee_pose"]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(ee[:, 0], ee[:, 1], ee[:, 2], c="green", s=3, alpha=0.5, label="Left EE")
    ax.scatter(ee[:, 7], ee[:, 8], ee[:, 9], c="blue", s=3, alpha=0.5, label="Right EE")

    ax.set_xlabel("X (cam)")
    ax.set_ylabel("Y (cam)")
    ax.set_zlabel("Z (cam)")
    ax.set_title("EE Positions in Camera Frame")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved 3D trajectory -> {out_path}")


def make_video(data: dict, out_path: str, K: np.ndarray, max_frames: int = 0,
               trail_len: int = 15):
    """Create a 2x2 camera grid video with EE overlays."""
    ee = data["observations.state.ee_pose"]
    ac = data["actions_cartesian"]
    T = ee.shape[0]
    if max_frames > 0:
        T = min(T, max_frames)

    first_frame = make_grid_frame(data["images"], 0, ee[0], ac[0], K)
    h, w = first_frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, 30, (w, h))

    trail = []
    for t in range(T):
        trail.append(ee[t])
        if len(trail) > trail_len:
            trail.pop(0)
        frame = make_grid_frame(data["images"], t, ee[t], ac[t], K, trail=list(trail))
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)

        if (t + 1) % 100 == 0 or t == T - 1:
            print(f"    Frame {t + 1}/{T}")

    writer.release()
    print(f"  Saved video ({T} frames) -> {out_path}")


def _make_combined_video(all_data: list[dict], out_path: str, K: np.ndarray,
                         trail_len: int = 15):
    """Stitch multiple episodes into one continuous video with EE overlays."""
    total_frames = sum(d["observations.state.ee_pose"].shape[0] for d in all_data)
    first = make_grid_frame(all_data[0]["images"], 0,
                            all_data[0]["observations.state.ee_pose"][0],
                            all_data[0]["actions_cartesian"][0], K)
    h, w = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, 30, (w, h))

    trail = []
    frame_idx = 0
    for ep_i, data in enumerate(all_data):
        ee = data["observations.state.ee_pose"]
        ac = data["actions_cartesian"]
        T = ee.shape[0]
        for t in range(T):
            trail.append(ee[t])
            if len(trail) > trail_len:
                trail.pop(0)
            frame = make_grid_frame(data["images"], t, ee[t], ac[t], K, trail=list(trail))
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)
            frame_idx += 1
            if frame_idx % 100 == 0 or frame_idx == total_frames:
                print(f"    Frame {frame_idx}/{total_frames} (episode {ep_i + 1}/{len(all_data)})")
        trail.clear()

    writer.release()
    print(f"  Saved combined video ({total_frames} frames) -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize Scale ALOHA robot Zarr episodes")
    parser.add_argument("zarr_paths", nargs="+", help="Path(s) to .zarr episodes or directories of episodes")
    parser.add_argument("--out-dir", default="./viz_out", help="Output directory")
    parser.add_argument("--max-frames", type=int, default=0, help="Max frames for video (0=all)")
    parser.add_argument("--no-video", action="store_true", help="Skip video generation")
    parser.add_argument("--trail-len", type=int, default=15, help="EE trail length in frames")
    parser.add_argument("--combine", action="store_true", help="Stitch all episodes into one combined video")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    episodes: list[Path] = []
    for zp in args.zarr_paths:
        zarr_path = Path(zp)
        if zarr_path.suffix == ".zarr":
            episodes.append(zarr_path)
        else:
            found = sorted(zarr_path.glob("*.zarr"))
            if not found:
                found = sorted(p for d in zarr_path.iterdir() if d.is_dir()
                                for p in d.glob("*.zarr"))
            episodes.extend(found)

    if not episodes:
        print(f"No .zarr episodes found")
        return 1

    K = CAM_INTRINSICS_640x480
    print(f"Visualizing {len(episodes)} episode(s)\n")

    for ep_path in episodes:
        name = ep_path.stem
        print(f"[{name}]")

        data = load_episode(str(ep_path))
        T = data["observations.state.ee_pose"].shape[0]
        emb = data["attrs"].get("embodiment", "unknown")
        fps = data["attrs"].get("fps", 30)
        print(f"  {T} frames, {emb}, {fps}Hz")

        ep_out = out_dir / name
        ep_out.mkdir(parents=True, exist_ok=True)

        plot_joint_trajectories(data, str(ep_out / "joints.png"))
        plot_cartesian_trajectories(data, str(ep_out / "cartesian.png"))
        plot_3d_trajectories(data, str(ep_out / "ee_3d.png"))

        snapshot = make_grid_frame(data["images"], T // 2,
                                   data["observations.state.ee_pose"][T // 2],
                                   data["actions_cartesian"][T // 2], K)
        snapshot_bgr = cv2.cvtColor(snapshot, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(ep_out / "snapshot.png"), snapshot_bgr)
        print(f"  Saved snapshot -> snapshot.png")

        if not args.no_video and not args.combine:
            make_video(data, str(ep_out / "video.mp4"), K,
                       max_frames=args.max_frames, trail_len=args.trail_len)

        print()

    if args.combine and not args.no_video and len(episodes) > 0:
        print("Creating combined video across all episodes...")
        combined_path = str(out_dir / "combined_video.mp4")
        all_data = [load_episode(str(ep)) for ep in episodes]
        _make_combined_video(all_data, combined_path, K,
                             trail_len=args.trail_len)

    print("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
