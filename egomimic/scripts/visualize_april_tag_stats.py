#!/usr/bin/env python
"""Visualize and summarize LeRobot AprilTag pose observations."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt


DEFAULT_KEY = "obs.april_tag"
COMPONENT_NAMES = ["x", "y", "z", "rx", "ry", "rz"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read an xyz+rotvec AprilTag key from a LeRobot dataset, save pose-frame "
            "plots, and compute mean/std over all frames."
        )
    )
    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="LeRobot dataset root, e.g. datasets/0524_ye_push_cart_new_mink_eef_only",
    )
    parser.add_argument(
        "--key",
        default=DEFAULT_KEY,
        help=f"Pose key to read from parquet files. Default: {DEFAULT_KEY}",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=10,
        help="Number of evenly spaced global timesteps to plot when --timesteps is omitted.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        nargs="+",
        help="Explicit global row indices to plot after sorting by episode_index/frame_index/index.",
    )
    parser.add_argument(
        "--axis-length",
        type=float,
        default=0.12,
        help="Length of the plotted AprilTag frame axes in dataset xyz units.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for plots and stats. Default: <dataset_dir>/plot",
    )
    parser.add_argument(
        "--ddof",
        type=int,
        default=0,
        help="Delta degrees of freedom for std. Default 0 computes population std.",
    )
    parser.add_argument(
        "--compare-pose",
        type=float,
        nargs=6,
        metavar=("X", "Y", "Z", "RX", "RY", "RZ"),
        help="Optional xyz+rotvec pose to overlay on the all-data plot and compare with stats.",
    )
    parser.add_argument(
        "--compare-name",
        default="rollout",
        help="Label used for --compare-pose outputs. Default: rollout",
    )
    return parser.parse_args()


def _find_episode_files(dataset_dir: Path) -> list[Path]:
    data_dir = dataset_dir / "data"
    files = sorted(data_dir.glob("chunk-*/episode_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {data_dir}")
    return files


def _load_pose_rows(dataset_dir: Path, key: str) -> pd.DataFrame:
    frames = []
    columns = [key, "episode_index", "frame_index", "timestamp", "index"]

    for path in _find_episode_files(dataset_dir):
        try:
            frame = pd.read_parquet(path, columns=columns)
        except ValueError:
            available = pd.read_parquet(path).columns.tolist()
            raise KeyError(
                f"Could not read key '{key}' from {path}. Available columns include: {available[:30]}"
            ) from None
        frame["_source_file"] = str(path.relative_to(dataset_dir))
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    sort_columns = [col for col in ("episode_index", "frame_index", "index") if col in data.columns]
    data = data.sort_values(sort_columns).reset_index(drop=True)
    data["_global_index"] = np.arange(len(data), dtype=np.int64)
    return data


def _stack_pose_column(data: pd.DataFrame, key: str) -> np.ndarray:
    values = np.stack(data[key].to_numpy()).astype(np.float64)
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError(f"Expected '{key}' to have shape (N, 6), got {values.shape}")
    return values


def _rotvec_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(rotvec))
    if theta < 1e-12:
        return np.eye(3)

    axis = rotvec / theta
    x, y, z = axis
    skew = np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=np.float64,
    )
    return np.eye(3) + np.sin(theta) * skew + (1.0 - np.cos(theta)) * (skew @ skew)


def _set_equal_3d_axes(ax, points: np.ndarray, padding_ratio: float = 0.12) -> None:
    finite_points = points[np.isfinite(points).all(axis=1)]
    if len(finite_points) == 0:
        finite_points = np.zeros((1, 3), dtype=np.float64)

    mins = finite_points.min(axis=0)
    maxs = finite_points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float((maxs - mins).max()) / 2.0, 0.1)
    radius *= 1.0 + padding_ratio

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _format_float_list(values: np.ndarray) -> list[float]:
    return [float(v) for v in values]


def _select_timesteps(total_frames: int, sample_count: int, timesteps: list[int] | None) -> np.ndarray:
    if timesteps:
        indices = np.array(timesteps, dtype=np.int64)
    else:
        count = min(max(sample_count, 1), total_frames)
        indices = np.linspace(0, total_frames - 1, count, dtype=np.int64)
        indices = np.unique(indices)

    if np.any(indices < 0) or np.any(indices >= total_frames):
        raise IndexError(f"Timesteps must be in [0, {total_frames - 1}], got {indices.tolist()}")
    return indices


def _plot_pose_frame(
    poses: np.ndarray,
    row: pd.Series,
    timestep: int,
    output_path: Path,
    axis_length: float,
    key: str,
) -> None:
    xyz = poses[:, :3]
    pose = poses[timestep]
    origin = pose[:3]
    rotmat = _rotvec_to_matrix(pose[3:])

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color="0.70", linewidth=0.8, alpha=0.8)
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], color="0.72", s=4, alpha=0.18, depthshade=False)
    ax.scatter([origin[0]], [origin[1]], [origin[2]], color="black", s=36, depthshade=False)

    axis_specs = [
        ("x", "#d62728", rotmat[:, 0]),
        ("y", "#2ca02c", rotmat[:, 1]),
        ("z", "#1f77b4", rotmat[:, 2]),
    ]
    for label, color, direction in axis_specs:
        ax.quiver(
            origin[0],
            origin[1],
            origin[2],
            direction[0],
            direction[1],
            direction[2],
            length=axis_length,
            normalize=True,
            color=color,
            linewidth=2.5,
            label=f"{label}-axis",
        )

    title = (
        f"{key} frame | global={int(row['_global_index'])} "
        f"episode={int(row['episode_index'])} frame={int(row['frame_index'])}"
    )
    if "timestamp" in row and np.isfinite(row["timestamp"]):
        title += f" time={float(row['timestamp']):.3f}s"
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(loc="upper right")
    ax.view_init(elev=25, azim=-55)

    frame_points = np.vstack([xyz, origin + axis_length * rotmat.T])
    _set_equal_3d_axes(ax, frame_points)

    text = (
        f"xyz=({pose[0]:.4f}, {pose[1]:.4f}, {pose[2]:.4f})\n"
        f"rotvec=({pose[3]:.4f}, {pose[4]:.4f}, {pose[5]:.4f})"
    )
    ax.text2D(0.02, 0.02, text, transform=ax.transAxes, family="monospace", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _draw_frame_axes(ax, pose: np.ndarray, axis_length: float, *, linewidth: float, prefix: str) -> None:
    origin = pose[:3]
    rotmat = _rotvec_to_matrix(pose[3:])
    axis_specs = [
        ("x", "#d62728", rotmat[:, 0]),
        ("y", "#2ca02c", rotmat[:, 1]),
        ("z", "#1f77b4", rotmat[:, 2]),
    ]
    for label, color, direction in axis_specs:
        ax.quiver(
            origin[0],
            origin[1],
            origin[2],
            direction[0],
            direction[1],
            direction[2],
            length=axis_length,
            normalize=True,
            color=color,
            linewidth=linewidth,
            label=f"{prefix} {label}-axis",
        )


def _plot_compare_pose(
    poses: np.ndarray,
    compare_pose: np.ndarray,
    output_path: Path,
    axis_length: float,
    compare_name: str,
    key: str,
) -> None:
    xyz = poses[:, :3]
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color="0.72", linewidth=0.8, alpha=0.55, label="dataset order")
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], color="0.70", s=4, alpha=0.20, depthshade=False, label="dataset xyz")
    ax.scatter(
        [compare_pose[0]],
        [compare_pose[1]],
        [compare_pose[2]],
        color="#ff7f0e",
        edgecolor="black",
        linewidth=0.8,
        marker="*",
        s=180,
        depthshade=False,
        label=compare_name,
    )
    _draw_frame_axes(ax, compare_pose, axis_length, linewidth=3.0, prefix=compare_name)

    ax.set_title(f"{compare_name} pose over all {key} xyz samples")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=25, azim=-55)
    frame_points = np.vstack([xyz, compare_pose[:3], compare_pose[:3] + axis_length * _rotvec_to_matrix(compare_pose[3:]).T])
    _set_equal_3d_axes(ax, frame_points)
    text = (
        f"{compare_name}\n"
        f"xyz=({compare_pose[0]:.4f}, {compare_pose[1]:.4f}, {compare_pose[2]:.4f})\n"
        f"rotvec=({compare_pose[3]:.4f}, {compare_pose[4]:.4f}, {compare_pose[5]:.4f})"
    )
    ax.text2D(0.02, 0.02, text, transform=ax.transAxes, family="monospace", fontsize=9)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _compare_pose_stats(
    poses: np.ndarray,
    compare_pose: np.ndarray,
    stats: dict,
    output_dir: Path,
    compare_name: str,
) -> dict:
    mean = np.array([stats["mean"][name] for name in COMPONENT_NAMES], dtype=np.float64)
    std = np.array([stats["std"][name] for name in COMPONENT_NAMES], dtype=np.float64)
    mins = np.nanmin(poses, axis=0)
    maxs = np.nanmax(poses, axis=0)
    zscore = np.divide(compare_pose - mean, std, out=np.full_like(compare_pose, np.nan), where=std > 0)

    xyz_dist = np.linalg.norm(poses[:, :3] - compare_pose[:3], axis=1)
    nearest_xyz_index = int(np.nanargmin(xyz_dist))
    normalized_dist = np.linalg.norm((poses - compare_pose) / std, axis=1)
    nearest_full_index = int(np.nanargmin(normalized_dist))

    comparison = {
        "name": compare_name,
        "pose": dict(zip(COMPONENT_NAMES, _format_float_list(compare_pose))),
        "zscore_vs_all_data": dict(zip(COMPONENT_NAMES, _format_float_list(zscore))),
        "min": dict(zip(COMPONENT_NAMES, _format_float_list(mins))),
        "max": dict(zip(COMPONENT_NAMES, _format_float_list(maxs))),
        "outside_min_max": {
            name: bool(compare_pose[idx] < mins[idx] or compare_pose[idx] > maxs[idx])
            for idx, name in enumerate(COMPONENT_NAMES)
        },
        "nearest_xyz": {
            "global_index": nearest_xyz_index,
            "distance": float(xyz_dist[nearest_xyz_index]),
            "pose": dict(zip(COMPONENT_NAMES, _format_float_list(poses[nearest_xyz_index]))),
        },
        "nearest_full_normalized": {
            "global_index": nearest_full_index,
            "distance": float(normalized_dist[nearest_full_index]),
            "pose": dict(zip(COMPONENT_NAMES, _format_float_list(poses[nearest_full_index]))),
        },
    }
    output_path = output_dir / f"{compare_name}_april_tag_comparison.json"
    output_path.write_text(json.dumps(comparison, indent=2) + "\n")
    return comparison


def _write_stats(output_dir: Path, key: str, poses: np.ndarray, ddof: int) -> dict:
    mean = np.nanmean(poses, axis=0)
    std = np.nanstd(poses, axis=0, ddof=ddof)
    stats = {
        "key": key,
        "count": int(poses.shape[0]),
        "components": COMPONENT_NAMES,
        "mean": dict(zip(COMPONENT_NAMES, _format_float_list(mean))),
        "std": dict(zip(COMPONENT_NAMES, _format_float_list(std))),
        "ddof": int(ddof),
    }
    stats_path = output_dir / "april_tag_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2) + "\n")
    return stats


def main() -> None:
    args = _parse_args()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    output_dir = (args.output_dir or dataset_dir / "plot").expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data = _load_pose_rows(dataset_dir, args.key)
    poses = _stack_pose_column(data, args.key)
    timesteps = _select_timesteps(len(poses), args.sample_count, args.timesteps)
    stats = _write_stats(output_dir, args.key, poses, args.ddof)

    for sample_idx, timestep in enumerate(timesteps):
        row = data.iloc[int(timestep)]
        output_path = output_dir / (
            f"april_tag_frame_sample_{sample_idx:02d}_"
            f"global_{int(row['_global_index']):06d}_"
            f"episode_{int(row['episode_index']):06d}_"
            f"frame_{int(row['frame_index']):06d}.png"
        )
        _plot_pose_frame(poses, row, int(timestep), output_path, args.axis_length, args.key)

    print(f"Loaded {stats['count']} poses from {dataset_dir}")
    print(f"Saved {len(timesteps)} plots and stats to {output_dir}")
    print("Mean:")
    for name in COMPONENT_NAMES:
        print(f"  {name}: {stats['mean'][name]: .8f}")
    print("Std:")
    for name in COMPONENT_NAMES:
        print(f"  {name}: {stats['std'][name]: .8f}")

    if args.compare_pose is not None:
        compare_pose = np.array(args.compare_pose, dtype=np.float64)
        compare_name = args.compare_name.replace("/", "_")
        compare_plot_path = output_dir / f"{compare_name}_april_tag_over_all_data.png"
        _plot_compare_pose(poses, compare_pose, compare_plot_path, args.axis_length, compare_name, args.key)
        comparison = _compare_pose_stats(poses, compare_pose, stats, output_dir, compare_name)
        print(f"Saved comparison plot to {compare_plot_path}")
        print("Compare pose z-scores vs all data:")
        for name in COMPONENT_NAMES:
            outside = " outside-range" if comparison["outside_min_max"][name] else ""
            print(f"  {name}: {comparison['zscore_vs_all_data'][name]: .3f}{outside}")
        print(
            "Nearest dataset xyz distance: "
            f"{comparison['nearest_xyz']['distance']:.4f} at global "
            f"{comparison['nearest_xyz']['global_index']}"
        )


if __name__ == "__main__":
    main()
