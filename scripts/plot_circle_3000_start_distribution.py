from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import zarr


def _first_nonzero_state(state: np.ndarray) -> np.ndarray:
    nz = np.flatnonzero(np.any(state != 0, axis=1))
    idx = int(nz[0]) if nz.size else 0
    return state[idx]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--world-size", type=float, default=512.0)
    args = p.parse_args()

    eps = sorted(glob.glob(os.path.join(args.dataset, "*.zarr")))
    if not eps:
        raise SystemExit(f"no .zarr episodes found under {args.dataset}")

    pusher_xy = []
    object_xy = []
    object_theta = []
    goal_xy = []
    goal_theta = []
    bad = []

    for ep in eps:
        try:
            z = zarr.open(ep, mode="r")
            state = np.asarray(z["observations.state"], dtype=np.float64)
            st0 = _first_nonzero_state(state)
            goal = np.asarray(z["goal_pose"], dtype=np.float64)
            if goal.ndim == 2:
                g0 = goal[0]
            else:
                g0 = goal.reshape(-1)
            pusher_xy.append(st0[:2])
            object_xy.append(st0[2:4])
            object_theta.append(st0[4])
            goal_xy.append(g0[:2])
            goal_theta.append(g0[2] if g0.size >= 3 else np.nan)
        except Exception as exc:  # keep plotting the good episodes
            bad.append((ep, repr(exc)))

    pusher_xy = np.asarray(pusher_xy, dtype=np.float64)
    object_xy = np.asarray(object_xy, dtype=np.float64)
    goal_xy = np.asarray(goal_xy, dtype=np.float64)
    object_theta = np.asarray(object_theta, dtype=np.float64)
    goal_theta = np.asarray(goal_theta, dtype=np.float64)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), constrained_layout=True)
    items = [
        ("Pusher Start", pusher_xy, "#2f80ed"),
        ("Object Start", object_xy, "#27ae60"),
        ("Target / Goal", goal_xy, "#eb5757"),
    ]
    for ax, (title, pts, color) in zip(axes, items):
        ax.scatter(pts[:, 0], pts[:, 1], s=6, alpha=0.28, c=color, edgecolors="none")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlim(0, args.world_size)
        ax.set_ylim(args.world_size, 0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True, linewidth=0.5, alpha=0.25)
        ax.axvline(0, color="black", linewidth=1.1)
        ax.axvline(args.world_size, color="black", linewidth=1.1)
        ax.axhline(0, color="black", linewidth=1.1)
        ax.axhline(args.world_size, color="black", linewidth=1.1)

    fig.suptitle(
        f"circle_3000 start/target position distribution ({len(pusher_xy)} episodes)",
        fontsize=16,
        fontweight="bold",
    )
    fig.savefig(out_dir / "circle_3000_start_target_distribution.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6.5), constrained_layout=True)
    ax.scatter(pusher_xy[:, 0], pusher_xy[:, 1], s=5, alpha=0.18, c="#2f80ed", label="pusher start")
    ax.scatter(object_xy[:, 0], object_xy[:, 1], s=5, alpha=0.18, c="#27ae60", label="object start")
    ax.scatter(goal_xy[:, 0], goal_xy[:, 1], s=10, alpha=0.65, c="#eb5757", label="target/goal")
    ax.set_xlim(0, args.world_size)
    ax.set_ylim(args.world_size, 0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("circle_3000 Combined Start/Target Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, linewidth=0.5, alpha=0.25)
    ax.legend(loc="upper right", frameon=True)
    fig.savefig(out_dir / "circle_3000_start_target_distribution_combined.png", dpi=180)
    plt.close(fig)

    def stats(name: str, pts: np.ndarray) -> dict:
        return {
            "name": name,
            "count": int(len(pts)),
            "x_min": float(np.min(pts[:, 0])),
            "x_max": float(np.max(pts[:, 0])),
            "x_mean": float(np.mean(pts[:, 0])),
            "x_std": float(np.std(pts[:, 0])),
            "y_min": float(np.min(pts[:, 1])),
            "y_max": float(np.max(pts[:, 1])),
            "y_mean": float(np.mean(pts[:, 1])),
            "y_std": float(np.std(pts[:, 1])),
        }

    summary = {
        "dataset": args.dataset,
        "episodes": int(len(pusher_xy)),
        "bad_episodes": bad[:20],
        "bad_count": int(len(bad)),
        "pusher_start": stats("pusher_start", pusher_xy),
        "object_start": stats("object_start", object_xy),
        "target_goal": stats("target_goal", goal_xy),
        "object_theta_min": float(np.nanmin(object_theta)),
        "object_theta_max": float(np.nanmax(object_theta)),
        "goal_theta_min": float(np.nanmin(goal_theta)),
        "goal_theta_max": float(np.nanmax(goal_theta)),
    }
    (out_dir / "circle_3000_start_target_distribution_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(json.dumps(summary, indent=2), flush=True)
    print(f"wrote {out_dir / 'circle_3000_start_target_distribution.png'}", flush=True)
    print(f"wrote {out_dir / 'circle_3000_start_target_distribution_combined.png'}", flush=True)


if __name__ == "__main__":
    main()
