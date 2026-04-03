#!/usr/bin/env python3
"""Visualize a local LeRobot v2 dataset: image grid and/or action vs time for one episode."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def _to_hwc(img) -> np.ndarray:
    if hasattr(img, "numpy"):
        img = img.numpy()
    x = np.asarray(img)
    if x.ndim == 3 and x.shape[0] in (1, 3):
        return np.transpose(x, (1, 2, 0))
    return x


def _episode_row_for_index(episodes: list, episode_index: int) -> int:
    """Map dataset episode_index (demo id) to row into episode_data_index from/to."""
    for row, ep in enumerate(episodes):
        if ep["episode_index"] == episode_index:
            return row
    raise ValueError(
        f"episode_index {episode_index} not found in meta/episodes.jsonl "
        f"(have {len(episodes)} episodes)."
    )


def _parse_dim_spec(spec: str | None, dim_max: int) -> list[int]:
    """Return list of dimension indices to plot. spec: 'all', '0:7', '0,2,5'."""
    if spec is None or spec.strip().lower() in ("", "all"):
        return list(range(dim_max))
    s = spec.strip()
    if ":" in s:
        parts = s.split(":")
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if len(parts) > 1 and parts[1] else dim_max
        return list(range(max(0, start), min(end, dim_max)))
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _tensor_to_numpy(v):
    if hasattr(v, "numpy"):
        return np.asarray(v.numpy(), dtype=np.float64)
    return np.asarray(v, dtype=np.float64)


def _scalar_float(v) -> float:
    return float(np.asarray(_tensor_to_numpy(v)).reshape(-1)[0])


def plot_action_in_episode(
    ds: LeRobotDataset,
    action_key: str,
    episode_index: int,
    dim_spec: str | None,
    out_path: Path,
    time_mode: str,
) -> None:
    """
    x-axis: time (seconds from episode start, or frame index).
    y-axis: one curve per selected action dimension.
    """
    if action_key not in ds.meta.features:
        raise ValueError(
            f"Unknown action key {action_key!r}. Known: {list(ds.meta.features.keys())}"
        )
    shape = ds.meta.shapes[action_key]
    if not shape or shape[0] < 1:
        raise ValueError(f"{action_key} has empty shape {shape}")
    dim_max = int(shape[0])

    row = _episode_row_for_index(ds.meta.episodes, episode_index)
    fr = int(ds.episode_data_index["from"][row].item())
    to_excl = int(ds.episode_data_index["to"][row].item())
    global_idxs = np.arange(fr, to_excl, dtype=np.int64)
    L = len(global_idxs)
    if L == 0:
        raise ValueError(f"Episode {episode_index} has length 0.")

    dim_indices = _parse_dim_spec(dim_spec, dim_max)
    if not dim_indices:
        raise ValueError("No dimensions to plot after --dims.")

    mat = np.zeros((L, len(dim_indices)), dtype=np.float64)
    times = np.zeros(L, dtype=np.float64)
    for i, gidx in enumerate(global_idxs):
        row_dict = ds[int(gidx)]
        vec = _tensor_to_numpy(row_dict[action_key]).reshape(-1)
        mat[i, :] = vec[dim_indices]
        times[i] = _scalar_float(row_dict["timestamp"])

    if time_mode == "frame":
        x = np.arange(L, dtype=np.float64)
        xlabel = "frame index (within episode)"
    else:
        x = times - times[0]
        xlabel = "time (s) from episode start"

    fig, ax = plt.subplots(figsize=(11, 5))
    for j, d in enumerate(dim_indices):
        ax.plot(x, mat[:, j], label=f"dim {d}", alpha=0.85, linewidth=1.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("value")
    ax.set_title(f"{action_key} | episode_index={episode_index} | L={L} | dims={dim_indices}")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(
        description="Visualize LeRobot v2 dataset: images and/or action vs time for one demo."
    )
    p.add_argument(
        "dataset_path",
        type=str,
        help="Root folder with meta/ and data/ (LeRobot v2).",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for PNGs. Default: <dataset_path>/viz_preview",
    )
    p.add_argument(
        "--action-key",
        "-k",
        type=str,
        default=None,
        help="LeRobot feature key, e.g. actions.joint. With --episode, plot y=value vs time.",
    )
    p.add_argument(
        "--episode",
        "-e",
        type=int,
        default=None,
        help="Dataset episode_index (one demo / trajectory).",
    )
    p.add_argument(
        "--dims",
        type=str,
        default=None,
        help="Which dimensions to plot: 'all' (default), slice '0:7', or list '0,1,5'.",
    )
    p.add_argument(
        "--time-axis",
        choices=("seconds", "frame"),
        default="seconds",
        help="seconds: x = timestamp - episode start (default). frame: 0..L-1.",
    )
    p.add_argument(
        "--only-action",
        action="store_true",
        help="Only write the action plot (--action-key and --episode required).",
    )
    p.add_argument("--n-frames", type=int, default=6, help="Images in the grid (preview).")
    p.add_argument(
        "--frame-indices",
        type=int,
        nargs="*",
        default=None,
        help="Explicit frame indices for image grid; default: linspace over dataset length.",
    )
    p.add_argument(
        "--no-images",
        action="store_true",
        help="Skip image grid (useful with --only-action).",
    )
    p.add_argument(
        "--list-keys",
        action="store_true",
        help="Print feature keys from meta/info.json and exit.",
    )
    args = p.parse_args()

    root = Path(args.dataset_path).resolve()
    out = Path(args.output_dir).resolve() if args.output_dir else root / "viz_preview"
    out.mkdir(parents=True, exist_ok=True)

    ds = LeRobotDataset(
        repo_id="local",
        root=str(root),
        local_files_only=True,
        download_videos=False,
    )

    if args.list_keys:
        for k in sorted(ds.meta.features.keys()):
            print(k)
        return

    if args.only_action and (args.action_key is None or args.episode is None):
        raise SystemExit("--only-action requires both --action-key and --episode")

    written = []

    # --- Optional: action vs time for one episode
    if args.action_key is not None and args.episode is not None:
        safe = re.sub(r"[^\w.\-]+", "_", args.action_key).replace(".", "_")
        action_path = out / f"action_{safe}_episode_{args.episode:06d}.png"
        plot_action_in_episode(
            ds,
            action_key=args.action_key,
            episode_index=args.episode,
            dim_spec=args.dims,
            out_path=action_path,
            time_mode=args.time_axis,
        )
        written.append(action_path)
        print(f"Wrote {action_path}")

    if args.only_action:
        return

    if args.no_images:
        if not written:
            raise SystemExit("Nothing to do: use --action-key/--episode or remove --no-images")
        return

    n = len(ds)
    cam_keys = ds.meta.camera_keys
    if not cam_keys:
        if not written:
            raise SystemExit("No image/video keys in dataset meta.")
        return

    if args.frame_indices is not None:
        idxs = [min(max(0, int(i)), n - 1) for i in args.frame_indices]
    else:
        idxs = [int(i) for i in np.linspace(0, n - 1, num=min(args.n_frames, n), dtype=int)]

    key = cam_keys[0]
    cols = min(3, len(idxs))
    rows = (len(idxs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, fi in zip(axes, idxs):
        im = _to_hwc(ds[fi][key])
        if im.shape[-1] == 1:
            ax.imshow(im.squeeze(-1), cmap="gray", vmin=0, vmax=1)
        else:
            ax.imshow(np.clip(im, 0, 1))
        ax.set_title(f"{key} idx={fi}")
        ax.axis("off")
    for ax in axes[len(idxs) :]:
        ax.axis("off")
    fig.suptitle(f"{root.name} | {key} | n={n} frames")
    fig.tight_layout()
    grid_path = out / "preview_images.png"
    fig.savefig(grid_path, dpi=120)
    plt.close(fig)
    written.append(grid_path)
    print(f"Wrote {grid_path}")

    # Legacy: first N frames joint preview (unless user only wanted episode plot)
    act_key = None
    for cand in ("actions.joint", "actions.joint_arm_hand"):
        if cand in ds.meta.features:
            act_key = cand
            break
    if act_key and args.action_key is None:
        span = min(300, n)
        D = min(12, ds.meta.shapes[act_key][0])
        mat = np.zeros((span, D), dtype=np.float64)
        for t in range(span):
            mat[t] = np.asarray(ds[t][act_key].numpy()[:D])
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        for d in range(D):
            ax2.plot(mat[:, d], alpha=0.7, linewidth=0.8)
        ax2.set_title(f"{act_key} first {D} dims (global frames 0..{span-1})")
        ax2.set_xlabel("global frame index")
        ax2.set_ylabel("value")
        fig2.tight_layout()
        trace_path = out / "preview_actions.png"
        fig2.savefig(trace_path, dpi=120)
        plt.close(fig2)
        written.append(trace_path)
        print(f"Wrote {trace_path}")


if __name__ == "__main__":
    main()
