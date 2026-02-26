#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)
from scipy.spatial.transform import Rotation as R


def _infer_per_arm_dims(total_dims, num_arms):
    if total_dims % num_arms != 0:
        raise ValueError(f"Cannot split {total_dims} dims across {num_arms} arms.")
    d = total_dims // num_arms
    if d not in (
        6,
        7,
        8,
    ):  # 6: pos+euler, 7: +grip, 8: pos+quat(+grip?) assumed w/o grip
        # try 16 -> 8+8 (pos+quat per arm), 14 -> 7+7, 12 -> 6+6
        pass
    return d


def _split_actions_per_arm(actions, arms_order):
    """
    actions: (T, D)
    arms_order: list like ['left'] or ['right'] or ['left','right'] or ['right','left']
    Returns dict arm -> (T, d_per_arm)
    """
    T, D = actions.shape
    num_arms = len(arms_order)
    d_per_arm = _infer_per_arm_dims(D, num_arms)
    out = {}
    for i, arm in enumerate(arms_order):
        out[arm] = actions[:, i * d_per_arm : (i + 1) * d_per_arm]
    return out, d_per_arm


def _as_pos_rot_per_timestep(arr, orient="auto", euler_seq="xyz", quat_w_first=False):
    """
    arr: (T, d) where d in {6,7,8}
      - 6: [x,y,z, r,p,y] (Euler)
      - 7: [x,y,z, r,p,y, grip]  (Euler + gripper)
      - 8: [x,y,z, q?,q?,q?,q?]  (Quaternion; w-first if quat_w_first else x-first)
    orient:
      'auto' -> 6/7 => Euler, 8 => quat
      'euler' or 'quat' to force
    Returns:
      pos: (T,3)
      Rmats: (T,3,3)
    """
    T, d = arr.shape
    pos = arr[:, :3]
    if orient == "auto":
        if d in (6, 7):
            orient = "euler"
        elif d == 8:
            orient = "quat"
        else:
            raise ValueError(f"Unsupported per-arm dim {d}; expected 6/7/8.")
    if orient == "euler":
        eul = arr[:, 3:6]
        Rmats = R.from_euler(euler_seq, eul, degrees=False).as_matrix()
    else:
        q = arr[:, 3:7]
        if quat_w_first:
            # input q = [w,x,y,z] -> SciPy needs [x,y,z,w]
            q = np.concatenate([q[:, 1:], q[:, :1]], axis=1)
        # else assume q = [x,y,z,w] already
        Rmats = R.from_quat(q).as_matrix()
    return pos, Rmats


def _auto_stride(n, target_triads=50):
    return max(1, n // max(1, target_triads))


def _set_axes_equal(ax):
    # quick equalization by data ranges
    xlims = ax.get_xlim3d()
    ylims = ax.get_ylim3d()
    zlims = ax.get_zlim3d()
    xlen = xlims[1] - xlims[0]
    ylen = ylims[1] - ylims[0]
    zlen = zlims[1] - zlims[0]
    maxlen = max(xlen, ylen, zlen)

    def mid(lo, hi):
        return 0.5 * (lo + hi)

    ax.set_xlim3d([mid(*xlims) - maxlen / 2, mid(*xlims) + maxlen / 2])
    ax.set_ylim3d([mid(*ylims) - maxlen / 2, mid(*ylims) + maxlen / 2])
    ax.set_zlim3d([mid(*zlims) - maxlen / 2, mid(*zlims) + maxlen / 2])


def plot_cartesian_3d(
    h5_path,
    dataset_key="action",
    arms="right",
    layout="left-right",
    orient="auto",
    euler_seq="xyz",
    quat_w_first=False,
    stride=None,
    triad_len=0.05,
    max_triad_count=60,
    out_path="cartesian_plot.png",
    dpi=150,
    fmt="png",
):
    with h5py.File(h5_path, "r") as f:
        if dataset_key not in f:
            raise KeyError(f"'{dataset_key}' not in HDF5.")
        A = np.asarray(f[dataset_key][...], dtype=np.float64)

    if A.ndim != 2:
        raise ValueError(f"Expected 2D dataset, got shape {A.shape}")
    T, D = A.shape
    if T < D and T <= 32 and D > 32:
        A = A.T
        T, D = A.shape

    if arms == "both":
        arms_order = ["left", "right"] if layout == "left-right" else ["right", "left"]
    else:
        arms_order = [arms]

    all_per_arm, d_per_arm = _split_actions_per_arm(
        A,
        arms_order
        if len(arms_order) > 1
        else (["left", "right"] if D % 2 == 0 else [arms]),
    )
    if len(arms_order) == 1 and len(all_per_arm) == 2:
        all_per_arm = {arms_order[0]: all_per_arm[arms_order[0]]}

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(f"Cartesian trajectory ({', '.join(all_per_arm.keys())})")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    # stride for triads
    if stride is None:
        n = next(iter(all_per_arm.values())).shape[0]
        stride = _auto_stride(n, target_triads=max_triad_count)

    # one color scale for whole sequence
    for arm, arr in all_per_arm.items():
        pos, Rmats = _as_pos_rot_per_timestep(
            arr, orient=orient, euler_seq=euler_seq, quat_w_first=quat_w_first
        )
        t_norm = np.linspace(0.0, 1.0, pos.shape[0])
        sc = ax.scatter(
            pos[:, 0],
            pos[:, 1],
            pos[:, 2],
            c=t_norm,
            cmap="viridis",
            s=12,
            depthshade=False,
            label=f"{arm}",
        )

        # sparse triads
        idxs = np.arange(0, pos.shape[0], stride)
        for i in idxs:
            p = pos[i]
            R_i = Rmats[i]
            ax.quiver(
                p[0], p[1], p[2], *(R_i[:, 0] * triad_len), arrow_length_ratio=0.2
            )
            ax.quiver(
                p[0], p[1], p[2], *(R_i[:, 1] * triad_len), arrow_length_ratio=0.2
            )
            ax.quiver(
                p[0], p[1], p[2], *(R_i[:, 2] * triad_len), arrow_length_ratio=0.2
            )

    _set_axes_equal(ax)
    plt.tight_layout()
    cbar = plt.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("timestep (0 → end)")
    ax.legend(loc="best")

    import os

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight", format=fmt)
    print(f"Saved 3D plot to {out_path}")


def main():
    ap = argparse.ArgumentParser(
        description="Plot 6-DoF Cartesian actions (trajectory + local axes)."
    )
    ap.add_argument("--h5", required=True, help="Path to dataset (HDF5).")
    ap.add_argument(
        "--key", default="action", help="Dataset key to read (default: action)."
    )
    ap.add_argument("--arms", default="right", choices=["left", "right", "both"])
    ap.add_argument(
        "--layout",
        default="left-right",
        choices=["left-right", "right-left"],
        help="Order of arms within each action vector if both arms are present.",
    )
    ap.add_argument(
        "--orient",
        default="auto",
        choices=["auto", "euler", "quat"],
        help="Interpretation of orientation. 'auto': 6/7->euler, 8->quat.",
    )
    ap.add_argument(
        "--euler-seq",
        default="xyz",
        help="Euler sequence if using Euler (default xyz; intrinsic).",
    )
    ap.add_argument(
        "--quat-w-first",
        action="store_true",
        help="Set if quaternion stored as [w,x,y,z] (default assumes [x,y,z,w]).",
    )
    ap.add_argument(
        "--stride", type=int, default=None, help="Sample stride for drawing triads."
    )
    ap.add_argument(
        "--triad-len", type=float, default=0.05, help="Axis length for triads."
    )
    ap.add_argument(
        "--max-triads", type=int, default=60, help="Auto stride target triad count."
    )
    ap.add_argument("--out", default="cartesian_plot.png", help="Output image path")
    ap.add_argument("--dpi", type=int, default=150, help="Output DPI")
    ap.add_argument(
        "--format", default="png", choices=["png", "pdf", "svg", "jpg", "jpeg"]
    )
    args = ap.parse_args()

    plot_cartesian_3d(
        h5_path=args.h5,
        dataset_key=args.key,
        arms=args.arms,
        layout=args.layout,
        orient=args.orient,
        euler_seq=args.euler_seq,
        quat_w_first=args.quat_w_first,
        stride=args.stride,
        triad_len=args.triad_len,
        max_triad_count=args.max_triads,
        out_path=args.out,
        dpi=args.dpi,
        fmt=args.format,
    )


if __name__ == "__main__":
    main()
