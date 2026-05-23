"""End-effector linear-velocity plots for every Zarr recording under a
data folder, grouped by embodiment.

Three outputs (all under `--out-dir`):

  1. distribution/<embodiment>.png
     Per-embodiment histogram of speeds (m/s). One faint curve per
     recording, bold red curve for the across-recording aggregate.
     Three panels: left arm, right arm, both.

  2. distribution/comparison.png
     All embodiments overlaid on one figure (one curve per embodiment,
     using the aggregate of every frame in that embodiment). Same three
     panels. This is the aria-vs-eva-vs-... view.

  3. timeseries/<embodiment>/grid.png
     Speed vs time, one small subplot per recording (capped at
     `--max-timeseries`). Two lines per subplot (left + right arm).
     Plus per-recording PNGs in timeseries/<embodiment>/<recording>.png
     when --per-recording is set.

Plus summary.csv with per-recording mean/median/p95/max speed.

Usage:
  python egomimic/scripts/data_visualization/velocity_distribution.py \\
      --data-dir /storage/project/r-dxu345-0/agao81/pick_place \\
      --out-dir  logs/pick_place/velocity_distribution
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import zarr

ARMS = ("left", "right")
EMB_PALETTE = {
    "aria_bimanual": "#1f77b4",
    "aria_left_arm": "#17becf",
    "aria_right_arm": "#9467bd",
    "eva_bimanual": "#ff7f0e",
    "eva_right_arm": "#d62728",
    "scale": "#2ca02c",
}


def _color_for(emb: str) -> str:
    if emb in EMB_PALETTE:
        return EMB_PALETTE[emb]
    # deterministic fallback for unseen embodiments
    h = abs(hash(emb)) % 10
    return [
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
    ][h]


def _load_recording(zarr_path: str) -> dict | None:
    meta_path = os.path.join(zarr_path, "zarr.json")
    if not os.path.isfile(meta_path):
        return None
    with open(meta_path) as f:
        attrs = json.load(f).get("attributes", {})
    emb = attrs.get("embodiment")
    fps = attrs.get("fps")
    T = attrs.get("total_frames")
    if emb is None or not fps:
        return None
    store = zarr.open(zarr_path, mode="r")
    out = {"embodiment": emb, "fps": float(fps), "name": os.path.basename(zarr_path)}
    for arm in ARMS:
        key = f"{arm}.obs_ee_pose"
        if key in store:
            arr = store[key][:]
            if T is not None and arr.shape[0] > T:
                arr = arr[:T]
            if arr.ndim == 2 and arr.shape[-1] >= 3:
                out[arm] = arr[:, :3].astype(np.float64)
    if "left" not in out and "right" not in out:
        return None
    return out


# Empirical correction: the reported fps × position-diff is 10× the true
# linear speed for these recordings (likely due to fps being stored as
# 10× the actual sample rate). Apply once here so every plot, summary,
# and comparison consumes the corrected speed in m/s.
SPEED_SCALE = 0.1


def _velocity(xyz: np.ndarray, fps: float) -> np.ndarray:
    """Per-frame linear speed (m/s). Length is len(xyz)-1; aligned with
    the gap *between* consecutive frames (so element i is the speed
    going from frame i to frame i+1)."""
    if xyz is None or len(xyz) < 2:
        return np.zeros(0, dtype=np.float64)
    diffs = np.diff(xyz, axis=0)
    speeds = np.linalg.norm(diffs, axis=1) * fps * SPEED_SCALE
    return np.where(np.isfinite(speeds), speeds, 0.0)


def _velocity_per_axis(xyz: np.ndarray, fps: float) -> np.ndarray:
    """Per-frame, per-axis signed velocity (m/s). Shape (T-1, 3)."""
    if xyz is None or len(xyz) < 2:
        return np.zeros((0, 3), dtype=np.float64)
    v = np.diff(xyz, axis=0) * fps * SPEED_SCALE
    return np.where(np.isfinite(v), v, 0.0)


def _filter_low(speeds: np.ndarray, threshold: float) -> np.ndarray:
    """Drop samples below `threshold` (m/s). 0 disables filtering. Used
    to remove "robot/hand is stationary" frames so the comparison focuses
    on actual motion — without this, eva's parked left arm produces a
    huge zero-spike that drowns everything else in the density plot."""
    if threshold > 0 and speeds.size:
        return speeds[speeds >= threshold]
    return speeds


def _equalize_counts(
    per_emb: dict[str, np.ndarray], seed: int = 0
) -> dict[str, np.ndarray]:
    """Subsample each per-embodiment array to the smallest non-empty
    count so the comparison legend shows equal n across embodiments."""
    nonempty = {k: v for k, v in per_emb.items() if len(v) > 0}
    if len(nonempty) < 2:
        return per_emb
    min_n = min(len(v) for v in nonempty.values())
    rng = np.random.default_rng(seed)
    out = {}
    for emb, v in per_emb.items():
        if len(v) > min_n:
            out[emb] = rng.choice(v, min_n, replace=False)
        else:
            out[emb] = v
    return out


# ---------------------------------------------------------------------------
# Distribution plots
# ---------------------------------------------------------------------------
def _plot_distribution(
    emb: str,
    recs: list[dict],
    out_dir: str,
    *,
    bins: int,
    vmax: float | None,
    bin_width: float = 0.01,
) -> str:
    panels = [("left", "Left arm"), ("right", "Right arm"), ("combined", "Both arms")]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)

    all_speeds = [_velocity(r[a], r["fps"]) for r in recs for a in ARMS if a in r]
    if not all_speeds:
        plt.close(fig)
        return ""
    flat = np.concatenate(all_speeds)
    auto_max = float(np.quantile(flat, 0.999)) if flat.size else 1.0
    upper = vmax if vmax is not None else max(auto_max, 1e-3)
    # Fixed-width bins (default 0.1 m/s). Round upper up to a clean multiple
    # so the rightmost bin isn't a stub.
    upper = float(np.ceil(upper / bin_width) * bin_width)
    edges = np.arange(0.0, upper + bin_width / 2, bin_width)

    for ax, (which, title) in zip(axes, panels):
        per_rec, agg = [], []
        for r in recs:
            if which == "combined":
                speeds = np.concatenate(
                    [_velocity(r[a], r["fps"]) for a in ARMS if a in r] or [np.zeros(0)]
                )
            else:
                if which not in r:
                    continue
                speeds = _velocity(r[which], r["fps"])
            if speeds.size == 0:
                continue
            per_rec.append(speeds)
            agg.append(speeds)

        if agg:
            agg_speeds = np.concatenate(agg)
            ax.hist(
                agg_speeds,
                bins=edges,
                color="crimson",
                alpha=0.7,
                edgecolor="white",
                linewidth=0.5,
                label=f"aggregate (n={len(agg_speeds):,})",
            )
            ax.axvline(
                float(np.median(agg_speeds)),
                color="black",
                linestyle="--",
                linewidth=1.0,
                label=f"median={np.median(agg_speeds):.3f}",
            )
            ax.axvline(
                float(np.mean(agg_speeds)),
                color="darkgreen",
                linestyle=":",
                linewidth=1.0,
                label=f"mean={np.mean(agg_speeds):.3f}",
            )
        ax.set_title(f"{title}  ({len(per_rec)} recordings)")
        ax.set_xlabel(f"Linear speed (m/s)  ·  {bin_width} m/s bins")
        ax.set_ylabel("Frequency")
        ax.set_xlim(0.0, upper)
        if ax.get_legend_handles_labels()[1]:
            ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"EE linear velocity — {emb}  ({len(recs)} recordings)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path = os.path.join(out_dir, f"{emb}.png")
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def _gather_speeds(recs: list[dict], which: str, exclude_below: float) -> np.ndarray:
    """Concat all per-frame speeds for `which` arm across `recs`, dropping
    near-zero "stationary" samples below `exclude_below`."""
    agg = []
    for r in recs:
        if which == "combined":
            parts = [
                _filter_low(_velocity(r[a], r["fps"]), exclude_below)
                for a in ARMS
                if a in r
            ]
            s = np.concatenate(parts) if parts else np.zeros(0)
        else:
            if which not in r:
                continue
            s = _filter_low(_velocity(r[which], r["fps"]), exclude_below)
        if s.size:
            agg.append(s)
    return np.concatenate(agg) if agg else np.zeros(0)


def _plot_comparison(
    by_emb: dict[str, list[dict]],
    out_dir: str,
    *,
    bins: int,
    vmax: float | None,
    exclude_below: float = 0.02,
    equal_frames: bool = True,
    log_y: bool = True,
    bin_width: float = 0.01,
    seed: int = 0,
) -> str:
    """One PNG with all embodiments overlaid, three panels (left/right/both).
    Each curve is the aggregate distribution for that embodiment.

    `exclude_below` drops near-zero samples where the arm is essentially
    stationary (default 0.02 m/s — robot motors at rest, human hand
    settled). Without it eva's parked left arm produces a huge spike at
    0 that hides every other feature.

    `equal_frames` subsamples each embodiment to the smallest count so
    legend `n=` is balanced — apples-to-apples comparison.

    `log_y` keeps the small differences in the tail visible even when
    the mode is much higher than the rest of the distribution."""
    panels = [("left", "Left arm"), ("right", "Right arm"), ("combined", "Both arms")]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)

    # x-axis upper bound from the FILTERED data, otherwise the long tail
    # of pre-filter speeds (most of which we drop) gives a sparse plot.
    flat_all = []
    for recs in by_emb.values():
        for r in recs:
            for a in ARMS:
                if a in r:
                    flat_all.append(
                        _filter_low(_velocity(r[a], r["fps"]), exclude_below)
                    )
    if not flat_all:
        plt.close(fig)
        return ""
    flat = np.concatenate(flat_all)
    # 99.9th percentile (was 99th — was clipping the visible top of every
    # embodiment's tail). Override with --vmax for an explicit cap.
    auto_max = float(np.quantile(flat, 0.999)) if flat.size else 1.0
    upper = vmax if vmax is not None else max(auto_max, 1e-3)
    lower = max(exclude_below, 0.0)
    # Snap [lower, upper] to clean multiples of bin_width so the leftmost
    # and rightmost bins aren't stubs.
    lower = float(np.floor(lower / bin_width) * bin_width)
    upper = float(np.ceil(upper / bin_width) * bin_width)
    edges = np.arange(lower, upper + bin_width / 2, bin_width)

    for ax, (which, title) in zip(axes, panels):
        per_emb: dict[str, np.ndarray] = {}
        for emb in sorted(by_emb):
            s = _gather_speeds(by_emb[emb], which, exclude_below)
            if s.size:
                per_emb[emb] = s

        if equal_frames:
            per_emb = _equalize_counts(per_emb, seed=seed)

        # Use histtype="step" (line outlines, no fill) so multiple
        # embodiments can be overlaid without occluding each other.
        for emb, agg_speeds in per_emb.items():
            ax.hist(
                agg_speeds,
                bins=edges,
                histtype="step",
                color=_color_for(emb),
                linewidth=1.8,
                label=(
                    f"{emb} (n={len(agg_speeds):,}, "
                    f"med={np.median(agg_speeds):.2f}, "
                    f"p95={np.quantile(agg_speeds, 0.95):.2f})"
                ),
            )
        ax.set_title(title)
        ax.set_xlabel(f"Linear speed (m/s)  ·  {bin_width} m/s bins")
        ax.set_ylabel("Frequency")
        ax.set_xlim(lower, upper)
        if log_y:
            ax.set_yscale("log")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3, which="both")

    suptitle = "EE linear-velocity comparison across embodiments"
    if exclude_below > 0:
        suptitle += f"  (excl. < {exclude_below:.2g} m/s)"
    if equal_frames:
        suptitle += "  ·  equal-frame subsample"
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path = os.path.join(out_dir, "comparison.png")
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def _plot_comparison_ecdf(
    by_emb: dict[str, list[dict]],
    out_dir: str,
    *,
    vmax: float | None,
    exclude_below: float = 0.02,
    equal_frames: bool = True,
    seed: int = 0,
) -> str:
    """ECDF (cumulative distribution): y = fraction of frames with
    speed ≤ x. Reads cleanly even when one embodiment has a huge zero
    spike — the spike just becomes a steep initial rise but the rest of
    the curve stays visible. Median = where the curve crosses 0.5;
    p95 = where it crosses 0.95. This is usually the most informative
    single comparison plot for distributions like these."""
    panels = [("left", "Left arm"), ("right", "Right arm"), ("combined", "Both arms")]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)

    flat_all = []
    for recs in by_emb.values():
        for r in recs:
            for a in ARMS:
                if a in r:
                    flat_all.append(
                        _filter_low(_velocity(r[a], r["fps"]), exclude_below)
                    )
    if not flat_all:
        plt.close(fig)
        return ""
    flat = np.concatenate(flat_all)
    auto_max = float(np.quantile(flat, 0.99)) if flat.size else 1.0
    upper = vmax if vmax is not None else max(auto_max, 1e-3)

    for ax, (which, title) in zip(axes, panels):
        per_emb: dict[str, np.ndarray] = {}
        for emb in sorted(by_emb):
            s = _gather_speeds(by_emb[emb], which, exclude_below)
            if s.size:
                per_emb[emb] = s
        if equal_frames:
            per_emb = _equalize_counts(per_emb, seed=seed)

        for emb, speeds in per_emb.items():
            speeds_sorted = np.sort(speeds)
            ecdf = np.arange(1, len(speeds_sorted) + 1) / len(speeds_sorted)
            ax.plot(
                speeds_sorted,
                ecdf,
                color=_color_for(emb),
                linewidth=1.8,
                label=(
                    f"{emb} (n={len(speeds):,}, "
                    f"med={np.median(speeds):.2f}, "
                    f"p95={np.quantile(speeds, 0.95):.2f})"
                ),
            )
        ax.axhline(0.5, color="gray", linewidth=0.5, alpha=0.6)
        ax.axhline(0.95, color="gray", linewidth=0.5, alpha=0.6)
        ax.set_title(title)
        ax.set_xlabel("Linear speed (m/s)")
        ax.set_ylabel("Cumulative fraction")
        ax.set_xlim(max(exclude_below, 0.0), upper)
        ax.set_ylim(0.0, 1.0)
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.3)

    suptitle = "EE linear-velocity ECDF across embodiments"
    if exclude_below > 0:
        suptitle += f"  (excl. < {exclude_below:.2g} m/s)"
    if equal_frames:
        suptitle += "  ·  equal-frame subsample"
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path = os.path.join(out_dir, "comparison_ecdf.png")
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Per-axis comparison (eva vs aria, x/y/z separately for each arm)
# ---------------------------------------------------------------------------
def _plot_comparison_per_axis(
    by_emb: dict[str, list[dict]],
    out_dir: str,
    *,
    bins: int,
    vmax: float | None,
    exclude_below: float = 0.02,
    equal_frames: bool = True,
    log_y: bool = True,
    bin_width: float = 0.01,
    seed: int = 0,
) -> str:
    """Six panels: rows = arm (left, right), cols = axis (x, y, z).
    Each panel overlays one signed-velocity curve per embodiment so you
    can compare eva vs aria component-by-component.

    For SIGNED velocity, `exclude_below` filters by absolute value (drops
    samples where |v| < threshold so the |v|≈0 spike from parked frames
    doesn't dominate). `equal_frames` subsamples per embodiment to the
    smallest count, and `log_y` makes the tails visible."""
    axes_names = ("x", "y", "z")

    flat_all = []
    for recs in by_emb.values():
        for r in recs:
            for a in ARMS:
                if a in r:
                    v = _velocity_per_axis(r[a], r["fps"]).ravel()
                    if exclude_below > 0:
                        v = v[np.abs(v) >= exclude_below]
                    flat_all.append(v)
    if not flat_all:
        return ""
    flat = np.concatenate(flat_all)
    auto_max = float(np.quantile(np.abs(flat), 0.999)) if flat.size else 1.0
    upper = vmax if vmax is not None else max(auto_max, 1e-3)
    upper = float(np.ceil(upper / bin_width) * bin_width)
    edges = np.arange(-upper, upper + bin_width / 2, bin_width)

    fig, axes = plt.subplots(2, 3, figsize=(18, 8), sharey=False)
    for row, arm in enumerate(ARMS):
        for col, axis_name in enumerate(axes_names):
            ax = axes[row][col]
            per_emb: dict[str, np.ndarray] = {}
            for emb in sorted(by_emb):
                agg = []
                for r in by_emb[emb]:
                    if arm not in r:
                        continue
                    v = _velocity_per_axis(r[arm], r["fps"])
                    if not v.size:
                        continue
                    comp = v[:, col]
                    if exclude_below > 0:
                        comp = comp[np.abs(comp) >= exclude_below]
                    if comp.size:
                        agg.append(comp)
                if agg:
                    per_emb[emb] = np.concatenate(agg)

            if equal_frames:
                per_emb = _equalize_counts(per_emb, seed=seed)

            for emb, vals in per_emb.items():
                ax.hist(
                    vals,
                    bins=edges,
                    histtype="step",
                    color=_color_for(emb),
                    linewidth=1.8,
                    label=(f"{emb} (n={len(vals):,}, σ={vals.std():.2f})"),
                )
            ax.set_title(f"{arm} arm — {axis_name} velocity")
            ax.set_xlabel(f"Velocity (m/s)  ·  {bin_width} m/s bins")
            ax.set_ylabel("Frequency")
            ax.set_xlim(-upper, upper)
            if log_y:
                ax.set_yscale("log")
            ax.axvline(0.0, color="gray", linewidth=0.6, alpha=0.6)
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3, which="both")

    suptitle = "EE per-axis velocity comparison across embodiments"
    if exclude_below > 0:
        suptitle += f"  (excl. |v| < {exclude_below:.2g} m/s)"
    if equal_frames:
        suptitle += "  ·  equal-frame subsample"
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path = os.path.join(out_dir, "comparison_per_axis.png")
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Time-series plots
# ---------------------------------------------------------------------------
def _plot_timeseries_single(rec: dict, out_path: str, *, ymax: float | None):
    """One PNG for a single recording: speed vs time, both arms overlaid."""
    fig, ax = plt.subplots(figsize=(10, 4))
    fps = rec["fps"]
    plotted = False
    for arm, color in (("left", "#1f77b4"), ("right", "#d62728")):
        if arm not in rec:
            continue
        s = _velocity(rec[arm], fps)
        if s.size == 0:
            continue
        t = np.arange(s.size) / fps
        ax.plot(t, s, color=color, linewidth=0.8, alpha=0.85, label=f"{arm} arm")
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Linear speed (m/s)")
    ax.set_title(f"{rec['embodiment']} — {rec['name']}  (fps={fps:g})")
    if ymax is not None:
        ax.set_ylim(0.0, ymax)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _plot_timeseries_grid(
    emb: str, recs: list[dict], out_path: str, *, max_recs: int, ymax: float | None
) -> str:
    """One PNG with a grid of small subplots, one per recording (capped)."""
    sel = recs[:max_recs] if max_recs > 0 else recs
    if not sel:
        return ""
    n = len(sel)
    cols = 4 if n >= 8 else min(n, 4)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * 4.0, rows * 2.2), squeeze=False
    )

    for ax, rec in zip(axes.flat, sel):
        fps = rec["fps"]
        plotted = False
        for arm, color in (("left", "#1f77b4"), ("right", "#d62728")):
            if arm not in rec:
                continue
            s = _velocity(rec[arm], fps)
            if s.size == 0:
                continue
            t = np.arange(s.size) / fps
            ax.plot(t, s, color=color, linewidth=0.6, alpha=0.85, label=arm)
            plotted = True
        ax.set_title(rec["name"], fontsize=8)
        ax.set_xlabel("t (s)", fontsize=8)
        ax.set_ylabel("m/s", fontsize=8)
        ax.tick_params(labelsize=7)
        if ymax is not None:
            ax.set_ylim(0.0, ymax)
        ax.grid(True, alpha=0.25)
        if plotted:
            ax.legend(loc="upper right", fontsize=6)

    # Hide unused axes.
    for ax in axes.flat[len(sel) :]:
        ax.set_visible(False)

    title = f"{emb} — speed vs time  (showing {len(sel)}/{len(recs)} recordings)"
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Summary CSV
# ---------------------------------------------------------------------------
def _write_summary_csv(out_dir: str, by_emb: dict[str, list[dict]]):
    import csv

    path = os.path.join(out_dir, "summary.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "embodiment",
                "recording",
                "arm",
                "n_frames",
                "mean_speed",
                "median_speed",
                "p95_speed",
                "max_speed",
            ]
        )
        for emb, recs in sorted(by_emb.items()):
            for r in recs:
                for arm in ARMS:
                    if arm not in r:
                        continue
                    s = _velocity(r[arm], r["fps"])
                    if s.size == 0:
                        continue
                    w.writerow(
                        [
                            emb,
                            r["name"],
                            arm,
                            s.size,
                            f"{s.mean():.6f}",
                            f"{np.median(s):.6f}",
                            f"{np.quantile(s, 0.95):.6f}",
                            f"{s.max():.6f}",
                        ]
                    )
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--data-dir",
        default="/storage/project/r-dxu345-0/agao81/pick_place",
        help="Folder containing per-recording Zarr directories.",
    )
    p.add_argument(
        "--out-dir",
        default="logs/pick_place/velocity_distribution",
        help="Output root. Distribution PNGs go in <out>/distribution/, "
        "time-series in <out>/timeseries/<embodiment>/.",
    )
    p.add_argument("--bins", type=int, default=80, help="Histogram bin count.")
    p.add_argument(
        "--bin-width",
        type=float,
        default=0.01,
        help="Histogram bin width in m/s (default 0.01). Overrides "
        "--bins for the distribution/comparison/per-axis plots; "
        "those use fixed-width bins so x-axis ticks line up.",
    )
    p.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="Override speed-axis upper bound (m/s). "
        "Default: 99.9th percentile across all data.",
    )
    p.add_argument(
        "--ts-ymax",
        type=float,
        default=None,
        help="Y-axis cap for time-series plots (m/s). Default: same as --vmax.",
    )
    p.add_argument(
        "--max-timeseries",
        type=int,
        default=16,
        help="Max recordings to show in each embodiment's grid PNG. "
        "0 = all (grids can get very tall).",
    )
    p.add_argument(
        "--per-recording",
        action="store_true",
        help="Also write one time-series PNG per recording. "
        "Disabled by default (would emit 1k+ files).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process at most this many recordings (0 = all).",
    )
    p.add_argument(
        "--hashes",
        type=str,
        default=None,
        help="JSON list (or comma-separated) of recording hashes "
        "to keep. Embodiment is read from each recording's "
        "zarr.json, so mislabeled hashes are auto-bucketed.",
    )
    p.add_argument(
        "--balance",
        action="store_true",
        help="After hash filtering, subsample each embodiment to "
        "the smallest per-embodiment count for a balanced "
        "comparison.",
    )
    p.add_argument(
        "--exclude-below",
        type=float,
        default=0.02,
        help="Drop velocity samples below this threshold (m/s) "
        "in the comparison plots. Default 0.02 keeps the "
        "comparison focused on actual motion instead of "
        "the parked-arm zero-spike. Set to 0 to disable.",
    )
    p.add_argument(
        "--no-equal-frames",
        dest="equal_frames",
        action="store_false",
        default=True,
        help="Disable per-embodiment frame-count equalization in "
        "the comparison plots (default ON: subsamples each "
        "embodiment's velocity samples to the smallest n).",
    )
    p.add_argument(
        "--no-log-y",
        dest="log_y",
        action="store_false",
        default=True,
        help="Use linear y-axis on comparison density plots "
        "(default ON: log scale, so tail differences show).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for --balance and --equal-frames sampling.",
    )
    args = p.parse_args()

    keep: set[str] | None = None
    if args.hashes:
        s = args.hashes.strip()
        try:
            keep = set(json.loads(s))
        except json.JSONDecodeError:
            keep = {h.strip() for h in s.split(",") if h.strip()}
        print(f"[info] hash filter: keeping {len(keep)} hashes")

    if not os.path.isdir(args.data_dir):
        raise SystemExit(f"data-dir not found: {args.data_dir}")

    entries = sorted(
        d
        for d in os.listdir(args.data_dir)
        if os.path.isdir(os.path.join(args.data_dir, d))
    )
    if keep is not None:
        missing = keep - set(entries)
        if missing:
            print(f"[warn] {len(missing)} requested hash(es) not present on disk:")
            for m in sorted(missing):
                print(f"         {m}")
        entries = [e for e in entries if e in keep]
    if args.limit > 0:
        entries = entries[: args.limit]

    by_emb: dict[str, list[dict]] = defaultdict(list)
    skipped = 0
    for i, name in enumerate(entries, 1):
        path = os.path.join(args.data_dir, name)
        try:
            rec = _load_recording(path)
        except Exception as e:
            print(f"[skip] {name}: {e}")
            skipped += 1
            continue
        if rec is None:
            skipped += 1
            continue
        by_emb[rec["embodiment"]].append(rec)
        if i % 50 == 0 or i == len(entries):
            print(f"  loaded {i}/{len(entries)} (skipped {skipped})")

    if not by_emb:
        raise SystemExit("No recordings loaded — nothing to plot.")

    print("[info] per-embodiment recording counts after hash filter:")
    for emb in sorted(by_emb):
        print(f"         {emb}: {len(by_emb[emb])}")

    if args.balance:
        target = min(len(v) for v in by_emb.values())
        rng = random.Random(args.seed)
        balanced: dict[str, list[dict]] = {}
        for emb, recs in by_emb.items():
            if len(recs) > target:
                balanced[emb] = rng.sample(recs, target)
            else:
                balanced[emb] = recs
        by_emb = defaultdict(list, balanced)
        print(f"[info] --balance: subsampled to {target} recordings per embodiment")
        for emb in sorted(by_emb):
            kept = sorted(r["name"] for r in by_emb[emb])
            print(f"         {emb}: {kept}")

    dist_dir = os.path.join(args.out_dir, "distribution")
    ts_dir = os.path.join(args.out_dir, "timeseries")
    os.makedirs(dist_dir, exist_ok=True)
    os.makedirs(ts_dir, exist_ok=True)

    # Distribution: per-embodiment + comparison.
    for emb in sorted(by_emb):
        out = _plot_distribution(
            emb,
            by_emb[emb],
            dist_dir,
            bins=args.bins,
            vmax=args.vmax,
            bin_width=args.bin_width,
        )
        if out:
            print(f"[ok] dist  {emb}: {len(by_emb[emb])} recs -> {out}")
    cmp_out = _plot_comparison(
        by_emb,
        dist_dir,
        bins=args.bins,
        vmax=args.vmax,
        exclude_below=args.exclude_below,
        equal_frames=args.equal_frames,
        log_y=args.log_y,
        bin_width=args.bin_width,
        seed=args.seed,
    )
    print(f"[ok] dist  comparison -> {cmp_out}")
    cmp_ecdf_out = _plot_comparison_ecdf(
        by_emb,
        dist_dir,
        vmax=args.vmax,
        exclude_below=args.exclude_below,
        equal_frames=args.equal_frames,
        seed=args.seed,
    )
    if cmp_ecdf_out:
        print(f"[ok] dist  comparison_ecdf -> {cmp_ecdf_out}")
    cmp_axis_out = _plot_comparison_per_axis(
        by_emb,
        dist_dir,
        bins=args.bins,
        vmax=args.vmax,
        exclude_below=args.exclude_below,
        equal_frames=args.equal_frames,
        log_y=args.log_y,
        bin_width=args.bin_width,
        seed=args.seed,
    )
    if cmp_axis_out:
        print(f"[ok] dist  comparison_per_axis -> {cmp_axis_out}")

    # Time-series: per-embodiment grid (+ optional per-recording).
    ts_ymax = args.ts_ymax if args.ts_ymax is not None else args.vmax
    for emb in sorted(by_emb):
        emb_dir = os.path.join(ts_dir, emb)
        grid_path = _plot_timeseries_grid(
            emb,
            by_emb[emb],
            os.path.join(emb_dir, "grid.png"),
            max_recs=args.max_timeseries,
            ymax=ts_ymax,
        )
        if grid_path:
            print(f"[ok] ts    {emb}: grid -> {grid_path}")
        if args.per_recording:
            for r in by_emb[emb]:
                _plot_timeseries_single(
                    r,
                    os.path.join(emb_dir, f"{r['name']}.png"),
                    ymax=ts_ymax,
                )
            print(
                f"[ok] ts    {emb}: {len(by_emb[emb])} per-recording PNGs in {emb_dir}/"
            )

    csv_path = _write_summary_csv(args.out_dir, by_emb)
    print(f"[ok] per-recording stats -> {csv_path}")


if __name__ == "__main__":
    main()
