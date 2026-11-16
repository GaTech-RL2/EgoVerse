"""KEPT / DROPPED video of the Aria Gen 2 frame filter for one recording.

Renders the whole recording's RGB at reduced size with a banner on every
frame: green "KEPT", orange "DROPPED: <reason>" for an in-demo error, or red
for a demo boundary (reset / opening calibration). A timeline along the bottom
shows every frame's outcome, white ticks mark demo starts and ends, and a
cursor marks the current frame. The per-frame decisions come from the same
code the converter uses (aria_utils.gen2_frame_signals +
gen2_filter.classify_frames), so the video matches the zarr output.

Writes <out-dir>/<name>_filter.mp4 and <name>_summary.txt (totals, dropped
time per reason, and per-demo spans with their episode hashes).

Usage:
  python egomimic/scripts/data_visualization/aria_gen2_filter_viz.py \\
      --vrs /data/aria_gen2/1794800772000.vrs --out-dir viz/aria_gen2 \\
      [--source mps] [--mps-dir /data/aria_gen2/mps_1794800772000_vrs]
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from projectaria_tools.core import data_provider, mps
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from projectaria_tools.core.stream_id import StreamId

from egomimic.scripts.aria_process.aria_to_zarr import gen2_demo_hashes
from egomimic.scripts.aria_process.aria_utils import (
    AriaVRSExtractor,
    gen2_ee_frame_ok,
    gen2_frame_signals,
    gen2_utc_timestamps_ns,
    is_aria_gen2,
    load_gen2_on_device_gaze,
    load_gen2_on_device_mp,
)
from egomimic.scripts.aria_process.gen2_filter import (
    BOUNDARY_REASONS,
    REASONS,
    Gen2FilterConfig,
    classify_frames,
    demo_segments,
    first_reason,
    runs,
)
from egomimic.utils.aws.aws_sql import timestamp_ms_to_episode_hash
from egomimic.utils.video_utils import write_mp4_frames

RGB_STREAM = StreamId("214-1")
KEPT_COLOR = (40, 170, 80)
BOUNDARY_COLOR = (190, 30, 30)
ERROR_COLOR = (200, 110, 20)
# Timeline colour per REASONS entry.
REASON_COLORS = {
    "not_tracking": (140, 60, 200),
    "reset": (220, 40, 40),
    "calibration": (230, 140, 20),
    "gaze_dropout": (150, 110, 60),
    "head_sweep": (230, 200, 30),
    "hand_at_headset": (200, 60, 150),
    "no_hands": (120, 120, 120),
    "left_hand_not_tracked": (90, 150, 230),
    "right_hand_not_tracked": (60, 200, 220),
}
SHORT_PIECE = "short piece between demos"


def _load(vrs: Path, source: str, mps_dir: Path, arm: str, cfg: Gen2FilterConfig):
    provider = data_provider.create_vrs_data_provider(str(vrs))
    if not is_aria_gen2(provider):
        raise ValueError(f"{vrs} is not an Aria Gen 2 recording")
    rgb = np.asarray(provider.get_timestamps_ns(RGB_STREAM, TimeDomain.DEVICE_TIME))
    has_mps = (mps_dir / "hand_tracking" / "hand_tracking_results.csv").exists()
    if source == "mps" or (source == "auto" and has_mps):
        hands = mps.hand_tracking.read_hand_tracking_results(
            str(mps_dir / "hand_tracking" / "hand_tracking_results.csv")
        )
        poses = mps.read_closed_loop_trajectory(
            str(mps_dir / "slam" / "closed_loop_trajectory.csv")
        )
        gaze = load_gen2_on_device_gaze(provider)
        tracking_ok, used = np.ones(len(rgb), bool), "mps"
    else:
        hands, poses, gaze, tracking_ok = load_gen2_on_device_mp(provider, rgb)
        used = "on_device"
    signals = gen2_frame_signals(rgb, poses, hands, gaze, tracking_ok, cfg)
    # Frames the converter drops last: no keypoint-based EE frame.
    ee_ok = gen2_ee_frame_ok(
        AriaVRSExtractor.get_hand_keypoints(
            world_device_T=poses,
            stream_timestamps_ns={"rgb": rgb},
            hand_tracking_results=hands,
            arm=arm,
        ),
        arm,
    )
    return provider, rgb, classify_frames(signals, arm, cfg), ee_ok, used


def _demo_hashes(provider, rgb, kept, demos, parent_hash):
    """Episode hashes exactly as the converter assigns them (see
    aria_to_zarr.gen2_demo_hashes): UTC of each demo's first kept frame."""
    utc = gen2_utc_timestamps_ns(provider, rgb)
    firsts = [s + int(np.argmax(kept[s:e])) for s, e in demos if kept[s:e].any()]
    hashes = iter(gen2_demo_hashes([int(utc[f]) for f in firsts], parent_hash))
    return [next(hashes) if kept[s:e].any() else "-" for s, e in demos]


def _summary(name, used, rgb, masks, label, demos, removed, hashes, cfg):
    t_s = (rgb - rgb[0]) / 1e9
    fps = (len(rgb) - 1) / max(t_s[-1], 1e-6)
    kept = (label < 0) & (_demo_index(len(rgb), demos) >= 0)
    lines = [
        f"{name} [{used} poses+hands]: {len(rgb)} RGB frames ({t_s[-1]:.1f} s)",
        f"  kept {kept.sum()} ({kept.sum() / fps:.1f} s), "
        f"dropped {(~kept).sum()} ({(~kept).sum() / fps:.1f} s)",
    ]
    for k, reason in enumerate(REASONS):
        n = int((label == k).sum())
        if n:
            lines.append(f"  dropped for {reason}: {n} frames ({n / fps:.1f} s)")
    lines.append(
        f"  demos (split at {' / '.join(BOUNDARY_REASONS)}, "
        f"min {cfg.min_demo_s:g} s): {len(demos)}"
    )
    for k, (s, e) in enumerate(demos):
        err = int(masks["error"][s:e].sum())
        lines.append(
            f"    demo {k + 1}/{len(demos)}  {hashes[k]}  "
            f"{t_s[s]:7.1f}-{t_s[e - 1]:7.1f} s"
            f"  ({t_s[e - 1] - t_s[s]:5.1f} s)  {e - s} frames, {err} in-demo "
            f"error frames ({err / fps:.1f} s), {e - s - err} kept"
        )
    for s, e in removed:
        lines.append(
            f"    removed {SHORT_PIECE} {t_s[s]:.1f}-{t_s[e - 1]:.1f} s "
            f"({t_s[e - 1] - t_s[s]:.1f} s < {cfg.min_demo_s:g} s)"
        )
    lines.append("  dropped spans (>= 0.5 s):")
    for s, e in runs(~kept):
        if t_s[e - 1] - t_s[s] < 0.5:
            continue
        seg = label[s:e]
        counts = np.bincount(seg[seg >= 0], minlength=len(REASONS))
        why = ", ".join(
            f"{REASONS[k]} {counts[k] / fps:.1f}s"
            for k in np.argsort(-counts)
            if counts[k]
        )
        lines.append(
            f"    {t_s[s]:7.1f}-{t_s[e - 1]:7.1f} s  ({t_s[e - 1] - t_s[s]:5.1f} s)  "
            f"{why or SHORT_PIECE}"
        )
    return lines


def _demo_index(T, demos):
    demo = np.full(T, -1)
    for k, (s, e) in enumerate(demos):
        demo[s:e] = k
    return demo


def _timeline(label, demo, demos, width, height):
    T = len(label)
    strip = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(strip)
    for x in range(width):
        a = int(x * T / width)
        b = max(int((x + 1) * T / width), a + 1)
        seg, seg_demo = label[a:b], demo[a:b]
        if ((seg < 0) & (seg_demo >= 0)).mean() >= 0.5:
            col = KEPT_COLOR
        elif (seg >= 0).any():
            col = REASON_COLORS[REASONS[np.bincount(seg[seg >= 0]).argmax()]]
        else:
            col = BOUNDARY_COLOR  # short piece between demos
        draw.line([(x, 0), (x, height)], fill=col)
    for s, e in demos:
        for f in (s, e - 1):
            x = int(f * width / T)
            draw.line([(x, 0), (x, height)], fill=(255, 255, 255), width=2)
    return strip


def _font(size):
    for name in ("DejaVuSans.ttf", "/System/Library/Fonts/Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def render(
    provider, rgb, label, demos, out_mp4: Path, fps: int, width: int, show_demos: bool
):
    """Stream the annotated video to out_mp4."""
    T = len(rgb)
    height = (width * 3 // 4) // 2 * 2  # Gen 2 RGB is 4:3
    banner, bar = 34, 26
    demo = _demo_index(T, demos)
    strip = _timeline(label, demo, demos, width, bar - 6)
    font, small = _font(17), _font(12)
    t_s = (rgb - rgb[0]) / 1e9
    step = max(1, round((T - 1) / max(t_s[-1], 1e-6) / fps))

    def frames():
        for i in range(0, T, step):
            img = provider.get_image_data_by_time_ns(
                RGB_STREAM,
                int(rgb[i]),
                TimeDomain.DEVICE_TIME,
                TimeQueryOptions.CLOSEST,
            )[0].to_numpy_array()
            frame = Image.new("RGB", (width, height + banner + bar), (20, 20, 22))
            frame.paste(Image.fromarray(img).resize((width, height)), (0, banner))
            draw = ImageDraw.Draw(frame)
            reason = REASONS[label[i]] if label[i] >= 0 else SHORT_PIECE
            tag = f"DEMO {demo[i] + 1}/{len(demos)}" if show_demos else ""
            if label[i] < 0 and demo[i] >= 0:
                text, fill = f"{tag} - KEPT" if tag else "KEPT", KEPT_COLOR
            elif demo[i] < 0:
                prefix = "BETWEEN DEMOS - " if show_demos else "DROPPED - "
                text, fill = prefix + reason, BOUNDARY_COLOR
            else:
                text = f"{tag} - DROPPED: {reason}" if tag else f"DROPPED - {reason}"
                fill = ERROR_COLOR
            draw.rectangle([0, 0, width, banner], fill=fill)
            draw.text((10, 7), text, fill=(255, 255, 255), font=font)
            draw.rectangle(
                [width - 66, banner + height - 20, width, banner + height], fill=0
            )
            draw.text(
                (width - 60, banner + height - 17),
                f"{t_s[i]:6.1f} s",
                fill=(255, 255, 255),
                font=small,
            )
            frame.paste(strip, (0, banner + height + 3))
            x = int(i * width / T)
            draw.rectangle(
                [x - 1, banner + height, x + 1, banner + height + bar],
                fill=(255, 255, 255),
            )
            yield np.asarray(frame)

    write_mp4_frames(
        frames(), out_mp4, width, height + banner + bar, fps, crf=26, preset="veryfast"
    )


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--vrs", type=Path, required=True, help="Aria Gen 2 .vrs file")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument(
        "--name", default=None, help="Output file prefix (default: VRS stem)"
    )
    p.add_argument("--source", choices=["auto", "mps", "on_device"], default="auto")
    p.add_argument(
        "--mps-dir",
        type=Path,
        default=None,
        help="MPS output folder (default: mps_<stem>_vrs next to the VRS)",
    )
    p.add_argument("--arm", choices=["left", "right", "both"], default="both")
    p.add_argument(
        "--no-demos",
        action="store_true",
        help="Do not mark demos (single-episode view)",
    )
    p.add_argument("--filter-config", type=Path, default=None)
    p.add_argument("--min-demo-s", type=float, default=None)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--width", type=int, default=480)
    args = p.parse_args()

    cfg = (
        Gen2FilterConfig.from_yaml(args.filter_config)
        if args.filter_config
        else Gen2FilterConfig()
    )
    if args.min_demo_s is not None:
        cfg = dataclasses.replace(cfg, min_demo_s=args.min_demo_s)
    name = args.name or args.vrs.stem
    mps_dir = args.mps_dir or args.vrs.parent / f"mps_{args.vrs.stem}_vrs"

    provider, rgb, masks, ee_ok, used = _load(
        args.vrs, args.source, mps_dir, args.arm, cfg
    )
    label = first_reason(masks)
    if args.no_demos:
        demos, removed = [(s, e) for s, e in runs(~masks["boundary"])], []
    else:
        _, demos, removed = demo_segments(
            rgb, masks["boundary"], cfg.min_demo_s, usable=~masks["error"]
        )
    parent_hash = (
        timestamp_ms_to_episode_hash(int(args.vrs.stem))
        if args.vrs.stem.isdigit()
        else ""
    )
    # Exactly the frames the converter writes (incl. its EE-frame drop).
    kept = (label < 0) & (_demo_index(len(rgb), demos) >= 0) & ee_ok
    hashes = _demo_hashes(provider, rgb, kept, demos, parent_hash)
    lines = _summary(name, used, rgb, masks, label, demos, removed, hashes, cfg)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"{name}_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    render(
        provider,
        rgb,
        label,
        demos,
        args.out_dir / f"{name}_filter.mp4",
        args.fps,
        args.width,
        not args.no_demos,
    )
    print("wrote", args.out_dir / f"{name}_filter.mp4")


if __name__ == "__main__":
    main()
