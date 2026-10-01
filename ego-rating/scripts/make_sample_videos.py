"""Generate browser-playable H.264 sample clips for every video referenced in
config.yaml, sized/length-matched to each span's [start, end] window.

These are throwaway demo clips (videos/ is gitignored) — in real use you drop
your own footage in there. Run from the ego-rating/ root:

    python scripts/make_sample_videos.py

Uses the ffmpeg bundled with imageio-ffmpeg, so no system ffmpeg is needed.
"""

import math
import re
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config.yaml"
VIDEOS = ROOT / "videos"
FPS = 24
W, H = 640, 360

# Distinct background tint per operator so clips are visually distinguishable.
PALETTE = [
    (32, 58, 102),  # blue
    (28, 84, 64),  # green
    (74, 40, 96),  # purple
    (96, 64, 28),  # amber
    (96, 36, 48),  # rose
]


def operator_color(operator: str, operators: list[str]) -> tuple[int, int, int]:
    return PALETTE[operators.index(operator) % len(PALETTE)]


def take_label(video_path: str) -> str:
    m = re.search(r"(take\d+)", video_path)
    return m.group(1) if m else Path(video_path).stem


def draw_frame(t: float, dur: float, header: str, bg: tuple) -> np.ndarray:
    """One RGB frame at time t (seconds)."""
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[:] = bg

    # Header: operator · scene · take
    cv2.putText(
        img,
        header,
        (24, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (235, 238, 245),
        2,
        cv2.LINE_AA,
    )

    # Big running timer (this is what makes the fragment seek/loop visible).
    cv2.putText(
        img,
        f"t = {t:5.1f}s",
        (150, 205),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.8,
        (255, 255, 255),
        4,
        cv2.LINE_AA,
    )

    # Moving marker that sweeps left->right with time, plus a progress bar.
    x = int(40 + (W - 80) * (t / dur))
    cv2.circle(img, (x, 270), 14, (91, 140, 255), -1, cv2.LINE_AA)
    cv2.rectangle(img, (40, 320), (W - 40, 332), (70, 78, 96), -1)
    cv2.rectangle(img, (40, 320), (x, 332), (124, 91, 255), -1)

    return img


def main() -> None:
    cfg = yaml.safe_load(CONFIG.read_text())
    spans = cfg.get("spans") or []
    operators = sorted({s["operator"] for s in spans})

    # Collapse to one clip per file; length must cover the largest end it's used for.
    by_file: dict[str, dict] = {}
    for s in spans:
        v = s["video"]
        info = by_file.setdefault(
            v, {"end": 0.0, "scene": s["scene"], "operator": s["operator"]}
        )
        info["end"] = max(info["end"], float(s["end"]))

    VIDEOS.mkdir(parents=True, exist_ok=True)
    for video_path, info in by_file.items():
        out = ROOT / video_path
        out.parent.mkdir(parents=True, exist_ok=True)
        dur = max(8.0, math.ceil(info["end"]) + 2)  # buffer past the span end
        header = f"{info['operator']} | {info['scene']} | {take_label(video_path)}"
        bg = operator_color(info["operator"], operators)

        writer = imageio.get_writer(
            str(out),
            fps=FPS,
            codec="libx264",
            quality=8,
            pixelformat="yuv420p",  # required for broad browser playback
            macro_block_size=8,  # 640x360 divisible by 8 -> no resize
            ffmpeg_params=["-movflags", "+faststart"],  # web streaming
        )
        n = int(dur * FPS)
        for i in range(n):
            writer.append_data(draw_frame(i / FPS, dur, header, bg))
        writer.close()
        print(f"  wrote {video_path}  ({dur:.0f}s, {n} frames)")

    print(f"\nGenerated {len(by_file)} clips in {VIDEOS}")


if __name__ == "__main__":
    main()
