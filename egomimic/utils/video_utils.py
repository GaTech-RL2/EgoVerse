import logging
import shutil
import subprocess
from collections.abc import Iterable
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def write_mp4_frames(
    frames_hwc: Iterable[np.ndarray],
    output_path: Path,
    width: int,
    height: int,
    fps: int,
    *,
    crf: int = 23,
    preset: str = "fast",
) -> None:
    """Stream uint8 HxWx3 RGB frames into an H.264 mp4 via ffmpeg.

    Frames whose size differs from (height, width) are resized; width and
    height must be even.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if width <= 0 or height <= 0 or width % 2 or height % 2:
        raise ValueError(f"Invalid output size: {width}x{height}")

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found in PATH")

    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        str(crf),
        "-preset",
        preset,
        str(output_path),
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    try:
        assert proc.stdin is not None
        assert proc.stderr is not None

        for frame_hwc in frames_hwc:
            if frame_hwc.dtype != np.uint8:
                frame_hwc = frame_hwc.astype(np.uint8, copy=False)
            if frame_hwc.shape[0] != height or frame_hwc.shape[1] != width:
                frame_hwc = cv2.resize(
                    frame_hwc, (width, height), interpolation=cv2.INTER_AREA
                )
            proc.stdin.write(np.ascontiguousarray(frame_hwc).tobytes())

        proc.stdin.close()

        stderr = proc.stderr.read()
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(
                f"ffmpeg failed (rc={rc}): {stderr.decode(errors='replace')}"
            )
    finally:
        try:
            if proc.poll() is None:
                proc.kill()
        except Exception:
            pass


def save_preview_mp4(
    images_tchw: np.ndarray,
    output_path: Path,
    fps: int,
    *,
    half_res: bool = True,
    crf: int = 23,
    preset: str = "fast",
) -> None:
    arr = np.asarray(images_tchw)
    if arr.ndim != 4 or arr.shape[0] == 0:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        return

    T, C, H, W = arr.shape

    if half_res:
        outW, outH = W // 2, H // 2
    else:
        outW, outH = W, H
    outW -= outW % 2
    outH -= outH % 2
    if outW <= 0 or outH <= 0:
        raise ValueError(f"Invalid output size: {outW}x{outH}")

    def frames():
        for t in range(T):
            frame_chw = arr[t]
            if frame_chw.shape[0] == 1:
                frame_chw = np.repeat(frame_chw, 3, axis=0)
            elif frame_chw.shape[0] == 4:
                frame_chw = frame_chw[:3]
            yield np.transpose(frame_chw, (1, 2, 0))  # view

    write_mp4_frames(frames(), output_path, outW, outH, fps, crf=crf, preset=preset)


def resize_video_thwc(video: np.ndarray) -> np.ndarray:
    out = np.empty((video.shape[0], 480, 640, video.shape[3]), dtype=video.dtype)
    for t in range(video.shape[0]):
        out[t] = cv2.resize(video[t], (640, 480), interpolation=cv2.INTER_AREA)
    return out
