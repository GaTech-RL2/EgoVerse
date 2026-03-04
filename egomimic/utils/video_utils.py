import logging
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def save_preview_mp4(
    images_tchw: np.ndarray,
    output_path: Path,
    fps: int,
    *,
    half_res: bool = True,
    crf: int = 23,
    preset: str = "fast",
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    arr = np.asarray(images_tchw)
    if arr.ndim != 4 or arr.shape[0] == 0:
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
        f"{outW}x{outH}",
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

        for t in range(T):
            frame_chw = arr[t]

            if frame_chw.dtype != np.uint8:
                frame_chw = frame_chw.astype(np.uint8, copy=False)

            if frame_chw.shape[0] == 1:
                frame_chw = np.repeat(frame_chw, 3, axis=0)
            elif frame_chw.shape[0] == 4:
                frame_chw = frame_chw[:3]

            frame_hwc = np.transpose(frame_chw, (1, 2, 0))  # view

            if frame_hwc.shape[0] != outH or frame_hwc.shape[1] != outW:
                frame_hwc = cv2.resize(
                    frame_hwc, (outW, outH), interpolation=cv2.INTER_AREA
                )

            proc.stdin.write(frame_hwc.tobytes())

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
