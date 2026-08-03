"""H.264 image storage for EgoVerse zarr episodes.

WHY: per-frame JPEG codes every frame independently, so it cannot exploit the
temporal redundancy of 30fps video. Measured on real fold episodes:

    JPEG (q75, as shipped)   44.8 KB/frame   8.33 ms/frame decode
    h264 crf15               20.2 KB/frame   1.68 ms/frame decode
                             2.2x smaller    5.0x faster

At EQUAL bytes h264 also beats JPEG on PSNR (+2.4 to +2.9 dB in the 4-8 KB
range), so this is not a quality-for-size trade -- it is strictly better coding.

LAYOUT: one mp4 per GROUP of ``frames_per_chunk`` frames, stored as the elements
of a VariableLengthBytes array. NOT one blob per episode: a span read would then
have to pull the whole episode (~68 MB) to decode any window, and we read ~13
spans per episode. Frame f lives in chunk ``f // frames_per_chunk`` at offset
``f % frames_per_chunk``.

The reader learns everything it needs from ``_features[key]``::

    {"dtype": "h264", "shape": [H, W, 3], "video": {
        "codec", "crf", "gop", "pix_fmt", "fps", "frames_per_chunk",
        "total_frames", "n_chunks"}}

``dtype`` stays "jpeg" for JPEG episodes, so existing data and readers are
untouched -- dispatch on it.
"""
from __future__ import annotations

import io
import os

import numpy as np

DEFAULT_CODEC = "h264"
DEFAULT_CRF = 15          # decode speed is flat across CRF, so don't over-compress
DEFAULT_GOP = 30          # 1 s at 30 fps; bounds worst-case seek within a chunk
DEFAULT_PIX_FMT = "yuv420p"
DEFAULT_FRAMES_PER_CHUNK = 300   # 10 s; a multiple of GOP so chunks start on a keyframe


def codec_settings_from_env() -> dict:
    """Read encode settings from env so a run is reproducible from its launch.

    These names are forwarded to Ray workers by run_conversion.py; without that
    forwarding a worker silently falls back to these defaults.
    """
    fpc = int(os.environ.get("EGOVERSE_VIDEO_FRAMES_PER_CHUNK", DEFAULT_FRAMES_PER_CHUNK))
    gop = int(os.environ.get("EGOVERSE_VIDEO_GOP", DEFAULT_GOP))
    if fpc % gop:
        raise ValueError(
            f"frames_per_chunk ({fpc}) must be a multiple of gop ({gop}) so every "
            f"chunk starts on a keyframe -- otherwise decoding a chunk requires "
            f"the previous one."
        )
    return {
        "codec": os.environ.get("EGOVERSE_IMAGE_CODEC", DEFAULT_CODEC),
        "crf": int(os.environ.get("EGOVERSE_VIDEO_CRF", DEFAULT_CRF)),
        "gop": gop,
        "pix_fmt": os.environ.get("EGOVERSE_VIDEO_PIXFMT", DEFAULT_PIX_FMT),
        "frames_per_chunk": fpc,
    }


def image_codec_enabled() -> bool:
    """True when episodes should be written as video instead of per-frame JPEG."""
    return os.environ.get("EGOVERSE_IMAGE_CODEC", "jpeg").lower() in ("h264", "avc", "video")


def encode_chunk(frames_hwc_rgb: np.ndarray, settings: dict, fps: int = 30) -> bytes:
    """Encode a (T, H, W, 3) uint8 RGB block to a self-contained mp4.

    Self-contained matters: the first frame of every chunk must be a keyframe so
    a reader can decode the chunk without touching its neighbours.
    """
    import av

    if frames_hwc_rgb.dtype != np.uint8:
        raise ValueError(f"expected uint8 frames, got {frames_hwc_rgb.dtype}")
    if frames_hwc_rgb.ndim != 4 or frames_hwc_rgb.shape[-1] != 3:
        raise ValueError(f"expected (T,H,W,3), got {frames_hwc_rgb.shape}")

    T, H, W, _ = frames_hwc_rgb.shape
    buf = io.BytesIO()
    enc = "libx264rgb" if settings["pix_fmt"] == "rgb24" else "libx264"
    with av.open(buf, mode="w", format="mp4") as container:
        stream = container.add_stream(enc, rate=fps)
        stream.width, stream.height = W, H
        stream.pix_fmt = settings["pix_fmt"]
        stream.options = {
            "crf": str(settings["crf"]),
            "g": str(settings["gop"]),
            # keyint_min=g stops x264 inserting EXTRA keyframes on scene cuts,
            # which would desync the frame->chunk arithmetic the reader relies on.
            "keyint_min": str(settings["gop"]),
            "sc_threshold": "0",
        }
        for i in range(T):
            container.mux(stream.encode(
                av.VideoFrame.from_ndarray(np.ascontiguousarray(frames_hwc_rgb[i]),
                                           format="rgb24")))
        container.mux(stream.encode())
    return buf.getvalue()


def decode_chunk(blob: bytes, n_expected: int | None = None) -> np.ndarray:
    """Decode an mp4 chunk back to (T, H, W, 3) uint8 RGB. Reader-side helper.

    Kept here so writer and reader share one definition of the container format.
    """
    import av

    # zarr hands back VLenBytes elements wrapped in (possibly nested) 0-d
    # object arrays; bytes() on those yields garbage, not the payload.
    while isinstance(blob, np.ndarray):
        blob = blob.item() if blob.shape == () else blob[0]
    if isinstance(blob, memoryview):
        blob = blob.tobytes()

    out = []
    with av.open(io.BytesIO(bytes(blob))) as container:
        for frame in container.decode(video=0):
            out.append(frame.to_ndarray(format="rgb24"))
    arr = np.stack(out, axis=0) if out else np.zeros((0, 0, 0, 3), np.uint8)
    if n_expected is not None and len(arr) != n_expected:
        raise ValueError(f"chunk decoded {len(arr)} frames, expected {n_expected}")
    return arr


def build_feature_entry(img_shape, settings: dict, total_frames: int, fps: int) -> dict:
    fpc = settings["frames_per_chunk"]
    return {
        "dtype": settings["codec"],          # "h264" -- readers dispatch on this
        "shape": list(img_shape),
        "names": ["height", "width", "channel"],
        "video": {
            "codec": settings["codec"],
            "crf": settings["crf"],
            "gop": settings["gop"],
            "pix_fmt": settings["pix_fmt"],
            "fps": fps,
            "frames_per_chunk": fpc,
            "total_frames": int(total_frames),
            "n_chunks": int((total_frames + fpc - 1) // fpc),
        },
    }
