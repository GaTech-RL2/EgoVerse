"""Tiny AV1 MP4s for tests, encoded with PyAV (libaom)."""

import numpy as np


def write_av1(path, n: int, faststart: bool = True, gop: int = 2) -> None:
    """n distinct 64x48 frames (a white bar growing by 4 px per frame), keyframe every `gop`."""
    import av

    out = av.open(
        str(path), "w", options={"movflags": "+faststart"} if faststart else {}
    )
    s = out.add_stream("libaom-av1", rate=30)
    s.width, s.height, s.pix_fmt = 64, 48, "yuv420p"
    s.options = {"cpu-used": "8", "crf": "30"}
    s.codec_context.gop_size = gop
    for i in range(n):
        img = np.zeros((48, 64, 3), np.uint8)
        img[:, : min(64, 4 * (i + 1))] = 255
        for p in s.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
            out.mux(p)
    for p in s.encode():
        out.mux(p)
    out.close()
