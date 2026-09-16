"""Shared helpers for the zarr episode viz scripts (abc_process/viz_eva_episode.py,
mecka_process/viz_mecka_episode.py)."""

import io
import os

import numpy as np
from PIL import Image, ImageDraw

AXIS_COLORS = {"x": (255, 0, 0), "y": (0, 255, 0), "z": (0, 0, 255)}  # RGB


def episode_path(episode: str, folder: str) -> str:
    """A full .zarr path, or a bare episode hash resolved under ``folder``."""
    return (
        episode
        if episode.endswith(".zarr")
        else os.path.join(folder, f"{episode}.zarr")
    )


def decode(elem):
    """An image stored in the object array -> HxWx3 uint8.

    Frames are JPEG bytes wrapped in a 0-d object ndarray (sometimes nested), so
    unwrap to the raw bytes and decode with PIL (imageio can't sniff a bare BytesIO).
    """
    while isinstance(elem, np.ndarray) and elem.dtype == object and elem.ndim == 0:
        elem = elem.item()
    if isinstance(elem, np.ndarray) and elem.ndim >= 2:
        return elem[..., :3].astype(np.uint8)
    raw = bytes(elem) if not isinstance(elem, (bytes, bytearray)) else elem
    return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.uint8)


def project(K, pts_cam: np.ndarray) -> np.ndarray:
    """(N,3) cam-frame points -> (N,2) pixels; NaN where behind the camera."""
    px = np.full((len(pts_cam), 2), np.nan)
    front = pts_cam[:, 2] > 1e-6
    if front.any():
        p = np.concatenate([pts_cam[front], np.ones((front.sum(), 1))], axis=1)
        uv = K @ p.T
        px[front] = (uv[:2] / uv[2]).T
    return px


def draw_triad(img, K, M_cam_pose, length, label, thickness=2, font_scale=0.45):
    """Draw an XYZ triad for a cam<-frame pose matrix onto img (RGB) in place."""
    import cv2

    o = M_cam_pose[:3, 3]
    pts = np.stack([o] + [o + M_cam_pose[:3, k] * length for k in range(3)])
    px = project(K, pts)
    if np.isnan(px[0]).any():
        return
    oi = tuple(np.round(px[0]).astype(int))
    for k, ax in enumerate("xyz"):
        if np.isnan(px[k + 1]).any():
            continue
        ei = tuple(np.round(px[k + 1]).astype(int))
        cv2.line(img, oi, ei, AXIS_COLORS[ax], thickness, cv2.LINE_AA)
        cv2.putText(
            img,
            ax,
            (ei[0] + 3, ei[1] - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            AXIS_COLORS[ax],
            1,
            cv2.LINE_AA,
        )
    cv2.circle(img, oi, 3, (255, 255, 255), -1)
    cv2.putText(
        img,
        label,
        (oi[0] + 5, oi[1] + 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale + 0.05,
        (255, 255, 0),
        1,
        cv2.LINE_AA,
    )


def write_episode_video(out, fps, nf, render, header, stills=()):
    """Write ``render(i)`` for i < nf to ``out`` with ``header`` burned in; also
    save the frames in ``stills`` as PNGs next to it."""
    import imageio.v2 as imageio

    stills = set(stills)
    writer = imageio.get_writer(out, fps=fps, macro_block_size=None)
    for i in range(nf):
        im = Image.fromarray(render(i))
        ImageDraw.Draw(im).text((6, 6), f"{i}/{nf}  {header}", fill=(255, 255, 0))
        if i in stills:
            still = f"{os.path.splitext(out)[0]}_f{i}.png"
            im.save(still)
            print(f"[viz] still -> {still}")
        writer.append_data(np.asarray(im))
    writer.close()
    print(f"[viz] wrote {nf} frames -> {out}")
