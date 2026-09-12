"""Episode keypoint overlays shared by the inspector and render_check.

All points in a horizon use the camera at the displayed observation frame.
Projection and drawing live in Embodiment.viz, also used by training previews.
"""

from __future__ import annotations

import numpy as np
import simplejpeg

from egomimic.rldb.embodiment import Embodiment
from egomimic.rldb.zarr.calibration import read_calibration
from egomimic.utils.pose_utils import _xyzwxyz_to_matrix, cam_frame_to_cam_pixels


class OverlayUnavailable(ValueError):
    """The requested view cannot interpret the stored representation."""


def episode_length(group) -> int:
    value = group.attrs.get("total_frames")
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise OverlayUnavailable("total_frames must be a positive integer")
    return value


def array_for(group, key):
    """Accept flat dotted keys and the inspector's legacy nested layout."""
    for candidate in (key, key.replace(".", "/")):
        try:
            return group[candidate]
        except KeyError:
            pass
    return None


def decode_frame(group, frame: int, camera="front_1") -> np.ndarray:
    if not 0 <= frame < episode_length(group):
        raise OverlayUnavailable("frame lies outside total_frames")
    array = array_for(group, f"images.{camera}")
    if array is None or array.shape[0] <= frame:
        raise OverlayUnavailable(f"images.{camera} is missing or too short")
    value = array[frame]
    if isinstance(value, np.ndarray) and value.ndim == 3:
        return np.asarray(value, dtype=np.uint8)
    # Zarr's VLenBytes codec can return several nested scalar object arrays.
    for _ in range(8):
        if not isinstance(value, np.ndarray) or value.ndim != 0:
            break
        value = value.item()
    if hasattr(value, "to_bytes"):
        value = value.to_bytes()
    if isinstance(value, np.ndarray):
        value = value.tobytes()
    return simplejpeg.decode_jpeg(bytes(value), colorspace="RGB")


def camera_context(group, frame, resolved, camera="front_1"):
    """Resolve current optical camera poses for the stored point frame."""
    calibration = read_calibration(group.attrs)
    if calibration is None or calibration.K(camera) is None:
        raise OverlayUnavailable(f"{camera}: no intrinsics")
    head = array_for(group, "obs_head_pose")
    if camera == "front_1" and head is not None:
        if head.shape[1:] != (7,) or head.shape[0] < episode_length(group):
            raise OverlayUnavailable("obs_head_pose must have shape (T, 7)")
        pose = np.asarray(head[frame])
        if not np.isfinite(pose).all() or not np.isclose(
            np.linalg.norm(pose[3:]), 1, atol=1e-3
        ):
            raise OverlayUnavailable(f"invalid obs_head_pose at frame {frame}")
        transform = _xyzwxyz_to_matrix(pose[None])[0]
        return calibration.K(camera), {s: transform for s in resolved.end_effectors}
    transforms = {}
    for side in resolved.end_effectors:
        if side in calibration.arm_bases:
            transform = calibration.base_T_cam(side, camera)
        else:
            transform = calibration.ref_T_cam(camera)
        if transform is None:
            raise OverlayUnavailable(
                f"{camera}: no transform from stored {side} point frame"
            )
        transforms[side] = transform
    return calibration.K(camera), transforms


def keypoint_chunk(group, frame, horizon=1, camera="front_1"):
    """Load canonical stored keypoints into a single current camera frame."""
    total = episode_length(group)
    if horizon < 1 or not 0 <= frame < total:
        raise OverlayUnavailable("frame/horizon outside episode")
    resolved = Embodiment.from_attrs(group.attrs)
    K, transforms = camera_context(group, frame, resolved, camera)
    end = min(total, frame + horizon)
    parts, owned = [], []
    for side in ("left", "right"):
        spec = resolved.end_effectors.get(side)
        if spec is None:
            parts.append(np.full((end - frame, 63), np.nan))
            owned.append(np.zeros(21, dtype=bool))
            continue
        suffix = (
            "obs_hand_keypoints"
            if spec.ee_class == "dexterous_hand"
            else "obs_keypoints"
        )
        array = array_for(group, f"{side}.{suffix}")
        n = spec.keypoints.n_slots
        if array is None or array.shape[0] < total or array.shape[1:] != (3 * n,):
            raise OverlayUnavailable(f"{side}.{suffix} needs shape (T, {3 * n})")
        points = np.asarray(array[frame:end], dtype=float).reshape(end - frame, n, 3)
        invalid = ~np.isfinite(points).all(axis=-1) | (np.abs(points) >= 1e8).any(
            axis=-1
        )
        cam_T_source = np.linalg.inv(transforms[side])
        points = points @ cam_T_source[:3, :3].T + cam_T_source[:3, 3]
        points[invalid] = np.nan
        parts.append(points.reshape(end - frame, 3 * n))
        owned.append(np.isin(np.arange(n), spec.keypoints.valid))
    return resolved, K, np.concatenate(parts, axis=-1), np.concatenate(owned)


def render_keypoints(group, frame, *, image=None, horizon=1, camera="front_1"):
    """Return an RGB overlay and projection diagnostics for the requested view."""
    image = decode_frame(group, frame, camera) if image is None else image
    resolved, K, chunk, owned = keypoint_chunk(group, frame, horizon, camera)
    points = chunk.reshape(len(chunk), -1, 3)[:, owned].reshape(-1, 3)
    finite = np.isfinite(points).all(axis=-1)
    positive = finite & (points[:, 2] > 0.01)
    with np.errstate(invalid="ignore", divide="ignore"):
        pixels = cam_frame_to_cam_pixels(points, K)
    height, width = image.shape[:2]
    inside = (
        positive
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )
    diagnostics = {
        "points": len(points),
        "finite": int(finite.sum()),
        "positive_depth": int(positive.sum()),
        "inside_image": int(inside.sum()),
        "inside_fraction": float(inside.mean()) if len(inside) else 0.0,
    }
    return resolved.viz(image, chunk, intrinsics=K), diagnostics
