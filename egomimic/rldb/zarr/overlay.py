"""Episode keypoint overlays shared by the inspector and render_check.

All points in a horizon use the camera at the displayed observation frame.
Projection and drawing live in Embodiment.viz, also used by training previews.
"""

from __future__ import annotations

import numpy as np
import simplejpeg

from egomimic.rldb.embodiment import Embodiment
from egomimic.rldb.zarr.camera_coverage import camera_coverage
from egomimic.utils.pose_utils import cam_frame_to_cam_pixels


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
    coverage = camera_coverage(group, camera, frame, resolved=resolved)
    if not coverage.available:
        raise OverlayUnavailable("; ".join(coverage.missing))
    return coverage.K, coverage.source_T_cam


def keypoint_source(group, side, spec):
    """Choose supplied points and their actual topology, without reconstruction."""
    suffix = "obs_hand_keypoints" if spec.ee_class == "dexterous_hand" else "obs_keypoints"
    key = f"{side}.{suffix}"
    array = array_for(group, key)
    edges, ranges, root = Embodiment.FINGER_EDGES, Embodiment.FINGER_EDGE_RANGES, 0
    if array is None and spec.ee_class == "human_hand":
        from egomimic.rldb.embodiment.human import (
            ARIA_FINGER_EDGE_RANGES,
            ARIA_FINGER_EDGES,
        )

        key = f"{side}.obs_aria_keypoints"
        array = array_for(group, key)
        edges, ranges, root = ARIA_FINGER_EDGES, ARIA_FINGER_EDGE_RANGES, 5
    return array, key, edges, ranges, root


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
        array, key, _, _, _ = keypoint_source(group, side, spec)
        n = spec.keypoints.n_slots
        if array is None or array.shape[0] < total or array.shape[1:] != (3 * n,):
            raise OverlayUnavailable(f"{key}: supplied keypoints need shape (T, {3 * n})")
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
    coverage = camera_coverage(group, camera, frame, image_shape=image.shape)
    if not coverage.available:
        raise OverlayUnavailable("; ".join(coverage.missing))
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
    diagnostics["coverage"] = coverage.to_jsonable()
    diagnostics["warnings"] = []
    if diagnostics["inside_fraction"] < 0.5:
        diagnostics["warnings"].append(
            "fewer than half of owned keypoints project inside the image"
        )
    if inside.any() and np.linalg.norm(np.ptp(pixels[inside], axis=0)) < 3:
        diagnostics["warnings"].append(
            "projected points collapse to a near-single pixel"
        )
    edge_pixels, edge_metres = [], []
    full_points = chunk.reshape(len(chunk), 2, -1, 3)
    sources = {}
    for index, side in enumerate(("left", "right")):
        if side not in resolved.end_effectors:
            continue
        slots = resolved.keypoints(side).valid
        _, key, edges, ranges, root = keypoint_source(group, side, resolved.end_effectors[side])
        sources[side] = (key, edges, ranges, root)
        for a, b in edges:
            if a not in slots or b not in slots:
                continue
            pair = full_points[:, index, [a, b], :]
            good = np.isfinite(pair).all(axis=(1, 2)) & (pair[:, :, 2] > 0.01).all(
                axis=1
            )
            if not good.any():
                continue
            pair = pair[good]
            projected = cam_frame_to_cam_pixels(pair.reshape(-1, 3), K)[:, :2].reshape(
                -1, 2, 2
            )
            edge_pixels.extend(
                np.linalg.norm(projected[:, 1] - projected[:, 0], axis=1).tolist()
            )
            edge_metres.extend(np.linalg.norm(pair[:, 1] - pair[:, 0], axis=1).tolist())
    diagnostics["skeleton_edge_px_range"] = (
        [min(edge_pixels), max(edge_pixels)] if edge_pixels else None
    )
    diagnostics["skeleton_edge_m_range"] = (
        [min(edge_metres), max(edge_metres)] if edge_metres else None
    )
    diagnostics["keypoint_sources"] = {side: source[0] for side, source in sources.items()}
    if any(source[3] == 5 for source in sources.values()):
        # Mixed canonical/raw-Aria sides keep independent topology and masks.
        for index, side in enumerate(("left", "right")):
            if side not in sources:
                continue
            _, edges, ranges, root = sources[side]
            one_side = np.full_like(chunk, np.nan)
            one_side[:, index * 63:(index + 1) * 63] = chunk[:, index * 63:(index + 1) * 63]
            image = resolved.viz(image, one_side, intrinsics=K, finger_edges=edges,
                                 finger_edge_ranges=ranges, label_slot=root)
        return image, diagnostics
    return resolved.viz(image, chunk, intrinsics=K), diagnostics


def pose_chunk(group, frame, horizon=1, camera="front_1"):
    """Load the familiar Cartesian pose window into the displayed camera frame."""
    from scipy.spatial.transform import Rotation

    total = episode_length(group)
    if horizon < 1 or not 0 <= frame < total:
        raise OverlayUnavailable("frame/horizon outside episode")
    resolved = Embodiment.from_attrs(group.attrs)
    K, transforms = camera_context(group, frame, resolved, camera)
    end = min(total, frame + horizon)
    parts = []
    for side in ("left", "right"):
        output = np.full((end - frame, 6), np.nan)
        if side in resolved.end_effectors:
            array = array_for(group, f"{side}.obs_ee_pose")
            if array is None or array.shape[0] < total or array.shape[1:] != (7,):
                raise OverlayUnavailable(f"{side}.obs_ee_pose needs shape (T, 7)")
            poses = np.asarray(array[frame:end], dtype=float)
            missing = (np.abs(poses) >= 1e8).all(axis=-1) if resolved.platform.kind == "human" else np.zeros(len(poses), bool)
            valid = ~missing
            if not np.isfinite(poses[valid]).all() or not np.allclose(
                np.linalg.norm(poses[valid, 3:], axis=-1), 1, atol=1e-3
            ):
                raise OverlayUnavailable(f"{side}.obs_ee_pose: invalid retained pose")
            transform = np.linalg.inv(transforms[side])
            output[valid, :3] = poses[valid, :3] @ transform[:3, :3].T + transform[:3, 3]
            if valid.any():
                rotations = transform[:3, :3] @ Rotation.from_quat(poses[valid][:, [4, 5, 6, 3]]).as_matrix()
                output[valid, 3:] = Rotation.from_matrix(rotations).as_euler("ZYX")
        parts.append(output)
    return resolved, K, np.concatenate(parts, axis=-1)


def render_overlay(group, frame, *, mode="keypoint", image=None, horizon=1, camera="front_1"):
    """Shared inspector/artifact modes, backed by the established renderers."""
    image = decode_frame(group, frame, camera) if image is None else image
    if mode == "none":
        return image, {"coverage": {}, "warnings": []}
    if mode == "keypoint":
        return render_keypoints(group, frame, image=image, horizon=horizon, camera=camera)
    if mode not in ("cartesian", "orientation"):
        raise ValueError(f"unknown overlay mode {mode!r}")
    coverage = camera_coverage(group, camera, frame, image_shape=image.shape)
    if not coverage.available:
        raise OverlayUnavailable("; ".join(coverage.missing))
    # Orientation shows the current axes; Cartesian shows the future trajectory.
    resolved, K, chunk = pose_chunk(group, frame, 1 if mode == "orientation" else horizon, camera)
    rendered = resolved.viz(image, chunk, mode="axes" if mode == "orientation" else "traj", intrinsics=K)
    return rendered, {"coverage": coverage.to_jsonable(), "pose_frames": len(chunk), "warnings": []}
