"""Report calibration capabilities for a particular view and observation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from egomimic.rldb.embodiment import Embodiment
from egomimic.rldb.zarr.calibration import CalibrationError, read_calibration
from egomimic.utils.pose_utils import _xyzwxyz_to_matrix


@dataclass
class CameraCoverage:
    camera: str
    missing: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    pose_source: str | None = None
    estimated: bool = False
    K: np.ndarray | None = None
    source_T_cam: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def available(self):
        return not self.missing

    def to_jsonable(self):
        return {
            "camera": self.camera,
            "available": self.available,
            "missing": self.missing,
            "limitations": self.limitations,
            "pose_source": self.pose_source,
            "estimated": self.estimated,
        }


def _array(group, key):
    for name in (key, key.replace(".", "/")):
        try:
            return group[name]
        except KeyError:
            pass
    return None


def _rigid(transform):
    return (
        transform is not None
        and transform.shape == (4, 4)
        and np.isfinite(transform).all()
        and np.allclose(transform[3], [0, 0, 0, 1], atol=1e-5)
        and np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-4)
        and np.isclose(np.linalg.det(transform[:3, :3]), 1, atol=1e-4)
    )


def camera_coverage(
    group, camera="front_1", frame=0, *, resolved=None, image_shape=None
):
    """Describe the requested view without requiring calibration of other views.

    ``obs_head_pose[t]`` is the optical front camera pose, never a wrist pose.
    Supplied trajectories take precedence over static metadata. Missing motion
    is a limitation of the operation needing it, not a universal data error.
    """
    result = CameraCoverage(camera)
    attrs = dict(group.attrs)
    resolved = Embodiment.from_attrs(attrs) if resolved is None else resolved
    total = attrs.get("total_frames", 0)
    image = _array(group, f"images.{camera}")
    if image is None or not image.shape or image.shape[0] < total:
        result.missing.append(f"images.{camera} missing or shorter than total_frames")
    if not 0 <= frame < total:
        result.missing.append("observation frame outside total_frames")
        return result
    # An unrelated malformed wrist block must not gate an ego-only consumer.
    block = attrs.get("calibration")
    if isinstance(block, dict) and isinstance(block.get("cameras"), dict):
        names = {camera, str(block.get("reference_frame", "")).removeprefix("camera:")}
        attrs["calibration"] = {
            **block,
            "cameras": {k: v for k, v in block["cameras"].items() if k in names},
        }
        if not attrs["calibration"]["cameras"]:
            attrs["calibration"]["cameras"] = {camera: {}}
    try:
        calibration = read_calibration(attrs)
    except CalibrationError as exc:
        result.missing.append(str(exc))
        return result
    if calibration is None or calibration.K(camera) is None:
        shape = image_shape or (
            attrs.get("features", {}).get(f"images.{camera}", {}).get("shape")
        )
        if (
            calibration is None
            and str(attrs.get("embodiment", "")).lower().startswith("aria_")
            and camera == "front_1"
            and shape is not None
            and tuple(shape[:2]) in ((480, 640), (240, 320))
        ):
            from egomimic.rldb.embodiment.human import (
                ARIA_INTRINSICS,
                ARIA_INTRINSICS_HALF,
            )

            result.K = (
                ARIA_INTRINSICS.copy()
                if shape[0] == 480
                else ARIA_INTRINSICS_HALF.copy()
            )
            result.limitations.append(
                "using the established rectified Aria intrinsics fallback; episode calibration missing"
            )
        else:
            result.missing.append(f"{camera}: no intrinsics")
    else:
        entry = calibration.cameras[camera]
        result.K = entry.K
        if result.K[0, 0] <= 0 or result.K[1, 1] <= 0:
            result.missing.append("camera focal lengths must be positive")
        if not entry.rectified and (entry.model != "PINHOLE" or any(entry.distortion)):
            result.missing.append(
                f"unrectified {entry.model} projection is not implemented"
            )
        if (
            image_shape is not None
            and entry.resolution is not None
            and tuple(image_shape[:2][::-1]) != entry.resolution
        ):
            result.missing.append(
                f"calibration resolution {entry.resolution} differs from stored image {image_shape[:2][::-1]}"
            )
    provenance = attrs.get("preview_provenance") or {}
    result.estimated = camera in provenance.get("estimated_cameras", [])
    if result.estimated:
        result.limitations.append(
            "camera mount/intrinsics estimated for analysis; not measured calibration"
        )
    trajectory = _array(group, "obs_head_pose") if camera == "front_1" else None
    ref_T_cam = None
    if trajectory is not None:
        result.pose_source = "obs_head_pose at displayed observation"
        if trajectory.shape[1:] != (7,) or trajectory.shape[0] < total:
            result.missing.append(
                "obs_head_pose needs shape (T, 7) on the retained timeline"
            )
        else:
            pose = np.asarray(trajectory[frame])
            if not np.isfinite(pose).all() or not np.isclose(
                np.linalg.norm(pose[3:]), 1, atol=1e-3
            ):
                result.missing.append(f"invalid obs_head_pose at frame {frame}")
            else:
                ref_T_cam = _xyzwxyz_to_matrix(pose[None])[0]
    elif calibration is not None:
        ref_T_cam = calibration.ref_T_cam(camera)
        result.pose_source = "static calibration"
        # Legacy human recordings without head poses follow has_head_pose=False.
        if (
            calibration.legacy
            and resolved.platform.kind == "human"
            and camera == "front_1"
        ):
            result.limitations.append(
                "legacy human points interpreted in the target camera frame (no head trajectory)"
            )
    for side in resolved.end_effectors:
        transform = ref_T_cam
        if (
            trajectory is None
            and calibration is not None
            and calibration.legacy
            and resolved.platform.kind == "robot"
            and side not in calibration.arm_bases
        ):
            # Legacy intrinsics imply no robot-base-to-camera relationship.
            transform = None
        if (
            transform is not None
            and calibration is not None
            and side in calibration.arm_bases
        ):
            transform = np.linalg.inv(calibration.arm_bases[side]) @ transform
        # Retain EVA's existing compatibility constants only when no trajectory
        # or arm extrinsics were declared. Never apply them to another platform.
        elif (
            trajectory is None
            and resolved.platform.name == "eva_x5"
            and camera == "front_1"
            and (calibration is None or calibration.legacy)
        ):
            from egomimic.rldb.embodiment.eva import Eva

            transform = Eva.EXTRINSICS[side]
            result.pose_source = "legacy EVA fallback"
            result.limitations.append(f"{side}: using EVA's compatibility camera pose")
        if _rigid(transform):
            result.source_T_cam[side] = transform
        else:
            result.missing.append(
                f"{camera}: no valid transform from stored {side} point frame"
            )
    return result


def camera_coverage_report(group, frame=0):
    """Report ego and wrist coverage independently for all stored image views."""
    names = [
        key.removeprefix("images.")
        for key in group.array_keys()
        if key.startswith("images.")
    ]
    return {name: camera_coverage(group, name, frame).to_jsonable() for name in names}
