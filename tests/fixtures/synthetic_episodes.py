"""Tiny synthetic zarr episodes per vendor, written through the real ZarrWriter.

Vendor differences that matter to the data pipeline are reproduced (camera set,
stride, head pose); everything else is random but non-degenerate so norm stats
are finite.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from egomimic.rldb.embodiment.eva import Eva
from egomimic.rldb.zarr import dexmate_rules as dex
from egomimic.rldb.zarr.required_metadata import MissingMetadata
from egomimic.rldb.zarr.zarr_writer import ZarrWriter
from egomimic.utils.pose_utils import _matrix_to_xyzwxyz, _xyzwxyz_to_matrix


@dataclass(frozen=True)
class Vendor:
    embodiment: str
    stride: int
    cameras: tuple[str, ...]  # zarr image keys without the "images." prefix
    has_head_pose: bool


VENDORS: dict[str, Vendor] = {
    "eva": Vendor("eva_bimanual", 1, ("front_1", "right_wrist", "left_wrist"), False),
    "aria": Vendor("human_bimanual", 3, ("front_1",), True),
    "mecka": Vendor("human_bimanual", 1, ("front_1",), True),
    "scale": Vendor("human_bimanual", 1, ("front_1",), True),
}


def _quat_wxyz(T: int, rng: np.random.Generator) -> np.ndarray:
    q = R.from_rotvec(rng.normal(0, 0.05, (T, 3))).as_quat()  # xyzw
    return np.concatenate([q[:, 3:4], q[:, :3]], axis=1)


def _pose(T: int, rng: np.random.Generator) -> np.ndarray:
    """One random pose held constant for all T frames (xyz + wxyz).

    Constant-in-time on purpose: the loader's outlier check compares every
    action-chunk cell against per-timestep 99.99% quantiles of the norm-stat
    sample. With only a few hundred synthetic samples, white-noise
    trajectories make nearly every sample the maximum of some cell and the
    whole dataset gets rejected; per-episode constants tie at the extremes
    instead, while three episodes still give non-degenerate norm stats."""
    one = np.concatenate([rng.normal(0, 0.1, (1, 3)), _quat_wxyz(1, rng)], axis=1)
    return np.repeat(one, T, axis=0).astype(np.float64)


def _scalar(T: int, rng: np.random.Generator) -> np.ndarray:
    """One uniform(0,1) value held constant for all T frames, shape (T, 1)."""
    return np.full((T, 1), rng.uniform(0.0, 1.0))


def _pose_in_frame(T: int, rng: np.random.Generator, frame: np.ndarray) -> np.ndarray:
    """A small random pose expressed in the world frame *through* ``frame``
    (4x4). The eva pipeline re-expresses cmd/obs poses relative to the real
    camera extrinsics; poses composed this way come out near identity there,
    so the ypr representation stays away from the +-pi wrap that the
    quantile bounds check would otherwise reject."""
    mats = _xyzwxyz_to_matrix(_pose(T, rng))
    return _matrix_to_xyzwxyz(frame[None] @ mats)


def write_episode(
    root: Path, vendor: str, *, T: int = 48, H: int = 64, W: int = 64, seed: int = 0
) -> Path:
    v = VENDORS[vendor]
    rng = np.random.default_rng(seed)
    K = np.array(
        [[100.0, 0.0, W / 2, 0.0], [0.0, 100.0, H / 2, 0.0], [0.0, 0.0, 1.0, 0.0]]
    )
    images = {
        f"images.{c}": rng.integers(0, 255, (T, H, W, 3), dtype=np.uint8)
        for c in v.cameras
    }
    numeric = {"left.obs_ee_pose": _pose(T, rng), "right.obs_ee_pose": _pose(T, rng)}
    extrinsics = None
    if v.embodiment == "eva_bimanual":
        ext_l, ext_r = Eva.EXTRINSICS["left"], Eva.EXTRINSICS["right"]
        numeric.update(
            {
                "left.obs_ee_pose": _pose_in_frame(T, rng, ext_l),
                "right.obs_ee_pose": _pose_in_frame(T, rng, ext_r),
                "left.cmd_ee_pose": _pose_in_frame(T, rng, ext_l),
                "right.cmd_ee_pose": _pose_in_frame(T, rng, ext_r),
                "left.obs_gripper": _scalar(T, rng),
                "right.obs_gripper": _scalar(T, rng),
                "left.cmd_gripper": _scalar(T, rng),
                "right.cmd_gripper": _scalar(T, rng),
            }
        )
        extrinsics = {
            "left": ext_l,
            "right": ext_r,
        }  # what the poses were composed through
    if v.has_head_pose:
        numeric["obs_head_pose"] = _pose(T, rng)
    return ZarrWriter.create_and_write(
        root / f"{vendor}_{seed:02d}.zarr",
        numeric_data=numeric,
        image_data=images,
        embodiment=v.embodiment,
        fps=30,
        task_name="synthetic",
        operator=f"op_{seed:02d}",
        lab="synthetic",
        scene="test_scene",
        objects=["red_cube", "bin"],
        vendor_recording_id=f"{vendor}_session_{seed:02d}",
        annotations=[
            ("pick up the red cube", 0, T // 2),
            ("place it in the bin", T // 2, T - 1),
        ],
        intrinsics={"front_1": K},
        extrinsics=extrinsics,
    )


def write_dexmate_episode(
    root: Path,
    *,
    T: int = 48,
    H: int = 64,
    W: int = 64,
    seed: int = 0,
    wrist_cameras: bool = True,
    intrinsics: dict | MissingMetadata = MissingMetadata(),
    arrays: dict[str, np.ndarray | None] | None = None,
    name: str | None = None,
) -> Path:
    """A dexmate_bimanual episode holding every array dexmate_rules requires,
    plus cmd_hand_keypoints and cmd_ee_pose; held constant in time like the
    vendor episodes above (see _pose). Hand joints sit inside the URDF limits.
    SharpaDex has no camera calibration, so intrinsics default to
    MissingMetadata(). ``arrays`` replaces arrays, or drops them with None.
    Separate from write_episode so the vendor episodes stay byte-identical."""
    rng = np.random.default_rng(seed)
    cameras = (
        ("front_1", "right_wrist", "left_wrist") if wrist_cameras else ("front_1",)
    )
    images = {
        f"images.{c}": rng.integers(0, 255, (T, H, W, 3), dtype=np.uint8)
        for c in cameras
    }

    def held(row: np.ndarray) -> np.ndarray:
        return np.repeat(np.asarray(row, dtype=np.float64)[None], T, axis=0)

    numeric = {
        dex.HEAD_LINK_POSE: _pose(T, rng),
        "obs_aux_joints": held(rng.uniform(-0.3, 0.3, dex.AUX_DOF)),
        "cmd_aux_joints": held(rng.uniform(-0.3, 0.3, dex.AUX_DOF)),
    }
    for side in dex.SIDES:
        lower, upper = np.asarray(dex.HAND_JOINT_LIMITS[side]).T
        hand = held(lower + rng.uniform(0.2, 0.8, dex.HAND_DOF) * (upper - lower))
        palm = _pose(T, rng)
        keypoints = palm[:, None, :3] + rng.normal(0, 0.05, (1, 21, 3))
        numeric.update(
            {
                f"{side}.obs_ee_pose": palm,
                f"{side}.cmd_ee_pose": palm,
                f"{side}.obs_joints": held(rng.uniform(-1, 1, dex.ARM_DOF)),
                f"{side}.cmd_joints": held(rng.uniform(-1, 1, dex.ARM_DOF)),
                f"{side}.obs_hand_joints": hand,
                f"{side}.cmd_hand_joints": hand,
                f"{side}.obs_hand_keypoints": keypoints.reshape(T, dex.KEYPOINT_DIM),
                f"{side}.cmd_hand_keypoints": keypoints.reshape(T, dex.KEYPOINT_DIM),
                f"{side}.obs_tactile": held(rng.normal(0, 1, dex.TACTILE_DIM)),
            }
        )
    for key, value in (arrays or {}).items():
        if value is None:
            numeric.pop(key)
        else:
            numeric[key] = value
    return ZarrWriter.create_and_write(
        root / f"{name or f'dexmate_{seed:02d}'}.zarr",
        numeric_data=numeric,
        image_data=images,
        embodiment=dex.EMBODIMENT,
        fps=30,
        task_name="synthetic",
        operator="sharpa",
        lab="synthetic",
        scene="test_scene",
        objects=["red_cube", "bin"],
        vendor_recording_id=f"task__season__{seed:06d}",
        annotations=[
            ("pick up the red cube", 0, T // 2),
            ("place it in the bin", T // 2, T - 1),
        ],
        intrinsics=intrinsics,
    )
