"""EVA hardware pose convention used by dataset conversion and live rollout."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

# ``eva_to_zarr`` left-multiplies every observed and commanded hardware
# orientation by this fixed rotation before writing the training dataset.
# Translation is intentionally unchanged. Live rollout must apply the same
# conversion on input and its inverse on output.
EVA_DATASET_FROM_HARDWARE_ROTATION = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)
EVA_HARDWARE_FROM_DATASET_ROTATION = EVA_DATASET_FROM_HARDWARE_ROTATION.T.copy()
EVA_DATASET_FROM_HARDWARE_ROTATION.setflags(write=False)
EVA_HARDWARE_FROM_DATASET_ROTATION.setflags(write=False)


def _require_last_dim(value: np.ndarray, width: int, name: str) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if value.ndim == 0 or value.shape[-1] != width:
        raise ValueError(f"{name} must have last dimension {width}, got {value.shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} contains non-finite values")
    return value


def hardware_ypr_pose_to_dataset_ypr(pose: np.ndarray) -> np.ndarray:
    """Convert hardware ``xyz + ZYX-ypr`` poses to the EVA Zarr convention."""

    pose = _require_last_dim(pose, 6, "pose")
    hardware_rotation = Rotation.from_euler("ZYX", pose[..., 3:6]).as_matrix()
    dataset_rotation = EVA_DATASET_FROM_HARDWARE_ROTATION @ hardware_rotation
    dataset_ypr = Rotation.from_matrix(dataset_rotation).as_euler("ZYX")
    return np.concatenate([pose[..., :3], dataset_ypr], axis=-1)


def hardware_ypr_pose_to_dataset_wxyz(pose: np.ndarray) -> np.ndarray:
    """Convert hardware ``xyz + ZYX-ypr`` poses to stored ``xyz + wxyz``."""

    pose = _require_last_dim(pose, 6, "pose")
    hardware_rotation = Rotation.from_euler("ZYX", pose[..., 3:6]).as_matrix()
    dataset_rotation = EVA_DATASET_FROM_HARDWARE_ROTATION @ hardware_rotation
    dataset_xyzw = Rotation.from_matrix(dataset_rotation).as_quat()
    dataset_wxyz = np.concatenate(
        [dataset_xyzw[..., 3:4], dataset_xyzw[..., :3]], axis=-1
    )
    return np.concatenate([pose[..., :3], dataset_wxyz], axis=-1)


def dataset_ypr_pose_to_hardware_ypr(pose: np.ndarray) -> np.ndarray:
    """Convert EVA Zarr ``xyz + ZYX-ypr`` poses back to hardware convention."""

    pose = _require_last_dim(pose, 6, "pose")
    dataset_rotation = Rotation.from_euler("ZYX", pose[..., 3:6]).as_matrix()
    hardware_rotation = EVA_HARDWARE_FROM_DATASET_ROTATION @ dataset_rotation
    hardware_ypr = Rotation.from_matrix(hardware_rotation).as_euler("ZYX")
    return np.concatenate([pose[..., :3], hardware_ypr], axis=-1)
