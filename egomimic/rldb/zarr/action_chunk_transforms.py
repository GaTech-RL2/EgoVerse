"""
Embodiment-dependent action chunk transforms for ZarrDataset.

Replicates the prestacking transformations from aria_to_lerobot.py / eva_to_lerobot.py,
applied at load time instead of at data creation time. Raw action frames are loaded
as (action_horizon, action_dim) and interpolated to (chunk_length, action_dim).

Translation (xyz) and gripper dimensions use linear interpolation.
Rotation (euler ypr) dimensions use np.unwrap before interpolation and rewrap after,
matching the behaviour of egomimicUtils.interpolate_arr_euler.
"""

from __future__ import annotations
from abc import abstractmethod

import numpy as np
from scipy.interpolate import interp1d
from projectaria_tools.core.sophus import SE3
from scipy.spatial.transform import Rotation as R


# ---------------------------------------------------------------------------
# Helper interpolation functions
# ---------------------------------------------------------------------------


def _interpolate_euler(seq: np.ndarray, chunk_length: int) -> np.ndarray:
    """Euler-aware interpolation for a single (T, 6) or (T, 7) sequence.

    Layout: [x, y, z, yaw, pitch, roll, (optional gripper)]

    - xyz: linear interpolation
    - ypr: np.unwrap + linear interp + rewrap to [-pi, pi)
    - gripper: linear interpolation (if present)
    - Bad-data sentinel: any value >= 1e8 -> fill with 1e9
    """
    T, D = seq.shape
    assert D in (6, 7), f"Expected 6 or 7 dims, got {D}"

    # Bad-data sentinel check
    if np.any(seq >= 1e8):
        return np.full((chunk_length, D), 1e9)

    old_time = np.linspace(0, 1, T)
    new_time = np.linspace(0, 1, chunk_length)

    # Translation
    trans_interp = interp1d(old_time, seq[:, :3], axis=0, kind="linear")(new_time)

    # Rotation (euler) – unwrap before interp, rewrap after
    rot_unwrapped = np.unwrap(seq[:, 3:6], axis=0)
    rot_interp = interp1d(old_time, rot_unwrapped, axis=0, kind="linear")(new_time)
    rot_interp = (rot_interp + np.pi) % (2 * np.pi) - np.pi

    if D == 6:
        return np.concatenate([trans_interp, rot_interp], axis=-1)

    # Gripper
    grip_interp = interp1d(old_time, seq[:, 6:7], axis=0, kind="linear")(new_time)
    return np.concatenate([trans_interp, rot_interp, grip_interp], axis=-1)


def _interpolate_linear(seq: np.ndarray, chunk_length: int) -> np.ndarray:
    """Simple linear interpolation for arbitrary (T, D) arrays."""
    T, D = seq.shape
    old_time = np.linspace(0, 1, T)
    new_time = np.linspace(0, 1, chunk_length)
    return interp1d(old_time, seq, axis=0, kind="linear")(new_time)


# ---------------------------------------------------------------------------
# Base Transform
# ---------------------------------------------------------------------------


class Transform:
    """Base Class for all transforms."""

    @abstractmethod
    def transform(self, batch: dict) -> dict:
        """Transform the data."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Interpolation Transforms
# ---------------------------------------------------------------------------


class InterpolatePose(Transform):
    """Interpolate a pose chunk of shape (T, 6) using Euler-aware interpolation."""

    def __init__(
        self,
        new_chunk_length: int,
        action_key: str,
        output_action_key: str,
        stride: int = 1,
    ):
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")
        self.new_chunk_length = new_chunk_length
        self.action_key = action_key
        self.output_action_key = output_action_key
        self.stride = int(stride)

    def transform(self, batch: dict) -> dict:
        actions = np.asarray(batch[self.action_key])
        if actions.ndim != 2 or actions.shape[-1] != 6:
            raise ValueError(
                f"InterpolatePose expects (T, 6), got {actions.shape} for key "
                f"'{self.action_key}'"
            )
        actions = actions[:: self.stride]
        batch[self.output_action_key] = _interpolate_euler(
            actions, self.new_chunk_length
        )
        return batch


class InterpolateLinear(Transform):
    """Interpolate any chunk of shape (T, D) with linear interpolation."""

    def __init__(
        self,
        new_chunk_length: int,
        action_key: str,
        output_action_key: str,
        stride: int = 1,
    ):
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")
        self.new_chunk_length = new_chunk_length
        self.action_key = action_key
        self.output_action_key = output_action_key
        self.stride = int(stride)

    def transform(self, batch: dict) -> dict:
        actions = np.asarray(batch[self.action_key])
        if actions.ndim != 2:
            raise ValueError(
                f"InterpolateLinear expects (T, D), got {actions.shape} for key "
                f"'{self.action_key}'"
            )
        actions = actions[:: self.stride]
        batch[self.output_action_key] = _interpolate_linear(
            actions, self.new_chunk_length
        )
        return batch


# ---------------------------------------------------------------------------
# Coordinate Transforms
# ---------------------------------------------------------------------------


def _xyzypr_to_matrix(xyzypr: np.ndarray) -> np.ndarray:
    """
    args:
        xyzypr: (B, 6) np.array of [[x, y, z, yaw, pitch, roll]]
    returns:
        (B, 4, 4) array of SE3 transformation matrices
    """
    if xyzypr.ndim != 2 or xyzypr.shape[-1] != 6:
        raise ValueError(f"Expected (B, 6) array, got shape {xyzypr.shape}")

    B = xyzypr.shape[0]
    dtype = xyzypr.dtype if np.issubdtype(xyzypr.dtype, np.floating) else np.float64

    mats = np.broadcast_to(np.eye(4, dtype=dtype), (B, 4, 4)).copy()
    # Input is [yaw, pitch, roll], so use ZYX order (Rz @ Ry @ Rx).
    mats[:, :3, :3] = R.from_euler("ZYX", xyzypr[:, 3:6], degrees=False).as_matrix()
    mats[:, :3, 3] = xyzypr[:, :3]

    return mats


def _matrix_to_xyzypr(mats: np.ndarray) -> np.ndarray:
    """
    args:
        mats: (B, 4, 4) array of SE3 transformation matrices
    returns:
        (B, 6) np.array of [[x, y, z, yaw, pitch, roll]]
    """
    if mats.ndim != 3 or mats.shape[-2:] != (4, 4):
        raise ValueError(f"Expected (B, 4, 4) array, got shape {mats.shape}")

    mats = np.asarray(mats)
    dtype = mats.dtype if np.issubdtype(mats.dtype, np.floating) else np.float64

    xyz = mats[:, :3, 3]
    # Match _xyzypr_to_matrix convention: ypr is extracted in ZYX order.
    ypr = R.from_matrix(mats[:, :3, :3]).as_euler("ZYX", degrees=False)

    return np.concatenate([xyz, ypr], axis=-1).astype(dtype, copy=False)


class ActionChunkCoordinateFrameTransform(Transform):
    def __init__(
        self,
        target_world: str,
        chunk_world: str,
        transformed_key_name: str,
        extra_batch_key: dict = None,
    ):
        """
        args:
            target_world:
            chunk_world:
            transformed_key_name:
        """
        self.target_world = target_world
        self.chunk_world = chunk_world
        self.transformed_key_name = transformed_key_name
        self.extra_batch_key = extra_batch_key

    def transform(self, batch):
        """
        args:
            batch:
                target_world: numpy(6): xyz + ypr
                chunk_world: numpy(T, 6): xyz + ypr
                transformed_key_name: str, name of the new key to store the transformed chunk world in

        returns
            batch with new key containing transformed chunk world in target frame: (T, 6)
        """
        batch.update(self.extra_batch_key or {})
        target_world = np.asarray(batch[self.target_world])
        if target_world.shape != (6,):
            raise ValueError(
                f"Expected target_world shape (6,), got {target_world.shape}"
            )
        chunk_world = np.asarray(batch[self.chunk_world])
        if chunk_world.ndim != 2 or chunk_world.shape[1] != 6:
            raise ValueError(
                f"Expected chunk_world shape (T, 6), got {chunk_world.shape}"
            )

        # Convert to SE3 for transformation
        target_se3 = SE3.from_matrix(
            _xyzypr_to_matrix(target_world[None, :])[0]
        )  # (4, 4)
        chunk_se3 = SE3.from_matrix(_xyzypr_to_matrix(chunk_world))  # (T, 4, 4)

        # Compute relative transform and apply to chunk
        chunk_in_target_frame = target_se3.inverse() @ chunk_se3
        chunk_mats = chunk_in_target_frame.to_matrix()
        if chunk_mats.ndim == 2:
            chunk_mats = chunk_mats[None, ...]
        chunk_in_target_frame = _matrix_to_xyzypr(chunk_mats)

        # Store transformed chunk back in batch
        batch[self.transformed_key_name] = chunk_in_target_frame

        return batch


class QuaternionPoseToYPR(Transform):
    """Convert a single pose from xyz + quat(x,y,z,w) to xyz + ypr."""

    def __init__(self, pose_key: str, output_key: str):
        self.pose_key = pose_key
        self.output_key = output_key

    def transform(self, batch: dict) -> dict:
        pose = np.asarray(batch[self.pose_key])
        if pose.shape != (7,):
            raise ValueError(
                f"QuaternionPoseToYPR expects shape (7,), got {pose.shape} for key "
                f"'{self.pose_key}'"
            )
        xyz = pose[:3]
        ypr = R.from_quat(pose[3:7]).as_euler("ZYX", degrees=False)
        batch[self.output_key] = np.concatenate([xyz, ypr], axis=0)
        return batch


class PoseCoordinateFrameTransform(Transform):
    """Transform a single pose (6,) into a target frame pose (6,)."""

    def __init__(self, target_world: str, pose_world: str, transformed_key_name: str):
        self.target_world = target_world
        self.pose_world = pose_world
        self.transformed_key_name = transformed_key_name
        self._chunk_transform = ActionChunkCoordinateFrameTransform(
            target_world=target_world,
            chunk_world=pose_world,
            transformed_key_name=transformed_key_name,
        )

    def transform(self, batch: dict) -> dict:
        pose_world = np.asarray(batch[self.pose_world])
        if pose_world.shape != (6,):
            raise ValueError(f"Expected pose_world shape (6,), got {pose_world.shape}")

        transformed = self._chunk_transform.transform(
            {
                self.target_world: batch[self.target_world],
                self.pose_world: pose_world[None, :],
            }
        )
        batch[self.transformed_key_name] = np.asarray(
            transformed[self.transformed_key_name]
        )[0]
        return batch


class DeleteKeys(Transform):
    def __init__(self, keys_to_delete):
        self.keys_to_delete = keys_to_delete

    def transform(self, batch):
        for key in self.keys_to_delete:
            batch.pop(key, None)
        return batch


class CartesianWithGripperCoordinateTransform(Transform):
    def __init__(
        self,
        left_target_world: str,
        right_target_world: str,
        chunk_world: str,
        transformed_key_name: str,
        extra_batch_key: dict = None,
    ):
        """
        args:
            left_target_world: string key for left target world pose in batch (6D: xyz + ypr)
            right_target_world: string key for right target world pose in batch (6D: xyz + ypr)
            chunk_world: string key for chunk world pose in batch (14D: xyz + ypr + gripper * 2 arms)
            transformed_key_name: string key to store transformed chunk world in batch (14D)
        """
        self.left_target_world = left_target_world
        self.right_target_world = right_target_world
        self.chunk_world = chunk_world
        self.transformed_key_name = transformed_key_name
        self.extra_batch_key = extra_batch_key

    def transform(self, batch):
        """
        args:
            batch:
                left_target_world: numpy(6): xyz + ypr
                right_target_world: numpy(6): xyz + ypr
                chunk_world: numpy(T, 14): [left xyz+ypr+gripper, right xyz+ypr+gripper]
                transformed_key_name: str, name of the new key to store the transformed chunk world in

        returns
            batch with new key containing transformed chunk world in target frame: (T, 14)
        """
        batch.update(self.extra_batch_key or {})
        left_target_world = batch[self.left_target_world]
        right_target_world = batch[self.right_target_world]
        chunk_world = batch[self.chunk_world]

        if left_target_world.shape != (6,):
            raise ValueError(
                f"Expected left_target_world shape (6,), got {left_target_world.shape}"
            )
        if right_target_world.shape != (6,):
            raise ValueError(
                f"Expected right_target_world shape (6,), got {right_target_world.shape}"
            )
        if chunk_world.ndim != 2 or chunk_world.shape[1] != 14:
            raise ValueError(
                f"Expected chunk_world shape (T, 14), got {chunk_world.shape}"
            )

        # Chunk layout: [left xyz+ypr+gripper, right xyz+ypr+gripper]
        left_pose_world = chunk_world[:, :6]
        right_pose_world = chunk_world[:, 7:13]

        left_target_se3 = SE3.from_matrix(
            _xyzypr_to_matrix(left_target_world[None, :])[0]
        )
        right_target_se3 = SE3.from_matrix(
            _xyzypr_to_matrix(right_target_world[None, :])[0]
        )
        left_target_inv = left_target_se3.inverse()
        right_target_inv = right_target_se3.inverse()

        left_pose_in_target = _matrix_to_xyzypr(
            (
                left_target_inv @ SE3.from_matrix(_xyzypr_to_matrix(left_pose_world))
            ).to_matrix()
        )
        right_pose_in_target = _matrix_to_xyzypr(
            (
                right_target_inv @ SE3.from_matrix(_xyzypr_to_matrix(right_pose_world))
            ).to_matrix()
        )

        chunk_in_target_frame = np.empty_like(chunk_world)
        chunk_in_target_frame[:, :6] = left_pose_in_target
        chunk_in_target_frame[:, 6] = chunk_world[:, 6]  # left gripper unchanged
        chunk_in_target_frame[:, 7:13] = right_pose_in_target
        chunk_in_target_frame[:, 13] = chunk_world[:, 13]  # right gripper unchanged

        batch[self.transformed_key_name] = chunk_in_target_frame
        return batch


# ---------------------------------------------------------------------------
# Shape Transforms
# ---------------------------------------------------------------------------


class ConcatKeys(Transform):
    def __init__(self, key_list, new_key_name, delete_old_keys=False):
        self.key_list = list(key_list)
        self.new_key_name = new_key_name
        self.delete_old_keys = delete_old_keys

    def transform(self, batch):
        arrays = [np.asarray(batch[k]) for k in self.key_list]
        try:
            batch[self.new_key_name] = np.concatenate(arrays, axis=-1)
        except ValueError as e:
            shapes = {k: np.asarray(batch[k]).shape for k in self.key_list}
            raise ValueError(
                f"ConcatKeys failed for keys {self.key_list} with shapes {shapes}"
            ) from e

        if self.delete_old_keys:
            for k in self.key_list:
                batch.pop(k, None)

        return batch


# ---------------------------------------------------------------------------
# Transform List Factories
# ---------------------------------------------------------------------------


def build_eva_bimanual_transform_list(
    *,
    left_target_world: str = "left_extrinsics_pose",
    right_target_world: str = "right_extrinsics_pose",
    left_cmd_world: str = "left.cmd_ee_pose",
    right_cmd_world: str = "right.cmd_ee_pose",
    left_obs_pose: str = "left.obs_ee_pose",
    right_obs_pose: str = "right.obs_ee_pose",
    left_obs_gripper: str = "left.obs_gripper",
    right_obs_gripper: str = "right.obs_gripper",
    left_gripper: str = "left.gripper",
    right_gripper: str = "right.gripper",
    left_cmd_camframe: str = "left.cmd_ee_pose_camframe",
    right_cmd_camframe: str = "right.cmd_ee_pose_camframe",
    actions_key: str = "actions_cartesian",
    obs_key: str = "observations.state.ee_pose",
    chunk_length: int = 100,
    stride: int = 1,
    left_extra_batch_key: dict | None = None,
    right_extra_batch_key: dict | None = None,
) -> list[Transform]:
    """Canonical EVA bimanual transform pipeline used by tests and notebooks."""
    return [
        ActionChunkCoordinateFrameTransform(
            target_world=left_target_world,
            chunk_world=left_cmd_world,
            transformed_key_name=left_cmd_camframe,
            extra_batch_key=left_extra_batch_key,
        ),
        ActionChunkCoordinateFrameTransform(
            target_world=right_target_world,
            chunk_world=right_cmd_world,
            transformed_key_name=right_cmd_camframe,
            extra_batch_key=right_extra_batch_key,
        ),
        PoseCoordinateFrameTransform(
            target_world=left_target_world,
            pose_world=left_obs_pose,
            transformed_key_name=left_obs_pose,
        ),
        PoseCoordinateFrameTransform(
            target_world=right_target_world,
            pose_world=right_obs_pose,
            transformed_key_name=right_obs_pose,
        ),
        InterpolatePose(
            new_chunk_length=chunk_length,
            action_key=left_cmd_camframe,
            output_action_key=left_cmd_camframe,
            stride=stride,
        ),
        InterpolatePose(
            new_chunk_length=chunk_length,
            action_key=right_cmd_camframe,
            output_action_key=right_cmd_camframe,
            stride=stride,
        ),
        InterpolateLinear(
            new_chunk_length=chunk_length,
            action_key=left_gripper,
            output_action_key=left_gripper,
            stride=stride,
        ),
        InterpolateLinear(
            new_chunk_length=chunk_length,
            action_key=right_gripper,
            output_action_key=right_gripper,
            stride=stride,
        ),
        ConcatKeys(
            key_list=[
                left_cmd_camframe,
                left_gripper,
                right_cmd_camframe,
                right_gripper,
            ],
            new_key_name=actions_key,
            delete_old_keys=True,
        ),
        ConcatKeys(
            key_list=[
                left_obs_pose,
                left_obs_gripper,
                right_obs_pose,
                right_obs_gripper,
            ],
            new_key_name=obs_key,
            delete_old_keys=True,
        ),
        DeleteKeys(
            keys_to_delete=[
                left_cmd_world,
                right_cmd_world,
                left_target_world,
                right_target_world,
            ]
        ),
    ]


def build_aria_bimanual_transform_list(
    *,
    target_world: str = "obs_head_pose",
    target_world_ypr: str = "obs_head_pose_ypr",
    target_world_is_quat: bool = True,
    left_action_world: str = "left.obs_ee_pose",
    right_action_world: str = "right.obs_ee_pose",
    left_obs_pose: str = "left.obs_ee_pose",
    right_obs_pose: str = "right.obs_ee_pose",
    left_action_headframe: str = "left.action_ee_pose_headframe",
    right_action_headframe: str = "right.action_ee_pose_headframe",
    left_obs_headframe: str = "left.obs_ee_pose_headframe",
    right_obs_headframe: str = "right.obs_ee_pose_headframe",
    actions_key: str = "actions_cartesian",
    obs_key: str = "observations.state.ee_pose",
    chunk_length: int = 100,
    stride: int = 3,
    delete_target_world: bool = True,
) -> list[Transform]:
    """Canonical ARIA bimanual transform pipeline used by tests and notebooks.

    Aria human data does not have commanded ee poses; action chunks are built
    from stacked observed ee poses (typically with a horizon on
    ``left/right.action_ee_pose`` mapped from ``left/right.obs_ee_pose``).
    """
    keys_to_delete = list(
        {
            left_action_world,
            right_action_world,
            left_obs_pose,
            right_obs_pose,
        }
    )
    target_pose_key = target_world_ypr if target_world_is_quat else target_world
    if delete_target_world:
        keys_to_delete.append(target_world)
        if target_world_is_quat:
            keys_to_delete.append(target_world_ypr)

    transform_list: list[Transform] = []
    if target_world_is_quat:
        transform_list.append(
            QuaternionPoseToYPR(
                pose_key=target_world,
                output_key=target_world_ypr,
            )
        )

    transform_list.extend(
        [
            ActionChunkCoordinateFrameTransform(
                target_world=target_pose_key,
                chunk_world=left_action_world,
                transformed_key_name=left_action_headframe,
            ),
            ActionChunkCoordinateFrameTransform(
                target_world=target_pose_key,
                chunk_world=right_action_world,
                transformed_key_name=right_action_headframe,
            ),
            PoseCoordinateFrameTransform(
                target_world=target_pose_key,
                pose_world=left_obs_pose,
                transformed_key_name=left_obs_headframe,
            ),
            PoseCoordinateFrameTransform(
                target_world=target_pose_key,
                pose_world=right_obs_pose,
                transformed_key_name=right_obs_headframe,
            ),
            InterpolatePose(
                new_chunk_length=chunk_length,
                action_key=left_action_headframe,
                output_action_key=left_action_headframe,
                stride=stride,
            ),
            InterpolatePose(
                new_chunk_length=chunk_length,
                action_key=right_action_headframe,
                output_action_key=right_action_headframe,
                stride=stride,
            ),
            ConcatKeys(
                key_list=[left_action_headframe, right_action_headframe],
                new_key_name=actions_key,
                delete_old_keys=True,
            ),
            ConcatKeys(
                key_list=[left_obs_headframe, right_obs_headframe],
                new_key_name=obs_key,
                delete_old_keys=True,
            ),
            DeleteKeys(keys_to_delete=keys_to_delete),
        ]
    )
    return transform_list
