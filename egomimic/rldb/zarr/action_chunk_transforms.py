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
from typing import Literal

import numpy as np
import torch
from projectaria_tools.core.sophus import SE3
from scipy.spatial.transform import Rotation as R

from egomimic.utils.pose_utils import (
    _interpolate_euler,
    _interpolate_linear,
    _interpolate_quat_wxyz,
    _interpolate_xyz,
    _matrix_to_xyz,
    _matrix_to_xyzwxyz,
    _matrix_to_xyzypr,
    _xyz_to_matrix,
    _xyzwxyz_to_matrix,
    _xyzypr_to_matrix,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)

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
    """Interpolate a pose chunk of shape (T, 6) or (T, 7)."""

    def __init__(
        self,
        new_chunk_length: int,
        action_key: str,
        output_action_key: str,
        stride: int = 1,
        mode: Literal["xyzwxyz", "xyzypr"] = "xyzwxyz",
    ):
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")
        self.new_chunk_length = new_chunk_length
        self.action_key = action_key
        self.output_action_key = output_action_key
        self.stride = int(stride)
        self.mode = mode

    def transform(self, batch: dict) -> dict:
        actions = np.asarray(batch[self.action_key])
        actions = actions[:: self.stride]
        if self.mode == "xyzwxyz":
            if actions.ndim != 2 or actions.shape[-1] != 7:
                raise ValueError(
                    f"InterpolatePose expects (T, 7) when is_quat=True, got "
                    f"{actions.shape} for key '{self.action_key}'"
                )
            batch[self.output_action_key] = _interpolate_quat_wxyz(
                actions, self.new_chunk_length
            )
        elif self.mode == "xyzypr":
            if actions.ndim != 2 or actions.shape[-1] != 6:
                raise ValueError(
                    f"InterpolatePose expects (T, 6), got {actions.shape} for key "
                    f"'{self.action_key}'"
                )
            batch[self.output_action_key] = _interpolate_euler(
                actions, self.new_chunk_length
            )
        else:
            if actions.shape[-1] != 3:
                raise ValueError(
                    f"InterpolatePose expects (T, 3) or (T, K, 3), got {actions.shape} for key "
                    f"'{self.action_key}'"
                )
            batch[self.output_action_key] = _interpolate_xyz(
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


class ActionChunkCoordinateFrameTransform(Transform):
    def __init__(
        self,
        target_world: str,
        chunk_world: str,
        transformed_key_name: str,
        extra_batch_key: dict = None,
        mode: Literal["xyz", "xyzwxyz", "xyzypr"] = "xyzwxyz",
        inverse: bool = True,
    ):
        """
        args:
            target_world:
            chunk_world:
            transformed_key_name:
            is_quat: if True, inputs are xyz + quat(wxyz); otherwise xyz + ypr.
        """
        self.target_world = target_world
        self.chunk_world = chunk_world
        self.transformed_key_name = transformed_key_name
        self.extra_batch_key = extra_batch_key
        self.mode = mode
        self.inverse = inverse

    def transform(self, batch):
        """
        args:
            batch:
                if is_quat=False, inputs are xyz + ypr.
                if is_quat=True, inputs are xyz + quat(wxyz).
                Input shape validation is delegated to the selected to-matrix helper.
                transformed_key_name: str, name of the new key to store the transformed chunk world in

        returns
            batch with new key containing transformed chunk world in target frame:
                if is_quat=False: (T, 6) xyz + ypr
                if is_quat=True: (T, 7) xyz + quat(wxyz)
        """
        # flatten to (T, D)
        # target world is head pose, chunk world is keypoints
        batch.update(self.extra_batch_key or {})
        target_world = np.asarray(batch[self.target_world])
        chunk_world = np.asarray(batch[self.chunk_world])
        chunk_world_shape = None

        if chunk_world.ndim > 2:
            chunk_world_shape = chunk_world.shape
            chunk_world = chunk_world.reshape(-1, chunk_world_shape[-1])

        to_matrix_fn = None
        if self.mode == "xyzwxyz":
            to_matrix_fn = _xyzwxyz_to_matrix
        elif self.mode == "xyzypr":
            to_matrix_fn = _xyzypr_to_matrix
        elif self.mode == "xyz":
            to_matrix_fn = _xyz_to_matrix
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        target_world_to_matrix_fn = (
            _xyzwxyz_to_matrix if target_world.shape[-1] == 7 else _xyzypr_to_matrix
        )
        # Convert to SE3 for transformation
        target_se3 = SE3.from_matrix(
            target_world_to_matrix_fn(target_world[None, :])[0]
        )  # (4, 4)
        chunk_se3 = SE3.from_matrix(to_matrix_fn(chunk_world))  # (T, 4, 4)

        # Compute relative transform and apply to chunk
        if self.inverse:
            chunk_in_target_frame = target_se3.inverse() @ chunk_se3
        else:
            chunk_in_target_frame = target_se3 @ chunk_se3
        chunk_mats = chunk_in_target_frame.to_matrix()
        if chunk_mats.ndim == 2:
            chunk_mats = chunk_mats[None, ...]

        if self.mode == "xyzwxyz":
            chunk_in_target_frame = _matrix_to_xyzwxyz(chunk_mats)
        elif self.mode == "xyzypr":
            chunk_in_target_frame = _matrix_to_xyzypr(chunk_mats)
        elif self.mode == "xyz":
            chunk_in_target_frame = _matrix_to_xyz(chunk_mats)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        if chunk_world_shape is not None:
            chunk_in_target_frame = chunk_in_target_frame.reshape(*chunk_world_shape)

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
        xyzw = wxyz_to_xyzw(pose[3:7])
        ypr = R.from_quat(xyzw).as_euler("ZYX", degrees=False)
        batch[self.output_key] = np.concatenate([xyz, ypr], axis=0)
        return batch


class YPRToQuaternionPose(Transform):
    """Convert a single pose from xyz + ypr to xyz + quat(x,y,z,w)."""

    def __init__(self, pose_key: str, output_key: str):
        self.pose_key = pose_key
        self.output_key = output_key

    def transform(self, batch: dict) -> dict:
        pose = np.asarray(batch[self.pose_key])
        if pose.shape != (6,):
            raise ValueError(
                f"YPRToQuaternionPose expects shape (6,), got {pose.shape} for key "
                f"'{self.pose_key}'"
            )
        xyz = pose[:3]
        quat = R.from_euler("ZYX", pose[3:6], degrees=False).as_quat()  # (x,y,z,w)
        quat = xyzw_to_wxyz(quat)
        batch[self.output_key] = np.concatenate([xyz, quat], axis=0)
        return batch


class BatchQuaternionPoseToYPR(Transform):
    """Convert a batch of poses from xyz + quat(x,y,z,w) to xyz + ypr."""

    def __init__(self, pose_key: str, output_key: str):
        self.pose_key = pose_key
        self.output_key = output_key

    def transform(self, batch: dict) -> dict:
        pose = np.asarray(batch[self.pose_key])
        if pose.ndim != 2 or pose.shape[-1] != 7:
            raise ValueError(
                f"BatchQuaternionPoseToYPR expects shape (N, 7), got {pose.shape} for key "
                f"'{self.pose_key}'"
            )
        xyz = pose[:, :3]
        xyzw = wxyz_to_xyzw(pose[:, 3:7])
        ypr = R.from_quat(xyzw).as_euler("ZYX", degrees=False)  # (N, 3)
        batch[self.output_key] = np.concatenate([xyz, ypr], axis=1)
        return batch


class BatchYPRToQuaternionPose(Transform):
    """Convert a batch of poses from xyz + ypr to xyz + quat(x,y,z,w)."""

    def __init__(self, pose_key: str, output_key: str):
        self.pose_key = pose_key
        self.output_key = output_key

    def transform(self, batch: dict) -> dict:
        pose = np.asarray(batch[self.pose_key])
        if pose.ndim != 2 or pose.shape[-1] != 6:
            raise ValueError(
                f"BatchYPRToQuaternionPose expects shape (N, 6), got {pose.shape} for key "
                f"'{self.pose_key}'"
            )
        xyz = pose[:, :3]
        quat = R.from_euler("ZYX", pose[:, 3:6], degrees=False).as_quat()  # (N, 4)
        quat = xyzw_to_wxyz(quat)
        batch[self.output_key] = np.concatenate([xyz, quat], axis=1)
        return batch


class PoseCoordinateFrameTransform(Transform):
    """Transform a single pose into a target frame pose."""

    def __init__(
        self,
        target_world: str,
        pose_world: str,
        transformed_key_name: str,
        mode: Literal["xyzwxyz", "xyzypr", "xyz"] = "xyzwxyz",
    ):
        self.target_world = target_world
        self.pose_world = pose_world
        self.transformed_key_name = transformed_key_name
        self.mode = mode
        self._chunk_transform = ActionChunkCoordinateFrameTransform(
            target_world=target_world,
            chunk_world=pose_world,
            transformed_key_name=transformed_key_name,
            mode=mode,
        )

    def transform(self, batch: dict) -> dict:
        pose_world = np.asarray(batch[self.pose_world])
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


class XYZWXYZ_to_XYZYPR(Transform):
    """Convert listed keys from xyz+quat(wxyz) to xyz+ypr in-place."""

    def __init__(self, keys: list[str]):
        self.keys = list(keys)

    def transform(self, batch: dict) -> dict:
        for key in self.keys:
            value = np.asarray(batch[key])
            if value.ndim == 1 and value.shape[0] == 7:
                batch[key] = _matrix_to_xyzypr(_xyzwxyz_to_matrix(value[None, :]))[0]
            elif value.ndim == 2 and value.shape[1] == 7:
                batch[key] = _matrix_to_xyzypr(_xyzwxyz_to_matrix(value))
            else:
                raise ValueError(
                    f"XYZWXYZ_to_XYZYPR expects key '{key}' to have shape (7,) "
                    f"or (T, 7), got {value.shape}"
                )
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
class SplitKeys(Transform):
    def __init__(self, input_key: str, output_key_list: list[(str, int)]):
        self.input_key = input_key
        self.output_key_list = list(output_key_list)

    def transform(self, batch: dict) -> dict:
        prev_end = 0
        for key, size in self.output_key_list:
            batch[key] = batch[self.input_key][..., prev_end : prev_end + size]
            prev_end += size
        return batch


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


class PadGripperZeros(Transform):
    """Pad a 12D bimanual cartesian action chunk to 14D by inserting a zero
    gripper slot at position 6 (end of left arm) and position 13 (end of right
    arm), matching the canonical [L xyz ypr g, R xyz ypr g] layout used by Eva.

    Used so aria (which has no gripper signal) can share an FM denoiser head
    sized for 14D actions without needing in-model padding branches.
    """

    def __init__(self, action_key: str = "actions_cartesian"):
        self.action_key = action_key

    def transform(self, batch: dict) -> dict:
        actions = batch[self.action_key]
        is_tensor = isinstance(actions, torch.Tensor)
        arr = actions.cpu().numpy() if is_tensor else np.asarray(actions)
        if arr.shape[-1] != 12:
            raise ValueError(
                f"PadGripperZeros expects last-dim 12, got {arr.shape} for "
                f"'{self.action_key}'"
            )
        pad_shape = (*arr.shape[:-1], 1)
        pad = np.zeros(pad_shape, dtype=arr.dtype)
        padded = np.concatenate((arr[..., :6], pad, arr[..., 6:], pad), axis=-1)
        batch[self.action_key] = torch.from_numpy(padded) if is_tensor else padded
        return batch


class Reshape(Transform):
    def __init__(self, input_key: str, output_key: str, shape: tuple):
        self.input_key = input_key
        self.output_key = output_key
        self.shape = shape

    def transform(self, batch: dict) -> dict:
        batch[self.output_key] = batch[self.input_key].reshape(*self.shape)
        return batch


# ---------------------------------------------------------------------------
# Type Transforms
# ---------------------------------------------------------------------------


class NumpyToTensor(Transform):
    def __init__(self, keys: list[str]):
        self.keys = keys

    def transform(self, batch: dict) -> dict:
        for key in self.keys:
            if isinstance(batch[key], np.ndarray):
                batch[key] = torch.from_numpy(batch[key])
            elif isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].clone()
            else:
                raise ValueError(
                    f"NumpyToTensor expects key '{key}' to be a numpy array or torch tensor, got {type(batch[key])}"
                )
        return batch


def _to_float64_numpy(value) -> np.ndarray:
    """Detach a numeric value into a NumPy compute buffer.

    NumPy has no native ``bfloat16`` dtype, so crossing that boundary directly
    raises ``TypeError``. Route BF16 tensors through CPU FP32 first, then use
    FP64 for deterministic planar-geometry computation. Other tensor dtypes
    keep their normal NumPy conversion before the common FP64 compute cast.
    """
    if torch.is_tensor(value):
        cpu_value = value.detach().cpu()
        if cpu_value.dtype == torch.bfloat16:
            cpu_value = cpu_value.float()
        value = cpu_value.numpy()
    return np.asarray(value, dtype=np.float64)


def _restore_numeric_type(value: np.ndarray, template):
    """Restore a NumPy compute result to ``template``'s type/device/dtype."""
    value = np.ascontiguousarray(value)
    if torch.is_tensor(template):
        dtype = template.dtype if template.is_floating_point() else torch.float32
        return torch.from_numpy(value).to(device=template.device, dtype=dtype)
    template_array = np.asarray(template)
    dtype = (
        template_array.dtype
        if np.issubdtype(template_array.dtype, np.floating)
        else np.float32
    )
    return value.astype(dtype, copy=False)


class RequireLastDim(Transform):
    """Fail early when a present array does not have the configured width."""

    def __init__(self, keys: list[str], width: int):
        self.keys = list(keys)
        self.width = int(width)
        if self.width <= 0:
            raise ValueError("width must be positive")

    def transform(self, batch: dict) -> dict:
        for key in self.keys:
            if key not in batch:
                continue
            value = _to_float64_numpy(batch[key])
            if value.ndim == 0 or value.shape[-1] != self.width:
                raise ValueError(
                    f"RequireLastDim expects key '{key}' to have last dimension "
                    f"{self.width}, got {value.shape}"
                )
            if not np.isfinite(value).all():
                raise ValueError(f"RequireLastDim found non-finite values in '{key}'")
        return batch


class ChainGripperNative4ToPoints6(Transform):
    """Map native ``[x, y, theta, grip]`` chunks to ordered ``[L, C, R]``.

    This is the ChainGripper training-boundary analogue of
    :class:`ThetaToRotVec`: datasets stay in their native control space, while
    the model target is derived at load time. Geometry and clipping rules are
    delegated to the authoritative Sim V2 implementation.
    """

    def __init__(self, keys: list[str], world_size: float = 512.0):
        self.keys = list(keys)
        self.world_size = float(world_size)
        if self.world_size <= 0.0:
            raise ValueError("world_size must be positive")

    def transform(self, batch: dict) -> dict:
        from Tsimulation.sim_v2.pushshapes.chain_gripper_control import (
            pose_control_to_points,
        )

        for key in self.keys:
            if key not in batch:
                continue
            template = batch[key]
            controls = _to_float64_numpy(template)
            if controls.ndim == 0 or controls.shape[-1] != 4:
                raise ValueError(
                    "ChainGripperNative4ToPoints6 expects key "
                    f"'{key}' to have last dimension 4, got {controls.shape}"
                )
            points = pose_control_to_points(controls, world_size=self.world_size)
            batch[key] = _restore_numeric_type(points, template)
        return batch


class ChainGripperPoints6ToNative4(Transform):
    """Sequentially project ordered ChainGripper points to native controls.

    Arbitrary model predictions generally do not lie on the four-dimensional
    ChainGripper kinematic manifold. Each trajectory is therefore projected
    one timestep at a time through Sim V2's constrained IK. A prior native
    control, or the latest ``x, y, theta`` from rollout state, seeds the first
    timestep; subsequent timesteps use their own previous projection. When a
    point pair is degenerate, temporal orientation is retained explicitly.

    Diagnostics from the most recent call are available through
    ``last_projection_diagnostics`` and ``projection_diagnostics_by_key``.
    """

    def __init__(
        self,
        keys: list[str],
        world_size: float = 512.0,
        grid_size: int = 33,
        refinements: int = 6,
        context_state_key: str = "state_agent_obj",
        previous_control_key: str = "previous_control",
    ):
        self.keys = list(keys)
        self.world_size = float(world_size)
        self.grid_size = int(grid_size)
        self.refinements = int(refinements)
        self.context_state_key = str(context_state_key)
        self.previous_control_key = str(previous_control_key)
        self.last_projection_diagnostics: dict | None = None
        self.projection_diagnostics_by_key: dict[str, dict] = {}
        if self.world_size <= 0.0:
            raise ValueError("world_size must be positive")
        if self.grid_size < 5 or self.grid_size % 2 == 0:
            raise ValueError("grid_size must be an odd integer >= 5")
        if self.refinements < 1:
            raise ValueError("refinements must be >= 1")

    @staticmethod
    def _context_rows(value, trajectory_count: int, *, key: str) -> np.ndarray:
        array = _to_float64_numpy(value)
        if array.ndim == 0:
            raise ValueError(f"Rollout context {key!r} must be an array")
        width = array.shape[-1]
        rows = array.reshape(-1, width)
        if rows.shape[0] % trajectory_count != 0:
            raise ValueError(
                f"Rollout context {key!r} shape {array.shape} cannot be aligned "
                f"to {trajectory_count} predicted trajectories"
            )
        return rows.reshape(trajectory_count, -1, width)[:, -1]

    def _initial_previous_controls(
        self,
        batch: dict,
        trajectory_count: int,
    ) -> np.ndarray | None:
        if self.previous_control_key in batch:
            rows = self._context_rows(
                batch[self.previous_control_key],
                trajectory_count,
                key=self.previous_control_key,
            )
            if rows.shape[-1] != 4:
                raise ValueError(
                    f"Rollout context {self.previous_control_key!r} must have "
                    f"last dimension 4, got {rows.shape}"
                )
            return rows
        if self.context_state_key not in batch:
            return None
        rows = self._context_rows(
            batch[self.context_state_key],
            trajectory_count,
            key=self.context_state_key,
        )
        if rows.shape[-1] < 3:
            raise ValueError(
                f"Rollout context {self.context_state_key!r} must contain x, y, "
                f"theta, got {rows.shape}"
            )
        previous = np.zeros((trajectory_count, 4), dtype=np.float64)
        previous[:, :3] = rows[:, :3]
        return previous

    def _project(
        self,
        points: np.ndarray,
        batch: dict,
    ) -> tuple[np.ndarray, dict]:
        from Tsimulation.sim_v2.pushshapes.chain_gripper_control import (
            pose_control_to_points,
            project_points_to_pose_control,
        )

        if points.ndim < 2 or points.shape[-1] != 6:
            raise ValueError(
                "ChainGripperPoints6ToNative4 expects (..., horizon, 6), "
                f"got {points.shape}"
            )
        trajectory_shape = points.shape[:-2]
        horizon = points.shape[-2]
        trajectory_count = int(np.prod(trajectory_shape)) if trajectory_shape else 1
        trajectories = points.reshape(trajectory_count, horizon, 6)
        controls = np.empty((trajectory_count, horizon, 4), dtype=np.float64)
        point_rmse = np.empty((trajectory_count, horizon), dtype=np.float64)
        wrong_chirality = np.empty((trajectory_count, horizon), dtype=bool)
        degenerate = np.empty((trajectory_count, horizon), dtype=bool)
        used_exact_inverse = np.empty((trajectory_count, horizon), dtype=bool)
        initial_previous = self._initial_previous_controls(batch, trajectory_count)

        for trajectory_index, trajectory in enumerate(trajectories):
            previous = (
                None if initial_previous is None else initial_previous[trajectory_index]
            )
            for time_index, predicted_points in enumerate(trajectory):
                projection = project_points_to_pose_control(
                    predicted_points,
                    previous_control=previous,
                    world_size=self.world_size,
                    grid_size=self.grid_size,
                    refinements=self.refinements,
                )
                control = np.asarray(projection.control, dtype=np.float64)
                is_degenerate = bool(projection.degenerate)
                if previous is not None and is_degenerate:
                    control = control.copy()
                    control[2] = previous[2]
                    fitted_points = pose_control_to_points(
                        control,
                        world_size=self.world_size,
                    )
                    projection_rmse = float(
                        np.sqrt(np.mean(np.square(fitted_points - predicted_points)))
                    )
                else:
                    projection_rmse = float(projection.point_rmse)
                controls[trajectory_index, time_index] = control
                point_rmse[trajectory_index, time_index] = projection_rmse
                wrong_chirality[trajectory_index, time_index] = bool(
                    projection.wrong_chirality
                )
                degenerate[trajectory_index, time_index] = is_degenerate
                used_exact_inverse[trajectory_index, time_index] = bool(
                    projection.used_exact_inverse
                )
                previous = control

        diagnostic_shape = (*trajectory_shape, horizon)
        diagnostics = {
            "point_rmse": point_rmse.reshape(diagnostic_shape),
            "wrong_chirality": wrong_chirality.reshape(diagnostic_shape),
            "degenerate": degenerate.reshape(diagnostic_shape),
            "used_exact_inverse": used_exact_inverse.reshape(diagnostic_shape),
            "mean_point_rmse": float(np.mean(point_rmse)),
            "max_point_rmse": float(np.max(point_rmse)),
            "wrong_chirality_count": int(np.count_nonzero(wrong_chirality)),
            "degenerate_count": int(np.count_nonzero(degenerate)),
            "trajectory_count": trajectory_count,
            "horizon": horizon,
        }
        return controls.reshape(*points.shape[:-1], 4), diagnostics

    def transform(self, batch: dict) -> dict:
        self.last_projection_diagnostics = None
        self.projection_diagnostics_by_key = {}
        for key in self.keys:
            if key not in batch:
                continue
            template = batch[key]
            points = _to_float64_numpy(template)
            controls, diagnostics = self._project(points, batch)
            batch[key] = _restore_numeric_type(controls, template)
            self.projection_diagnostics_by_key[key] = diagnostics
            self.last_projection_diagnostics = diagnostics
        return batch


class ThetaToRotVec(Transform):
    """Replace a planar angle with its continuous ``(cos, sin)`` encoding.

    For an input whose last dimension is ``[..., x, y, theta, ...]``, the
    angle at ``angle_col`` becomes two values and the output gains one column.
    Missing optional keys are ignored, while malformed present keys fail
    loudly so action-shape mistakes cannot silently reach normalization.
    """

    def __init__(self, keys: list[str], angle_col: int = 2):
        self.keys = list(keys)
        self.angle_col = int(angle_col)
        if self.angle_col < 0:
            raise ValueError("angle_col must be non-negative")

    def transform(self, batch: dict) -> dict:
        for key in self.keys:
            if key not in batch:
                continue
            value = batch[key]
            if torch.is_tensor(value):
                if value.ndim == 0 or value.shape[-1] <= self.angle_col:
                    raise ValueError(
                        f"ThetaToRotVec needs angle_col={self.angle_col} in key "
                        f"'{key}', got shape {tuple(value.shape)}"
                    )
                theta = value[..., self.angle_col]
                batch[key] = torch.cat(
                    [
                        value[..., : self.angle_col],
                        torch.cos(theta).unsqueeze(-1),
                        torch.sin(theta).unsqueeze(-1),
                        value[..., self.angle_col + 1 :],
                    ],
                    dim=-1,
                )
                continue

            value = np.asarray(value)
            if value.ndim == 0 or value.shape[-1] <= self.angle_col:
                raise ValueError(
                    f"ThetaToRotVec needs angle_col={self.angle_col} in key "
                    f"'{key}', got shape {value.shape}"
                )
            theta = value[..., self.angle_col]
            batch[key] = np.concatenate(
                [
                    value[..., : self.angle_col],
                    np.cos(theta)[..., None].astype(value.dtype, copy=False),
                    np.sin(theta)[..., None].astype(value.dtype, copy=False),
                    value[..., self.angle_col + 1 :],
                ],
                axis=-1,
            )
        return batch


class PlanarAgentStateToRotVec4(Transform):
    """Encode an observed ``[agent x,y,theta,...]`` pose as rotvec4.

    Unlike :class:`ThetaToRotVec`, this deliberately drops trailing object
    state. It is the exact U-Socket proprio contract, not an action transform.
    """

    def __init__(
        self,
        keys: list[str],
        angle_col: int = 2,
        source_dim: int = 6,
    ):
        self.keys = list(keys)
        self.angle_col = int(angle_col)
        self.source_dim = int(source_dim)
        if self.angle_col != 2:
            raise ValueError("PlanarAgentStateToRotVec4 requires angle_col=2")
        if self.source_dim < 3:
            raise ValueError("PlanarAgentStateToRotVec4 source_dim must be >= 3")

    def transform(self, batch: dict) -> dict:
        for key in self.keys:
            if key not in batch:
                continue
            value = batch[key]
            if torch.is_tensor(value):
                if value.ndim == 0 or value.shape[-1] != self.source_dim:
                    raise ValueError(
                        f"PlanarAgentStateToRotVec4 needs exact source width "
                        f"{self.source_dim} in key '{key}', got shape "
                        f"{tuple(value.shape)}"
                    )
                theta = value[..., 2]
                batch[key] = torch.cat(
                    [
                        value[..., :2],
                        torch.cos(theta).unsqueeze(-1),
                        torch.sin(theta).unsqueeze(-1),
                    ],
                    dim=-1,
                )
                continue

            value = np.asarray(value)
            if value.ndim == 0 or value.shape[-1] != self.source_dim:
                raise ValueError(
                    f"PlanarAgentStateToRotVec4 needs exact source width "
                    f"{self.source_dim} in key '{key}', got shape {value.shape}"
                )
            theta = value[..., 2]
            batch[key] = np.concatenate(
                [
                    value[..., :2],
                    np.cos(theta)[..., None].astype(value.dtype, copy=False),
                    np.sin(theta)[..., None].astype(value.dtype, copy=False),
                ],
                axis=-1,
            )
        return batch


class RotVecToTheta(Transform):
    """Invert :class:`ThetaToRotVec` with ``atan2(sin, cos)``.

    The pair does not need to be unit length; only its direction determines
    the reconstructed angle.  This is useful for model predictions before a
    U-socket simulator step.
    """

    def __init__(self, keys: list[str], angle_col: int = 2):
        self.keys = list(keys)
        self.angle_col = int(angle_col)
        if self.angle_col < 0:
            raise ValueError("angle_col must be non-negative")

    def transform(self, batch: dict) -> dict:
        for key in self.keys:
            if key not in batch:
                continue
            value = batch[key]
            if torch.is_tensor(value):
                if value.ndim == 0 or value.shape[-1] <= self.angle_col + 1:
                    raise ValueError(
                        f"RotVecToTheta needs cos/sin at columns "
                        f"{self.angle_col}:{self.angle_col + 2} in key '{key}', "
                        f"got shape {tuple(value.shape)}"
                    )
                theta = torch.atan2(
                    value[..., self.angle_col + 1], value[..., self.angle_col]
                )
                batch[key] = torch.cat(
                    [
                        value[..., : self.angle_col],
                        theta.unsqueeze(-1),
                        value[..., self.angle_col + 2 :],
                    ],
                    dim=-1,
                )
                continue

            value = np.asarray(value)
            if value.ndim == 0 or value.shape[-1] <= self.angle_col + 1:
                raise ValueError(
                    f"RotVecToTheta needs cos/sin at columns "
                    f"{self.angle_col}:{self.angle_col + 2} in key '{key}', "
                    f"got shape {value.shape}"
                )
            theta = np.arctan2(
                value[..., self.angle_col + 1], value[..., self.angle_col]
            )
            batch[key] = np.concatenate(
                [
                    value[..., : self.angle_col],
                    theta[..., None].astype(value.dtype, copy=False),
                    value[..., self.angle_col + 2 :],
                ],
                axis=-1,
            )
        return batch
