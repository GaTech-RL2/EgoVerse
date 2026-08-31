"""PushShapes-specific adapters for Pipeline policy rollout."""

from __future__ import annotations

import numpy as np
import torch

from egomimic.rldb.zarr.action_chunk_transforms import (
    ChainGripperPoints6ToNative4,
    _restore_numeric_type,
    _to_float64_numpy,
)
from egomimic.rldb.zarr.arc_length_tokenizer import (
    CHAIN_GRIPPER_POINT_ARC_DIM,
    PLANAR_ARC_DIM,
    USOCKET_ARC_DIM,
    TokenizeChainGripperPointArcLength,
    TokenizeUSocketArcLength,
)


class USocketRotVecRolloutAdapter:
    """Decode ``[x, y, cos(theta), sin(theta)]`` into simulator actions."""

    preserves_decoded_timing = True

    def decode(self, actions, context: dict | None = None):
        del context
        if torch.is_tensor(actions):
            if actions.ndim < 2 or actions.shape[-1] != 4:
                raise ValueError(
                    "USocketRotVecRolloutAdapter expects (..., 4) actions, "
                    f"got {tuple(actions.shape)}"
                )
            theta = torch.atan2(actions[..., 3], actions[..., 2])
            return torch.cat((actions[..., :2], theta.unsqueeze(-1)), dim=-1)

        value = np.asarray(actions)
        if value.ndim < 2 or value.shape[-1] != 4:
            raise ValueError(
                "USocketRotVecRolloutAdapter expects (..., 4) actions, "
                f"got {value.shape}"
            )
        theta = np.arctan2(value[..., 3], value[..., 2])
        return np.concatenate((value[..., :2], theta[..., None]), axis=-1)

    __call__ = decode


def _planar_common5_to_native(actions, *, native_action_dim: int, adapter: str):
    """Decode ``[x, y, cos(theta), sin(theta), grip]`` without changing time."""
    if native_action_dim not in (2, 3, 4):
        raise ValueError("native_action_dim must be one of 2, 3, or 4")
    if torch.is_tensor(actions):
        if actions.ndim < 2 or actions.shape[-1] != 5:
            raise ValueError(
                f"{adapter} expects (..., 5) common-planar actions, "
                f"got {tuple(actions.shape)}"
            )
        theta = torch.atan2(actions[..., 3], actions[..., 2])
        native = torch.cat(
            (actions[..., :2], theta.unsqueeze(-1), actions[..., 4:5]), dim=-1
        )
        return native[..., :native_action_dim]

    value = np.asarray(actions)
    if value.ndim < 2 or value.shape[-1] != 5:
        raise ValueError(
            f"{adapter} expects (..., 5) common-planar actions, got {value.shape}"
        )
    theta = np.arctan2(value[..., 3], value[..., 2])
    native = np.concatenate(
        (value[..., :2], theta[..., None], value[..., 4:5]), axis=-1
    )
    return native[..., :native_action_dim]


class PlanarCommon5RolloutAdapter:
    """Decode a fixed-rate common-planar chunk into native simulator controls."""

    preserves_decoded_timing = True

    def __init__(self, action_horizon: int, native_action_dim: int):
        self.action_horizon = int(action_horizon)
        self.native_action_dim = int(native_action_dim)
        if self.action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        if self.native_action_dim not in (2, 3, 4):
            raise ValueError("native_action_dim must be one of 2, 3, or 4")

    def decode(self, actions, context: dict | None = None):
        del context
        if torch.is_tensor(actions):
            value = actions.unsqueeze(0) if actions.ndim == 2 else actions
            expected = (self.action_horizon, 5)
            if value.ndim != 3 or tuple(value.shape[1:]) != expected:
                raise ValueError(
                    "PlanarCommon5RolloutAdapter expects "
                    f"(B, {expected[0]}, {expected[1]}), got {tuple(value.shape)}"
                )
        else:
            value = np.asarray(actions)
            value = value[None] if value.ndim == 2 else value
            expected = (self.action_horizon, 5)
            if value.ndim != 3 or tuple(value.shape[1:]) != expected:
                raise ValueError(
                    "PlanarCommon5RolloutAdapter expects "
                    f"(B, {expected[0]}, {expected[1]}), got {value.shape}"
                )
        return _planar_common5_to_native(
            value,
            native_action_dim=self.native_action_dim,
            adapter="PlanarCommon5RolloutAdapter",
        )

    __call__ = decode


class PlanarArcWaypointZeroRolloutAdapter:
    """Decode the anchored planar waypoint and replan every control step.

    The generic v2 token stores ``M`` common-layout waypoints followed by one
    timing row. A full fixed-rate inverse is intentionally not inferred from
    that scalar timing payload. Rollout therefore executes waypoint zero and
    asks the policy for a fresh token at the next simulator step.
    """

    preserves_decoded_timing = True

    def __init__(self, resampled_vector_length: int, native_action_dim: int):
        self.resampled_vector_length = int(resampled_vector_length)
        self.native_action_dim = int(native_action_dim)
        self.action_horizon = 1
        if self.resampled_vector_length < 2:
            raise ValueError("resampled_vector_length must be at least 2")
        if self.native_action_dim not in (2, 3, 4):
            raise ValueError("native_action_dim must be one of 2, 3, or 4")

    def decode(self, actions, context: dict | None = None):
        del context
        expected = (self.resampled_vector_length + 1, PLANAR_ARC_DIM)
        if torch.is_tensor(actions):
            value = actions.unsqueeze(0) if actions.ndim == 2 else actions
            if value.ndim != 3 or tuple(value.shape[1:]) != expected:
                raise ValueError(
                    "PlanarArcWaypointZeroRolloutAdapter expects "
                    f"(B, {expected[0]}, {expected[1]}), got {tuple(value.shape)}"
                )
            return _planar_common5_to_native(
                value[:, :1],
                native_action_dim=self.native_action_dim,
                adapter="PlanarArcWaypointZeroRolloutAdapter",
            )

        value = np.asarray(actions)
        value = value[None] if value.ndim == 2 else value
        if value.ndim != 3 or tuple(value.shape[1:]) != expected:
            raise ValueError(
                "PlanarArcWaypointZeroRolloutAdapter expects "
                f"(B, {expected[0]}, {expected[1]}), got {value.shape}"
            )
        return _planar_common5_to_native(
            value[:, :1],
            native_action_dim=self.native_action_dim,
            adapter="PlanarArcWaypointZeroRolloutAdapter",
        )

    __call__ = decode


class USocketArcLengthRolloutAdapter:
    """Decode planar U-socket arc tokens into fixed-rate simulator actions."""

    preserves_decoded_timing = True

    def __init__(
        self,
        min_distance_unit: float = 200.0,
        resampled_vector_length: int = 25,
        action_horizon: int = 100,
        dt: float = 1.0 / 30.0,
        rotation_radius: float = 40.0,
    ):
        self.resampled_vector_length = int(resampled_vector_length)
        self.action_horizon = int(action_horizon)
        self.detokenizer = TokenizeUSocketArcLength(
            min_distance_unit=min_distance_unit,
            resampled_vector_length=self.resampled_vector_length,
            dt=dt,
            rotation_radius=rotation_radius,
        )

    def decode(self, actions, context: dict | None = None):
        del context
        value = _to_float64_numpy(actions)
        if value.ndim == 2:
            value = value[None]
        expected = (self.resampled_vector_length + 1, USOCKET_ARC_DIM)
        if value.ndim != 3 or tuple(value.shape[1:]) != expected:
            raise ValueError(
                "USocketArcLengthRolloutAdapter expects "
                f"(B, {expected[0]}, {expected[1]}), got {value.shape}"
            )
        decoded = np.stack(
            [self.detokenizer.detokenize(token, self.action_horizon) for token in value]
        )
        return _restore_numeric_type(decoded, actions)

    __call__ = decode


class ChainGripperPointRolloutAdapter:
    """Apply the reusable constrained-IK revert transform at rollout."""

    preserves_decoded_timing = True

    def __init__(
        self,
        action_horizon: int = 16,
        world_size: float = 512.0,
        grid_size: int = 33,
        refinements: int = 6,
        context_state_key: str = "state_agent_obj",
        previous_control_key: str = "previous_control",
    ):
        self.action_horizon = int(action_horizon)
        if self.action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        self.revert_transform = ChainGripperPoints6ToNative4(
            keys=["actions"],
            world_size=world_size,
            grid_size=grid_size,
            refinements=refinements,
            context_state_key=context_state_key,
            previous_control_key=previous_control_key,
        )

    @property
    def last_projection_diagnostics(self) -> dict | None:
        return self.revert_transform.last_projection_diagnostics

    def decode(self, actions, context: dict | None = None):
        batch = dict(context or {})
        batch["actions"] = actions
        return self.revert_transform.transform(batch)["actions"]

    __call__ = decode


class ChainGripperPointArcLengthRolloutAdapter:
    """Detokenize point arcs at fixed rate, then IK-project to native control."""

    preserves_decoded_timing = True

    def __init__(
        self,
        min_distance_unit: float = 200.0,
        resampled_vector_length: int = 25,
        action_horizon: int = 100,
        dt: float = 1.0 / 30.0,
        world_size: float = 512.0,
        grid_size: int = 33,
        refinements: int = 6,
        context_state_key: str = "state_agent_obj",
        previous_control_key: str = "previous_control",
    ):
        self.resampled_vector_length = int(resampled_vector_length)
        self.action_horizon = int(action_horizon)
        self.detokenizer = TokenizeChainGripperPointArcLength(
            min_distance_unit=min_distance_unit,
            resampled_vector_length=self.resampled_vector_length,
            dt=dt,
        )
        self.point_adapter = ChainGripperPointRolloutAdapter(
            action_horizon=self.action_horizon,
            world_size=world_size,
            grid_size=grid_size,
            refinements=refinements,
            context_state_key=context_state_key,
            previous_control_key=previous_control_key,
        )

    @property
    def last_projection_diagnostics(self) -> dict | None:
        return self.point_adapter.last_projection_diagnostics

    def decode(self, actions, context: dict | None = None):
        value = _to_float64_numpy(actions)
        if value.ndim == 2:
            value = value[None]
        expected = (
            self.resampled_vector_length + 1,
            CHAIN_GRIPPER_POINT_ARC_DIM,
        )
        if value.ndim != 3 or tuple(value.shape[1:]) != expected:
            raise ValueError(
                "ChainGripperPointArcLengthRolloutAdapter expects "
                f"(B, {expected[0]}, {expected[1]}), got {value.shape}"
            )
        points = np.stack(
            [self.detokenizer.detokenize(token, self.action_horizon) for token in value]
        )
        decoded = self.point_adapter.decode(points, context=context)
        return _restore_numeric_type(decoded, actions)

    __call__ = decode
