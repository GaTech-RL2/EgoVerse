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
