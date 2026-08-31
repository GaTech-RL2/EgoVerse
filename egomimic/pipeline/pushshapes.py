"""PushShapes-specific adapters for Pipeline policy rollout."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from egomimic.rldb.zarr.action_chunk_transforms import (
    ChainGripperPoints6ToNative4,
    PlanarAgentStateToRotVec4,
    _restore_numeric_type,
    _to_float64_numpy,
)
from egomimic.rldb.zarr.arc_length_tokenizer import (
    CHAIN_GRIPPER_POINT_ARC_DIM,
    USOCKET_ARC_DIM,
    TokenizeChainGripperPointArcLength,
    TokenizeUSocketArcLength,
)
from Tsimulation.sim_v2.pushshapes.chain_gripper_constants import (
    CHAIN_GRIPPER_CLOSED_ANGLE,
    CHAIN_GRIPPER_LINK_LEN,
    CHAIN_GRIPPER_OPEN_ANGLE,
)


class USocketModelStateObservationAdapter:
    """Add U-Socket rotvec4 model proprio while preserving native state."""

    def __init__(
        self,
        raw_state_key: str = "state_agent_obj",
        model_state_key: str = "state_agent_model",
    ):
        self.raw_state_key = str(raw_state_key)
        self.model_state_key = str(model_state_key)
        self.transform = PlanarAgentStateToRotVec4(keys=[self.model_state_key])

    def encode(self, batch: dict) -> dict:
        out = dict(batch)
        if self.raw_state_key not in out:
            raise KeyError(f"Missing raw U-Socket state {self.raw_state_key!r}")
        out[self.model_state_key] = out[self.raw_state_key]
        return self.transform.transform(out)


class ChainModelStateObservationAdapter:
    """Add Chain raw6 model proprio while preserving native IK context."""

    def __init__(
        self,
        raw_state_key: str = "state_agent_obj",
        model_state_key: str = "state_agent_model",
    ):
        self.raw_state_key = str(raw_state_key)
        self.model_state_key = str(model_state_key)

    def encode(self, batch: dict) -> dict:
        out = dict(batch)
        if self.raw_state_key not in out:
            raise KeyError(f"Missing raw Chain state {self.raw_state_key!r}")
        out[self.model_state_key] = out[self.raw_state_key]
        return out


class USocketRotVecRolloutAdapter:
    """Decode ``[x, y, cos(theta), sin(theta)]`` into simulator actions."""

    preserves_decoded_timing = True
    native_angle_index = 2

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


class USocketRotVecActionCanonicalizer(nn.Module):
    """Project physical U-Socket rotvec4 tokens onto the unit circle.

    Rollout converts the final two coordinates with ``atan2``. Their radius is
    therefore unobservable to the simulator and must not be rewarded as an
    independent source of sample diversity by a distributional objective.
    """

    action_dim = 4

    def __init__(self, world_size: float = 512.0, epsilon: float = 1e-8):
        super().__init__()
        self.world_size = float(world_size)
        self.epsilon = float(epsilon)
        if self.world_size <= 0.0:
            raise ValueError("world_size must be positive")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive")

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.ndim == 0 or actions.shape[-1] != self.action_dim:
            raise ValueError(
                "USocketRotVecActionCanonicalizer expects (..., 4), got "
                f"{tuple(actions.shape)}"
            )
        value = actions.float()
        rotvec = value[..., 2:4]
        radius = torch.linalg.vector_norm(rotvec, dim=-1, keepdim=True)
        safe_radius = radius.clamp_min(self.epsilon)
        unit = rotvec / safe_radius
        fallback = torch.zeros_like(unit)
        fallback[..., 0] = 1.0
        unit = torch.where(radius > self.epsilon, unit, fallback)
        xy = value[..., :2].clamp(0.0, self.world_size)
        return torch.cat((xy, unit), dim=-1)

    def diagnostics(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        if actions.ndim == 0 or actions.shape[-1] != self.action_dim:
            raise ValueError(
                "USocketRotVecActionCanonicalizer expects (..., 4), got "
                f"{tuple(actions.shape)}"
            )
        radius = torch.linalg.vector_norm(actions.float()[..., 2:4], dim=-1)
        return {
            "raw_rotvec_radius_min": radius.min(),
            "raw_rotvec_radius_median": radius.median(),
            "raw_rotvec_radius_mean": radius.mean(),
        }


class ChainGripperPointActionCanonicalizer(nn.Module):
    """Differentiably map physical point6 tokens onto ChainGripper FK.

    The inverse is exact for realizable ``[left, center, right]`` points. For
    arbitrary decoder outputs it anchors translation at the predicted center,
    obtains orientation from the tip chord, obtains the joint angle from the
    two ordered center-to-tip rays, clips to the native controller limits, and
    reconstructs valid points with the authoritative Sim V2 geometry.

    This intentionally has no temporal state. The rollout adapter still uses
    sequential constrained IK (including the previous-control fallback); the
    validation diagnostics measure that exact executed representation.
    """

    action_dim = 6

    def __init__(self, world_size: float = 512.0, epsilon: float = 1e-8):
        super().__init__()
        self.world_size = float(world_size)
        self.epsilon = float(epsilon)
        if self.world_size <= 0.0:
            raise ValueError("world_size must be positive")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive")

    def to_native(self, actions: torch.Tensor) -> torch.Tensor:
        """Return analytic native4 controls for arbitrary ordered points."""

        if actions.ndim == 0 or actions.shape[-1] != self.action_dim:
            raise ValueError(
                "ChainGripperPointActionCanonicalizer expects (..., 6), got "
                f"{tuple(actions.shape)}"
            )
        value = actions.float()
        left = value[..., 0:2]
        center = value[..., 2:4].clamp(0.0, self.world_size)
        right = value[..., 4:6]

        left_ray = center - left
        right_ray = right - center
        cross = (
            left_ray[..., 0] * right_ray[..., 1] - left_ray[..., 1] * right_ray[..., 0]
        )
        dot = (left_ray * right_ray).sum(dim=-1)
        ray_product = torch.linalg.vector_norm(
            left_ray, dim=-1
        ) * torch.linalg.vector_norm(right_ray, dim=-1)
        valid_rays = ray_product > self.epsilon
        # Avoid evaluating atan2(0, 0): masked NaNs can still poison backward.
        safe_cross = torch.where(valid_rays, cross, torch.zeros_like(cross))
        safe_dot = torch.where(valid_rays, dot, torch.ones_like(dot))
        joint_angle = 0.5 * torch.atan2(safe_cross, safe_dot)
        joint_angle = joint_angle.clamp(
            CHAIN_GRIPPER_OPEN_ANGLE, CHAIN_GRIPPER_CLOSED_ANGLE
        )

        chord = right - left
        chord_norm = torch.linalg.vector_norm(chord, dim=-1)
        valid_chord = chord_norm > self.epsilon
        safe_chord_x = torch.where(
            valid_chord, chord[..., 0], torch.ones_like(chord[..., 0])
        )
        safe_chord_y = torch.where(
            valid_chord, chord[..., 1], torch.zeros_like(chord[..., 1])
        )
        theta = torch.atan2(safe_chord_y, safe_chord_x)

        grip = (joint_angle - CHAIN_GRIPPER_OPEN_ANGLE) / (
            CHAIN_GRIPPER_CLOSED_ANGLE - CHAIN_GRIPPER_OPEN_ANGLE
        )
        return torch.cat((center, theta.unsqueeze(-1), grip.unsqueeze(-1)), dim=-1)

    @staticmethod
    def native_to_points(controls: torch.Tensor) -> torch.Tensor:
        center = controls[..., 0:2]
        theta = controls[..., 2]
        grip = controls[..., 3].clamp(0.0, 1.0)
        joint_angle = CHAIN_GRIPPER_OPEN_ANGLE + grip * (
            CHAIN_GRIPPER_CLOSED_ANGLE - CHAIN_GRIPPER_OPEN_ANGLE
        )
        radius = 2.0 * CHAIN_GRIPPER_LINK_LEN * torch.cos(joint_angle / 2.0)
        left_angle = theta - joint_angle
        right_angle = theta + joint_angle
        fitted_left = center - radius.unsqueeze(-1) * torch.stack(
            (torch.cos(left_angle), torch.sin(left_angle)), dim=-1
        )
        fitted_right = center + radius.unsqueeze(-1) * torch.stack(
            (torch.cos(right_angle), torch.sin(right_angle)), dim=-1
        )
        return torch.cat((fitted_left, center, fitted_right), dim=-1)

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        return self.native_to_points(self.to_native(actions))


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
    native_angle_index = 2

    def __init__(
        self,
        action_horizon: int = 16,
        world_size: float = 512.0,
        grid_size: int = 33,
        refinements: int = 6,
        context_state_key: str = "state_agent_obj",
        previous_control_key: str = "previous_control",
        input_is_canonical: bool = False,
    ):
        self.action_horizon = int(action_horizon)
        if self.action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        self.input_is_canonical = bool(input_is_canonical)
        self.canonicalizer = ChainGripperPointActionCanonicalizer(world_size=world_size)
        self._canonical_projection_diagnostics: dict | None = None
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
        if self._canonical_projection_diagnostics is not None:
            return self._canonical_projection_diagnostics
        return self.revert_transform.last_projection_diagnostics

    def decode(self, actions, context: dict | None = None):
        self._canonical_projection_diagnostics = None
        if self.input_is_canonical:
            del context
            template = actions
            value = actions if torch.is_tensor(actions) else torch.as_tensor(actions)
            controls = self.canonicalizer.to_native(value)
            fitted = self.canonicalizer.native_to_points(controls)
            point_rmse = (
                (fitted.float() - value.float()).square().mean(dim=-1).sqrt().detach()
            )
            diagnostic_shape = tuple(point_rmse.shape)
            self._canonical_projection_diagnostics = {
                "point_rmse": point_rmse.cpu().numpy(),
                "wrong_chirality": np.zeros(diagnostic_shape, dtype=bool),
                "degenerate": np.zeros(diagnostic_shape, dtype=bool),
                "used_exact_inverse": np.ones(diagnostic_shape, dtype=bool),
                "mean_point_rmse": float(point_rmse.mean().item()),
                "max_point_rmse": float(point_rmse.max().item()),
                "wrong_chirality_count": 0,
                "degenerate_count": 0,
                "trajectory_count": int(np.prod(diagnostic_shape[:-1]))
                if len(diagnostic_shape) > 1
                else 1,
                "horizon": int(diagnostic_shape[-1]),
            }
            if torch.is_tensor(template):
                dtype = (
                    template.dtype if template.is_floating_point() else torch.float32
                )
                return controls.to(device=template.device, dtype=dtype)
            return _restore_numeric_type(controls.detach().cpu().numpy(), template)
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
