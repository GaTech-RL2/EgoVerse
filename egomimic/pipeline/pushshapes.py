"""PushShapes-specific adapters for Pipeline policy rollout."""

from __future__ import annotations

import numpy as np
import torch

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
        input_is_tensor = torch.is_tensor(actions)
        if input_is_tensor:
            device, dtype = actions.device, actions.dtype
            value = actions.detach().cpu().numpy().astype(np.float64)
        else:
            value = np.asarray(actions, dtype=np.float64)
            device = dtype = None
        if value.ndim == 2:
            value = value[None]
        expected = (self.resampled_vector_length + 1, USOCKET_ARC_DIM)
        if value.ndim != 3 or tuple(value.shape[1:]) != expected:
            raise ValueError(
                "USocketArcLengthRolloutAdapter expects "
                f"(B, {expected[0]}, {expected[1]}), got {value.shape}"
            )
        decoded = np.stack(
            [
                self.detokenizer.detokenize(token, self.action_horizon)
                for token in value
            ]
        )
        if input_is_tensor:
            return torch.from_numpy(decoded).to(device=device, dtype=dtype)
        return decoded.astype(np.float32)

    __call__ = decode


class ChainGripperPointRolloutAdapter:
    """Project ordered six-dimensional point commands to native pose control.

    The implementation deliberately delegates to the Sim V2 projection utility
    so dataset conversion, direct point-mode replay, and learned-policy rollout
    share one FK/IK definition and one set of geometry constants.
    """

    preserves_decoded_timing = True

    def __init__(
        self,
        action_horizon: int = 16,
        world_size: float = 512.0,
        grid_size: int = 33,
        refinements: int = 6,
        context_state_key: str = "state_agent_obj",
    ):
        self.action_horizon = int(action_horizon)
        self.world_size = float(world_size)
        self.grid_size = int(grid_size)
        self.refinements = int(refinements)
        self.context_state_key = str(context_state_key)
        self.last_projection_diagnostics: dict | None = None
        if self.action_horizon <= 0:
            raise ValueError("action_horizon must be positive")

    @staticmethod
    def _context_rows(value, trajectory_count: int, *, key: str) -> np.ndarray:
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        array = np.asarray(value)
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
        context: dict | None,
        trajectory_count: int,
    ) -> np.ndarray | None:
        if not context:
            return None
        if "previous_control" in context:
            rows = self._context_rows(
                context["previous_control"],
                trajectory_count,
                key="previous_control",
            )
            if rows.shape[-1] != 4:
                raise ValueError(
                    "Rollout context 'previous_control' must have last dimension 4, "
                    f"got {rows.shape}"
                )
            return rows.astype(np.float64, copy=False)
        if self.context_state_key not in context:
            return None
        rows = self._context_rows(
            context[self.context_state_key],
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

    def _decode_numpy(
        self,
        actions: np.ndarray,
        context: dict | None = None,
    ) -> np.ndarray:
        value = np.asarray(actions)
        if value.ndim < 2 or value.shape[-1] != 6:
            raise ValueError(
                "ChainGripperPointRolloutAdapter expects (..., 6) ordered "
                f"points, got {value.shape}"
            )
        from Tsimulation.sim_v2.pushshapes.chain_gripper_control import (
            pose_control_to_points,
            project_points_to_pose_control,
        )

        trajectory_shape = value.shape[:-2]
        horizon = value.shape[-2]
        trajectory_count = int(np.prod(trajectory_shape)) if trajectory_shape else 1
        trajectories = value.reshape(trajectory_count, horizon, 6)
        controls = np.empty((trajectory_count, horizon, 4), dtype=np.float64)
        point_rmse = np.empty((trajectory_count, horizon), dtype=np.float64)
        wrong_chirality = np.empty((trajectory_count, horizon), dtype=bool)
        degenerate = np.empty((trajectory_count, horizon), dtype=bool)
        used_exact_inverse = np.empty((trajectory_count, horizon), dtype=bool)
        initial_previous = self._initial_previous_controls(context, trajectory_count)

        # Projection is deliberately sequential: a degenerate row inherits the
        # most recent projected orientation from its own trajectory instead of
        # falling back to a global theta=0 convention.
        for trajectory_index, trajectory in enumerate(trajectories):
            previous = (
                None if initial_previous is None else initial_previous[trajectory_index]
            )
            for time_index, points in enumerate(trajectory):
                projection = project_points_to_pose_control(
                    points,
                    previous_control=previous,
                    world_size=self.world_size,
                    grid_size=self.grid_size,
                    refinements=self.refinements,
                )
                control = np.asarray(projection.control, dtype=np.float64)
                is_degenerate = bool(projection.degenerate)
                if previous is not None and is_degenerate:
                    # The shared projector supplies the bounded center/grip fit.
                    # Preserve only temporal orientation when endpoint geometry
                    # is ill-conditioned, as specified by the rollout contract.
                    control = control.copy()
                    control[2] = previous[2]
                    fitted_points = pose_control_to_points(
                        control,
                        world_size=self.world_size,
                    )
                    projection_rmse = float(
                        np.sqrt(np.mean(np.square(fitted_points - points)))
                    )
                else:
                    projection_rmse = float(projection.point_rmse)
                controls[trajectory_index, time_index] = control
                point_rmse[trajectory_index, time_index] = projection_rmse
                wrong_chirality[trajectory_index, time_index] = (
                    projection.wrong_chirality
                )
                degenerate[trajectory_index, time_index] = is_degenerate
                used_exact_inverse[trajectory_index, time_index] = (
                    projection.used_exact_inverse
                )
                previous = control

        diagnostic_shape = (*trajectory_shape, horizon)
        self.last_projection_diagnostics = {
            "point_rmse": point_rmse.reshape(diagnostic_shape),
            "wrong_chirality": wrong_chirality.reshape(diagnostic_shape),
            "degenerate": degenerate.reshape(diagnostic_shape),
            "used_exact_inverse": used_exact_inverse.reshape(diagnostic_shape),
            "mean_point_rmse": float(np.mean(point_rmse)),
            "max_point_rmse": float(np.max(point_rmse)),
            "wrong_chirality_count": int(np.count_nonzero(wrong_chirality)),
            "degenerate_count": int(np.count_nonzero(degenerate)),
        }
        decoded = controls.reshape(*value.shape[:-1], 4)
        output_dtype = (
            value.dtype if np.issubdtype(value.dtype, np.floating) else np.float32
        )
        return decoded.astype(output_dtype, copy=False)

    def decode(self, actions, context: dict | None = None):
        if torch.is_tensor(actions):
            device, dtype = actions.device, actions.dtype
            decoded = self._decode_numpy(
                actions.detach().cpu().numpy(),
                context=context,
            )
            return torch.from_numpy(decoded).to(device=device, dtype=dtype)
        return self._decode_numpy(actions, context=context)

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
        )

    @property
    def last_projection_diagnostics(self) -> dict | None:
        return self.point_adapter.last_projection_diagnostics

    def decode(self, actions, context: dict | None = None):
        input_is_tensor = torch.is_tensor(actions)
        if input_is_tensor:
            device, dtype = actions.device, actions.dtype
            value = actions.detach().cpu().numpy().astype(np.float64)
        else:
            raw = np.asarray(actions)
            value = raw.astype(np.float64)
            device = dtype = None
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
        if input_is_tensor:
            return torch.from_numpy(decoded).to(device=device, dtype=dtype)
        output_dtype = (
            raw.dtype if np.issubdtype(raw.dtype, np.floating) else np.float32
        )
        return decoded.astype(output_dtype, copy=False)

    __call__ = decode
