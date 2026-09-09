"""Arc-length tokenize / detokenize stages for configured Planar graphs.

The U-Socket tokenizer is deliberately separate from the existing robot
``TokenizePlanarArcLength`` transform: the robot schema remains unchanged,
while U-Socket uses independent translation and rotation velocity streams.

These two stages move that boundary INTO the graph, as nodes with contracts:

* :class:`ArcTokenizeStage` replaces ``ActionTargetBuilder`` -- it reads the
  loader's time-indexed action chunk and writes the arc token as ``target``.
  It is the only writer of ``target``, so the rest of a DP or flow graph is
  unchanged; it simply models arc tokens.
* :class:`ArcDetokenizeStage` is the inverse, reading a predicted arc token and
  writing back a time-indexed chunk. It is declared ``inference_only``, the
  mirror of ``train_only``: ``pred_action`` exists only in the inference graph,
  and this runner treats an unsatisfied read as a configuration error rather
  than a reason to skip, so the restriction is stated rather than inferred.

Both stages are deliberately Planar-specific and live behind explicit
contracts: the generic runner in ``pipeline/core.py`` still knows nothing about
what an action, an angle, or a waypoint is.
"""

from __future__ import annotations

import numpy as np
import torch

from egomimic.pipeline.core import Stage
from egomimic.rldb.zarr.planar_arc import (
    PLANAR_ARC_STACKED_DIM,
    PLANAR_ACTION_DIM,
    TokenizeUSocketArcVelocity,
)


def _as_batched(actions: torch.Tensor, label: str) -> torch.Tensor:
    if not torch.is_tensor(actions):
        raise TypeError(f"{label} must be a tensor, got {type(actions).__name__}")
    if actions.ndim != 3:
        raise ValueError(f"{label} must be (B, T, D), got {tuple(actions.shape)}")
    return actions


class ArcTokenizeStage(Stage):
    """Encode a time-indexed action chunk as an arc-length token.

    Writes ``target`` directly rather than a namespaced key, so this stage takes
    ``ActionTargetBuilder``'s place in a stage list instead of sitting after it.
    Two writers of ``target`` would be a duplicate-writer lint error, and the
    graph would be ambiguous about which one the denoiser is modelling.

    Tokenization runs per sample through ``TokenizeUSocketArcVelocity``.
    """

    train_only = True
    writes = ("target",)

    def __init__(
        self,
        action_key: str = "actions",
        min_distance_unit: float = 200.0,
        resampled_vector_length: int = 100,
        dt: float = 1.0 / 30.0,
        rotation_distance_unit: float | None = None,
    ):
        super().__init__()
        self.action_key = str(action_key)
        if not self.action_key:
            raise ValueError("action_key must be non-empty")
        self.reads = (self.action_key,)
        self.num_waypoints = int(resampled_vector_length)
        self.tokenizer = TokenizeUSocketArcVelocity(
            action_key="actions",
            output_action_key="actions",
            min_distance_unit=min_distance_unit,
            resampled_vector_length=resampled_vector_length,
            dt=dt,
            rotation_distance_unit=rotation_distance_unit,
        )

    def forward(self, batch: dict) -> dict:
        actions = _as_batched(batch[self.action_key], "ArcTokenizeStage input")
        native = actions.detach().cpu().numpy().astype(np.float64, copy=False)
        tokens = np.stack(
            [self.tokenizer.tokenize(sample) for sample in native], axis=0
        )
        batch["target"] = torch.as_tensor(
            tokens, dtype=actions.dtype, device=actions.device
        )
        batch.pop(self.action_key, None)
        return batch


class ArcDetokenizeStage(Stage):
    """Decode independent XY-speed and angle-velocity streams in real time."""

    # Inference-only rather than "blocked in train by a missing read": a stage
    # whose reads are unavailable is a configuration error in this runner, so a
    # mode restriction has to be declared, not inferred.
    inference_only = True
    reads = ("pred_action",)
    writes = ("pred_action_native", "log/*")

    def __init__(
        self,
        resampled_vector_length: int = 100,
        action_horizon: int = 16,
        dt: float = 1.0 / 30.0,
        native_action_dim: int = 3,
        zero_dist_epsilon: float = 1e-9,
    ):
        super().__init__()
        self.num_waypoints = int(resampled_vector_length)
        self.action_horizon = int(action_horizon)
        self.dt = float(dt)
        self.native_action_dim = int(native_action_dim)
        self.zero_dist_epsilon = float(zero_dist_epsilon)
        if self.num_waypoints < 2:
            raise ValueError("resampled_vector_length must be at least two")
        if self.action_horizon <= 0 or self.dt <= 0:
            raise ValueError("action_horizon and dt must be positive")
        if self.native_action_dim not in (2, 3):
            raise ValueError("native_action_dim must be 2 or 3 for U-Socket")

    def _decode_stream(
        self, geometry: torch.Tensor, interval_distance: torch.Tensor, rate: torch.Tensor
    ) -> torch.Tensor:
        """Invert local interval rates into a geometry sample at every ``dt``."""
        active = interval_distance > self.zero_dist_epsilon
        usable = rate.abs() > self.zero_dist_epsilon
        duration = torch.zeros_like(interval_distance)
        duration = torch.where(
            active & usable,
            interval_distance / rate.abs().clamp_min(self.zero_dist_epsilon),
            duration,
        )
        # A predicted nonzero interval with zero velocity must not teleport.
        stop_duration = self.dt * (self.action_horizon + 1)
        duration = torch.where(
            active & ~usable, torch.full_like(duration, stop_duration), duration
        )
        return self._interpolate_at_times(geometry, duration)

    def _interpolate_at_times(
        self, geometry: torch.Tensor, duration: torch.Tensor
    ) -> torch.Tensor:
        """Sample ``geometry`` every ``dt`` given per-interval durations."""
        cumulative_time = torch.cat(
            (torch.zeros_like(duration[:, :1]), torch.cumsum(duration, dim=1)), dim=1
        )
        targets = self.dt * torch.arange(
            self.action_horizon, device=geometry.device, dtype=geometry.dtype
        )[None, :]
        upper = torch.searchsorted(
            cumulative_time.contiguous(), targets.expand(len(geometry), -1).contiguous(),
            right=True,
        ).clamp(1, self.num_waypoints - 1)
        lower = upper - 1
        t_lo = torch.gather(cumulative_time, 1, lower)
        t_hi = torch.gather(cumulative_time, 1, upper)
        alpha = ((targets - t_lo) / (t_hi - t_lo).clamp_min(self.zero_dist_epsilon))
        alpha = alpha.clamp(0.0, 1.0).unsqueeze(-1)
        width = geometry.shape[-1]
        lo = torch.gather(geometry, 1, lower.unsqueeze(-1).expand(-1, -1, width))
        hi = torch.gather(geometry, 1, upper.unsqueeze(-1).expand(-1, -1, width))
        return (1.0 - alpha) * lo + alpha * hi

    def forward(self, batch: dict) -> dict:
        tokens = _as_batched(batch["pred_action"], "ArcDetokenizeStage input")
        expected = (2 * self.num_waypoints, PLANAR_ACTION_DIM)
        if tuple(tokens.shape[1:]) != expected:
            raise ValueError(
                f"ArcDetokenizeStage expects (B, {expected[0]}, {expected[1]}), "
                f"got {tuple(tokens.shape)}"
            )

        translation = tokens[:, : self.num_waypoints]
        rotation = tokens[:, self.num_waypoints :]
        xy_distance = torch.linalg.vector_norm(
            translation[:, 1:, :2] - translation[:, :-1, :2], dim=-1
        )
        xy = self._decode_stream(
            translation[..., :2], xy_distance, translation[:, :-1, 4].clamp_min(0.0)
        )
        heading_geometry = rotation[..., 2:4]
        delta_cos = (
            heading_geometry[:, 1:, 0] * heading_geometry[:, :-1, 0]
            + heading_geometry[:, 1:, 1] * heading_geometry[:, :-1, 1]
        )
        delta_sin = (
            heading_geometry[:, 1:, 1] * heading_geometry[:, :-1, 0]
            - heading_geometry[:, 1:, 0] * heading_geometry[:, :-1, 1]
        )
        angle_distance = torch.atan2(delta_sin, delta_cos).abs()
        heading = self._decode_stream(
            heading_geometry, angle_distance, rotation[:, :-1, 4]
        )
        norm = torch.linalg.vector_norm(heading, dim=-1, keepdim=True)
        heading = heading / norm.clamp_min(1e-8)
        theta = torch.atan2(heading[..., 1], heading[..., 0]).unsqueeze(-1)
        native = torch.cat((xy, theta), dim=-1)

        batch["pred_action_native"] = native[..., : self.native_action_dim]
        batch["log/ArcLinearSpeed"] = translation[..., 4].clamp_min(0.0).mean()
        batch["log/ArcAngularVelocityAbs"] = rotation[..., 4].abs().mean()
        batch["log/ArcTranslationDistance"] = xy_distance.sum(dim=1).mean()
        batch["log/ArcRotationDistance"] = angle_distance.sum(dim=1).mean()
        return batch


class ArcDetokenizeStackedStage(ArcDetokenizeStage):
    """Decode the stacked ``[M, 6]`` ARC token.

    Identical decode mathematics to the parent -- same local-rate inversion,
    same heading renormalisation -- but the two streams are separated by COLUMN
    rather than by row:

        ``[x, y, v_xy, cos, sin, omega]``

    translation geometry ``[..., 0:2]`` with rate ``[..., 2]``; heading
    geometry ``[..., 3:5]`` with rate ``[..., 5]``.
    """

    def forward(self, batch: dict) -> dict:
        tokens = _as_batched(batch["pred_action"], "ArcDetokenizeStackedStage input")
        expected = (self.num_waypoints, PLANAR_ARC_STACKED_DIM)
        if tuple(tokens.shape[1:]) != expected:
            raise ValueError(
                f"ArcDetokenizeStackedStage expects (B, {expected[0]}, "
                f"{expected[1]}), got {tuple(tokens.shape)}"
            )

        xy_distance = torch.linalg.vector_norm(
            tokens[:, 1:, :2] - tokens[:, :-1, :2], dim=-1
        )
        xy = self._decode_stream(
            tokens[..., :2], xy_distance, tokens[:, :-1, 2].clamp_min(0.0)
        )
        heading_geometry = tokens[..., 3:5]
        delta_cos = (
            heading_geometry[:, 1:, 0] * heading_geometry[:, :-1, 0]
            + heading_geometry[:, 1:, 1] * heading_geometry[:, :-1, 1]
        )
        delta_sin = (
            heading_geometry[:, 1:, 1] * heading_geometry[:, :-1, 0]
            - heading_geometry[:, 1:, 0] * heading_geometry[:, :-1, 1]
        )
        angle_distance = torch.atan2(delta_sin, delta_cos).abs()
        heading = self._decode_stream(
            heading_geometry, angle_distance, tokens[:, :-1, 5]
        )
        norm = torch.linalg.vector_norm(heading, dim=-1, keepdim=True)
        heading = heading / norm.clamp_min(1e-8)
        theta = torch.atan2(heading[..., 1], heading[..., 0]).unsqueeze(-1)
        native = torch.cat((xy, theta), dim=-1)

        batch["pred_action_native"] = native[..., : self.native_action_dim]
        batch["log/ArcLinearSpeed"] = tokens[..., 2].clamp_min(0.0).mean()
        batch["log/ArcAngularVelocityAbs"] = tokens[..., 5].abs().mean()
        batch["log/ArcTranslationDistance"] = xy_distance.sum(dim=1).mean()
        batch["log/ArcRotationDistance"] = angle_distance.sum(dim=1).mean()
        return batch


class ArcDetokenizeDurationStage(ArcDetokenizeStackedStage):
    """Decode the stacked token whose timing channels are DURATIONS.

    ``[x, y, dt_translation, cos, sin, dt_rotation]``

    Strictly simpler than the velocity decoder: the per-interval durations are
    read directly instead of being recovered by dividing a decoded distance by
    a predicted rate. That removes the division, the ``rate.abs()`` (which
    silently discarded the predicted sign of omega), and the synthetic
    ``stop_duration`` fallback for a nonzero interval with zero predicted
    velocity -- a hold is just a long duration here.

    Durations are clamped non-negative: time cannot run backwards, and an
    unclamped negative prediction would make the cumulative clock
    non-monotone, breaking the searchsorted lookup.
    """

    def forward(self, batch: dict) -> dict:
        tokens = _as_batched(batch["pred_action"], "ArcDetokenizeDurationStage input")
        expected = (self.num_waypoints, PLANAR_ARC_STACKED_DIM)
        if tuple(tokens.shape[1:]) != expected:
            raise ValueError(
                f"ArcDetokenizeDurationStage expects (B, {expected[0]}, "
                f"{expected[1]}), got {tuple(tokens.shape)}"
            )

        xy_duration = tokens[:, :-1, 2].clamp_min(0.0)
        theta_duration = tokens[:, :-1, 5].clamp_min(0.0)
        xy = self._interpolate_at_times(tokens[..., :2], xy_duration)
        heading_geometry = tokens[..., 3:5]
        heading = self._interpolate_at_times(heading_geometry, theta_duration)
        norm = torch.linalg.vector_norm(heading, dim=-1, keepdim=True)
        heading = heading / norm.clamp_min(1e-8)
        theta = torch.atan2(heading[..., 1], heading[..., 0]).unsqueeze(-1)
        native = torch.cat((xy, theta), dim=-1)

        batch["pred_action_native"] = native[..., : self.native_action_dim]
        batch["log/ArcTranslationDuration"] = xy_duration.sum(dim=1).mean()
        batch["log/ArcRotationDuration"] = theta_duration.sum(dim=1).mean()
        return batch
