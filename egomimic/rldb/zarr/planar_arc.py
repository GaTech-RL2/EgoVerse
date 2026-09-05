"""Small, explicit SE(2) arc tokenizer used by Planar PushShapes models."""

from __future__ import annotations

import math

import numpy as np

PLANAR_ACTION_DIM = 5  # [x, y, cos(theta), sin(theta), grip]


def lambda_for_radius(radius: float) -> float:
    """Convert a physical rotation radius into the SE(2) metric weight."""
    radius = float(radius)
    if radius < 0:
        raise ValueError("radius must be non-negative")
    return 2.0 * math.sqrt(2.0) * radius


def rotation_step_metric_planar(theta: np.ndarray) -> np.ndarray:
    """Return the scaled chordal rotation distance between adjacent angles."""
    theta = np.unwrap(np.asarray(theta, dtype=np.float64))
    return math.sqrt(2.0) * np.sin(np.abs(np.diff(theta)) / 4.0)


def planar_step_distance(
    xy: np.ndarray,
    theta: np.ndarray | None = None,
    lambda_rot: float = 0.0,
) -> np.ndarray:
    """Return adjacent SE(2) distances: translation plus weighted rotation."""
    xy = np.asarray(xy, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"xy must have shape (T, 2), got {xy.shape}")
    if len(xy) < 2:
        return np.zeros(0, dtype=np.float64)
    distance = np.linalg.norm(np.diff(xy, axis=0), axis=-1)
    if lambda_rot:
        if lambda_rot < 0 or theta is None or len(theta) != len(xy):
            raise ValueError("positive lambda_rot requires one theta per xy")
        distance += float(lambda_rot) * rotation_step_metric_planar(theta)
    return distance


def _bracket_segment(cumulative: np.ndarray, target: float) -> tuple[int, float]:
    if target <= cumulative[0]:
        return 0, 0.0
    if target >= cumulative[-1]:
        return len(cumulative) - 2, 1.0
    index = int(np.searchsorted(cumulative, target, side="left") - 1)
    index = max(0, min(index, len(cumulative) - 2))
    span = cumulative[index + 1] - cumulative[index]
    alpha = 0.0 if span <= 1e-12 else (target - cumulative[index]) / span
    return index, float(alpha)


def _interpolate(values: np.ndarray, cumulative: np.ndarray, target: float):
    if target <= cumulative[0]:
        return values[0].copy()
    if target >= cumulative[-1]:
        return values[-1].copy()
    index, alpha = _bracket_segment(cumulative, target)
    return (1.0 - alpha) * values[index] + alpha * values[index + 1]


class PadPlanarAction:
    """Widen native ``[x,y[,theta[,grip]]]`` actions to a common five-vector."""

    def __init__(self, keys: list[str] | None = None):
        self.keys = list(keys or ["actions"])

    def transform(self, batch: dict) -> dict:
        for key in self.keys:
            value = np.asarray(batch[key])
            if value.ndim != 2 or value.shape[-1] not in (2, 3, 4):
                raise ValueError(
                    f"PadPlanarAction expects (T, 2|3|4) for {key!r}, got {value.shape}"
                )
            work = value.astype(np.float64, copy=False)
            theta = work[:, 2] if work.shape[1] >= 3 else np.zeros(len(work))
            grip = work[:, 3] if work.shape[1] == 4 else np.zeros(len(work))
            output = np.column_stack((work[:, :2], np.cos(theta), np.sin(theta), grip))
            dtype = (
                value.dtype if np.issubdtype(value.dtype, np.floating) else np.float32
            )
            batch[key] = output.astype(dtype, copy=False)
        return batch


class TokenizeUSocketArcVelocity:
    """Encode U-Socket actions as independent translation and angle streams.

    The token has ``2 * M`` rows of width five. Rows ``[:M]`` are
    ``[x, y, 0, 0, v_xy]`` and rows ``[M:]`` are
    ``[0, 0, cos(theta), sin(theta), omega]``. ``v_xy`` is the local
    non-negative tangential speed of each translation interval and ``omega``
    is the local *signed* angular velocity of each angle interval. There is no
    chunk-level or mean velocity row.

    Translation and rotation are sampled on their own cumulative arc clocks
    and have independent distance budgets. Their local velocities recover the
    elapsed-time parameterization when the streams are decoded.
    """

    def __init__(
        self,
        action_key: str = "actions",
        output_action_key: str = "actions",
        min_distance_unit: float = 200.0,
        resampled_vector_length: int = 100,
        dt: float = 1.0 / 30.0,
        rotation_distance_unit: float | None = None,
        zero_dist_epsilon: float = 1e-9,
    ):
        if min_distance_unit <= 0 or dt <= 0:
            raise ValueError("min_distance_unit and dt must be positive")
        if resampled_vector_length < 2:
            raise ValueError("resampled_vector_length must be at least two")
        if rotation_distance_unit is not None and (
            not math.isfinite(rotation_distance_unit) or rotation_distance_unit <= 0
        ):
            raise ValueError("rotation_distance_unit must be finite and positive")
        self.action_key = str(action_key)
        self.output_action_key = str(output_action_key)
        self.distance = float(min_distance_unit)
        self.num_waypoints = int(resampled_vector_length)
        self.dt = float(dt)
        self.rotation_distance = (
            None
            if rotation_distance_unit is None
            else float(rotation_distance_unit)
        )
        self.zero_dist_epsilon = float(zero_dist_epsilon)

    @staticmethod
    def _components(actions: np.ndarray):
        xy = actions[:, :2]
        if actions.shape[1] == PLANAR_ACTION_DIM:
            theta = np.unwrap(np.arctan2(actions[:, 3], actions[:, 2]))
        elif actions.shape[1] == 3:
            theta = np.unwrap(actions[:, 2])
        else:
            theta = np.zeros(len(actions))
        return xy, theta

    def _sample_stream(
        self,
        values: np.ndarray,
        cumulative: np.ndarray,
        end: float,
        *,
        signed_rate: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample geometry and one local interval rate at each waypoint."""
        if end <= self.zero_dist_epsilon:
            points = np.repeat(values[:1], self.num_waypoints, axis=0)
            return points, np.zeros(self.num_waypoints, dtype=np.float64)

        targets = np.linspace(0.0, end, self.num_waypoints)
        points = np.stack(
            [_interpolate(values, cumulative, point) for point in targets]
        )
        points[0] = values[0]

        # Recover the source time at every sampled arc position. Using the
        # bracketing source frames (rather than end / total_time) retains local
        # speed changes and also accounts for stationary frames before motion.
        times = np.empty(self.num_waypoints, dtype=np.float64)
        times[0] = 0.0
        for index, target in enumerate(targets[1:], start=1):
            source_index, alpha = _bracket_segment(cumulative, float(target))
            times[index] = (source_index + alpha) * self.dt

        delta_t = np.diff(times)
        if signed_rate:
            delta_geometry = np.diff(points[:, 0])
        else:
            delta_geometry = np.linalg.norm(np.diff(points, axis=0), axis=-1)
        interval_rate = np.divide(
            delta_geometry,
            delta_t,
            out=np.zeros_like(delta_geometry),
            where=delta_t > self.zero_dist_epsilon,
        )
        rates = np.zeros(self.num_waypoints, dtype=np.float64)
        rates[:-1] = interval_rate
        rates[-1] = interval_rate[-1]
        return points, rates

    def tokenize(self, actions: np.ndarray) -> np.ndarray:
        xy, theta = self._components(actions)
        translation_arc = np.concatenate(
            (np.zeros(1), np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=-1)))
        )
        angle_arc = np.concatenate((np.zeros(1), np.cumsum(np.abs(np.diff(theta)))))
        translation_end = min(self.distance, float(translation_arc[-1]))
        rotation_end = float(angle_arc[-1])
        if self.rotation_distance is not None:
            rotation_end = min(self.rotation_distance, rotation_end)

        xy_waypoints, linear_speed = self._sample_stream(
            xy, translation_arc, translation_end, signed_rate=False
        )
        theta_waypoints, angular_velocity = self._sample_stream(
            theta[:, None], angle_arc, rotation_end, signed_rate=True
        )

        translation = np.zeros((self.num_waypoints, PLANAR_ACTION_DIM))
        translation[:, :2] = xy_waypoints
        translation[:, 4] = linear_speed
        rotation = np.zeros((self.num_waypoints, PLANAR_ACTION_DIM))
        rotation[:, 2] = np.cos(theta_waypoints[:, 0])
        rotation[:, 3] = np.sin(theta_waypoints[:, 0])
        rotation[:, 4] = angular_velocity
        return np.concatenate((translation, rotation), axis=0)

    def transform(self, batch: dict) -> dict:
        value = np.asarray(batch[self.action_key])
        if value.ndim != 2 or value.shape[1] not in (2, 3, PLANAR_ACTION_DIM):
            raise ValueError(
                "TokenizeUSocketArcVelocity expects U-Socket native (T, 2|3) or "
                f"common-five (T, 5), got {value.shape}"
            )
        if len(value) < 2 or not np.isfinite(value).all():
            raise ValueError("planar actions need at least two finite timesteps")
        output = self.tokenize(value.astype(np.float64, copy=False))
        dtype = value.dtype if np.issubdtype(value.dtype, np.floating) else np.float32
        batch[self.output_action_key] = output.astype(dtype, copy=False)
        return batch


class TokenizePlanarArcLength:
    """Legacy robot/Planar SE(2) tokenizer; its schema remains unchanged."""

    def __init__(
        self,
        action_key: str = "actions",
        output_action_key: str = "actions",
        min_distance_unit: float = 200.0,
        resampled_vector_length: int = 100,
        dt: float = 1.0 / 30.0,
        rotation_radius: float = 0.0,
        hybrid_rotation_unit: float | None = None,
        zero_dist_epsilon: float = 1e-9,
    ):
        if min_distance_unit <= 0 or dt <= 0:
            raise ValueError("min_distance_unit and dt must be positive")
        if resampled_vector_length < 2:
            raise ValueError("resampled_vector_length must be at least two")
        if rotation_radius < 0:
            raise ValueError("rotation_radius must be non-negative")
        if hybrid_rotation_unit is not None and (
            not math.isfinite(hybrid_rotation_unit) or hybrid_rotation_unit <= 0
        ):
            raise ValueError("hybrid_rotation_unit must be finite and positive")
        self.action_key = str(action_key)
        self.output_action_key = str(output_action_key)
        self.distance = float(min_distance_unit)
        self.num_waypoints = int(resampled_vector_length)
        self.dt = float(dt)
        self.rotation_radius = float(rotation_radius)
        self.hybrid_rotation_unit = (
            None if hybrid_rotation_unit is None else float(hybrid_rotation_unit)
        )
        self.zero_dist_epsilon = float(zero_dist_epsilon)

    @staticmethod
    def _components(actions: np.ndarray):
        xy = actions[:, :2]
        theta = (
            np.unwrap(actions[:, 2])
            if actions.shape[1] >= 3
            else np.zeros(len(actions))
        )
        grip = actions[:, 3] if actions.shape[1] == 4 else np.zeros(len(actions))
        return xy, theta, grip

    def _window_end(self, cumulative: np.ndarray, theta: np.ndarray) -> float:
        end = min(self.distance, float(cumulative[-1]))
        if self.hybrid_rotation_unit is None:
            return end
        rotation = np.concatenate(
            (np.zeros(1), np.cumsum(rotation_step_metric_planar(theta)))
        )
        if (
            cumulative[-1] > self.zero_dist_epsilon
            and rotation[-1] > self.zero_dist_epsilon
        ):
            end = min(
                end,
                float(cumulative[-1])
                * min(1.0, self.hybrid_rotation_unit / float(rotation[-1])),
            )
        return end

    def tokenize(self, actions: np.ndarray) -> np.ndarray:
        xy, theta, grip = self._components(actions)
        steps = planar_step_distance(
            xy, theta, lambda_for_radius(self.rotation_radius)
        )
        cumulative = np.concatenate((np.zeros(1), np.cumsum(steps)))
        end = self._window_end(cumulative, theta)
        if end <= self.zero_dist_epsilon:
            xy_waypoints = np.repeat(xy[:1], self.num_waypoints, axis=0)
            theta_waypoints = np.repeat(theta[0], self.num_waypoints)
            grip_waypoints = np.repeat(grip[0], self.num_waypoints)
            speed = 0.0
        else:
            targets = np.linspace(0.0, end, self.num_waypoints)
            xy_waypoints = np.stack(
                [_interpolate(xy, cumulative, point) for point in targets]
            )
            theta_waypoints = np.array(
                [
                    _interpolate(theta[:, None], cumulative, point)[0]
                    for point in targets
                ]
            )
            grip_waypoints = np.array(
                [
                    _interpolate(grip[:, None], cumulative, point)[0]
                    for point in targets
                ]
            )
            last_index = int(np.searchsorted(cumulative, end, side="left"))
            speed = end / (max(1, last_index) * self.dt)
        xy_waypoints[0] = xy[0]
        theta_waypoints[0] = theta[0]
        grip_waypoints[0] = grip[0]
        waypoints = np.column_stack(
            (
                xy_waypoints,
                np.cos(theta_waypoints),
                np.sin(theta_waypoints),
                grip_waypoints,
            )
        )
        timing = np.zeros((1, PLANAR_ACTION_DIM), dtype=np.float64)
        timing[0, 0] = speed
        return np.concatenate((waypoints, timing), axis=0)

    def transform(self, batch: dict) -> dict:
        value = np.asarray(batch[self.action_key])
        if value.ndim != 2 or value.shape[1] not in (2, 3, 4):
            raise ValueError(
                f"TokenizePlanarArcLength expects (T, 2|3|4), got {value.shape}"
            )
        if len(value) < 2 or not np.isfinite(value).all():
            raise ValueError("planar actions need at least two finite timesteps")
        output = self.tokenize(value.astype(np.float64, copy=False))
        dtype = value.dtype if np.issubdtype(value.dtype, np.floating) else np.float32
        batch[self.output_action_key] = output.astype(dtype, copy=False)
        return batch
