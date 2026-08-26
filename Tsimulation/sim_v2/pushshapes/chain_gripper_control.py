"""Dual action encodings for the Sim V2 chain gripper.

The native/default action remains ``[x, y, theta, grip]``.  Point mode accepts
three permanently ordered command points::

    [left_free_tip_x, left_free_tip_y,
     middle_joint_x, middle_joint_y,
     right_free_tip_x, right_free_tip_y]

The names left/right are link-topology labels, not image-space ordering.  Point
mode is an adapter: it maps the prediction back to the native controller; it
does not define a second physics implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .chain_gripper_constants import (
    CHAIN_GRIPPER_CLOSED_ANGLE,
    CHAIN_GRIPPER_LINK_LEN,
    CHAIN_GRIPPER_OPEN_ANGLE,
)

CHAIN_GRIPPER_POSE_MODE = "pose"
CHAIN_GRIPPER_POINT_MODE = "points"
CHAIN_GRIPPER_CONTROL_MODES = (
    CHAIN_GRIPPER_POSE_MODE,
    CHAIN_GRIPPER_POINT_MODE,
)
CHAIN_GRIPPER_POINT_ACTION_SPEC = (
    "left_free_tip_x",
    "left_free_tip_y",
    "middle_joint_x",
    "middle_joint_y",
    "right_free_tip_x",
    "right_free_tip_y",
)


def wrap_angle(angle):
    angle = np.asarray(angle, dtype=np.float64)
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _require_last_dim(value, width: int, name: str) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if value.ndim == 0 or value.shape[-1] != width:
        raise ValueError(f"{name} must have last dimension {width}, got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
    return value


def canonicalize_pose_control(control, *, world_size: float = 512.0) -> np.ndarray:
    """Apply the clipping/wrapping rules used by the native Sim V2 path."""
    control = _require_last_dim(control, 4, "control")
    result = control.copy()
    result[..., 0:2] = np.clip(result[..., 0:2], 0.0, world_size)
    result[..., 2] = wrap_angle(result[..., 2])
    result[..., 3] = np.clip(result[..., 3], 0.0, 1.0)
    return result


def grip_to_joint_angle(grip):
    grip = np.clip(np.asarray(grip, dtype=np.float64), 0.0, 1.0)
    return CHAIN_GRIPPER_OPEN_ANGLE + grip * (
        CHAIN_GRIPPER_CLOSED_ANGLE - CHAIN_GRIPPER_OPEN_ANGLE
    )


def joint_angle_to_grip(joint_angle):
    joint_angle = np.clip(
        np.asarray(joint_angle, dtype=np.float64),
        CHAIN_GRIPPER_OPEN_ANGLE,
        CHAIN_GRIPPER_CLOSED_ANGLE,
    )
    return (joint_angle - CHAIN_GRIPPER_OPEN_ANGLE) / (
        CHAIN_GRIPPER_CLOSED_ANGLE - CHAIN_GRIPPER_OPEN_ANGLE
    )


def pose_control_to_points(control, *, world_size: float = 512.0) -> np.ndarray:
    """Convert native commands to canonical, ordered command points."""
    control = canonicalize_pose_control(control, world_size=world_size)
    center = control[..., 0:2]
    theta = control[..., 2]
    joint_angle = grip_to_joint_angle(control[..., 3])
    radius = 2.0 * CHAIN_GRIPPER_LINK_LEN * np.cos(joint_angle / 2.0)
    left_angle = theta - joint_angle
    right_angle = theta + joint_angle
    left = center - radius[..., None] * np.stack(
        (np.cos(left_angle), np.sin(left_angle)), axis=-1
    )
    right = center + radius[..., None] * np.stack(
        (np.cos(right_angle), np.sin(right_angle)), axis=-1
    )
    return np.concatenate((left, center, right), axis=-1)


def points_to_pose_control_exact(points, *, world_size: float = 512.0) -> np.ndarray:
    """Invert an already-realizable point command without approximation."""
    points = _require_last_dim(points, 6, "points")
    left, center, right = points[..., 0:2], points[..., 2:4], points[..., 4:6]
    left_ray = center - left
    right_ray = right - center
    if np.any(np.linalg.norm(left_ray, axis=-1) <= 1e-12) or np.any(
        np.linalg.norm(right_ray, axis=-1) <= 1e-12
    ):
        raise ValueError("point rays must have non-zero length")
    cross = left_ray[..., 0] * right_ray[..., 1] - left_ray[..., 1] * right_ray[..., 0]
    dot = np.sum(left_ray * right_ray, axis=-1)
    joint_angle = 0.5 * np.arctan2(cross, dot)
    tolerance = 1e-9
    if np.any(joint_angle < CHAIN_GRIPPER_OPEN_ANGLE - tolerance) or np.any(
        joint_angle > CHAIN_GRIPPER_CLOSED_ANGLE + tolerance
    ):
        raise ValueError("points have invalid chirality or joint angle")
    chord = right - left
    theta = np.arctan2(chord[..., 1], chord[..., 0])
    control = np.concatenate(
        (center, theta[..., None], joint_angle_to_grip(joint_angle)[..., None]),
        axis=-1,
    )
    return canonicalize_pose_control(control, world_size=world_size)


@dataclass(frozen=True)
class PointProjection:
    control: np.ndarray
    fitted_points: np.ndarray
    point_rmse: np.ndarray
    wrong_chirality: np.ndarray
    degenerate: np.ndarray
    used_exact_inverse: np.ndarray


def project_points_to_pose_control(
    points,
    *,
    previous_control=None,
    world_size: float = 512.0,
    grid_size: int = 33,
    refinements: int = 6,
) -> PointProjection:
    """Project arbitrary point predictions onto the valid 4-DOF manifold.

    The predicted middle point anchors translation.  A bounded one-dimensional
    search chooses the shared joint angle; at each candidate angle, planar
    Procrustes gives the optimal orientation.  Exact dataset points take a
    lossless fast path so point replay is numerically identical to pose replay.
    """
    points = _require_last_dim(points, 6, "points")
    if grid_size < 5 or grid_size % 2 == 0:
        raise ValueError("grid_size must be an odd integer >= 5")
    if refinements < 1:
        raise ValueError("refinements must be >= 1")

    shape = points.shape[:-1]
    flat = points.reshape(-1, 6)
    count = flat.shape[0]
    previous = None
    if previous_control is not None:
        previous = canonicalize_pose_control(previous_control, world_size=world_size)
        previous = np.broadcast_to(previous, (*shape, 4)).reshape(-1, 4)

    # Preserve exact replay whenever every input point already lies on-manifold.
    try:
        exact_control = points_to_pose_control_exact(points, world_size=world_size)
        exact_fit = pose_control_to_points(exact_control, world_size=world_size)
        exact_rmse = np.sqrt(np.mean((exact_fit - points) ** 2, axis=-1))
        if np.all(exact_rmse <= 1e-9):
            false = np.zeros(shape, dtype=bool)
            return PointProjection(
                control=exact_control,
                fitted_points=exact_fit,
                point_rmse=exact_rmse,
                wrong_chirality=false,
                degenerate=false,
                used_exact_inverse=np.ones(shape, dtype=bool),
            )
    except ValueError:
        pass

    left, predicted_center, right = flat[:, 0:2], flat[:, 2:4], flat[:, 4:6]
    center = np.clip(predicted_center, 0.0, world_size)
    observed_left, observed_right = left - center, right - center
    left_ray, right_ray = center - left, right - center
    cross = left_ray[:, 0] * right_ray[:, 1] - left_ray[:, 1] * right_ray[:, 0]
    wrong_chirality = cross <= 0.0
    degenerate = (
        (np.linalg.norm(left_ray, axis=-1) <= 1e-8)
        | (np.linalg.norm(right_ray, axis=-1) <= 1e-8)
    )

    lower = np.full(count, CHAIN_GRIPPER_OPEN_ANGLE)
    upper = np.full(count, CHAIN_GRIPPER_CLOSED_ANGLE)
    best_joint = (lower + upper) / 2.0
    best_theta = previous[:, 2].copy() if previous is not None else np.zeros(count)
    fractions = np.linspace(0.0, 1.0, grid_size)[None, :]
    rows = np.arange(count)
    for _ in range(refinements):
        candidates = lower[:, None] + (upper - lower)[:, None] * fractions
        radius = 2.0 * CHAIN_GRIPPER_LINK_LEN * np.cos(candidates / 2.0)
        cosine, sine = np.cos(candidates), np.sin(candidates)
        left_x, left_y = -radius * cosine, radius * sine
        right_x, right_y = radius * cosine, radius * sine
        correlation_real = (
            observed_left[:, None, 0] * left_x
            + observed_left[:, None, 1] * left_y
            + observed_right[:, None, 0] * right_x
            + observed_right[:, None, 1] * right_y
        )
        correlation_imag = (
            observed_left[:, None, 1] * left_x
            - observed_left[:, None, 0] * left_y
            + observed_right[:, None, 1] * right_x
            - observed_right[:, None, 0] * right_y
        )
        theta = np.arctan2(correlation_imag, correlation_real)
        if previous is not None:
            theta = np.where(
                np.hypot(correlation_real, correlation_imag) <= 1e-8,
                previous[:, None, 2],
                theta,
            )
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        fit_left_x = cos_theta * left_x - sin_theta * left_y
        fit_left_y = sin_theta * left_x + cos_theta * left_y
        fit_right_x = cos_theta * right_x - sin_theta * right_y
        fit_right_y = sin_theta * right_x + cos_theta * right_y
        error = (
            (observed_left[:, None, 0] - fit_left_x) ** 2
            + (observed_left[:, None, 1] - fit_left_y) ** 2
            + (observed_right[:, None, 0] - fit_right_x) ** 2
            + (observed_right[:, None, 1] - fit_right_y) ** 2
        )
        index = np.argmin(error, axis=1)
        best_joint, best_theta = candidates[rows, index], theta[rows, index]
        lower = candidates[rows, np.maximum(index - 1, 0)]
        upper = candidates[rows, np.minimum(index + 1, grid_size - 1)]

    control = np.column_stack(
        (center, wrap_angle(best_theta), joint_angle_to_grip(best_joint))
    )
    fitted = pose_control_to_points(control, world_size=world_size)
    rmse = np.sqrt(np.mean((fitted - flat) ** 2, axis=-1))
    return PointProjection(
        control=control.reshape(*shape, 4),
        fitted_points=fitted.reshape(*shape, 6),
        point_rmse=rmse.reshape(shape),
        wrong_chirality=wrong_chirality.reshape(shape),
        degenerate=degenerate.reshape(shape),
        used_exact_inverse=np.zeros(shape, dtype=bool),
    )


def point_action_bounds(world_size: float = 512.0) -> dict[str, tuple[float, float]]:
    """Bounds for point mode; free tips may extend outside the arena."""
    reach = 2.0 * CHAIN_GRIPPER_LINK_LEN
    tip = (-reach, world_size + reach)
    center = (0.0, world_size)
    return {
        "left_free_tip_x": tip,
        "left_free_tip_y": tip,
        "middle_joint_x": center,
        "middle_joint_y": center,
        "right_free_tip_x": tip,
        "right_free_tip_y": tip,
    }
