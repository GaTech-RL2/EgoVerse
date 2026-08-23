"""Fail-closed validation for commands crossing into live robot control."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def validate_action_vector(action, expected_dim: int) -> np.ndarray:
    action = np.asarray(action, dtype=np.float64)
    if action.shape != (expected_dim,):
        raise ValueError(
            f"Expected rollout action shape ({expected_dim},), got {action.shape}"
        )
    if not np.all(np.isfinite(action)):
        raise ValueError("Rollout action contains NaN or infinity")
    return action


def validate_gripper(gripper: float) -> None:
    if not np.isfinite(gripper) or not 0.0 <= float(gripper) <= 1.0:
        raise ValueError(f"Normalized gripper command must be in [0, 1], got {gripper}")


def validate_joint_command(
    command,
    current,
    joint_min,
    joint_max,
    max_delta,
) -> np.ndarray:
    command = validate_action_vector(command, 7)
    current = validate_action_vector(current, 7)
    validate_gripper(command[6])
    joints = command[:6]
    joint_min = np.asarray(joint_min, dtype=np.float64)
    joint_max = np.asarray(joint_max, dtype=np.float64)
    max_delta = np.asarray(max_delta, dtype=np.float64)
    if joint_min.shape != (6,) or joint_max.shape != (6,) or max_delta.shape != (6,):
        raise ValueError("Joint safety limits must each have shape (6,)")
    if np.any(joints < joint_min) or np.any(joints > joint_max):
        raise ValueError("Joint command exceeds the configured robot position limits")
    delta = np.abs(joints - current[:6])
    if np.any(delta > max_delta):
        raise ValueError(
            "Joint command exceeds the preview-window delta limit: "
            f"max requested={delta.max():.4f}, max allowed={max_delta.max():.4f} rad"
        )
    return command


def validate_cartesian_command(
    command,
    current_pose,
    *,
    max_translation_step_m: float,
    max_rotation_step_rad: float,
) -> np.ndarray:
    command = validate_action_vector(command, 7)
    current_pose = validate_action_vector(current_pose, 7)
    validate_gripper(command[6])
    translation_step = float(np.linalg.norm(command[:3] - current_pose[:3]))
    if translation_step > max_translation_step_m:
        raise ValueError(
            f"Cartesian translation jump {translation_step:.4f} m exceeds "
            f"{max_translation_step_m:.4f} m"
        )
    current_rotation = Rotation.from_euler("ZYX", current_pose[3:6])
    target_rotation = Rotation.from_euler("ZYX", command[3:6])
    rotation_step = float((target_rotation * current_rotation.inv()).magnitude())
    if rotation_step > max_rotation_step_rad:
        raise ValueError(
            f"Cartesian rotation jump {rotation_step:.4f} rad exceeds "
            f"{max_rotation_step_rad:.4f} rad"
        )
    return command
