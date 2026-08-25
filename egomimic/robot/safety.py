"""Fail-closed validation for commands crossing into live robot control."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


class CartesianTranslationConfirmationRequired(ValueError):
    """A finite Cartesian command is safe only with attended confirmation."""

    def __init__(
        self,
        translation_step_m: float,
        automatic_limit_m: float,
        hard_limit_m: float,
    ):
        self.translation_step_m = float(translation_step_m)
        self.automatic_limit_m = float(automatic_limit_m)
        self.hard_limit_m = float(hard_limit_m)
        super().__init__(
            f"Cartesian translation jump {self.translation_step_m:.4f} m exceeds "
            f"the automatic limit {self.automatic_limit_m:.4f} m; attended "
            f"confirmation is required below the hard limit "
            f"{self.hard_limit_m:.4f} m"
        )


class CartesianTranslationHardLimitExceeded(ValueError):
    """A Cartesian command must not be sent because it reached the hard limit."""

    def __init__(self, translation_step_m: float, hard_limit_m: float):
        self.translation_step_m = float(translation_step_m)
        self.hard_limit_m = float(hard_limit_m)
        super().__init__(
            f"Cartesian translation jump {self.translation_step_m:.4f} m reaches "
            f"or exceeds the hard limit {self.hard_limit_m:.4f} m"
        )


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
    *,
    allow_preview_window_jump: bool = False,
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
    if np.any(delta > max_delta) and not allow_preview_window_jump:
        raise ValueError(
            "Joint command exceeds the preview-window delta limit: "
            f"max requested={delta.max():.4f}, max allowed={max_delta.max():.4f} rad"
        )
    return command


def validate_cartesian_command(
    command,
    current_pose,
    *,
    max_translation_step_m: float | None,
    max_rotation_step_rad: float | None,
    hard_max_translation_step_m: float | None = None,
    allow_soft_translation_jump: bool = False,
) -> np.ndarray:
    command = validate_action_vector(command, 7)
    current_pose = validate_action_vector(current_pose, 7)
    validate_gripper(command[6])
    if max_translation_step_m is not None and max_translation_step_m <= 0.0:
        raise ValueError("Cartesian translation limit must be positive")
    if hard_max_translation_step_m is not None:
        if hard_max_translation_step_m <= 0.0:
            raise ValueError("Hard Cartesian translation limit must be positive")
        if (
            max_translation_step_m is not None
            and hard_max_translation_step_m <= max_translation_step_m
        ):
            raise ValueError(
                "Hard Cartesian translation limit must exceed the automatic limit"
            )
    translation_step = float(np.linalg.norm(command[:3] - current_pose[:3]))
    if (
        hard_max_translation_step_m is not None
        and translation_step >= hard_max_translation_step_m
    ):
        raise CartesianTranslationHardLimitExceeded(
            translation_step,
            hard_max_translation_step_m,
        )
    if (
        hard_max_translation_step_m is None
        and max_translation_step_m is not None
        and translation_step > max_translation_step_m
    ):
        raise ValueError(
            f"Cartesian translation jump {translation_step:.4f} m exceeds "
            f"{max_translation_step_m:.4f} m"
        )
    if max_rotation_step_rad is not None:
        current_rotation = Rotation.from_euler("ZYX", current_pose[3:6])
        target_rotation = Rotation.from_euler("ZYX", command[3:6])
        rotation_step = float((target_rotation * current_rotation.inv()).magnitude())
        if rotation_step > max_rotation_step_rad:
            raise ValueError(
                f"Cartesian rotation jump {rotation_step:.4f} rad exceeds "
                f"{max_rotation_step_rad:.4f} rad"
            )
    if (
        hard_max_translation_step_m is not None
        and max_translation_step_m is not None
        and translation_step > max_translation_step_m
        and not allow_soft_translation_jump
    ):
        raise CartesianTranslationConfirmationRequired(
            translation_step,
            max_translation_step_m,
            hard_max_translation_step_m,
        )
    return command
