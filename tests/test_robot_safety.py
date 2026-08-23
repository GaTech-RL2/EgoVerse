import numpy as np
import pytest

from egomimic.robot.eva.eva_ws.src.eva.robot_interface import OfflineARXInterface
from egomimic.robot.safety import (
    validate_action_vector,
    validate_cartesian_command,
    validate_joint_command,
)


def test_action_vector_rejects_bad_shape_and_nonfinite_values():
    with pytest.raises(ValueError, match="shape"):
        validate_action_vector(np.zeros(6), 7)
    action = np.zeros(7)
    action[2] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        validate_action_vector(action, 7)


def test_joint_command_enforces_gripper_limits_and_maximum_delta():
    current = np.zeros(7)
    current[6] = 0.5
    command = current.copy()
    command[0] = 0.1
    validate_joint_command(command, current, -np.ones(6), np.ones(6), np.ones(6) * 0.2)
    command[0] = 0.3
    with pytest.raises(ValueError, match="delta"):
        validate_joint_command(
            command, current, -np.ones(6), np.ones(6), np.ones(6) * 0.2
        )
    command = current.copy()
    command[6] = 1.1
    with pytest.raises(ValueError, match="gripper"):
        validate_joint_command(
            command, current, -np.ones(6), np.ones(6), np.ones(6) * 0.2
        )


def test_cartesian_command_uses_geodesic_rotation_delta():
    current = np.zeros(7)
    current[6] = 0.5
    command = current.copy()
    command[:3] = [0.01, 0.02, 0.01]
    command[3] = 0.1
    validate_cartesian_command(
        command,
        current,
        max_translation_step_m=0.05,
        max_rotation_step_rad=0.2,
    )
    command[3] = 0.4
    with pytest.raises(ValueError, match="rotation jump"):
        validate_cartesian_command(
            command,
            current,
            max_translation_step_m=0.05,
            max_rotation_step_rad=0.2,
        )


def test_offline_home_resets_state_and_episode_cursor():
    interface = OfflineARXInterface.__new__(OfflineARXInterface)
    interface._joint_positions = {arm: np.ones(7) for arm in ("left", "right")}
    interface._ee_pose = {arm: np.ones(7) for arm in ("left", "right")}
    interface.frame_idx = 17
    interface.set_home()
    assert interface.frame_idx == 0
    for arm in ("left", "right"):
        np.testing.assert_array_equal(interface._joint_positions[arm], np.zeros(7))
        np.testing.assert_array_equal(interface._ee_pose[arm], np.zeros(7))
