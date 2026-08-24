from types import SimpleNamespace

import numpy as np
import pytest

from egomimic.robot.eva.eva_ws.src.eva.robot_interface import (
    ARXInterface,
    OfflineARXInterface,
)
from egomimic.robot.safety import (
    CartesianTranslationConfirmationRequired,
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
    command[6] = -1e-6
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


def test_cartesian_translation_confirmation_and_hard_limit_boundaries():
    current = np.zeros(7)
    current[6] = 0.5
    command = current.copy()

    command[0] = 0.08
    validate_cartesian_command(
        command,
        current,
        max_translation_step_m=0.08,
        hard_max_translation_step_m=0.15,
        max_rotation_step_rad=0.5,
    )

    command[0] = np.nextafter(0.08, np.inf)
    with pytest.raises(CartesianTranslationConfirmationRequired) as exc_info:
        validate_cartesian_command(
            command,
            current,
            max_translation_step_m=0.08,
            hard_max_translation_step_m=0.15,
            max_rotation_step_rad=0.5,
        )
    assert exc_info.value.translation_step_m == pytest.approx(command[0])
    assert exc_info.value.automatic_limit_m == pytest.approx(0.08)
    assert exc_info.value.hard_limit_m == pytest.approx(0.15)

    command[0] = np.nextafter(0.15, 0.0)
    validate_cartesian_command(
        command,
        current,
        max_translation_step_m=0.08,
        hard_max_translation_step_m=0.15,
        max_rotation_step_rad=0.5,
        allow_soft_translation_jump=True,
    )

    command[0] = 0.15
    with pytest.raises(ValueError, match="hard limit"):
        validate_cartesian_command(
            command,
            current,
            max_translation_step_m=0.08,
            hard_max_translation_step_m=0.15,
            max_rotation_step_rad=0.5,
            allow_soft_translation_jump=True,
        )


def test_cartesian_confirmation_never_bypasses_other_hard_checks():
    current = np.zeros(7)
    current[6] = 0.5
    command = current.copy()
    command[0] = 0.09
    command[3] = 0.6
    with pytest.raises(ValueError, match="rotation jump") as exc_info:
        validate_cartesian_command(
            command,
            current,
            max_translation_step_m=0.08,
            hard_max_translation_step_m=0.15,
            max_rotation_step_rad=0.5,
        )
    assert not isinstance(exc_info.value, CartesianTranslationConfirmationRequired)

    command[3] = 0.0
    command[6] = 1.1
    with pytest.raises(ValueError, match="gripper"):
        validate_cartesian_command(
            command,
            current,
            max_translation_step_m=0.08,
            hard_max_translation_step_m=0.15,
            max_rotation_step_rad=0.5,
            allow_soft_translation_jump=True,
        )


def test_live_pose_validation_requires_one_shot_soft_jump_override():
    interface = ARXInterface.__new__(ARXInterface)
    interface.get_pose_6d = lambda arm: np.zeros(6)
    interface.get_joints = lambda arm: np.asarray([0, 0, 0, 0, 0, 0, 0.5])
    command = np.asarray([0.09, 0, 0, 0, 0, 0, 0.5])

    with pytest.raises(CartesianTranslationConfirmationRequired):
        interface.validate_pose_command(command, "right")
    interface.validate_pose_command(
        command,
        "right",
        allow_soft_translation_jump=True,
    )

    command[0] = 0.15
    with pytest.raises(ValueError, match="hard limit"):
        interface.validate_pose_command(
            command,
            "right",
            allow_soft_translation_jump=True,
        )


def test_set_pose_propagates_only_the_explicit_soft_jump_override():
    interface = ARXInterface.__new__(ARXInterface)
    validations = []
    sent = []

    def validate(pose, arm, *, allow_soft_translation_jump=False):
        validations.append((arm, allow_soft_translation_jump))
        return np.asarray(pose)

    interface.validate_pose_command = validate
    interface.solve_ik = lambda pose, arm: np.zeros(6)
    interface.set_joints = lambda joints, arm: sent.append((arm, joints.copy()))
    command = np.asarray([0.09, 0, 0, 0, 0, 0, 0.5])

    interface.set_pose(command, "right", allow_soft_translation_jump=True)

    assert validations == [("right", True)]
    assert len(sent) == 1
    np.testing.assert_array_equal(sent[0][1], np.asarray([0, 0, 0, 0, 0, 0, 0.5]))


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


def test_live_joint_command_maps_normalized_gripper_to_configured_endpoints():
    sent = []
    controller = SimpleNamespace(
        get_joint_state=lambda: SimpleNamespace(timestamp=2.0),
        set_joint_cmd=sent.append,
    )
    interface = ARXInterface.__new__(ARXInterface)
    interface.controller = {"left": controller}
    interface.arx_joint_state = lambda pos, vel, torque, timestamp: SimpleNamespace(
        pos=pos,
        vel=vel,
        torque=torque,
        timestamp=timestamp,
    )
    interface.gripper_close = {"left": -0.012}
    configured_open = 0.09445506587172998
    interface.gripper_width = {"left": configured_open + 0.012}
    interface.ts_offset = 0.2
    interface.validate_joints_command = lambda command, arm: np.asarray(command)

    interface.set_joints(np.zeros(7), "left")
    open_command = np.zeros(7)
    open_command[6] = 1.0
    interface.set_joints(open_command, "left")

    assert len(sent) == 2
    assert sent[0].gripper_pos == pytest.approx(-0.012)
    assert sent[1].gripper_pos == pytest.approx(configured_open)
