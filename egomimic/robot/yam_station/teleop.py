"""Quest clutch adapter and shared Eva/Yam command loop.

The pose mapping delegates to the byte-for-byte ``yam-pipeline`` mapper in
``egomimic.robot.yam.quest_mapper``. Yam additionally exposes the reference
streaming IK path; Eva keeps using its existing robot-interface IK.
"""

import numpy as np

from egomimic.robot.yam.quest_mapper import QuestBimanualMapper, _input_value
from egomimic.robot.yam_station.interface import (
    ARM_OFFSET,
    joint_vector,
    pose_matrix,
    pose_vector,
)

ARM_TO_SIDE = {"left": "l", "right": "r"}


def analog(buttons, key):
    return _input_value((buttons or {}).get(key, 0.0))


class ButtonEdge:
    def __init__(self, key):
        self.key, self.down = key, False

    def pressed(self, buttons):
        down = analog(buttons, self.key) > 0.5
        rising, self.down = down and not self.down, down
        return rising


def rigid_transform(value):
    matrix = np.asarray(value, dtype=float)
    if (
        matrix.shape != (4, 4)
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix[3], [0, 0, 0, 1])
        or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-3)
        or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-3)
    ):
        raise ValueError("Expected a finite rigid 4x4 pose")
    return matrix.copy()


class WorldFrameTeleop:
    """Arm-name adapter around yam-pipeline's exact world-frame mapper."""

    def __init__(
        self,
        arms,
        translation_gain,
        orientation_gain,
        target_lpf,
        max_controller_step_m,
        max_controller_step_deg,
        headset_yaw_degrees=0.0,
        orientation_rx_degrees=0.0,
        side_yaw=None,
    ):
        self.arms = tuple(arms)
        if not self.arms or not set(self.arms) <= ARM_TO_SIDE.keys():
            raise ValueError("Teleop arms must select left, right, or both")
        normalized_yaw = {}
        for name, yaw in (side_yaw or {}).items():
            side = ARM_TO_SIDE.get(name, name)
            if side not in ("l", "r"):
                raise ValueError(f"Unknown controller side: {name!r}")
            normalized_yaw[side] = float(yaw)
        self.side_for_arm = {arm: ARM_TO_SIDE[arm] for arm in self.arms}
        self.mapper = QuestBimanualMapper(
            translation_gain=translation_gain,
            orientation_gain=orientation_gain,
            headset_yaw_degrees=headset_yaw_degrees,
            side_yaw=normalized_yaw,
            sides=tuple(self.side_for_arm.values()),
            target_lpf=target_lpf,
            orientation_rx_degrees=orientation_rx_degrees,
            max_controller_step_m=max_controller_step_m,
            max_controller_step_deg=max_controller_step_deg,
        )
        self._joy = {
            arm: ButtonEdge("LJ" if arm == "left" else "RJ") for arm in self.arms
        }
        self.anchored = set()

    def reset(self):
        self.mapper.reset_clutches()
        self.anchored = set()

    def will_clutch(self, arm, buttons):
        return self.mapper.will_clutch(self.side_for_arm[arm], buttons or {})

    def update_states(self, poses, buttons, ee_poses):
        buttons = buttons or {}
        will_anchor = {
            arm: (
                (self.will_clutch(arm, buttons) or self._joy[arm].pressed(buttons))
                and analog(buttons, "LG" if arm == "left" else "RG") > 0.5
            )
            for arm in self.arms
        }
        side_poses = {
            self.side_for_arm[arm]: rigid_transform(ee_poses[arm]) for arm in self.arms
        }
        mapped = self.mapper.update(poses or {}, buttons, side_poses)
        states = {arm: mapped[self.side_for_arm[arm]] for arm in self.arms}
        self.anchored = {
            arm
            for arm, state in states.items()
            if will_anchor[arm] and state["engaged"]
        }
        return states

    def update(self, poses, buttons, measured):
        states = self.update_states(poses, buttons, measured)
        return {
            arm: np.r_[pose_vector(state["target_pose"]), state["gripper"]]
            for arm, state in states.items()
            if state["engaged"]
        }


class TeleopControl:
    def __init__(self, robot, mapper, frequency, max_joint_velocity):
        self.robot, self.mapper = robot, mapper
        self.frequency = float(frequency)
        self.max_joint_velocity = float(max_joint_velocity)
        self.step_limit = self.max_joint_velocity / self.frequency
        if not np.isfinite(self.step_limit) or self.step_limit <= 0:
            raise ValueError("Joint velocity and frequency must be positive")
        validate = getattr(robot, "validate_teleop_config", None)
        if validate is not None:
            validate(self.frequency, self.max_joint_velocity)
        self.last = None

    def reset(self):
        self.last = None
        self.mapper.reset()

    def _streaming_step(self, poses, buttons, obs):
        """Match yam-pipeline's command-seeded FK and local streaming IK loop."""
        for arm in self.robot.arms:
            if self.mapper.will_clutch(arm, buttons):
                offset = ARM_OFFSET[arm]
                self.last[offset : offset + 7] = obs["joint_positions"][
                    offset : offset + 7
                ]
        anchor_poses = {
            arm: self.robot.teleop_fk(
                self.last[ARM_OFFSET[arm] : ARM_OFFSET[arm] + 7], arm
            )
            for arm in self.robot.arms
        }
        states = self.mapper.update_states(poses, buttons, anchor_poses)
        commands = self.last.copy()
        for arm in self.robot.arms:
            offset = ARM_OFFSET[arm]
            command = commands[offset : offset + 7].copy()
            state = states[arm]
            if state["engaged"]:
                ok, solved = self.robot.solve_teleop_ik(
                    state["target_pose"], command, arm
                )
                solved = np.asarray(solved, dtype=float)
                if ok and solved.shape == (6,) and np.isfinite(solved).all():
                    command[:6] += np.clip(
                        solved - command[:6], -self.step_limit, self.step_limit
                    )
            command[6] = float(np.clip(state["gripper"], 0.0, 1.0))
            commands[offset : offset + 7] = joint_vector(command)

        # Solve both arms first, then issue their targets back-to-back.
        for arm in self.robot.arms:
            offset = ARM_OFFSET[arm]
            self.robot.set_joints(commands[offset : offset + 7], arm)
        ee_commands = np.zeros(14)
        for arm in self.robot.arms:
            offset = ARM_OFFSET[arm]
            command = commands[offset : offset + 7]
            ee_commands[offset : offset + 7] = np.r_[
                pose_vector(self.robot.teleop_fk(command, arm)), command[6]
            ]
        self.last = commands
        return commands.copy(), ee_commands

    def _generic_step(self, poses, buttons, obs):
        measured = {
            arm: pose_matrix(obs["ee_poses"][ARM_OFFSET[arm] : ARM_OFFSET[arm] + 6])
            for arm in self.robot.arms
        }
        targets = self.mapper.update(poses, buttons, measured)
        for arm in self.mapper.anchored:
            offset = ARM_OFFSET[arm]
            self.last[offset : offset + 7] = obs["joint_positions"][offset : offset + 7]
        commands, ee_commands = self.last.copy(), np.zeros(14)
        for arm in self.robot.arms:
            offset = ARM_OFFSET[arm]
            if arm in targets:
                target = targets[arm]
                try:
                    solved = np.asarray(
                        self.robot.solve_ik(target[:6], arm), dtype=float
                    )
                    if solved.shape != (6,) or not np.isfinite(solved).all():
                        raise ValueError("IK did not return six finite joints")
                    command = joint_vector(np.r_[solved, target[6]])
                except (TypeError, ValueError, RuntimeError):
                    command = commands[offset : offset + 7].copy()
                command[:6] = commands[offset : offset + 6] + np.clip(
                    command[:6] - commands[offset : offset + 6],
                    -self.step_limit,
                    self.step_limit,
                )
                commands[offset : offset + 7] = command
            command = joint_vector(commands[offset : offset + 7])
            self.robot.set_joints(command, arm)
            ee_commands[offset : offset + 7] = np.r_[
                self.robot.forward_kinematics(command[:6], arm), command[6]
            ]
        self.last = commands
        return commands.copy(), ee_commands

    def step(self, poses, buttons, obs):
        if self.last is None:
            self.last = np.asarray(obs["joint_positions"], dtype=float).copy()
        if hasattr(self.robot, "solve_teleop_ik"):
            return self._streaming_step(poses, buttons, obs)
        return self._generic_step(poses, buttons, obs)
