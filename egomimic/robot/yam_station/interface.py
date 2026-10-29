"""Shared Eva/Yam contract. Vectors are left-7, right-7; grippers are [0, 1]."""

from typing import Protocol

import numpy as np
from scipy.spatial.transform import Rotation

ARM_OFFSET = {"left": 0, "right": 7}


def pose_matrix(pose):
    pose = np.asarray(pose, dtype=float)
    if pose.shape != (6,) or not np.isfinite(pose).all():
        raise ValueError("Expected xyz + intrinsic ZYX Euler angles (radians)")
    matrix = np.eye(4)
    matrix[:3, 3] = pose[:3]
    matrix[:3, :3] = Rotation.from_euler("ZYX", pose[3:]).as_matrix()
    return matrix


def pose_vector(matrix):
    return np.r_[matrix[:3, 3], Rotation.from_matrix(matrix[:3, :3]).as_euler("ZYX")]


def joint_vector(value):
    value = np.asarray(value, dtype=float)
    if value.shape != (7,) or not np.isfinite(value).all() or not 0 <= value[6] <= 1:
        raise ValueError("Expected six finite joints and gripper opening in [0, 1]")
    return value.copy()


class RobotInterface(Protocol):
    """Arm poses are in each arm's base frame; camera images are uint8 BGR.

    get_obs returns joint_positions and ee_poses (14D) and named images.
    Unselected arms occupy zero-filled slots. Implementations own hardware cleanup.
    """

    arms: list[str]
    camera_res: dict[str, tuple[int, int]]
    recorders: dict

    def forward_kinematics(self, joints, arm: str) -> np.ndarray: ...
    def get_obs(self) -> dict: ...
    def get_joints(self, arm: str) -> np.ndarray: ...
    def get_pose(self, arm: str, se3: bool = False): ...
    def solve_ik(self, ee_pose: np.ndarray, arm: str): ...
    def set_joints(self, desired_position: np.ndarray, arm: str): ...
    def set_home(self): ...
    def close(self): ...


def create_robot(config) -> RobotInterface:
    config = dict(config)
    kind = config.pop("kind")
    arms = list(config.pop("arms"))
    if not arms or len(set(arms)) != len(arms) or not set(arms) <= ARM_OFFSET.keys():
        raise ValueError("arms must select left, right, or both without duplicates")
    if kind == "eva":
        from egomimic.robot.eva.eva_ws.src.eva.robot_interface import ARXInterface

        return ARXInterface(arms=arms, **config)
    if kind == "yam":
        from egomimic.robot.yam.interface import YamInterface

        return YamInterface(arms=arms, **config)
    raise ValueError(f"Unsupported robot kind: {kind}")
