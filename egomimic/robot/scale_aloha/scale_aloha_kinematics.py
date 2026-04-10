"""
Kinematics solver for the Scale/Trossen WXAI dual-arm robot.

Uses a stripped MJCF model (model_scale_wxai.xml) containing only the
kinematic chain.  Both arms share the same single-arm model; they are
distinguished only by the world-frame base offset defined in the full
URDF (external/scale/reference.urdf).

Joint ordering in the MCAP data (per arm):
    shoulder_pan, shoulder_lift, elbow_flex, wrist_angle, wrist_rotate, gripper, gripper_finger

Maps to URDF/MJCF joints:
    joint_0  (shoulder_pan)   – revolute, Z axis
    joint_1  (shoulder_lift)  – revolute, Y axis
    joint_2  (elbow_flex)     – revolute, -Y axis
    joint_3  (wrist_angle)    – revolute, -Y axis
    joint_4  (wrist_rotate)   – revolute, -Z axis
    joint_5  (gripper)        – revolute, X axis   *see note below*

NOTE: The MCAP reports 7 values per arm.  Indices 0-4 are the 5 arm
revolute joints that map 1:1 to joint_0 – joint_4.  Index 5 ("gripper")
maps to joint_5 (the wrist roll).  Index 6 ("gripper_finger") is the
prismatic gripper opening and does NOT participate in FK.

If the mapping above turns out to be wrong after real-data validation,
update MCAP_TO_FK_INDICES accordingly.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from egomimic.robot.kinematics import MinkKinematicsSolver

_RESOURCES_DIR = Path(__file__).resolve().parents[2] / "resources"
DEFAULT_MODEL_PATH = str(_RESOURCES_DIR / "model_scale_wxai.xml")

JOINT_NAMES = [
    "joint_0",
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
]

DEFAULT_VELOCITY_LIMITS = {name: 1.0 for name in JOINT_NAMES}

# Indices into the 7-element MCAP JointState position array that correspond
# to the 6 revolute FK joints.  The 7th value (index 6, "gripper_finger")
# is the prismatic gripper and is not used in FK.
MCAP_TO_FK_INDICES = [0, 1, 2, 3, 4, 5]


class ScaleAlohaMinkKinematicsSolver(MinkKinematicsSolver):
    """Mink-based FK/IK solver for a single WXAI arm."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        eef_link_name: str = "ee_gripper",
        eef_frame_type: str = "site",
        velocity_limits: dict | None = None,
        solver: str = "daqp",
        max_iterations: int = 100,
        position_tolerance: float = 1e-3,
        orientation_tolerance: float = 1e-3,
    ):
        if velocity_limits is None:
            velocity_limits = DEFAULT_VELOCITY_LIMITS.copy()

        super().__init__(
            model_path=model_path,
            base_link_name="base_link",
            eef_link_name=eef_link_name,
            num_joints=6,
            joint_names=JOINT_NAMES,
            eef_frame_type=eef_frame_type,
            velocity_limits=velocity_limits,
            solver=solver,
            max_iterations=max_iterations,
            position_tolerance=position_tolerance,
            orientation_tolerance=orientation_tolerance,
        )


class ScaleAlohaDualArmFK:
    """Convenience wrapper that computes FK for both arms of the WXAI robot.

    Accepts the full 14-element joint vector [left_7 | right_7] produced by
    the MCAP converter and returns a 14-element Cartesian vector:
        [L_xyz(3), L_ypr(3), L_gripper(1), R_xyz(3), R_ypr(3), R_gripper(1)]
    """

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        self.solver = ScaleAlohaMinkKinematicsSolver(model_path=model_path)

    def _extract_fk_joints(self, joints_7: np.ndarray) -> np.ndarray:
        """Extract the 6 revolute joint angles from a 7-element MCAP vector."""
        return joints_7[MCAP_TO_FK_INDICES]

    def fk_single_arm(self, joints_7: np.ndarray) -> np.ndarray:
        """FK for one arm.

        Args:
            joints_7: (7,) array — the 7 MCAP joint values for one arm.

        Returns:
            (7,) array — [xyz(3), ypr(3), gripper(1)]
        """
        fk_joints = self._extract_fk_joints(joints_7)
        pos, rot = self.solver.fk_single_pos_rot(fk_joints)
        ypr = rot.as_euler("ZYX", degrees=False)
        gripper = np.array([joints_7[6]])
        return np.concatenate([pos, ypr, gripper])

    def fk_bimanual(self, joints_14: np.ndarray) -> np.ndarray:
        """FK for both arms.

        Args:
            joints_14: (14,) array — [left_7 | right_7] from MCAP.

        Returns:
            (14,) array — [L_xyz, L_ypr, L_grip, R_xyz, R_ypr, R_grip]
        """
        left = self.fk_single_arm(joints_14[:7])
        right = self.fk_single_arm(joints_14[7:])
        return np.concatenate([left, right])

    def fk_bimanual_batch(self, joints_batch: np.ndarray) -> np.ndarray:
        """Batch FK for a (T, 14) joint array.

        Returns:
            (T, 14) Cartesian array.
        """
        T = joints_batch.shape[0]
        out = np.empty((T, 14), dtype=np.float32)
        for t in range(T):
            out[t] = self.fk_bimanual(joints_batch[t])
        return out
