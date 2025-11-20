"""
Generic kinematics solver.
"""

from pyexpat import model
from typing import List, Tuple, Optional
import numpy as np
from sklearn import base
import os
import re
import tempfile
from scipy.spatial.transform import Rotation as R
from pathlib import Path

try:
    from trac_ik import TracIK

    _HAS_TRACIK = True
except ImportError:
    _HAS_TRACIK = False

import mink
import mujoco


class TracKinematicsSolver:
    """
    Generic kinematics solver using PyTracik

    Args:
        urdf_path: Path to the URDF file
        base_link_name: Name of the base link in the urdf
        ee_link_name: Name of the link you are solving for
    """

    def __init__(
        self, urdf_path: str, base_link_name: str, eef_link_name: str, num_joints: int
    ):
        if not _HAS_TRACIK:
            raise ImportError(
                "trac_ik module not found. Install it from pytracik"
                "or use MinkKinematicsSolver instead."
            )

        self.num_joints = num_joints
        self.urdf_path = self._resolve_urdf_path(urdf_path)
        self.solver = self.kinematics_solver = TracIK(
            base_link_name=base_link_name,
            tip_link_name=eef_link_name,
            urdf_path=self.urdf_path,
        )

    def _resolve_urdf_path(self, urdf_path: str) -> str:
        """
        Resolve URDF path, handling package:// URIs if present.
        """
        # Make path absolute
        if not os.path.isabs(urdf_path):
            urdf_path = os.path.abspath(urdf_path)

        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF path does not exist: {urdf_path}")

        # Directory where the original URDF sits
        model_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(urdf_path)))
        print(f"Model root path is {model_root_dir}")

        try:
            with open(urdf_path, "r", encoding="utf-8") as f:
                urdf_text = f.read()

            if "package://" in urdf_text:
                # Replace every package://<pkg_name>/... with model_root_dir/...
                # This is a cheap fallback when you don't have ROS package paths
                def _replace_pkg(match: re.Match) -> str:
                    # match.group() is like "package://X5A/"
                    return model_root_dir + "/"

                urdf_text = re.sub(r"package://[^/]+/", _replace_pkg, urdf_text)

                tmp = tempfile.NamedTemporaryFile(
                    delete=False, suffix=".urdf", mode="w"
                )
                tmp.write(urdf_text)
                tmp.flush()
                tmp.close()
                return tmp.name
        except Exception:
            # If anything goes wrong, just return the original path
            return urdf_path

        return urdf_path

    def ik(self, pos_xyz, rot_mat, cur_jnts):
        """
        Inverse kinematics

        Args:
            pos_xyz: numpy array of xyz
            rot_mat: 3x3 rotation matrix in numpy
            cur_jnts: numpy array of length num_joints

        Return:
            solved_jnts: numpy array of length num_joints
        """
        return self.kinematics_solver.ik(
            pos_xyz, rot_mat, seed_jnt_values=cur_jnts[: self.num_joints]
        )

    def fk(self, jnts):
        """
        Forward Kinematics

        Args:
            jnts: numpy array of length num_joints

        Return:
            pos: xyz, rot: Scipy rotation matrix
        """
        return self.kinematics_solver.fk(jnts[: self.num_joints])


class MinkKinematicsSolver:
    """
    Kinematics solver using mink (MuJoCo-based IK).

    This solver provides a similar interface to KinematicsSolver but uses
    mink's optimization-based IK.

    Args:
        model_path: Path to the URDF/XML file
        base_link_name: Name of the base link in the urdf
        eef_link_name: Name of the end-effector link/site
        num_joints: Number of joints to control
        joint_names: List of joint names to control (in order)
        eef_frame_type: Type of end-effector frame ("site" or "body")
        velocity_limits: Optional dict of joint velocity limits
        solver: QP solver to use ("daqp", "quadprog", "proxqp", etc.)
    """

    def __init__(
        self,
        model_path: str,
        base_link_name: str,
        eef_link_name: str,
        num_joints: int,
        joint_names: List[str],
        eef_frame_type: str = "site",
        velocity_limits: Optional[dict] = None,
        solver: str = "daqp",
        max_iterations: int = 100,
        position_tolerance: float = 1e-3,
        orientation_tolerance: float = 1e-3,
    ):
        self.num_joints = num_joints
        self.joint_names = joint_names
        self.eef_link_name = eef_link_name
        self.eef_frame_type = eef_frame_type
        self.solver = solver
        self.max_iterations = max_iterations
        self.position_tolerance = position_tolerance
        self.orientation_tolerance = orientation_tolerance

        # Convert URDF to MuJoCo XML or load directly
        self.model_path = self._resolve_model_path(model_path)

        # Load MuJoCo model
        try:
            self.model = mujoco.MjModel.from_xml_path(self.model_path)
        except:
            # If direct loading fails, try creating a scene XML
            self.model = self._create_mujoco_model_from_urdf(model_path)

        self.data = mujoco.MjData(self.model)

        # Get joint indices
        self.dof_ids = np.array([self.model.joint(name).id for name in joint_names])

        # Set up velocity limits
        if velocity_limits is None:
            velocity_limits = {name: 0.5 for name in joint_names}
        self.velocity_limits = velocity_limits

        # Create mink configuration
        self.configuration = mink.Configuration(self.model)

        # Define IK tasks
        # Position cost is higher because position errors are typically in meters (0.001-0.1m)
        # while orientation errors are in radians (0.001-0.1 rad), so we need to balance them
        # Using higher position cost to prioritize position convergence
        self.ee_task = mink.FrameTask(
            frame_name=eef_link_name,
            frame_type=eef_frame_type,
            position_cost=10.0,  # Increased to prioritize position convergence
            orientation_cost=1.0,
            lm_damping=0.1,  # Reduced from 1.0 to allow faster convergence
        )

        self.posture_task = mink.PostureTask(self.model, cost=1e-3)

        self.tasks = [self.ee_task, self.posture_task]

        # Define limits
        self.limits = [
            mink.ConfigurationLimit(model=self.model),
            mink.VelocityLimit(self.model, velocity_limits),
        ]

    def _resolve_model_path(self, model_path: str) -> str:
        """Resolve URDF path, handling package:// URIs if present."""
        if not os.path.isabs(model_path):
            model_path = os.path.abspath(model_path)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"URDF path does not exist: {model_path}")

        return model_path

    def _create_mujoco_model_from_urdf(self, model_path: str):
        """Create a MuJoCo model from URDF (simplified version)."""
        # For now, just try to load the URDF directly
        # In production, you might need to convert URDF to MJCF
        raise NotImplementedError(
            "Direct URDF loading failed. Please provide a MuJoCo XML file or implement URDF conversion."
        )

    def ik(self, pos_xyz, rot_mat, cur_jnts, dt=0.05):
        """
        Inverse kinematics using mink.

        Args:
            pos_xyz: numpy array of xyz position (3,)
            rot_mat: 3x3 rotation matrix in numpy
            cur_jnts: numpy array of current joint values (num_joints,)
            dt: time step for integration (default 0.05 for faster convergence)

        Return:
            solved_jnts: numpy array of joint values (num_joints,) or None if no solution
        """
        # Convert rotation matrix to quaternion (w, x, y, z)
        scipy_rot = R.from_matrix(rot_mat)
        quat_xyzw = scipy_rot.as_quat()  # Returns [x, y, z, w]
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

        # Set target pose
        target_transform = mink.SE3.from_rotation_and_translation(
            mink.SO3(quat_wxyz), pos_xyz
        )
        self.ee_task.set_target(target_transform)

        # Initialize configuration from current joints
        # Reset entire configuration first to avoid stale values
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.dof_ids] = cur_jnts[: self.num_joints]
        mujoco.mj_forward(self.model, self.data)
        self.configuration.update(self.data.qpos)

        # Update posture task to bias towards current configuration
        self.posture_task.set_target_from_configuration(self.configuration)

        # Adaptive damping: start with lower damping for faster convergence,
        # increase if we're close to target to avoid overshoot
        best_solution = None
        best_error = float("inf")

        # Solve IK iteratively
        for i in range(self.max_iterations):
            # Use adaptive damping: lower when far, higher when close
            current_err = self.ee_task.compute_error(self.configuration)
            current_pos_err = np.linalg.norm(current_err[:3])
            current_ori_err = np.linalg.norm(current_err[3:])
            total_error = current_pos_err + current_ori_err

            # Track best solution
            if total_error < best_error:
                best_error = total_error
                best_solution = self.configuration.q[self.dof_ids].copy()

            # Adaptive damping: lower when far from target, higher when close
            if total_error > 0.01:
                damping = 1e-4  # Lower damping for faster convergence when far
            else:
                damping = 1e-3  # Higher damping when close to avoid overshoot

            vel = mink.solve_ik(
                self.configuration,
                self.tasks,
                dt,
                self.solver,
                limits=self.limits,
                damping=damping,
            )
            self.configuration.integrate_inplace(vel, dt)

            # Check convergence
            if (
                current_pos_err < self.position_tolerance
                and current_ori_err < self.orientation_tolerance
            ):
                # Converged successfully
                return self.configuration.q[self.dof_ids]

        # Return best solution found (even if not fully converged)
        if best_solution is not None:
            return best_solution
        return self.configuration.q[self.dof_ids]
    
    def fk_single_pos_rot(self, jnts):
        """
        Forward Kinematics using MuJoCo.

        Args:
            jnts: numpy array of joint values (num_joints,)

        Return:
            pos: xyz position (numpy array)
            rot: scipy Rotation object
        """
        # Set joint positions
        self.data.qpos[self.dof_ids] = jnts[: self.num_joints]

        # Compute forward kinematics
        mujoco.mj_forward(self.model, self.data)

        # Get end-effector pose
        if self.eef_frame_type == "site":
            site_id = self.model.site(self.eef_link_name).id
            pos = self.data.site_xpos[site_id].copy()
            rot_mat = self.data.site_xmat[site_id].reshape(3, 3).copy()
        elif self.eef_frame_type == "body":
            body_id = self.model.body(self.eef_link_name).id
            pos = self.data.xpos[body_id].copy()
            rot_mat = self.data.xmat[body_id].reshape(3, 3).copy()
        else:
            raise ValueError(f"Unknown frame type: {self.eef_frame_type}")

        # Convert to scipy Rotation
        rot = R.from_matrix(rot_mat)

        return pos, rot

    def fk_6dof(self, jnts):
        """
        Forward Kinematics for 6DOF joints.
        """
        pos, rot = self.fk_single_pos_rot(jnts[:6])
        ypr = rot.as_euler("ZYX", degrees=False)
        return np.concatenate([pos, ypr], axis=0)
    
    def fk_pos(self, jnts):
        """
        Forward Kinematics for position only.
        """
        if jnts.ndim == 1:
            return self.fk_single_pos_rot(jnts)[0]
        elif jnts.ndim == 2:
            return np.stack([self.fk_single_pos_rot(jnts[i])[0] for i in range(jnts.shape[0])], axis=0)
        else:
            raise ValueError(f"Unknown joint shape: {jnts.shape}")
    
    def fk(self, jnts):
        """
        Same format as aloha_fk.fk(qpos)
        """
        if jnts.ndim == 1:
            if jnts.shape[-1] == 6:
                return self.fk_single_pos_rot(jnts)
            elif jnts.shape[-1] == 7:
                return self.fk_single_pos_rot(jnts)
            else:
                raise ValueError(f"Unknown joint shape: {jnts.shape}")
        elif jnts.ndim == 2:
            if jnts.shape[-1] == 6:
                return np.stack([self.fk_single_pos_rot(jnts[i]) for i in range(jnts.shape[0])], axis=0)
            elif jnts.shape[-1] == 7:
                return np.stack([self.fk_single_pos_rot(jnts[i]) for i in range(jnts.shape[0])], axis=0)
            else:
                raise ValueError(f"Unknown joint shape: {jnts.shape}")
        else:
            raise ValueError(f"Unknown joint shape: {jnts.shape}")
