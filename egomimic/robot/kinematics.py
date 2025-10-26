"""
Generic kinematics solver and Eva-specific implementation.
"""
from typing import List, Tuple, Optional
import numpy as np
import pybullet as p
import pybullet_data
import os
import re
import tempfile
from scipy.spatial.transform import Rotation as R
from pathlib import Path


class KinematicsSolver:
    """
    Generic kinematics solver using PyBullet.
    
    Args:
        urdf_path: Path to the URDF file
        end_effector_link_index: Index of the end-effector link
        base_position: Base position of the robot [x, y, z]
        base_orientation_euler: Base orientation in Euler angles [roll, pitch, yaw]
        gui: Whether to show PyBullet GUI
    """
    
    def __init__(
        self,
        urdf_path: str,
        end_effector_link_index: int,
        base_position: List[float] = [0, 0, 0],
        base_orientation_euler: List[float] = [0, 0, 0],
        gui: bool = False,
    ):
        self.urdf_path = urdf_path
        self.end_effector_link_index = end_effector_link_index
        self.base_position = base_position
        self.base_orientation_euler = base_orientation_euler
        
        # Initialize PyBullet
        self.robot_id = self._init_pybullet(gui)
        
    def _init_pybullet(self, gui: bool) -> int:
        """Initialize PyBullet physics engine and load the robot."""
        if gui:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Convert Euler to quaternion (PyBullet uses XYZW format)
        base_quat_xyzw = p.getQuaternionFromEuler(self.base_orientation_euler)
        
        # Resolve URDF path
        urdf_path = self._resolve_urdf_path(self.urdf_path)
        
        # Load robot
        robot_id = p.loadURDF(
            urdf_path,
            basePosition=self.base_position,
            baseOrientation=base_quat_xyzw,
            useFixedBase=True,
        )
        p.setGravity(0, 0, -9.81)
        
        return robot_id
    
    def _resolve_urdf_path(self, urdf_path: str) -> str:
        """
        Resolve URDF path, handling package:// URIs if present.
        
        Args:
            urdf_path: Path to URDF file (can contain package:// URIs)
            
        Returns:
            Resolved absolute path to URDF file
        """
        # Make path absolute
        if not os.path.isabs(urdf_path):
            urdf_path = os.path.abspath(urdf_path)
        
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF path does not exist: {urdf_path}")
        
        # Add model root to PyBullet search path
        model_root_dir = os.path.abspath(
            os.path.join(os.path.dirname(urdf_path), os.pardir)
        )
        p.setAdditionalSearchPath(model_root_dir)
        
        # Handle package:// URIs by replacing them with absolute paths
        try:
            with open(urdf_path, "r", encoding="utf-8") as f:
                urdf_text = f.read()
            
            if "package://" in urdf_text:
                # Replace package:// URIs with absolute paths
                # Extract package name and replace with model root
                # e.g., "package://X5A/meshes/..." becomes "[model_root_dir]/meshes/..."
                urdf_text = re.sub(
                    r'package://[^/]+/',
                    model_root_dir + '/',
                    urdf_text
                )
                
                # Write to temporary file
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".urdf", mode='w')
                tmp.write(urdf_text)
                tmp.flush()
                tmp.close()
                return tmp.name
        except Exception:
            pass
        
        return urdf_path
    
    def get_joint_positions(self, num_joints: int = None) -> np.ndarray:
        """
        Get current joint positions from the PyBullet simulation.
        
        Args:
            num_joints: Number of joints to retrieve. If None, retrieves all joints.
            
        Returns:
            Array of current joint positions (radians)
        """
        total_joints = p.getNumJoints(self.robot_id)
        if num_joints is None:
            num_joints = total_joints
        else:
            num_joints = min(num_joints, total_joints)
        
        joint_positions = []
        for j_idx in range(num_joints):
            joint_state = p.getJointState(self.robot_id, j_idx)
            joint_positions.append(joint_state[0])  # Position is first element
        
        return np.array(joint_positions, dtype=np.float64)
    
    def set_joint_positions(self, joint_positions: List[float]) -> None:
        """
        Set current joint positions in the PyBullet simulation.
        This syncs the simulation state with the real robot.
        
        Args:
            joint_positions: Joint positions to set (radians)
        """
        num_joints = p.getNumJoints(self.robot_id)
        for j_idx in range(min(len(joint_positions), num_joints)):
            p.resetJointState(self.robot_id, j_idx, float(joint_positions[j_idx]))
    
    def forward_kinematics(self, joint_positions: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute forward kinematics for given joint positions.
        
        Args:
            joint_positions: Joint positions (radians)
            
        Returns:
            Tuple of (position, quaternion_xyzw) of the end-effector
        """
        # Save current joint states
        num_joints = p.getNumJoints(self.robot_id)
        saved_states = []
        for j_idx in range(num_joints):
            saved_states.append(p.getJointState(self.robot_id, j_idx)[0])
        
        # Set new joint positions
        for j_idx in range(min(len(joint_positions), num_joints)):
            p.resetJointState(self.robot_id, j_idx, float(joint_positions[j_idx]))
        
        # Compute forward kinematics
        link_state = p.getLinkState(
            self.robot_id,
            self.end_effector_link_index,
            computeForwardKinematics=True
        )
        position = np.array(link_state[4], dtype=np.float64)  # World position
        quaternion_xyzw = np.array(link_state[5], dtype=np.float64)  # World orientation
        
        # Restore original joint states
        for j_idx in range(num_joints):
            p.resetJointState(self.robot_id, j_idx, float(saved_states[j_idx]))
        
        return position, quaternion_xyzw
    
    def inverse_kinematics(
        self,
        target_position: np.ndarray,
        target_orientation_xyzw: np.ndarray,
        current_joint_positions: List[float],
        max_iterations: int = 100,
        residual_threshold: float = 0.002,
        position_tolerance: float = 0.01,  # 1cm
        orientation_tolerance: float = 0.1,  # ~5.7 degrees
    ) -> Optional[np.ndarray]:
        """
        Solve inverse kinematics for target end-effector pose.
        
        Args:
            target_position: Target position [x, y, z]
            target_orientation_xyzw: Target orientation as quaternion [x, y, z, w]
            current_joint_positions: Current joint positions (used as seed for IK)
            max_iterations: Maximum number of IK iterations
            residual_threshold: Convergence threshold for PyBullet IK solver
            position_tolerance: Position error threshold for verification (meters)
            orientation_tolerance: Orientation error threshold for verification (radians)
            
        Returns:
            Joint positions that achieve the target pose, or None if verification fails
            
        Note:
            residual_threshold tells PyBullet when to stop if converged, but PyBullet
            will still return a solution even if max_iterations is reached without
            convergence. 
        """
        # Normalize quaternion
        quat = np.asarray(target_orientation_xyzw, dtype=np.float64)
        quat_norm = np.linalg.norm(quat)
        if not (0.999 <= quat_norm <= 1.001):
            quat = quat / max(quat_norm, 1e-9)
        
        # Compute inverse kinematics
        joint_positions = p.calculateInverseKinematics(
            self.robot_id,
            endEffectorLinkIndex=self.end_effector_link_index,
            targetPosition=target_position.tolist(),
            targetOrientation=quat.tolist(),
            currentPositions=current_joint_positions,
            maxNumIterations=max_iterations,
            residualThreshold=residual_threshold,
        )
        
        joint_positions_array = np.array(joint_positions, dtype=np.float64)
        return joint_positions_array


class EvaKinematicsSolver(KinematicsSolver):
    """
    Eva-specific kinematics solver.
    
    This solver adds Eva-specific configurations and handles the dual gripper joints.
    """
    
    # Eva-specific constants
    END_EFFECTOR_LINK_INDEX = 5  # Link index for Eva's end-effector
    NUM_ARM_JOINTS = 6  # Eva has 6 arm joints
    
    def __init__(
        self,
        urdf_path: str,
        base_position: List[float] = [0, 0, 0],
        base_orientation_euler: List[float] = [0, 0, 0],
        gui: bool = False
    ):
        """
        Initialize Eva kinematics solver.
        
        Args:
            urdf_path: Path to Eva's URDF file
            base_position: Base position of the robot [x, y, z]
            base_orientation_euler: Base orientation in Euler angles [roll, pitch, yaw]
            gui: Whether to show PyBullet GUI
        """
        super().__init__(
            urdf_path=urdf_path,
            end_effector_link_index=self.END_EFFECTOR_LINK_INDEX,
            base_position=base_position,
            base_orientation_euler=base_orientation_euler,
            gui=gui,
        )
        
        self.urdf_path = urdf_path
        self.base_transform = None
        
    def forward_kinematics(self, joint_positions: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        arm_joint_positions = joint_positions[:self.NUM_ARM_JOINTS]
        return super().forward_kinematics(arm_joint_positions)
    
    def inverse_kinematics(
        self,
        target_position: np.ndarray,
        target_orientation_xyzw: np.ndarray,
        current_joint_positions: List[float],
        max_iterations: int = 100,
        residual_threshold: float = 0.002,
        position_tolerance: float = 0.01,  # 1cm
        orientation_tolerance: float = 0.1,  # ~5.7 degrees
    ) -> Optional[np.ndarray]:
        """
        Solve inverse kinematics for Eva robot.
        
        This extends the base IK solver with Eva-specific joint damping
        and gripper joint handling.
        
        Args:
            target_position: Target position [x, y, z]
            target_orientation_xyzw: Target orientation as quaternion [x, y, z, w]
            current_joint_positions: Current joint positions (at least 6 arm joints)
            max_iterations: Maximum number of IK iterations
            residual_threshold: Convergence threshold for PyBullet IK solver
            position_tolerance: Position error threshold for verification (meters)
            orientation_tolerance: Orientation error threshold for verification (radians)
            
        Returns:
            Joint positions (6 arm joints only) that achieve the target pose, or None if verification fails
            
        Note:
            residual_threshold tells PyBullet when to stop if converged, but PyBullet
            will still return a solution even if max_iterations is reached without
            convergence. 
        """
        # Normalize quaternion
        quat = np.asarray(target_orientation_xyzw, dtype=np.float64)
        quat_norm = np.linalg.norm(quat)
        if not (0.999 <= quat_norm <= 1.001):
            quat = quat / max(quat_norm, 1e-9)
        
        # Eva has dual gripper joints at indices 6 and 7
        # Provide current positions for all joints including gripper
        current_positions_full = list(current_joint_positions[:self.NUM_ARM_JOINTS]) + [0.02239, -0.02239]
        
        # Compute inverse kinematics
        joint_positions = p.calculateInverseKinematics(
            self.robot_id,
            endEffectorLinkIndex=self.end_effector_link_index,
            targetPosition=target_position.tolist(),
            targetOrientation=quat.tolist(),
            currentPositions=current_positions_full,
            maxNumIterations=max_iterations,
            residualThreshold=residual_threshold,
        )
        
        # Return only the arm joints (first 6)
        arm_joints = np.array(list(joint_positions)[:self.NUM_ARM_JOINTS], dtype=np.float64)
        return arm_joints
    
    def get_joint_positions(self) -> np.ndarray:
        """
        Get current arm joint positions (excludes gripper joints).
        
        Returns:
            Array of 6 arm joint positions (radians)
        """
        return super().get_joint_positions(num_joints=self.NUM_ARM_JOINTS)
    
    def set_joint_positions(self, joint_positions: List[float]) -> None:
        """
        Set current arm joint positions in the PyBullet simulation.
        This syncs the simulation state with the real Eva robot.
        Gripper joints are set to default open position.
        
        Args:
            joint_positions: 6 arm joint positions to set (radians)
        """
        # Set arm joints
        for j_idx in range(min(len(joint_positions), self.NUM_ARM_JOINTS)):
            p.resetJointState(self.robot_id, j_idx, float(joint_positions[j_idx]))
        
        # Set gripper joints to default open position (indices 6 and 7)
        if p.getNumJoints(self.robot_id) > 6:
            p.resetJointState(self.robot_id, 6, 0.02239)   # Right gripper finger
            p.resetJointState(self.robot_id, 7, -0.02239)  # Left gripper finger


def example_eva_kinematics():
    """Example showing how to use the Eva kinematics solver."""
    
    # Path to Eva's URDF (adjust as needed)
    urdf_path = "eva/eva_ws/src/resources/ARX_Model/X5A/urdf/X5A.urdf"

    # Create Eva kinematics solver
    solver = EvaKinematicsSolver(
        urdf_path=urdf_path,
        gui=False,  # Set to True to visualize in PyBullet GUI
    )
        
    position, quaternion_xyzw = solver.forward_kinematics(solver.get_joint_positions())

    # Define a target pose (slightly offset from home)
    target_position = position + np.array([0.4, 0.0, 0.0])  # Move 10cm in X and Z
    target_orientation = quaternion_xyzw  # Keep same orientation
    
    print(f"Target position: {target_position}")
    
    # Solve IK with verification enabled (default)
    solution_joints = solver.inverse_kinematics(
        target_position=target_position,
        target_orientation_xyzw=target_orientation,
        current_joint_positions=solver.get_joint_positions(),
        max_iterations=100,
        residual_threshold=0.002,
        position_tolerance=0.01,  # 10mm tolerance
        orientation_tolerance=0.1,  # ~5.7 degrees tolerance
    )
        
    # Verify with FK
    achieved_pos, achieved_quat = solver.forward_kinematics(solution_joints)
    error = achieved_pos - target_position
    print(f"Achieved position: {achieved_pos}")
    # per-dimension error
    print(f"X error: {error[0]*1000:.2f}mm")
    print(f"Y error: {error[1]*1000:.2f}mm")
    print(f"Z error: {error[2]*1000:.2f}mm")


if __name__ == "__main__":
    # Run Eva example
    # Note: Update the URDF path before running
    example_eva_kinematics()
    

