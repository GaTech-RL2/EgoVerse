from egomimic.robot.kinematics import MinkKinematicsSolver

class EvaMinkKinematicsSolver(MinkKinematicsSolver):
    """
    Eva-specific kinematics solver using mink.
    
    This solver provides optimization-based IK for the Eva (ARX X5) robot arm
    using the mink library built on MuJoCo.
    """
    
    # Joint names for ARX X5 arm
    JOINT_NAMES = [
        "joint1",
        "joint2", 
        "joint3",
        "joint4",
        "joint5",
        "joint6",
    ]
    
    # Conservative velocity limits (rad/s)
    DEFAULT_VELOCITY_LIMITS = {
        "joint1": 1.0,
        "joint2": 1.0,
        "joint3": 1.0,
        "joint4": 1.0,
        "joint5": 1.0,
        "joint6": 1.0,
    }
    
    def __init__(
        self,
        urdf_path: str,
        eef_link_name: str = "gripper",
        eef_frame_type: str = "site",
        velocity_limits: dict = None,
        solver: str = "daqp",
        max_iterations: int = 20,
        position_tolerance: float = 1e-3,
        orientation_tolerance: float = 1e-3,
    ):
        """
        Initialize Eva mink kinematics solver.
        
        Args:
            urdf_path: Path to Eva's URDF/XML file (should be MuJoCo XML format)
            eef_link_name: Name of end-effector frame (default: "gripper")
            eef_frame_type: Type of frame - "site" or "body" (default: "site")
            velocity_limits: Optional dict of joint velocity limits
            solver: QP solver to use (default: "daqp")
            max_iterations: Maximum IK iterations (default: 100)
            position_tolerance: Position convergence tolerance in meters (default: 1e-3)
            orientation_tolerance: Orientation convergence tolerance in radians (default: 1e-3)
        """
        if velocity_limits is None:
            velocity_limits = self.DEFAULT_VELOCITY_LIMITS.copy()
        
        super().__init__(
            urdf_path=urdf_path,
            base_link_name="base_link",
            eef_link_name=eef_link_name,
            num_joints=6,
            joint_names=self.JOINT_NAMES,
            eef_frame_type=eef_frame_type,
            velocity_limits=velocity_limits,
            solver=solver,
            max_iterations=max_iterations,
            position_tolerance=position_tolerance,
            orientation_tolerance=orientation_tolerance,
        )
        
        self.urdf_path = urdf_path
        self.base_transform = None
    
    def set_base_transform(self, transform):
        """
        Set base transform for the robot (useful for mobile base).
        
        Args:
            transform: 4x4 transformation matrix
        """
        self.base_transform = transform
    
    def ik_with_retries(self, pos_xyz, rot_mat, cur_jnts, num_retries=3, dt=0.01):
        """
        Solve IK with multiple retries and different random seeds.
        
        Args:
            pos_xyz: Target position (3,)
            rot_mat: Target rotation matrix (3, 3)
            cur_jnts: Current joint values (6,)
            num_retries: Number of retries with random perturbations
            dt: Time step for integration
            
        Returns:
            solved_jnts: Solution joint values or None if all retries fail
        """
        import numpy as np
        
        # First try with current configuration
        result = self.ik(pos_xyz, rot_mat, cur_jnts, dt=dt)
        
        # Verify the solution
        if result is not None:
            # Check if solution is valid by computing FK
            fk_pos, fk_rot = self.fk(result)
            pos_error = np.linalg.norm(fk_pos - pos_xyz)
            rot_error = np.linalg.norm(fk_rot.as_matrix() - rot_mat)
            
            if pos_error < self.position_tolerance * 10 and rot_error < self.orientation_tolerance * 10:
                return result
        
        # Try with random perturbations
        for i in range(num_retries):
            # Add small random perturbation to seed
            perturbed_jnts = cur_jnts + np.random.randn(self.num_joints) * 0.1
            result = self.ik(pos_xyz, rot_mat, perturbed_jnts, dt=dt)
            
            if result is not None:
                fk_pos, fk_rot = self.fk(result)
                pos_error = np.linalg.norm(fk_pos - pos_xyz)
                rot_error = np.linalg.norm(fk_rot.as_matrix() - rot_mat)
                
                if pos_error < self.position_tolerance * 10 and rot_error < self.orientation_tolerance * 10:
                    return result
        
        # All retries failed
        return None