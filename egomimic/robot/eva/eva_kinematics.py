from egomimic.robot.kinematics import KinematicsSolver

class EvaKinematicsSolver(KinematicsSolver):
    """
    Eva-specific kinematics solver.

    This solver adds Eva-specific configurations and handles the dual gripper joints.
    """

    def __init__(
        self,
        urdf_path: str,
    ):
        """
        Initialize Eva kinematics solver.

        Args:
            urdf_path: Path to Eva's URDF file
        """
        super().__init__(
            urdf_path=urdf_path,
            base_link_name="base_link",
            eef_link_name="link6",
            num_joints=6
        )

        self.urdf_path = urdf_path
        self.base_transform = None