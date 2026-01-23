"""
Robot interfaces for teleoperation.

Imports are deferred to class instantiation based on robot_type and camera config.
- ARXInterface: imports arx5 SDK when instantiated
- YAMInterface: imports i2rt library when instantiated  
- Camera recorders: import aria/realsense based on camera type in config
"""

import os
import sys
import time
import yaml
import numpy as np
from scipy.spatial.transform import Rotation as R
from abc import ABC, abstractmethod


class Robot_Interface(ABC):
    def __init__(self):
        self.cfg = {}
        try:
            self.cfg = self._get_config()
        except Exception as e:
            print(f"Failed to load configs.yaml: {e}")

    def _get_config(self):
        cfg_path = (
            "/home/robot/robot_ws/egomimic/robot/eva/eva_ws/src/config/configs.yaml"
        )
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        return cfg

    @abstractmethod
    def _create_controllers(self, cfg):
        pass

    @abstractmethod
    def set_joints(self, desired_position):
        pass

    @abstractmethod
    def set_pose(self, desired_position):
        pass

    @abstractmethod
    def get_obs(self):
        pass

    @staticmethod
    def solve_ik():
        pass

    @abstractmethod
    def get_joints(self):
        pass

    @abstractmethod
    def get_pose(self):
        pass

    @abstractmethod
    def set_home(self):
        pass


class ARXInterface(Robot_Interface):
    def __init__(self, arms):
        # Import ARX dependencies when this class is used
        import arx5.arx5_interface as arx5
        from arx5.arx5_interface import Arx5JointController, JointState as ArxJointState, Gain
        from egomimic.robot.eva.eva_kinematics import EvaMinkKinematicsSolver
        
        self._arx5 = arx5
        self._Arx5JointController = Arx5JointController
        self._ArxJointState = ArxJointState
        self._Gain = Gain

        super().__init__()

        # ARX-specific robot config setup
        model = self.cfg.get("model", "X5")
        self.robot_urdf = self.cfg.get("urdf", None)

        self.robot_config = arx5.RobotConfigFactory.get_instance().get_config(model)
        if self.robot_urdf:
            self.robot_config.urdf_path = self.robot_urdf
        self.robot_config.base_link_name = "base_link"
        self.robot_config.eef_link_name = "link6"

        self.controller_config = arx5.ControllerConfigFactory.get_instance().get_config(
            "joint_controller", self.robot_config.joint_dof
        )

        self.arms = arms
        self.controller = dict()
        self._create_controllers(self.cfg)
        self._create_cam_recorders(self.cfg["cameras"])
        self.save_resolution = self.cfg["save_resolution"]
        self.kinematics_solver = EvaMinkKinematicsSolver(
            model_path="/home/robot/robot_ws/egomimic/resources/model_x5.xml"
        )

    def _create_controllers(self, cfg):
        interfaces_cfg = cfg.get("interfaces", {})
        for arm in self.arms:
            if arm == "right":
                default_iface = "can2"
                selected_interface = interfaces_cfg.get("right", default_iface)
            elif arm == "left":
                default_iface = "can1"
                selected_interface = interfaces_cfg.get("left", default_iface)

            self.controller[arm] = self._Arx5JointController(
                self.robot_config, self.controller_config, selected_interface
            )
            self.controller[arm].reset_to_home()

            gain = self.controller[arm].get_gain()

            kp = (
                np.array(
                    [6.225, 17.225, 18.225, 14.225, 8.225, 6.225], dtype=np.float64
                )
                * 0.8
            )
            kd = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0], dtype=np.float64) * 0.6 
            # zeros = np.zeros(6)
            # kp = zeros
            # kd = zeros
            gain.kp()[:] = kp
            gain.kd()[:] = kd
            gain.gripper_kp = 1.0
            gain.gripper_kd = 0.1

            self.ts_offset = 0.2

            self.controller[arm].set_gain(gain)

            self.gripper_offset = 0.000

            # self.engaged = True

            # self.gripper_width = self.cfg.get("gripper_width", None)

            # if self.gripper_width is None:
            #     raise RuntimeError("Gripper value not initialized in config.yaml")

    def _create_cam_recorders(self, cameras_cfg):
        self.recorders = dict()
        self.camera_config = cameras_cfg  # Store camera config for later use
        for name, cam_cfg in cameras_cfg.items():
            if not cam_cfg["enabled"]:
                continue
            cam_type = cam_cfg["type"]
            if cam_type == "aria":
                # Import Aria only when aria camera is configured
                from stream_aria import AriaRecorder
                self.recorders[name] = AriaRecorder(
                    profile_name="profile15", use_security=True
                )
                self.recorders[name].start()
            elif cam_type == "d405":
                # Import RealSense only when d405 camera is configured
                from stream_d405 import RealSenseRecorder
                self.recorders[name] = RealSenseRecorder(
                    str(cam_cfg["serial_number"]),
                    width=cam_cfg["width"],
                    height=cam_cfg["height"],
                )
            else:
                raise ValueError(f"Unknown camera type '{cam_type}' in config")

    def set_joints(self, desired_position, arm):
        """

        Args:
            desired_position (np.array): 6 joints + gripper values (0 to 1)
        """
        if desired_position.shape != (7,):
            raise ValueError(
                "For Eva, desired position must be of shape (7,) for single arm"
            )

        gripper_cmd = desired_position[6]
        desired_position = desired_position[:6]

        velocity = np.zeros_like(desired_position) + 0.1
        torque = np.zeros_like(desired_position) + 0.1

        # you need to set the timestamp this way since timestamp tells controller interpolator what the target it should reach at absolute timestamps
        cur_joint_state = self.controller[arm].get_joint_state()
        current_ts = getattr(cur_joint_state, "timestamp", 0.0)
        self.timestamp = current_ts + self.ts_offset

        requested = self._ArxJointState(
            desired_position.astype(np.float32),
            velocity.astype(np.float32),
            torque.astype(np.float32),
            float(self.timestamp),
        )

        # print(f"gripper val: {gripper_cmd}")
        requested.gripper_pos = float(gripper_cmd)
        requested.gripper_vel = 0.1
        requested.gripper_torque = 0.2

        self.controller[arm].set_joint_cmd(requested)

    # x,y,z,y,p,r
    def set_pose(self, pose, arm):
        if pose.shape != (7,):
            raise ValueError(
                f"For Eva, target position must be of shape (7,), current shape: {pose.shape}"
            )
        arm_joints = self.solve_ik(pose[:6], arm)
        joints = np.concatenate([arm_joints, [pose[6]]])
        self.set_joints(joints, arm)
        return joints

    def get_obs(self):
        obs = {}
        joint_positions = np.zeros(14)
        ee_poses = np.zeros(14)
        for arm in self.arms:
            arm_offset = 0
            if arm == "right":
                arm_offset = 7
            joint_positions[arm_offset : arm_offset + 7] = self.get_joints(arm)
            xyz, rot = self.get_pose(arm, se3=False)
            ee_poses[arm_offset : arm_offset + 7] = np.concatenate([xyz, rot.as_euler("ZYX", degrees=False), [joint_positions[arm_offset + 6]]])
        obs["joint_positions"] = joint_positions
        obs["ee_poses"] = ee_poses

        # camera logic
        for name, recorder in self.recorders.items():
            obs[name] = recorder.get_image()
        return obs

    # removed static since can't figure out how to create ik when robot_urdf is not static
    # take in ypr
    def solve_ik(self, ee_pose, arm):
        if ee_pose.shape != (6,):
            raise ValueError(
                "For Eva, target position must be of shape (6,) for single arm"
            )
        pos_xyz = ee_pose[:3]
        # ypr_euler = ee_pose[3:6]
        # ypr_euler[[0,2]] = ypr_euler[[2,0]]
        rot_mat = R.from_euler(
            "ZYX", ee_pose[3:6], degrees=False
        ).as_matrix()  # scipy output xyzw
        # rot_mat2 = R.from_euler(
        #     "XYZ", ee_pose[3:6], degrees=False
        # )
        # breakpoint()
        arm_joints = self.kinematics_solver.ik(
            pos_xyz, rot_mat, cur_jnts=self.get_joints(arm)[:6]
        )
        return arm_joints

    def get_joints(self, arm):
        joints = self.controller[arm].get_joint_state()
        arm_joints = joints.pos()
        gripper = getattr(joints, "gripper_pos", 0.0)
        joints = np.array(
            [
                arm_joints[0],
                arm_joints[1],
                arm_joints[2],
                arm_joints[3],
                arm_joints[4],
                arm_joints[5],
                gripper,
            ]
        )
        return joints

    def get_pose(self, arm, se3=False):
        """

        Returns:
           xyz: np.array, quat: np.array (xyzw)
        """
        joints = self.get_joints(arm)
        pos, rot = self.kinematics_solver.fk(joints[:6])
        if se3:
            # Return 4x4 SE(3) transformation matrix (world to end-effector)
            T = np.eye(4)
            T[:3, :3] = rot.as_matrix()
            T[:3, 3] = pos
            return T

        return pos, rot
    
    def get_pose_6d(self, arm):
        pos, rot = self.get_pose(arm, se3=False)
        return np.concatenate([pos, rot.as_euler("ZYX", degrees=False)])

    def set_home(self):
        for arm in self.arms:
            self.controller[arm].reset_to_home()


class YAMInterface:
    """
    Robot interface for I2RT YAM robot using the i2rt library.
    
    This interface wraps MotorChainRobot from i2rt and provides the same API
    as ARXInterface for use with VR teleoperation and demo collection.
    
    Joint Limits (from i2rt/robots/get_robot.py line 47):
        The i2rt library adds ±0.15 rad buffer internally for safety.
        Actual mechanical limits are:
        - Joint 1 (base):     [-2.617, 3.13] rad
        - Joint 2 (shoulder): [0, 3.65] rad       ← NOTE: min is 0!
        - Joint 3 (elbow):    [0, 3.13] rad       ← NOTE: min is 0!
        - Joint 4 (wrist 1):  [-1.57, 1.57] rad
        - Joint 5 (wrist 2):  [-1.57, 1.57] rad
        - Joint 6 (wrist 3):  [-2.09, 2.09] rad
        - Gripper: normalized 0-1 (0=closed, 1=open), auto-calibrated
    
    Safety Features (from i2rt library):
        - Joint limits enforced with RuntimeError on violation
        - Gravity compensation enabled by default (factor=1.3)
        - Gripper force limiting (50N default)
        - 400ms motor timeout (can be disabled via i2rt tools)
    
    Additional Safety (added in this interface):
        - Software joint limit checking before each command
        - Dry-run mode for testing without actuation
    """
    
    # Default CAN interfaces for left/right arms
    DEFAULT_INTERFACES = {
        "left": "can_left",
        "right": "can_right",
    }
    
    # Joint limits - from i2rt/robots/get_robot.py lines 47-49
    # Including the ±0.15 rad buffer that i2rt applies for safety
    JOINT_LIMITS = np.array([
        [-2.767, 3.28],   # Joint 1 (base rotation): base [-2.617, 3.13] + buffer
        [-0.15, 3.80],    # Joint 2 (shoulder): base [0, 3.65] + buffer
        [-0.15, 3.28],    # Joint 3 (elbow): base [0, 3.13] + buffer
        [-1.72, 1.72],    # Joint 4 (wrist 1): base [-1.57, 1.57] + buffer
        [-1.72, 1.72],    # Joint 5 (wrist 2): base [-1.57, 1.57] + buffer
        [-2.24, 2.24],    # Joint 6 (wrist 3): base [-2.09, 2.09] + buffer
    ])
    
    # Home position for YAM robot (in radians)
    HOME_POSITION = np.array([0.0, 0.05, 0.05, 0.0, 0.0, 0.0, 1.0])
    # HOME_POSITION = np.array([4.864e-2, 1.467, 5.205e-1, 1.25e-1, -2.88e-2, -6.847e-2, 1.0])
    HOME_POSITION_ABOVE = np.array([
        1.051e-1,
        8.875e-1,
        5.285e-1,
        -2.840e-1,
        5.207e-2,
        -7.153e-2
        , 0.80])


    def __init__(
        self,
        arms: list,
        gripper_type = None,
        interfaces: dict = None,
        cameras_cfg: dict = None,
        zero_gravity_mode: bool = False,
        dry_run: bool = False,
    ):
        """
        Initialize YAM robot interface.
        
        Args:
            arms: List of arm names to control, e.g. ["left"], ["right"], or ["left", "right"]
            gripper_type: GripperType enum or string name (default: "linear_4310")
            interfaces: Dict mapping arm names to CAN interfaces, e.g. {"left": "can0", "right": "can1"}
            cameras_cfg: Camera configuration dict (same format as ARXInterface)
            zero_gravity_mode: If True, robot starts in gravity compensation mode (movable by hand).
                              If False, robot holds current position on startup.
            dry_run: If True, don't connect to real robot - just log commands and simulate responses.
        """
        # Import i2rt dependencies when this class is used
        sys.path.insert(0, os.path.expanduser("~/i2rt"))
        from i2rt.robots.get_robot import get_yam_robot
        from i2rt.robots.kinematics import Kinematics as I2RTKinematics
        from i2rt.robots.utils import GripperType
        
        self._get_yam_robot = get_yam_robot
        self._I2RTKinematics = I2RTKinematics
        self._GripperType = GripperType

        self.arms = arms
        self.interfaces = interfaces if interfaces is not None else self.DEFAULT_INTERFACES
        self.zero_gravity_mode = zero_gravity_mode
        self.dry_run = dry_run
        
        # Handle gripper_type
        if gripper_type is None:
            self.gripper_type = GripperType.LINEAR_4310
        elif isinstance(gripper_type, str):
            self.gripper_type = GripperType.from_string_name(gripper_type)
        else:
            self.gripper_type = gripper_type
        
        # Track simulated joint positions for dry run mode
        self._simulated_joints = {}
        for arm in arms:
            self._simulated_joints[arm] = self.HOME_POSITION.copy()
        
        # Track last command time for velocity limiting (per arm)
        self._last_cmd_time = {}
        
        # Initialize robot controllers (one per arm)
        self.robot = {}
        self.kinematics = {}
        
        # Get the XML path for kinematics (same for all arms)
        model_path = self.gripper_type.get_xml_path()
        
        for arm in self.arms:
            channel = self.interfaces.get(arm, self.DEFAULT_INTERFACES.get(arm, "can0"))
            
            if self.dry_run:
                print(f"[YAMInterface] DRY RUN: Would initialize {arm} arm on {channel}")
                print(f"[YAMInterface] DRY RUN: Gripper type: {self.gripper_type}")
                print(f"[YAMInterface] DRY RUN: Model path: {model_path}")
                self.robot[arm] = None
            else:
                print(f"[YAMInterface] Initializing {arm} arm on {channel} with gripper type {self.gripper_type}")
                self.robot[arm] = get_yam_robot(
                    channel=channel,
                    gripper_type=self.gripper_type,
                    zero_gravity_mode=zero_gravity_mode,
                )
            
            print(f"[YAMInterface] Using model: {model_path}")
            
            self.kinematics[arm] = I2RTKinematics(
                xml_path=model_path,
                site_name="grasp_site",
            )
        
        # Initialize camera recorders and load save_resolution from config
        self.recorders = {}
        if cameras_cfg:
            self._create_cam_recorders(cameras_cfg)
            # Still need to load save_resolution from config file
            self._load_save_resolution()
        else:
            self._load_default_camera_config()
        
        print(f"[YAMInterface] Initialized with arms: {self.arms}")
        if self.dry_run:
            print("[YAMInterface] *** DRY RUN MODE ENABLED - NO ROBOT ACTUATION ***")
    
    def print_config(self):
        """Print current configuration summary."""
        print("\n" + "="*60)
        print("YAM ROBOT CONFIGURATION")
        print("="*60)
        print(f"Arms:              {self.arms}")
        print(f"Gripper type:      {self.gripper_type}")
        print(f"CAN interfaces:    {self.interfaces}")
        print(f"Zero gravity mode: {self.zero_gravity_mode}")
        print(f"Dry run mode:      {self.dry_run}")
        print(f"Model path:        {self.gripper_type.get_xml_path()}")
        print(f"Cameras:           {list(self.recorders.keys())}")
        print("\nJoint Limits (rad):")
        joint_names = ["J1 (base)", "J2 (shoulder)", "J3 (elbow)", "J4 (wrist1)", "J5 (wrist2)", "J6 (wrist3)"]
        for i, (name, (lo, hi)) in enumerate(zip(joint_names, self.JOINT_LIMITS)):
            print(f"  {name}: [{lo:+.3f}, {hi:+.3f}]")
        print(f"\nHome position: {np.array2string(self.HOME_POSITION, precision=3)}")
        print("="*60 + "\n")
    
    def _load_save_resolution(self):
        """Load save_resolution from default YAML config file."""
        config_paths = [
            "/home/robot/robot_ws/egomimic/robot/eva/eva_ws/src/config/configs_yam.yaml",
            os.path.expanduser("~/robot_ws/egomimic/robot/eva/eva_ws/src/config/configs_yam.yaml"),
        ]
        
        for cfg_path in config_paths:
            if os.path.exists(cfg_path):
                with open(cfg_path, "r") as f:
                    cfg = yaml.safe_load(f) or {}
                self.save_resolution = cfg["save_resolution"]
                return
        
        raise FileNotFoundError(f"[YAMInterface] No config found in paths: {config_paths}")
    
    def _load_default_camera_config(self):
        """Load camera configuration from default YAML config file."""
        config_paths = [
            "/home/robot/robot_ws/egomimic/robot/eva/eva_ws/src/config/configs_yam.yaml",
            os.path.expanduser("~/robot_ws/egomimic/robot/eva/eva_ws/src/config/configs_yam.yaml"),
        ]
        
        for cfg_path in config_paths:
            if os.path.exists(cfg_path):
                with open(cfg_path, "r") as f:
                    cfg = yaml.safe_load(f) or {}
                if "cameras" not in cfg:
                    raise ValueError(f"[YAMInterface] No 'cameras' section in config: {cfg_path}")
                self._create_cam_recorders(cfg["cameras"])
                self.save_resolution = cfg["save_resolution"]
                print(f"[YAMInterface] Loaded camera config from {cfg_path}")
                return
        
        raise FileNotFoundError(f"[YAMInterface] No camera config found in paths: {config_paths}")
    
    def _create_cam_recorders(self, cameras_cfg: dict):
        """Create camera recorders based on config - imports only what's needed."""
        self.camera_config = cameras_cfg  # Store camera config for later use
        for name, cam_cfg in cameras_cfg.items():
            if not cam_cfg.get("enabled", False):
                continue
            cam_type = cam_cfg.get("type", "")
            
            if cam_type == "aria":
                # Import Aria only when aria camera is configured
                from stream_aria import AriaRecorder
                self.recorders[name] = AriaRecorder(
                    profile_name="profile15", use_security=True
                )
                self.recorders[name].start()
                print(f"[YAMInterface] Started Aria camera: {name}")
                    
            elif cam_type == "d405":
                # Import RealSense only when d405 camera is configured
                from stream_d405 import RealSenseRecorder
                serial = str(cam_cfg["serial_number"])
                width = cam_cfg["width"]
                height = cam_cfg["height"]
                self.recorders[name] = RealSenseRecorder(serial, width=width, height=height)
                print(f"[YAMInterface] Started RealSense D405 camera: {name} (serial: {serial}, {width}x{height})")
            else:
                raise ValueError(f"Unknown camera type '{cam_type}' for {name}")
    
    def _check_joint_limits(self, joints: np.ndarray, arm: str, tolerance: float = 0.01) -> bool:
        """Check if joint positions are within safe limits.
        
        Args:
            joints: Joint positions array
            arm: Arm name for logging
            tolerance: Small tolerance for floating point comparison (default 0.01 rad ≈ 0.6°)
            
        Returns:
            True if within limits (with tolerance), False otherwise
        """
        arm_joints = joints[:6]
        for i, (j, (lo, hi)) in enumerate(zip(arm_joints, self.JOINT_LIMITS)):
            if j < lo - tolerance or j > hi + tolerance:
                print(f"[YAMInterface] WARNING: {arm} joint {i} = {j:.3f} rad is outside limits [{lo:.3f}, {hi:.3f}]")
                return False
        return True
    
    def _clamp_to_limits(self, joints: np.ndarray, margin: float = 0.005) -> np.ndarray:
        """Clamp joint positions to be within safe limits with a small margin.
        
        Args:
            joints: (7,) array of joint positions + gripper
            margin: Small margin inside limits to avoid edge cases (default 0.005 rad)
            
        Returns:
            Clamped joint positions
        """
        clamped = joints.copy()
        for i, (lo, hi) in enumerate(self.JOINT_LIMITS):
            clamped[i] = np.clip(joints[i], lo + margin, hi - margin)
        return clamped
    
    def set_joints(self, desired_position: np.ndarray, arm: str, velocity_limit: float = None):
        """Command joint positions for a specific arm.
        
        Args:
            desired_position: (7,) array of 6 joint positions + gripper value
            arm: Arm name ("left" or "right")
            velocity_limit: Maximum joint velocity in rad/s. If None, no velocity limit is applied.
                           The limit is applied to joints only (not gripper).
        """
        if desired_position.shape != (7,):
            raise ValueError(
                f"For YAM, desired position must be of shape (7,), got {desired_position.shape}"
            )
        
        if not self._check_joint_limits(desired_position, arm):
            print(f"[YAMInterface] WARNING: Skipping command due to joint limit violation")
            return
        
        # Apply velocity limiting if specified
        cmd_position = desired_position.copy()
        if velocity_limit is not None:
            current_time = time.time()
            
            # Compute dt from last command (default to 1/30s if first command)
            if arm in self._last_cmd_time:
                dt = current_time - self._last_cmd_time[arm]
                # Clamp dt to reasonable range to handle pauses/delays
                dt = np.clip(dt, 0.001, 0.1)
            else:
                dt = 1.0 / 30.0  # Default assumption: 30Hz control
            
            self._last_cmd_time[arm] = current_time
            
            # Get current joint positions
            current_joints = self.get_joints(arm)
            
            # Compute max change per step for joints (not gripper)
            max_delta = velocity_limit * dt
            
            # Clip the joint delta (first 6 values, not gripper)
            delta = cmd_position[:6] - current_joints[:6]
            clipped_delta = np.clip(delta, -max_delta, max_delta)
            cmd_position[:6] = current_joints[:6] + clipped_delta
        
        if self.dry_run:
            print(f"[YAMInterface] DRY RUN: set_joints({arm}) = {np.array2string(cmd_position, precision=3)}, velocity_limit={velocity_limit}")
            self._simulated_joints[arm] = cmd_position.copy()
        else:
            self.robot[arm].command_joint_pos(cmd_position)
    
    def get_joints(self, arm: str) -> np.ndarray:
        """Get current joint positions for a specific arm."""
        if self.dry_run:
            return self._simulated_joints[arm].copy()
        return self.robot[arm].get_joint_pos()
    
    def set_pose(self, pose: np.ndarray, arm: str, velocity_limit: float = None) -> np.ndarray:
        """Command end-effector pose using inverse kinematics.
        
        Args:
            pose: (7,) array of [x, y, z, yaw, pitch, roll, gripper]
            arm: Arm name ("left" or "right")
            velocity_limit: Maximum joint velocity in rad/s. If None, no velocity limit is applied.
            
        Returns:
            joints: (7,) array of joint positions that were commanded, or None if skipped
        """
        if pose.shape != (7,):
            raise ValueError(
                f"For YAM, target pose must be of shape (7,), got {pose.shape}"
            )
        
        arm_joints = self.solve_ik(pose[:6], arm)
        if arm_joints is None:
            print(f"[YAMInterface] ERROR: IK completely failed for arm {arm}, skipping command")
            return None
        
        joints = np.concatenate([arm_joints, [pose[6]]])
        
        # set_joints will check limits and skip if still violated after clamping
        self.set_joints(joints, arm, velocity_limit=velocity_limit)
        return joints
    
    def solve_ik(
        self, 
        ee_pose: np.ndarray, 
        arm: str,
        damping: float = 1e-4,
        max_iters: int = 500,
        pos_threshold: float = 1e-3,
        ori_threshold: float = 1e-3,
        verbose: bool = False,
    ) -> np.ndarray:
        """Solve inverse kinematics for target end-effector pose.
        
        Args:
            ee_pose: (6,) array of [x, y, z, yaw, pitch, roll]
            arm: Arm name ("left" or "right")
            damping: Levenberg-Marquardt damping for stability (higher = more stable, slower)
            max_iters: Maximum IK iterations
            pos_threshold: Position convergence threshold (meters)
            ori_threshold: Orientation convergence threshold (radians)
            verbose: If True, print debug info on IK failure
            
        Returns:
            Solved joint positions (6,) or None if IK fails completely
        """
        if ee_pose.shape != (6,):
            raise ValueError(
                f"For YAM, target pose must be of shape (6,), got {ee_pose.shape}"
            )
        
        pos_xyz = ee_pose[:3]
        ypr = ee_pose[3:6]
        
        rot_mat = R.from_euler("ZYX", ypr, degrees=False).as_matrix()
        target_T = np.eye(4)
        target_T[:3, :3] = rot_mat
        target_T[:3, 3] = pos_xyz
        
        cur_jnts = self.get_joints(arm)[:6]
        
        # Use improved IK parameters for better convergence
        success, solved_joints = self.kinematics[arm].ik(
            target_pose=target_T,
            site_name="grasp_site",
            init_q=cur_jnts,
            damping=damping,
            max_iters=max_iters,
            pos_threshold=pos_threshold,
            ori_threshold=ori_threshold,
            verbose=verbose,
        )
        
        # Compute achieved pose to see how far off we are
        achieved_T = self.kinematics[arm].fk(solved_joints)
        pos_err = np.linalg.norm(achieved_T[:3, 3] - pos_xyz)
        
        # Print diagnostic info for large errors (> 5mm)
        if pos_err > 0.005:
            current_pose = self.kinematics[arm].fk(cur_jnts)
            print(f"[YAMInterface] IK {'FAILED' if not success else 'large error'} for arm {arm}:")
            print(f"  Target XYZ:   [{pos_xyz[0]:.4f}, {pos_xyz[1]:.4f}, {pos_xyz[2]:.4f}] m")
            print(f"  Target YPR:   [{ypr[0]:.4f}, {ypr[1]:.4f}, {ypr[2]:.4f}] rad")
            print(f"  Current XYZ:  [{current_pose[0,3]:.4f}, {current_pose[1,3]:.4f}, {current_pose[2,3]:.4f}] m")
            print(f"  Achieved XYZ: [{achieved_T[0,3]:.4f}, {achieved_T[1,3]:.4f}, {achieved_T[2,3]:.4f}] m")
            print(f"  Position error: {pos_err*1000:.1f}mm")
        
        # Clamp solved joints to be within limits (helps when IK solution is at boundary)
        solved_joints = self._clamp_to_limits(
            np.concatenate([solved_joints, [0.0]])  # Add dummy gripper for clamping
        )[:6]
        
        return solved_joints
    
    def get_pose(self, arm: str, se3: bool = False):
        """Get current end-effector pose via forward kinematics."""
        joints = self.get_joints(arm)[:6]
        T = self.kinematics[arm].fk(joints)
        
        if se3:
            return T
        
        pos = T[:3, 3]
        rot = R.from_matrix(T[:3, :3])
        return pos, rot
    
    def get_pose_6d(self, arm: str) -> np.ndarray:
        """Get current end-effector pose as 6D vector."""
        pos, rot = self.get_pose(arm, se3=False)
        ypr = rot.as_euler("ZYX", degrees=False)
        return np.concatenate([pos, ypr])
    
    def get_obs(self) -> dict:
        """Get full observation dictionary including joint states and camera images."""
        obs = {}
        joint_positions = np.zeros(14)
        ee_poses = np.zeros(14)
        
        for arm in self.arms:
            arm_offset = 0 if arm == "left" else 7
            joint_positions[arm_offset:arm_offset + 7] = self.get_joints(arm)
            xyz, rot = self.get_pose(arm, se3=False)
            ypr = rot.as_euler("ZYX", degrees=False)
            gripper = joint_positions[arm_offset + 6]
            ee_poses[arm_offset:arm_offset + 7] = np.concatenate([xyz, ypr, [gripper]])
        
        obs["joint_positions"] = joint_positions
        obs["ee_poses"] = ee_poses
        
        for name, recorder in self.recorders.items():
            obs[name] = recorder.get_image()
        
        return obs
    
    def set_home(self):
        """Move all arms to home position."""
        for arm in self.arms:
            # home_position = self.HOME_POSITION if arm == 'right' else self.HOME_POSITION_ABOVE
            home_position = self.HOME_POSITION
            if self.dry_run:
                print(f"[YAMInterface] DRY RUN: Would move {arm} arm to home position: {home_position}")
                self._simulated_joints[arm] = home_position.copy()
            else:
                print(f"[YAMInterface] Moving {arm} arm to home position...")
                self.robot[arm].move_joints(home_position, time_interval_s=2.0)
    
    def close(self):
        """Safely close robot connections and camera streams."""
        for arm in self.arms:
            if self.dry_run:
                print(f"[YAMInterface] DRY RUN: Would close {arm} arm controller")
            else:
                try:
                    self.robot[arm].close()
                    print(f"[YAMInterface] Closed {arm} arm controller")
                except Exception as e:
                    print(f"[YAMInterface] Error closing {arm} arm: {e}")
        
        for name, recorder in self.recorders.items():
            try:
                recorder.stop()
                print(f"[YAMInterface] Stopped camera: {name}")
            except Exception as e:
                print(f"[YAMInterface] Error stopping camera {name}: {e}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == "__main__":
    # Run Eva example
    # Note: Update the URDF path before running
    ri = ARXInterface(arms=["left"])
    joints = ri.get_joints("left")
    breakpoint()
