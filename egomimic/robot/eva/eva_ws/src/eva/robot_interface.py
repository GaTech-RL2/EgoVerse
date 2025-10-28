import arx5.arx5_interface as arx5
from arx5.arx5_interface import Arx5JointController, JointState as ArxJointState, Gain
from ament_index_python.packages import get_package_share_directory
import os
import yaml
import numpy as np
import pytorch_kinematics as pk
from scipy.spatial.transform import Rotation as R
from abc import ABC, abstractmethod
from stream_aria import AriaRecorder
from stream_d405 import RealSenseRecorder
from egomimic.robot.kinematics import EvaKinematicsSolver

# from egomimic.robot.eva.kinematics import EvaKinematicsSolver


class Robot_Interface(ABC):

    def __init__(self):
        self.cfg = {}
        try:
            self.cfg = self.__get_config(self.cfg)
        except Exception as e:
            print(f"Failed to load configs.yaml: {e}")

        # self.arm = arm

        model = self.cfg.get("model", "X5")
        self.robot_urdf = self.cfg.get("urdf", None)

        self.robot_config = arx5.RobotConfigFactory.get_instance().get_config(model)
        if self.robot_urdf:
            self.robot_config.urdf_path = self.robot_urdf
            # Match X5A URDF link names
        self.robot_config.base_link_name = "base_link"
        self.robot_config.eef_link_name = "link6"

        self.controller_config = arx5.ControllerConfigFactory.get_instance().get_config(
            "joint_controller", self.robot_config.joint_dof
        )

    def __get_config(self, cfg):
        share = get_package_share_directory("eva")
        cfg_path = os.path.join(share, "config", "configs.yaml")
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


class SingleARXInterface(Robot_Interface):
    def __init__(self, arm):
        super().__init__()

        self.arm = arm
        self._create_controllers(self.cfg)
        self.__create_cam_recorders(self.cfg)
        self.kinematics_solver = EvaKinematicsSolver(self.robot_urdf)

    def _create_controllers(self, cfg):
        interfaces_cfg = cfg.get("interfaces", {})
        if self.arm == "right":
            default_iface = "can2"
            selected_interface = interfaces_cfg.get("right", default_iface)
        elif self.arm == "left":
            default_iface = "can1"
            selected_interface = interfaces_cfg.get("left", default_iface)

        self.controller = Arx5JointController(self.robot_config, self.controller_config, selected_interface)

        gain = self.controller.get_gain()

        kp = np.array([6.225, 17.225, 18.225, 12.225, 8.225, 6.225], dtype=np.float64)
        kd = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0], dtype=np.float64)
        gain.kp()[:] = kp
        gain.kd()[:] = kd
        gain.gripper_kp = 1.0
        gain.gripper_kd = 0.1

        self.ts_offset = 0.2

        self.controller.set_gain(gain)

        self.gripper_offset = 0.018

        # self.engaged = True

        
        # self.gripper_width = self.cfg.get("gripper_width", None)

        # if self.gripper_width is None:
        #     raise RuntimeError("Gripper value not initialized in config.yaml")
    
    def __create_rs_recorder(self, cfg):
        rs_cfg = cfg.get("realsense", {})
        
        if self.arm == "right":
            default_serial_num = "230322277025"
            serial_num = rs_cfg.get("right", default_serial_num)
        elif self.arm == "left":
            default_serial_num = "218622279810"
            serial_num = rs_cfg.get("left", default_serial_num)
        
        self.rs_recorder = RealSenseRecorder(serial_number=serial_num)

    def __create_cam_recorders(self, cfg):
        self.aria_recorder = AriaRecorder()
        self.__create_rs_recorder(self.cfg)

        # camera_cfg = self.cfg.get("cameras")
        # self.cameras = dict()
        # if camera_cfg is None:
        #     raise RuntimeError("Camera not configured in config.yaml")
        # if ("cam_high" in camera_cfg) and (
        #     camera_cfg["cam_high"].get("enabled", False)
        # ):
        #     self.cameras["cam_high"] = AriaRecorder()
        # if ("cam_right_wrist" in camera_cfg) and (
        #     camera_cfg["cam_right_wrist"].get("enabled", False)
        # ):
        #     self.cameras["cam_right_wrist"] = RealSenseRecorder(
        #         camera_cfg["cam_right_wrist"]["serial_number"]
        #     )
        # if ("cam_left_wrist" in camera_cfg) and (
        #     camera_cfg["cam_left_wrist"].get("enabled", False)
        # ):
        #     self.cameras["cam_left_wrist"] = RealSenseRecorder(
        #         camera_cfg["cam_left_wrist"]["serial_number"]
        #     )



    def set_joints(self, desired_position):
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
        cur_joint_state = self.controller.get_joint_state()
        current_ts = getattr(cur_joint_state, "timestamp", 0.0)
        self.timestamp = current_ts + self.ts_offset

        requested = ArxJointState(
            desired_position.astype(np.float32),
            velocity.astype(np.float32),
            torque.astype(np.float32),
            float(self.timestamp),
        )

        requested.gripper_pos = float(gripper_cmd) - self.gripper_offset
        requested.gripper_vel = 0.2
        requested.gripper_torque = 0.1

        self.controller.set_joint_cmd(requested)

    #x,y,z,y,p,r
    def set_pose(self, pose):
        if pose.shape != (7,):
            raise ValueError(
                "For Eva, target position must be of shape (7,) for single arm"
            )
        arm_joints = self.solve_ik(pose[:6])
        joints = np.concatenate([arm_joints, [pose[6]]])
        self.set_joints(joints)

    def get_obs(self):
        obs = {}
        obs["joint_positions"] = self.get_joints()
        obs["ee_pose"] = self.get_pose()

        # camera logic
        obs["front_img_1"] = self.aria_recorder.get_image()
        
        if self.arm == "right":
            obs["right_wrist_img"] = self.rs_recorder.get_image()
        elif self.arm == "left":
            obs["left_wrist_img"] = self.rs_recorder.get_image()
        return obs

    # removed static since can't figure out how to create ik when robot_urdf is not static
    def solve_ik(self, ee_pose):
        if ee_pose.shape != (6,):
            raise ValueError(
                "For Eva, target position must be of shape (7,) for single arm"
            )
        pos_xyz = ee_pose[:3]
        quat_xyzw = R.from_euler(
            "ZYX", ee_pose[3:6], degrees=False
        ).as_quat()  # scipy output xyzw
        arm_joints = self.kinematics_solver.inverse_kinematics(pos_xyz, quat_xyzw, self.get_joints())
        return arm_joints

    def get_joints(self):
        joints = self.controller.get_joint_state()
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

    def get_pose(self):
        """

        Returns:
           xyz: np.array, quat: np.array (xyzw)
        """
        joints = self.get_joints()
        pos, quat = self.kinematics_solver.forward_kinematics(joints)        
        rot = R.from_quat(quat)

        return pos, rot

    def set_home(self):
        self.controller.reset_to_home()


class DualARXInterface(Robot_Interface):
    def __init__(self, arm):
        super.__init__()

        self.arm = arm
        self.__create_controllers(self.cfg)

    def _create_controllers(self, cfg):
        if self.arm == "both":
            interfaces_cfg = cfg.get("interfaces", {})

            default_right_iface = "can2"
            selected_right_interface = interfaces_cfg.get("right", default_right_iface)

            default_left_iface = "can1"
            selected_left_interface = interfaces_cfg.get("left", default_left_iface)

            self.right_controller = Arx5JointController(
                self.robot_config, self.controller_config, selected_right_interface
            )
            self.left_controller = Arx5JointController(
                self.robot_config, self.controller_config, selected_left_interface
            )

            right_gain = self.right_controller.get_gain()
            left_gain = self.left_controller.get_gain()

            kp = np.array(
                [6.225, 17.225, 18.225, 12.225, 8.225, 6.225], dtype=np.float64
            )
            kd = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0], dtype=np.float64)

            right_gain.kp()[:] = kp
            right_gain.kd()[:] = kd
            right_gain.gripper_kp = 1.0
            right_gain.gripper_kd = 0.1

            left_gain.kp()[:] = kp
            left_gain.kd()[:] = kd
            left_gain.gripper_kp = 1.0
            left_gain.gripper_kd = 0.1

            self.right_controller.set_gain(right_gain)
            self.left_controller.set_gain(left_gain)
        elif self.arm == "right":
            interfaces_cfg = cfg.get("interfaces", {})
            default_iface = "can2"
            selected_interface = interfaces_cfg.get("right", default_iface)

            self.right_controller = Arx5JointController(
                self.robot_config, self.controller_config, selected_interface
            )

            gain = self.right_controller.get_gain()

            kp = np.array(
                [6.225, 17.225, 18.225, 12.225, 8.225, 6.225], dtype=np.float64
            )
            kd = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0], dtype=np.float64)
            gain.kp()[:] = kp
            gain.kd()[:] = kd
            gain.gripper_kp = 1.0
            gain.gripper_kd = 0.1

            self.right_controller.set_gain(gain)
        elif self.arm == "left":
            interfaces_cfg = cfg.get("interfaces", {})
            default_iface = "can1"
            selected_interface = interfaces_cfg.get("left", default_iface)

            self.right_controller = Arx5JointController(
                self.robot_config, self.controller_config, selected_interface
            )

            gain = self.right_controller.get_gain()

            kp = np.array(
                [6.225, 17.225, 18.225, 12.225, 8.225, 6.225], dtype=np.float64
            )
            kd = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0], dtype=np.float64)
            gain.kp()[:] = kp
            gain.kd()[:] = kd
            gain.gripper_kp = 1.0
            gain.gripper_kd = 0.1

            self.right_controller.set_gain(gain)

    def set_joints(self, desired_position):
        if (self.arm == "right" or self.arm == "left") and desired_position.shape != (
            7,
        ):
            raise ValueError(
                "For Eva, desired position must be of shape (7,) for arm: " + self.arm
            )
        elif self.arm == "both" and desired_position.shape != (14,):
            raise ValueError(
                "For Eva, desired position must be of shape (14,) for arm: " + self.arm
            )

        velocity = np.zeros_like(desired_position) + 0.1
        torque = np.zeros_like(desired_position) + 0.1

        requested = ArxJointState(
            desired_position.astype(np.float32),
            velocity.astype(np.float32),
            torque.astype(np.float32),
            float(self.timestamp),
        )

        requested.gripper_pos = float(
            self.gripper_cmd
        )  # Elmo had a "- 0.018" here. Check on this
        requested.gripper_vel = 0.2
        requested.gripper_torque = 0.1

        if self.arm == "right":
            self.right_controller.set_joint_cmd(requested)
        elif self.arm == "left":
            self.left_controller.set_joint_cmd(requested)
        elif self.arm == "both":
            self.left_controller.set_joint_cmd(requested[:7])
            self.right_controller.set_joint_cmd(requested[7:])

        self.timestamp += 1

    def set_pose(self):
        pass

    def get_obs(self):
        obs = {}
        obs["joint_positions"] = self.get_joints()
        obs["ee_pose"] = self.get_pose()

        # camera logic
        obs["front_img_1"] = self.cameras["cam_high"].get_image()

        if self.arm == "right":
            obs["right_wrist_img"] = self.cameras["right_wrist_img"].get_image()
        elif self.arm == "left":
            obs["left_wrist_img"] = self.cameras["left_wrist_img"].get_image()

        return obs

    @staticmethod
    def solve_ik():
        pass

    def get_joints(self):
        if self.arm == "left":
            curr_joints = self.left_controller.get_joint_state()
        elif self.arm == "right":
            curr_joints = self.right_controller.get_joint_state()
        elif self.arm == "both":
            curr_right_joints = self.right_controller.get_joint_state()
            curr_left_joints = self.left_controller.get_joint_state()
            curr_joints = np.concatenate((curr_left_joints, curr_right_joints), axis=0)
        return curr_joints

    # Add docstrings. EEpose has a lot of conventions.
    def get_pose(self):
        joints = self.get_joints()
        chain = pk.build_serial_chain_from_urdf(open(self.robot_urdf).read(), "link6")
        matrix = chain.forward_kinematics(joints, end_only=True).get_matrix()
        x, y, z = matrix[:, :3, 3]
        quat = pk.matrix_to_quaternion(matrix[:, :3, :3])
        R = R.from_quat(quat)
        r, p, yaw = R.as_euler("xyz", degrees=True)
        return [x, y, z, r, p, yaw]

    def set_home(self):
        if self.arm == "right":
            self.right_controller.reset_to_home()
        elif self.arm == "left":
            self.left_controller.reset_to_home()
        elif self.arm == "both":
            self.right_controller.reset_to_home()
            self.left_controller.reset_to_home()
