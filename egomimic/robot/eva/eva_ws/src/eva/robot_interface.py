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

        self.controller_config = (
            arx5.ControllerConfigFactory.get_instance().get_config(
                "joint_controller", self.robot_config.joint_dof
            )
        )

        self.engaged = True
        self.timestamp = 0

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
    def set_joint(self, desired_position):
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
    def get_joint(self):
        pass

    @abstractmethod
    def get_pose(self):
        pass

    @abstractmethod
    def set_home(self):
        pass



class SingleARXInterface(Robot_Interface):
    def __init__(self, arm):
        super.__init__()
        
        self.arm = arm        
        self._create_controllers(self.cfg)
    
    def _create_controllers(self, cfg):
        if self.arm == "right":
            interfaces_cfg = cfg.get("interfaces", {})
            default_iface = "can2"
            selected_interface = interfaces_cfg.get("right", default_iface)
        elif self.arm == "left":
            interfaces_cfg = cfg.get("interfaces", {})
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

        self.controller.set_gain(gain)


    def set_joint(self, desired_position):
        if desired_position.shape != (7,):
            raise ValueError("For Eva, desired position must be of shape (7,) for single arm")
        
        velocity = np.zeros_like(desired_position) + 0.1
        torque = np.zeros_like(desired_position) + 0.1

        requested= ArxJointState(
            desired_position.astype(np.float32),
            velocity.astype(np.float32),
            torque.astype(np.float32),
            float(self.timestamp)
        )

        requested.gripper_pos = float(self.gripper_cmd) # Elmo had a "- 0.018" here. Check on this
        requested.gripper_vel = 0.2
        requested.gripper_torque = 0.1

        self.controller.set_joint_cmd(requested)

        self.timestamp += 1


    def set_pose(self):
        pass

    def get_obs(self):
        obs = {}
        obs["joint_positions"] = self.get_joint()
        obs["ee_pose"] = self.get_pose()

        #camera logic
        obs["front_img_1"] = 1
        
        if self.arm == "right":
            obs["right_wrist_img"] = 1
        elif self.arm == "left":
            obs["left_wrist_img"] = 1

        return obs

    @staticmethod
    def solve_ik():
        pass

    def get_joint(self):
        curr_joints = self.controller.get_joint_state()
        return curr_joints

    #Add docstrings. EEpose has a lot of conventions.
    def get_pose(self):
        joints = self.get_joint()

        chain = pk.build_serial_chain_from_urdf(open(self.robot_urdf).read(), "link6")
        matrix = chain.forward_kinematics(joints, end_only=True).get_matrix()
        x, y, z = matrix[:, :3, 3]

        quat = pk.matrix_to_quaternion(matrix[:, :3, :3])
        R = R.from_quat(quat)
        r, p, yaw = R.as_euler('xyz', degrees=True)
        
        return [x, y, z, r, p, yaw]


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

            self.right_controller = Arx5JointController(self.robot_config, self.controller_config, selected_right_interface)
            self.left_controller = Arx5JointController(self.robot_config, self.controller_config, selected_left_interface)

            right_gain = self.right_controller.get_gain()
            left_gain = self.left_controller.get_gain()

            kp = np.array([6.225, 17.225, 18.225, 12.225, 8.225, 6.225], dtype=np.float64)
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

            self.right_controller = Arx5JointController(self.robot_config, self.controller_config, selected_interface)

            gain = self.right_controller.get_gain()

            kp = np.array([6.225, 17.225, 18.225, 12.225, 8.225, 6.225], dtype=np.float64)
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

            self.right_controller = Arx5JointController(self.robot_config, self.controller_config, selected_interface)

            gain = self.right_controller.get_gain()

            kp = np.array([6.225, 17.225, 18.225, 12.225, 8.225, 6.225], dtype=np.float64)
            kd = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0], dtype=np.float64)
            gain.kp()[:] = kp
            gain.kd()[:] = kd
            gain.gripper_kp = 1.0
            gain.gripper_kd = 0.1

            self.right_controller.set_gain(gain)


    def set_joint(self, desired_position):
        if (self.arm == "right" or self.arm == "left") and desired_position.shape != (7,):
            raise ValueError("For Eva, desired position must be of shape (7,) for arm: " + self.arm)
        elif self.arm == "both" and desired_position.shape != (14,):
            raise ValueError("For Eva, desired position must be of shape (14,) for arm: " + self.arm)
        
        velocity = np.zeros_like(desired_position) + 0.1
        torque = np.zeros_like(desired_position) + 0.1

        requested= ArxJointState(
            desired_position.astype(np.float32),
            velocity.astype(np.float32),
            torque.astype(np.float32),
            float(self.timestamp)
        )

        requested.gripper_pos = float(self.gripper_cmd) # Elmo had a "- 0.018" here. Check on this
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
        obs["joint_positions"] = self.get_joint()
        obs["ee_pose"] = self.get_pose()

        #camera logic
        obs["front_img_1"] = 1
        
        if self.arm == "right":
            obs["right_wrist_img"] = 1
        elif self.arm == "left":
            obs["left_wrist_img"] = 1
        elif self.arm == "both":
            obs["right_wrist_img"] = 1
            obs["left_wrist_img"] = 1

        return obs

    @staticmethod
    def solve_ik():
        pass

    def get_joint(self):
        if self.arm == "left":
            curr_joints = self.left_controller.get_joint_state()
        elif self.arm == "right":
            curr_joints = self.right_controller.get_joint_state()
        elif self.arm == "both":
            curr_right_joints = self.right_controller.get_joint_state()
            curr_left_joints = self.left_controller.get_joint_state()
            curr_joints = np.concatenate((curr_left_joints, curr_right_joints), axis=0)
        return curr_joints

    #Add docstrings. EEpose has a lot of conventions.
    def get_pose(self):
        joints = self.get_joint()
        chain = pk.build_serial_chain_from_urdf(open(self.robot_urdf).read(), "link6")
        matrix = chain.forward_kinematics(joints, end_only=True).get_matrix()
        x, y, z = matrix[:, :3, 3]
        quat = pk.matrix_to_quaternion(matrix[:, :3, :3])
        R = R.from_quat(quat)
        r, p, yaw = R.as_euler('xyz', degrees=True)
        return [x, y, z, r, p, yaw]


    def set_home(self):
        if self.arm == "right":
            self.right_controller.reset_to_home()
        elif self.arm == "left":
            self.left_controller.reset_to_home()
        elif self.arm == "both":
            self.right_controller.reset_to_home()
            self.left_controller.reset_to_home()