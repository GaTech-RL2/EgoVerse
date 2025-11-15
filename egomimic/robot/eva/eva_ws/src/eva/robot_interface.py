import arx5.arx5_interface as arx5
from arx5.arx5_interface import Arx5JointController, JointState as ArxJointState, Gain
from arx5.arx5_interface import Arx5CartesianController, EEFState, Gain, LogLevel
from ament_index_python.packages import get_package_share_directory
import os
import yaml
import numpy as np
import pytorch_kinematics as pk
from scipy.spatial.transform import Rotation as R
from abc import ABC, abstractmethod
from stream_aria import AriaRecorder, update_iptables
from stream_d405 import RealSenseRecorder
from egomimic.robot.eva.eva_kinematics import EvaMinkKinematicsSolver


class Robot_Interface(ABC):
    def __init__(self):
        self.cfg = {}
        try:
            self.cfg = self.__get_config(self.cfg)
        except Exception as e:
            print(f"Failed to load configs.yaml: {e}")

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

    def __get_config(self, cfg):
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
        super().__init__()

        self.arms = arms
        self.controller = dict()
        self._create_controllers(self.cfg)
        self.__create_cam_recorders(self.cfg["cameras"])
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

            self.controller[arm] = Arx5JointController(
                self.robot_config, self.controller_config, selected_interface
            )
            self.controller[arm].reset_to_home()

            gain = self.controller[arm].get_gain()

            kp = self.cfg.get("kp", None)
            kd = self.cfg.get("kd", None)
            if kp is None or kd is None:
                raise RuntimeError(
                    "Both kp and kd must be specified in the config file."
                )
            kp = np.array(kp, dtype=np.float64)
            kd = np.array(kd, dtype=np.float64) * 0.6
            gain.kp()[:] = kp
            gain.kd()[:] = kd
            gain.gripper_kp = 1.0
            gain.gripper_kd = 0.1

            self.ts_offset = 0.2

            self.controller[arm].set_gain(gain)

            self.gripper_open_value = self.cfg.get("gripper_open_value", None)
            self.gripper_close_value = self.cfg.get("gripper_close_value", None)

            if self.gripper_close_value is None or self.gripper_open_value is None:
                raise RuntimeError(
                    "Gripper open/close values must be initialized in config.yaml (gripper_open_value and gripper_close_value or gripper_min)"
                )

            self.gripper_width = self.gripper_open_value - self.gripper_close_value

    def __create_cam_recorders(self, cameras_cfg):
        self.recorders = dict()
        for name, cam_cfg in cameras_cfg.items():
            if not cam_cfg["enabled"]:
                continue
            cam_type = cam_cfg["type"]
            if cam_type == "aria":
                self.recorders[name] = AriaRecorder(
                    profile_name="profile15", use_security=True
                )
                self.recorders[name].start()
            elif cam_type == "d405":
                self.recorders[name] = RealSenseRecorder(str(cam_cfg["serial_number"]))
            else:
                raise ValueError("Invalid value in the config")

    def norm_gripper_to_real_gripper(self, norm_gripper_val):
        norm_gripper_val = max(0.0, min(1.0, norm_gripper_val))
        return norm_gripper_val * self.gripper_width + self.gripper_close_value

    def real_gripper_to_norm_gripper(self, real_gripper_val):
        norm_gripper_val = (
            real_gripper_val - self.gripper_close_value
        ) / self.gripper_width
        return max(0.0, min(1.0, norm_gripper_val))

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

        cur_joint_state = self.controller[arm].get_joint_state()
        current_ts = getattr(cur_joint_state, "timestamp", 0.0)
        self.timestamp = current_ts + self.ts_offset

        requested = ArxJointState(
            desired_position.astype(np.float32),
            velocity.astype(np.float32),
            torque.astype(np.float32),
            float(self.timestamp),
        )

        requested.gripper_pos = self.norm_gripper_to_real_gripper(float(gripper_cmd))
        requested.gripper_vel = 0.1
        requested.gripper_torque = 0.2

        self.controller[arm].set_joint_cmd(requested)

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
            ee_poses[arm_offset : arm_offset + 7] = np.concatenate(
                [
                    xyz,
                    rot.as_euler("ZYX", degrees=False),
                    [joint_positions[arm_offset + 6]],
                ]
            )
        obs["joint_positions"] = joint_positions
        obs["ee_poses"] = ee_poses

        # camera logic
        for name, recorder in self.recorders.items():
            obs[name] = recorder.get_image()
        return obs

    def solve_ik(self, ee_pose, arm):
        if ee_pose.shape != (6,):
            raise ValueError(
                "For Eva, target position must be of shape (6,) for single arm"
            )
        pos_xyz = ee_pose[:3]
        rot_mat = R.from_euler("ZYX", ee_pose[3:6], degrees=False).as_matrix()
        arm_joints = self.kinematics_solver.ik(
            pos_xyz, rot_mat, cur_jnts=self.get_joints(arm)[:6]
        )
        return arm_joints

    def get_joints(self, arm):
        joints = self.controller[arm].get_joint_state()
        arm_joints = joints.pos()
        gripper = getattr(joints, "gripper_pos", 0.0)
        gripper = self.real_gripper_to_norm_gripper(float(gripper))
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


if __name__ == "__main__":
    ri = ARXInterface(arms=["left"])
    joints = ri.get_joints("left")
    breakpoint()
