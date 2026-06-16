from abc import ABC, abstractmethod
from pathlib import Path

import time
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

from egomimic.rldb.embodiment.eva import Eva
from egomimic.rldb.zarr.zarr_dataset_multi import ZarrDataset
from egomimic.robot.eva.eva_kinematics import EvaMinkKinematicsSolver

try:
    import arx5.arx5_interface as arx5
    from arx5.arx5_interface import Arx5JointController
    from arx5.arx5_interface import JointState as ArxJointState
except ImportError:
    arx5 = None
    Arx5JointController = None
    ArxJointState = None

try:
    from stream_aria import AriaRecorder
    from stream_d405 import RealSenseRecorder
except ImportError:
    AriaRecorder = None
    RealSenseRecorder = None


def _get_model_xml_path():
    candidates = [
        "/home/robot/robot_ws/egomimic/resources/model_x5.xml",
        Path(__file__).resolve().parents[5] / "resources" / "model_x5.xml",
    ]
    for candidate in candidates:
        candidate = str(candidate)
        if Path(candidate).exists():
            return candidate
    return str(candidates[-1])


class Robot_Interface(ABC):
    def __init__(self):
        if arx5 is None or Arx5JointController is None or ArxJointState is None:
            raise ImportError(
                "Live robot interface dependencies are unavailable. Use offline debug mode or install the ARX interface stack."
            )
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
        # Raise gripper torque limit so the torque protection (triggered at
        # gripper_torque_max/2) doesn't freeze the close command before the
        # gripper reaches position zero.  Default 1.5 Nm gives only a 0.75 Nm
        # threshold — lower than the ~1 Nm the motor draws closing from open.
        self.robot_config.gripper_torque_max = 4.0

        self.controller_config = arx5.ControllerConfigFactory.get_instance().get_config(
            "joint_controller", self.robot_config.joint_dof
        )

    def __get_config(self, cfg):
        # share = get_package_share_directory("eva")
        # cfg_path = os.path.join(share, "config", "configs.yaml")
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
            model_path=_get_model_xml_path()
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
            gain.gripper_kp = 5.0
            gain.gripper_kd = 0.2

            self.ts_offset = 0.2

            self.controller[arm].set_gain(gain)

            self.gripper_offset = 0.000

        gripper_cfg = self.cfg.get("gripper", {})
        self.gripper_open = {}
        self.gripper_close = {}
        self.gripper_width = {}
        for arm in self.arms:
            arm_cfg = gripper_cfg.get(arm, {})
            self.gripper_open[arm] = arm_cfg.get("open", 0.08)
            self.gripper_close[arm] = arm_cfg.get("close", 0.0)
            self.gripper_width[arm] = self.gripper_open[arm] - self.gripper_close[arm]

    def __create_cam_recorders(self, cameras_cfg):
        if AriaRecorder is None or RealSenseRecorder is None:
            raise ImportError(
                "Camera recorder dependencies are unavailable. Install the live robot streaming stack to use ARXInterface."
            )
        self.recorders = dict()
        self.camera_res = dict()
        for name, cam_cfg in cameras_cfg.items():
            if not cam_cfg["enabled"]:
                continue
            cam_type = cam_cfg["type"]
            if cam_type == "aria":
                self.recorders[name] = AriaRecorder(
                    profile_name="profile15",
                    use_security=True,
                    height=cam_cfg["height"],
                    width=cam_cfg["width"],
                )
                self.recorders[name].start()
            elif cam_type == "d405":
                self.recorders[name] = RealSenseRecorder(str(cam_cfg["serial_number"]))
            else:
                raise ValueError("Invalid value in the config")
            self.camera_res[name] = (cam_cfg["height"], cam_cfg["width"])

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

        requested = ArxJointState(
            desired_position.astype(np.float32),
            velocity.astype(np.float32),
            torque.astype(np.float32),
            float(self.timestamp),
        )

        # Denormalize gripper from [0, 1] to hardware range
        gripper_cmd = (
            float(gripper_cmd) * self.gripper_width[arm] + self.gripper_close[arm]
        )
        requested.gripper_pos = gripper_cmd
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
        gripper_raw = getattr(joints, "gripper_pos", 0.0)
        gripper = (gripper_raw - self.gripper_close[arm]) / self.gripper_width[arm]
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
            joints = self.get_joints(arm)
            joints[6] = 1.0
            self.set_joints(joints, arm)
            time.sleep(1)
            self.controller[arm].reset_to_home()


class OfflineARXInterface:
    DEFAULT_IMAGE_SHAPE = (480, 640, 3)

    def __init__(self, arms, dataset_path=None):
        self.arms = arms
        self.recorders = {}
        self._joint_positions = {
            arm: np.zeros(7, dtype=np.float64) for arm in ("left", "right")
        }
        self._ee_pose = {
            arm: np.zeros(7, dtype=np.float64) for arm in ("left", "right")
        }
        self.dataset_path = str(dataset_path) if dataset_path is not None else None
        self.dataset = None
        self.frame_idx = 0
        if self.dataset_path is not None:
            self.dataset = self._build_dataset(self.dataset_path)
            if len(self.dataset) == 0:
                raise ValueError(f"Offline dataset is empty: {self.dataset_path}")

    def _build_dataset(self, dataset_path):
        episode_path = Path(dataset_path)
        if not episode_path.exists():
            raise FileNotFoundError(f"Offline episode path not found: {dataset_path}")
        return ZarrDataset(
            episode_path,
            key_map=Eva.get_keymap(),
            transform_list=None,
        )

    def _create_controllers(self, cfg):
        return None

    @staticmethod
    def _to_numpy(value):
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    @classmethod
    def _image_to_bgr_uint8(cls, image):
        image = cls._to_numpy(image)
        if image.ndim != 3:
            raise ValueError(f"Expected image with 3 dims, got shape {image.shape}")
        if image.shape[0] in (1, 3):
            image = np.transpose(image, (1, 2, 0))
        if image.shape[-1] != 3:
            raise ValueError(f"Expected image channels-last RGB, got {image.shape}")
        if image.dtype != np.uint8:
            if np.issubdtype(image.dtype, np.floating) and np.max(image) <= 1.0:
                image = image * 255.0
            image = np.clip(image, 0, 255).astype(np.uint8)
        return image[..., [2, 1, 0]]

    @classmethod
    def _pose_wxyz_to_xyzypr(cls, pose):
        pose = cls._to_numpy(pose).astype(np.float64, copy=False).reshape(-1)
        if pose.shape != (7,):
            raise ValueError(f"Expected xyz+quat(wxyz) pose, got shape {pose.shape}")
        quat_xyzw = pose[[4, 5, 6, 3]]
        ypr = R.from_quat(quat_xyzw).as_euler("ZYX", degrees=False)
        return np.concatenate([pose[:3], ypr], axis=-1)

    def _next_dataset_sample(self):
        if self.dataset is None:
            return None
        sample = self.dataset[self.frame_idx % len(self.dataset)]
        self.frame_idx += 1
        return sample

    def _obs_from_dataset_sample(self, sample):
        sample = {k: self._to_numpy(v) for k, v in sample.items()}
        left_pose = self._pose_wxyz_to_xyzypr(sample["left.obs_ee_pose"])
        right_pose = self._pose_wxyz_to_xyzypr(sample["right.obs_ee_pose"])
        left_gripper = float(np.asarray(sample["left.obs_gripper"]).reshape(-1)[0])
        right_gripper = float(np.asarray(sample["right.obs_gripper"]).reshape(-1)[0])

        ee_poses = np.zeros(14, dtype=np.float64)
        ee_poses[:6] = left_pose
        ee_poses[6] = left_gripper
        ee_poses[7:13] = right_pose
        ee_poses[13] = right_gripper

        obs = {
            "front_img_1": self._image_to_bgr_uint8(
                sample["observations.images.front_img_1"]
            ),
            "left_wrist_img": self._image_to_bgr_uint8(
                sample["observations.images.left_wrist_img"]
            ),
            "right_wrist_img": self._image_to_bgr_uint8(
                sample["observations.images.right_wrist_img"]
            ),
            "joint_positions": np.concatenate(
                [self.get_joints("left"), self.get_joints("right")], axis=0
            ),
            "ee_poses": ee_poses,
        }
        return obs

    def set_joints(self, desired_position, arm):
        desired_position = np.asarray(desired_position, dtype=np.float64)
        if desired_position.shape != (7,):
            raise ValueError(
                "For Eva, desired position must be of shape (7,) for single arm"
            )
        self._joint_positions[arm] = desired_position.copy()

    def set_pose(self, pose, arm):
        pose = np.asarray(pose, dtype=np.float64)
        if pose.shape != (7,):
            raise ValueError(
                f"For Eva, target position must be of shape (7,), current shape: {pose.shape}"
            )
        self._ee_pose[arm] = pose.copy()
        joints = np.zeros(7, dtype=np.float64)
        joints[6] = pose[6]
        self._joint_positions[arm] = joints
        return joints

    def get_obs(self):
        sample = self._next_dataset_sample()
        if sample is not None:
            return self._obs_from_dataset_sample(sample)

        obs = {
            "front_img_1": np.zeros(self.DEFAULT_IMAGE_SHAPE, dtype=np.uint8),
            "left_wrist_img": np.zeros(self.DEFAULT_IMAGE_SHAPE, dtype=np.uint8),
            "right_wrist_img": np.zeros(self.DEFAULT_IMAGE_SHAPE, dtype=np.uint8),
        }
        joint_positions = np.concatenate(
            [self.get_joints("left"), self.get_joints("right")], axis=0
        )
        ee_poses = np.concatenate([self._ee_pose["left"], self._ee_pose["right"]])
        obs["joint_positions"] = joint_positions
        obs["ee_poses"] = ee_poses
        return obs

    def solve_ik(self, ee_pose, arm):
        ee_pose = np.asarray(ee_pose, dtype=np.float64)
        if ee_pose.shape != (6,):
            raise ValueError(
                "For Eva, target position must be of shape (6,) for single arm"
            )
        return np.zeros(6, dtype=np.float64)

    def get_joints(self, arm):
        return self._joint_positions[arm].copy()

    def get_pose(self, arm, se3=False):
        pose = self._ee_pose[arm]
        pos = pose[:3].copy()
        rot = R.from_euler("ZYX", pose[3:6], degrees=False)
        if se3:
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = rot.as_matrix()
            T[:3, 3] = pos
            return T
        return pos, rot

    def set_home(self):
        for arm in ("left", "right"):
            self._joint_positions[arm] = np.zeros(7, dtype=np.float64)
            self._ee_pose[arm] = np.zeros(7, dtype=np.float64)
        self.frame_idx = 0


if __name__ == "__main__":
    # Run Eva example
    # Note: Update the URDF path before running
    ri = ARXInterface(arms=["right"])
    joints = ri.get_joints("right")
    breakpoint()
