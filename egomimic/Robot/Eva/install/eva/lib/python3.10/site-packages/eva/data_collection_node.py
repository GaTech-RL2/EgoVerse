import os
from tracemalloc import start
import h5py
from tqdm import tqdm
import argparse
from typing import Optional, List, Dict

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState, Image
from std_msgs.msg import Int8, Bool, Float32
from cv_bridge import CvBridge

import arx5.arx5_interface as arx5
from arx5.arx5_interface import Arx5JointController, JointState as ArxJointState, Gain

from functools import partial
from threading import Thread, Event

import yaml
from ament_index_python.packages import get_package_share_directory


class EvaDataCollectionNode(Node):
    def __init__(
        self,
    ):
        super().__init__("eva_data_collect")
        self.declare_parameter("topics_file", "")
        self.declare_parameter("task_config_file", "")
        topics_file = (
            self.get_parameter("topics_file").get_parameter_value().string_value
        )
        task_config_file = (
            self.get_parameter("task_config_file").get_parameter_value().string_value
        )
        share = get_package_share_directory("eva")
        if not topics_file:
            topics_file = os.path.join(share, "config", "topics.yaml")
        if not task_config_file:
            task_config_file = os.path.join(
                share, "config", "task_config", "test_task.yaml"
            )

        with open(topics_file, "r") as f:
            topics_cfg = yaml.safe_load(f)
            if topics_cfg is None:
                raise ValueError("Topics config is not loaded properly.")
        with open(task_config_file, "r") as f:
            task_config = yaml.safe_load(f)
            if task_config is None:
                raise ValueError("Task config is not loaded properly.")

        self.robot_prefix_in = topics_cfg.get("robot_prefix", "eva").lstrip("/")
        self.vr_prefix_in = topics_cfg.get("vr_prefix", "/vr").lstrip("/")
        self.ik_prefix_in = topics_cfg.get("ik_prefix", "/eva_ik").lstrip("/")
        cameras_cfg = topics_cfg.get("cameras", {})
        self.cameras = [
            (name, cfg)
            for name, cfg in cameras_cfg.items()
            if cfg.get("enabled", False)
        ]
        arms_cfg = topics_cfg.get("arms", {})
        self.arms = [
            (name, cfg) for name, cfg in arms_cfg.items() if cfg.get("enabled", False)
        ]
        self.task_config = task_config
        self.hz = self.task_config.get("hz")

        self.overwrite = False

        qos = QoSProfile(depth=50)
        self.bridge = CvBridge()

        self.cur_robot_joint: Dict[str, np.ndarray] = {}
        self.cur_cmd_joint: Dict[str, np.ndarray] = {}
        self.cur_target_ee = dict()
        self.cur_img: Dict[str, np.ndarray] = {}

        self.button_x = False

        for name, cfg in self.arms:
            self.create_subscription(
                JointState,
                cfg.get("robot_joint_topic"),
                partial(self._on_robot_joint, name),
                qos,
            )
            self.create_subscription(
                JointState,
                cfg.get("ik_joint_topic"),
                partial(self._on_cmd_joint, name),
                qos,
            )
            self.create_subscription(
                PoseStamped,
                cfg.get("ik_ee_topic"),
                partial(self._on_target_ee, name),
                qos,
            )

        for camera_name, camera_cfg in self.cameras:
            self.create_subscription(
                Image, camera_cfg["topic"], partial(self._on_images, camera_name), qos
            )

        self.create_subscription(
            Bool, f"/{self.vr_prefix_in}/button_x", self._on_button_x, qos
        )

        # Stage logic
        self.stage = 0
        self._recording = Event()
        self.dt = 1.0 / max(1e-6, self.hz)
        self.create_timer(1.0 / self.hz, self.collection_control)

    def _on_robot_joint(self, arm: str, msg: JointState):
        if msg.position:
            self.cur_robot_joint[arm] = np.asarray(msg.position, dtype=float)

    def _on_cmd_joint(self, arm: str, msg: JointState):
        if msg.position:
            self.cur_cmd_joint[arm] = np.asarray(msg.position, dtype=float)

    def _on_target_ee(self, arm: str, msg: PoseStamped):
        self.cur_target_ee[arm] = msg

    def _on_images(self, camera: str, msg: Image):
        self.cur_img[camera] = self.bridge.imgmsg_to_cv2(
            msg, desired_encoding="passthrough"
        )

    def _on_button_x(self, msg: Bool):
        self.button_x = bool(msg.data)

    def get_robot_joint(self):
        output = np.zeros((14,))
        for name, cfg in self.arms:
            if name == "r":
                output[7:] = self.cur_robot_joint[name]
            else:
                output[:7] = self.cur_robot_joint[name]
        return output

    def get_cmd_joint(self):
        output = np.zeros((14,))
        for name, cfg in self.arms:
            if name == "r":
                output[7:] = self.cur_cmd_joint[name]
            else:
                output[:7] = self.cur_cmd_joint[name]
        return output

    def get_target_ee(self):
        output = []
        for name, cfg in self.arms:
            output.append(self.cur_target_ee[name])
        return output

    def process_pose_stamped(self, ee_msgs, joints):
        output = []
        for i, msgs in enumerate(ee_msgs):
            target_ee_output = np.zeros(14)
            for j, arm_msg in enumerate(msgs):
                p = arm_msg.pose.position
                q = arm_msg.pose.orientation
                xyz = np.array([p.x, p.y, p.z], dtype=np.float64)
                xyzw = np.array([q.x, q.y, q.z, q.w], dtype=np.float64)
                ypr = R.from_quat(xyzw).as_euler("ZYX", degrees=False)
                start_idx = j * 7
                target_ee_output[start_idx : start_idx + 3] = xyz
                target_ee_output[start_idx + 3 : start_idx + 6] = ypr
                target_ee_output[start_idx + 6] = joints[i][j * 7 + 6]
            output.append(target_ee_output)
        return output

    def print_dt_diagnosis(self, actual_dt_history):
        for val in actual_dt_history:
            t0 = val[0].nanoseconds * 1e-9
            t1 = val[1].nanoseconds * 1e-9
            val[0] = t0
            val[1] = t1

        actual_dt_history = np.array(actual_dt_history)
        total_time = actual_dt_history[:, 1] - actual_dt_history[:, 0]

        dt_mean = np.mean(total_time)
        freq_mean = 1 / dt_mean

        return freq_mean

    def collection_control(self):
        if self.stage == 0:
            self.episode_id = int(input("Enter episode number: "))
            self.stage = 1
            self.get_logger().info(f"/{self.vr_prefix_in}/button_x")
        if self.stage == 1:
            if self.button_x == True:
                self.stage = 2
            # wait for button x from controller
        elif self.stage == 2:
            if (
                not all(
                    a in self.cur_robot_joint for a in [name for name, cfg in self.arms]
                )
                or not all(
                    a in self.cur_cmd_joint for a in [name for name, cfg in self.arms]
                )
                or not all(
                    a in self.cur_target_ee for a in [name for name, cfg in self.arms]
                )
                or not all(
                    a in self.cur_img
                    for a in [camera_name for camera_name, camera_cfg in self.cameras]
                )
            ):
                self.get_logger().info(
                    f"robot keys {self.cur_robot_joint.keys()}, cmd keys {self.cur_cmd_joint.keys()}, target ee keys {self.cur_target_ee.keys()}, camera keys {self.cur_img.keys()}"
                )
                self.get_logger().info(
                    f"Comp list {[name for name, cfg in self.arms]} and {[camera_name for camera_name, camera_cfg in self.cameras]}"
                )
            elif not self._recording.is_set():
                self._recording.set()
                Thread(target=self._record_worker, daemon=True).start()
        else:
            self.get_logger().info(
                f"Episode {self.episode_id} data collection is completed"
            )
            self.stage = 0

    def _record_worker(self):
        try:
            dataset_dir = self.task_config.get("dataset_dir")
            dataset_name = self.task_config.get("dataset_name")
            dataset_path = os.path.join(dataset_dir, dataset_name)
            os.makedirs(dataset_path, exist_ok=True)
            episode_path = os.path.join(dataset_path, f"episode_{self.episode_id}")

            max_timesteps = int(self.task_config.get("episode_len"))

            self.get_logger().info("Start recording data")
            ok = self.record_one_episode(max_timesteps, episode_path)
            self.stage = 3 if ok else 0
        finally:
            self._recording.clear()

    def record_one_episode(self, max_timesteps, dataset_path):
        data_dict = dict()
        for camera_name, camera_cfg in self.cameras:
            data_dict[f"/observations/images/{camera_name}"] = []

        time0 = self.get_clock().now()
        timesteps = []
        cmd_joint_actions = []
        robot_joint_actions = []
        joint_vels = []
        target_ees = []
        actual_dt_history = []

        rate = self.create_rate(self.hz, self.get_clock())
        for t in tqdm(range(max_timesteps)):
            rclpy.spin_once(self, timeout_sec=0.0)

            t0 = self.get_clock().now()
            robot_joint = self.get_robot_joint()
            cmd_joint = self.get_cmd_joint()
            target_ee = self.get_target_ee()

            # l_cart_action = node.get_obs().cartesian
            # r_cart_action = node.getFollowerCartesianPos("right")
            # cart_action = np.concatenate([l_cart_action, r_cart_action])

            timesteps.append(t0)
            robot_joint_actions.append(robot_joint)
            cmd_joint_actions.append(cmd_joint)
            target_ees.append(target_ee)
            for camera_name, camera_cfg in self.cameras:
                data_dict[f"/observations/images/{camera_name}"].append(
                    self.cur_img[camera_name]
                )
            t1 = self.get_clock().now()
            actual_dt_history.append([t0, t1])
            rate.sleep()
        self.get_logger().info(
            f"Avg fps: {max_timesteps / ((self.get_clock().now() - time0).nanoseconds * 1e-9) }"
        )

        freq_mean = self.print_dt_diagnosis(actual_dt_history)
        if freq_mean < 30:
            self.get_logger().info(
                f"\n\nfreq mean is {freq_mean}, lower than 30 recollecting... \n\n\n"
            )
            return False

        # Process ee
        target_ee_proc = self.process_pose_stamped(target_ees, cmd_joint_actions)

        data_dict["/observations/joint_positions"] = robot_joint_actions
        # data_dict["/observations/qjointvel"] = joint_vels
        data_dict["/observations/target_ee_pose"] = target_ee_proc
        data_dict["/action"] = cmd_joint_actions

        t0 = self.get_clock().now()
        with h5py.File(dataset_path + ".hdf5", "w", rdcc_nbytes=1024**2 * 2) as root:
            root.attrs["sim"] = False
            obs = root.create_group("observations")
            image = obs.create_group("images")
            for cam_name, camera_cfg in self.cameras:
                _ = image.create_dataset(
                    cam_name,
                    (max_timesteps, 480, 640, 3),
                    dtype="uint8",
                    chunks=(1, 480, 640, 3),
                )
            _ = obs.create_dataset("joint_positions", (max_timesteps, 14))
            # _ = obs.create_dataset("qjointvel", (max_timesteps, 16))
            _ = obs.create_dataset("target_ee_pose", (max_timesteps, 14))
            _ = root.create_dataset("action", (max_timesteps, 14))

            for name, array in data_dict.items():
                root[name][...] = array

        self.get_logger().info(
            f"Saving: {((self.get_clock().now() - t0).nanoseconds * 1e-9):.1f} secs"
        )
        return True


def main(args=None) -> None:
    rclpy.init(args=args)
    node = EvaDataCollectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
