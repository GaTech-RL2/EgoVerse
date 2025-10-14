import os
import h5py
from typing import List, Dict

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import JointState

from tqdm import tqdm

from threading import Thread


class EvaReplayNode(Node):
    def __init__(self, ik_prefix_in: str, arms: List[str], hz: int, task_config: dict):
        super().__init__("eva_replay")
        self.ik_prefix_in = ik_prefix_in
        self.arms = arms
        self.hz = hz

        dataset_dir = str(task_config["dataset_dir"])
        dataset_name = str(task_config["dataset_name"])
        self.dataset_path = os.path.join(dataset_dir, dataset_name) + ".hdf5"

        if not os.path.isfile(self.dataset_path):
            raise FileNotFoundError(f"HDF5 not found: {self.dataset_path}")

        qos = QoSProfile(depth=10)

        self.pub_cmd_joint: Dict[str, rclpy.publisher.Publisher] = {}
        for arm in self.arms:
            topic = f"/{self.ik_prefix_in}/{arm}/joint_state"
            self.pub_cmd_joint[arm] = self.create_publisher(JointState, topic, qos)

        with h5py.File(self.dataset_path, "r") as f:
            self.action = np.asarray(f["action"][...], dtype=np.float32)
        self.T = int(self.action.shape[0])
        total_cmd = int(self.action.shape[1])
        self.per_arm_cmd = total_cmd // len(self.arms)
        Thread(target=self.replay_once, daemon=True).start()

    def replay_once(self):
        rate = self.create_rate(self.hz, self.get_clock())
        for t in tqdm(range(self.T)):
            now = self.get_clock().now().to_msg()
            row = self.action[t]

            for i, arm in enumerate(self.arms):
                start = i * self.per_arm_cmd
                end = start + self.per_arm_cmd
                vec = row[start:end]

                msg = JointState()
                msg.header.stamp = now
                msg.position = vec.tolist()

                self.pub_cmd_joint[arm].publish(msg)

            rate.sleep()


def main():
    rclpy.init()

    task_config = {
        "dataset_dir": ".",
        "dataset_name": "testeva",
    }
    node = EvaReplayNode(
        ik_prefix_in="eva_ik",
        arms=["l", "r"],
        hz=50,
        task_config=task_config,
    )

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
