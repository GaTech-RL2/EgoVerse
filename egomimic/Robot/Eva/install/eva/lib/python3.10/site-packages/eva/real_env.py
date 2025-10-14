import collections
import time
from typing import Optional

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import JointState


class RealEnv(Node):
    """
    Minimal real environment wrapper that coordinates VR controller, IK, and robot nodes.
    This mirrors the interface of EgoMimic-Eve's real_env where applicable, but uses VR
    teleop instead of leader/follower robots.

    Exposed methods:
      - reset(fake=False): prepare system and return initial observation placeholder
      - step(action): no-op placeholder; robot motion is driven by VR/IK nodes
      - get_observation(): returns latest qpos from robot node
    """

    def __init__(self, active_arms: str = "right", setup_robots: bool = True) -> None:
        super().__init__("eva_clean_real_env")
        self.active_arms = active_arms
        self.qos = QoSProfile(depth=50)

        # Buffers
        self._latest_qpos: Optional[np.ndarray] = None

        # Subscriptions
        # Robot node publishes current executed joints on /eva/r/joints
        self.create_subscription(JointState, "/eva/r/joints", self._on_robot_joints, self.qos)

        # Placeholders for initialization of VR/IK/Robot nodes: those are expected to run as separate processes
        # launched externally or via ROS 2 launch files.

    def _on_robot_joints(self, msg: JointState) -> None:
        if not msg.position:
            return
        vals = list(msg.position)
        if len(vals) < 7:
            vals = vals + [0.0] * (7 - len(vals))
        # Store 7 values [6 joints + gripper]
        self._latest_qpos = np.asarray(vals[:7], dtype=np.float64)

    def get_observation(self):
        obs = collections.OrderedDict()
        # qpos placeholder: right arm 6 + gripper 1; left omitted in this VR setup
        obs["qpos"] = self._latest_qpos if self._latest_qpos is not None else np.zeros(7, dtype=np.float64)
        # qvel/effort not implemented
        obs["qvel"] = np.zeros_like(obs["qpos"])  # placeholder
        obs["effort"] = np.zeros(14, dtype=np.float64)  # placeholder to match shape used elsewhere
        # images not collected here; handled by separate camera nodes / recorder
        obs["images"] = {}
        return obs

    def reset(self, fake: bool = False):
        # No hardware resets here; assume VR/IK/Robot nodes manage homing/anchors.
        return self.get_observation()

    def step(self, action, get_obs: bool = True):
        # Action is ignored: VR/IK drives motion; return latest observation
        if get_obs:
            return self.get_observation()
        return None


def make_real_env(node: Node = None, active_arms: str = "right", setup_robots: bool = True):
    # This helper mirrors EgoMimic-Eve naming, but just constructs the wrapper node.
    return RealEnv(active_arms=active_arms, setup_robots=setup_robots)


