from typing import Optional, List

import os
import yaml
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from ament_index_python.packages import get_package_share_directory

import arx5.arx5_interface as arx5
from arx5.arx5_interface import Arx5JointController, JointState as ArxJointState, Gain


class IkToJointsNode(Node):
    def __init__(self) -> None:
        super().__init__("ik_to_joints_node")

        qos = QoSProfile(depth=50)

        # Load configuration
        cfg = {}
        try:
            share = get_package_share_directory("eva")
            cfg_path = os.path.join(share, "config", "configs.yaml")
            with open(cfg_path, "r") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception as e:
            self.get_logger().warn(f"Failed to load configs.yaml: {e}")

        # Determine arm from node name
        name = self.get_name().lower()
        if name.endswith("_l") or "left" in name:
            self.arm = "l"
        elif name.endswith("_r") or "right" in name:
            self.arm = "r"
        else:
            self.arm = "r"

        # Topic namespaces
        joint_topic_prefix_in = cfg.get("topic_prefix_out", "eva_ik").rstrip("/")
        gripper_topic_prefix_in = cfg.get("topic_prefix_in", "eva_ik").rstrip("/")
        self.joint_prefix = joint_topic_prefix_in
        self.gripper_prefix = gripper_topic_prefix_in

        # Timing
        self.ts_offset = float(cfg.get("ts_offset", 0.4))
        self.only_when_engaged = False

        # Interface selection: default mapping per user is left->can2, right->can1
        interfaces_cfg = cfg.get("interfaces", {})
        default_iface = "can1" if self.arm == "l" else "can2"
        selected_interface = interfaces_cfg.get(
            "left" if self.arm == "l" else "right", default_iface
        )

        # Robot configuration
        model = cfg.get("model", "X5")
        robot_urdf = cfg.get("urdf", None)

        self.get_logger().info("Initializing ARX5 controller...")
        try:
            self.robot_config = arx5.RobotConfigFactory.get_instance().get_config(model)
            if robot_urdf:
                self.robot_config.urdf_path = robot_urdf
            # Match X5A URDF link names
            self.robot_config.base_link_name = "base_link"
            self.robot_config.eef_link_name = "link6"

            self.controller_config = (
                arx5.ControllerConfigFactory.get_instance().get_config(
                    "joint_controller", self.robot_config.joint_dof
                )
            )

            self.controller = Arx5JointController(
                self.robot_config, self.controller_config, selected_interface
            )
        except Exception as e:
            self.get_logger().fatal(f"Failed to initialize ARX5 controller: {e}")
            raise

        # Gains and initial home move
        try:
            gain: Gain = self.controller.get_gain()
            kp = np.array(
                [6.225, 17.225, 18.225, 12.225, 8.225, 6.225], dtype=np.float64
            )
            kd = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0], dtype=np.float64)
            gain.kp()[:] = kp
            gain.kd()[:] = kd
            gain.gripper_kp = 1.0
            gain.gripper_kd = 0.1
            cur_joint = self.controller.get_joint_state()
            cur_joint.timestamp = cur_joint.timestamp + 0.1
            self.controller.set_joint_cmd(cur_joint)
            self.controller.set_gain(gain)
        except Exception as e:
            self.get_logger().warn(f"Failed to set gains/home pose: {e}")

        # Publisher for current robot joint state feedback
        self.pub_eva_joint_state = self.create_publisher(
            JointState, f"eva/{self.arm}/joints", qos
        )

        # Subscriptions (from IK)
        self.create_subscription(
            JointState,
            f"/{self.joint_prefix}/{self.arm}/joint_state",
            self._on_joint_state,
            qos,
        )
        self.create_subscription(
            Bool,
            f"/{self.joint_prefix}/{self.arm}/engaged",
            self._on_engaged,
            qos,
        )

        self.engaged: bool = True  # force-enable streaming by default
        self._dbg_every = 10
        self._step = 0

        self.get_logger().info(
            f"Subscribed joints from '{self.joint_prefix}' and gripper from '{self.gripper_prefix}'."
        )

    def _on_engaged(self, msg: Bool) -> None:
        self.engaged = bool(msg.data)

    def _on_joint_state(self, msg: JointState) -> None:
        if self.only_when_engaged and not self.engaged:
            return
        if not msg.position or len(msg.position) < 6:
            return

        full_positions = np.array(list(msg.position), dtype=np.float64)
        # Extract first 6 joint angles for arm
        desired_position = full_positions[:6]
        # Gripper command comes from index 6 of IK JointState
        if full_positions.shape[0] >= 7:
            self.gripper_cmd = float(full_positions[6])
        else:
            self.gripper_cmd = float(getattr(self, "gripper_cmd", 0.0))

        # Compose and send joint+gripper command (gripper fields optional)
        zero_vec = np.zeros_like(desired_position) + 0.1
        cur_joint_state = self.controller.get_joint_state()
        current_ts = getattr(cur_joint_state, "timestamp", 0.0)
        ts = current_ts + self.ts_offset

        requested = ArxJointState(
            desired_position.astype(np.float32),
            zero_vec.astype(np.float32),
            zero_vec.astype(np.float32),
            float(ts),
        )
        # Apply hardware gripper mapping consistent with working node
        requested.gripper_pos = float(self.gripper_cmd) - 0.018
        requested.gripper_vel = 0.2
        requested.gripper_torque = 0.1

        try:
            self.controller.set_joint_cmd(requested)
        except Exception as e:
            self.get_logger().error(f"Failed to set joint cmd: {e}")
            return

        # Publish feedback
        try:
            cur = self.controller.get_joint_state()
            joints = cur.pos()
            gripper = getattr(cur, "gripper_pos", 0.0)
            js = JointState()
            js.header.stamp = self.get_clock().now().to_msg()
            js.name = [
                "joint_1",
                "joint_2",
                "joint_3",
                "joint_4",
                "joint_5",
                "joint_6",
                "gripper",
            ]
            js.position = [
                float(joints[0]),
                float(joints[1]),
                float(joints[2]),
                float(joints[3]),
                float(joints[4]),
                float(joints[5]),
                float(gripper),
            ]
            self.pub_eva_joint_state.publish(js)
        except Exception as e:
            self.get_logger().warn(f"Failed to publish feedback: {e}")


def main(argv=None) -> None:
    rclpy.init()
    node = IkToJointsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
