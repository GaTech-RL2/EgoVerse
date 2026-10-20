#!/usr/bin/env python3
"""ROS 2 deployment node for the real_mug v2 bimanual policy (FR3 + ORCA hands, cvg-dex-teleop rig).

Thin successor of egoverse/model_inference_node.py: all model conventions live in
egomimic.robot.mug2_policy (tool-frame flip, per-arm base frames, key names, normalization,
chunk resampling); this node only moves ROS messages in and out.

Topics (parameters, defaults mirror the old node):
  in : <camera_topics[i]> (sensor_msgs/Image, 3 cameras in the order of camera_names =
       ["front_img_1", "left_wrist_img", "right_wrist_img"]),
       /{left,right}/arm/end_effector_pose (PoseStamped, TCP in that arm's base frame),
       /{left,right}/hand/joint_state (Float32MultiArray, 17 measured joints; falls back to the
       last published hand command if the topic never arrives),
       /model_inference_command (String: start | stop | reset | quit)
  out: /{left,right}/arm/end_effector_pose_cmd (PoseStamped, frame_id panda_link0 of that arm),
       /{left,right}/hand/policy_output (Float32MultiArray, 17 targets)

Timing: the policy is queried every `query_period_s` (default 0.5 s, i.e. receding horizon) and
the latest 1.5 s chunk, resampled to `control_hz` (default 30), is streamed step by step.
"""
import os
import sys
import time
from copy import deepcopy
from threading import Lock

import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from std_msgs.msg import String as ROSString

from egomimic.robot.mug2_policy import HAND_DIM, Mug2Observation, Mug2Policy

QOS = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
CBG = ReentrantCallbackGroup()
SIDES = ("left", "right")


def to_f32_multiarray(a):
    msg = Float32MultiArray(); a = np.asarray(a, dtype=np.float32).ravel()
    msg.layout.dim.append(MultiArrayDimension(label="x", size=len(a), stride=len(a))); msg.data = a.tolist(); return msg


def pose_msg(xyz_quat, frame_id):
    m = PoseStamped(); m.header.frame_id = frame_id
    m.pose.position.x, m.pose.position.y, m.pose.position.z = map(float, xyz_quat[:3])
    m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z, m.pose.orientation.w = map(float, xyz_quat[3:7]); return m


class Mug2InferenceNode(Node):
    def __init__(self):
        super().__init__("mug2_policy")
        p = self.declare_parameter
        p("policy_ckpt_path", ""); p("camera_topics", rclpy.Parameter.Type.STRING_ARRAY)
        p("camera_names", ["front_img_1", "left_wrist_img", "right_wrist_img"])
        p("control_hz", 30.0); p("query_period_s", 0.5); p("num_inference_steps", 50)
        p("hand_state_topic", "/{side}/hand/joint_state"); p("arm_pose_topic", "/{side}/arm/end_effector_pose")
        p("arm_cmd_topic", "/{side}/arm/end_effector_pose_cmd"); p("hand_cmd_topic", "/{side}/hand/policy_output")
        p("arm_frame_id", "panda_link0"); p("log_dir", "")
        g = lambda k: self.get_parameter(k).value
        self.control_hz = float(g("control_hz")); self.query_period = float(g("query_period_s"))
        self.bridge = CvBridge(); self.lock = Lock()
        self.images = {}; self.ee = {s: None for s in SIDES}; self.hand = {s: None for s in SIDES}; self.last_hand_cmd = {s: None for s in SIDES}

        names = list(g("camera_names")); topics = list(g("camera_topics"))
        assert len(names) == 3 and len(topics) == 3, "need exactly 3 cameras: front_img_1, left_wrist_img, right_wrist_img"
        for name, topic in zip(names, topics):
            self.create_subscription(Image, topic, lambda msg, n=name: self._img_cb(n, msg), QOS, callback_group=CBG)
        self.arm_pub, self.hand_pub = {}, {}
        for s in SIDES:
            self.create_subscription(PoseStamped, g("arm_pose_topic").format(side=s), lambda msg, side=s: self._pose_cb(side, msg), QOS, callback_group=CBG)
            self.create_subscription(Float32MultiArray, g("hand_state_topic").format(side=s), lambda msg, side=s: self._hand_cb(side, msg), QOS, callback_group=CBG)
            self.arm_pub[s] = self.create_publisher(PoseStamped, g("arm_cmd_topic").format(side=s), QOS)
            self.hand_pub[s] = self.create_publisher(Float32MultiArray, g("hand_cmd_topic").format(side=s), QOS)
        self.create_subscription(ROSString, "/model_inference_command", self._cmd_cb, QOS, callback_group=CBG)
        self.frame_id = g("arm_frame_id")

        self.get_logger().info(f"loading policy {g('policy_ckpt_path')}")
        self.policy = Mug2Policy(g("policy_ckpt_path"), num_inference_steps=int(g("num_inference_steps")))
        self.get_logger().info("policy loaded")

        self.running = False; self.chunk = None; self.chunk_idx = 0; self.last_query = 0.0
        self.create_timer(1.0 / self.control_hz, self._control_cb, callback_group=CBG)

    # ---- inputs
    def _img_cb(self, name, msg):
        with self.lock: self.images[name] = msg
    def _pose_cb(self, side, msg):
        q = msg.pose
        with self.lock: self.ee[side] = np.array([q.position.x, q.position.y, q.position.z, q.orientation.x, q.orientation.y, q.orientation.z, q.orientation.w])
    def _hand_cb(self, side, msg):
        with self.lock: self.hand[side] = np.asarray(msg.data, dtype=float)[:HAND_DIM]
    def _cmd_cb(self, msg):
        c = msg.data.strip().lower()
        if c == "start": self.running = True; self.chunk = None; self.get_logger().info("policy started")
        elif c == "stop": self.running = False; self.get_logger().info("policy stopped")
        elif c == "reset": self.running = False; self.chunk = None; self.get_logger().info("reset (chunk cleared)")
        elif c == "quit": rclpy.shutdown(); sys.exit(0)

    def _observation(self):
        with self.lock:
            imgs = {k: deepcopy(v) for k, v in self.images.items()}; ee = dict(self.ee); hand = dict(self.hand)
        if len(imgs) < 3 or any(ee[s] is None for s in SIDES):
            return None
        for s in SIDES:   # hand proprio: measured joints if published, else the last command we sent
            if hand[s] is None:
                hand[s] = self.last_hand_cmd[s] if self.last_hand_cmd[s] is not None else np.zeros(HAND_DIM)
        rgb = {k: self.bridge.imgmsg_to_cv2(v, "rgb8") for k, v in imgs.items()}
        return Mug2Observation(rgb["front_img_1"], rgb["left_wrist_img"], rgb["right_wrist_img"], ee["left"], ee["right"], hand["left"], hand["right"])

    # ---- control loop
    def _control_cb(self):
        if not self.running:
            return
        now = time.time()
        if self.chunk is None or now - self.last_query >= self.query_period or self.chunk_idx >= len(self.chunk.ee_left):
            obs = self._observation()
            if obs is None:
                self.get_logger().warn("waiting for cameras / arm poses", throttle_duration_sec=2.0); return
            t0 = time.time(); self.chunk = self.policy.predict(obs).at_rate(self.control_hz); self.chunk_idx = 0; self.last_query = now
            self.get_logger().info(f"policy query {1000 * (time.time() - t0):.0f} ms", throttle_duration_sec=5.0)
        i = self.chunk_idx; self.chunk_idx += 1
        stamp = self.get_clock().now().to_msg()
        for s in SIDES:
            m = pose_msg(getattr(self.chunk, f"ee_{s}")[i], self.frame_id); m.header.stamp = stamp; self.arm_pub[s].publish(m)
            h = getattr(self.chunk, f"hand_{s}")[i]; self.hand_pub[s].publish(to_f32_multiarray(h)); self.last_hand_cmd[s] = h


def main(args=None):
    rclpy.init(args=args)
    node = Mug2InferenceNode()
    ex = rclpy.executors.MultiThreadedExecutor(num_threads=4); ex.add_node(node)
    try:
        ex.spin()
    finally:
        node.destroy_node(); rclpy.shutdown()


if __name__ == "__main__":
    main()
