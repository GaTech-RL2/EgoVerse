#!/usr/bin/env python3

import argparse
import os
import time
from typing import Dict, Optional

import numpy as np
import h5py

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, Int8, Float32

from cv_bridge import CvBridge
import yaml


class LatestBuffer:
    def __init__(self):
        self.msg = None
        self.stamp = 0.0

    def set(self, msg, stamp):
        self.msg = msg
        self.stamp = stamp

    def get(self):
        return self.msg, self.stamp


class DataCollector(Node):
    def __init__(self, cfg: Dict, fps: float, max_timesteps: int, dataset_path: str, wait_for_save_demo: bool) -> None:
        super().__init__("eva_clean_recorder")
        qos = QoSProfile(depth=50)
        self.bridge = CvBridge()

        self.fps = float(fps)
        self.period = 1.0 / max(self.fps, 1e-6)
        self.max_timesteps = int(max_timesteps)
        self.dataset_path = dataset_path
        self.wait_for_save_demo = bool(wait_for_save_demo)

        # Topic prefixes
        self.vr_prefix = cfg.get("vr_prefix", "/vr").rstrip("/")
        self.ik_prefix = cfg.get("ik_prefix", "/eva_ik").rstrip("/")

        cams = cfg.get("cameras", {})
        self.cams_cfg = cams

        # Buffers
        self.buffers: Dict[str, LatestBuffer] = {}

        def b(key):
            self.buffers[key] = LatestBuffer()

        # Recorder now only tracks robot action (from IK) and robot executed joints.
        b("robot_action")
        b("robot_joints")

        # Cameras
        for cam_name, cam in self.cams_cfg.items():
            if bool(cam.get("enabled", True)):
                b(f"cam_{cam_name}")

        # Subscriptions
        self._create_subscriptions(qos, cfg)

        # Sampling timer
        self.step = 0
        self.samples = []  # will flush to HDF5 progressively
        self.timer = self.create_timer(self.period, self._on_timer)
        self.started = not self.wait_for_save_demo

        self.get_logger().info(
            f"Recording to {self.dataset_path} at {self.fps:.1f} FPS for up to {self.max_timesteps} steps"
        )

    # ----- Subscriptions -----
    def _create_subscriptions(self, qos: QoSProfile, cfg: Dict) -> None:
        vr = self.vr_prefix
        ik = self.ik_prefix

        def clk():
            return self.get_clock().now().nanoseconds / 1e9

        # Helpers to set buffers
        def set_buf(name, msg):
            self.buffers[name].set(msg, clk())

        # Robot action (IK-produced desired joints) and executed joints from robot node
        self.create_subscription(JointState, "/eva_clean/robot_action", lambda m: set_buf("robot_action", m), qos)
        self.create_subscription(JointState, "/eva/r/joints", lambda m: set_buf("robot_joints", m), qos)

        # Cameras
        for cam_name, cam in self.cams_cfg.items():
            if not bool(cam.get("enabled", True)):
                continue
            topic = cam.get("topic")
            self.create_subscription(Image, topic, lambda m, n=cam_name: set_buf(f"cam_{n}", m), qos)

    # ----- Timer -----
    def _on_timer(self) -> None:
        if self.step >= self.max_timesteps:
            self.get_logger().info("Max timesteps reached; stopping.")
            rclpy.shutdown()
            return

        # Start immediately (VR gating removed) unless wait_for_save_demo is used; placeholder no-op
        if not self.started:
            self.started = True
            # No gating; begin recording

        t_now = time.time()

        sample = {
            "t_wall": t_now,
        }

        # Robot action and executed joints
        act_msg, _ = self.buffers["robot_action"].get()
        if isinstance(act_msg, JointState) and act_msg.position:
            vals = list(act_msg.position)
            if len(vals) < 7:
                vals = vals + [0.0] * (7 - len(vals))
            sample["robot_action"] = np.array(vals[:7], dtype=np.float64)
        else:
            sample["robot_action"] = None

        qpos_msg, _ = self.buffers["robot_joints"].get()
        if isinstance(qpos_msg, JointState) and qpos_msg.position:
            vals = list(qpos_msg.position)
            if len(vals) < 7:
                vals = vals + [0.0] * (7 - len(vals))
            sample["robot_qpos"] = np.array(vals[:7], dtype=np.float64)
        else:
            sample["robot_qpos"] = None

        # Cameras -> store raw latest frame
        for cam_name, cam in self.cams_cfg.items():
            if not bool(cam.get("enabled", True)):
                continue
            img_msg, _ = self.buffers.get(f"cam_{cam_name}", LatestBuffer()).get()
            if isinstance(img_msg, Image):
                try:
                    cv_img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding=cam.get("encoding", "rgb8"))
                except Exception:
                    cv_img = None
                sample[f"cam_{cam_name}"] = cv_img
            else:
                sample[f"cam_{cam_name}"] = None

        self.samples.append(sample)
        self.step += 1

        if self.step == self.max_timesteps:
            self.get_logger().info("Reached max timesteps. Writing HDF5...")
            self._write_hdf5()
            rclpy.shutdown()

    # ----- HDF5 write -----
    def _write_hdf5(self) -> None:
        T = len(self.samples)
        cams = {k: v for k, v in self.cams_cfg.items() if bool(v.get("enabled", True))}

        # Prepare arrays with None -> zeros
        def arr_or_default(key, shape, dtype):
            out = np.zeros(shape, dtype=dtype)
            for t, s in enumerate(self.samples):
                v = s.get(key, None)
                if v is None:
                    continue
                try:
                    out[t] = v
                except Exception:
                    pass
            return out

        os.makedirs(os.path.dirname(self.dataset_path), exist_ok=True)
        with h5py.File(self.dataset_path + ".hdf5", "w") as f:
            f.attrs["fps"] = float(self.fps)
            f.attrs["num_steps"] = int(T)

            # timestamps
            ts = np.array([s["t_wall"] for s in self.samples], dtype=np.float64)
            f.create_dataset("/timestamps/now", data=ts)

            grp_obs = f.create_group("observations")
            # Robot action and executed joints
            grp_obs.create_dataset("robot_action", data=arr_or_default("robot_action", (T, 7), np.float64))
            grp_obs.create_dataset("robot_qpos", data=arr_or_default("robot_qpos", (T, 7), np.float64))

            # Images
            grp_images = grp_obs.create_group("images")
            for cam_name, cam in cams.items():
                H = int(cam.get("height", 480))
                W = int(cam.get("width", 640))
                ds = grp_images.create_dataset(cam_name, (T, H, W, 3), dtype=np.uint8, chunks=(1, H, W, 3))
                for t, s in enumerate(self.samples):
                    img = s.get(f"cam_{cam_name}")
                    if img is None:
                        continue
                    try:
                        # Resize if unexpected size
                        if img.shape[0] != H or img.shape[1] != W:
                            import cv2
                            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
                        if img.shape[2] == 4:
                            img = img[:, :, :3]
                        ds[t] = img.astype(np.uint8)
                    except Exception:
                        pass


def main(argv=None):
    parser = argparse.ArgumentParser(description="Eva Clean data recorder (ROS2)")
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--config", type=str, default="/home/rl2-bonjour/Eva_Clean/config/topics.yaml")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-timesteps", type=int, default=3000)
    parser.add_argument("--wait-for-save-demo", action="store_true")

    args = parser.parse_args(argv)

    dataset_path = os.path.join(args.dataset_dir, args.dataset_name)
    with open(args.config, "r") as fh:
        cfg = yaml.safe_load(fh) or {}

    rclpy.init()
    node = DataCollector(cfg, args.fps, args.max_timesteps, dataset_path, args.wait_for_save_demo)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node._write_hdf5()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()


