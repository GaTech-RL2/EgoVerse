#!/usr/bin/env python3
"""
DWM Streaming Client for YAM Robot.

Standalone client that runs on Jetson Orin to:
- Collect observations from YAM robot and cameras
- Stream observations to desktop inference server  
- Receive actions and execute on robot

Minimal dependencies - no egomimic required.

Usage:
    python dwm_yam_client.py \
        --server tcp://192.168.1.100:5555 \
        --arms both \
        --query-frequency 16 \
        --frequency 30
"""

import os
import sys
import time
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R

# Add i2rt and robot interface paths
sys.path.insert(0, os.path.expanduser("~/i2rt"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "eva/eva_ws/src/eva"))

import zmq
import msgpack
import msgpack_numpy
msgpack_numpy.patch()

from robot_interface import YAMInterface
from i2rt.robots.utils import GripperType


# =============================================================================
# Camera Transforms (standalone, no egomimic dependency)
# =============================================================================

# Extrinsics for different camera configurations
# Format: 4x4 transformation matrix from camera frame to base frame
EXTRINSICS = {
    "x5Dec13_2": {
        "left": np.array([
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float64),
        "right": np.array([
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float64),
    },
    # Add more extrinsic configs as needed
}


def cam_to_base_frame(actions_cam: np.ndarray, T_cam_base: np.ndarray) -> np.ndarray:
    """Transform actions from camera frame to robot base frame.
    
    Args:
        actions_cam: (T, 6) array of [x, y, z, yaw, pitch, roll] in camera frame
        T_cam_base: 4x4 transformation matrix from camera to base
        
    Returns:
        actions_base: (T, 6) array of [x, y, z, yaw, pitch, roll] in base frame
    """
    T = actions_cam.shape[0]
    
    # Build SE(3) matrices for each timestep
    se3 = np.zeros((T, 4, 4), dtype=np.float64)
    se3[:, :3, 3] = actions_cam[:, :3]  # xyz
    
    # Convert ypr to rotation matrices
    for i in range(T):
        ypr = actions_cam[i, 3:6]
        se3[i, :3, :3] = R.from_euler('ZYX', ypr).as_matrix()
    se3[:, 3, 3] = 1.0
    
    # Transform to base frame
    base_frame = T_cam_base @ se3
    
    # Extract xyz and ypr
    xyz = base_frame[:, :3, 3]
    ypr = R.from_matrix(base_frame[:, :3, :3]).as_euler('ZYX')
    
    return np.concatenate([xyz, ypr], axis=1)


# =============================================================================
# DWM Streaming Client
# =============================================================================

class DWMYAMClient:
    """Streaming client for YAM robot with DWM inference server."""
    
    def __init__(
        self,
        server_addr: str,
        arms: str,
        query_freq: int,
        extrinsics_key: str,
        timeout_ms: int,
        gripper_type: str,
        can_interfaces: dict,
        dry_run: bool,
    ):
        """Initialize YAM client.
        
        Args:
            server_addr: ZMQ server address (e.g., "tcp://192.168.1.100:5555")
            arms: Which arm(s) to control ("left", "right", or "both")
            query_freq: How often to request new actions (in timesteps)
            extrinsics_key: Key for camera extrinsics
            timeout_ms: Request timeout in milliseconds
            gripper_type: Gripper type string
            can_interfaces: CAN interface mapping {"left": "can0", "right": "can1"}
            dry_run: If True, simulate robot without actual hardware
        """
        self.server_addr = server_addr
        self.arms = arms
        self.query_freq = query_freq
        self.timeout_ms = timeout_ms
        self.dry_run = dry_run
        
        # Setup extrinsics
        if extrinsics_key not in EXTRINSICS:
            raise ValueError(f"Unknown extrinsics_key: {extrinsics_key}")
        self.extrinsics = EXTRINSICS[extrinsics_key]
        
        # Determine arm list
        if arms == "both":
            self.arms_list = ["left", "right"]
        else:
            self.arms_list = [arms]
        
        # Initialize robot interface
        print(f"Initializing YAM robot interface...")
        gripper_enum = GripperType.from_string_name(gripper_type)
        self.robot = YAMInterface(
            arms=self.arms_list,
            gripper_type=gripper_enum,
            interfaces=can_interfaces,
            zero_gravity_mode=False,
            dry_run=dry_run,
        )
        
        # Setup ZMQ connection
        print(f"Connecting to inference server: {server_addr}")
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.connect(server_addr)
        
        # Test connection
        self._ping()
        
        # Action buffer
        self.actions = None
        
    def _ping(self):
        """Test server connection."""
        self.sock.send(msgpack.packb({'cmd': 'ping'}, use_bin_type=True))
        reply = msgpack.unpackb(self.sock.recv(), raw=False)
        if reply.get('status') != 'ok':
            raise RuntimeError(f"Server ping failed: {reply}")
        print("Server connection verified.")
    
    def _request_actions(self, obs: dict, step: int) -> np.ndarray:
        """Request inference from server.
        
        Args:
            obs: Observation dictionary with images and joint_positions
            step: Current timestep
            
        Returns:
            actions: (T, 14) array in ypr format [left_xyz_ypr_grip, right_xyz_ypr_grip]
        """
        # Prepare observation data
        obs_data = {}
        for key, val in obs.items():
            if isinstance(val, np.ndarray):
                obs_data[key] = val
            elif val is not None:
                obs_data[key] = np.asarray(val)
        
        msg = {'cmd': 'infer', 'obs': obs_data, 'step': step}
        
        t0 = time.time()
        self.sock.send(msgpack.packb(msg, use_bin_type=True))
        reply = msgpack.unpackb(self.sock.recv(), raw=False)
        rtt = time.time() - t0
        
        if 'error' in reply:
            raise RuntimeError(f"Server error: {reply['error']}")
        
        actions = reply['actions']
        infer_t = reply.get('infer_time', 0)
        print(f"Step {step}: roundtrip {rtt*1000:.1f}ms (inference {infer_t*1000:.1f}ms)")
        
        return actions
    
    def _transform_actions(self, actions_ypr: np.ndarray) -> np.ndarray:
        """Transform actions from camera frame to base frame.
        
        Args:
            actions_ypr: (T, 14) array with [left_7D, right_7D] in ypr format
            
        Returns:
            transformed: (T, 14) array in base frame
        """
        if self.arms == "both":
            left = actions_ypr[:, :7]
            right = actions_ypr[:, 7:14]
            
            left_6d = cam_to_base_frame(left[:, :6], self.extrinsics["left"])
            right_6d = cam_to_base_frame(right[:, :6], self.extrinsics["right"])
            
            left_out = np.hstack([left_6d, left[:, 6:7]])
            right_out = np.hstack([right_6d, right[:, 6:7]])
            
            return np.hstack([left_out, right_out])
        else:
            arm_offset = 7 if self.arms == "right" else 0
            arm_actions = actions_ypr[:, arm_offset:arm_offset + 7]
            
            transformed_6d = cam_to_base_frame(
                arm_actions[:, :6],
                self.extrinsics[self.arms]
            )
            
            return np.hstack([transformed_6d, arm_actions[:, 6:7]])
    
    def step(self, i: int) -> bool:
        """Execute one control step.
        
        Args:
            i: Current timestep
            
        Returns:
            True if step succeeded, False to terminate
        """
        # Get observations
        obs = self.robot.get_obs()
        
        # Request new actions at query frequency
        if i % self.query_freq == 0:
            actions_ypr = self._request_actions(obs, i)
            self.actions = self._transform_actions(actions_ypr)
        
        if self.actions is None:
            return False
        
        # Get action at current index
        act_idx = i % self.query_freq
        if act_idx >= self.actions.shape[0]:
            act_idx = self.actions.shape[0] - 1
        
        action = self.actions[act_idx]
        
        # Execute on robot
        for arm in self.arms_list:
            arm_offset = 7 if arm == "right" and self.arms == "both" else 0
            if self.arms != "both" and arm == "right":
                arm_offset = 0
            arm_action = action[arm_offset:arm_offset + 7]
            self.robot.set_pose(arm_action, arm)
        
        return True
    
    def reset(self):
        """Reset robot to home position."""
        print("Resetting to home position...")
        self.robot.set_home()
        self.actions = None
    
    def close(self):
        """Clean up resources."""
        self.sock.close()
        self.ctx.term()
        print("Client closed.")


# =============================================================================
# Rate Control
# =============================================================================

class RateController:
    """Simple rate limiter for control loop."""
    
    def __init__(self, freq_hz: float):
        self.period = 1.0 / freq_hz
        self.last_t = None
        
    def sleep(self):
        """Sleep to maintain desired rate."""
        now = time.time()
        if self.last_t is not None:
            elapsed = now - self.last_t
            if elapsed < self.period:
                time.sleep(self.period - elapsed)
        self.last_t = time.time()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="DWM Streaming Client for YAM Robot")
    
    # Server connection
    parser.add_argument("--server", type=str, required=True,
                        help="Inference server address (e.g., tcp://192.168.1.100:5555)")
    
    # Robot configuration
    parser.add_argument("--arms", type=str, default="both",
                        choices=["left", "right", "both"],
                        help="Which arm(s) to control")
    parser.add_argument("--gripper-type", type=str, default="linear_4310",
                        choices=["crank_4310", "linear_3507", "linear_4310", "yam_teaching_handle", "no_gripper"],
                        help="YAM gripper type")
    parser.add_argument("--left-can", type=str, default="can0",
                        help="CAN interface for left arm")
    parser.add_argument("--right-can", type=str, default="can1",
                        help="CAN interface for right arm")
    
    # Control parameters
    parser.add_argument("--frequency", type=float, default=30.0,
                        help="Control loop frequency in Hz")
    parser.add_argument("--query-frequency", type=int, default=16,
                        help="How often to request new actions (in timesteps)")
    parser.add_argument("--timeout-ms", type=int, default=5000,
                        help="Server request timeout in milliseconds")
    
    # Camera configuration  
    parser.add_argument("--extrinsics", type=str, default="x5Dec13_2",
                        help="Camera extrinsics configuration key")
    
    # Debug
    parser.add_argument("--dry-run", action="store_true",
                        help="Simulate robot without actual hardware")
    
    args = parser.parse_args()
    
    # Print configuration
    print("=" * 60)
    print("DWM YAM Client Configuration")
    print("=" * 60)
    print(f"Server:          {args.server}")
    print(f"Arms:            {args.arms}")
    print(f"Gripper type:    {args.gripper_type}")
    print(f"Frequency:       {args.frequency} Hz")
    print(f"Query frequency: {args.query_frequency}")
    print(f"Extrinsics:      {args.extrinsics}")
    print(f"Dry run:         {args.dry_run}")
    print("=" * 60)
    
    # Initialize client
    can_interfaces = {"left": args.left_can, "right": args.right_can}
    
    client = DWMYAMClient(
        server_addr=args.server,
        arms=args.arms,
        query_freq=args.query_frequency,
        extrinsics_key=args.extrinsics,
        timeout_ms=args.timeout_ms,
        gripper_type=args.gripper_type,
        can_interfaces=can_interfaces,
        dry_run=args.dry_run,
    )
    
    # Initialize rate controller
    rate = RateController(args.frequency)
    
    # Reset to home
    client.reset()
    print("\nStarting control loop. Press Ctrl+C to stop.\n")
    
    step = 0
    try:
        while True:
            if not client.step(step):
                print("Rollout complete.")
                break
            rate.sleep()
            step += 1
            
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        raise
    finally:
        client.close()
        print("Done.")


if __name__ == "__main__":
    main()
