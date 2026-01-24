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
# Action Space Utilities
# =============================================================================

def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """Convert 6D rotation representation back to rotation matrix.
    
    The 6D representation is the first two columns of the rotation matrix, flattened
    in row-major (C) order. Given a (3, 2) slice of columns, reshape(-1, 6) produces:
        [r00, r01, r10, r11, r20, r21]
    where r_ij is element (row i, col j) of the original rotation matrix.
    
    We recover the third column via cross product and orthonormalize.
    
    Args:
        rot6d: (..., 6) array of 6D rotation representation
        
    Returns:
        rot_matrix: (..., 3, 3) rotation matrix
    """
    # Handle batched input
    original_shape = rot6d.shape[:-1]
    rot6d = rot6d.reshape(-1, 6)
    
    # The 6D representation was created by taking the first two columns of the
    # rotation matrix (shape 3x2) and flattening in row-major order:
    #   [[r00, r01], [r10, r11], [r20, r21]] -> [r00, r01, r10, r11, r20, r21]
    # So we reshape back to (N, 3, 2) to correctly extract the columns.
    rot6d_reshaped = rot6d.reshape(-1, 3, 2)
    col1 = rot6d_reshaped[:, :, 0]  # (N, 3) - first column [r00, r10, r20]
    col2 = rot6d_reshaped[:, :, 1]  # (N, 3) - second column [r01, r11, r21]
    
    # Normalize first column
    col1 = col1 / (np.linalg.norm(col1, axis=1, keepdims=True) + 1e-8)
    
    # Make second column orthogonal to first and normalize
    col2 = col2 - np.sum(col1 * col2, axis=1, keepdims=True) * col1
    col2 = col2 / (np.linalg.norm(col2, axis=1, keepdims=True) + 1e-8)
    
    # Third column is cross product
    col3 = np.cross(col1, col2)
    
    # Assemble rotation matrix
    rot_matrix = np.stack([col1, col2, col3], axis=-1)  # (N, 3, 3)
    
    # Reshape back to original batch shape
    rot_matrix = rot_matrix.reshape(*original_shape, 3, 3)
    
    return rot_matrix


def convert_6drot_to_ypr(actions_6drot: np.ndarray) -> np.ndarray:
    """Convert actions from 6D rotation format to YPR format.
    
    6D rotation format (20D total):
        [left_xyz(3), left_rot6d(6), left_grip(1), right_xyz(3), right_rot6d(6), right_grip(1)]
        
    YPR format (14D total):
        [left_xyz(3), left_ypr(3), left_grip(1), right_xyz(3), right_ypr(3), right_grip(1)]
    
    Args:
        actions_6drot: (T, 20) array in 6D rotation format
        
    Returns:
        actions_ypr: (T, 14) array in YPR format
    """
    T = actions_6drot.shape[0]
    
    # Extract components for left arm
    left_xyz = actions_6drot[:, 0:3]
    left_rot6d = actions_6drot[:, 3:9]
    left_grip = actions_6drot[:, 9:10]
    
    # Extract components for right arm
    right_xyz = actions_6drot[:, 10:13]
    right_rot6d = actions_6drot[:, 13:19]
    right_grip = actions_6drot[:, 19:20]
    
    # Convert rot6d to rotation matrices
    left_rot_mat = rot6d_to_matrix(left_rot6d)   # (T, 3, 3)
    right_rot_mat = rot6d_to_matrix(right_rot6d)  # (T, 3, 3)
    
    # Convert rotation matrices to YPR (ZYX Euler angles)
    left_ypr = R.from_matrix(left_rot_mat).as_euler('ZYX')   # (T, 3)
    right_ypr = R.from_matrix(right_rot_mat).as_euler('ZYX')  # (T, 3)
    
    # Assemble YPR format: [left_xyz, left_ypr, left_grip, right_xyz, right_ypr, right_grip]
    actions_ypr = np.hstack([
        left_xyz, left_ypr, left_grip,
        right_xyz, right_ypr, right_grip,
    ])
    
    return actions_ypr


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
        timeout_ms: int,
        gripper_type: str,
        can_interfaces: dict,
        dry_run: bool,
        velocity_limit: float = None,
        wait_for_server: bool = False,
        zero_action_threshold: float = None,
        action_horizon: int = None,
    ):
        """Initialize YAM client.
        
        Args:
            server_addr: ZMQ server address (e.g., "tcp://192.168.1.100:5555")
            arms: Which arm(s) to control ("left", "right", or "both")
            query_freq: How often to request new actions (in timesteps)
            extrinsics_key: Key for camera extrinsics
            apply_extrinsics: Whether to transform actions from camera to base frame
            timeout_ms: Request timeout in milliseconds
            gripper_type: Gripper type string
            can_interfaces: CAN interface mapping {"left": "can0", "right": "can1"}
            dry_run: If True, connect to real robot for proprioception but do NOT execute actions.
                     Robot will be homed and observations will be real, but set_pose() is skipped.
            velocity_limit: Maximum joint velocity in rad/s. If None, no velocity limit is applied.
            wait_for_server: If True, wait indefinitely for the server to become available.
            zero_action_threshold: If set, replace near-zero actions (xyz norm < threshold) with
                                   current pose. Workaround for models trained with zeros for 
                                   non-engaged arms.
            action_horizon: If set, only execute the first H actions from each action chunk.
                           If None, execute all actions in the chunk.
        """
        self.server_addr = server_addr
        self.arms = arms
        self.query_freq = query_freq
        self.timeout_ms = timeout_ms
        self.dry_run = dry_run
        self.velocity_limit = velocity_limit
        self.wait_for_server = wait_for_server
        self.zero_action_threshold = zero_action_threshold
        self.action_horizon = action_horizon
        self._zero_arms = {}
        
        # Determine arm list
        if arms == "both":
            self.arms_list = ["left", "right"]
        else:
            self.arms_list = [arms]
        
        # Setup ZMQ connection FIRST (before robot to avoid motor timeout during connection)
        print(f"Connecting to inference server: {server_addr}")
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.connect(server_addr)
        
        # Test connection (with retry if wait_for_server is enabled)
        self._connect_with_retry()
        
        # Initialize robot interface AFTER ZMQ is ready
        # Note: We always connect to real robot (dry_run=False in YAMInterface)
        # The client's dry_run flag only controls whether we execute actions
        print(f"Initializing YAM robot interface...")
        gripper_enum = GripperType.from_string_name(gripper_type)
        self.robot = YAMInterface(
            arms=self.arms_list,
            gripper_type=gripper_enum,
            interfaces=can_interfaces,
            zero_gravity_mode=False,
            dry_run=False,  # Always use real robot for proprioception
        )
        
        if dry_run:
            print("[DWMYAMClient] DRY RUN MODE: Real robot connected for proprioception, but actions will NOT be executed")

        time.sleep(2)
        
        # Action buffer
        self.actions = None
        
    def _connect_with_retry(self):
        """Connect to server, optionally waiting indefinitely if wait_for_server is enabled."""
        retry_interval = 2.0  # seconds between retries
        attempt = 0
        
        while True:
            attempt += 1
            try:
                self._ping()
                return  # Success
            except zmq.error.Again:
                # Timeout - server not responding
                if not self.wait_for_server:
                    raise RuntimeError(
                        f"Server not responding at {self.server_addr}. "
                        "Use --wait-for-server to wait indefinitely."
                    )
                print(f"Server not available (attempt {attempt}), retrying in {retry_interval}s...")
                # Reset socket for retry (ZMQ REQ socket can get stuck after timeout)
                self.sock.close()
                self.sock = self.ctx.socket(zmq.REQ)
                self.sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
                self.sock.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
                self.sock.setsockopt(zmq.LINGER, 0)
                self.sock.connect(self.server_addr)
                time.sleep(retry_interval)
            except Exception as e:
                if not self.wait_for_server:
                    raise
                print(f"Connection error (attempt {attempt}): {e}, retrying in {retry_interval}s...")
                # Reset socket for retry
                self.sock.close()
                self.sock = self.ctx.socket(zmq.REQ)
                self.sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
                self.sock.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
                self.sock.setsockopt(zmq.LINGER, 0)
                self.sock.connect(self.server_addr)
                time.sleep(retry_interval)
    
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
            actions: Raw actions from server (shape depends on action_space)
                     - ypr: (T, 14) [left_xyz_ypr_grip, right_xyz_ypr_grip]
                     - 6drot: (T, 20) [left_xyz_rot6d_grip, right_xyz_rot6d_grip]
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
    
    def _process_actions(self, raw_actions: np.ndarray) -> np.ndarray:
        """Process raw actions from server to YPR format for robot execution.
        
        Auto-detects action space from dimensionality:
            - 14D → YPR format [left_xyz_ypr_grip, right_xyz_ypr_grip]
            - 20D → 6drot format [left_xyz_rot6d_grip, right_xyz_rot6d_grip]
        
        Actions are assumed to already be in robot base frame (no coordinate 
        transformation needed).
        
        If action_horizon is set, only the first H actions are returned.
        
        Args:
            raw_actions: Raw actions from server (T, 14) or (T, 20)
            
        Returns:
            actions_ypr: (H, 14) array in YPR format for robot execution,
                         where H = min(T, action_horizon) if action_horizon is set, else T
        """

        action_dim = raw_actions.shape[1]
        if action_dim == 20:
            actions_ypr = convert_6drot_to_ypr(raw_actions)
        elif action_dim == 14:
            actions_ypr = raw_actions

        # Slice to action_horizon if specified
        if self.action_horizon is not None:
            original_len = actions_ypr.shape[0]
            actions_ypr = actions_ypr[:self.action_horizon]
            print(f"[DWMClient] Sliced actions from {original_len} to {actions_ypr.shape[0]} (action_horizon={self.action_horizon})")
        
        return actions_ypr
    
    def _fix_zero_actions(self, actions: np.ndarray, ee_poses: np.ndarray) -> np.ndarray:
        """Replace near-zero actions with current pose.
        
        Workaround for models trained with zeros for non-engaged arms.
        If an arm's xyz position norm is below the threshold, replace that
        arm's action with the current ee pose.
        
        Also updates self._zero_arms to track which arms are outputting zeros,
        so we can zero out their observations to match training distribution.
        
        Args:
            actions: (T, 14) array of actions in YPR format
            ee_poses: (14,) current end-effector poses [left_7D, right_7D]
            
        Returns:
            Fixed actions array
        """
        if self.zero_action_threshold is None:
            return actions
        
        actions = actions.copy()
        threshold = self.zero_action_threshold
        
        # Check first action to detect zero-action arms
        left_xyz_norm = np.linalg.norm(actions[0, 0:3])
        right_xyz_norm = np.linalg.norm(actions[0, 7:10])
        
        # Track which arms are outputting zeros (for observation zeroing)
        if left_xyz_norm < threshold:
            if not self._zero_arms.get("left", False):
                print(f"[DWMClient] Detected zero-action left arm (xyz_norm={left_xyz_norm:.4f}) - will zero observations")
            self._zero_arms["left"] = True
        
        if right_xyz_norm < threshold:
            if not self._zero_arms.get("right", False):
                print(f"[DWMClient] Detected zero-action right arm (xyz_norm={right_xyz_norm:.4f}) - will zero observations")
            self._zero_arms["right"] = True
        
        # Fix all timesteps
        for t in range(actions.shape[0]):
            # Check left arm (indices 0-6)
            if np.linalg.norm(actions[t, 0:3]) < threshold:
                actions[t, 0:7] = ee_poses[0:7]  # Replace with current left pose
            
            # Check right arm (indices 7-13)
            if np.linalg.norm(actions[t, 7:10]) < threshold:
                actions[t, 7:14] = ee_poses[7:14]  # Replace with current right pose
        
        return actions
    
    def _zero_obs_for_bad_arms(self, obs: dict) -> dict:
        """Zero out ee_poses for arms that output near-zero actions.
        
        Workaround to match training distribution where non-engaged arms
        had zeros in their observations.
        
        Args:
            obs: Observation dictionary with 'ee_poses' key
            
        Returns:
            Modified observation dictionary
        """
        if self.zero_action_threshold is None:
            return obs
        
        ee_poses = obs.get('ee_poses')
        if ee_poses is None:
            return obs
        
        # Make a copy to avoid modifying the original
        obs = obs.copy()
        ee_poses = ee_poses.copy()
        
        if self._zero_arms.get("left", False):
            ee_poses[0:7] = 0.0  # Zero left arm ee_pose
        
        if self._zero_arms.get("right", False):
            ee_poses[7:14] = 0.0  # Zero right arm ee_pose
        
        obs['ee_poses'] = ee_poses
        return obs
    
    def step(self, i: int) -> bool:
        """Execute one control step.
        
        Args:
            i: Current timestep
            
        Returns:
            True if step succeeded, False to terminate
        """
        # Get observations
        obs = self.robot.get_obs()
        
        # Store actual ee_poses before any modifications (for action fixing)
        actual_ee_poses = obs.get('ee_poses')
        if actual_ee_poses is not None:
            actual_ee_poses = actual_ee_poses.copy()
        
        # Request new actions at query frequency
        if i % self.query_freq == 0:
            # Print current robot state
            print(f"[DWMClient] Current robot ee_poses: {actual_ee_poses}")
            
            # Zero out observations for arms that output near-zero actions
            # (matches training distribution where non-engaged arms had zeros)
            obs_for_model = self._zero_obs_for_bad_arms(obs)
            if self._zero_arms.get("left") or self._zero_arms.get("right"):
                print(f"[DWMClient] Zeroed obs ee_poses: {obs_for_model.get('ee_poses')}")
            
            raw_actions = self._request_actions(obs_for_model, i)
            print(f"[DWMClient] Raw actions from inference (shape {raw_actions.shape}):")
            print(f"  First action: {raw_actions[0]}")
            
            # Convert to YPR format if needed (auto-detects from dimension)
            self.actions = self._process_actions(raw_actions)
            
            # Fix near-zero actions (replace with actual current pose for safety)
            if self.zero_action_threshold is not None and actual_ee_poses is not None:
                self.actions = self._fix_zero_actions(self.actions, actual_ee_poses)
            
            print(f"[DWMClient] Final actions (shape {self.actions.shape}):")
            print(f"  First action [left_7D, right_7D]: {self.actions[0]}")
        
        if self.actions is None:
            return False
        
        # Get action at current index
        act_idx = i % self.query_freq
        if act_idx >= self.actions.shape[0]:
            act_idx = self.actions.shape[0] - 1
        
        action = self.actions[act_idx]
        
        # Execute on robot (skip if dry_run - only read proprioception)
        if self.dry_run:
            if act_idx == 0:  # Only print once per action batch
                print(f"[DRY RUN] Would execute action: {action}")
        else:
            for arm in self.arms_list:
                arm_offset = 7 if arm == "right" and self.arms == "both" else 0
                if self.arms != "both" and arm == "right":
                    arm_offset = 0
                arm_action = action[arm_offset:arm_offset + 7]
                print(f"[DWMClient] Executing {arm} arm action: xyz=[{arm_action[0]:.4f}, {arm_action[1]:.4f}, {arm_action[2]:.4f}] ypr=[{arm_action[3]:.4f}, {arm_action[4]:.4f}, {arm_action[5]:.4f}] grip={arm_action[6]:.3f}")
                self.robot.set_pose(arm_action, arm, velocity_limit=self.velocity_limit)
        
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
    
    # Debug
    parser.add_argument("--dry-run", action="store_true",
                        help="Connect to real robot for proprioception but do NOT execute actions (robot will be homed)")
    
    # Safety
    parser.add_argument("--velocity-limit", type=float, default=1.0,
                        help="Maximum joint velocity limit in rad/s. If not specified, no limit is applied.")
    
    # Connection
    parser.add_argument("--wait-for-server", action="store_true",
                        help="Wait indefinitely for the inference server to become available")
    
    # Workarounds
    parser.add_argument("--zero-action-threshold", type=float, default=None,
                        help="Replace near-zero actions (xyz norm < threshold) with current pose. "
                             "Workaround for models trained with zeros for non-engaged arms. "
                             "Suggested value: 0.05 (5cm)")
    
    # Action horizon
    parser.add_argument("--action-horizon", type=int, default=None,
                        help="Only execute the first H actions from each action chunk. "
                             "If not specified, all actions in the chunk are executed.")
    
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
    print(f"Dry run:         {args.dry_run}")
    print(f"Velocity limit:  {args.velocity_limit} rad/s" if args.velocity_limit else "Velocity limit:  None (unlimited)")
    print(f"Wait for server: {args.wait_for_server}")
    print(f"Zero action fix: {args.zero_action_threshold}m threshold" if args.zero_action_threshold else "Zero action fix: disabled")
    print(f"Action horizon:  {args.action_horizon}" if args.action_horizon else "Action horizon:  None (use full chunk)")
    print("=" * 60)
    
    # Initialize client
    can_interfaces = {"left": args.left_can, "right": args.right_can}
    
    client = DWMYAMClient(
        server_addr=args.server,
        arms=args.arms,
        query_freq=args.query_frequency,
        timeout_ms=args.timeout_ms,
        gripper_type=args.gripper_type,
        can_interfaces=can_interfaces,
        dry_run=args.dry_run,
        velocity_limit=args.velocity_limit,
        wait_for_server=args.wait_for_server,
        zero_action_threshold=args.zero_action_threshold,
        action_horizon=args.action_horizon,
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
