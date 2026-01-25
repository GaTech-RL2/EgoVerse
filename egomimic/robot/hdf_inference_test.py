#!/usr/bin/env python3
"""
HDF5 Demo Replay to Inference Server.

This script loads an HDF5 demo file and streams observations to a DWM inference 
server (similar to dwm_yam_client.py) to test model predictions against ground
truth actions. It tracks and reports deviation statistics.

Usage:
    python hdf_inference_test.py \
        --hdf5 /path/to/demo.hdf5 \
        --server tcp://192.168.1.100:5555 \
        --query-frequency 16 \
        [--arm both|left|right] \
        [--output-dir ./inference_results]
"""

import os
import sys
import time
import argparse
import numpy as np
import h5py
import json
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, field, asdict
from scipy.spatial.transform import Rotation as R
from collections import defaultdict

# Optional plotting
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# ZMQ for server communication
try:
    import zmq
    import msgpack
    import msgpack_numpy
    msgpack_numpy.patch()
    ZMQ_AVAILABLE = True
except ImportError:
    ZMQ_AVAILABLE = False


# =============================================================================
# Action Format Utilities (YPR <-> rot6d)
# =============================================================================

def _reconstruct_rot_from_6d(rot6d: np.ndarray) -> np.ndarray:
    """Reconstruct rotation matrices from 6D representation.

    The 6D vector is stored as a row-major flatten of the first two
    columns of the rotation matrix (as saved by collect_demo.py):
    [R00, R01, R10, R11, R20, R21].

    Args:
        rot6d: (T, 6) array of first two rotation matrix columns.

    Returns:
        R: (T, 3, 3) rotation matrices.
    """
    eps = 1e-8
    rot_cols = rot6d.reshape(-1, 3, 2)
    c1 = rot_cols[:, :, 0]
    c2 = rot_cols[:, :, 1]
    c1n = c1 / (np.linalg.norm(c1, axis=-1, keepdims=True) + eps)
    proj = np.sum(c2 * c1n, axis=-1, keepdims=True) * c1n
    c2o = c2 - proj
    c2n = c2o / (np.linalg.norm(c2o, axis=-1, keepdims=True) + eps)
    c3n = np.cross(c1n, c2n)
    return np.stack([c1n, c2n, c3n], axis=-1)


def _rot_matrix_to_ypr(Rm: np.ndarray) -> np.ndarray:
    """Convert rotation matrices to yaw-pitch-roll (ZYX)."""
    eps = 1e-6
    sy = -Rm[:, 2, 0]
    sy = np.clip(sy, -1 + eps, 1 - eps)
    pitch = np.arcsin(sy)
    cy = np.cos(pitch)
    cy = np.clip(cy, eps, None)
    yaw = np.arctan2(Rm[:, 1, 0], Rm[:, 0, 0])
    roll = np.arctan2(Rm[:, 2, 1], Rm[:, 2, 2])
    return np.stack([yaw, pitch, roll], axis=-1)


def convert_rot6d_to_ypr(actions: np.ndarray) -> np.ndarray:
    """Convert bimanual 20D rot6d actions to 14D ypr actions."""
    squeeze = False
    if actions.ndim == 1:
        actions = actions[None, :]
        squeeze = True
    if actions.shape[1] != 20:
        raise ValueError(f"Expected 20D rot6d actions, got {actions.shape}")

    left_xyz = actions[:, 0:3]
    left_rot6d = actions[:, 3:9]
    left_grip = actions[:, 9:10]
    right_xyz = actions[:, 10:13]
    right_rot6d = actions[:, 13:19]
    right_grip = actions[:, 19:20]

    left_R = _reconstruct_rot_from_6d(left_rot6d)
    right_R = _reconstruct_rot_from_6d(right_rot6d)
    left_ypr = _rot_matrix_to_ypr(left_R)
    right_ypr = _rot_matrix_to_ypr(right_R)

    out = np.concatenate(
        [left_xyz, left_ypr, left_grip, right_xyz, right_ypr, right_grip],
        axis=-1,
    )
    return out[0] if squeeze else out


def convert_ypr_to_rot6d(actions: np.ndarray) -> np.ndarray:
    """Convert bimanual 14D ypr actions to 20D rot6d actions."""
    squeeze = False
    if actions.ndim == 1:
        actions = actions[None, :]
        squeeze = True
    if actions.shape[1] != 14:
        raise ValueError(f"Expected 14D ypr actions, got {actions.shape}")

    left_xyz = actions[:, 0:3]
    left_ypr = actions[:, 3:6]
    left_grip = actions[:, 6:7]
    right_xyz = actions[:, 7:10]
    right_ypr = actions[:, 10:13]
    right_grip = actions[:, 13:14]

    left_R = R.from_euler("ZYX", left_ypr).as_matrix()
    right_R = R.from_euler("ZYX", right_ypr).as_matrix()
    left_rot6d = left_R[:, :, :2].reshape(-1, 6)
    right_rot6d = right_R[:, :, :2].reshape(-1, 6)

    out = np.concatenate(
        [left_xyz, left_rot6d, left_grip, right_xyz, right_rot6d, right_grip],
        axis=-1,
    )
    return out[0] if squeeze else out

# =============================================================================
# Deviation Statistics
# =============================================================================

@dataclass
class DeviationStats:
    """Track deviation statistics between predicted and ground truth actions."""
    
    # Position deviation (xyz)
    pos_errors: List[float] = field(default_factory=list)
    # Orientation deviation (ypr in radians)  
    rot_errors: List[float] = field(default_factory=list)
    # Gripper deviation
    gripper_errors: List[float] = field(default_factory=list)
    
    # Per-dimension errors (for detailed analysis)
    per_dim_errors: Dict[str, List[float]] = field(default_factory=lambda: {
        'x': [], 'y': [], 'z': [],
        'yaw': [], 'pitch': [], 'roll': [],
        'gripper': []
    })
    
    # Per-step tracking
    step_indices: List[int] = field(default_factory=list)
    pred_actions: List[np.ndarray] = field(default_factory=list)
    gt_actions: List[np.ndarray] = field(default_factory=list)
    
    # Arm tracking (for bimanual)
    arm_labels: List[str] = field(default_factory=list)
    
    def add(self, step: int, pred: np.ndarray, gt: np.ndarray, arm: str = "both"):
        """Add a comparison between predicted and ground truth actions.
        
        Args:
            step: Step index
            pred: Predicted action (7 or 14 dims: xyz + ypr + gripper per arm)
            gt: Ground truth action (same format)
            arm: Which arm ("left", "right", "both")
        """
        self.step_indices.append(step)
        self.pred_actions.append(pred.copy())
        self.gt_actions.append(gt.copy())
        
        if arm == "both":
            # Process both arms
            for arm_idx, arm_name in enumerate(["left", "right"]):
                offset = arm_idx * 7
                self._compute_errors(pred[offset:offset+7], gt[offset:offset+7], arm_name)
        else:
            # For single-arm tests, select the correct arm slice if data is bimanual
            offset = 0 if arm == "left" else 7
            pred_arm = pred
            gt_arm = gt
            if pred.shape[0] >= 14:
                pred_arm = pred[offset:offset+7]
            if gt.shape[0] >= 14:
                gt_arm = gt[offset:offset+7]
            self._compute_errors(pred_arm[:7], gt_arm[:7], arm)
    
    def _compute_errors(self, pred: np.ndarray, gt: np.ndarray, arm_label: str = ""):
        """Compute errors for a single arm (7D: xyz + ypr + gripper)."""
        self.arm_labels.append(arm_label)
        
        # Position error (Euclidean distance)
        pos_err = np.linalg.norm(pred[:3] - gt[:3])
        self.pos_errors.append(pos_err)
        
        # Per-dimension position errors
        self.per_dim_errors['x'].append(pred[0] - gt[0])
        self.per_dim_errors['y'].append(pred[1] - gt[1])
        self.per_dim_errors['z'].append(pred[2] - gt[2])
        
        # Rotation error (angle between orientations)
        pred_R = R.from_euler('ZYX', pred[3:6])
        gt_R = R.from_euler('ZYX', gt[3:6])
        rot_diff = (pred_R.inv() * gt_R).as_rotvec()
        rot_err = np.linalg.norm(rot_diff)  # radians
        self.rot_errors.append(rot_err)
        
        # Per-dimension rotation errors (wrapped to [-pi, pi])
        yaw_diff = (pred[3] - gt[3] + np.pi) % (2 * np.pi) - np.pi
        pitch_diff = (pred[4] - gt[4] + np.pi) % (2 * np.pi) - np.pi  
        roll_diff = (pred[5] - gt[5] + np.pi) % (2 * np.pi) - np.pi
        self.per_dim_errors['yaw'].append(yaw_diff)
        self.per_dim_errors['pitch'].append(pitch_diff)
        self.per_dim_errors['roll'].append(roll_diff)
        
        # Gripper error
        gripper_err = pred[6] - gt[6]
        self.gripper_errors.append(abs(gripper_err))
        self.per_dim_errors['gripper'].append(gripper_err)
    
    def summary(self) -> Dict:
        """Compute summary statistics."""
        if not self.pos_errors:
            return {"error": "No data collected"}
        
        def dim_stats(arr, scale=1.0):
            """Compute stats for an array with optional scale."""
            arr = np.array(arr) * scale
            return {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "max": float(np.max(arr)),
                "min": float(np.min(arr)),
                "abs_mean": float(np.mean(np.abs(arr))),
            }
        
        return {
            "num_samples": len(self.step_indices),
            "position": {
                "mean_mm": float(np.mean(self.pos_errors) * 1000),
                "std_mm": float(np.std(self.pos_errors) * 1000),
                "max_mm": float(np.max(self.pos_errors) * 1000),
                "min_mm": float(np.min(self.pos_errors) * 1000),
            },
            "rotation": {
                "mean_deg": float(np.rad2deg(np.mean(self.rot_errors))),
                "std_deg": float(np.rad2deg(np.std(self.rot_errors))),
                "max_deg": float(np.rad2deg(np.max(self.rot_errors))),
                "min_deg": float(np.rad2deg(np.min(self.rot_errors))),
            },
            "gripper": {
                "mean": float(np.mean(self.gripper_errors)),
                "std": float(np.std(self.gripper_errors)),
                "max": float(np.max(self.gripper_errors)),
                "min": float(np.min(self.gripper_errors)),
            },
            "per_dimension_mm": {
                "x": dim_stats(self.per_dim_errors['x'], 1000),
                "y": dim_stats(self.per_dim_errors['y'], 1000),
                "z": dim_stats(self.per_dim_errors['z'], 1000),
            },
            "per_dimension_deg": {
                "yaw": dim_stats(self.per_dim_errors['yaw'], 180/np.pi),
                "pitch": dim_stats(self.per_dim_errors['pitch'], 180/np.pi),
                "roll": dim_stats(self.per_dim_errors['roll'], 180/np.pi),
            }
        }
    
    def print_summary(self):
        """Print a formatted summary."""
        stats = self.summary()
        if "error" in stats:
            print(f"[DeviationStats] {stats['error']}")
            return
        
        print("\n" + "=" * 60)
        print("INFERENCE DEVIATION SUMMARY")
        print("=" * 60)
        print(f"Total samples: {stats['num_samples']}")
        
        print("\nPosition Error (Euclidean, mm):")
        print(f"  Mean: {stats['position']['mean_mm']:.2f}")
        print(f"  Std:  {stats['position']['std_mm']:.2f}")
        print(f"  Max:  {stats['position']['max_mm']:.2f}")
        print(f"  Min:  {stats['position']['min_mm']:.2f}")
        
        print("\nPer-Axis Position Error (mm):")
        for axis in ['x', 'y', 'z']:
            d = stats['per_dimension_mm'][axis]
            print(f"  {axis.upper()}: mean={d['mean']:+.2f}, |mean|={d['abs_mean']:.2f}, "
                  f"std={d['std']:.2f}, range=[{d['min']:.2f}, {d['max']:.2f}]")
        
        print("\nRotation Error (angle, degrees):")
        print(f"  Mean: {stats['rotation']['mean_deg']:.2f}")
        print(f"  Std:  {stats['rotation']['std_deg']:.2f}")
        print(f"  Max:  {stats['rotation']['max_deg']:.2f}")
        print(f"  Min:  {stats['rotation']['min_deg']:.2f}")
        
        print("\nPer-Axis Rotation Error (degrees):")
        for axis in ['yaw', 'pitch', 'roll']:
            d = stats['per_dimension_deg'][axis]
            print(f"  {axis.capitalize()}: mean={d['mean']:+.2f}, |mean|={d['abs_mean']:.2f}, "
                  f"std={d['std']:.2f}, range=[{d['min']:.2f}, {d['max']:.2f}]")
        
        print("\nGripper Error (normalized 0-1):")
        print(f"  Mean: {stats['gripper']['mean']:.4f}")
        print(f"  Std:  {stats['gripper']['std']:.4f}")
        print(f"  Max:  {stats['gripper']['max']:.4f}")
        print(f"  Min:  {stats['gripper']['min']:.4f}")
        print("=" * 60)


@dataclass 
class ChunkDeviationStats:
    """Track deviation statistics for entire action chunks (not just first action)."""
    
    chunk_start_steps: List[int] = field(default_factory=list)
    chunk_pos_errors: List[np.ndarray] = field(default_factory=list)  # List of (chunk_len,) arrays
    chunk_rot_errors: List[np.ndarray] = field(default_factory=list)
    chunk_predictions: List[np.ndarray] = field(default_factory=list)  # Full predicted chunks
    chunk_ground_truths: List[np.ndarray] = field(default_factory=list)  # Corresponding GT windows
    
    def add_chunk(self, step: int, pred_chunk: np.ndarray, gt_window: np.ndarray, arm: str = "both"):
        """Add a chunk comparison.
        
        Args:
            step: Starting step index
            pred_chunk: Predicted action chunk (T, 7) or (T, 14)
            gt_window: Ground truth actions for same window (T, 7) or (T, 14)
            arm: Which arm
        """
        self.chunk_start_steps.append(step)
        self.chunk_predictions.append(pred_chunk.copy())
        self.chunk_ground_truths.append(gt_window.copy())
        
        # Compute per-timestep errors for the chunk
        chunk_len = min(pred_chunk.shape[0], gt_window.shape[0])
        pos_errors = np.zeros(chunk_len)
        rot_errors = np.zeros(chunk_len)
        
        for t in range(chunk_len):
            if arm == "both":
                # Average over both arms
                for arm_idx in range(2):
                    offset = arm_idx * 7
                    pos_err = np.linalg.norm(pred_chunk[t, offset:offset+3] - gt_window[t, offset:offset+3])
                    
                    pred_R = R.from_euler('ZYX', pred_chunk[t, offset+3:offset+6])
                    gt_R = R.from_euler('ZYX', gt_window[t, offset+3:offset+6])
                    rot_err = np.linalg.norm((pred_R.inv() * gt_R).as_rotvec())
                    
                    pos_errors[t] += pos_err / 2  # Average
                    rot_errors[t] += rot_err / 2
            else:
                pos_errors[t] = np.linalg.norm(pred_chunk[t, :3] - gt_window[t, :3])
                pred_R = R.from_euler('ZYX', pred_chunk[t, 3:6])
                gt_R = R.from_euler('ZYX', gt_window[t, 3:6])
                rot_errors[t] = np.linalg.norm((pred_R.inv() * gt_R).as_rotvec())
        
        self.chunk_pos_errors.append(pos_errors)
        self.chunk_rot_errors.append(rot_errors)
    
    def summary(self) -> Dict:
        """Compute chunk-level statistics."""
        if not self.chunk_pos_errors:
            return {"error": "No chunk data collected"}
        
        # Stack all errors
        all_pos = np.concatenate(self.chunk_pos_errors)
        all_rot = np.concatenate(self.chunk_rot_errors)
        
        # Per-position-in-chunk analysis
        max_len = max(len(e) for e in self.chunk_pos_errors)
        pos_by_idx = defaultdict(list)
        rot_by_idx = defaultdict(list)
        
        for pos_arr, rot_arr in zip(self.chunk_pos_errors, self.chunk_rot_errors):
            for i, (p, r) in enumerate(zip(pos_arr, rot_arr)):
                pos_by_idx[i].append(p)
                rot_by_idx[i].append(r)
        
        pos_by_idx_stats = {}
        rot_by_idx_stats = {}
        for i in range(max_len):
            if i in pos_by_idx:
                pos_by_idx_stats[i] = {
                    "mean_mm": float(np.mean(pos_by_idx[i]) * 1000),
                    "std_mm": float(np.std(pos_by_idx[i]) * 1000),
                }
                rot_by_idx_stats[i] = {
                    "mean_deg": float(np.rad2deg(np.mean(rot_by_idx[i]))),
                    "std_deg": float(np.rad2deg(np.std(rot_by_idx[i]))),
                }
        
        return {
            "num_chunks": len(self.chunk_pos_errors),
            "overall_position_mm": {
                "mean": float(np.mean(all_pos) * 1000),
                "std": float(np.std(all_pos) * 1000),
            },
            "overall_rotation_deg": {
                "mean": float(np.rad2deg(np.mean(all_rot))),
                "std": float(np.rad2deg(np.std(all_rot))),
            },
            "by_chunk_position": pos_by_idx_stats,
            "by_chunk_position_idx": rot_by_idx_stats,
        }
    
    def print_summary(self):
        """Print chunk deviation summary."""
        stats = self.summary()
        if "error" in stats:
            print(f"[ChunkStats] {stats['error']}")
            return
        
        print("\n" + "-" * 60)
        print("CHUNK-LEVEL DEVIATION ANALYSIS")
        print("-" * 60)
        print(f"Number of chunks: {stats['num_chunks']}")
        print(f"Overall position error: {stats['overall_position_mm']['mean']:.2f} ± {stats['overall_position_mm']['std']:.2f} mm")
        print(f"Overall rotation error: {stats['overall_rotation_deg']['mean']:.2f} ± {stats['overall_rotation_deg']['std']:.2f} deg")
        
        print("\nError by position in chunk (position mm, rotation deg):")
        for i in sorted(stats['by_chunk_position'].keys()):
            pos = stats['by_chunk_position'][i]
            rot = stats['by_chunk_position_idx'][i]
            print(f"  t+{i}: pos={pos['mean_mm']:.1f}±{pos['std_mm']:.1f}mm, rot={rot['mean_deg']:.1f}±{rot['std_deg']:.1f}°")
        print("-" * 60)


# =============================================================================
# HDF5 Demo Loader
# =============================================================================

class HDF5DemoLoader:
    """Load and iterate through HDF5 demo files."""
    
    def __init__(self, hdf5_path: str, arm: str = "both"):
        """Initialize demo loader.
        
        Args:
            hdf5_path: Path to HDF5 demo file
            arm: Which arm(s) to process ("left", "right", "both")
        """
        self.hdf5_path = Path(hdf5_path)
        if not self.hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")
        
        self.arm = arm
        self._load_demo()
    
    def _load_demo(self):
        """Load demo data from HDF5 file."""
        print(f"Loading HDF5 demo: {self.hdf5_path}")
        
        with h5py.File(self.hdf5_path, 'r') as f:
            # Print available keys for debugging
            print("HDF5 structure:")
            self._print_hdf5_structure(f, indent=2)
            
            # Load images
            self.images = {}
            if 'observations' in f and 'images' in f['observations']:
                for cam_name in f['observations']['images'].keys():
                    img_data = f['observations']['images'][cam_name][:]
                    self.images[cam_name] = img_data
                    print(f"  Loaded {cam_name}: {img_data.shape}")
            
            # Load joint positions
            if 'observations' in f and 'joint_positions' in f['observations']:
                self.joint_positions = f['observations']['joint_positions'][:]
            elif 'observations' in f and 'joints' in f['observations']:
                self.joint_positions = f['observations']['joints'][:]
            else:
                raise KeyError("Missing joint_positions in HDF5 file")
            
            # Load ground truth actions
            # Try multiple possible locations
            self.gt_actions_ypr = None
            self.gt_actions_rot6d = None

            if 'actions' in f:
                if 'eepose_ypr' in f['actions']:
                    self.gt_actions_ypr = f['actions']['eepose_ypr'][:]
                    print(f"  Loaded eepose_ypr actions: {self.gt_actions_ypr.shape}")
                # Support both naming variants
                if 'eepose_6drot' in f['actions']:
                    self.gt_actions_rot6d = f['actions']['eepose_6drot'][:]
                    print(f"  Loaded eepose_6drot actions: {self.gt_actions_rot6d.shape}")
                elif 'eepose_rot6d' in f['actions']:
                    self.gt_actions_rot6d = f['actions']['eepose_rot6d'][:]
                    print(f"  Loaded eepose_rot6d actions: {self.gt_actions_rot6d.shape}")
                if 'eepose' in f['actions'] and self.gt_actions_ypr is None:
                    self.gt_actions_ypr = f['actions']['eepose'][:]
                    print(f"  Loaded eepose actions: {self.gt_actions_ypr.shape}")

            if 'observations' in f and 'eepose' in f['observations'] and self.gt_actions_ypr is None:
                self.gt_actions_ypr = f['observations']['eepose'][:]
                print(f"  Loaded observations/eepose: {self.gt_actions_ypr.shape}")

            self.gt_actions_eepose = (
                self.gt_actions_ypr if self.gt_actions_ypr is not None else self.gt_actions_rot6d
            )
            if self.gt_actions_eepose is None:
                print("  Warning: No eepose actions found in HDF5")
            
            # Load joint actions for reference
            if 'action' in f:
                self.gt_actions_joints = f['action'][:]
            elif 'actions' in f and 'joints' in f['actions']:
                self.gt_actions_joints = f['actions']['joints'][:]
            else:
                self.gt_actions_joints = None
        
        self.num_steps = self.joint_positions.shape[0]
        print(f"  Total steps: {self.num_steps}")
        
        # Determine available camera views
        self.camera_names = list(self.images.keys())
        print(f"  Available cameras: {self.camera_names}")
    
    def _print_hdf5_structure(self, group, indent=0):
        """Recursively print HDF5 structure."""
        for key in group.keys():
            item = group[key]
            prefix = " " * indent
            if isinstance(item, h5py.Group):
                print(f"{prefix}{key}/")
                if indent < 4:  # Limit depth
                    self._print_hdf5_structure(item, indent + 2)
            else:
                print(f"{prefix}{key}: {item.shape} {item.dtype}")
    
    def get_observation(self, step: int) -> Dict:
        """Get observation at a specific step.
        
        Args:
            step: Step index
            
        Returns:
            Observation dict with images and joint_positions
        """
        obs = {
            'joint_positions': self.joint_positions[step].copy()
        }
        
        for cam_name in self.camera_names:
            img = self.images[cam_name][step]
            # Ensure BGR format (if stored as RGB, convert)
            if img.ndim == 3 and img.shape[2] == 3:
                # Assume stored as RGB, convert to BGR for inference server
                obs[cam_name] = img[..., ::-1].copy()
            else:
                obs[cam_name] = img.copy()
        
        return obs
    
    def get_ground_truth(self, step: int, action_dim: Optional[int] = None) -> Optional[np.ndarray]:
        """Get ground truth eepose action at a specific step.
        
        Args:
            step: Step index
            
        Returns:
            Ground truth action (14D: xyz+ypr+grip for each arm) or None
        """
        if action_dim is None or action_dim in (7, 14):
            if self.gt_actions_ypr is not None and step < len(self.gt_actions_ypr):
                return self.gt_actions_ypr[step].copy()
            if self.gt_actions_rot6d is not None and step < len(self.gt_actions_rot6d):
                return convert_rot6d_to_ypr(self.gt_actions_rot6d[step])
        if action_dim == 20:
            if self.gt_actions_rot6d is not None and step < len(self.gt_actions_rot6d):
                return self.gt_actions_rot6d[step].copy()
            if self.gt_actions_ypr is not None and step < len(self.gt_actions_ypr):
                return convert_ypr_to_rot6d(self.gt_actions_ypr[step])
        return None
    
    def __len__(self):
        return self.num_steps
    
    def __iter__(self):
        for i in range(self.num_steps):
            yield i, self.get_observation(i), self.get_ground_truth(i)


# =============================================================================
# Inference Client
# =============================================================================

class InferenceClient:
    """Client to communicate with DWM inference server."""
    
    def __init__(
        self,
        server_addr: str,
        timeout_ms: int = 10000,
    ):
        """Initialize inference client.
        
        Args:
            server_addr: ZMQ server address (e.g., "tcp://192.168.1.100:5555")
            timeout_ms: Request timeout in milliseconds
        """
        if not ZMQ_AVAILABLE:
            raise ImportError("zmq and msgpack-numpy required: pip install pyzmq msgpack msgpack-numpy")
        
        self.server_addr = server_addr
        self.timeout_ms = timeout_ms
        
        # Setup ZMQ
        print(f"Connecting to inference server: {server_addr}")
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.connect(server_addr)
        
        # Test connection
        self._ping()
    
    def _ping(self):
        """Test server connection."""
        try:
            self.sock.send(msgpack.packb({'cmd': 'ping'}, use_bin_type=True))
            reply = msgpack.unpackb(self.sock.recv(), raw=False)
            if reply.get('status') == 'ok':
                print("Server connection verified.")
            else:
                raise RuntimeError(f"Server ping failed: {reply}")
        except zmq.error.Again:
            raise RuntimeError(f"Server not responding at {self.server_addr}")
    
    def request_inference(self, obs: Dict, step: int) -> Tuple[np.ndarray, float]:
        """Request inference from server.
        
        Args:
            obs: Observation dictionary with images and joint_positions
            step: Current timestep
            
        Returns:
            Tuple of (actions array, inference time in seconds)
        """
        # Prepare observation data
        obs_data = {}
        for key, val in obs.items():
            if isinstance(val, np.ndarray):
                obs_data[key] = val
            elif val is not None:
                obs_data[key] = np.asarray(val)
        
        print(f'{step=} {obs_data=}')
        msg = {'cmd': 'infer', 'obs': obs_data, 'step': step}
        
        t0 = time.time()
        self.sock.send(msgpack.packb(msg, use_bin_type=True))
        reply = msgpack.unpackb(self.sock.recv(), raw=False)
        rtt = time.time() - t0
        
        if 'error' in reply:
            raise RuntimeError(f"Server error: {reply['error']}")
        
        actions = reply['actions']
        infer_t = reply.get('infer_time', 0)
        
        return np.asarray(actions), infer_t
    
    def close(self):
        """Close connection."""
        self.sock.close()
        self.ctx.term()
        print("Client closed.")


# =============================================================================
# Camera Frame Transforms
# =============================================================================

# Try to import from egomimicUtils, otherwise use local fallback
try:
    from egomimic.utils.egomimicUtils import EXTRINSICS, cam_frame_to_base_frame
    EGOMIMIC_UTILS_AVAILABLE = True
except ImportError:
    EGOMIMIC_UTILS_AVAILABLE = False
    
    def cam_frame_to_base_frame(actions_cam: np.ndarray, T_cam_base: np.ndarray) -> np.ndarray:
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

    # Fallback extrinsics
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
    }


def list_available_extrinsics():
    """Print available extrinsics keys."""
    print("\nAvailable extrinsics keys:")
    for key in sorted(EXTRINSICS.keys()):
        val = EXTRINSICS[key]
        if isinstance(val, dict):
            arms = list(val.keys())
            print(f"  {key}: arms={arms}")
        else:
            print(f"  {key}: single matrix")


# =============================================================================
# Main Test Runner
# =============================================================================

def run_inference_test(
    hdf5_path: str,
    server_addr: str,
    query_frequency: int,
    arm: str = "both",
    extrinsics_key: str = "x5Dec13_2",
    output_dir: Optional[str] = None,
    max_steps: Optional[int] = None,
    skip_transform: bool = False,
):
    """Run inference test on HDF5 demo.
    
    Args:
        hdf5_path: Path to HDF5 demo file
        server_addr: Inference server address
        query_frequency: How often to request new actions
        arm: Which arm(s) to test
        extrinsics_key: Camera extrinsics key
        output_dir: Optional directory to save results
        max_steps: Maximum number of steps to process
        skip_transform: If True, skip camera-to-base transform (for debugging)
    """
    # Load demo
    loader = HDF5DemoLoader(hdf5_path, arm)
    
    # Connect to server
    client = InferenceClient(server_addr)
    
    # Setup extrinsics
    extrinsics = EXTRINSICS.get(extrinsics_key, EXTRINSICS["x5Dec13_2"])
    
    # Track statistics
    stats = DeviationStats()
    timing_stats = {
        'roundtrip_times': [],
        'inference_times': [],
    }
    
    # Action buffer for temporal execution
    actions_buffer = None
    
    num_steps = len(loader) if max_steps is None else min(len(loader), max_steps)
    
    print(f"\nRunning inference test on {num_steps} steps...")
    print(f"Query frequency: {query_frequency}")
    print(f"Arm: {arm}")
    print(f"Skip transform: {skip_transform}")
    print("-" * 40)
    
    try:
        for i in range(num_steps):
            obs = loader.get_observation(i)
            gt = loader.get_ground_truth(i, action_dim=14)
            
            # Request new actions at query frequency
            if i % query_frequency == 0:
                t0 = time.time()
                actions_pred, infer_time = client.request_inference(obs, i)
                rtt = time.time() - t0

                # Validate action shape and normalize to ypr for comparison
                if actions_pred.ndim != 2:
                    raise ValueError(f"Expected actions with shape (T, D), got {actions_pred.shape}")
                action_dim = actions_pred.shape[1]
                if action_dim == 20:
                    actions_ypr = convert_rot6d_to_ypr(actions_pred)
                    if not hasattr(run_inference_test, "_logged_rot6d_to_ypr"):
                        print("Converting 20D rot6d actions to 14D ypr for evaluation.")
                        run_inference_test._logged_rot6d_to_ypr = True
                elif action_dim in (7, 14):
                    actions_ypr = actions_pred
                else:
                    raise ValueError(f"Unexpected action dim {action_dim} (expected 7, 14, or 20)")

                if arm == "both" and actions_ypr.shape[1] != 14:
                    raise ValueError(
                        f"Arm='both' requires 14D ypr actions, got {actions_ypr.shape[1]}D"
                    )
                
                timing_stats['roundtrip_times'].append(rtt)
                timing_stats['inference_times'].append(infer_time)
                
                # Transform to base frame if needed
                if not skip_transform:
                    if arm == "both":
                        left = actions_ypr[:, :7]
                        right = actions_ypr[:, 7:14]
                        
                        left_6d = cam_frame_to_base_frame(left[:, :6], extrinsics["left"])
                        right_6d = cam_frame_to_base_frame(right[:, :6], extrinsics["right"])
                        
                        left_out = np.hstack([left_6d, left[:, 6:7]])
                        right_out = np.hstack([right_6d, right[:, 6:7]])
                        
                        actions_buffer = np.hstack([left_out, right_out])
                    else:
                        arm_offset = 7 if arm == "right" else 0
                        arm_actions = actions_ypr[:, arm_offset:arm_offset + 7]
                        
                        transformed_6d = cam_frame_to_base_frame(
                            arm_actions[:, :6],
                            extrinsics[arm]
                        )
                        actions_buffer = np.hstack([transformed_6d, arm_actions[:, 6:7]])
                else:
                    actions_buffer = actions_ypr
                
                print(f"Step {i}: RTT={rtt*1000:.1f}ms, infer={infer_time*1000:.1f}ms, "
                      f"actions shape={actions_buffer.shape}")
            
            # Get action at current index
            if actions_buffer is not None and gt is not None:
                act_idx = i % query_frequency
                if act_idx >= actions_buffer.shape[0]:
                    act_idx = actions_buffer.shape[0] - 1
                
                pred_action = actions_buffer[act_idx]
                
                # Compare with ground truth
                stats.add(i, pred_action, gt, arm)
                
                if i % 50 == 0 or i == num_steps - 1:
                    # Print periodic progress
                    pos_err = stats.pos_errors[-1] * 1000 if stats.pos_errors else 0
                    rot_err = np.rad2deg(stats.rot_errors[-1]) if stats.rot_errors else 0
                    print(f"  Step {i}/{num_steps}: pos_err={pos_err:.1f}mm, rot_err={rot_err:.1f}deg")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        raise
    finally:
        client.close()
    
    # Print summary
    stats.print_summary()
    
    # Timing summary
    if timing_stats['roundtrip_times']:
        print("\nTiming Statistics:")
        print(f"  Roundtrip: mean={np.mean(timing_stats['roundtrip_times'])*1000:.1f}ms, "
              f"std={np.std(timing_stats['roundtrip_times'])*1000:.1f}ms")
        print(f"  Inference: mean={np.mean(timing_stats['inference_times'])*1000:.1f}ms, "
              f"std={np.std(timing_stats['inference_times'])*1000:.1f}ms")
    
    # Save results
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save statistics
        results = {
            'hdf5_path': str(hdf5_path),
            'server_addr': server_addr,
            'query_frequency': query_frequency,
            'arm': arm,
            'extrinsics_key': extrinsics_key,
            'num_steps': num_steps,
            'deviation_stats': stats.summary(),
            'timing_stats': {
                'roundtrip_mean_ms': float(np.mean(timing_stats['roundtrip_times'])) * 1000,
                'roundtrip_std_ms': float(np.std(timing_stats['roundtrip_times'])) * 1000,
                'inference_mean_ms': float(np.mean(timing_stats['inference_times'])) * 1000,
                'inference_std_ms': float(np.std(timing_stats['inference_times'])) * 1000,
            }
        }
        
        results_file = output_path / 'inference_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_file}")
        
        # Save detailed data
        np.savez(
            output_path / 'detailed_data.npz',
            step_indices=np.array(stats.step_indices),
            pred_actions=np.array(stats.pred_actions) if stats.pred_actions else np.array([]),
            gt_actions=np.array(stats.gt_actions) if stats.gt_actions else np.array([]),
            pos_errors=np.array(stats.pos_errors),
            rot_errors=np.array(stats.rot_errors),
            gripper_errors=np.array(stats.gripper_errors),
            roundtrip_times=np.array(timing_stats['roundtrip_times']),
            inference_times=np.array(timing_stats['inference_times']),
        )
        print(f"Detailed data saved to: {output_path / 'detailed_data.npz'}")
        
        # Generate plots if matplotlib available
        if MATPLOTLIB_AVAILABLE and stats.pos_errors:
            _generate_plots(stats, timing_stats, output_path)
    
    return stats


def _generate_plots(stats: DeviationStats, timing_stats: Dict, output_path: Path):
    """Generate deviation plots."""
    
    # Main summary plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Position error over time
    ax1 = axes[0, 0]
    ax1.plot(np.array(stats.pos_errors) * 1000, 'b-', alpha=0.7, linewidth=0.8)
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Position Error (mm)')
    ax1.set_title('Position Error Over Time')
    ax1.axhline(np.mean(stats.pos_errors) * 1000, color='red', linestyle='--', 
                alpha=0.7, label=f'Mean: {np.mean(stats.pos_errors)*1000:.1f}mm')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Rotation error over time
    ax2 = axes[0, 1]
    ax2.plot(np.rad2deg(stats.rot_errors), 'r-', alpha=0.7, linewidth=0.8)
    ax2.set_xlabel('Sample Index')
    ax2.set_ylabel('Rotation Error (degrees)')
    ax2.set_title('Rotation Error Over Time')
    ax2.axhline(np.rad2deg(np.mean(stats.rot_errors)), color='blue', linestyle='--',
                alpha=0.7, label=f'Mean: {np.rad2deg(np.mean(stats.rot_errors)):.1f}°')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Position error histogram
    ax3 = axes[1, 0]
    ax3.hist(np.array(stats.pos_errors) * 1000, bins=50, color='blue', alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Position Error (mm)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Position Error Distribution')
    ax3.axvline(np.mean(stats.pos_errors) * 1000, color='red', linestyle='--', linewidth=2, label='Mean')
    ax3.axvline(np.median(stats.pos_errors) * 1000, color='green', linestyle=':', linewidth=2, label='Median')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Timing histogram
    ax4 = axes[1, 1]
    if timing_stats['inference_times']:
        ax4.hist(np.array(timing_stats['inference_times']) * 1000, bins=30, color='green', alpha=0.7, edgecolor='black')
        ax4.axvline(np.mean(timing_stats['inference_times']) * 1000, color='red', linestyle='--', linewidth=2, label='Mean')
        ax4.legend()
    ax4.set_xlabel('Inference Time (ms)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Inference Time Distribution')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_path / 'deviation_plots.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Plots saved to: {plot_path}")
    
    # Per-dimension plots
    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 8))
    
    # Position per-axis
    for i, (ax, dim) in enumerate(zip(axes2[0], ['x', 'y', 'z'])):
        errors = np.array(stats.per_dim_errors[dim]) * 1000
        ax.plot(errors, alpha=0.7, linewidth=0.8)
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax.axhline(np.mean(errors), color='red', linestyle='--', alpha=0.7)
        ax.set_xlabel('Sample Index')
        ax.set_ylabel(f'{dim.upper()} Error (mm)')
        ax.set_title(f'{dim.upper()} Error: mean={np.mean(errors):.2f}mm')
        ax.grid(True, alpha=0.3)
    
    # Rotation per-axis
    for i, (ax, dim) in enumerate(zip(axes2[1], ['yaw', 'pitch', 'roll'])):
        errors = np.rad2deg(stats.per_dim_errors[dim])
        ax.plot(errors, alpha=0.7, linewidth=0.8, color='orange')
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax.axhline(np.mean(errors), color='red', linestyle='--', alpha=0.7)
        ax.set_xlabel('Sample Index')
        ax.set_ylabel(f'{dim.capitalize()} Error (deg)')
        ax.set_title(f'{dim.capitalize()}: mean={np.mean(errors):.2f}°')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path2 = output_path / 'per_dimension_plots.png'
    plt.savefig(plot_path2, dpi=150)
    plt.close()
    print(f"Per-dimension plots saved to: {plot_path2}")


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Test DWM inference server with HDF5 demo replay",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic test
  python hdf_inference_test.py --hdf5 demo.hdf5 --server tcp://192.168.1.100:5555

  # Test with specific arm and save results  
  python hdf_inference_test.py --hdf5 demo.hdf5 --server tcp://localhost:5555 \\
      --arm right --query-frequency 16 --output-dir ./results

  # Quick test with limited steps
  python hdf_inference_test.py --hdf5 demo.hdf5 --server tcp://localhost:5555 \\
      --max-steps 100
      
  # List available extrinsics
  python hdf_inference_test.py --list-extrinsics
  
  # Inspect HDF5 structure only (no server needed)
  python hdf_inference_test.py --hdf5 demo.hdf5 --inspect-only
        """
    )
    
    # HDF5 and server arguments (required unless using utility flags)
    parser.add_argument("--hdf5", type=str, default=None,
                        help="Path to HDF5 demo file")
    parser.add_argument("--server", type=str, default=None,
                        help="Inference server address (e.g., tcp://192.168.1.100:5555)")
    
    # Optional arguments
    parser.add_argument("--query-frequency", type=int, default=16,
                        help="How often to request new actions (in timesteps)")
    parser.add_argument("--arm", type=str, default="both",
                        choices=["left", "right", "both"],
                        help="Which arm(s) to test")
    parser.add_argument("--extrinsics", type=str, default="x5Dec13_2",
                        help="Camera extrinsics key")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save results (optional)")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Maximum number of steps to process")
    parser.add_argument("--skip-transform", action="store_true",
                        help="Skip camera-to-base frame transform (for debugging)")
    
    # Utility flags
    parser.add_argument("--list-extrinsics", action="store_true",
                        help="List available extrinsics keys and exit")
    parser.add_argument("--inspect-only", action="store_true",
                        help="Only inspect HDF5 structure, don't run inference")
    
    args = parser.parse_args()
    
    # Handle utility flags
    if args.list_extrinsics:
        list_available_extrinsics()
        return
    
    if args.inspect_only:
        if not args.hdf5:
            parser.error("--hdf5 is required with --inspect-only")
        loader = HDF5DemoLoader(args.hdf5, args.arm)
        print(f"\nDemo has {len(loader)} steps")
        print(f"Camera views: {loader.camera_names}")
        print(f"Joint positions shape: {loader.joint_positions.shape}")
        if loader.gt_actions_ypr is not None:
            print(f"EE pose YPR actions shape: {loader.gt_actions_ypr.shape}")
        if loader.gt_actions_rot6d is not None:
            print(f"EE pose rot6d actions shape: {loader.gt_actions_rot6d.shape}")
        return
    
    # Validate required arguments for inference
    if not args.hdf5:
        parser.error("--hdf5 is required")
    if not args.server:
        parser.error("--server is required")
    
    # Check extrinsics
    if args.extrinsics not in EXTRINSICS:
        print(f"Warning: extrinsics key '{args.extrinsics}' not found.")
        list_available_extrinsics()
        print(f"\nUsing first available key instead...")
        args.extrinsics = list(EXTRINSICS.keys())[0]
    
    # Print configuration
    print("=" * 60)
    print("HDF5 Inference Test Configuration")
    print("=" * 60)
    print(f"HDF5 file:        {args.hdf5}")
    print(f"Server:           {args.server}")
    print(f"Query frequency:  {args.query_frequency}")
    print(f"Arm:              {args.arm}")
    print(f"Extrinsics:       {args.extrinsics}")
    print(f"Output dir:       {args.output_dir or 'None (no save)'}")
    print(f"Max steps:        {args.max_steps or 'All'}")
    print(f"Skip transform:   {args.skip_transform}")
    print("=" * 60)
    
    # Run test
    run_inference_test(
        hdf5_path=args.hdf5,
        server_addr=args.server,
        query_frequency=args.query_frequency,
        arm=args.arm,
        extrinsics_key=args.extrinsics,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        skip_transform=args.skip_transform,
    )


if __name__ == "__main__":
    main()
