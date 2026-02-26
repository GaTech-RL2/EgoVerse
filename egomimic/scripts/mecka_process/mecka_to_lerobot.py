#!/usr/bin/env python3
"""
Convert Mecka RL2 dataset to LeRobot format.
"""

import argparse
import json
import logging
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation

from egomimic.rldb.embodiment import EMBODIMENT

sys.path.insert(0, str(Path(__file__).parent / "lerobot"))

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EPISODE_LENGTH = 300
CHUNK_SIZE = 100


def download_with_retry(url, dest_path, max_retries=5):
    """Download file with retry logic for network connections."""
    import subprocess

    if Path(dest_path).exists():
        file_size = Path(dest_path).stat().st_size
        logger.info(
            f"File {Path(dest_path).name} already exists ({file_size} bytes), skipping download"
        )
        return

    for attempt in range(max_retries):
        try:
            logger.info(
                f"Downloading {Path(dest_path).name} (attempt {attempt + 1}/{max_retries})..."
            )

            subprocess.run(
                [
                    "curl",
                    "-L",
                    "-C",
                    "-",
                    "--retry",
                    "3",
                    "--retry-delay",
                    "2",
                    "-o",
                    str(dest_path),
                    url,
                ],
                check=True,
                capture_output=True,
                timeout=600,
                text=True,
            )

            if Path(dest_path).exists() and Path(dest_path).stat().st_size > 0:
                logger.info(
                    f"Successfully downloaded {Path(dest_path).name} ({Path(dest_path).stat().st_size} bytes)"
                )
                return
            else:
                raise Exception("Download completed but file is empty or missing")

        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"Download failed, retrying in 3 seconds... ({e})")
                if Path(dest_path).exists():
                    Path(dest_path).unlink()
                time.sleep(3)
            else:
                logger.error(
                    f"Failed to download {Path(dest_path).name} after {max_retries} attempts"
                )
                raise


# ROTATION_MATRIX = np.array([[1, 0, 0],
#                             [0, 1, 0],
#                             [0, 0, 1]])


def pose_to_transform(pose: np.ndarray) -> np.ndarray:
    """Convert 6DOF pose [x, y, z, yaw, pitch, roll] to 4x4 transform matrix."""
    x, y, z, yaw, pitch, roll = pose
    rotation = Rotation.from_euler("ZYX", [yaw, pitch, roll])
    T = np.eye(4)
    T[:3, :3] = rotation.as_matrix()
    T[:3, 3] = [x, y, z]
    return T


def transform_to_pose(T: np.ndarray) -> np.ndarray:
    """Convert 4x4 transform matrix to 6DOF pose [x, y, z, yaw, pitch, roll]."""
    x, y, z = T[:3, 3]
    rotation = Rotation.from_matrix(T[:3, :3])
    yaw, pitch, roll = rotation.as_euler("ZYX")
    return np.array([x, y, z, yaw, pitch, roll])


def compute_camera_relative_pose(
    pose: np.ndarray, cam_prev_inv: np.ndarray, cam_curr: np.ndarray
) -> np.ndarray:
    """
    Transform pose from world frame to camera-t frame.

    Args:
        pose: (6,) array [x, y, z, yaw, pitch, roll] in world frame at offset time
        cam_t_inv: (4, 4) inverse camera transform at timestep t
        cam_offset: (4, 4) camera transform at offset time

    Returns:
        (6,) array pose in camera-t frame [x, y, z, yaw, pitch, roll]
    """
    T_pose = pose_to_transform(pose)
    T_final = cam_prev_inv @ cam_curr @ T_pose
    pose_t = transform_to_pose(T_final)

    return pose_t


def compute_hand_pose_6dof(keypoints: np.ndarray) -> np.ndarray:
    """
    Compute 6DOF pose (x, y, z, yaw, pitch, roll) from hand keypoints.

    Args:
        keypoints: (21, 3) array of hand keypoints

    Returns:
        (6,) array [x, y, z, yaw, pitch, roll]
    """
    if np.allclose(keypoints, 0):
        return np.zeros(6)

    position = keypoints[0]
    wrist = keypoints[0]
    middle_base = keypoints[9]

    forward = middle_base - wrist
    if np.linalg.norm(forward) < 1e-6:
        return np.concatenate([position, np.zeros(3)])
    forward = forward / np.linalg.norm(forward)

    thumb_dir = keypoints[5] - wrist
    pinky_dir = keypoints[17] - wrist
    up = np.cross(thumb_dir, pinky_dir)
    if np.linalg.norm(up) < 1e-6:
        return np.concatenate([position, np.zeros(3)])
    up = up / np.linalg.norm(up)

    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)

    R = np.column_stack([forward, right, up])

    try:
        rotation = Rotation.from_matrix(R)
        yaw, pitch, roll = rotation.as_euler("zyx")
    except Exception:
        yaw, pitch, roll = 0, 0, 0

    return np.concatenate([position, [yaw, pitch, roll]])


class MeckaExtractor:
    """Extract features from Mecka RL2 format."""

    TAGS = ["mecka", "robotics", "human_hands"]

    @staticmethod
    def process_episode(
        episode_json_path: str,
        arm: str = "both",
        prestack: bool = True,
        local_data_dir: Optional[Path] = None,
    ) -> dict:
        """
        Extract all features from episode JSON.

        Args:
            episode_json_path: Path to episode JSON file
            arm: "left", "right", or "both"
            prestack: Whether to prestack action chunks
            local_data_dir: Optional directory containing pre-downloaded files (video, hands.csv, egomotion.txt, frames.csv, annotations.csv). If provided, downloads are skipped.

        Returns:
            episode_feats: Dictionary with observations and action keys
                {
                    "observations": {
                        "state.ee_pose_cam": (T, 12),
                        "images.front_img_1": (T, H, W, 3)
                    },
                    "actions_ee_cartesian_cam": (T, 100, 12),
                    "actions_ee_keypoints_world": (T, 100, 126),
                    "actions_head_cartesian_world": (T, 10),
                    "timestamp": (T,),
                    "frame_index": (T,)
                }
        """
        logger.info(f"Processing episode: {episode_json_path}")

        # Load episode metadata
        with open(episode_json_path, "r") as f:
            data = json.load(f)
            # Handle both list and dict formats
            episode_meta = data[0] if isinstance(data, list) else data

        local_data_dir = Path(local_data_dir) if local_data_dir is not None else None

        # Prepare data file paths, using local directory when provided
        try:
            episode_id = episode_meta.get("id", "unknown")

            if local_data_dir is None:
                temp_dir = Path(episode_json_path).parent / "temp_download"
                temp_dir.mkdir(exist_ok=True)

                logger.info("Downloading data files...")
                hands_path = temp_dir / "hands.csv"
                egomotion_path = temp_dir / "egomotion.txt"
                frames_path = temp_dir / "frames.csv"
                annotations_path = temp_dir / "annotations.csv"
                video_path = temp_dir / "video.mp4"

                root_video = Path(episode_json_path).parent / f"{episode_id}_video.mp4"
                if root_video.exists():
                    video_path = root_video
                    logger.info(f"Using pre-downloaded video: {video_path.name}")
                else:
                    download_with_retry(episode_meta["urls"]["video"], video_path)

                download_with_retry(episode_meta["urls"]["hands"], hands_path)
                download_with_retry(episode_meta["urls"]["egomotion"], egomotion_path)
                download_with_retry(episode_meta["urls"]["frames"], frames_path)
                download_with_retry(
                    episode_meta["urls"]["annotations"], annotations_path
                )
            else:
                logger.info(
                    f"Loading data files from local directory: {local_data_dir}"
                )
                hands_path = local_data_dir / "hands.csv"
                egomotion_path = local_data_dir / "egomotion.txt"
                frames_path = local_data_dir / "frames.csv"
                annotations_path = local_data_dir / "annotations.csv"

                candidate_videos = [
                    local_data_dir / f"{episode_id}_video.mp4",
                    local_data_dir / "video.mp4",
                ]
                video_path = next((p for p in candidate_videos if p.exists()), None)
                if video_path is None:
                    raise FileNotFoundError(
                        f"Could not find video in {local_data_dir}; expected one of {[str(p) for p in candidate_videos]}"
                    )

                for p in [hands_path, egomotion_path, frames_path, annotations_path]:
                    if not p.exists():
                        raise FileNotFoundError(
                            f"Missing required file for local load: {p}"
                        )

            hands_df = pd.read_csv(hands_path)
            egomotion = np.loadtxt(egomotion_path)
            frames_df = pd.read_csv(frames_path)
            annotations_df = pd.read_csv(annotations_path)

            camera_transforms = MeckaExtractor._extract_camera_transforms(egomotion)

            hand_poses_world, hand_keypoints_world = MeckaExtractor._extract_hand_data(
                hands_df, frames_df, arm
            )

            images = MeckaExtractor._extract_video_frames(video_path, len(frames_df))

            num_frames = min(len(images), len(frames_df), len(egomotion))
            logger.info(
                f"Syncing to {num_frames} frames (video={len(images)}, frames_df={len(frames_df)}, egomotion={len(egomotion)})"
            )

            images = images[:num_frames]

            # Downsample images to 640x360 (W x H)
            target_w, target_h = 640, 360
            downsampled_images = []
            for img in images:
                # Ensure RGB uint8
                if img.dtype != np.uint8:
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    else:
                        img = img.astype(np.uint8)
                ds = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
                downsampled_images.append(ds)
            images = np.array(downsampled_images)
            frames_df = frames_df.iloc[:num_frames]
            egomotion = egomotion[:num_frames]
            camera_transforms = camera_transforms[:num_frames]
            hand_poses_world = hand_poses_world[:num_frames]
            hand_keypoints_world = hand_keypoints_world[:num_frames]

            fps = 30
            timestamps = np.arange(num_frames) / fps
            frame_indices = np.arange(num_frames)

            ee_pose_cam = MeckaExtractor._compute_camera_relative_poses(
                hand_poses_world, camera_transforms
            )

            if prestack:
                actions_ee_cartesian_cam = MeckaExtractor._prestack_actions(
                    ee_pose_cam, chunk_size=CHUNK_SIZE
                )
                actions_ee_keypoints_world = MeckaExtractor._prestack_keypoints(
                    hand_keypoints_world, chunk_size=CHUNK_SIZE
                )
            else:
                actions_ee_cartesian_cam = ee_pose_cam
                actions_ee_keypoints_world = hand_keypoints_world

            actions_head_cartesian_world = MeckaExtractor._extract_head_poses(
                camera_transforms
            )

            if arm == "both":
                enum = EMBODIMENT.MECKA_BIMANUAL
            elif arm == "left":
                enum = EMBODIMENT.MECKA_LEFT_ARM
            else:
                enum = EMBODIMENT.MECKA_RIGHT_ARM

            embodiment = np.full((num_frames, 1), enum.value, dtype=np.int32)

            episode_feats = {
                "observations": {
                    "state.ee_pose_cam": ee_pose_cam,
                    "images.front_img_1": images,
                },
                "actions_ee_cartesian_cam": actions_ee_cartesian_cam,
                "actions_ee_keypoints_world": actions_ee_keypoints_world,
                "actions_head_cartesian_world": actions_head_cartesian_world,
                "metadata.embodiment": embodiment,
                "timestamp": timestamps,
                "frame_index": frame_indices,
                "annotations": annotations_df,
                "episode_meta": episode_meta,
            }

            logger.info(f"Extracted {len(frame_indices)} frames")
            return episode_feats

        finally:
            pass

    @staticmethod
    def _extract_camera_transforms(egomotion: np.ndarray) -> List[np.ndarray]:
        """
        Extract 4x4 camera transform matrices from egomotion data.

        Egomotion format (space-separated, no headers):
        Col 0: frame_index
        Col 1-3: x, y, z (position)
        Col 4-6: yaw, pitch, roll
        Col 7-10: quat_x, quat_y, quat_z, quat_w
        """
        transforms = []
        for row in egomotion:
            T = np.eye(4)
            # Position
            T[:3, 3] = row[1:4]  # x, y, z
            # Rotation from quaternion
            quat = row[7:11]  # quat_x, quat_y, quat_z, quat_w
            R = Rotation.from_quat(quat)
            T[:3, :3] = R.as_matrix()
            transforms.append(T)
        return transforms

    @staticmethod
    def _extract_hand_data(
        hands_df: pd.DataFrame, frames_df: pd.DataFrame, arm: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract hand poses and keypoints from hands CSV.

        Returns:
            hand_poses_world: (T, 12) - [left_6dof, right_6dof] in world frame
            hand_keypoints_world: (T, 126) - [left_21kp, right_21kp] flattened
        """
        num_frames = len(frames_df)
        hand_poses = np.zeros((num_frames, 12))
        hand_keypoints = np.zeros((num_frames, 2, 21, 3))

        for frame_idx in range(num_frames):
            for hand_index in [0, 1]:
                hand_data = hands_df[
                    (hands_df["frame"] == frame_idx)
                    & (hands_df["hand_index"] == hand_index)
                ].sort_values("landmark_index")

                if len(hand_data) == 21:
                    kp = hand_data[["world_x", "world_y", "world_z"]].values
                    hand_keypoints[frame_idx, hand_index] = kp

                    pose_6dof = compute_hand_pose_6dof(kp)

                    # remapping axes
                    x, y, z = pose_6dof[0], pose_6dof[1], pose_6dof[2]
                    remapped_xyz = np.array([-y, -z, x], dtype=np.float64)
                    pose_6dof[:3] = remapped_xyz

                    hand_poses[frame_idx, hand_index * 6 : (hand_index + 1) * 6] = (
                        pose_6dof
                    )

        hand_keypoints_flat = hand_keypoints.reshape(num_frames, 126)

        return hand_poses, hand_keypoints_flat

    @staticmethod
    def _extract_video_frames(video_path: Path, num_frames: int) -> np.ndarray:
        """Extract frames from video file."""
        cap = cv2.VideoCapture(str(video_path))
        frames = []

        for _ in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"Could not read frame {len(frames)}")
                break
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()

        if len(frames) < num_frames:
            logger.warning(f"Expected {num_frames} frames, got {len(frames)}")

        return np.array(frames)  # (T, H, W, 3)

    @staticmethod
    def _compute_camera_relative_poses(
        hand_poses_world: np.ndarray, camera_transforms: List[np.ndarray]
    ) -> np.ndarray:
        """
        Compute current hand poses in camera frame.

        Args:
            hand_poses_world: (T, 12) poses in world frame
            camera_transforms: List of (4, 4) camera transforms

        Returns:
            (T, 12) poses in camera frame
        """
        num_frames = len(hand_poses_world)
        ee_pose_cam = np.zeros((num_frames, 12))

        for t in range(num_frames):
            cam_curr = camera_transforms[t]
            cam_prev_inv = (
                np.eye(4) if t == 0 else np.linalg.inv(camera_transforms[t - 1])
            )

            for hand_idx in range(2):
                pose_world = hand_poses_world[t, hand_idx * 6 : (hand_idx + 1) * 6]
                pose_cam = compute_camera_relative_pose(
                    pose_world, cam_prev_inv, cam_curr
                )
                ee_pose_cam[t, hand_idx * 6 : (hand_idx + 1) * 6] = pose_cam

        return ee_pose_cam

    @staticmethod
    def _prestack_actions(ee_pose_cam: np.ndarray, chunk_size: int = 100) -> np.ndarray:
        """
        Prestack future actions into chunks.

        Args:
            ee_pose_cam: (T, 12) current poses in camera frame
            chunk_size: Number of future timesteps per chunk

        Returns:
            (T, chunk_size, 12) prestacked action chunks
        """
        num_frames = len(ee_pose_cam)
        actions = np.zeros((num_frames, chunk_size, 12))

        for t in range(num_frames):
            for offset in range(chunk_size):
                future_t = min(t + offset, num_frames - 1)
                actions[t, offset] = ee_pose_cam[future_t]

        return actions

    @staticmethod
    def _prestack_keypoints(keypoints: np.ndarray, chunk_size: int = 100) -> np.ndarray:
        """
        Prestack future keypoints into chunks.

        Args:
            keypoints: (T, 126) flattened keypoints
            chunk_size: Number of future timesteps per chunk

        Returns:
            (T, chunk_size, 126) prestacked keypoint chunks
        """
        num_frames = len(keypoints)
        keypoint_dim = keypoints.shape[1]
        actions = np.zeros((num_frames, chunk_size, keypoint_dim))

        for t in range(num_frames):
            for offset in range(chunk_size):
                future_t = min(t + offset, num_frames - 1)
                actions[t, offset] = keypoints[future_t]

        return actions

    @staticmethod
    def _extract_head_poses(camera_transforms: List[np.ndarray]) -> np.ndarray:
        """
        Extract head poses in world frame.

        Returns:
            (T, 10) - [x, y, z, yaw, pitch, roll, qx, qy, qz, qw]
        """
        num_frames = len(camera_transforms)
        head_poses = np.zeros((num_frames, 10))

        for t, T in enumerate(camera_transforms):
            head_poses[t, :3] = T[:3, 3]

            R = Rotation.from_matrix(T[:3, :3])
            yaw, pitch, roll = R.as_euler("ZYX")
            head_poses[t, 3:6] = [yaw, pitch, roll]

            quat = R.as_quat()
            head_poses[t, 6:] = quat

        return head_poses

    @staticmethod
    def _get_task_for_frame(
        frame_time: float, annotations_df: pd.DataFrame, episode_task: str
    ) -> str:
        """
        Get the task label for a specific frame based on annotations.

        Args:
            frame_time: Frame timestamp in seconds
            annotations_df: DataFrame with Labels, start_time, end_time columns
            episode_task: Fallback episode-level task label

        Returns:
            Task label string for this frame
        """
        for _, row in annotations_df.iterrows():
            if row["start_time"] <= frame_time <= row["end_time"]:
                return row["Labels"]

        return episode_task

    @staticmethod
    def define_features(
        episode_feats: dict,
        image_compressed: bool = False,
        encode_as_video: bool = True,
    ) -> Tuple[dict, dict]:
        """
        Define features dictionary for LeRobotDataset.create().

        Args:
            episode_feats: Output from process_episode()
            image_compressed: Whether to compress images
            encode_as_video: Whether to encode images as video

        Returns:
            features: Features dict with dtype, shape, names
            metadata: Additional metadata dict
        """
        sample_image = episode_feats["observations"]["images.front_img_1"][0]
        H, W, C = sample_image.shape

        features = {
            "observations.state.ee_pose_cam": {
                "dtype": "float32",
                "shape": (12,),
                "names": ["dual_hand_6dof"],
            },
            "observations.images.front_img_1": {
                "dtype": "video" if encode_as_video else "image",
                "shape": (C, H, W),
                "names": ["channel", "height", "width"],
            },
            "actions_ee_cartesian_cam": {
                "dtype": "prestacked_float32",
                "shape": (CHUNK_SIZE, 12),
                "names": ["chunk_size", "dual_hand_6dof"],
            },
            "actions_ee_keypoints_world": {
                "dtype": "prestacked_float32",
                "shape": (CHUNK_SIZE, 126),
                "names": ["chunk_size", "dual_hand_keypoints"],
            },
            "actions_head_cartesian_world": {
                "dtype": "float32",
                "shape": (10,),
                "names": ["head_pose_10d"],
            },
            "metadata.embodiment": {
                "dtype": "int32",
                "shape": (1,),
                "names": ["dim_0"],
            },
        }

        episode_meta = episode_feats["episode_meta"]
        metadata = {
            "robot_type": "MECKA_BIMANUAL",  # TODO: make dynamic based on arm
            "fps": 30,
            "episode_id": episode_meta["id"],
            "user_id": episode_meta.get("user_id"),
            "duration": episode_meta.get("duration"),
            "environment_id": episode_meta.get("environment_id"),
            "scene_id": episode_meta.get("scene_id"),
            "scene_desc": episode_meta.get("scene_desc"),
            "objects": episode_meta.get("objects", []),
        }

        return features, metadata

    @staticmethod
    def extract_episode_frames(
        episode_feats: dict, features: dict
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Convert episode features to list of frame dictionaries with torch tensors.

        Args:
            episode_feats: Output from process_episode()
            features: Features dict from define_features()

        Returns:
            List of frame dicts ready for dataset.add_frame()
        """
        num_frames = len(episode_feats["frame_index"])
        frames = []

        for t in range(num_frames):
            frame_dict = {}

            frame_dict["observations.state.ee_pose_cam"] = torch.from_numpy(
                episode_feats["observations"]["state.ee_pose_cam"][t]
            ).float()

            img = episode_feats["observations"]["images.front_img_1"][t]
            img = cv2.resize(img, (640, 360), interpolation=cv2.INTER_AREA)

            img_tensor = torch.from_numpy(img).permute(2, 0, 1)
            frame_dict["observations.images.front_img_1"] = img_tensor

            frame_dict["actions_ee_cartesian_cam"] = torch.from_numpy(
                episode_feats["actions_ee_cartesian_cam"][t]
            ).float()

            frame_dict["actions_ee_keypoints_world"] = torch.from_numpy(
                episode_feats["actions_ee_keypoints_world"][t]
            ).float()

            frame_dict["actions_head_cartesian_world"] = torch.from_numpy(
                episode_feats["actions_head_cartesian_world"][t]
            ).float()

            emb_arr = episode_feats.get("metadata.embodiment")
            emb_val = int(np.asarray(emb_arr)[t].item())
            frame_dict["metadata.embodiment"] = torch.tensor(
                [emb_val], dtype=torch.int32
            )

            frame_dict["timestamp"] = float(episode_feats["timestamp"][t])

            frames.append(frame_dict)

        return frames


class MeckaDatasetConverter:
    """Convert Mecka episodes to LeRobot dataset format."""

    def __init__(
        self,
        episode_json_path: str,
        repo_id: str,
        arm: str = "both",
        prestack: bool = True,
        video_encoding: bool = False,
        local_data_dir: Optional[Path] = None,
    ):
        """
        Initialize converter.

        Args:
            episode_json_path: Path to episode JSON
            repo_id: Dataset repo ID
            arm: Which arm data to include
            prestack: Whether to prestack actions
            video_encoding: Whether to encode images as video
        """
        self.episode_json_path = episode_json_path
        self.repo_id = repo_id
        self.arm = arm
        self.prestack = prestack
        self.video_encoding = video_encoding
        self.local_data_dir = (
            Path(local_data_dir) if local_data_dir is not None else None
        )
        self.output_dir = None
        self._mp4_path = None

        if arm == "both":
            robotype = EMBODIMENT.MECKA_BIMANUAL
        elif arm == "left":
            robotype = EMBODIMENT.MECKA_LEFT_ARM
        else:
            robotype = EMBODIMENT.MECKA_RIGHT_ARM

        self.robot_type = robotype.name

        logger.info("Processing episode to extract features...")
        self.episode_feats = MeckaExtractor.process_episode(
            episode_json_path,
            arm=arm,
            prestack=prestack,
            local_data_dir=self.local_data_dir,
        )

        self.features, self.metadata = MeckaExtractor.define_features(
            self.episode_feats, encode_as_video=video_encoding
        )

        self.dataset = None
        self.buffer = []

    def init_lerobot_dataset(self, output_dir: str):
        """
        Initialize LeRobotDataset using create() API.

        Args:
            output_dir: Root directory for dataset output
        """
        self.output_dir = output_dir
        logger.info(f"Creating LeRobotDataset at {output_dir}")
        logger.info(f"Repository ID: {self.repo_id}")

        self.dataset = LeRobotDataset.create(
            repo_id=self.repo_id,
            fps=30,
            root=output_dir,
            robot_type=self.robot_type,
            features=self.features,
            use_videos=self.video_encoding,
        )

        logger.info("LeRobotDataset initialized successfully")

    def extract_episode(self, task_description: str):
        """
        Extract episode frames and add to dataset in sub-episodes of EPISODE_LENGTH.

        Args:
            task_description: Task description for this episode
        """
        logger.info(f"Extracting episode with task: {task_description}")

        self.task_description = task_description

        frames = MeckaExtractor.extract_episode_frames(
            self.episode_feats, self.features
        )

        logger.info(
            f"Processing {len(frames)} frames in sub-episodes of {EPISODE_LENGTH}"
        )

        mp4_frames = []
        sub_episode_idx = 0
        for t, frame in enumerate(frames):
            if (
                self._mp4_path is not None
                and "observations.images.front_img_1" in frame
            ):
                mp4_frames.append(frame["observations.images.front_img_1"])
            self.buffer.append(frame)

            if len(self.buffer) == EPISODE_LENGTH:
                logger.info(
                    f"Saving sub-episode {sub_episode_idx} ({len(self.buffer)} frames)"
                )

                for i, f in enumerate(self.buffer):
                    f["timestamp"] = float(i / 30)
                    self.dataset.add_frame(f)

                self.dataset.save_episode(task=self.task_description)

                self._save_annotations(sub_episode_idx)

                self.buffer.clear()
                sub_episode_idx += 1

        if len(self.buffer) > 0:
            logger.info(
                f"Saving final sub-episode {sub_episode_idx} ({len(self.buffer)} frames)"
            )

            for i, f in enumerate(self.buffer):
                f["timestamp"] = float(i / 30)
                self.dataset.add_frame(f)

            self.dataset.save_episode(task=self.task_description)
            self._save_annotations(sub_episode_idx)

            self.buffer.clear()

        if self._mp4_path is not None and mp4_frames:
            episode_id = self.episode_feats["episode_meta"].get("id", "unknown")
            self.save_preview_mp4(
                mp4_frames, self._mp4_path / f"{episode_id}_video.mp4"
            )

    def _save_annotations(self, sub_episode_idx: int):
        """
        Save annotations CSV for sub-episode.

        Args:
            sub_episode_idx: Index of sub-episode
        """
        annotations_df = self.episode_feats["annotations"]

        start_frame = sub_episode_idx * EPISODE_LENGTH
        end_frame = start_frame + EPISODE_LENGTH

        timestamps = self.episode_feats["timestamp"][start_frame:end_frame]
        if len(timestamps) == 0:
            return

        start_time_s = timestamps[0]
        end_time_s = timestamps[-1]

        filtered_annotations = annotations_df[
            (annotations_df["end_time"] >= start_time_s)
            & (annotations_df["start_time"] <= end_time_s)
        ].copy()

        filtered_annotations["start_time"] = (
            filtered_annotations["start_time"] - start_time_s
        ).clip(lower=0)
        filtered_annotations["end_time"] = (
            filtered_annotations["end_time"] - start_time_s
        ).clip(upper=end_time_s - start_time_s)

        annotations_path = (
            Path(self.output_dir)
            / "annotations"
            / f"episode_{sub_episode_idx:06d}_annotations.csv"
        )
        annotations_path.parent.mkdir(parents=True, exist_ok=True)
        filtered_annotations.to_csv(annotations_path, index=False)
        logger.info(f"Saved annotations: {annotations_path}")

    def save_preview_mp4(self, image_frames: list, output_path: Path, fps: int = 30):
        """Save a half-resolution H.264 MP4 preview."""
        import subprocess

        import torch.nn.functional as F

        if not image_frames:
            return

        C, H, W = image_frames[0].shape
        outW, outH = (W // 2) & ~1, (H // 2) & ~1
        if outW <= 0 or outH <= 0:
            raise ValueError(f"Invalid output size: {outW}x{outH}")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        rgb_frames = []
        for chw in image_frames:
            t = chw.detach().cpu()
            if t.dtype != torch.uint8:
                t = t.to(torch.uint8)
            if t.shape[0] == 1:
                t = t.repeat(3, 1, 1)
            t_resized = (
                F.interpolate(
                    t.unsqueeze(0).float(),
                    size=(outH, outW),
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze(0)
                .to(torch.uint8)
            )
            rgb_frames.append(t_resized.permute(1, 2, 0).contiguous())

        video_tensor = torch.stack(rgb_frames, dim=0)

        try:
            from torchvision.io import write_video

            write_video(
                filename=str(output_path),
                video_array=video_tensor,
                fps=float(fps),
                video_codec="libx264",
                options={"crf": "23", "preset": "veryfast"},
            )
            logger.info(f"Saved MP4: {output_path}")
            return
        except Exception as e:
            logger.warning(f"torchvision failed ({e}), trying ffmpeg...")

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("Neither torchvision nor ffmpeg available")

        cmd = [
            ffmpeg,
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{outW}x{outH}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-profile:v",
            "baseline",
            "-level",
            "3.0",
            "-movflags",
            "+faststart",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            str(output_path),
        ]

        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        for hwc in rgb_frames:
            proc.stdin.write(hwc.numpy()[..., ::-1].tobytes())
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed: {proc.stderr.read().decode()}")
        logger.info(f"Saved MP4: {output_path}")

    def consolidate(self):
        """Consolidate dataset and update metadata."""
        logger.info("Consolidating dataset...")
        self.dataset.consolidate()

        self._add_mecka_metadata()

        logger.info("Dataset consolidation complete")

    def _add_mecka_metadata(self):
        """Add Mecka-specific metadata to info.json."""
        info_path = Path(self.output_dir) / "meta" / "info.json"

        if not info_path.exists():
            logger.warning(f"info.json not found at {info_path}")
            return

        with open(info_path, "r") as f:
            info = json.load(f)

        episode_meta = self.episode_feats["episode_meta"]
        info["mecka"] = {
            "episode_id": episode_meta["id"],
            "user_id": episode_meta.get("user_id"),
            "duration": episode_meta.get("duration"),
            "environment_id": episode_meta.get("environment_id"),
            "scene_id": episode_meta.get("scene_id"),
            "scene_desc": episode_meta.get("scene_desc"),
            "objects": episode_meta.get("objects", []),
            "intrinsics": episode_meta.get("intrinsics", {}),
        }

        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)

        logger.info("Added Mecka metadata to info.json")


def main():
    """Main conversion function."""
    parser = argparse.ArgumentParser(description="Convert Mecka RL2 to LeRobot format")
    parser.add_argument(
        "--episode-json", required=True, help="Path to episode JSON file"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Output directory for dataset"
    )
    parser.add_argument("--repo-id", default="mecka/demo", help="Dataset repo ID")
    parser.add_argument(
        "--arm",
        default="both",
        choices=["left", "right", "both"],
        help="Which arm(s) to include",
    )
    parser.add_argument(
        "--no-prestack", action="store_true", help="Disable action prestacking"
    )
    parser.add_argument(
        "--video-encoding",
        action="store_true",
        help="Encode images as video. Default is to embed in parquet.",
    )
    parser.add_argument(
        "--local-data-dir",
        type=str,
        default=None,
        help="Path to directory containing pre-downloaded episode files (video.mp4 or <id>_video.mp4, hands.csv, egomotion.txt, frames.csv, annotations.csv). If set, downloads are skipped.",
    )
    parser.add_argument(
        "--save-mp4", action="store_true", help="Save half-res H.264 preview MP4"
    )

    args = parser.parse_args()

    output_path = Path(args.output_dir)
    if output_path.exists():
        logger.warning(f"Output directory {args.output_dir} exists, removing...")
        shutil.rmtree(output_path)

    try:
        converter = MeckaDatasetConverter(
            episode_json_path=args.episode_json,
            repo_id=args.repo_id,
            arm=args.arm,
            prestack=not args.no_prestack,
            video_encoding=args.video_encoding,
            local_data_dir=args.local_data_dir,
        )
        if args.save_mp4:
            converter._mp4_path = Path(args.output_dir)

        converter.init_lerobot_dataset(args.output_dir)

        task = converter.episode_feats["episode_meta"].get("task", "unknown_task")
        converter.extract_episode(task_description=task)

        converter.consolidate()

        logger.info("✅ Conversion complete!")
        logger.info(f"Dataset saved to: {args.output_dir}")

    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
