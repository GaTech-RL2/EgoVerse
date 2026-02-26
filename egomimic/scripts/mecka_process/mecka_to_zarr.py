"""
Convert Mecka RL2 episode data to Zarr format for use with EgoMimic/LeRobot pipelines.


This module downloads or loads Mecka episode assets (video, hands, egomotion, frames,
annotations), extracts features (hand poses, head poses, keypoints), and writes
them to a Zarr dataset compatible with the ZarrWriter interface.
"""

import argparse
import json
import logging
import shutil
import subprocess
import time
import traceback
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from egomimic.rldb.embodiment import EMBODIMENT
from egomimic.rldb.zarr.zarr_writer import ZarrWriter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_with_retry(
    url: str, dest_path: Union[Path, str], max_retries: int = 5
) -> None:
    """
    Download a file from URL to destination with retry logic for unreliable networks.


    Skips download if the file already exists. Uses curl with resume support (-C -)
    and retry flags. On failure, removes partial file and retries after 3 seconds.


    Args:
        url: Source URL to download from.
        dest_path: Local filesystem path to save the file.
        max_retries: Maximum number of download attempts (default 5).


    Raises:
        Exception: If download fails after all retries.
    """
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


def pose_to_transform(pose: np.ndarray) -> np.ndarray:
    """
    Convert 7DOF pose to a 4x4 homogeneous transform matrix.

    Args:
        pose: Array of shape (7,) with [x, y, z, qw, qx, qy, qz] (quat in WXYZ).

    Returns:
        Array of shape (4, 4) representing SE(3) transform (rotation + translation).
    """
    x, y, z, qw, qx, qy, qz = pose
    # SciPy from_quat expects (x, y, z, w); pose[3:7] is (w, x, y, z) = WXYZ
    rotation = Rotation.from_quat([qx, qy, qz, qw])
    T = np.eye(4)
    T[:3, :3] = rotation.as_matrix()
    T[:3, 3] = [x, y, z]
    return T


def extract_mecka_metadata(
    episode_meta: dict,
    data_dir: Optional[Path] = None,
) -> dict:
    """
    Extract Mecka metadata dict from episode JSON for Zarr metadata_override.

    Args:
        episode_meta: Raw episode dict from RL2 JSON (id, user_id, duration, etc.).
        data_dir: Optional directory containing intrinsics.json to load.

    Returns:
        Dict with episode_id, user_id, duration, environment_id, scene_id,
        scene_desc, objects, intrinsics (from file or {}).
    """
    intrinsics: dict = {}
    if data_dir is not None:
        intrinsics_path = data_dir / "intrinsics.json"
        if intrinsics_path.exists():
            try:
                with open(intrinsics_path, "r") as f:
                    intrinsics = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load intrinsics from {intrinsics_path}: {e}")

    return {
        "episode_id": episode_meta.get("id"),
        "user_id": episode_meta.get("user_id"),
        "duration": episode_meta.get("duration"),
        "environment_id": episode_meta.get("environment_id"),
        "scene_id": episode_meta.get("scene_id"),
        "scene_desc": episode_meta.get("scene_desc"),
        "objects": episode_meta.get("objects", []),
        "intrinsics": intrinsics if intrinsics else episode_meta.get("intrinsics", {}),
    }


def transform_to_pose(T: np.ndarray) -> np.ndarray:
    """
    Convert 4x4 homogeneous transform matrix to 7DOF pose in WXYZ format.

    Args:
        T: Array of shape (4, 4) representing SE(3) transform.

    Returns:
        Array of shape (7,) with [x, y, z, qw, qx, qy, qz] (quat in WXYZ).
    """
    pos = T[:3, 3]
    rotation = Rotation.from_matrix(T[:3, :3])
    quat_xyzw = rotation.as_quat()  # SciPy returns (x, y, z, w)
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    return np.concatenate([pos, quat_wxyz])


def compute_hand_pose_xyzquat(keypoints: np.ndarray, hand_index: int) -> np.ndarray:
    """
    Compute 7DOF pose from 21 hand keypoints in camera frame.

    Uses wrist position (keypoint 0) and constructs a right-handed frame from:
    - Forward: direction from wrist to middle finger base (keypoint 9)
    - Up: cross product of thumb (keypoint 5) and pinky (keypoint 17) directions
    - Right: cross(forward, up), then orthonormalized

    Args:
        keypoints: Array of shape (21, 3) with 3D positions of hand landmarks.

    Returns:
        Tuple of (pose_7dof, wrist_7dof), each shape (7,) with [x, y, z, qw, qx, qy, qz] (WXYZ).
        Returns (zeros, zeros) or (position+identity_quat, copy) in degenerate cases.
    """
    # Identity quaternion in WXYZ for degenerate cases
    quat_wxyz_identity = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    rot_left = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    rot_right = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])

    if np.allclose(keypoints, 0):
        fallback = np.concatenate([np.zeros(3), quat_wxyz_identity])
        return fallback, fallback.copy()

    # Use wrist as origin; keypoint 0 is wrist in MediaPipe-style hand model
    centroid = np.mean(
        [keypoints[0], keypoints[17], keypoints[13], keypoints[9], keypoints[5]], axis=0
    )
    position = centroid
    wrist = keypoints[0]
    middle_base = keypoints[9]  # Base of middle finger

    # Forward axis: wrist → middle finger base
    forward = middle_base - wrist
    if np.linalg.norm(forward) < 1e-6:
        logger.warning(
            "Forward direction norm too small, returning position + identity quat"
        )
        fallback = np.concatenate([position, quat_wxyz_identity])
        return fallback, fallback.copy()
    forward = forward / np.linalg.norm(forward)

    # Up axis: cross product of thumb→wrist and pinky→wrist
    thumb_dir = keypoints[5] - wrist
    pinky_dir = keypoints[17] - wrist
    up = np.cross(thumb_dir, pinky_dir)

    if np.linalg.norm(up) < 1e-6:
        logger.warning(
            "Up direction norm too small, returning position + identity quat"
        )
        fallback = np.concatenate([position, quat_wxyz_identity])
        return fallback, fallback.copy()
    up = up / np.linalg.norm(up)

    # Right axis: cross(forward, up), then re-orthonormalize
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    right = right * -1  # Flip for right-handed frame

    rot_matrix = np.column_stack([forward, right, up])

    if hand_index == 0:
        rot_matrix = rot_matrix @ rot_left
    else:
        rot_matrix = rot_matrix @ rot_right

    quat_xyzw = Rotation.from_matrix(rot_matrix).as_quat()  # SciPy returns (x, y, z, w)
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    return (
        np.concatenate([position, quat_wxyz]),
        np.concatenate([wrist, quat_wxyz]),
    )


class MeckaExtractor:
    """
    Extract observation features from Mecka RL2 episode data.


    Handles downloading/loading episode assets, parsing hand keypoints and egomotion,
    and producing a feature dict compatible with the ZarrWriter (hand poses, head poses,
    keypoints, images). Supports both remote download and local data directories.
    """

    TAGS = ["mecka", "robotics", "human_hands"]

    @staticmethod
    def process_episode(
        episode_json_path: str,
        arm: str = "both",
        local_data_dir: Optional[Path] = None,
    ) -> Tuple[dict, pd.DataFrame, dict, dict]:
        """
        Load and process a single Mecka episode into feature tensors and metadata.


        Reads episode JSON, fetches or loads hands.csv, egomotion.txt, frames.csv,
        annotations.csv, and video, then extracts hand poses (world frame), head poses,
        keypoints, and downsampled RGB frames.


        Args:
            episode_json_path: Path to episode JSON (RL2 format with 'urls' and 'id').
            arm: Which hand(s) to include ("left", "right", or "both").
            local_data_dir: If set, load from this directory instead of downloading.
                Expects video.mp4 or <id>_video.mp4, hands.csv, egomotion.txt,
                frames.csv, annotations.csv.


        Returns:
            Tuple of (episode_feats, annotations_df, episode_meta, mecka_metadata):
                - episode_feats: Dict with keys like right.obs_ee_pose, left.obs_ee_pose,
                  right.obs_keypoints, left.obs_keypoints, obs_head_pose, images.front_1, etc.
                - annotations_df: DataFrame with annotation labels and time ranges.
                - episode_meta: Raw episode metadata from JSON.
                - mecka_metadata: Dict for Zarr metadata_override (episode_id, user_id, etc.).
        """
        logger.info(f"Processing episode: {episode_json_path}")

        with open(episode_json_path, "r") as f:
            data = json.load(f)
            episode_meta = data[0] if isinstance(data, list) else data

        local_data_dir = Path(local_data_dir) if local_data_dir is not None else None

        try:
            episode_id = episode_meta["id"]
            # Download from URLs or load from local directory
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
                # Prefer video in episode directory (e.g. <id>_video.mp4)
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
                intrinsics_path = temp_dir / "intrinsics.json"
                if "intrinsics" in episode_meta.get("urls", {}):
                    try:
                        download_with_retry(
                            episode_meta["urls"]["intrinsics"], intrinsics_path
                        )
                    except Exception as e:
                        logger.warning(f"Could not download intrinsics: {e}")
                data_dir = temp_dir
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
                data_dir = local_data_dir

            hands_df = pd.read_csv(hands_path)
            egomotion = np.loadtxt(egomotion_path)
            frames_df = pd.read_csv(frames_path)
            annotations_df = pd.read_csv(annotations_path)

            # Per-frame camera poses (world-to-camera) for hand transform
            camera_transforms = MeckaExtractor._extract_camera_transforms(egomotion)
            hand_poses_world, hand_keypoints_world, wrist_poses_world = (
                MeckaExtractor._extract_hand_data(
                    hands_df, frames_df, arm, camera_transforms
                )
            )

            images = MeckaExtractor._extract_video_frames(video_path, len(frames_df))

            # Sync to shortest stream to ensure aligned observations
            num_frames = min(len(images), len(frames_df), len(egomotion))
            logger.info(
                f"Syncing to {num_frames} frames (video={len(images)}, frames_df={len(frames_df)}, egomotion={len(egomotion)})"
            )

            images = images[:num_frames]

            # Downsample images to 640x360 (W x H) for storage and training
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
            hand_poses_world = hand_poses_world[:num_frames]
            wrist_poses_world = wrist_poses_world[:num_frames]
            hand_keypoints_world = hand_keypoints_world[:num_frames]
            actions_head_cartesian_world = MeckaExtractor._extract_head_poses(egomotion)
            # Flatten 21×3 keypoints to 63 per hand for Zarr schema
            # hand_index 0=left, 1=right
            right_keypoints = hand_keypoints_world[:, 1, :, :].reshape(num_frames, 63)
            left_keypoints = hand_keypoints_world[:, 0, :, :].reshape(num_frames, 63)

            # RL2/Mecka hands.csv: hand_index 0=left, 1=right (per mecka_visualisation)
            episode_feats = {
                "right.obs_ee_pose": hand_poses_world[:, 7:14],
                "left.obs_ee_pose": hand_poses_world[:, :7],
                "right.obs_keypoints": right_keypoints,
                "left.obs_keypoints": left_keypoints,
                "obs_head_pose": actions_head_cartesian_world,
                "images.front_1": images,
                "right.obs_wrist_pose": wrist_poses_world[:, 7:14],
                "left.obs_wrist_pose": wrist_poses_world[:, :7],
            }

            mecka_metadata = extract_mecka_metadata(episode_meta, data_dir)

            logger.info(f"Extracted {num_frames} frames")
            return episode_feats, annotations_df, episode_meta, mecka_metadata

        finally:
            pass

    @staticmethod
    def _extract_camera_transforms(egomotion: np.ndarray) -> List[np.ndarray]:
        """
        Parse egomotion array into per-frame camera poses in world frame.

        Egomotion format: each row has columns [?, X, Y, Z, ?, ?, ?, qx, qy, qz, qw].
        Quaternion is XYZW (SciPy); used internally, not exposed.


        Args:
            egomotion: Array of shape (T, 11+), one row per frame.


        Returns:
            List of T 4x4 homogeneous transforms (wTc), one per frame.
        """
        transforms = []

        for i in range(len(egomotion)):
            # Translation in world frame (columns 1-3)
            t = egomotion[i, 1:4]  # X, Y, Z

            # Quaternion (columns 7-10): SciPy expects (x, y, z, w)
            q = egomotion[i, 7:11]

            # Convert quaternion -> rotation matrix
            R_mat = Rotation.from_quat(q).as_matrix()

            # Build homogeneous SE(3)
            T = np.eye(4)
            T[:3, :3] = R_mat
            T[:3, 3] = t

            transforms.append(T)

        return transforms

    @staticmethod
    def _extract_hand_data(
        hands_df: pd.DataFrame,
        frames_df: pd.DataFrame,
        arm: str,
        camera_transforms: List[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract hand poses and keypoints from hands CSV and transform to world frame.


        Hands CSV has columns: frame, hand_index (0=left, 1=right), landmark_index,
        world_x, world_y, world_z. Hand poses are computed from keypoints in camera
        frame via compute_hand_pose_xyzquat, then transformed to world using wTc.


        Args:
            hands_df: DataFrame with hand landmark positions per frame.
            frames_df: DataFrame defining frame count and sync (used for shape).
            arm: Unused; kept for API compatibility.
            camera_transforms: List of wTc 4x4 transforms per frame.


        Returns:
            Tuple of (hand_poses_world, hand_keypoints_world):
                - hand_poses_world: (T, 14) [left_7dof, right_7dof] xyz+quat(WXYZ) in world frame.
                - hand_keypoints_world: (T, 2, 21, 3) [left_21kp, right_21kp] in world.
        """
        num_frames = len(frames_df)
        hand_poses = np.zeros((num_frames, 14))
        hand_keypoints = np.zeros((num_frames, 2, 21, 3))
        wrist_poses = np.zeros((num_frames, 14))

        for frame_idx in range(num_frames):
            for hand_index in [0, 1]:
                hand_data = hands_df[
                    (hands_df["frame"] == frame_idx)
                    & (hands_df["hand_index"] == hand_index)
                ].sort_values("landmark_index")

                if len(hand_data) == 21:
                    kp = hand_data[
                        ["world_x", "world_y", "world_z"]
                    ].values  # (21, 3) in camera frame
                    wTc = camera_transforms[frame_idx]

                    # Transform keypoints from camera frame to world frame (same as hand poses)
                    kp_h = np.concatenate([kp, np.ones((21, 1))], axis=1)  # (21, 4)
                    kp_world = (wTc @ kp_h.T).T[:, :3]  # (21, 3)
                    hand_keypoints[frame_idx, hand_index] = kp_world

                    # Hand pose in camera frame -> transform to world via wTc
                    pose_xyzquat, wrist_xyzquat = compute_hand_pose_xyzquat(
                        kp, hand_index
                    )

                    T_hand_cam = pose_to_transform(pose_xyzquat)
                    T_wrist_cam = pose_to_transform(wrist_xyzquat)

                    T_hand_world = wTc @ T_hand_cam
                    pose_xyzquat = transform_to_pose(T_hand_world)

                    T_wrist_world = wTc @ T_wrist_cam
                    wrist_xyzquat = transform_to_pose(T_wrist_world)

                    hand_poses[frame_idx, hand_index * 7 : (hand_index + 1) * 7] = (
                        pose_xyzquat
                    )
                    wrist_poses[frame_idx, hand_index * 7 : (hand_index + 1) * 7] = (
                        wrist_xyzquat
                    )

        return hand_poses, hand_keypoints, wrist_poses

    @staticmethod
    def _extract_video_frames(video_path: Path, num_frames: int) -> np.ndarray:
        """
        Decode video file and extract the first num_frames as RGB arrays.


        Args:
            video_path: Path to MP4 or other OpenCV-supported video.
            num_frames: Number of frames to read (stops earlier if video ends).


        Returns:
            Array of shape (T, H, W, 3) in RGB, uint8. T may be less than num_frames.
        """
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
    def _extract_head_poses(egomotion: np.ndarray) -> np.ndarray:
        """
        Extract head (camera) poses from egomotion with axis remap for pipeline convention.


        Applies a basis change so output matches expected coordinate frame:
        (x, y, z) -> (-y, -z, x). Output is position + quaternion for each frame.


        Args:
            egomotion: Array of shape (T, 11+), one row per frame.


        Returns:
            Array of shape (T, 7) with [x, y, z, qw, qx, qy, qz] (quat in WXYZ) per frame.
        """
        # Frame remap: (x, y, z) -> (-y, -z, x) to match expected head pose convention
        R_fix = np.array(
            [
                [0, -1, 0],
                [0, 0, -1],
                [1, 0, 0],
            ],
            dtype=np.float64,
        )

        T_fix = np.eye(4, dtype=np.float64)
        T_fix[:3, :3] = R_fix

        # Construct inverse SE(3) transform matrix for the rotation remap
        R_fix_inv = R_fix.T
        T_fix_inv = np.eye(4, dtype=np.float64)
        T_fix_inv[:3, :3] = R_fix_inv

        adjusted_head_poses = []
        for i in range(len(egomotion)):
            # Build camera pose in world frame: wTc
            t = np.asarray(egomotion[i, 1:4], dtype=np.float64)  # world translation
            q = np.asarray(egomotion[i, 7:11], dtype=np.float64)  # quat (x, y, z, w)
            R_wc = Rotation.from_quat(q).as_matrix()

            # Build SE(3) transform matrix from translation and rotation
            T_wc = np.eye(4, dtype=np.float64)
            T_wc[:3, :3] = R_wc
            T_wc[:3, 3] = t

            # Apply axis remap: wTc' = wTc @ inv(T_fix)
            T_wc_adj = T_wc @ T_fix_inv

            # Output head pose as [x, y, z, qw, qx, qy, qz] (WXYZ)
            xyz_adj = T_wc_adj[:3, 3]
            R_wc_adj = Rotation.from_matrix(T_wc_adj[:3, :3])
            quat_xyzw = R_wc_adj.as_quat()  # SciPy returns (x, y, z, w)
            quat_wxyz = np.array(
                [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
            )
            adjusted_head_poses.append(np.concatenate([xyz_adj, quat_wxyz]))

        return np.asarray(adjusted_head_poses)


class MeckaDatasetConverter:
    """Convert Mecka episodes to Zarr dataset format using ZarrWriter."""

    def __init__(
        self,
        episode_json_path: str,
        output_dir: str,
        repo_id: str,
        arm: str = "both",
        local_data_dir: Optional[Path] = None,
        task_description: str = "",
    ):
        """
        Initialize the Mecka-to-Zarr converter and run feature extraction.


        Loads episode via MeckaExtractor.process_episode and stores results for
        subsequent extract_episode() call, which writes the Zarr dataset.


        Args:
            episode_json_path: Path to episode JSON (RL2 format).
            output_dir: Directory to write Zarr store and preview MP4.
            repo_id: Dataset repository identifier (e.g. "mecka/demo").
            arm: Which hand(s) to include ("left", "right", or "both").
            local_data_dir: Optional path to pre-downloaded episode files; skips download.
            task_description: Optional task label for annotations.
        """
        self.episode_json_path = episode_json_path
        self.repo_id = repo_id
        self.arm = arm
        self.local_data_dir = (
            Path(local_data_dir) if local_data_dir is not None else None
        )
        self.output_dir = Path(output_dir)
        self.fps = 30
        self.task_description = task_description

        if arm == "both":
            emb = EMBODIMENT.MECKA_BIMANUAL
        elif arm == "left":
            emb = EMBODIMENT.MECKA_LEFT_ARM
        else:
            emb = EMBODIMENT.MECKA_RIGHT_ARM

        self.embodiment = emb.name

        logger.info("Processing episode to extract features...")
        (
            self.episode_feats,
            self.annotations_df,
            self.episode_meta,
            self.mecka_metadata,
        ) = MeckaExtractor.process_episode(
            episode_json_path,
            arm=arm,
            local_data_dir=self.local_data_dir,
        )

    def extract_episode(self) -> None:
        """
        Write extracted episode features to Zarr and save a preview MP4.


        Splits episode_feats into numeric_data and image_data, parses annotations
        from annotations_df (label, start_time, end_time), and calls ZarrWriter.
        Generates a half-resolution H.264 preview video alongside the Zarr store.
        """
        numeric_data = {}
        image_data = {}
        annotations = []
        image_frames = np.empty((0, 360, 640, 3), dtype=np.uint8)

        # Split episode_feats into numeric vs image arrays for ZarrWriter
        for key, value in self.episode_feats.items():
            if "images" in key:
                image_data[key] = value
                image_frames = np.vstack([image_frames, value])
            else:
                numeric_data[key] = value

        # Parse annotations: label, start_time, end_time -> (label, start_idx, end_idx)
        fps = 30  # Mecka/RL2 video fps
        for _, row in self.annotations_df.iterrows():
            label = row.get("label", row.get("Labels", ""))
            label = label.replace("_", " ")
            start_idx = int(row["start_time"] * fps)
            end_idx = int(row["end_time"] * fps)
            annotations.append((label, start_idx, end_idx))

        episode_zarr_path = self.output_dir / f"{self.episode_meta['id']}.zarr"
        ZarrWriter.create_and_write(
            episode_path=episode_zarr_path,
            numeric_data=numeric_data,
            image_data=image_data,
            annotations=annotations,
            fps=self.fps,
            embodiment=self.embodiment,
            task=self.task_description,
            enable_sharding=False,  # Avoid zarr sharding codec ZeroDivisionError with certain array shapes
            metadata_override=self.mecka_metadata,
        )
        mp4_path = self.output_dir / f"{self.episode_meta['id']}.mp4"
        self.save_preview_mp4(image_frames, mp4_path)

    def save_preview_mp4(
        self, image_frames: np.ndarray, output_path: Path, fps: int = 30
    ) -> None:
        """
        Save a half-resolution H.264 MP4 preview of the episode.


        Uses torchvision.write_video when available, otherwise falls back to ffmpeg.
        Output is resized to half resolution with even dimensions for H.264 compatibility.


        Args:
            image_frames: Numpy array of shape (N, H, W, 3) in RGB, uint8.
            output_path: Path for the output MP4 file.
            fps: Frames per second (default 30).


        Raises:
            ValueError: If channels != 3 or output dimensions invalid.
            RuntimeError: If neither torchvision nor ffmpeg is available.
        """
        if image_frames.shape[0] == 0:
            return

        N, H, W, C = image_frames.shape
        if C != 3:
            raise ValueError(f"Expected 3 channels, got {C}")

        # Half resolution; ensure dimensions are even for H.264
        outW, outH = (W // 2) & ~1, (H // 2) & ~1
        if outW <= 0 or outH <= 0:
            raise ValueError(f"Invalid output size: {outW}x{outH}")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Resize frames to half resolution (NHWC numpy)
        if image_frames.dtype != np.uint8:
            if image_frames.max() <= 1.0:
                image_frames = (image_frames * 255).astype(np.uint8)
            else:
                image_frames = image_frames.astype(np.uint8)

        resized = np.array(
            [
                cv2.resize(f, (outW, outH), interpolation=cv2.INTER_AREA)
                for f in image_frames
            ]
        )

        # Try torchvision first (expects THWC, uint8)
        try:
            import torch
            from torchvision.io import write_video

            video_tensor = torch.from_numpy(resized)  # (N, H, W, C)
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

        # Fallback: ffmpeg with raw RGB input piped via stdin
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
            "rgb24",
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
        for frame in resized:
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed: {proc.stderr.read().decode()}")
        logger.info(f"Saved MP4: {output_path}")


def main() -> None:
    """
    CLI entrypoint for Mecka RL2 to Zarr conversion.


    Parses episode JSON path, output directory, arm selection, and optional
    local data directory. Creates MeckaDatasetConverter and runs extract_episode.
    Removes existing output directory before conversion. Exits with traceback on error.
    """
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
        "--video-encoding", action="store_true", help="Encode images as video in zarr"
    )
    parser.add_argument(
        "--local-data-dir",
        type=str,
        default=None,
        help="Path to directory containing pre-downloaded episode files (video.mp4 or <id>_video.mp4, hands.csv, egomotion.txt, frames.csv, annotations.csv). If set, downloads are skipped.",
    )

    args = parser.parse_args()

    output_path = Path(args.output_dir)
    if output_path.exists():
        logger.warning(f"Output directory {args.output_dir} exists, removing...")
        shutil.rmtree(output_path)

    try:
        converter = MeckaDatasetConverter(
            episode_json_path=args.episode_json,
            output_dir=args.output_dir,
            repo_id=args.repo_id,
            arm=args.arm,
            local_data_dir=args.local_data_dir,
        )

        converter.extract_episode()

        logger.info("✅ Conversion complete!")
        logger.info(f"Dataset saved to: {args.output_dir}")

    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
