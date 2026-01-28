#!/usr/bin/env python3
"""
Convert Mecka RL2 dataset to Zarr format.

This script creates a Zarr-based dataset that is compatible with the ZarrDataset
class and can be used as a drop-in replacement for LeRobot datasets.

Features:
- JPEG-compressed images (quality 85)
- Prestacked actions (T, 100, 12) for fast training I/O
- Self-contained annotations inside Zarr
- Compatible with existing DataSchematic and training pipeline

Usage:
    python mecka_to_zarr.py --episode-json /path/to/episode.json --output-dir /path/to/output.zarr
"""

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import numcodecs
import zarr

# Import the existing MeckaExtractor (reuse extraction logic)
from egomimic.scripts.mecka_process.mecka_to_lerobot import (
    MeckaExtractor,
    EPISODE_LENGTH,
    CHUNK_SIZE,
)
# Import from zarr_utils which is self-contained (no lerobot dependency)
from egomimic.rldb.zarr_utils import SimplejpegCodec, EMBODIMENT, get_embodiment_id

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ZarrDatasetWriter:
    """
    Writer for Zarr-based datasets.

    Creates the following structure:
    dataset_root.zarr/
    ├── .zattrs                           # Dataset-level metadata
    ├── episodes/
    │   └── ep_{hash}/                    # Per-episode group
    │       ├── .zattrs                   # Episode metadata
    │       ├── rgb/front_img_1/          # (T, H, W, 3), JPEG compressed
    │       ├── state/ee_pose_cam/        # (T, 12)
    │       ├── actions/ee_cartesian_cam/ # (T, 100, 12) - PRESTACKED
    │       ├── keypoints/hand_keypoints_world/ # (T, 100, 126) - PRESTACKED
    │       ├── head/pose_world/          # (T, 10)
    │       └── annotations/              # label_id, start_frame, end_frame
    └── meta/
        ├── info.json
        ├── stats.json
        └── episodes.jsonl
    """

    def __init__(
        self,
        output_root: Path,
        robot_type: str = "MECKA_BIMANUAL",
        fps: int = 30,
        jpeg_quality: int = 85,
        overwrite: bool = False,
    ):
        """
        Initialize ZarrDatasetWriter.

        Args:
            output_root: Path to output zarr store
            robot_type: Robot type string (e.g., "MECKA_BIMANUAL")
            fps: Frames per second
            jpeg_quality: JPEG compression quality (0-100)
            overwrite: Whether to overwrite existing store
        """
        self.output_root = Path(output_root)
        self.robot_type = robot_type
        self.fps = fps
        self.jpeg_quality = jpeg_quality

        if self.output_root.exists():
            if overwrite:
                logger.warning(f"Removing existing store: {self.output_root}")
                shutil.rmtree(self.output_root)
            else:
                raise ValueError(f"Output exists: {self.output_root}. Use --overwrite.")

        # Create zarr store
        self.store = zarr.open(str(self.output_root), mode="w")

        # Initialize dataset-level metadata
        self.store.attrs["fps"] = fps
        self.store.attrs["robot_type"] = robot_type
        self.store.attrs["total_episodes"] = 0
        self.store.attrs["total_frames"] = 0

        # Create groups
        self.store.create_group("episodes")

        # Create meta directory
        (self.output_root / "meta").mkdir(parents=True, exist_ok=True)

        # Episode tracking
        self.episodes_meta = []
        self.stats_accumulator = StatsAccumulator()
        self.image_shape = None  # Track actual image dimensions (H, W)

        logger.info(f"Created Zarr store at {self.output_root}")

    def add_episode(
        self,
        episode_feats: dict,
        episode_hash: str,
        task_description: str = "",
    ) -> str:
        """
        Add an episode to the Zarr store.

        Args:
            episode_feats: Output from MeckaExtractor.process_episode()
            episode_hash: Unique hash for this episode
            task_description: Task description string

        Returns:
            Episode hash
        """
        logger.info(f"Adding episode: {episode_hash}")

        # Create episode group
        ep_group = self.store["episodes"].create_group(episode_hash)

        # Get episode metadata
        episode_meta = episode_feats.get("episode_meta", {})
        num_frames = len(episode_feats["frame_index"])

        # Episode attributes
        ep_group.attrs["episode_id"] = episode_meta.get("id", episode_hash)
        ep_group.attrs["hash"] = episode_hash
        ep_group.attrs["length"] = num_frames
        ep_group.attrs["fps"] = self.fps
        ep_group.attrs["embodiment_id"] = get_embodiment_id(self.robot_type)
        ep_group.attrs["task"] = task_description
        ep_group.attrs["task_id"] = 0  # Can be updated later

        # Additional metadata from episode_meta
        for key in ["user_id", "duration", "environment_id", "scene_id", "scene_desc", "objects"]:
            if key in episode_meta:
                ep_group.attrs[key] = episode_meta[key]

        # Write arrays
        self._write_images(ep_group, episode_feats)
        self._write_state(ep_group, episode_feats)
        self._write_actions(ep_group, episode_feats)
        self._write_keypoints(ep_group, episode_feats)
        self._write_head_pose(ep_group, episode_feats)
        self._write_annotations(ep_group, episode_feats)

        # Track episode metadata
        self.episodes_meta.append({
            "hash": episode_hash,
            "episode_id": episode_meta.get("id", episode_hash),
            "length": num_frames,
            "task": task_description,
            "embodiment_id": get_embodiment_id(self.robot_type),
        })

        # Update stats
        self._accumulate_stats(episode_feats)

        # Update dataset-level counters
        self.store.attrs["total_episodes"] = len(self.episodes_meta)
        self.store.attrs["total_frames"] = sum(ep["length"] for ep in self.episodes_meta)

        logger.info(f"Episode {episode_hash}: {num_frames} frames added")
        return episode_hash

    def _write_images(self, ep_group: zarr.Group, episode_feats: dict):
        """Write JPEG-compressed images."""
        images = episode_feats["observations"]["images.front_img_1"]
        T, H, W, C = images.shape

        # Track image dimensions for metadata
        if self.image_shape is None:
            self.image_shape = (H, W)

        # Create array with custom simplejpeg compressor (pass dimensions for reshape)
        jpeg_compressor = SimplejpegCodec(quality=self.jpeg_quality, height=H, width=W)

        rgb_group = ep_group.create_group("rgb")
        arr = rgb_group.create_dataset(
            "front_img_1",
            shape=(T, H, W, C),
            chunks=(1, H, W, C),  # Per-frame chunks for random access
            dtype=np.uint8,
            compressor=jpeg_compressor,
        )

        # Write images frame by frame
        for t in range(T):
            arr[t] = images[t]

        logger.debug(f"Wrote {T} images with JPEG compression (quality={self.jpeg_quality})")

    def _write_state(self, ep_group: zarr.Group, episode_feats: dict):
        """Write end-effector pose in camera frame."""
        state = episode_feats["observations"]["state.ee_pose_cam"]
        T, D = state.shape

        state_group = ep_group.create_group("state")
        arr = state_group.create_dataset(
            "ee_pose_cam",
            data=state.astype(np.float32),
            chunks=(min(256, T), D),
            compressor=numcodecs.Zstd(level=3),
        )
        logger.debug(f"Wrote state: {state.shape}")

    def _write_actions(self, ep_group: zarr.Group, episode_feats: dict):
        """
        Write prestacked actions (T, chunk_size, action_dim).

        Prestacking avoids expensive on-the-fly I/O during training.
        Each frame stores the next 100 actions for action chunking.
        """
        actions = episode_feats.get("actions_ee_cartesian_cam")

        if actions is None:
            return

        if len(actions.shape) != 3:
            raise ValueError(f"Expected prestacked actions (T, chunk_size, dim), got {actions.shape}")

        T, chunk_size, D = actions.shape

        actions_group = ep_group.create_group("actions")
        arr = actions_group.create_dataset(
            "ee_cartesian_cam",
            data=actions.astype(np.float32),
            chunks=(1, chunk_size, D),  # Per-frame chunks for random access
            compressor=numcodecs.Zstd(level=3),
        )
        logger.debug(f"Wrote actions: {actions.shape} (prestacked)")

    def _write_keypoints(self, ep_group: zarr.Group, episode_feats: dict):
        """Write prestacked hand keypoints (T, chunk_size, 126)."""
        keypoints = episode_feats.get("actions_ee_keypoints_world")

        if keypoints is None:
            return

        if len(keypoints.shape) != 3:
            raise ValueError(f"Expected prestacked keypoints (T, chunk_size, dim), got {keypoints.shape}")

        T, chunk_size, D = keypoints.shape

        kp_group = ep_group.create_group("keypoints")
        arr = kp_group.create_dataset(
            "hand_keypoints_world",
            data=keypoints.astype(np.float32),
            chunks=(1, chunk_size, D),  # Per-frame chunks for random access
            compressor=numcodecs.Zstd(level=3),
        )
        logger.debug(f"Wrote keypoints: {keypoints.shape} (prestacked)")

    def _write_head_pose(self, ep_group: zarr.Group, episode_feats: dict):
        """Write head pose (world frame)."""
        head_pose = episode_feats.get("actions_head_cartesian_world")

        if head_pose is None:
            return

        T, D = head_pose.shape

        head_group = ep_group.create_group("head")
        arr = head_group.create_dataset(
            "pose_world",
            data=head_pose.astype(np.float32),
            chunks=(min(256, T), D),
            compressor=numcodecs.Zstd(level=3),
        )
        logger.debug(f"Wrote head pose: {head_pose.shape}")

    def _write_annotations(self, ep_group: zarr.Group, episode_feats: dict):
        """Write annotations as integer arrays with vocabulary."""
        annotations_df = episode_feats.get("annotations")

        if annotations_df is None or len(annotations_df) == 0:
            return

        # Build label vocabulary
        labels = annotations_df["Labels"].unique().tolist()
        label_to_id = {label: idx for idx, label in enumerate(labels)}

        # Store vocabulary in attrs
        ep_group.attrs["label_vocab"] = labels

        # Convert to arrays
        label_ids = annotations_df["Labels"].map(label_to_id).values.astype(np.int32)

        # Convert times to frames
        fps = self.fps
        start_frames = (annotations_df["start_time"] * fps).astype(np.int32).values
        end_frames = (annotations_df["end_time"] * fps).astype(np.int32).values

        # Create annotations group
        ann_group = ep_group.create_group("annotations")
        ann_group.create_dataset("label_id", data=label_ids)
        ann_group.create_dataset("start_frame", data=start_frames)
        ann_group.create_dataset("end_frame", data=end_frames)

        logger.debug(f"Wrote {len(label_ids)} annotations with {len(labels)} unique labels")

    def _accumulate_stats(self, episode_feats: dict):
        """Accumulate statistics for normalization.

        For prestacked arrays, we use the first timestep of each chunk
        (the current action at time t) for computing statistics.
        """
        # State
        state = episode_feats["observations"]["state.ee_pose_cam"]
        self.stats_accumulator.update("observations.state.ee_pose_cam", state)

        # Actions (prestacked) - use first action in each chunk for stats
        actions = episode_feats.get("actions_ee_cartesian_cam")
        if actions is not None and len(actions.shape) == 3:
            # (T, chunk_size, D) -> (T, D) for stats computation
            self.stats_accumulator.update("actions_ee_cartesian_cam", actions[:, 0, :])

        # Keypoints (prestacked)
        keypoints = episode_feats.get("actions_ee_keypoints_world")
        if keypoints is not None and len(keypoints.shape) == 3:
            self.stats_accumulator.update("actions_ee_keypoints_world", keypoints[:, 0, :])

        # Head pose (not prestacked)
        head = episode_feats.get("actions_head_cartesian_world")
        if head is not None:
            self.stats_accumulator.update("actions_head_cartesian_world", head)

    def consolidate(self):
        """Finalize dataset: write metadata, compute stats, consolidate."""
        logger.info("Consolidating dataset...")

        # Write episodes.jsonl
        episodes_path = self.output_root / "meta" / "episodes.jsonl"
        with open(episodes_path, "w") as f:
            for ep in self.episodes_meta:
                f.write(json.dumps(ep) + "\n")

        # Write info.json
        H, W = self.image_shape if self.image_shape else (360, 640)
        info = {
            "robot_type": self.robot_type,
            "fps": self.fps,
            "total_episodes": len(self.episodes_meta),
            "total_frames": sum(ep["length"] for ep in self.episodes_meta),
            "codebase_version": "zarr_v1",
            "features": {
                "observations.images.front_img_1": {
                    "dtype": "uint8",
                    "shape": [3, H, W],
                    "compression": "jpeg",
                    "quality": self.jpeg_quality,
                },
                "observations.state.ee_pose_cam": {
                    "dtype": "float32",
                    "shape": [12],
                },
                "actions_ee_cartesian_cam": {
                    "dtype": "float32",
                    "shape": [CHUNK_SIZE, 12],
                },
                "actions_ee_keypoints_world": {
                    "dtype": "float32",
                    "shape": [CHUNK_SIZE, 126],
                },
                "actions_head_cartesian_world": {
                    "dtype": "float32",
                    "shape": [10],
                },
            },
        }
        info_path = self.output_root / "meta" / "info.json"
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)

        # Write stats.json
        stats = self.stats_accumulator.compute_stats()
        stats_path = self.output_root / "meta" / "stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

        # Write tasks.json
        tasks = list(set(ep.get("task", "") for ep in self.episodes_meta))
        tasks_path = self.output_root / "meta" / "tasks.json"
        with open(tasks_path, "w") as f:
            json.dump({"tasks": tasks}, f, indent=2)

        # Consolidate zarr metadata for faster loading
        zarr.consolidate_metadata(str(self.output_root))

        logger.info(f"Consolidated dataset: {len(self.episodes_meta)} episodes, "
                    f"{sum(ep['length'] for ep in self.episodes_meta)} frames")


class StatsAccumulator:
    """Accumulate statistics for normalization."""

    def __init__(self):
        self.data = {}

    def update(self, key: str, values: np.ndarray):
        """Add values for a key."""
        if key not in self.data:
            self.data[key] = []
        self.data[key].append(values)

    def compute_stats(self) -> dict:
        """Compute mean, std, min, max for each key."""
        stats = {}

        for key, values_list in self.data.items():
            all_values = np.concatenate(values_list, axis=0)

            stats[key] = {
                "mean": all_values.mean(axis=0).tolist(),
                "std": all_values.std(axis=0).tolist(),
                "min": all_values.min(axis=0).tolist(),
                "max": all_values.max(axis=0).tolist(),
                "median": np.median(all_values, axis=0).tolist(),
                "quantile_1": np.percentile(all_values, 1, axis=0).tolist(),
                "quantile_99": np.percentile(all_values, 99, axis=0).tolist(),
            }

        return stats


class MeckaZarrConverter:
    """Convert Mecka episodes to Zarr format."""

    def __init__(
        self,
        episode_json_path: str,
        output_root: str,
        arm: str = "both",
        local_data_dir: Optional[Path] = None,
        jpeg_quality: int = 85,
    ):
        """
        Initialize converter.

        Args:
            episode_json_path: Path to episode JSON
            output_root: Path to output zarr store
            arm: Which arm data to include ("left", "right", "both")
            local_data_dir: Optional local data directory
            jpeg_quality: JPEG compression quality
        """
        self.episode_json_path = Path(episode_json_path)
        self.output_root = Path(output_root)
        self.arm = arm
        self.local_data_dir = Path(local_data_dir) if local_data_dir else None
        self.jpeg_quality = jpeg_quality

        # Determine robot type
        if arm == "both":
            self.robot_type = EMBODIMENT.MECKA_BIMANUAL.name
        elif arm == "left":
            self.robot_type = EMBODIMENT.MECKA_LEFT_ARM.name
        else:
            self.robot_type = EMBODIMENT.MECKA_RIGHT_ARM.name

        # Process episode
        logger.info(f"Processing episode: {episode_json_path}")
        self.episode_feats = MeckaExtractor.process_episode(
            str(episode_json_path),
            arm=arm,
            prestack=True,  # We'll extract raw from prestacked
            local_data_dir=self.local_data_dir,
        )

    def convert(self, overwrite: bool = False):
        """Convert episode to Zarr format."""
        # Get episode metadata
        episode_meta = self.episode_feats.get("episode_meta", {})
        episode_hash = episode_meta.get("id", self.episode_json_path.stem)
        task = episode_meta.get("task", "unknown_task")

        # Create writer
        writer = ZarrDatasetWriter(
            output_root=self.output_root,
            robot_type=self.robot_type,
            fps=30,
            jpeg_quality=self.jpeg_quality,
            overwrite=overwrite,
        )

        # Add episode
        writer.add_episode(
            episode_feats=self.episode_feats,
            episode_hash=episode_hash,
            task_description=task,
        )

        # Consolidate
        writer.consolidate()

        logger.info(f"Conversion complete! Dataset saved to: {self.output_root}")
        return self.output_root


def main():
    """Main conversion function."""
    parser = argparse.ArgumentParser(description="Convert Mecka RL2 to Zarr format")
    parser.add_argument("--episode-json", required=True, help="Path to episode JSON file")
    parser.add_argument("--output-dir", required=True, help="Output directory for Zarr dataset")
    parser.add_argument("--arm", default="both", choices=["left", "right", "both"],
                        help="Which arm(s) to include")
    parser.add_argument("--local-data-dir", type=str, default=None,
                        help="Path to directory with pre-downloaded files")
    parser.add_argument("--jpeg-quality", type=int, default=85,
                        help="JPEG compression quality (0-100)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing output")

    args = parser.parse_args()

    converter = MeckaZarrConverter(
        episode_json_path=args.episode_json,
        output_root=args.output_dir,
        arm=args.arm,
        local_data_dir=args.local_data_dir,
        jpeg_quality=args.jpeg_quality,
    )

    converter.convert(overwrite=args.overwrite)


if __name__ == "__main__":
    main()
