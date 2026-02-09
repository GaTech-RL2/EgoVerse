#!/usr/bin/env python3
"""
Scale SFS -> EgoVerse Zarr converter.

Output keys per episode:
    left.obs_ee_pose                 (T, 6)         xyzypr
    right.obs_ee_pose                (T, 6)         xyzypr
    left.obs_keypoints               (T, 63)        21 keypoints * 3 (xyz)
    right.obs_keypoints              (T, 63)        21 keypoints * 3 (xyz)
    left.obs_wrist_pose              (T, 6)         xyzypr
    right.obs_wrist_pose             (T, 6)         xyzypr
    obs_head_pose                    (T, 6)         xyzypr
    images.front_1                   (T, H, W, 3)   JPEG-compressed by ZarrWriter

Usage:
  python sfs_to_egoverse_zarr.py --task-ids TASK1 TASK2 --output-dir ./zarr_out
"""

from __future__ import annotations

import argparse
import os
import shutil
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R

from egomimic.rldb.zarr.zarr_writer import ZarrWriter

from sfsEgoverseUtils import (
    download_from_simple_response_dict,
    get_intrinsics,
    get_posepath,
    get_simple_response_dict_egocentric,
    load_annotation_file,
    load_scene,
)


MANO_LABELS = [
    "hand_wrist",
    "hand_thumb1", "hand_thumb2", "hand_thumb3", "hand_thumb4",
    "hand_index1", "hand_index2", "hand_index3", "hand_index4",
    "hand_middle1", "hand_middle2", "hand_middle3", "hand_middle4",
    "hand_ring1", "hand_ring2", "hand_ring3", "hand_ring4",
    "hand_pinky1", "hand_pinky2", "hand_pinky3", "hand_pinky4",
]
PALM_INDICES = [0, 5, 9, 13, 17]
NUM_KEYPOINTS = 21

INVALID_VALUE = 1e9
ACTION_WINDOW = 30
SUB_EPISODE_LENGTH = 300
IMAGE_SIZE = (640, 480)  # (W, H) for cv2.resize




# ---------------------------------------------------------------------------
# Data structures & extraction (unchanged from original)
# ---------------------------------------------------------------------------

@dataclass
class HandKeypoints:
    left: np.ndarray | None = None
    right: np.ndarray | None = None


@dataclass
class CameraPose:
    position: np.ndarray
    quaternion: np.ndarray
    rotation_matrix: np.ndarray

    @classmethod
    def from_pose_array(cls, pose: list[float]) -> CameraPose:
        position = np.array(pose[:3], dtype=np.float64)
        quaternion = np.array(pose[3:7], dtype=np.float64)
        rotation = R.from_quat(quaternion).as_matrix()
        return cls(position=position, quaternion=quaternion, rotation_matrix=rotation)

    def get_transform_matrix(self) -> np.ndarray:
        t = np.eye(4, dtype=np.float64)
        t[:3, :3] = self.rotation_matrix
        t[:3, 3] = self.position
        return t


@dataclass
class FrameData:
    frame_index: int
    timestamp_us: int
    camera_pose: CameraPose
    hand_keypoints: HandKeypoints
    text_annotations: list[dict[str, Any]] = field(default_factory=list)
    subgoal: dict[str, Any] | None = None
    collector_issue: dict[str, Any] | None = None


class SFSDataExtractor:
    """Extracts per-frame metadata from SFS + annotation files."""

    def __init__(self, sfs_path: str, annotation_path: str, video_path: str):
        self.video_path = video_path
        self.sfs_data = load_scene(sfs_path)
        self.annotation_data = load_annotation_file(annotation_path)

        if self.sfs_data is None or self.annotation_data is None:
            raise ValueError("Failed to load SFS or annotation data")

        self.camera_sensor_id = "left_rectified"
        self.intrinsics = get_intrinsics(self.sfs_data, self.camera_sensor_id)
        self.posepath = get_posepath(self.sfs_data, self.camera_sensor_id)
        if self.intrinsics is None or self.posepath is None:
            raise ValueError(f"Missing camera data for {self.camera_sensor_id}")

        self.timestamps = self.posepath.get("timestamps", [])
        self.pose_values = self.posepath.get("values", [])

        self._build_keypoint_lookup()
        self._build_annotation_lookup()

    # -- keypoint lookup (unchanged) --
    def _build_keypoint_lookup(self) -> None:
        self.keypoint_paths: dict[str, dict[int, dict[int, Any]]] = {"left": {}, "right": {}}
        for annotation in self.annotation_data.get("annotations", []):
            if annotation.get("type") != "points":
                continue
            labels = annotation.get("labels", [])
            paths = annotation.get("paths", [])
            for i, label in enumerate(labels):
                if i >= len(paths):
                    continue
                hand_type = "left" if label.startswith("left_") else "right" if label.startswith("right_") else None
                if hand_type is None:
                    continue
                prefix_len = 5 if hand_type == "left" else 6
                keypoint_name = label[prefix_len:]
                kp_idx = next((idx for idx, v in enumerate(MANO_LABELS) if v == keypoint_name), None)
                if kp_idx is None:
                    continue
                path = paths[i]
                values = path.get("values", [])
                for ts_idx, ts in enumerate(path.get("timestamps", [])):
                    self.keypoint_paths[hand_type].setdefault(ts, {})
                    if ts_idx < len(values):
                        self.keypoint_paths[hand_type][ts][kp_idx] = values[ts_idx]

    def _build_annotation_lookup(self) -> None:
        self.text_annotations: list[dict] = []
        self.subgoal_annotations: list[dict] = []
        self.collector_issues: list[dict] = []
        self.demonstration_metadata: dict[str, Any] = {}

        for attr in self.annotation_data.get("attributes", []):
            values = attr.get("values", [])
            if values:
                self.demonstration_metadata[attr.get("name", "")] = values[0]

        for annotation in self.annotation_data.get("annotations", []):
            if annotation.get("type") != "text_annotation":
                continue
            label = annotation.get("label", "")
            for clip in annotation.get("clips", []):
                start_ts = clip.get("timestamp", 0)
                end_ts = start_ts + clip.get("duration", 0)
                text = clip.get("text", "")
                attr_dict = {}
                for attr in clip.get("attributes", []):
                    vals = attr.get("values", [])
                    if vals:
                        attr_dict[attr.get("name", "")] = vals[0]

                if label == "Sub-goal":
                    self.subgoal_annotations.append(
                        {"start_ts": start_ts, "end_ts": end_ts, "text": text}
                    )
                elif label == "Collector Issue":
                    self.collector_issues.append(
                        {"start_ts": start_ts, "end_ts": end_ts,
                         "issue_type": attr_dict.get("Collector Quality Issue", "")}
                    )
                self.text_annotations.append(
                    {"label": label, "text": text, "start_ts": start_ts,
                     "end_ts": end_ts, "attributes": attr_dict}
                )

    def get_hand_keypoints_at_timestamp(self, timestamp: int) -> HandKeypoints:
        result = HandKeypoints()
        for hand_type in ("left", "right"):
            if timestamp not in self.keypoint_paths[hand_type]:
                continue
            kp_dict = self.keypoint_paths[hand_type][timestamp]
            if len(kp_dict) < NUM_KEYPOINTS // 2:
                continue
            keypoints = np.full((NUM_KEYPOINTS, 3), INVALID_VALUE, dtype=np.float32)
            for kp_idx, xyz in kp_dict.items():
                keypoints[kp_idx] = xyz
            if hand_type == "left":
                result.left = keypoints
            else:
                result.right = keypoints
        return result

    def get_subgoal_at_timestamp(self, timestamp: int) -> dict[str, Any] | None:
        for item in self.subgoal_annotations:
            if item["start_ts"] <= timestamp <= item["end_ts"]:
                return item
        return None

    def get_collector_issue_at_timestamp(self, timestamp: int) -> dict[str, Any] | None:
        for item in self.collector_issues:
            if item["start_ts"] <= timestamp <= item["end_ts"]:
                return item
        return None

    def get_text_annotations_at_timestamp(self, timestamp: int) -> list[dict[str, Any]]:
        return [
            ann for ann in self.text_annotations
            if ann["start_ts"] <= timestamp <= ann["end_ts"]
        ]

    def extract_all_frames_metadata(self) -> list[FrameData]:
        frames = []
        for i, ts in enumerate(self.timestamps):
            pose = self.pose_values[i]
            frames.append(
                FrameData(
                    frame_index=i,
                    timestamp_us=ts,
                    camera_pose=CameraPose.from_pose_array(pose),
                    hand_keypoints=self.get_hand_keypoints_at_timestamp(ts),
                    text_annotations=self.get_text_annotations_at_timestamp(ts),
                    subgoal=self.get_subgoal_at_timestamp(ts),
                    collector_issue=self.get_collector_issue_at_timestamp(ts),
                )
            )
        return frames

    def load_images_for_range(self, start_idx: int, end_idx: int) -> list[np.ndarray | None]:
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return [None] * (end_idx - start_idx)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
        images: list[np.ndarray | None] = []
        for _ in range(end_idx - start_idx):
            ret, frame = cap.read()
            if not ret:
                images.append(None)
                continue
            images.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return images


# ---------------------------------------------------------------------------
# Palm pose computation (vectorised where possible)
# ---------------------------------------------------------------------------

def _compute_palm_centroid(keypoints: np.ndarray) -> np.ndarray:
    palm_kps = keypoints[PALM_INDICES]
    valid_mask = ~np.any(palm_kps >= INVALID_VALUE - 1, axis=1)
    if not np.any(valid_mask):
        return np.full(3, INVALID_VALUE, dtype=np.float32)
    return np.mean(palm_kps[valid_mask], axis=0).astype(np.float32)


def _compute_palm_orientation(keypoints: np.ndarray, mirror_y: bool = False) -> np.ndarray:
    wrist, index1, middle1, pinky1 = keypoints[0], keypoints[5], keypoints[9], keypoints[17]
    if any(np.any(kp >= INVALID_VALUE - 1) for kp in (wrist, index1, pinky1)):
        return np.zeros(3, dtype=np.float32)
    x_axis = middle1 - wrist
    x_axis /= np.linalg.norm(x_axis) + 1e-8
    temp_y = pinky1 - wrist
    z_axis = np.cross(x_axis, temp_y)
    z_axis /= np.linalg.norm(z_axis) + 1e-8
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis) + 1e-8
    if mirror_y:
        y_axis = -y_axis
        z_axis = np.cross(x_axis, y_axis)
        z_axis /= np.linalg.norm(z_axis) + 1e-8
    rot = np.column_stack([x_axis, y_axis, z_axis])
    try:
        return R.from_matrix(rot).as_euler("ZYX", degrees=False).astype(np.float32)
    except Exception:
        return np.zeros(3, dtype=np.float32)


def _compute_palm_6dof(keypoints: np.ndarray, mirror_y: bool = False) -> np.ndarray:
    centroid = _compute_palm_centroid(keypoints)
    if np.any(centroid >= INVALID_VALUE - 1):
        return np.full(6, INVALID_VALUE, dtype=np.float32)
    ypr = _compute_palm_orientation(keypoints, mirror_y=mirror_y)
    return np.concatenate([centroid, ypr]).astype(np.float32)


def _compute_wrist_position(keypoints: np.ndarray) -> np.ndarray:
    wrist = keypoints[0]
    if np.any(wrist >= INVALID_VALUE - 1):
        return np.full(3, INVALID_VALUE, dtype=np.float32)
    return wrist.astype(np.float32)


def _compute_wrist_orientation(keypoints: np.ndarray, mirror_y: bool = False) -> np.ndarray:
    wrist, index1, middle1, pinky1 = keypoints[0], keypoints[5], keypoints[9], keypoints[17]
    if any(np.any(kp >= INVALID_VALUE - 1) for kp in (wrist, index1, pinky1)):
        return np.zeros(3, dtype=np.float32)

    x_axis = middle1 - wrist
    x_axis /= np.linalg.norm(x_axis) + 1e-8
    temp_y = pinky1 - wrist
    z_axis = np.cross(x_axis, temp_y)
    z_axis /= np.linalg.norm(z_axis) + 1e-8
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis) + 1e-8
    if mirror_y:
        y_axis = -y_axis
        z_axis = np.cross(x_axis, y_axis)
        z_axis /= np.linalg.norm(z_axis) + 1e-8
    rot = np.column_stack([x_axis, y_axis, z_axis])
    try:
        return R.from_matrix(rot).as_euler("ZYX", degrees=False).astype(np.float32)
    except Exception:
        return np.zeros(3, dtype=np.float32)


def _compute_wrist_6dof(keypoints: np.ndarray, mirror_y: bool = False) -> np.ndarray:
    wrist_xyz = _compute_wrist_position(keypoints)
    if np.any(wrist_xyz >= INVALID_VALUE - 1):
        return np.full(6, INVALID_VALUE, dtype=np.float32)
    wrist_ypr = _compute_wrist_orientation(keypoints, mirror_y=mirror_y)
    return np.concatenate([wrist_xyz, wrist_ypr]).astype(np.float32)


# ---------------------------------------------------------------------------
# Language annotations
# ---------------------------------------------------------------------------

def _build_language_annotations(sub_frames: list[FrameData]) -> list[tuple[str, int, int]]:
    rows: list[tuple[str, int, int]] = []
    current_text: str | None = None
    start_idx: int | None = None
    for idx, frame in enumerate(sub_frames):
        label = frame.subgoal.get("text", "").strip() if frame.subgoal else ""
        text = label if label else None
        if text != current_text:
            if current_text is not None and start_idx is not None:
                rows.append((current_text, start_idx, idx - 1))
            current_text = text
            start_idx = idx if text is not None else None
    if current_text is not None and start_idx is not None:
        rows.append((current_text, start_idx, len(sub_frames) - 1))
    return rows


def _task_description(frames: list[FrameData], demo_meta: dict[str, Any]) -> str:
    candidate = str(demo_meta.get("Demonstration", "")).strip()
    if candidate:
        return candidate
    skip = {"Inactive Time", "Collector Issue", "inactive time", "collector issue"}
    for frame in frames:
        for ann in frame.text_annotations:
            text = str(ann.get("text", "")).strip()
            label = str(ann.get("label", "")).strip()
            if text and text not in skip and label not in skip:
                return text
    return "Unknown task"


# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------

def convert_task_to_zarr(
    task_id: str,
    output_dir: str,
    download_dir: str,
    robot_type: str = "scale_bimanual",
    fps: int = 30,
) -> int:
    """Convert one Scale task to one or more Zarr episodes. Returns count."""
    t_start = time.perf_counter()

    print(f"[{task_id}] Fetching task metadata...")
    task_download_path = os.path.join(download_dir, task_id)
    os.makedirs(task_download_path, exist_ok=True)

    response = get_simple_response_dict_egocentric(task_id)
    if response is None:
        raise ValueError(f"Task {task_id} not found or Scale API failed")

    print(f"[{task_id}] Downloading files...")
    t_dl = time.perf_counter()
    local_paths = download_from_simple_response_dict(task_download_path, response)
    sfs_path = local_paths.get("sfs")
    annotations_path = local_paths.get("annotations")
    video_path = local_paths.get("left_rectified") or local_paths.get("left_rgb")
    if not all([sfs_path, annotations_path, video_path]):
        raise ValueError(f"Missing SFS/annotation/video files for task {task_id}")

    def _nonempty(p: str | None) -> bool:
        return bool(p) and os.path.exists(p) and os.path.getsize(p) > 0

    if not (_nonempty(sfs_path) and _nonempty(annotations_path)):
        raise ValueError(f"Downloaded SFS/annotation files are empty for task {task_id}")
    print(f"[{task_id}] Downloaded in {time.perf_counter() - t_dl:.1f}s")

    print(f"[{task_id}] Loading SFS metadata...")
    try:
        extractor = SFSDataExtractor(sfs_path, annotations_path, video_path)
    except ValueError:
        print(f"[{task_id}] Load failed — re-downloading SFS + annotations...")
        for p in (sfs_path, annotations_path):
            if p and os.path.exists(p):
                os.remove(p)
        local_paths = download_from_simple_response_dict(task_download_path, response)
        sfs_path = local_paths.get("sfs")
        annotations_path = local_paths.get("annotations")
        video_path = local_paths.get("left_rectified") or local_paths.get("left_rgb")
        extractor = SFSDataExtractor(sfs_path, annotations_path, video_path)
    frames = extractor.extract_all_frames_metadata()
    n_frames = len(frames)
    if n_frames <= ACTION_WINDOW:
        raise ValueError(f"Task {task_id} has too few frames ({n_frames})")

    task_desc = _task_description(frames, extractor.demonstration_metadata)
    valid_frame_count = n_frames - ACTION_WINDOW

    # ------------------------------------------------------------------
    # Precompute all per-frame data into dense arrays (once)
    # ------------------------------------------------------------------
    left_world = np.full((n_frames, 6), INVALID_VALUE, dtype=np.float32)
    right_world = np.full((n_frames, 6), INVALID_VALUE, dtype=np.float32)
    left_wrist = np.full((n_frames, 6), INVALID_VALUE, dtype=np.float32)
    right_wrist = np.full((n_frames, 6), INVALID_VALUE, dtype=np.float32)
    left_kps = np.full((n_frames, 63), INVALID_VALUE, dtype=np.float32)
    right_kps = np.full((n_frames, 63), INVALID_VALUE, dtype=np.float32)
    head_pose_world = np.zeros((n_frames, 6), dtype=np.float32)

    for i, frame in enumerate(frames):
        if frame.hand_keypoints.left is not None:
            left_world[i] = _compute_palm_6dof(frame.hand_keypoints.left)
            left_wrist[i] = _compute_wrist_6dof(frame.hand_keypoints.left)
            left_kps[i] = frame.hand_keypoints.left.flatten().astype(np.float32)
        if frame.hand_keypoints.right is not None:
            right_world[i] = _compute_palm_6dof(frame.hand_keypoints.right, mirror_y=True)
            right_wrist[i] = _compute_wrist_6dof(frame.hand_keypoints.right, mirror_y=True)
            right_kps[i] = frame.hand_keypoints.right.flatten().astype(np.float32)
        head_pose_world[i, :3] = frame.camera_pose.position.astype(np.float32)
        head_pose_world[i, 3:] = R.from_matrix(frame.camera_pose.rotation_matrix).as_euler(
            "ZYX", degrees=False
        ).astype(np.float32)

    # ------------------------------------------------------------------
    # Filter valid frame indices (same criteria as old script)
    # ------------------------------------------------------------------
    valid_indices: list[int] = []
    for t in range(valid_frame_count):
        if (
            frames[t].collector_issue is not None
            and frames[t].collector_issue.get("issue_type") == "Inactive Time"
        ):
            continue
        window = slice(t, t + ACTION_WINDOW)
        n_invalid = (
            np.sum(np.any(left_world[window] >= INVALID_VALUE - 1, axis=1))
            + np.sum(np.any(right_world[window] >= INVALID_VALUE - 1, axis=1))
        )
        if n_invalid > ACTION_WINDOW:  # >50% of 2*ACTION_WINDOW
            continue
        valid_indices.append(t)

    if not valid_indices:
        raise ValueError(f"Task {task_id} has no valid frames after filtering")

    print(f"[{task_id}] {len(valid_indices)} valid frames out of {valid_frame_count}")

    # ------------------------------------------------------------------
    # Write sub-episodes
    # ------------------------------------------------------------------
    folder = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H-%M-%S-%f")
    task_output_dir = Path(output_dir) / folder
    task_output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for ep_start in range(0, len(valid_indices), SUB_EPISODE_LENGTH):
        sub = valid_indices[ep_start : ep_start + SUB_EPISODE_LENGTH]
        if len(sub) < 10:
            continue

        min_frame = min(sub)
        max_frame = max(sub)
        image_batch = extractor.load_images_for_range(min_frame, max_frame+1)

        # First pass: figure out which frames have images
        kept: list[int] = []
        for t in sub:
            img = image_batch[t - min_frame]
            if img is not None:
                kept.append(t)
        if len(kept) < 10:
            continue

        T = len(kept)

        # ---- Per-frame current state (vectorised) ----
        kept_arr = np.array(kept)
        left_curr_6 = left_world[kept_arr]   # (T, 6)
        right_curr_6 = right_world[kept_arr]
        left_curr_6 = np.where(left_curr_6 >= INVALID_VALUE - 1, 0.0, left_curr_6).astype(
            np.float32
        )
        right_curr_6 = np.where(right_curr_6 >= INVALID_VALUE - 1, 0.0, right_curr_6).astype(
            np.float32
        )
        left_wrist_curr_6 = np.where(
            left_wrist[kept_arr] >= INVALID_VALUE - 1, 0.0, left_wrist[kept_arr]
        ).astype(np.float32)
        right_wrist_curr_6 = np.where(
            right_wrist[kept_arr] >= INVALID_VALUE - 1, 0.0, right_wrist[kept_arr]
        ).astype(np.float32)

        # Head pose & keypoints
        actions_head = head_pose_world[kept_arr]
        left_keypoints = np.where(
            left_kps[kept_arr] >= INVALID_VALUE - 1, 0.0, left_kps[kept_arr]
        ).astype(np.float32)
        right_keypoints = np.where(
            right_kps[kept_arr] >= INVALID_VALUE - 1, 0.0, right_kps[kept_arr]
        ).astype(np.float32)

        # ---- Build image array ----
        images = np.stack(
            [cv2.resize(image_batch[t - min_frame], IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)
             for t in kept],
            axis=0,
        ).astype(np.uint8)

        # ---- Numeric data ----
        numeric_data = {
            "left.obs_ee_pose": left_curr_6,
            "right.obs_ee_pose": right_curr_6,
            "left.obs_keypoints": left_keypoints,
            "right.obs_keypoints": right_keypoints,
            "left.obs_wrist_pose": left_wrist_curr_6,
            "right.obs_wrist_pose": right_wrist_curr_6,
            "obs_head_pose": actions_head,
        }
        image_data = {
            "images.front_1": images,
        }

        used_frames = [frames[t] for t in kept]
        lang_ann = _build_language_annotations(used_frames)

        episode_path = task_output_dir / f"{task_id}_episode_{written:06d}.zarr"
        ZarrWriter.create_and_write(
            episode_path=episode_path,
            numeric_data=numeric_data,
            image_data=image_data,
            embodiment=robot_type,
            fps=fps,
            task=task_desc,
            annotations=lang_ann if lang_ann else None,
            enable_sharding=False,
        )
        written += 1
        print(f"[{task_id}] Wrote episode {written} ({T} frames) -> {episode_path.name}")

    # Clean download cache
    if os.path.exists(task_download_path):
        shutil.rmtree(task_download_path)

    elapsed = time.perf_counter() - t_start
    print(f"[{task_id}] Done: {written} episode(s) in {elapsed:.1f}s -> {task_output_dir}")
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert Scale SFS tasks to EgoVerse Zarr episodes"
    )
    parser.add_argument("--task-ids", nargs="+", required=True, help="Scale task IDs")
    parser.add_argument("--output-dir", default="egoverse_zarr_dataset", help="Output root")
    parser.add_argument("--download-dir", default="scale_data", help="Temp download cache")
    parser.add_argument("--robot-type", default="scale_bimanual", help="Embodiment tag")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.download_dir).mkdir(parents=True, exist_ok=True)

    total_episodes = 0
    failed: list[str] = []
    for idx, task_id in enumerate(args.task_ids, start=1):
        print(f"\n[{idx}/{len(args.task_ids)}] {task_id}")
        try:
            n = convert_task_to_zarr(
                task_id=task_id,
                output_dir=args.output_dir,
                download_dir=args.download_dir,
                robot_type=args.robot_type,
                fps=args.fps,
            )
            total_episodes += n
        except Exception as exc:
            print(f"[{task_id}] ERROR: {exc}")
            traceback.print_exc()
            failed.append(task_id)

    print(f"\n{'=' * 60}")
    print(f"Conversion complete: {len(args.task_ids)} tasks, "
          f"{len(args.task_ids) - len(failed)} ok, {len(failed)} failed")
    print(f"Episodes written: {total_episodes}")
    if failed:
        print(f"Failed: {failed}")
    print(f"Output: {Path(args.output_dir).resolve()}")
    print("=" * 60)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
