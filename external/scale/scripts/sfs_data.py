"""
SFS data structures, extraction, and hand pose geometry.

Provides:
  - Data classes: HandKeypoints, CameraPose, FrameData
  - SFSDataExtractor: parses SFS + annotation files into per-frame metadata
  - Hand pose computation: palm 6DoF, wrist 6DoF, batch euler-to-quat
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

from scale_api import (
    get_intrinsics,
    get_posepath,
    load_annotation_file,
    load_scene,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Data classes
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
    hand_tracking_error: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# SFS data extraction
# ---------------------------------------------------------------------------


class SFSDataExtractor:
    """Extracts per-frame metadata from SFS + annotation files."""

    def __init__(self, sfs_path: str, annotation_path: str, video_path: str):
        self.video_path = video_path
        self.sfs_data = load_scene(sfs_path)
        self.annotation_data = load_annotation_file(annotation_path)

        if self.sfs_data is None or self.annotation_data is None:
            raise ValueError("Failed to load SFS or annotation data")

        self.camera_sensor_id = "left_rectified"
        self.posepath = get_posepath(self.sfs_data, self.camera_sensor_id)
        if self.posepath is None:
            raise ValueError(f"Missing pose data for {self.camera_sensor_id}")

        self.timestamps = self.posepath.get("timestamps", [])
        self.pose_values = self.posepath.get("values", [])

        self._build_keypoint_lookup()
        self._build_annotation_lookup()

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
        self.hand_tracking_errors: list[dict] = []
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
                elif label == "Hand Tracking Error":
                    error_type = attr_dict.get("Hand Tracking Error", text)
                    hand = attr_dict.get("Hand", "Both")
                    self.hand_tracking_errors.append(
                        {"start_ts": start_ts, "end_ts": end_ts,
                         "error_type": error_type, "hand": hand}
                    )
                # Promote clip-level "Hand Used" to demonstration metadata
                if "Hand Used" in attr_dict and "Hand Used" not in self.demonstration_metadata:
                    self.demonstration_metadata["Hand Used"] = attr_dict["Hand Used"]

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

    def get_hand_tracking_error_at_timestamp(self, timestamp: int) -> dict[str, Any] | None:
        for item in self.hand_tracking_errors:
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
                    hand_tracking_error=self.get_hand_tracking_error_at_timestamp(ts),
                )
            )
        return frames


# ---------------------------------------------------------------------------
# Hand pose geometry
# ---------------------------------------------------------------------------


def _batch_euler_to_quat(euler_zyx: np.ndarray) -> np.ndarray:
    """(N, 3) euler ZYX -> (N, 4) quaternion wxyz."""
    q_xyzw = R.from_euler("ZYX", euler_zyx, degrees=False).as_quat()
    return q_xyzw[..., [3, 0, 1, 2]].astype(np.float32)


def batch_pose6_to_pose7(pose6: np.ndarray) -> np.ndarray:
    """(N, 6) [xyz ypr] -> (N, 7) [xyz quat_wxyz].  Invalid sentinels -> zeros."""
    N = pose6.shape[0]
    out = np.zeros((N, 7), dtype=np.float32)
    valid = ~np.any(pose6 >= INVALID_VALUE - 1, axis=1)
    if valid.any():
        out[valid, :3] = pose6[valid, :3]
        out[valid, 3:] = _batch_euler_to_quat(pose6[valid, 3:6])
    return out


def _compute_palm_centroid(keypoints: np.ndarray) -> np.ndarray:
    palm_kps = keypoints[PALM_INDICES]
    valid_mask = ~np.any(palm_kps >= INVALID_VALUE - 1, axis=1)
    if not np.any(valid_mask):
        return np.full(3, INVALID_VALUE, dtype=np.float32)
    return np.mean(palm_kps[valid_mask], axis=0).astype(np.float32)


def _compute_hand_orientation(keypoints: np.ndarray, flip_x: bool = False) -> np.ndarray:
    """Hand frame: x=right, y=down (palm normal toward ground), z=forward (toward fingers).

    flip_x=True for the right hand so that x is rightward for both hands.
    Shared by both palm and wrist pose computation.
    """
    wrist, index1, middle1, pinky1 = keypoints[0], keypoints[5], keypoints[9], keypoints[17]
    if any(np.any(kp >= INVALID_VALUE - 1) for kp in (wrist, index1, middle1, pinky1)):
        return np.zeros(3, dtype=np.float32)
    z_axis = middle1 - wrist
    z_axis /= np.linalg.norm(z_axis) + 1e-8
    across = (pinky1 - index1) if flip_x else (index1 - pinky1)
    across -= np.dot(across, z_axis) * z_axis
    x_axis = across / (np.linalg.norm(across) + 1e-8)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis) + 1e-8
    rot = np.column_stack([x_axis, y_axis, z_axis])
    try:
        return R.from_matrix(rot).as_euler("ZYX", degrees=False).astype(np.float32)
    except Exception:
        return np.zeros(3, dtype=np.float32)


def compute_palm_6dof(keypoints: np.ndarray, flip_x: bool = False) -> np.ndarray:
    centroid = _compute_palm_centroid(keypoints)
    if np.any(centroid >= INVALID_VALUE - 1):
        return np.full(6, INVALID_VALUE, dtype=np.float32)
    ypr = _compute_hand_orientation(keypoints, flip_x=flip_x)
    return np.concatenate([centroid, ypr]).astype(np.float32)


def _compute_wrist_position(keypoints: np.ndarray) -> np.ndarray:
    wrist = keypoints[0]
    if np.any(wrist >= INVALID_VALUE - 1):
        return np.full(3, INVALID_VALUE, dtype=np.float32)
    return wrist.astype(np.float32)


def compute_wrist_6dof(keypoints: np.ndarray, flip_x: bool = False) -> np.ndarray:
    wrist_xyz = _compute_wrist_position(keypoints)
    if np.any(wrist_xyz >= INVALID_VALUE - 1):
        return np.full(6, INVALID_VALUE, dtype=np.float32)
    wrist_ypr = _compute_hand_orientation(keypoints, flip_x=flip_x)
    return np.concatenate([wrist_xyz, wrist_ypr]).astype(np.float32)
