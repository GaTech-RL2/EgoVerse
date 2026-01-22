#!/usr/bin/env python3
"""
SFS to Egoverse LeRobot Converter

Converts Scale Sensor Fusion Scene (SFS) data to the Egoverse LeRobot format
with pre-stacked action chunks for robotics training.

Usage:
    export SCALE_API_KEY="your_api_key"
    python sfsToEgoverse.py --task-ids TASK_ID_1 TASK_ID_2 --output-dir ./dataset

Output Structure:
dataset/
    ├── data/chunk-000/
    │   ├── episode_000000.parquet
    │   ├── episode_000000_annotations.csv
    │   └── ...
└── meta/
    ├── episodes.jsonl
    ├── info.json
    ├── stats.json
    ├── env.jsonl
    └── tasks.jsonl

Parquet Columns:
    - actions_ee_cartesian_cam: (100, 12) palm poses in camera frame
    - actions_ee_keypoints_world: (100, 126) hand keypoints in world frame
    - actions_head_cartesian_world: (10,) head pose
    - observations.images.front_img_1: (C, H, W) RGB image
    - observations.state.ee_pose_cam: (12,) current palm pose
    - timestamp: seconds
    - frame_index: int
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

import cv2
import numpy as np
import pandas as pd
import pyarrow as pa
import io
from PIL import Image
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as R
from datasets import Dataset, Features, Image as ImageFeature
from tqdm import tqdm

from sfsEgoverseUtils import (
    get_simple_response_dict_egocentric,
    download_from_simple_response_dict,
    load_scene,
    load_annotation_file,
    get_posepath,
    get_intrinsics,
)
from egomimic.utils.egomimicUtils import interpolate_arr, interpolate_arr_euler

# Constants
MANO_LABELS = [
    "hand_wrist",
    "hand_thumb1", "hand_thumb2", "hand_thumb3", "hand_thumb4",
    "hand_index1", "hand_index2", "hand_index3", "hand_index4",
    "hand_middle1", "hand_middle2", "hand_middle3", "hand_middle4",
    "hand_ring1", "hand_ring2", "hand_ring3", "hand_ring4",
    "hand_pinky1", "hand_pinky2", "hand_pinky3", "hand_pinky4",
]

PALM_INDICES = [0, 5, 9, 13, 17]  # wrist + finger bases
ACTION_CHUNK_LENGTH = 100
ACTION_INTERPOLATION_WINDOW = 30
SUB_EPISODE_LENGTH = 300
INVALID_VALUE = 1e9
NUM_KEYPOINTS = 21


@dataclass
class HandKeypoints:
    left: Optional[np.ndarray] = None
    right: Optional[np.ndarray] = None


@dataclass
class CameraPose:
    position: np.ndarray
    quaternion: np.ndarray
    rotation_matrix: np.ndarray
    
    @classmethod
    def from_pose_array(cls, pose: List[float]) -> 'CameraPose':
        position = np.array(pose[:3])
        quaternion = np.array(pose[3:7])
        rotation = R.from_quat(quaternion)
        return cls(position=position, quaternion=quaternion, rotation_matrix=rotation.as_matrix())
    
    def get_transform_matrix(self) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = self.rotation_matrix
        T[:3, 3] = self.position
        return T
    
    def get_euler_ypr(self) -> np.ndarray:
        return R.from_matrix(self.rotation_matrix).as_euler('zyx', degrees=False)


@dataclass
class FrameData:
    frame_index: int
    timestamp_us: int
    camera_pose: CameraPose
    hand_keypoints: HandKeypoints
    image: Optional[np.ndarray] = None
    text_annotations: List[Dict[str, Any]] = field(default_factory=list)
    subgoal: Optional[Dict[str, Any]] = None
    collector_issue: Optional[Dict[str, Any]] = None


class SFSDataExtractor:
    """Extracts data from SFS files and associated annotations/videos."""
    
    def __init__(self, sfs_path: str, annotation_path: str, video_path: str):
        self.video_path = video_path
        
        print(f"Loading SFS: {sfs_path}")
        self.sfs_data = load_scene(sfs_path)
        
        print(f"Loading annotations: {annotation_path}")
        self.annotation_data = load_annotation_file(annotation_path)
        
        if self.sfs_data is None or self.annotation_data is None:
            raise ValueError("Failed to load SFS or annotation data")
        
        self.camera_sensor_id = "left_rectified"
        self.intrinsics = get_intrinsics(self.sfs_data, self.camera_sensor_id)
        self.posepath = get_posepath(self.sfs_data, self.camera_sensor_id)
        
        if self.intrinsics is None or self.posepath is None:
            raise ValueError(f"Missing camera data for {self.camera_sensor_id}")
        
        self.timestamps = self.posepath.get('timestamps', [])
        self.pose_values = self.posepath.get('values', [])
        print(f"Found {len(self.timestamps)} frames")
        
        self._build_keypoint_lookup()
        self._build_annotation_lookup()
    
    def _build_keypoint_lookup(self):
        self.keypoint_paths = {'left': {}, 'right': {}}
        
        for annotation in self.annotation_data.get('annotations', []):
            if annotation.get('type') != 'points':
                continue
            
            labels = annotation.get('labels', [])
            paths = annotation.get('paths', [])
            
            for i, label in enumerate(labels):
                if i >= len(paths):
                    continue
                
                hand_type = 'left' if label.startswith('left_') else 'right' if label.startswith('right_') else None
                if not hand_type:
                    continue
                
                prefix_len = 5 if hand_type == 'left' else 6
                keypoint_name = label[prefix_len:]
                
                kp_idx = next((idx for idx, m in enumerate(MANO_LABELS) if keypoint_name == m), None)
                if kp_idx is None:
                    continue
                
                path = paths[i]
                for ts_idx, ts in enumerate(path.get('timestamps', [])):
                    if ts not in self.keypoint_paths[hand_type]:
                        self.keypoint_paths[hand_type][ts] = {}
                    values = path.get('values', [])
                    if ts_idx < len(values):
                        self.keypoint_paths[hand_type][ts][kp_idx] = values[ts_idx]
    
    def _build_annotation_lookup(self):
        self.text_annotations = []
        self.subgoal_annotations = []
        self.collector_issues = []
        self.demonstration_metadata = {}
        
        # Top-level attributes
        for attr in self.annotation_data.get('attributes', []):
            values = attr.get('values', [])
            if values:
                self.demonstration_metadata[attr.get('name', '')] = values[0]
        
        for annotation in self.annotation_data.get('annotations', []):
            if annotation.get('type') != 'text_annotation':
                continue
            
            label = annotation.get('label', '')
            
            for clip in annotation.get('clips', []):
                clip_start = clip.get('timestamp', 0)
                clip_end = clip_start + clip.get('duration', 0)
                clip_text = clip.get('text', '')
                
                attr_dict = {}
                for attr in clip.get('attributes', []):
                    values = attr.get('values', [])
                    if values:
                        attr_dict[attr.get('name', '')] = values[0]
                
                if label == 'Sub-goal':
                    self.subgoal_annotations.append({
                        'start_ts': clip_start, 'end_ts': clip_end, 'text': clip_text,
                        'subgoal_type': attr_dict.get('Sub-goal Type', ''),
                        'mistake': attr_dict.get('Mistake', ''),
                        'navigation': attr_dict.get('Navigation', ''),
                        'other_subgoal': attr_dict.get('Other Subgoal', ''),
                    })
                elif label == 'ClipExport':
                    self.demonstration_metadata.update({
                        'Demonstration Rating': attr_dict.get('Demonstration Rating', ''),
                        'Outcome Rating': attr_dict.get('Outcome Rating', ''),
                        'Hand Used': attr_dict.get('Hand Used', ''),
                        'Grasp Type': attr_dict.get('Grasp Type', ''),
                        'Demonstration': attr_dict.get('Demonstration', ''),
                    })
                elif label == 'Collector Issue':
                    self.collector_issues.append({
                        'start_ts': clip_start, 'end_ts': clip_end,
                        'issue_type': attr_dict.get('Collector Quality Issue', ''),
                    })
                
                self.text_annotations.append({
                    'label': label, 'text': clip_text,
                    'start_ts': clip_start, 'end_ts': clip_end, 'attributes': attr_dict,
                })
    
    def get_subgoal_at_timestamp(self, timestamp: int) -> Optional[Dict[str, Any]]:
        for sg in self.subgoal_annotations:
            if sg['start_ts'] <= timestamp <= sg['end_ts']:
                return sg
        return None
        
    def get_collector_issue_at_timestamp(self, timestamp: int) -> Optional[Dict[str, Any]]:
        for issue in self.collector_issues:
            if issue['start_ts'] <= timestamp <= issue['end_ts']:
                return issue
        return None
    
    def get_hand_keypoints_at_timestamp(self, timestamp: int) -> HandKeypoints:
        result = HandKeypoints()
        
        for hand_type in ['left', 'right']:
            if timestamp not in self.keypoint_paths[hand_type]:
                continue
            
            kp_dict = self.keypoint_paths[hand_type][timestamp]
            if len(kp_dict) < NUM_KEYPOINTS // 2:
                continue
            
            keypoints = np.full((NUM_KEYPOINTS, 3), INVALID_VALUE)
            for kp_idx, xyz in kp_dict.items():
                keypoints[kp_idx] = xyz
            
            if hand_type == 'left':
                result.left = keypoints
            else:
                result.right = keypoints
        
        return result
    
    def get_text_annotations_at_timestamp(self, timestamp: int) -> List[Dict[str, Any]]:
        return [
            {'label': ann['label'], 'text': ann['text'], 'attributes': ann['attributes']}
            for ann in self.text_annotations
            if ann['start_ts'] <= timestamp <= ann['end_ts']
        ]
    
    def extract_all_frames(self) -> List[FrameData]:
        frames = []
        
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {self.video_path}")
        
        video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        num_frames = min(len(self.timestamps), video_frame_count)
        print(f"Video: {video_frame_count} frames, SFS: {len(self.timestamps)} poses")
        
        for i in tqdm(range(num_frames), desc="Extracting frames"):
            timestamp = self.timestamps[i]
            pose = self.pose_values[i]
            
            ret, frame = cap.read()
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).copy() if ret else None
            
            frames.append(FrameData(
                frame_index=i,
                timestamp_us=timestamp,
                camera_pose=CameraPose.from_pose_array(pose),
                hand_keypoints=self.get_hand_keypoints_at_timestamp(timestamp),
                image=image,
                text_annotations=self.get_text_annotations_at_timestamp(timestamp),
                subgoal=self.get_subgoal_at_timestamp(timestamp),
                collector_issue=self.get_collector_issue_at_timestamp(timestamp),
            ))
        
        cap.release()
        return frames


class PoseComputer:
    """Computes palm poses and coordinate transformations."""
    
    @staticmethod
    def compute_palm_centroid(keypoints: np.ndarray) -> np.ndarray:
        palm_kps = keypoints[PALM_INDICES]
        valid_mask = ~np.any(palm_kps >= INVALID_VALUE - 1, axis=1)
        if not np.any(valid_mask):
            return np.full(3, INVALID_VALUE)
        return np.mean(palm_kps[valid_mask], axis=0)
    
    @staticmethod
    def compute_palm_orientation(keypoints: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        wrist, index1, middle1, pinky1 = keypoints[0], keypoints[5], keypoints[9], keypoints[17]
        
        if any(np.any(kp >= INVALID_VALUE - 1) for kp in [wrist, index1, pinky1]):
            return np.eye(3), np.zeros(3)
        
        x_axis = middle1 - wrist
        x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-8)
        
        temp_y = pinky1 - wrist
        z_axis = np.cross(x_axis, temp_y)
        z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-8)
        
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-8)
        
        rot_matrix = np.column_stack([x_axis, y_axis, z_axis])
        
        try:
            euler_ypr = R.from_matrix(rot_matrix).as_euler('zyx', degrees=False)
        except Exception:
            euler_ypr = np.zeros(3)
        
        return rot_matrix, euler_ypr
    
    @staticmethod
    def compute_palm_6dof(keypoints: np.ndarray) -> np.ndarray:
        centroid = PoseComputer.compute_palm_centroid(keypoints)
        _, euler_ypr = PoseComputer.compute_palm_orientation(keypoints)
        
        if np.any(centroid >= INVALID_VALUE - 1):
            return np.full(6, INVALID_VALUE)
        
        return np.concatenate([centroid, euler_ypr])
    
    @staticmethod
    def world_to_camera_frame(world_pose: np.ndarray, camera_pose_t: CameraPose, camera_pose_offset: CameraPose) -> np.ndarray:
        if np.any(world_pose >= INVALID_VALUE - 1):
            return np.full(6, INVALID_VALUE)
        
        position = world_pose[:3]
        euler = world_pose[3:6]
        
        T_world_pose = np.eye(4)
        T_world_pose[:3, :3] = R.from_euler('zyx', euler).as_matrix()
        T_world_pose[:3, 3] = position
        
        T_cam_t_inv = np.linalg.inv(camera_pose_t.get_transform_matrix())
        T_cam_t_pose = T_cam_t_inv @ T_world_pose
        
        position_cam = T_cam_t_pose[:3, 3]
        euler_cam = R.from_matrix(T_cam_t_pose[:3, :3]).as_euler('zyx', degrees=False)
        
        return np.concatenate([position_cam, euler_cam])


class EgoverseDatasetWriter:
    """Writes data in the Egoverse LeRobot format."""
    
    def __init__(self, output_dir: str, task_id: str):
        self.output_dir = Path(output_dir)
        self.task_id = task_id
        
        self.data_dir = self.output_dir / "data" / "chunk-000"
        self.meta_dir = self.output_dir / "meta"
        self.anno_dir = self.output_dir / "annotations"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        self.anno_dir.mkdir(parents=True, exist_ok=True)
        
        self.episodes = []
        self.tasks = []
        self.stats_accumulator = {
            'actions_ee_cartesian_cam': [],
            'actions_ee_keypoints_world': [],
            'actions_head_cartesian_world': [],
            'observations.state.ee_pose_cam': [],
        }
        self.next_index = 0
        # SCALE_BIMANUAL embodiment ID from EMBODIMENT enum
        self.metadata_embodiment = 12
    
    def process_and_write_episode(self, frames: List[FrameData], intrinsics: Dict[str, float], source_info: Dict[str, Any]) -> int:
        valid_frames = len(frames) - ACTION_INTERPOLATION_WINDOW
        if valid_frames <= 0:
            print("Warning: Not enough frames for action chunks")
            return 0
        
        task_desc = self._get_task_description(frames)
        task_index = self._get_or_create_task(task_desc)
        
        rows = []
        for t in tqdm(range(valid_frames), desc="Processing frames"):
            row = self._create_parquet_row(frames, t, task_index)
            if row is not None:
                rows.append(row)
        
        if not rows:
            print("Warning: No valid rows created")
            return 0
        
        num_subepisodes = 0
        for start_idx in range(0, len(rows), SUB_EPISODE_LENGTH):
            end_idx = min(start_idx + SUB_EPISODE_LENGTH, len(rows))
            subepisode_rows = rows[start_idx:end_idx]
            
            if len(subepisode_rows) < 10:
                continue
            
            episode_idx = len(self.episodes)
            self._write_parquet(episode_idx, subepisode_rows, task_index)
            self._write_annotations_csv(episode_idx, frames[start_idx:end_idx])
            
            self.episodes.append({
                'episode_index': episode_idx,
                'tasks': [task_desc],
                'length': len(subepisode_rows),
            })
            self._accumulate_stats(subepisode_rows)
            num_subepisodes += 1
        
        return num_subepisodes
    
    def _get_task_description(self, frames: List[FrameData]) -> str:
        for frame in frames:
            for ann in frame.text_annotations:
                if ann.get('text'):
                    return ann['text']
        return "Unknown task"
    
    def _get_or_create_task(self, task_desc: str) -> int:
        for i, task in enumerate(self.tasks):
            if task['task'] == task_desc:
                return i
        task_index = len(self.tasks)
        self.tasks.append({'task_index': task_index, 'task': task_desc})
        return task_index

    def _create_parquet_row(self, frames: List[FrameData], t: int, task_index: int) -> Optional[Dict[str, Any]]:
        frame_t = frames[t]
        camera_pose_t = frame_t.camera_pose
        base_window = ACTION_INTERPOLATION_WINDOW
        
        # Filter out frames with "Inactive Time" in collector_issue
        if frame_t.collector_issue is not None and frame_t.collector_issue.get('issue_type') == 'Inactive Time':
            #print("found")
            return None

        if t + base_window > len(frames):
            return None
        
        # Actions: EE Cartesian in Camera Frame (100, 12)
        left_cartesian_seq = []
        right_cartesian_seq = []
        for offset in range(base_window):
            frame_offset = frames[t + offset]
            
            left_pose = (PoseComputer.world_to_camera_frame(
                PoseComputer.compute_palm_6dof(frame_offset.hand_keypoints.left),
                camera_pose_t, frame_offset.camera_pose
            ) if frame_offset.hand_keypoints.left is not None else np.full(6, INVALID_VALUE))
            
            right_pose = (PoseComputer.world_to_camera_frame(
                PoseComputer.compute_palm_6dof(frame_offset.hand_keypoints.right),
                camera_pose_t, frame_offset.camera_pose
            ) if frame_offset.hand_keypoints.right is not None else np.full(6, INVALID_VALUE))
            
            left_cartesian_seq.append(left_pose)
            right_cartesian_seq.append(right_pose)

        left_cartesian_seq = np.array(left_cartesian_seq)
        right_cartesian_seq = np.array(right_cartesian_seq)

        # Interpolate 30 → 100 using egomimic utils
        left_interp = interpolate_arr_euler(left_cartesian_seq[None, :, :], ACTION_CHUNK_LENGTH)[0]
        right_interp = interpolate_arr_euler(right_cartesian_seq[None, :, :], ACTION_CHUNK_LENGTH)[0]
        actions_ee_cartesian = np.concatenate([left_interp, right_interp], axis=-1)
        
        # Actions: EE Keypoints in World Frame (100, 126)
        actions_ee_keypoints_base = []
        for offset in range(base_window):
            frame_offset = frames[t + offset]
            left_kps = frame_offset.hand_keypoints.left.flatten() if frame_offset.hand_keypoints.left is not None else np.full(63, INVALID_VALUE)
            right_kps = frame_offset.hand_keypoints.right.flatten() if frame_offset.hand_keypoints.right is not None else np.full(63, INVALID_VALUE)
            actions_ee_keypoints_base.append(np.concatenate([left_kps, right_kps]))

        actions_ee_keypoints_base = np.array(actions_ee_keypoints_base)
        actions_ee_keypoints = interpolate_arr(actions_ee_keypoints_base[None, :, :], ACTION_CHUNK_LENGTH)[0]
        
        # Actions: Head Cartesian in World Frame (10,)
        actions_head = np.concatenate([
            camera_pose_t.position,
            camera_pose_t.get_euler_ypr(),
            camera_pose_t.quaternion
        ])
                
        if frame_t.image is None:
            return None
        
        # Encode image to PNG bytes
        with io.BytesIO() as buf:
            img_pil = Image.fromarray(frame_t.image)
            img_pil = img_pil.resize((640, 480))
            img_pil.save(buf, format='PNG')
            image_bytes = buf.getvalue()
        timestamp_s = frame_t.timestamp_us/ 1000000
        
        # Filter frames with too many invalid values
        if np.sum(actions_ee_cartesian >= INVALID_VALUE - 1) > actions_ee_cartesian.size * 0.5:
            return None
        
        actions_ee_cartesian = np.where(actions_ee_cartesian >= INVALID_VALUE - 1, 0.0, actions_ee_cartesian)
        actions_ee_keypoints = np.where(actions_ee_keypoints >= INVALID_VALUE - 1, 0.0, actions_ee_keypoints)

        # Observations
        obs_ee_pose = actions_ee_cartesian[0]
        return {
            'actions_ee_cartesian_cam': actions_ee_cartesian.astype(np.float32),
            'actions_ee_keypoints_world': actions_ee_keypoints.astype(np.float32),
            'actions_head_cartesian_world': actions_head.astype(np.float32),
            'observations.images.front_img_1': {'bytes': image_bytes},
            'observations.state.ee_pose_cam': obs_ee_pose.astype(np.float32),
            'timestamp': timestamp_s,
            'frame_index': int(frame_t.frame_index),
            'task_index': int(task_index),
        }
    
    def _write_parquet(self, episode_idx: int, rows: List[Dict[str, Any]], task_index: int):
        filepath = self.data_dir / f"episode_{episode_idx:06d}.parquet"
        
        for row in rows:
            row['episode_index'] = int(episode_idx)
            row['task_index'] = int(task_index)
            row['index'] = int(self.next_index)
            row['metadata.embodiment'] = int(self.metadata_embodiment)
            self.next_index += 1
        
        # Use Hugging Face Datasets to write parquet with proper Image feature metadata
        data_dict = {}
        for key in rows[0].keys():
            if isinstance(rows[0][key], np.ndarray):
                data_dict[key] = [row[key].tolist() for row in rows]
            else:
                data_dict[key] = [row[key] for row in rows]

        ds = Dataset.from_dict(data_dict)

        # Cast image column to Image feature 
        if 'observations.images.front_img_1' in ds.column_names:
            ds = ds.cast_column('observations.images.front_img_1', ImageFeature())

        ds.to_parquet(filepath)
        print(f"Wrote {len(rows)} frames to {filepath}")
    
    def _write_annotations_csv(self, episode_idx: int, frames: List[FrameData]):
        filepath = self.anno_dir / f"episode_{episode_idx:06d}_annotations.csv"
        
        rows = []
        current_label: Optional[str] = None
        start_ts: Optional[float] = None
        end_ts: Optional[float] = None

        for frame in frames:
            label = frame.subgoal.get('text', '') if frame.subgoal else ''
            label = label if label else None
            ts = frame.timestamp_us / 1_000_000

            if label != current_label:
                if current_label is not None and start_ts is not None and end_ts is not None:
                    rows.append({'Labels': current_label, 'start_time': start_ts, 'end_time': end_ts})
                current_label = label
                start_ts = ts if label is not None else None
                end_ts = None

            if label is not None:
                end_ts = ts

        if current_label is not None and start_ts is not None and end_ts is not None:
            rows.append({'Labels': current_label, 'start_time': start_ts, 'end_time': end_ts})
        
        pd.DataFrame(rows, columns=['Labels', 'start_time', 'end_time']).to_csv(filepath, index=False)
    
    def _accumulate_stats(self, rows: List[Dict[str, Any]]):
        for row in rows:
            for key in self.stats_accumulator.keys():
                if key in row:
                    data = row[key].flatten()
                    valid = data[data < INVALID_VALUE - 1]
                    if len(valid) > 0:
                        self.stats_accumulator[key].append(valid)
    
    def write_metadata(self, intrinsics: Dict[str, float], demonstration_metadata: Dict[str, Any] = None, additional_info: Dict[str, Any] = None):
        # episodes.jsonl
        with open(self.meta_dir / "episodes.jsonl", 'w') as f:
            for ep in self.episodes:
                f.write(json.dumps(ep) + '\n')
        
        # tasks.jsonl
        with open(self.meta_dir / "tasks.jsonl", 'w') as f:
            for task in self.tasks:
                f.write(json.dumps(task) + '\n')
        
        # info.json
        total_frames = int(sum(ep.get('length', 0) for ep in self.episodes))
        chunk_dirs = sorted(self.data_dir.parent.glob("chunk-*"))
        total_chunks = len(chunk_dirs) if chunk_dirs else 1

        train_split = [ep['episode_index'] for ep in self.episodes]
        valid_split = []
        if len(train_split) > 1:
            valid_split = [train_split[-1]]
            train_split = train_split[:-1]

        image_height = 480
        image_width = 640

        feature_spec = {
            'observations.images.front_img_1': {
                'dtype': 'image',
                'shape': [3, image_height, image_width],
                'names': ['channel', 'height', 'width'],
            },
            'actions_ee_cartesian_cam': {
                'dtype': 'prestacked_float32',
                'shape': [ACTION_CHUNK_LENGTH, 12],
                'names': ['chunk_length', 'action_dim'],
            },
            'actions_ee_keypoints_world': {
                'dtype': 'prestacked_float32',
                'shape': [ACTION_CHUNK_LENGTH, 126],
                'names': ['chunk_length', 'keypoint_dim'],
            },
            'actions_head_cartesian_world': {
                'dtype': 'float32',
                'shape': [10],
                'names': ['dim_0'],
            },
            'observations.state.ee_pose_cam': {
                'dtype': 'float32',
                'shape': [12],
                'names': ['dim_0'],
            },
            'timestamp': {
                'dtype': 'float32',
                'shape': [1],
                'names': None,
            },
            'frame_index': {
                'dtype': 'int64',
                'shape': [1],
                'names': None,
            },
            'episode_index': {
                'dtype': 'int64',
                'shape': [1],
                'names': None,
            },
            'index': {
                'dtype': 'int64',
                'shape': [1],
                'names': None,
            },
            'task_index': {
                'dtype': 'int64',
                'shape': [1],
                'names': None,
            },
            'metadata.embodiment': {
                'dtype': 'int32',
                'shape': [1],
                'names': ['dim_0'],
            },
        }

        info = {
            'codebase_version': 'v2.0',
            'robot_type': 'scale_bimanual',
            'total_episodes': len(self.episodes),
            'total_frames': total_frames,
            'total_tasks': len(self.tasks),
            'total_videos': 0,
            'total_chunks': total_chunks,
            'chunks_size': 1000,
            'fps': 30,
            'splits': {
                'train': train_split,
                'valid': valid_split,
            },
            'data_path': 'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet',
            'video_path': 'videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4',
            'features': feature_spec,
        }

        if intrinsics:
            info['camera_intrinsics'] = intrinsics

        if demonstration_metadata:
            info['demonstration'] = {
                'rating': demonstration_metadata.get('Demonstration Rating', ''),
                'outcome_rating': demonstration_metadata.get('Outcome Rating', ''),
                'hand_used': demonstration_metadata.get('Hand Used', ''),
                'grasp_type': demonstration_metadata.get('Grasp Type', ''),
                'description': demonstration_metadata.get('Demonstration', ''),
                'lighting_conditions': demonstration_metadata.get('Lighting Conditions', ''),
                'is_good_video': demonstration_metadata.get('Is Good Video?', ''),
            }

        if additional_info:
            info.update(additional_info)

        with open(self.meta_dir / "info.json", 'w') as f:
            json.dump(info, f, indent=2)
        
        # stats.json
        stats = {}
        for key, values_list in self.stats_accumulator.items():
            if values_list:
                all_values = np.concatenate(values_list)
                stats[key] = {
                    'mean': float(np.mean(all_values)),
                    'std': float(np.std(all_values)),
                    'min': float(np.min(all_values)),
                    'max': float(np.max(all_values)),
                }
        
        with open(self.meta_dir / "stats.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        # env.jsonl
        with open(self.meta_dir / "env.jsonl", 'w') as f:
            f.write(json.dumps({'env_index': 0, 'env': 'egoverse'}) + '\n')
        
        print(f"Metadata written to {self.meta_dir}")


def convert_task_to_egoverse(task_id: str, output_dir: str, download_dir: str = "scale_data") -> int:
    """Convert a single Scale task to Egoverse format."""
    print(f"\n{'='*60}\nProcessing task: {task_id}\n{'='*60}")
    
    task_download_path = os.path.join(download_dir, task_id)
    os.makedirs(task_download_path, exist_ok=True)
    
    print("Retrieving task data...")
    simple_response_dict = get_simple_response_dict_egocentric(task_id)
    if simple_response_dict is None:
        print(f"Error: Could not retrieve task {task_id}")
        return 0
    
    print("Downloading files...")
    local_path_dict = download_from_simple_response_dict(task_download_path, simple_response_dict)
    
    sfs_path = local_path_dict.get('sfs')
    annotations_path = local_path_dict.get('annotations')
    video_path = local_path_dict.get('left_rectified') or local_path_dict.get('left_rgb')
    
    if not all([sfs_path, annotations_path, video_path]):
        print(f"Error: Missing required files")
        return 0
    
    extractor = SFSDataExtractor(sfs_path, annotations_path, video_path)
    frames = extractor.extract_all_frames()
    
    if not frames:
        print("Error: No frames extracted")
        return 0
    
    writer = EgoverseDatasetWriter(output_dir, task_id)
    num_episodes = writer.process_and_write_episode(frames, extractor.intrinsics, {'task_id': task_id})
    
    writer.write_metadata(
        extractor.intrinsics,
        demonstration_metadata=extractor.demonstration_metadata,
        additional_info={'source_task_id': task_id, 'total_source_frames': len(frames)}
    )
    
    print(f"Created {num_episodes} sub-episodes")
    return num_episodes


def main():
    parser = argparse.ArgumentParser(description="Convert Scale SFS data to Egoverse LeRobot format")
    parser.add_argument("--task-ids", type=str, nargs="+", required=True, help="Scale task IDs")
    parser.add_argument("--output-dir", type=str, default="egoverse_dataset", help="Output directory")
    parser.add_argument("--download-dir", type=str, default="scale_data", help="Download directory")
    args = parser.parse_args()
    
    print(f"Converting {len(args.task_ids)} tasks to Egoverse format")
    
    total_episodes = 0
    for task_id in args.task_ids:
        try:
            total_episodes += convert_task_to_egoverse(task_id, args.output_dir, args.download_dir)
        except Exception as e:
            print(f"Error processing {task_id}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}\nConversion complete! Total episodes: {total_episodes}\nOutput: {args.output_dir}\n{'='*60}")


if __name__ == "__main__":
    main()
