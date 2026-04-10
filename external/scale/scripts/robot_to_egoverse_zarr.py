#!/usr/bin/env python3
"""
Scale Robot MCAP+MP4 -> EgoVerse Zarr converter.

Matches Eva's per-arm data format.  Output keys per episode:
    left.obs_ee_pose   (T, 7)   FK(follower) left arm  [xyz_base, wxyz_R_t_e]
    right.obs_ee_pose  (T, 7)   FK(follower) right arm  [xyz_base, wxyz_R_t_e]
    left.cmd_ee_pose   (T, 7)   FK(leader)   left arm  [xyz_base, wxyz_R_t_e]
    right.cmd_ee_pose  (T, 7)   FK(leader)   right arm  [xyz_base, wxyz_R_t_e]
    left.obs_joints    (T, 6)   follower left  revolute joints
    right.obs_joints   (T, 6)   follower right revolute joints
    left.cmd_joints    (T, 6)   leader   left  revolute joints
    right.cmd_joints   (T, 6)   leader   right revolute joints
    left.gripper       (T, 1)   left  gripper
    right.gripper      (T, 1)   right gripper
    observations.images.front_img_1      (T, H, W, 3)  cam_high  JPEG-compressed
    observations.images.cam_low          (T, H, W, 3)  cam_low   JPEG-compressed
    observations.images.left_wrist_img   (T, H, W, 3)  cam_left  JPEG-compressed
    observations.images.right_wrist_img  (T, H, W, 3)  cam_right JPEG-compressed

Processing pipeline:
  1. Download MCAP + 4 MP4s + annotations from Scale API
  2. Parse MCAP: extract leader/follower JointState at ~490Hz
  3. Parse annotations: find active clip window, drop Inactive Time
  4. Trim to active clip, interpolate joints to 30Hz camera grid
  5. Decode MP4 frames, resize, JPEG-encode
  6. FK -> base-frame [xyz, ypr, grip], apply R_t_e, convert to wxyz
  7. Split into per-arm keys matching Eva format
  8. Write each as a Zarr episode via ZarrWriter

Usage:
  python robot_to_egoverse_zarr.py --task-ids TASK1 TASK2 --output-dir ./zarr_out
"""

from __future__ import annotations

import argparse
import os
import struct
import shutil
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import simplejpeg
from mcap.reader import make_reader
from decord import VideoReader, cpu as decord_cpu
from scipy.spatial.transform import Rotation as Rot

from egomimic.rldb.zarr.zarr_writer import ZarrWriter
from egomimic.robot.scale_aloha.scale_aloha_kinematics import ScaleAlohaDualArmFK
from egomimic.utils.pose_utils import xyzw_to_wxyz

def _lazy_scale_imports():
    """Deferred imports that require SCALE_API_KEY at module level."""
    from robot_scale_api import download_robot_task_files, get_robot_task_response
    from robot_mcap_data import (
        MCAPJointExtractor,
        RobotAnnotationExtractor,
        build_aligned_joint_arrays,
    )
    return (
        download_robot_task_files,
        get_robot_task_response,
        MCAPJointExtractor,
        RobotAnnotationExtractor,
        build_aligned_joint_arrays,
    )

MIN_EPISODE_FRAMES = 10
IMAGE_SIZE = (640, 480)  # (W, H) for cv2.resize
CAMERA_KEYS = {
    "cam_high": "observations.images.front_img_1",
    "cam_low": "observations.images.cam_low",
    "cam_left_wrist": "observations.images.left_wrist_img",
    "cam_right_wrist": "observations.images.right_wrist_img",
}
CAMERA_INFO_TOPICS = {
    "/robot/camera/cam_high/image_raw/compressed/camera_info": "cam_high",
    "/robot/camera/cam_low/image_raw/compressed/camera_info": "cam_low",
    "/robot/camera/cam_left_wrist/image_raw/compressed/camera_info": "cam_left_wrist",
    "/robot/camera/cam_right_wrist/image_raw/compressed/camera_info": "cam_right_wrist",
}

R_t_e = np.array(
    [
        [0, 0, 1],
        [-1, 0, 0],
        [0, -1, 0],
    ],
    dtype=float,
)


def _rot_orientation(quat_xyzw: np.ndarray) -> np.ndarray:
    """Apply R_t_e rotation to quaternion orientations (batch)."""
    rotation = Rot.from_quat(quat_xyzw).as_matrix()
    rotation = R_t_e @ rotation
    return Rot.from_matrix(rotation).as_quat()


def _split_per_arm(cart_base: np.ndarray) -> dict[str, np.ndarray]:
    """Split (T, 14) base-frame FK output into per-arm keys with R_t_e + wxyz.

    Input layout per arm: [xyz_base(3), ypr_base(3), gripper(1)] x2
    Output: per-arm keys matching Eva's format:
        {side}.{key} -> (T, 7) for poses [xyz_base, wxyz_R_t_e]
        {side}.gripper -> (T, 1)
    """
    left_xyz = cart_base[:, 0:3]
    left_ypr = cart_base[:, 3:6]
    left_grip = cart_base[:, 6:7]
    right_xyz = cart_base[:, 7:10]
    right_ypr = cart_base[:, 10:13]
    right_grip = cart_base[:, 13:14]

    left_quat = _rot_orientation(
        Rot.from_euler("ZYX", left_ypr, degrees=False).as_quat()
    )
    right_quat = _rot_orientation(
        Rot.from_euler("ZYX", right_ypr, degrees=False).as_quat()
    )
    left_quat = xyzw_to_wxyz(left_quat)
    right_quat = xyzw_to_wxyz(right_quat)

    return {
        "left": np.concatenate([left_xyz, left_quat], axis=-1).astype(np.float32),
        "right": np.concatenate([right_xyz, right_quat], axis=-1).astype(np.float32),
        "left_grip": left_grip.astype(np.float32),
        "right_grip": right_grip.astype(np.float32),
    }


# ---------------------------------------------------------------------------
# MCAP calibration extraction (intrinsics + extrinsics in one pass)
# ---------------------------------------------------------------------------

# Fallback extrinsics (cam-to-base convention) derived from a reference MCAP
# /tf_static.  Used only when tf_static is missing from a particular recording.
_FALLBACK_EXTRINSICS = {
    "left": np.array([
        [ 1.00000000, -0.00000000,  0.00000000,  0.44850000],
        [-0.00000000, -0.90632455,  0.42258232, -0.31017500],
        [ 0.00000000, -0.42258232, -0.90632455,  1.00777400],
        [ 0.00000000,  0.00000000,  0.00000000,  1.00000000],
    ]),
    "right": np.array([
        [-1.00000000, -0.00000000,  0.00000000,  0.46650000],
        [-0.00000000,  0.90632455, -0.42258232,  0.31017500],
        [ 0.00000000, -0.42258232, -0.90632455,  1.00777400],
        [ 0.00000000,  0.00000000,  0.00000000,  1.00000000],
    ]),
}


def _scan_camera_info_k_and_res(data: bytes) -> tuple[np.ndarray, int, int] | None:
    """Extract K matrix and native (width, height) from a CameraInfo CDR message.

    Returns (K_3x3, width, height) or None.
    """
    # Width and height are uint32 fields early in the message.  Scan for a
    # plausible (height, width) pair followed (after the distortion model
    # string and D array) by a valid K matrix.
    for off in range(0, max(0, len(data) - 72)):
        vals = struct.unpack_from("<9d", data, off)
        if (
            100 < vals[0] < 2000
            and abs(vals[1]) < 1e-2
            and 100 < vals[2] < 2000
            and abs(vals[3]) < 1e-2
            and 100 < vals[4] < 2000
            and 100 < vals[5] < 2000
            and abs(vals[6]) < 1e-2
            and abs(vals[7]) < 1e-2
            and 0.99 < vals[8] < 1.01
        ):
            K = np.array(
                [[vals[0], vals[1], vals[2]],
                 [vals[3], vals[4], vals[5]],
                 [vals[6], vals[7], vals[8]]],
                dtype=np.float64,
            )
            # Scan backwards from K for the (height, width) uint32 pair.
            w, h = 0, 0
            for hw_off in range(8, min(off, 200), 4):
                candidate_h = struct.unpack_from("<I", data, off - hw_off)[0]
                candidate_w = struct.unpack_from("<I", data, off - hw_off + 4)[0]
                if 100 < candidate_w < 8000 and 100 < candidate_h < 8000:
                    w, h = candidate_w, candidate_h
                    break
            if w == 0:
                w, h = int(round(vals[2] * 2)), int(round(vals[5] * 2))
            return K, w, h
    return None


def _scan_tf_static_transforms(data: bytes) -> list[dict]:
    """Extract transforms from a tf2_msgs/TFMessage by scanning for unit quaternions."""
    import math
    results = []
    for off in range(0, len(data) - 31):
        try:
            qx, qy, qz, qw = struct.unpack_from("<4d", data, off)
            qnorm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        except (OverflowError, ValueError):
            continue
        if not (0.9999 < qnorm < 1.0001):
            continue
        if off < 24:
            continue
        tx, ty, tz = struct.unpack_from("<3d", data, off - 24)
        if not all(abs(v) < 10 for v in (tx, ty, tz)):
            continue
        # Find the nearest frame-name string before the transform data.
        label = ""
        for s_off in range(max(0, off - 200), off):
            if s_off + 4 > len(data):
                break
            slen = struct.unpack_from("<I", data, s_off)[0]
            if 4 < slen < 80 and s_off + 4 + slen <= off:
                try:
                    s = data[s_off + 4 : s_off + 4 + slen - 1].decode("utf-8")
                    if s.isprintable() and len(s) > 3:
                        label = s
                except Exception:
                    pass
        results.append({"label": label, "xyz": (tx, ty, tz), "quat_xyzw": (qx, qy, qz, qw)})
    # Deduplicate overlapping byte hits.
    seen_offsets: list[tuple[float, ...]] = []
    deduped = []
    for r in results:
        key = r["quat_xyzw"]
        if key not in seen_offsets:
            seen_offsets.append(key)
            deduped.append(r)
    return deduped


def _build_extrinsics_from_tf(
    transforms: list[dict],
) -> dict[str, np.ndarray] | None:
    """Build cam-to-base extrinsics from parsed tf_static transforms.

    Looks for world->left_follower_arm/base_link, world->right_follower_arm/base_link,
    and world->cam_high_frame.  Returns {"left": T_base_from_cam, "right": ...} or None.
    """
    from scipy.spatial.transform import Rotation as Rot

    def _find(keyword: str) -> dict | None:
        for t in transforms:
            if keyword in t["label"]:
                return t
        return None

    cam_tf = _find("cam_high_frame")
    left_tf = _find("left_follower_arm/base_link")
    right_tf = _find("right_follower_arm/base_link")
    if cam_tf is None or left_tf is None or right_tf is None:
        return None

    def _to_mat(tf: dict) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = Rot.from_quat(tf["quat_xyzw"]).as_matrix()
        T[:3, 3] = tf["xyz"]
        return T

    T_world_from_cam = _to_mat(cam_tf)
    T_world_from_left = _to_mat(left_tf)
    T_world_from_right = _to_mat(right_tf)

    return {
        "left": np.linalg.inv(T_world_from_left) @ T_world_from_cam,
        "right": np.linalg.inv(T_world_from_right) @ T_world_from_cam,
    }


def extract_calibration_from_mcap(
    mcap_path: str,
    out_width: int = IMAGE_SIZE[0],
    out_height: int = IMAGE_SIZE[1],
) -> dict[str, Any]:
    """Single-pass MCAP scan that extracts both intrinsics and extrinsics.

    Returns {"intrinsics": {...}, "extrinsics": {"left": 4x4, "right": 4x4}}.
    Extrinsics are cam-to-base convention (inv gives cam_from_base for projection).
    """
    intrinsics: dict[str, dict[str, float]] = {}
    extrinsics: dict[str, np.ndarray] | None = None

    with open(mcap_path, "rb") as f:
        reader = make_reader(f)
        for _schema, channel, message in reader.iter_messages():
            # --- CameraInfo intrinsics ---
            cam_name = CAMERA_INFO_TOPICS.get(channel.topic)
            if cam_name is not None and cam_name not in intrinsics:
                parsed = _scan_camera_info_k_and_res(message.data)
                if parsed is not None:
                    K, native_w, native_h = parsed
                    sx = out_width / native_w
                    sy = out_height / native_h
                    intrinsics[cam_name] = {
                        "fx": float(K[0, 0] * sx),
                        "fy": float(K[1, 1] * sy),
                        "cx": float(K[0, 2] * sx),
                        "cy": float(K[1, 2] * sy),
                        "native_width": native_w,
                        "native_height": native_h,
                        "width": out_width,
                        "height": out_height,
                    }

            # --- tf_static extrinsics ---
            if extrinsics is None and channel.topic == "/tf_static":
                transforms = _scan_tf_static_transforms(message.data)
                extrinsics = _build_extrinsics_from_tf(transforms)

            # Early exit once we have everything.
            if len(intrinsics) == len(CAMERA_INFO_TOPICS) and extrinsics is not None:
                break

    if extrinsics is None:
        extrinsics = _FALLBACK_EXTRINSICS.copy()

    return {"intrinsics": intrinsics, "extrinsics": extrinsics}

# Singleton FK solver (lazy-initialized)
_fk_solver: ScaleAlohaDualArmFK | None = None


def _get_fk_solver() -> ScaleAlohaDualArmFK:
    global _fk_solver
    if _fk_solver is None:
        _fk_solver = ScaleAlohaDualArmFK()
    return _fk_solver


def joints_to_cartesian(joints: np.ndarray) -> np.ndarray:
    """Convert (T, 14) joint positions to (T, 14) Cartesian EE poses in base frame.

    Returns (T, 14): [L_xyz_base, L_ypr_base, L_grip, R_xyz_base, R_ypr_base, R_grip].
    """
    fk = _get_fk_solver()
    return fk.fk_bimanual_batch(joints)


# ---------------------------------------------------------------------------
# Video helpers
# ---------------------------------------------------------------------------


def decode_video_frames(
    video_path: str,
    indices: list[int] | None = None,
    resize: tuple[int, int] | None = IMAGE_SIZE,
) -> list[np.ndarray]:
    """Decode video frames, optionally selecting specific indices and resizing.

    If indices is None, decodes all frames. Returns list of RGB uint8 arrays.
    """
    vr = VideoReader(video_path, ctx=decord_cpu())
    total = len(vr)

    if indices is None:
        indices = list(range(total))
    else:
        indices = [i for i in indices if i < total]

    if not indices:
        return []

    frames = []
    chunk_size = 500
    for start in range(0, len(indices), chunk_size):
        chunk_idx = indices[start : start + chunk_size]
        batch = vr.get_batch(chunk_idx).asnumpy()
        for i in range(len(chunk_idx)):
            frame = batch[i]
            if resize is not None:
                frame = cv2.resize(frame, resize, interpolation=cv2.INTER_LINEAR)
            frames.append(frame)
        del batch

    return frames


def get_video_frame_count(video_path: str) -> int:
    vr = VideoReader(video_path, ctx=decord_cpu())
    return len(vr)


def encode_frames_jpeg(
    frames: list[np.ndarray],
    n_workers: int = 4,
) -> tuple[np.ndarray, list[int]]:
    """JPEG-encode a list of RGB frames. Returns (encoded_array, image_shape)."""

    def _encode_one(frame: np.ndarray) -> bytes:
        return simplejpeg.encode_jpeg(
            frame, quality=ZarrWriter.JPEG_QUALITY, colorspace="RGB"
        )

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        encoded_list = list(pool.map(_encode_one, frames))

    encoded = np.array(encoded_list, dtype=object)
    image_shape = list(frames[0].shape)
    return encoded, image_shape


# ---------------------------------------------------------------------------
# Annotation-based filtering
# ---------------------------------------------------------------------------


def get_active_window_us(
    annotations: RobotAnnotationExtractor,
) -> tuple[int, int] | None:
    """Return the (start_us, end_us) of the active demonstration window.

    Uses the ClipExport annotation to find the active recording segment,
    excluding Inactive Time at the beginning/end.
    """
    clip = annotations.get_active_clip()
    if clip is None:
        return None
    return clip["start_us"], clip["end_us"]


def build_validity_mask(
    timestamps_us: np.ndarray,
    annotations: RobotAnnotationExtractor,
    video_frame_counts: dict[str, int],
    fps: int = 30,
) -> np.ndarray:
    """Boolean mask: True = frame is usable.

    Drops:
      - Frames during Inactive Time collector issues
      - Frames beyond the shortest video's length
    """
    n = len(timestamps_us)
    mask = np.ones(n, dtype=bool)

    for i, ts_us in enumerate(timestamps_us):
        if annotations.is_inactive_time(int(ts_us)):
            mask[i] = False

    if video_frame_counts:
        min_video_frames = min(video_frame_counts.values())
        for i in range(n):
            if i >= min_video_frames:
                mask[i] = False

    return mask


def contiguous_runs(mask: np.ndarray, min_length: int) -> list[list[int]]:
    """Extract contiguous runs of True indices from a boolean mask."""
    runs: list[list[int]] = []
    current: list[int] = []
    for i, val in enumerate(mask):
        if val:
            current.append(i)
        else:
            if len(current) >= min_length:
                runs.append(current)
            current = []
    if len(current) >= min_length:
        runs.append(current)
    return runs


# ---------------------------------------------------------------------------
# Language annotations
# ---------------------------------------------------------------------------


def build_language_annotations(
    timestamps_us: np.ndarray,
    annotations: RobotAnnotationExtractor,
) -> list[tuple[str, int, int]]:
    """Build (text, start_idx, end_idx) tuples from Sub-goal annotations."""
    rows: list[tuple[str, int, int]] = []
    current_text: str | None = None
    start_idx: int | None = None

    for idx, ts_us in enumerate(timestamps_us):
        sg = annotations.get_subgoal_at(int(ts_us))
        text = sg["text"].strip() if sg else None

        if text != current_text:
            if current_text is not None and start_idx is not None:
                rows.append((current_text, start_idx, idx - 1))
            current_text = text
            start_idx = idx if text is not None else None

    if current_text is not None and start_idx is not None:
        rows.append((current_text, start_idx, len(timestamps_us) - 1))

    return rows


# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------


def convert_robot_task_to_zarr(
    task_id: str,
    output_dir: str,
    download_dir: str,
    fps: int = 30,
    img_workers: int | None = None,
) -> dict[str, Any]:
    """Convert one Scale robot task to one or more Zarr episodes.

    Returns a dict with keys: episodes, folder, task_desc, total_frames, output_dir.
    """
    (
        download_robot_task_files,
        get_robot_task_response,
        MCAPJointExtractor,
        RobotAnnotationExtractor,
        build_aligned_joint_arrays,
    ) = _lazy_scale_imports()

    t_start = time.perf_counter()
    if img_workers is None:
        img_workers = min(os.cpu_count() or 4, 8)
    task_download_path = os.path.join(download_dir, task_id)

    try:
        # 1. Fetch metadata and download files
        print(f"[{task_id}] Fetching task metadata...")
        task_response = get_robot_task_response(task_id)
        if task_response is None:
            raise ValueError(f"Task {task_id} not found or Scale API failed")

        print(f"[{task_id}] Downloading files...")
        local_paths = download_robot_task_files(download_dir, task_response)

        mcap_path = local_paths.get("mcap")
        ann_path = local_paths.get("annotations")
        if not mcap_path or not ann_path:
            raise ValueError(f"Missing MCAP or annotation files for task {task_id}")
        if not os.path.exists(mcap_path) or os.path.getsize(mcap_path) == 0:
            raise ValueError(f"MCAP file is empty for task {task_id}")

        calibration = extract_calibration_from_mcap(
            mcap_path, out_width=IMAGE_SIZE[0], out_height=IMAGE_SIZE[1]
        )
        camera_intrinsics = calibration["intrinsics"]
        task_extrinsics = calibration["extrinsics"]
        if camera_intrinsics:
            sample = next(iter(camera_intrinsics.values()))
            print(
                f"[{task_id}] Intrinsics: {len(camera_intrinsics)} cams, "
                f"native {sample.get('native_width', '?')}x{sample.get('native_height', '?')}"
            )
        else:
            print(f"[{task_id}] WARNING: Could not parse camera intrinsics from MCAP")
        used_fallback = calibration["extrinsics"] is _FALLBACK_EXTRINSICS
        if used_fallback:
            print(f"[{task_id}] WARNING: Using fallback extrinsics (no /tf_static in MCAP)")

        # 2. Parse MCAP
        print(f"[{task_id}] Parsing MCAP...")
        mcap = MCAPJointExtractor(mcap_path)
        episode = mcap.get_episode_data()
        print(
            f"[{task_id}] MCAP: {episode.duration_s:.1f}s, "
            f"LF={len(episode.left_follower.timestamps_ns)} "
            f"RF={len(episode.right_follower.timestamps_ns)} "
            f"LL={len(episode.left_leader.timestamps_ns)} "
            f"RL={len(episode.right_leader.timestamps_ns)} msgs"
        )

        # 3. Parse annotations
        print(f"[{task_id}] Parsing annotations...")
        annotations = RobotAnnotationExtractor(ann_path)
        task_desc = annotations.demonstration_label or task_response.get("demonstration", "Unknown task")

        active_window = get_active_window_us(annotations)
        if active_window is None:
            print(f"[{task_id}] WARNING: No ClipExport annotation, using full recording")
            start_ns = episode.epoch_ns
            end_ns = start_ns + int(episode.duration_s * 1e9)
        else:
            start_us, end_us = active_window
            start_ns = episode.epoch_ns + start_us * 1000
            end_ns = episode.epoch_ns + end_us * 1000
            print(f"[{task_id}] Active window: {start_us/1e6:.1f}s - {end_us/1e6:.1f}s")

        # 4. Interpolate joints to camera FPS
        aligned = build_aligned_joint_arrays(
            episode, fps=fps, start_ns=start_ns, end_ns=end_ns
        )
        obs_joints = aligned["observation_joints"]   # (T, 14)
        act_joints = aligned["action_joints"]        # (T, 14)
        timestamps_ns = aligned["timestamps_ns"]
        n_frames = len(timestamps_ns)
        print(f"[{task_id}] Aligned: {n_frames} frames at {fps}Hz")

        if n_frames < MIN_EPISODE_FRAMES:
            raise ValueError(f"Task {task_id} has too few frames ({n_frames})")

        # 4b. FK: joint positions -> Cartesian EE poses in base frame
        print(f"[{task_id}] Computing FK...")
        ee_pose_base = joints_to_cartesian(obs_joints)
        actions_cart_base = joints_to_cartesian(act_joints)
        print(f"[{task_id}] FK done: ee_pose {ee_pose_base.shape}, actions {actions_cart_base.shape}")

        # 5. Probe video frame counts
        video_frame_counts: dict[str, int] = {}
        cam_paths: dict[str, str] = {}
        for cam_name in ("cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"):
            path = local_paths.get(cam_name)
            if path and os.path.exists(path):
                fc = get_video_frame_count(path)
                video_frame_counts[cam_name] = fc
                cam_paths[cam_name] = path
                print(f"[{task_id}]   {cam_name}: {fc} frames")

        if not cam_paths:
            raise ValueError(f"No video files found for task {task_id}")

        # 6. Build validity mask
        timestamps_from_start_us = ((timestamps_ns - episode.epoch_ns) / 1000).astype(np.int64)
        validity = build_validity_mask(
            timestamps_from_start_us, annotations, video_frame_counts, fps
        )
        valid_count = int(validity.sum())
        dropped = n_frames - valid_count
        print(f"[{task_id}] Validity: {valid_count}/{n_frames} kept ({dropped} dropped)")

        runs = contiguous_runs(validity, min_length=MIN_EPISODE_FRAMES)
        if not runs:
            raise ValueError(f"Task {task_id} has no valid contiguous runs after filtering")

        print(f"[{task_id}] {len(runs)} episode(s) from contiguous runs")

        # Compute which video frame indices correspond to our aligned timestamps.
        # The active window start corresponds to a video frame offset.
        if active_window:
            start_us, _ = active_window
            video_start_frame = int(start_us / 1e6 * fps)
        else:
            video_start_frame = 0

        # 8. Write Zarr episodes
        folder = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H-%M-%S-%f")
        task_output_dir = Path(output_dir) / folder
        task_output_dir.mkdir(parents=True, exist_ok=True)

        written = 0
        for sub in runs:
            T = len(sub)
            sub_arr = np.array(sub)

            video_indices = [video_start_frame + idx for idx in sub]

            obs_split = _split_per_arm(ee_pose_base[sub_arr])
            act_split = _split_per_arm(actions_cart_base[sub_arr])

            numeric_data = {
                "left.obs_ee_pose": obs_split["left"],
                "right.obs_ee_pose": obs_split["right"],
                "left.cmd_ee_pose": act_split["left"],
                "right.cmd_ee_pose": act_split["right"],
                "left.obs_joints": obs_joints[sub_arr][:, :6].astype(np.float32),
                "right.obs_joints": obs_joints[sub_arr][:, 7:13].astype(np.float32),
                "left.cmd_joints": act_joints[sub_arr][:, :6].astype(np.float32),
                "right.cmd_joints": act_joints[sub_arr][:, 7:13].astype(np.float32),
                "left.gripper": act_split["left_grip"],
                "right.gripper": act_split["right_grip"],
            }

            pre_encoded: dict[str, tuple[np.ndarray, list[int]]] = {}
            for cam_name, zarr_key in CAMERA_KEYS.items():
                path = cam_paths.get(cam_name)
                if path is None:
                    continue
                frames = decode_video_frames(path, video_indices, resize=IMAGE_SIZE)
                if len(frames) < T:
                    print(
                        f"[{task_id}] WARNING: {cam_name} decoded {len(frames)}/{T} frames, padding"
                    )
                    while len(frames) < T:
                        frames.append(frames[-1])
                frames = frames[:T]

                encoded, img_shape = encode_frames_jpeg(frames, n_workers=img_workers)
                pre_encoded[zarr_key] = (encoded, img_shape)
                del frames

            lang_ann = build_language_annotations(
                timestamps_from_start_us[sub_arr], annotations
            )

            episode_path = task_output_dir / f"{task_id}_episode_{written:06d}.zarr"
            metadata_override = {"camera_intrinsics": camera_intrinsics} if camera_intrinsics else None
            ZarrWriter.create_and_write(
                episode_path=episode_path,
                numeric_data=numeric_data,
                pre_encoded_image_data=pre_encoded,
                embodiment="scale_aloha_bimanual",
                fps=fps,
                task_description=task_desc,
                annotations=lang_ann if lang_ann else None,
                metadata_override=metadata_override,
            )
            written += 1
            print(f"[{task_id}] Wrote episode {written} ({T} frames) -> {episode_path.name}")

        elapsed = time.perf_counter() - t_start
        print(f"[{task_id}] Done: {written} episode(s) in {elapsed:.1f}s -> {task_output_dir}")
        return {
            "episodes": written,
            "folder": folder,
            "task_desc": task_desc,
            "total_frames": n_frames,
            "output_dir": str(task_output_dir),
        }
    finally:
        if os.path.exists(task_download_path):
            shutil.rmtree(task_download_path, ignore_errors=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _find_existing_task_ids(output_dir: str) -> set[str]:
    """Scan output_dir for already-converted task IDs (from zarr folder names)."""
    existing = set()
    out = Path(output_dir)
    if not out.exists():
        return existing
    for zarr_dir in out.rglob("*.zarr"):
        # Folder names are like <task_id>_episode_000000.zarr
        stem = zarr_dir.stem
        if "_episode_" in stem:
            existing.add(stem.rsplit("_episode_", 1)[0])
    return existing


def _convert_one(args_tuple: tuple) -> tuple[str, dict | None, str | None]:
    """Top-level worker for ProcessPoolExecutor (must be picklable)."""
    task_id, output_dir, download_dir, fps = args_tuple
    try:
        result = convert_robot_task_to_zarr(
            task_id=task_id,
            output_dir=output_dir,
            download_dir=download_dir,
            fps=fps,
        )
        return (task_id, result, None)
    except Exception as exc:
        traceback.print_exc()
        return (task_id, None, str(exc))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert Scale robot MCAP+MP4 tasks to EgoVerse Zarr episodes"
    )
    parser.add_argument("--task-ids", nargs="+", help="Scale task IDs")
    parser.add_argument("--task-file", help="Text file with one task ID per line")
    parser.add_argument("--output-dir", default="robot_zarr_dataset", help="Output root")
    parser.add_argument("--download-dir", default="robot_data", help="Temp download dir")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--workers", "-j", type=int, default=4,
                        help="Parallel workers (default 4, set 1 for sequential)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Re-convert tasks even if output already exists")
    args = parser.parse_args()

    task_ids: list[str] = []
    if args.task_file:
        with open(args.task_file) as f:
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                task_ids.append(stripped)
    if args.task_ids:
        task_ids.extend(args.task_ids)
    if not task_ids:
        parser.error("Provide task IDs via --task-ids or --task-file")

    task_ids = list(dict.fromkeys(task_ids))

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.download_dir).mkdir(parents=True, exist_ok=True)

    skipped = 0
    if not args.no_resume:
        existing = _find_existing_task_ids(args.output_dir)
        before = len(task_ids)
        task_ids = [t for t in task_ids if t not in existing]
        skipped = before - len(task_ids)
        if skipped:
            print(f"Resuming: skipping {skipped} already-converted tasks")

    n_tasks = len(task_ids)
    n_workers = min(args.workers, n_tasks) if n_tasks > 0 else 1
    print(f"Tasks to convert: {n_tasks} ({n_workers} workers)")

    total_episodes = 0
    failed: list[str] = []
    t0 = time.perf_counter()

    work = [(tid, args.output_dir, args.download_dir, args.fps) for tid in task_ids]

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_convert_one, w): w[0] for w in work}
        for i, future in enumerate(as_completed(futures), start=1):
            task_id = futures[future]
            tid, result, error = future.result()
            if error:
                print(f"[{i}/{n_tasks}] FAILED {tid}: {error}")
                failed.append(tid)
            else:
                total_episodes += result["episodes"]
                print(f"[{i}/{n_tasks}] OK {tid}: {result['episodes']} episodes")

    elapsed = time.perf_counter() - t0
    print(f"\n{'=' * 60}")
    print(f"Conversion complete in {elapsed:.0f}s: {n_tasks + skipped} total tasks, "
          f"{skipped} skipped (already done), "
          f"{n_tasks - len(failed)} converted, {len(failed)} failed")
    print(f"Episodes written: {total_episodes}")
    if failed:
        print(f"Failed: {failed}")
    print(f"Output: {Path(args.output_dir).resolve()}")
    print("=" * 60)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
