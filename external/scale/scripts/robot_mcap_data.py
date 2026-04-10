"""
MCAP data extraction for robot (ALOHA-style) teleoperation recordings.

Parses ROS2 JointState messages from MCAP files, aligns leader/follower
joint data to a regular timestamp grid, and extracts annotation metadata.

Provides:
  - MCAPJointExtractor: parses MCAP → per-topic timestamped joint arrays
  - RobotAnnotationExtractor: parses Scale annotation JSON → clips + metadata
  - interpolate_to_grid: resamples irregular joint data to fixed FPS
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from mcap.reader import make_reader

from robot_scale_api import load_annotation_file

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

JOINT_TOPICS = {
    "/robot/arm_left_follower/joint_states": "left_follower",
    "/robot/arm_right_follower/joint_states": "right_follower",
    "/robot/arm_left_leader/joint_states": "left_leader",
    "/robot/arm_right_leader/joint_states": "right_leader",
}

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_angle",
    "wrist_rotate",
    "gripper",
    "gripper_finger",
]

NUM_JOINTS_PER_ARM = 7
POS_BYTE_OFFSET = 156  # verified offset for position float64s in CDR-encoded JointState


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class JointStream:
    """Timestamped joint position data for a single arm topic."""
    name: str
    timestamps_ns: np.ndarray
    positions: np.ndarray  # (N, 7) float64

    @property
    def duration_s(self) -> float:
        if len(self.timestamps_ns) < 2:
            return 0.0
        return (self.timestamps_ns[-1] - self.timestamps_ns[0]) / 1e9

    @property
    def rate_hz(self) -> float:
        if len(self.timestamps_ns) < 2:
            return 0.0
        dt = np.median(np.diff(self.timestamps_ns)) / 1e9
        return 1.0 / dt if dt > 0 else 0.0


@dataclass
class RobotEpisodeData:
    """All joint streams and metadata for a single episode."""
    left_follower: JointStream
    right_follower: JointStream
    left_leader: JointStream
    right_leader: JointStream
    epoch_ns: int
    duration_s: float

    # Annotation-derived fields
    demonstration: str = ""
    clips: list = field(default_factory=list)
    subgoals: list = field(default_factory=list)
    collector_issues: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# MCAP parsing
# ---------------------------------------------------------------------------


class MCAPJointExtractor:
    """Extracts JointState position data from an MCAP file.

    Uses a fixed byte offset (POS_BYTE_OFFSET) into the CDR-serialized
    JointState message to read positions without a full ROS2 deserializer.
    """

    def __init__(self, mcap_path: str):
        self.mcap_path = mcap_path
        self.streams: Dict[str, JointStream] = {}
        self._parse()

    def _parse(self) -> None:
        raw: Dict[str, Dict] = {}

        with open(self.mcap_path, "rb") as f:
            reader = make_reader(f)
            for _schema, channel, message in reader.iter_messages():
                arm_name = JOINT_TOPICS.get(channel.topic)
                if arm_name is None:
                    continue

                if arm_name not in raw:
                    raw[arm_name] = {"times": [], "positions": []}

                data = message.data
                sec = struct.unpack_from("<I", data, 4)[0]
                nsec = struct.unpack_from("<I", data, 8)[0]
                ts_ns = sec * 1_000_000_000 + nsec

                positions = np.array(
                    [
                        struct.unpack_from("<d", data, POS_BYTE_OFFSET + i * 8)[0]
                        for i in range(NUM_JOINTS_PER_ARM)
                    ]
                )

                raw[arm_name]["times"].append(ts_ns)
                raw[arm_name]["positions"].append(positions)

        for arm_name, d in raw.items():
            times = np.array(d["times"], dtype=np.int64)
            positions = np.array(d["positions"], dtype=np.float64)
            order = np.argsort(times)
            self.streams[arm_name] = JointStream(
                name=arm_name,
                timestamps_ns=times[order],
                positions=positions[order],
            )

    @property
    def epoch_ns(self) -> int:
        return min(s.timestamps_ns[0] for s in self.streams.values())

    @property
    def end_ns(self) -> int:
        return max(s.timestamps_ns[-1] for s in self.streams.values())

    @property
    def duration_s(self) -> float:
        return (self.end_ns - self.epoch_ns) / 1e9

    def get_episode_data(self) -> RobotEpisodeData:
        return RobotEpisodeData(
            left_follower=self.streams["left_follower"],
            right_follower=self.streams["right_follower"],
            left_leader=self.streams["left_leader"],
            right_leader=self.streams["right_leader"],
            epoch_ns=self.epoch_ns,
            duration_s=self.duration_s,
        )


# ---------------------------------------------------------------------------
# Interpolation to regular grid
# ---------------------------------------------------------------------------


def interpolate_to_grid(
    stream: JointStream,
    target_timestamps_ns: np.ndarray,
) -> np.ndarray:
    """Interpolate a JointStream's positions to a regular timestamp grid.

    Uses linear interpolation per joint. Clamps to nearest value outside
    the stream's time range (no extrapolation).

    Returns (T, 7) float32 array aligned to target_timestamps_ns.
    """
    T = len(target_timestamps_ns)
    n_joints = stream.positions.shape[1]
    result = np.zeros((T, n_joints), dtype=np.float32)

    src_times = stream.timestamps_ns.astype(np.float64)
    tgt_times = target_timestamps_ns.astype(np.float64)

    for j in range(n_joints):
        result[:, j] = np.interp(tgt_times, src_times, stream.positions[:, j])

    return result


def build_aligned_joint_arrays(
    episode: RobotEpisodeData,
    fps: int = 30,
    start_ns: Optional[int] = None,
    end_ns: Optional[int] = None,
) -> Dict[str, Any]:
    """Build time-aligned (T, 14) observation and action arrays at target FPS.

    Returns dict with:
        observation_joints: (T, 14) follower positions [left_7 | right_7]
        action_joints:      (T, 14) leader positions   [left_7 | right_7]
        timestamps_ns:      (T,) absolute timestamps
        timestamps_s:       (T,) relative timestamps from start
    """
    if start_ns is None:
        start_ns = episode.epoch_ns
    if end_ns is None:
        end_ns = start_ns + int(episode.duration_s * 1e9)

    dt_ns = int(1e9 / fps)
    target_ts = np.arange(start_ns, end_ns, dt_ns, dtype=np.int64)

    lf = interpolate_to_grid(episode.left_follower, target_ts)
    rf = interpolate_to_grid(episode.right_follower, target_ts)
    ll = interpolate_to_grid(episode.left_leader, target_ts)
    rl = interpolate_to_grid(episode.right_leader, target_ts)

    return {
        "observation_joints": np.concatenate([lf, rf], axis=1),  # (T, 14)
        "action_joints": np.concatenate([ll, rl], axis=1),       # (T, 14)
        "timestamps_ns": target_ts,
        "timestamps_s": (target_ts - target_ts[0]).astype(np.float64) / 1e9,
    }


# ---------------------------------------------------------------------------
# Annotation extraction
# ---------------------------------------------------------------------------


class RobotAnnotationExtractor:
    """Parses Scale annotation JSON for robot tasks.

    Extracts ClipExport segments, Sub-goal segments, Collector Issues,
    and demonstration-level metadata.
    """

    def __init__(self, annotation_path: str):
        self.data = load_annotation_file(annotation_path)
        if self.data is None:
            raise ValueError(f"Failed to load annotation file: {annotation_path}")

        self.clips: List[Dict[str, Any]] = []
        self.subgoals: List[Dict[str, Any]] = []
        self.collector_issues: List[Dict[str, Any]] = []
        self.demonstration_metadata: Dict[str, Any] = {}

        self._parse()

    def _parse(self) -> None:
        for attr in self.data.get("attributes", []):
            values = attr.get("values", [])
            if values:
                self.demonstration_metadata[attr.get("name", "")] = values[0]

        for annotation in self.data.get("annotations", []):
            atype = annotation.get("type")
            label = annotation.get("label", "")

            if atype != "text_annotation":
                continue

            for clip in annotation.get("clips", []):
                start_us = int(clip.get("timestamp", 0))
                duration_us = int(clip.get("duration", 0))
                text = clip.get("text", "")

                attr_dict = {}
                for attr in clip.get("attributes", []):
                    vals = attr.get("values", [])
                    if vals:
                        attr_dict[attr.get("name", "")] = vals[0]

                entry = {
                    "text": text,
                    "start_us": start_us,
                    "end_us": start_us + duration_us,
                    "label": label,
                    "attributes": attr_dict,
                }

                if label == "ClipExport":
                    self.clips.append(entry)
                elif label == "Sub-goal":
                    self.subgoals.append(entry)
                elif label == "Collector Issue":
                    issue_type = attr_dict.get("Collector Quality Issue", "")
                    entry["issue_type"] = issue_type
                    self.collector_issues.append(entry)

    @property
    def demonstration_label(self) -> str:
        return str(self.demonstration_metadata.get("Demonstration", "")).strip()

    def is_inactive_time(self, timestamp_us: int) -> bool:
        for issue in self.collector_issues:
            if (
                issue.get("issue_type") == "Inactive Time"
                and issue["start_us"] <= timestamp_us <= issue["end_us"]
            ):
                return True
        return False

    def get_subgoal_at(self, timestamp_us: int) -> Optional[Dict[str, Any]]:
        for sg in self.subgoals:
            if sg["start_us"] <= timestamp_us <= sg["end_us"]:
                return sg
        return None

    def get_active_clip(self) -> Optional[Dict[str, Any]]:
        """Return the main ClipExport entry (the active demonstration window)."""
        if self.clips:
            return self.clips[0]
        return None
