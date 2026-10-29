"""Local, display-only MP4 recording for the YAM rollout dashboard.

The rollout loop owns this recorder.  It observes the existing camera frames
only; it does not create a second robot command path.
"""

from __future__ import annotations

import json
import re
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import numpy as np

VIDEO_RECORDING_DEFAULTS = {
    "enabled": False,
    "directory": "/home/rohan/rollouts/yam_hptflow",
    "fps": 12,
}
_VIDEO_ID = re.compile(r"^rollout_\d{8}-\d{6}-\d{6}$")


def validate_video_recording(config: Mapping[str, object] | None) -> dict:
    """Validate the local MP4-output policy without touching cameras or disk."""
    config = {} if config is None else dict(config)
    unknown = config.keys() - VIDEO_RECORDING_DEFAULTS.keys()
    if unknown:
        raise ValueError(
            "Unknown video_recording option(s): " + ", ".join(sorted(unknown))
        )
    result = {**deepcopy(VIDEO_RECORDING_DEFAULTS), **config}
    if type(result["enabled"]) is not bool:
        raise ValueError("video_recording.enabled must be a boolean")
    directory = result["directory"]
    if not isinstance(directory, str) or not directory:
        raise ValueError("video_recording.directory must be a nonempty path")
    if not Path(directory).is_absolute():
        raise ValueError("video_recording.directory must be an absolute path")
    fps = result["fps"]
    if type(fps) is not int or not 1 <= fps <= 30:
        raise ValueError("video_recording.fps must be an integer in [1, 30]")
    return result


def _video_id() -> str:
    return "rollout_" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")


def _is_video_id(value: object) -> bool:
    return isinstance(value, str) and _VIDEO_ID.fullmatch(value) is not None


def list_rollout_videos(directory: str | Path) -> list[dict]:
    """Return completed, browser-safe rollout MP4 metadata newest first."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    videos = []
    for manifest_path in directory.glob("rollout_*.json"):
        try:
            payload = json.loads(manifest_path.read_text())
            video_id = payload["id"]
            filename = payload["filename"]
            frames = payload["frames"]
            fps = payload["fps"]
            if (
                not _is_video_id(video_id)
                or filename != f"{video_id}.mp4"
                or type(frames) is not int
                or frames <= 0
                or type(fps) is not int
                or fps <= 0
            ):
                continue
            path = directory / filename
            if not path.is_file() or path.parent != directory:
                continue
            videos.append(
                {
                    "id": video_id,
                    "filename": filename,
                    "frames": frames,
                    "fps": fps,
                    "duration_seconds": frames / fps,
                    "created_at": payload.get("created_at", ""),
                }
            )
        except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
            continue
    return sorted(videos, key=lambda video: video["id"], reverse=True)


def rollout_video_path(directory: str | Path, video_id: str) -> Path:
    """Resolve one completed MP4 by its manifest-backed public ID."""
    if not _is_video_id(video_id):
        raise FileNotFoundError("Unknown rollout video")
    for video in list_rollout_videos(directory):
        if video["id"] == video_id:
            return Path(directory) / video["filename"]
    raise FileNotFoundError("Rollout video does not exist or is incomplete")


class RolloutVideoRecorder:
    """Write a synchronized, labelled BGR camera mosaic at a bounded cadence."""

    def __init__(self, cameras, config: Mapping[str, object]) -> None:
        self.cameras = tuple(cameras)
        if not self.cameras or len(set(self.cameras)) != len(self.cameras):
            raise ValueError("A rollout video needs distinct configured camera names")
        self.config = validate_video_recording(config)
        self.directory = Path(self.config["directory"])
        self._writer = None
        self._video_id: str | None = None
        self._final_path: Path | None = None
        self._partial_path: Path | None = None
        self._created_at: str | None = None
        self._tile_size: tuple[int, int] | None = None
        self._frames = 0
        self._next_frame_at = 0.0

    @property
    def recording(self) -> bool:
        return self._video_id is not None

    def start(self) -> str:
        """Start a new video session; no file is published until ``close``."""
        if self.recording:
            raise RuntimeError("A rollout video is already recording")
        self.directory.mkdir(parents=True, exist_ok=True)
        video_id = _video_id()
        while (self.directory / f"{video_id}.mp4").exists() or (
            self.directory / f"{video_id}.json"
        ).exists():
            time.sleep(0.001)
            video_id = _video_id()
        self._video_id = video_id
        self._final_path = self.directory / f"{video_id}.mp4"
        self._partial_path = self.directory / f".{video_id}.partial.mp4"
        self._created_at = datetime.now(timezone.utc).isoformat()
        self._tile_size = None
        self._frames = 0
        self._next_frame_at = 0.0
        return video_id

    @staticmethod
    def _first_frame(frames: Mapping[str, object]) -> np.ndarray | None:
        for frame in frames.values():
            frame = np.asarray(frame)
            if frame.ndim == 3 and frame.shape[2] == 3 and frame.dtype == np.uint8:
                if frame.shape[0] > 0 and frame.shape[1] > 0:
                    return frame
        return None

    def _mosaic(self, frames: Mapping[str, object]) -> np.ndarray | None:
        import cv2

        first = self._first_frame(frames)
        if first is None:
            return None
        if self._tile_size is None:
            # Most common MPEG encoders require even dimensions.
            height = max(2, first.shape[0] - first.shape[0] % 2)
            width = max(2, first.shape[1] - first.shape[1] % 2)
            self._tile_size = (width, height)
        width, height = self._tile_size
        tiles = []
        for name in self.cameras:
            frame = frames.get(name)
            frame = None if frame is None else np.asarray(frame)
            if (
                frame is None
                or frame.ndim != 3
                or frame.shape[2] != 3
                or frame.dtype != np.uint8
            ):
                tile = np.zeros((height, width, 3), dtype=np.uint8)
                label = f"{name} unavailable"
                color = (100, 100, 255)
            else:
                tile = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                label = name.replace("_", " ")
                color = (230, 230, 230)
            cv2.rectangle(tile, (0, 0), (width, 30), (0, 0, 0), thickness=-1)
            cv2.putText(
                tile,
                label,
                (10, 21),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                1,
                cv2.LINE_AA,
            )
            tiles.append(tile)
        return np.concatenate(tiles, axis=1)

    def append(self, frames: Mapping[str, object], now: float | None = None) -> bool:
        """Append at most one sampled mosaic frame; return whether one was written."""
        if not self.recording:
            return False
        now = time.monotonic() if now is None else float(now)
        if now < self._next_frame_at:
            return False
        mosaic = self._mosaic(frames)
        if mosaic is None:
            return False
        import cv2

        if self._writer is None:
            assert self._partial_path is not None
            height, width = mosaic.shape[:2]
            writer = cv2.VideoWriter(
                str(self._partial_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                float(self.config["fps"]),
                (width, height),
            )
            if not writer.isOpened():
                writer.release()
                raise RuntimeError("Could not open the local MP4 video writer")
            self._writer = writer
        self._writer.write(mosaic)
        self._frames += 1
        self._next_frame_at = now + 1 / self.config["fps"]
        return True

    def close(self) -> dict | None:
        """Finalize and publish a valid MP4 plus manifest, or return no artifact."""
        if not self.recording:
            return None
        writer, video_id = self._writer, self._video_id
        final_path, partial_path = self._final_path, self._partial_path
        created_at, frames = self._created_at, self._frames
        self._writer = None
        self._video_id = self._final_path = self._partial_path = None
        self._created_at = None
        self._tile_size = None
        self._frames = 0
        if writer is None or frames == 0:
            if writer is not None:
                writer.release()
            return None
        writer.release()
        assert (
            video_id is not None
            and final_path is not None
            and partial_path is not None
            and created_at is not None
        )
        if not partial_path.is_file():
            raise RuntimeError("The MP4 video writer did not create an output file")
        partial_path.replace(final_path)
        payload = {
            "id": video_id,
            "filename": final_path.name,
            "frames": frames,
            "fps": self.config["fps"],
            "created_at": created_at,
            "cameras": list(self.cameras),
        }
        manifest = self.directory / f"{video_id}.json"
        temporary_manifest = self.directory / f".{video_id}.partial.json"
        temporary_manifest.write_text(json.dumps(payload, indent=2) + "\n")
        temporary_manifest.replace(manifest)
        return payload
