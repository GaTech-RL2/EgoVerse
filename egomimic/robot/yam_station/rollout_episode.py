"""Record a YAM rollout as a demonstration-format HDF5 episode.

The file's top level is exactly what ``collect_demo.EpisodeWriter`` writes for a
GELLO demonstration: measured joints and EEF poses, the joint command sent to
each arm, the forward kinematics of that command, and one RGB frame per camera
per row. A rollout episode therefore replays, converts and loads like a demo.
Rollout-only data lives under ``rollout/``: per-row timestamps and plan index,
the policy's Cartesian target, and every plan with its inference settings.

A background thread owns the file, so the multi-megabyte image rows are never
written inside the control tick; the rollout loop only enqueues copies. Like the
MP4 recorder, this observes the rollout and has no robot command path.
"""

from __future__ import annotations

import json
import math
import queue
import re
import shutil
import threading
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import h5py
import numpy as np

from egomimic.robot.yam_station.collect_demo import EpisodeWriter

EPISODE_RECORDING_DEFAULTS = {
    "enabled": False,
    "directory": "/home/rohan/rollouts/yam_hptflow/episodes",
    "min_free_gb": 20.0,
    "max_queued_rows": 60,
}
OUTCOMES = ("success", "failure", "unlabeled")
END_REASONS = (
    "saved",
    "restart",
    "model_change",
    "camera_reconnect",
    "stop",
    "error",
    "low_disk",
    "writer_error",
)
_EPISODE_ID = re.compile(r"^rollout_\d{8}-\d{6}-\d{6}$")
_DISK_CHECK_SECONDS = 1.0
_END_POLL_SECONDS = 0.05


def validate_episode_recording(config: Mapping[str, object] | None) -> dict:
    """Validate the local episode-output policy without touching cameras or disk."""
    config = {} if config is None else dict(config)
    unknown = config.keys() - EPISODE_RECORDING_DEFAULTS.keys()
    if unknown:
        raise ValueError(
            "Unknown episode_recording option(s): " + ", ".join(sorted(unknown))
        )
    result = {**deepcopy(EPISODE_RECORDING_DEFAULTS), **config}
    if type(result["enabled"]) is not bool:
        raise ValueError("episode_recording.enabled must be a boolean")
    directory = result["directory"]
    if not isinstance(directory, str) or not directory:
        raise ValueError("episode_recording.directory must be a nonempty path")
    if not Path(directory).is_absolute():
        raise ValueError("episode_recording.directory must be an absolute path")
    free = result["min_free_gb"]
    if (
        isinstance(free, bool)
        or not isinstance(free, (int, float))
        or not math.isfinite(free)
        or free < 0
    ):
        raise ValueError("episode_recording.min_free_gb must be a finite number >= 0")
    result["min_free_gb"] = float(free)
    rows = result["max_queued_rows"]
    if type(rows) is not int or not 1 <= rows <= 600:
        raise ValueError(
            "episode_recording.max_queued_rows must be an integer in [1, 600]"
        )
    return result


def _episode_id() -> str:
    return "rollout_" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")


def list_rollout_episodes(directory: str | Path) -> list[dict]:
    """Return finalized rollout episode manifests, newest first."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    episodes = []
    for manifest_path in directory.glob("rollout_*.json"):
        try:
            payload = json.loads(manifest_path.read_text())
            episode_id, filename = payload["id"], payload["filename"]
            if (
                not isinstance(episode_id, str)
                or _EPISODE_ID.fullmatch(episode_id) is None
                or filename != f"{episode_id}.hdf5"
                or type(payload["frames"]) is not int
                or payload["frames"] <= 0
            ):
                continue
            path = directory / filename
            if not path.is_file():
                continue
            episodes.append({**payload, "size_bytes": path.stat().st_size})
        except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
            continue
    return sorted(episodes, key=lambda episode: episode["id"], reverse=True)


def _attr(value):
    """HDF5-storable attribute: scalars as is, anything else as JSON text."""
    if value is None:
        return ""
    if isinstance(value, (bool, int, float, str, np.integer, np.floating)):
        return value
    return json.dumps(value, default=str, sort_keys=True)


class RolloutEpisodeRecorder:
    """One rollout episode at a time, written to disk by a background thread."""

    def __init__(self, camera_res, config: Mapping[str, object]) -> None:
        self.camera_res = {
            str(name): (int(height), int(width))
            for name, (height, width) in dict(camera_res).items()
        }
        if not self.camera_res:
            raise ValueError("A rollout episode needs at least one configured camera")
        self.config = validate_episode_recording(config)
        self.directory = Path(self.config["directory"])
        self._lock = threading.Lock()
        self._finished: list[dict] = []
        self._thread: threading.Thread | None = None
        self._queue: queue.Queue | None = None
        self._request: tuple | None = None
        self._episode_id: str | None = None
        self._failure: str | None = None
        self._failure_detail: str | None = None
        self._rows = 0
        self._plan_index = -1
        self._t0 = 0.0
        self._next_disk_check = 0.0

    @property
    def recording(self) -> bool:
        """True from ``start`` until ``close``/``discard`` (finalizing excluded)."""
        return self._episode_id is not None

    @property
    def busy(self) -> bool:
        """True while a closed episode is still being written out."""
        return self._thread is not None and self._thread.is_alive()

    @property
    def frames(self) -> int:
        return self._rows

    @property
    def failure(self) -> str | None:
        """``low_disk`` or ``writer_error`` once recording can no longer continue."""
        return self._failure

    def free_gb(self) -> float:
        return shutil.disk_usage(self.directory).free / 1e9

    def start(self, metadata: Mapping[str, object] | None = None) -> str:
        """Create the episode file; rows are accepted until ``close``."""
        if self.recording:
            raise RuntimeError("A rollout episode is already recording")
        if self.busy:
            raise RuntimeError("The previous rollout episode is still being saved")
        self.directory.mkdir(parents=True, exist_ok=True)
        free = self.free_gb()
        if free < self.config["min_free_gb"]:
            raise RuntimeError(
                f"Only {free:.1f} GB free under {self.directory}; episode recording "
                f"needs at least {self.config['min_free_gb']:g} GB"
            )
        episode_id = _episode_id()
        while (self.directory / f"{episode_id}.hdf5").exists() or (
            self.directory / f"{episode_id}.json"
        ).exists():
            time.sleep(0.001)
            episode_id = _episode_id()
        path = self.directory / f"{episode_id}.hdf5"
        writer = EpisodeWriter(path, self.camera_res)  # exclusive creation
        try:
            rollout = writer.file.create_group("rollout")
            rollout.create_dataset(
                "timestamps", (0,), maxshape=(None,), dtype="float64", chunks=(1024,)
            )
            rollout.create_dataset(
                "plan_index", (0,), maxshape=(None,), dtype="int32", chunks=(1024,)
            )
            rollout.create_dataset(
                "target_eepose",
                (0, 14),
                maxshape=(None, 14),
                dtype="float32",
                chunks=(256, 14),
            )
            plans = rollout.create_group("plans")
            for name, dtype in (
                ("start_row", "int64"),
                ("executed_rows", "int32"),
                ("horizon", "int32"),
                ("inference_seconds", "float64"),
            ):
                plans.create_dataset(
                    name, (0,), maxshape=(None,), dtype=dtype, chunks=(256,)
                )
            plans.create_dataset(
                "info", (0,), maxshape=(None,), dtype=h5py.string_dtype(), chunks=(256,)
            )
            created_at = datetime.now(timezone.utc).isoformat()
            attrs = {
                "kind": "rollout",
                "rollout_id": episode_id,
                "created_at": created_at,
                "outcome": "unlabeled",
                "end_reason": "recording",
                "cameras": list(self.camera_res),
            }
            attrs.update(dict(metadata or {}))
            for key, value in attrs.items():
                writer.file.attrs[key] = _attr(value)
            writer.file.flush()
        except BaseException:
            writer.close(complete=False)
            path.unlink(missing_ok=True)
            raise
        self._queue = queue.Queue(maxsize=self.config["max_queued_rows"])
        self._request = None
        self._episode_id = episode_id
        self._failure = self._failure_detail = None
        self._rows, self._plan_index = 0, -1
        self._t0 = time.monotonic()
        self._next_disk_check = self._t0 + _DISK_CHECK_SECONDS
        manifest = {
            "id": episode_id,
            "filename": path.name,
            "created_at": created_at,
            "checkpoint": (metadata or {}).get("checkpoint"),
            "cameras": list(self.camera_res),
        }
        self._thread = threading.Thread(
            target=self._run,
            args=(writer, self._queue, manifest),
            name=f"rollout_episode_{episode_id}",
            daemon=False,
        )
        self._thread.start()
        return episode_id

    def _enqueue(self, item) -> bool:
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            self._failure = "writer_error"
            self._failure_detail = (
                f"the writer fell {self.config['max_queued_rows']} rows behind the "
                "control loop (disk too slow)"
            )
            return False
        return True

    def append(
        self, obs, joints, eepose, target=None, now: float | None = None
    ) -> bool:
        """Queue one executed control tick. Never blocks the control loop."""
        if not self.recording or self._failure is not None:
            return False
        now = time.monotonic() if now is None else float(now)
        if now >= self._next_disk_check:
            self._next_disk_check = now + _DISK_CHECK_SECONDS
            try:
                free = self.free_gb()
            except OSError as error:
                free, self._failure_detail = 0.0, str(error)
            if free < self.config["min_free_gb"]:
                self._failure = "low_disk"
                self._failure_detail = self._failure_detail or (
                    f"free space fell to {free:.1f} GB"
                )
                return False
        row = {
            "joint_positions": np.array(obs["joint_positions"], dtype=np.float64),
            "ee_poses": np.array(obs["ee_poses"], dtype=np.float64),
        }
        for name in self.camera_res:
            frame = obs.get(name)
            row[name] = None if frame is None else np.array(frame, copy=True)
        target = (
            np.full(14, np.nan)
            if target is None
            else np.array(target, dtype=np.float64)
        )
        item = (
            "row",
            row,
            np.array(joints, dtype=np.float64),
            np.array(eepose, dtype=np.float64),
            target,
            now - self._t0,
            self._plan_index,
        )
        if not self._enqueue(item):
            return False
        self._rows += 1
        return True

    def mark_plan(
        self, actions, executed_rows: int, inference_seconds=None, info=None
    ) -> bool:
        """Record one policy plan; rows that follow carry its index."""
        if not self.recording or self._failure is not None:
            return False
        actions = np.array(actions, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 14:
            raise ValueError("A rollout plan must be shaped (H, 14)")
        seconds = math.nan if inference_seconds is None else float(inference_seconds)
        item = (
            "plan",
            actions,
            int(executed_rows),
            seconds,
            json.dumps(info or {}, default=str, sort_keys=True),
            self._rows,
        )
        if not self._enqueue(item):
            return False
        self._plan_index += 1
        return True

    def close(
        self,
        outcome: str = "unlabeled",
        complete: bool = True,
        end_reason: str = "saved",
    ):
        """Stop accepting rows and finalize in the background; return the episode id."""
        if outcome not in OUTCOMES:
            raise ValueError(f"outcome must be one of {OUTCOMES}")
        if end_reason not in END_REASONS:
            raise ValueError(f"end_reason must be one of {END_REASONS}")
        if not self.recording:
            return None
        if self._failure is not None:
            complete, end_reason = False, self._failure
        episode_id, self._episode_id = self._episode_id, None
        self._end(("close", outcome, bool(complete), end_reason, self._failure_detail))
        return episode_id

    def discard(self):
        """Stop recording and delete the episode file; return the episode id."""
        if not self.recording:
            return None
        episode_id, self._episode_id = self._episode_id, None
        self._end(("discard",))
        return episode_id

    def _end(self, request: tuple) -> None:
        """Hand the end request to the writer without ever blocking the control loop.

        The writer reads ``_request`` once its queue runs dry, so the request is
        never lost; the sentinel only shortens that wait when the queue has room.
        A blocking put here would stall control for as long as one HDF5 write
        takes on a slow disk, which is exactly when the queue is full.
        """
        with self._lock:
            self._request = request
        try:
            self._queue.put_nowait(("end",))
        except queue.Full:
            pass

    def wait(self, timeout: float | None = None) -> bool:
        """Wait for the writer to finish; return whether it did."""
        thread = self._thread
        if thread is not None:
            thread.join(timeout)
        return not self.busy

    def take_finished(self) -> list[dict]:
        """Return and clear the results of episodes finalized since the last call."""
        with self._lock:
            finished, self._finished = self._finished, []
        return finished

    def _run(self, writer: EpisodeWriter, items: queue.Queue, manifest: dict) -> None:
        error = None
        first_timestamp = last_timestamp = None
        plans = writer.file["rollout/plans"]
        while True:
            try:
                item = items.get(timeout=_END_POLL_SECONDS)
            except queue.Empty:
                with self._lock:
                    ended = self._request is not None
                if ended:
                    break  # the sentinel was dropped on a full queue; it is drained now
                continue
            kind = item[0]
            if kind == "end":
                break
            if error is not None:
                continue  # keep draining so the control loop never blocks on put
            try:
                if kind == "row":
                    _, row, joints, eepose, target, timestamp, plan_index = item
                    writer.append(row, joints, eepose)
                    n = writer.frames
                    rollout = writer.file["rollout"]
                    for name, value in (
                        ("timestamps", timestamp),
                        ("plan_index", plan_index),
                        ("target_eepose", target),
                    ):
                        dataset = rollout[name]
                        dataset.resize(n, axis=0)
                        dataset[n - 1] = value
                    first_timestamp = (
                        timestamp if first_timestamp is None else first_timestamp
                    )
                    last_timestamp = timestamp
                else:
                    _, actions, executed, seconds, info, start_row = item
                    if "actions" not in plans:
                        plans.create_dataset(
                            "actions",
                            (0, *actions.shape),
                            maxshape=(None, *actions.shape),
                            dtype="float32",
                            chunks=(1, *actions.shape),
                        )
                    stored = plans["actions"]
                    horizon = stored.shape[1]
                    padded = np.full((horizon, 14), np.nan, dtype=np.float32)
                    padded[: min(horizon, len(actions))] = actions[:horizon]
                    p = stored.shape[0] + 1
                    for name, value in (
                        ("actions", padded),
                        ("start_row", start_row),
                        ("executed_rows", executed),
                        ("horizon", len(actions)),
                        ("inference_seconds", seconds),
                        ("info", info),
                    ):
                        plans[name].resize(p, axis=0)
                        plans[name][p - 1] = value
            except Exception as caught:  # surfaced to the loop through ``failure``
                error = f"{type(caught).__name__}: {caught}"
                self._failure, self._failure_detail = "writer_error", error
        with self._lock:
            request = self._request
        result = {
            "id": manifest["id"],
            "saved": None,
            "discarded": False,
            "error": None,
        }
        path = writer.path
        try:
            if request[0] == "discard":
                writer.close(complete=False)
                path.unlink(missing_ok=True)
                result["discarded"] = True
            else:
                _, outcome, complete, end_reason, detail = request
                if error is not None:
                    complete, end_reason, detail = False, "writer_error", error
                frames = writer.frames
                ended_at = datetime.now(timezone.utc).isoformat()
                duration = (
                    0.0 if first_timestamp is None else last_timestamp - first_timestamp
                )
                attrs = {
                    "outcome": outcome,
                    "end_reason": end_reason,
                    "ended_at": ended_at,
                    "frames": frames,
                    "duration_seconds": duration,
                    "end_detail": detail or "",
                }
                for key, value in attrs.items():
                    writer.file.attrs[key] = _attr(value)
                plan_count = plans["start_row"].shape[0]
                writer.close(complete=complete)
                if frames == 0:
                    path.unlink(missing_ok=True)
                else:
                    payload = {
                        **manifest,
                        "frames": frames,
                        "plans": plan_count,
                        "duration_seconds": duration,
                        "outcome": outcome,
                        "complete": complete,
                        "end_reason": end_reason,
                        "end_detail": detail or "",
                        "ended_at": ended_at,
                    }
                    final = self.directory / f"{manifest['id']}.json"
                    temporary = self.directory / f".{manifest['id']}.partial.json"
                    temporary.write_text(json.dumps(payload, indent=2) + "\n")
                    temporary.replace(final)
                    result["saved"] = payload
        except Exception as caught:
            result["error"] = f"{type(caught).__name__}: {caught}"
            if writer.file is not None:
                try:
                    writer.close(complete=False)
                except Exception:
                    pass
        with self._lock:
            self._finished.append(result)
