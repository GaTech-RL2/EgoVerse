"""Streaming per-episode zarr writer for PushShapes demonstrations.

Wraps `egomimic.rldb.zarr.ZarrWriter` to match EgoVerse's per-episode
schema documented in `Tsimulation/SCHEMA_NOTES.md`. Each committed episode
becomes its own `episode_{N:06d}.zarr` under the output directory.

Episode commits are atomic: data is buffered in memory and only written
to disk in `commit_episode`. Crash mid-episode leaves the on-disk store
untouched.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from egomimic.rldb.zarr.zarr_writer import ZarrWriter

IMAGE_KEY = "observations.images.front_img_1"
STATE_KEY = "observations.state"
CMD_PUSHER_KEY = "observations.pusher_cmd_pose"
ACTION_KEY = "actions"
REWARD_KEY = "reward"
GOAL_KEY = "goal_pose"

_EPISODE_RE = re.compile(r"^episode_[A-Za-z0-9]+_[A-Za-z0-9]+_obs\d+_(\d+)\.zarr$")


class ZarrDemoWriter:
    """Per-episode zarr writer matching EgoVerse's loader schema."""

    def __init__(
        self,
        path: str | Path,
        env_args: dict[str, Any],
        image_size: int = 96,
        fps: int = 30,
        task_name: str = "pushshapes",
        embodiment: str = "pushshapes_sim",
        chunk_timesteps: int = 100,
    ):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.image_size = image_size
        self.fps = fps
        self.task_name = task_name
        self.embodiment = embodiment
        self.chunk_timesteps = chunk_timesteps
        self._object_shape = env_args.get("object_shape", "obj")
        self._pusher_shape = env_args.get("pusher_shape", "pusher")
        self._obstacle_level = env_args.get("obstacle_level", 0)
        self.task_description = json.dumps(
            {"env_args": env_args, "version": "0.3"}, separators=(",", ":")
        )
        self._buffer: dict[str, list[np.ndarray]] | None = None
        self._episode_idx = self._find_next_index()

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #

    @property
    def next_episode_index(self) -> int:
        return self._episode_idx

    @property
    def steps_in_episode(self) -> int:
        return 0 if self._buffer is None else len(self._buffer[IMAGE_KEY])

    @property
    def is_recording(self) -> bool:
        return self._buffer is not None

    def start_episode(self) -> None:
        if self._buffer is not None:
            raise RuntimeError("episode already started; commit or abort first")
        self._buffer = {
            IMAGE_KEY: [],
            STATE_KEY: [],
            CMD_PUSHER_KEY: [],
            ACTION_KEY: [],
            REWARD_KEY: [],
            GOAL_KEY: [],
        }

    def add_step(
        self,
        *,
        image: np.ndarray,
        pusher_obs_pose: np.ndarray,
        object_obs_pose: np.ndarray,
        pusher_cmd_pose: np.ndarray,
        action: np.ndarray,
        reward: float,
        goal_pose: np.ndarray,
    ) -> None:
        if self._buffer is None:
            raise RuntimeError("no episode started; call start_episode() first")
        img = np.asarray(image, dtype=np.uint8)
        if img.ndim != 3 or img.shape[-1] != 3:
            raise ValueError(
                f"image must be (H, W, 3) uint8, got {img.shape} {img.dtype}"
            )
        pusher = np.asarray(pusher_obs_pose, dtype=np.float32).reshape(-1)
        obj = np.asarray(object_obs_pose, dtype=np.float32).reshape(-1)
        self._buffer[IMAGE_KEY].append(img)
        self._buffer[STATE_KEY].append(np.concatenate([pusher, obj]))
        self._buffer[CMD_PUSHER_KEY].append(
            np.asarray(pusher_cmd_pose, dtype=np.float32).reshape(-1)
        )
        self._buffer[ACTION_KEY].append(
            np.asarray(action, dtype=np.float32).reshape(-1)
        )
        self._buffer[REWARD_KEY].append(np.asarray([reward], dtype=np.float32))
        self._buffer[GOAL_KEY].append(
            np.asarray(goal_pose, dtype=np.float32).reshape(-1)
        )

    def commit_episode(self) -> int:
        """Write the buffered episode to disk and return its index.

        Returns -1 if the buffer was empty (nothing written, buffer cleared).
        """
        if self._buffer is None:
            return -1
        if len(self._buffer[IMAGE_KEY]) == 0:
            self._buffer = None
            return -1

        ep_path = self.path / (
            f"episode_{self._object_shape}_{self._pusher_shape}"
            f"_obs{self._obstacle_level}_{self._episode_idx:06d}.zarr"
        )
        images = np.stack(self._buffer[IMAGE_KEY], axis=0)
        numeric = {
            STATE_KEY: np.stack(self._buffer[STATE_KEY], axis=0),
            CMD_PUSHER_KEY: np.stack(self._buffer[CMD_PUSHER_KEY], axis=0),
            ACTION_KEY: np.stack(self._buffer[ACTION_KEY], axis=0),
            REWARD_KEY: np.stack(self._buffer[REWARD_KEY], axis=0),
            GOAL_KEY: np.stack(self._buffer[GOAL_KEY], axis=0),
        }

        ZarrWriter.create_and_write(
            episode_path=ep_path,
            numeric_data=numeric,
            image_data={IMAGE_KEY: images},
            embodiment=self.embodiment,
            fps=self.fps,
            task_name=self.task_name,
            task_description=self.task_description,
            chunk_timesteps=self.chunk_timesteps,
        )

        idx = self._episode_idx
        self._episode_idx += 1
        self._buffer = None
        return idx

    def abort_episode(self) -> None:
        self._buffer = None

    def close(self) -> None:
        if self._buffer is not None:
            self.abort_episode()

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #

    def _find_next_index(self) -> int:
        """Resume index = (max existing `episode_NNNNNN.zarr` index) + 1.

        Only directories matching the regex are considered — stray files or
        unrelated subdirs are ignored. Returns 0 on a fresh output dir.
        """
        if not self.path.exists():
            return 0
        max_idx = -1
        for entry in self.path.iterdir():
            if not entry.is_dir():
                continue
            m = _EPISODE_RE.match(entry.name)
            if m:
                max_idx = max(max_idx, int(m.group(1)))
        return max_idx + 1

    def __enter__(self) -> "ZarrDemoWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False
