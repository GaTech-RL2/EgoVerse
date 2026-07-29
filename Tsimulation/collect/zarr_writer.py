"""Streaming per-episode zarr writer for PushShapes demonstrations.

Wraps `egomimic.rldb.zarr.ZarrWriter` to match EgoVerse's per-episode
schema documented in `Tsimulation/SCHEMA_NOTES.md`. Each committed episode
becomes its own `episode_{N:06d}.zarr` under the output directory.

Episode commits are atomic: data is buffered in memory and only written
to disk in `commit_episode`. Crash mid-episode leaves the on-disk store
untouched. Because commits are bulk writes, the resulting episode layout
matches the repo's compact one-chunk-per-array format.
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

# Non-greedy ``.+?`` for the object_shape + pusher_shape prefix so pusher names
# containing ``_`` (e.g. ``circle_small``) still match. Anchored by ``_obs\d+_``
# on the right so we don't over-match into the obstacle level / tag / index.
_EPISODE_RE = re.compile(r"^episode_.+?_obs\d+_(\d+)\.zarr$")
_TAGGED_EPISODE_RE_TMPL = r"^episode_.+?_obs\d+_{tag}_(\d+)\.zarr$"


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
        tag: str | None = None,
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
        # Optional filename tag — keeps mixed-purpose datasets distinguishable
        # and indexed independently in the same folder.
        if tag is not None:
            if not re.match(r"^[A-Za-z0-9]+$", tag):
                raise ValueError(
                    f"tag must be alphanumeric ([A-Za-z0-9]+), got {tag!r}"
                )
        self._tag = tag
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

    def start_episode(
        self,
        *,
        init_state: dict[str, Any] | None = None,
    ) -> None:
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
        self._episode_init = init_state

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
        pusher = np.asarray(pusher_obs_pose, dtype=np.float64).reshape(-1)
        obj = np.asarray(object_obs_pose, dtype=np.float64).reshape(-1)
        self._buffer[IMAGE_KEY].append(img)
        self._buffer[STATE_KEY].append(np.concatenate([pusher, obj]))
        self._buffer[CMD_PUSHER_KEY].append(np.asarray(pusher_cmd_pose, dtype=np.float64).reshape(-1))
        self._buffer[ACTION_KEY].append(np.asarray(action, dtype=np.float64).reshape(-1))
        self._buffer[REWARD_KEY].append(np.asarray([reward], dtype=np.float64))
        self._buffer[GOAL_KEY].append(np.asarray(goal_pose, dtype=np.float64).reshape(-1))

    def commit_episode(self) -> int:
        """Write the buffered episode to disk and return its index.

        Returns -1 if the buffer was empty (nothing written, buffer cleared).
        """
        if self._buffer is None:
            return -1
        if len(self._buffer[IMAGE_KEY]) == 0:
            self._buffer = None
            return -1

        tag_part = f"_{self._tag}" if self._tag else ""
        ep_path = self.path / (
            f"episode_{self._object_shape}_{self._pusher_shape}"
            f"_obs{self._obstacle_level}{tag_part}_{self._episode_idx:06d}.zarr"
        )
        images = np.stack(self._buffer[IMAGE_KEY], axis=0)
        numeric = {
            STATE_KEY: np.stack(self._buffer[STATE_KEY], axis=0),
            CMD_PUSHER_KEY: np.stack(self._buffer[CMD_PUSHER_KEY], axis=0),
            ACTION_KEY: np.stack(self._buffer[ACTION_KEY], axis=0),
            REWARD_KEY: np.stack(self._buffer[REWARD_KEY], axis=0),
            GOAL_KEY: np.stack(self._buffer[GOAL_KEY], axis=0),
        }

        metadata_override = None
        if self._episode_init is not None:
            metadata_override = {"episode_init": json.dumps(self._episode_init)}

        # The arena is an orthographic 512x512 top-down render, so there is no
        # pinhole camera to calibrate. create_and_write() requires a validated
        # 3x4 K, so go through the constructor -- where intrinsics is optional
        # -- keeping episodes consistent with the existing dataset, which
        # carries no intrinsics key.
        writer = ZarrWriter(
            episode_path=ep_path,
            embodiment=self.embodiment,
            fps=self.fps,
            task_name=self.task_name,
            task_description=self.task_description,
            chunk_timesteps=self.chunk_timesteps,
        )
        writer.write(
            numeric_data=numeric,
            image_data={IMAGE_KEY: images},
            metadata_override=metadata_override,
        )

        idx = self._episode_idx
        self._episode_idx += 1
        self._buffer = None
        self._episode_init = None
        return idx

    def abort_episode(self) -> None:
        self._buffer = None
        self._episode_init = None

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
        When ``tag`` was supplied, only episodes carrying the same tag in
        their filename are counted (so tagged + untagged sequences advance
        independently in a mixed-purpose folder).
        """
        if not self.path.exists():
            return 0
        if self._tag is not None:
            pattern = re.compile(_TAGGED_EPISODE_RE_TMPL.format(tag=re.escape(self._tag)))
        else:
            pattern = _EPISODE_RE
        max_idx = -1
        for entry in self.path.iterdir():
            if not entry.is_dir():
                continue
            m = pattern.match(entry.name)
            if m:
                max_idx = max(max_idx, int(m.group(1)))
        return max_idx + 1

    def __enter__(self) -> "ZarrDemoWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False
