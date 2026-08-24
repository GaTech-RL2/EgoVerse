"""EVA-specific observation and action codecs for the generic rollout graph."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
import torch
from torch.utils.data import default_collate

from egomimic.rldb.embodiment.eva_frames import (
    dataset_ypr_pose_to_hardware_ypr,
    hardware_ypr_pose_to_dataset_wxyz,
)
from egomimic.rldb.embodiment.fold_span_transforms import (
    build_bimanual_rot6d_wrist_revert_transforms,
    eva_rollout_obs_transforms,
)
from egomimic.rollout.core import RolloutNode

_GRIPPER_ENDPOINT_ATOL = 1e-6


def _ypr_pose_to_wxyz(pose: np.ndarray) -> np.ndarray:
    return hardware_ypr_pose_to_dataset_wxyz(pose).astype(np.float32)


class EvaObservationWindow(RolloutNode):
    """Maintain the same two-frame observation history used during training."""

    reads = ("obs",)
    writes = ("obs_window",)

    def __init__(self, n_obs_steps: int = 2):
        self.n_obs_steps = int(n_obs_steps)
        if self.n_obs_steps <= 0:
            raise ValueError("n_obs_steps must be positive")
        self._history: deque[dict[str, np.ndarray]] = deque(maxlen=self.n_obs_steps)

    def reset(self, state):
        self._history.clear()

    def __call__(self, state):
        obs = {
            key: np.asarray(value).copy() for key, value in dict(state["obs"]).items()
        }
        if not self._history:
            for _ in range(self.n_obs_steps - 1):
                self._history.append(dict(obs))
        self._history.append(obs)
        state["obs_window"] = {
            key: np.stack([frame[key] for frame in self._history], axis=0)
            for key in obs
        }
        return state


class EvaObservationCodec(RolloutNode):
    """Convert live BGR/YPR observations into the Fold training key space."""

    reads = ("obs_window", "should_query")
    writes = ("obs_batch", "native_state_ee_pose")

    def __init__(self, arms=("left", "right")):
        if tuple(arms) != ("left", "right"):
            raise ValueError("The Fold Pipeline checkpoint is bimanual")
        self.transforms = eva_rollout_obs_transforms()

    @staticmethod
    def _images_to_rgb_chw(images: np.ndarray) -> torch.Tensor:
        images = np.asarray(images)
        if images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError(f"Expected (N,H,W,3) images, got {images.shape}")
        rgb = images[..., ::-1].copy()
        return torch.from_numpy(rgb).permute(0, 3, 1, 2).float().div_(255.0)

    def __call__(self, state):
        if not state.get("should_query"):
            return state
        obs = state["obs_window"]
        ee = np.asarray(obs["ee_poses"], dtype=np.float64)
        if ee.ndim != 2 or ee.shape[-1] != 14:
            raise ValueError(f"Expected EVA ee_poses history (N,14), got {ee.shape}")
        batch: dict[str, Any] = {
            "front_img_1": self._images_to_rgb_chw(obs["front_img_1"]),
            "left_wrist_img": self._images_to_rgb_chw(obs["left_wrist_img"]),
            "right_wrist_img": self._images_to_rgb_chw(obs["right_wrist_img"]),
            "left.obs_ee_pose": _ypr_pose_to_wxyz(ee[:, :6]),
            "right.obs_ee_pose": _ypr_pose_to_wxyz(ee[:, 7:13]),
            "left.obs_gripper": ee[:, 6:7].astype(np.float32),
            "right.obs_gripper": ee[:, 13:14].astype(np.float32),
        }
        for transform in self.transforms:
            batch = transform.transform(batch)
        native_state = batch["state_ee_pose"]
        if torch.is_tensor(native_state):
            native_state = native_state.detach().cpu().numpy()
        state["native_state_ee_pose"] = np.asarray(native_state).copy()
        state["obs_batch"] = default_collate([batch])
        return state


class EvaActionCodec(RolloutNode):
    """Decode normalized 20-D wrist-frame actions to 14-D base-frame commands."""

    reads = ("action", "native_state_ee_pose")
    writes = ("command",)

    def __init__(self, norm_stats, emb_id: int, ac_key: str = "actions_cartesian"):
        self.norm_stats = norm_stats
        self.emb_id = int(emb_id)
        self.ac_key = str(ac_key)
        self.transforms = build_bimanual_rot6d_wrist_revert_transforms(
            self.ac_key, "state_ee_pose"
        )

    def __call__(self, state):
        action = state.get("action")
        if action is None:
            return state
        if not torch.is_tensor(action):
            action = torch.as_tensor(action, dtype=torch.float32)
        native = self.norm_stats.unnormalize(
            {self.ac_key: action[None, :]}, self.emb_id
        )[self.ac_key]
        batch = {
            self.ac_key: native.detach().cpu().numpy(),
            "state_ee_pose": np.asarray(state["native_state_ee_pose"]),
        }
        for transform in self.transforms:
            batch = transform.transform(batch)
        encoded = np.asarray(batch[self.ac_key], dtype=np.float64)
        if encoded.ndim != 2 or encoded.shape[-1] != 14:
            raise ValueError(f"Invalid encoded EVA command shape: {encoded.shape}")
        left = dataset_ypr_pose_to_hardware_ypr(encoded[:, :6])
        right = dataset_ypr_pose_to_hardware_ypr(encoded[:, 7:13])
        commands = np.concatenate(
            [left, encoded[:, 6:7], right, encoded[:, 13:14]], axis=-1
        )
        grippers = commands[:, [6, 13]]
        if (
            not np.all(np.isfinite(grippers))
            or np.any(grippers < -_GRIPPER_ENDPOINT_ATOL)
            or np.any(grippers > 1.0 + _GRIPPER_ENDPOINT_ATOL)
        ):
            raise ValueError(
                "Decoded EVA gripper command exceeds [0, 1] beyond numerical "
                f"tolerance: {grippers.tolist()}"
            )
        commands[:, [6, 13]] = np.clip(grippers, 0.0, 1.0)
        command = commands.astype(np.float32, copy=False)[0]
        if command.shape != (14,) or not np.all(np.isfinite(command)):
            raise ValueError(f"Invalid EVA command shape/value: {command.shape}")
        state["command"] = command
        return state
