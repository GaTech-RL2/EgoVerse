"""Algorithm-neutral contracts and shared EVA policy plumbing for live rollout."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from scipy.spatial.transform import Rotation
from torch.utils.data import default_collate

from egomimic.rldb.embodiment.embodiment import get_embodiment_id
from egomimic.rldb.embodiment.eva import Eva
from egomimic.utils.pose_utils import (
    cam_frame_to_base_frame,
    interpolate_arr,
    interpolate_arr_euler,
    xyzw_to_wxyz,
)

EMBODIMENT_NAME_BY_ARM = {
    "both": "eva_bimanual",
    "left": "eva_left_arm",
    "right": "eva_right_arm",
}
_R_TOOL_TO_EE = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=float)
_R_EE_TO_TOOL = np.linalg.inv(_R_TOOL_TO_EE)


@dataclass(frozen=True)
class RolloutPolicyConfig:
    arm: str
    query_frequency: int
    cartesian: bool
    resampled_action_len: int | None = None
    annotation_path: str | None = None
    action_frame: str = "base"
    require_cuda: bool = False

    def __post_init__(self):
        if self.arm not in EMBODIMENT_NAME_BY_ARM:
            raise ValueError(f"Unknown arm selection {self.arm!r}")
        if self.query_frequency <= 0:
            raise ValueError("query_frequency must be positive")
        if self.action_frame not in ("base", "cam"):
            raise ValueError("action_frame must be base or cam")

    @property
    def embodiment_id(self) -> int:
        return get_embodiment_id(self.embodiment_name)

    @property
    def embodiment_name(self) -> str:
        return EMBODIMENT_NAME_BY_ARM[self.arm]


@runtime_checkable
class RolloutPolicy(Protocol):
    debug_actions: np.ndarray | None
    just_queried: bool

    def act(self, obs: dict) -> np.ndarray: ...

    def reset(self) -> None: ...

    def load_annotation(self, annotation_path: str) -> bool: ...


def _checkpoint_algo_class(checkpoint: dict):
    config_tree = checkpoint.get("hyper_parameters", {}).get("config_tree")
    if config_tree is None:
        raise ValueError("Checkpoint has no hyper_parameters.config_tree")
    cfg = (
        config_tree
        if isinstance(config_tree, DictConfig)
        else OmegaConf.create(config_tree)
    )
    target = OmegaConf.select(cfg, "model.robomimic_model._target_")
    if not target:
        raise ValueError("Checkpoint config has no model.robomimic_model._target_")
    return hydra.utils.get_class(str(target))


def load_rollout_policy(checkpoint_path: str, config: RolloutPolicyConfig):
    """Load any Algo checkpoint and ask that Algo to construct its live policy."""
    from egomimic.pl_utils.pl_model import ModelWrapper

    print(f"[rollout] Loading policy from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    algo_cls = _checkpoint_algo_class(checkpoint)
    if config.require_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "Live policy rollout requires CUDA. Check Docker --gpus/device setup, "
            "or pass --allow-cpu-policy only for an intentional diagnostic run."
        )
    prepared_path = algo_cls.prepare_rollout_checkpoint(checkpoint_path)
    wrapper = ModelWrapper.load_from_checkpoint(
        prepared_path, weights_only=False, map_location="cpu"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wrapper = wrapper.to(device).eval()
    wrapper.model.device = device
    policy = wrapper.model.create_rollout_policy(config)
    policy._model_wrapper = wrapper  # Keep Lightning ownership alive for rollout.
    _report_model_device(wrapper.model)
    return policy


def _report_model_device(algo) -> None:
    try:
        parameter = next(algo.nets.parameters())
    except StopIteration:
        return
    print(f"[rollout] Model device: {parameter.device}, dtype: {parameter.dtype}")
    if not parameter.is_cuda:
        print("[rollout] WARNING: model is running on CPU")


def _ee_pose_to_rotated_frame(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose)
    rotation = Rotation.from_euler("ZYX", pose[..., 3:6]).as_matrix()
    rotated = _R_TOOL_TO_EE @ rotation
    return np.concatenate(
        [pose[..., :3], Rotation.from_matrix(rotated).as_euler("ZYX")], axis=-1
    )


def _rotated_frame_to_ee_pose_batch(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose)
    rotation = Rotation.from_euler("ZYX", pose[..., 3:6]).as_matrix()
    ee_rotation = _R_EE_TO_TOOL @ rotation
    return np.concatenate(
        [pose[..., :3], Rotation.from_matrix(ee_rotation).as_euler("ZYX")], axis=-1
    )


class EvaLegacyPolicy:
    """Shared EVA adapter used by legacy algorithms; inference stays Algo-owned."""

    def __init__(self, algo, config: RolloutPolicyConfig):
        self.algo = algo
        self.config = config
        self.arm = config.arm
        self.cartesian = config.cartesian
        self.action_frame = config.action_frame
        self.extrinsics = Eva.EXTRINSICS
        self.transform_list = Eva.get_transform_list(mode="cartesian_wristframe_ypr")
        self.annotation = None
        self.actions = None
        self.debug_actions = None
        self.just_queried = False
        self._action_index = 0
        if config.annotation_path:
            self.load_annotation(config.annotation_path)
        print(f"[rollout] action_frame={self.action_frame}")

    def reset(self) -> None:
        self.actions = None
        self.debug_actions = None
        self.just_queried = False
        self._action_index = 0
        self.algo.nets.eval()

    def load_annotation(self, annotation_path: str) -> bool:
        if not os.path.isfile(annotation_path):
            print(f"[rollout] WARNING: annotation file not found: {annotation_path}")
            return False
        with open(annotation_path) as stream:
            self.annotation = stream.read().strip()
        print(f"[rollout] Loaded annotation from {annotation_path}")
        return True

    def act(self, obs: dict) -> np.ndarray:
        self.just_queried = self.actions is None or self._action_index >= min(
            self.config.query_frequency, len(self.actions)
        )
        if self.just_queried:
            started = time.time()
            self.actions = self._predict_chunk(obs)
            self.debug_actions = self.actions.copy()
            self.actions = self._decode_actions(self.actions)
            if self.config.resampled_action_len is not None:
                self.actions = self._downsample_chunk(
                    self.actions, self.config.resampled_action_len
                )
            self._action_index = 0
            print(f"Inference time: {time.time() - started}s")
        action = self.actions[self._action_index]
        self._action_index += 1
        return action

    def _predict_chunk(self, obs: dict) -> np.ndarray:
        sample = self._process_observation(obs)
        for transform in self.transform_list:
            sample = transform.transform(sample)
        batch = {self.config.embodiment_name: default_collate([sample])}
        processed = self.algo.process_batch_for_training(batch)
        key = f"{self.config.embodiment_name}_actions_cartesian"
        prediction = self.algo.forward_eval(processed)[key]
        actions = prediction.detach().cpu().numpy().squeeze()
        if actions.ndim == 1:
            actions = actions[None, :]
        return actions

    @staticmethod
    def _rgb_chw(image: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(image[..., ::-1].copy()).permute(2, 0, 1).float() / 255

    def _process_observation(self, obs: dict) -> dict:
        front = self._rgb_chw(obs["front_img_1"])
        data = {
            "front_img_1": front,
            "base_0_rgb": front,
            "observations.images.front_img_1": front,
            "pad_mask": torch.ones((1, 100, 1), dtype=torch.bool),
        }
        ee = np.asarray(obs["ee_poses"])
        for arm, offset in (("right", 7), ("left", 0)):
            if self.arm not in (arm, "both"):
                continue
            wrist = self._rgb_chw(obs[f"{arm}_wrist_img"])
            data[f"{arm}_wrist_img"] = wrist
            data[f"{arm}_wrist_0_rgb"] = wrist
            data[f"observations.images.{arm}_wrist_img"] = wrist
            pose = _ee_pose_to_rotated_frame(ee[offset : offset + 6])
            quat = xyzw_to_wxyz(Rotation.from_euler("ZYX", pose[3:6]).as_quat())
            pose_quat = np.concatenate([ee[offset : offset + 3], quat])
            data[f"{arm}.obs_ee_pose"] = torch.from_numpy(pose_quat).reshape(-1)
            data[f"{arm}.obs_gripper"] = torch.from_numpy(
                ee[offset + 6 : offset + 7]
            ).reshape(-1)
            data[f"{arm}.cmd_gripper"] = (
                data[f"{arm}.obs_gripper"].view(1, 1).repeat(45, 1)
            )
            data[f"{arm}.cmd_ee_pose"] = (
                torch.from_numpy(pose_quat).view(1, 7).repeat(45, 1)
            )
        data["embodiment"] = [self.config.embodiment_name]
        if self.annotation is not None:
            data["annotations"] = [self.annotation]
        return data

    def _decode_actions(self, actions: np.ndarray) -> np.ndarray:
        if not self.cartesian or self.action_frame == "base":
            return actions.copy()
        if self.arm == "both":
            decoded = []
            for arm, action in zip(("left", "right"), (actions[:, :7], actions[:, 7:])):
                pose = cam_frame_to_base_frame(
                    action[:, :6].copy(), self.extrinsics[arm]
                )
                pose = _rotated_frame_to_ee_pose_batch(pose)
                decoded.append(
                    np.hstack([pose, action[:, 6:7]]) if action.shape[1] == 7 else pose
                )
            return np.hstack(decoded)
        pose = _rotated_frame_to_ee_pose_batch(actions[:, :6].copy())
        pose = cam_frame_to_base_frame(pose, self.extrinsics[self.arm])
        return np.hstack([pose, actions[:, 6:7]]) if actions.shape[1] == 7 else pose

    def _downsample_chunk(self, chunk: np.ndarray, target_len: int) -> np.ndarray:
        if target_len <= 0 or len(chunk) == target_len:
            return chunk.astype(np.float32, copy=False)
        if not self.cartesian:
            return interpolate_arr(chunk[None], target_len)[0].astype(np.float32)
        if self.arm != "both":
            return interpolate_arr_euler(chunk[None], target_len)[0].astype(np.float32)
        left = interpolate_arr_euler(chunk[None, :, :7], target_len)[0]
        right = interpolate_arr_euler(chunk[None, :, 7:14], target_len)[0]
        return np.hstack([left, right]).astype(np.float32)
