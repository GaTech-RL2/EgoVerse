"""PushShapes dataset schema for single-observation Pipeline policies."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import torch

from egomimic.rldb.embodiment.embodiment import get_embodiment
from egomimic.utils.viz_utils import draw_dot_on_frame

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_PUSHSHAPES_WORLD_SIZE = 512.0
_OVERLAY_UPSCALE = 4


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().float().numpy()
    return np.asarray(value)


def _denormalize_imagenet(images: np.ndarray) -> np.ndarray:
    mean = _IMAGENET_MEAN.reshape((1,) * (images.ndim - 3) + (3, 1, 1))
    std = _IMAGENET_STD.reshape((1,) * (images.ndim - 3) + (3, 1, 1))
    restored = np.clip(images * std + mean, 0.0, 1.0)
    return (np.moveaxis(restored, -3, -1) * 255).astype(np.uint8)


def _draw_xy_chunk(
    frame: np.ndarray,
    actions: np.ndarray,
    *,
    palette: str,
    alpha: float,
) -> np.ndarray:
    values = np.asarray(actions, dtype=np.float32)
    if values.ndim == 1:
        values = values[None]
    if values.ndim != 2 or values.shape[-1] < 2:
        raise ValueError(
            "PushShapes XY visualization requires (horizon, action_dim>=2), "
            f"got {values.shape}"
        )
    xy = values[:, :2]
    if not np.isfinite(xy).all():
        raise ValueError("PushShapes XY visualization received non-finite actions")
    pixels = xy * (frame.shape[1] / _PUSHSHAPES_WORLD_SIZE)
    overlay = draw_dot_on_frame(
        frame, pixels.tolist(), show=False, palette=palette, dot_size=3
    )
    if alpha >= 1.0:
        return overlay
    return cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0)


def viz_gt_preds_xy(
    predictions: dict,
    batch: dict,
    image_key: str = "front_img_1",
    action_key: str = "actions",
    gt_alpha: float = 0.6,
    pred_alpha: float = 1.0,
    **_unused_kwargs: Any,
) -> np.ndarray:
    """Draw physical XY action chunks: target green, prediction red.

    ``HumanRobotOverlayEval`` supplies actions after the checkpoint normalizer's
    inverse. For PlanarCommon5 this means columns 0 and 1 are already the native
    512-by-512 PushShapes cursor coordinates; theta and grip are intentionally
    ignored by this visualization.
    """

    embodiment_name = get_embodiment(int(batch["embodiment"][0].item())).lower()
    images = _denormalize_imagenet(_as_numpy(batch[image_key]))
    targets = _as_numpy(batch[action_key])
    predicted = _as_numpy(predictions[f"{embodiment_name}_{action_key}"])
    if images.ndim != 4:
        raise ValueError(
            f"Expected evaluator-selected images shaped (B,C,H,W), got {images.shape}"
        )
    if targets.shape != predicted.shape or targets.shape[0] != images.shape[0]:
        raise ValueError(
            "PushShapes XY overlay batch mismatch: "
            f"images={images.shape}, targets={targets.shape}, predictions={predicted.shape}"
        )

    frames = []
    for index, image in enumerate(images):
        height, width = image.shape[:2]
        frame = cv2.resize(
            image,
            (width * _OVERLAY_UPSCALE, height * _OVERLAY_UPSCALE),
            interpolation=cv2.INTER_LINEAR,
        )
        frame = _draw_xy_chunk(
            frame, targets[index], palette="Greens", alpha=float(gt_alpha)
        )
        frame = _draw_xy_chunk(
            frame, predicted[index], palette="Reds", alpha=float(pred_alpha)
        )
        frames.append(frame)
    return np.stack(frames)


def get_keymap_hpt(
    action_horizon: int = 16,
    observation_horizon: int = 1,
    action_target_offset: int = 0,
    norm_mode: bool = False,
    action_zarr_key: str = "actions",
    **kwargs,
):
    """Map an observation window to a future action chunk.

    With the default one-observation contract, observation keys intentionally
    have no redundant horizon axis.  Paper Diffusion Policy uses two
    observations: its sample starts one frame earlier, conditions on frames
    ``t,t+1``, and predicts actions starting at ``t+1``.  The matching
    ``action_target_offset=1`` fetches the extra leading action so the transform
    can align the target to the last observed frame without future leakage.
    """
    observation_horizon = int(observation_horizon)
    action_target_offset = int(action_target_offset)
    if observation_horizon <= 0:
        raise ValueError("observation_horizon must be positive")
    if action_target_offset < 0:
        raise ValueError("action_target_offset must be non-negative")
    keymap = {
        "front_img_1": {
            "key_type": "camera_keys",
            "zarr_key": "observations.images.front_img_1",
        },
        "state_agent_obj": {
            "key_type": "proprio_keys",
            "zarr_key": "observations.state",
        },
        "actions": {
            "key_type": "action_keys",
            "zarr_key": str(action_zarr_key),
            "horizon": int(action_horizon) + action_target_offset,
        },
    }
    if observation_horizon > 1:
        keymap["front_img_1"]["horizon"] = observation_horizon
        keymap["state_agent_obj"]["horizon"] = observation_horizon
    if norm_mode:
        keymap.pop("front_img_1")
    return keymap


def get_chain_gripper_point_validation_transform_list(
    keys: list[str] | None = None,
):
    """Validate direct ChainGripper point targets before normalization."""
    from egomimic.rldb.zarr.action_chunk_transforms import RequireLastDim

    return [RequireLastDim(keys=keys or ["actions"], width=6)]


def get_chain_gripper_point_transform_list(
    keys: list[str] | None = None,
    world_size: float = 512.0,
):
    """Derive ordered point targets from native ChainGripper controls."""
    from egomimic.rldb.zarr.action_chunk_transforms import (
        ChainGripperNative4ToPoints6,
    )

    return [
        ChainGripperNative4ToPoints6(
            keys=keys or ["actions"],
            world_size=world_size,
        )
    ]


def get_chain_gripper_point_revert_transform_list(
    keys: list[str] | None = None,
    world_size: float = 512.0,
    grid_size: int = 33,
    refinements: int = 6,
    context_state_key: str = "state_agent_obj",
    previous_control_key: str = "previous_control",
):
    """Project model point predictions back to native ChainGripper controls."""
    from egomimic.rldb.zarr.action_chunk_transforms import (
        ChainGripperPoints6ToNative4,
    )

    return [
        ChainGripperPoints6ToNative4(
            keys=keys or ["actions"],
            world_size=world_size,
            grid_size=grid_size,
            refinements=refinements,
            context_state_key=context_state_key,
            previous_control_key=previous_control_key,
        )
    ]


def get_rotvec_transform_list(keys: list[str] | None = None, angle_col: int = 2):
    """Encode U-socket ``theta`` targets as ``(cos(theta), sin(theta))``."""
    from egomimic.rldb.zarr.action_chunk_transforms import ThetaToRotVec

    return [ThetaToRotVec(keys=keys or ["actions"], angle_col=angle_col)]


def get_rotvec_revert_transform_list(keys: list[str] | None = None, angle_col: int = 2):
    """Decode U-socket rotation vectors before simulator consumption."""
    from egomimic.rldb.zarr.action_chunk_transforms import RotVecToTheta

    return [RotVecToTheta(keys=keys or ["actions"], angle_col=angle_col)]


def get_arc_length_transform_list(
    keys: list[str] | None = None,
    min_distance_unit: float = 200.0,
    resampled_vector_length: int = 25,
    dt: float = 1.0 / 30.0,
    rotation_radius: float = 40.0,
):
    """Create the planar SE(2) arc-length transform for U-socket actions."""
    from egomimic.rldb.zarr.arc_length_tokenizer import TokenizeUSocketArcLength

    action_keys = keys or ["actions"]
    if len(action_keys) != 1:
        raise ValueError("U-socket arc-length tokenization expects exactly one key")
    return [
        TokenizeUSocketArcLength(
            action_key=action_keys[0],
            output_action_key=action_keys[0],
            min_distance_unit=min_distance_unit,
            resampled_vector_length=resampled_vector_length,
            dt=dt,
        )
    ]


def get_chain_gripper_point_arc_length_transform_list(
    keys: list[str] | None = None,
    min_distance_unit: float = 200.0,
    resampled_vector_length: int = 25,
    dt: float = 1.0 / 30.0,
):
    """Create the three-point planar arc transform for ChainGripper actions."""
    from egomimic.rldb.zarr.arc_length_tokenizer import (
        TokenizeChainGripperPointArcLength,
    )

    action_keys = keys or ["actions"]
    if len(action_keys) != 1:
        raise ValueError("ChainGripper point arc tokenization expects exactly one key")
    return [
        TokenizeChainGripperPointArcLength(
            action_key=action_keys[0],
            output_action_key=action_keys[0],
            min_distance_unit=min_distance_unit,
            resampled_vector_length=resampled_vector_length,
            dt=dt,
        )
    ]


def get_chain_gripper_native_point_arc_length_transform_list(
    keys: list[str] | None = None,
    min_distance_unit: float = 200.0,
    resampled_vector_length: int = 25,
    dt: float = 1.0 / 30.0,
    world_size: float = 512.0,
):
    """Compose native4-to-points FK with point-space arc tokenization."""
    action_keys = keys or ["actions"]
    return get_chain_gripper_point_transform_list(
        keys=action_keys,
        world_size=world_size,
    ) + get_chain_gripper_point_arc_length_transform_list(
        keys=action_keys,
        min_distance_unit=min_distance_unit,
        resampled_vector_length=resampled_vector_length,
        dt=dt,
    )


class SliceActionTarget:
    """Align a fixed-horizon action target to the last observed frame."""

    def __init__(self, keys, start: int, horizon: int):
        self.keys = list(keys)
        self.start = int(start)
        self.horizon = int(horizon)
        if self.start < 0 or self.horizon <= 0:
            raise ValueError("SliceActionTarget requires start>=0 and horizon>0")

    def transform(self, batch):
        for key in self.keys:
            value = batch[key]
            sliced = value[self.start : self.start + self.horizon]
            if int(sliced.shape[0]) != self.horizon:
                raise ValueError(
                    f"{key} target has {sliced.shape[0]} steps after alignment; "
                    f"expected {self.horizon}"
                )
            batch[key] = sliced
        return batch


def get_planar_dense_transform_list(
    keys: list[str] | None = None,
    action_target_offset: int = 0,
    action_horizon: int = 16,
):
    """h16-style dense baseline, widened to the shared 5-channel layout.

    Pairs with the arc configs: identical action representation, the only
    difference being time-indexed dense chunks vs arc-length tokens. Without
    the shared layout the comparison would confound tokenization with a
    change of action space.
    """
    from egomimic.rldb.zarr.arc_length_tokenizer import PadPlanarAction

    action_keys = keys or ["actions"]
    transforms = []
    if int(action_target_offset) > 0:
        transforms.append(
            SliceActionTarget(
                keys=action_keys,
                start=int(action_target_offset),
                horizon=int(action_horizon),
            )
        )
    transforms.append(PadPlanarAction(keys=action_keys))
    return transforms


def get_planar_arc_length_transform_list(
    keys: list[str] | None = None,
    min_distance_unit: float = 100.0,
    resampled_vector_length: int = 100,
    dt: float = 1.0 / 30.0,
    rotation_radius: float = 0.0,
    hybrid_rotation_unit: float | None = None,
    velocity_mode: str = "mean_scalar",
    velocity_layout: str = "append",
):
    """Build an explicitly configured embodiment-agnostic planar tokenizer."""
    from egomimic.rldb.zarr.arc_length_tokenizer import TokenizePlanarArcLength

    action_keys = keys or ["actions"]
    if len(action_keys) != 1:
        raise ValueError("planar arc tokenization expects exactly one action key")
    return [
        TokenizePlanarArcLength(
            action_key=action_keys[0],
            output_action_key=action_keys[0],
            min_distance_unit=min_distance_unit,
            resampled_vector_length=resampled_vector_length,
            dt=dt,
            rotation_radius=rotation_radius,
            hybrid_rotation_unit=hybrid_rotation_unit,
            velocity_mode=velocity_mode,
            velocity_layout=velocity_layout,
        )
    ]
