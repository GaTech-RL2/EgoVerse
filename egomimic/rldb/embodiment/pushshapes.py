"""Keymap helper and validation-time visualization for the pushshapes_sim
embodiment.

`get_keymap()` is referenced from `egomimic/hydra_configs/data/tsimulation.yaml`
so it sets the shape of every training batch. `viz_gt_preds()` is referenced
from `egomimic/hydra_configs/visualization/cartesian.yaml` and is called
during validation to render the GT/predicted trajectories overlaid on each
observation image.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import torch

from egomimic.rldb.embodiment.embodiment import get_embodiment

# ImageNet normalization constants applied by eval_image_augs; we invert them
# before rendering so the image looks correct in the viz frame.
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# PushShapes world is always 512x512 px; observations are rendered at
# image_size (default 96). We upscale the obs image by UP_SCALE before
# drawing trajectories so the video is readable.
_WORLD_SIZE = 512.0
UP_SCALE = 4

# Trajectory styling. Colors are RGB (cv2 will draw them in the obs frame,
# which is RGB by the time it reaches us).
_GT_COLOR = (0, 200, 0)
_PRED_COLOR = (220, 50, 50)
_TRAJ_THICKNESS = 2
_TRAJ_DOT_RADIUS = 3
_TRAJ_DOT_EVERY = 2  # draw a dot at every Nth trajectory point


def get_keymap(action_horizon: int = 32, **kwargs) -> dict:
    """Return the key_map for pushshapes_sim ZarrDataset.

    Args:
        action_horizon: number of future actions returned per sample. Must
            match the model's ``trunk.action_horizon`` / head ``act_seq``.
            Wire it from the data config (see
            ``egomimic/hydra_configs/data/tsimulation.yaml``) so there's one
            source of truth shared with the model config.

    Extra kwargs (e.g. ``norm_mode=True`` injected by trainHydra) are accepted
    and ignored so that norm-stat collection doesn't crash by passing
    sentinel keys through to the inner key_map iteration.
    """
    return {
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
            "zarr_key": "actions",
            "horizon": int(action_horizon),
        },
    }


# ---------------------------------------------------------------------- #
# Validation viz
# ---------------------------------------------------------------------- #


def _as_numpy(arr: Any) -> np.ndarray:
    """Detach torch → cpu → float32 numpy. Pass-through for ndarrays."""
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().float().numpy()
    return np.asarray(arr)


def _denormalize_imagenet(images: np.ndarray) -> np.ndarray:
    """Reverse ImageNet normalisation on a (B, C, H, W) float batch and return
    a (B, H, W, C) uint8 array."""
    out = images * _IMAGENET_STD[:, None, None] + _IMAGENET_MEAN[:, None, None]
    out = np.clip(out, 0.0, 1.0)
    out = np.transpose(out, (0, 2, 3, 1))  # (B, H, W, C)
    return (out * 255).astype(np.uint8)


def _draw_traj(frame: np.ndarray, traj: np.ndarray, scale: float, color: tuple) -> None:
    """Overlay a 2-D trajectory (lines + sparse dots) onto `frame` in place."""
    h, w = frame.shape[:2]
    pts: list[tuple[int, int]] = []
    for xy in traj:
        px = max(0, min(w - 1, int(xy[0] * scale)))
        py = max(0, min(h - 1, int(xy[1] * scale)))
        pts.append((px, py))
    for j in range(1, len(pts)):
        cv2.line(
            frame,
            pts[j - 1],
            pts[j],
            color,
            thickness=_TRAJ_THICKNESS,
            lineType=cv2.LINE_AA,
        )
    for px, py in pts[::_TRAJ_DOT_EVERY]:
        cv2.circle(frame, (px, py), _TRAJ_DOT_RADIUS, color, thickness=-1)


def viz_gt_preds(
    predictions: dict,
    batch: dict,
    image_key: str = "front_img_1",
    action_key: str = "actions",
    mode: str = "traj",
    **_unused_kwargs: Any,
) -> np.ndarray:
    """Render validation frames with GT (green) and predicted (red) trajectories.

    Extra kwargs (``gt_alpha``, ``pred_alpha``, ``annotation_key`` … that the
    base ``Embodiment.viz_gt_preds`` accepts) are ignored — the pushshapes
    trajectory is a flat 2-D path and doesn't need them.

    Returns:
        (B, H', W', 3) uint8 array, where H'/W' = image_size * UP_SCALE.
    """
    embodiment_name = get_embodiment(int(batch["embodiment"][0].item())).lower()

    images = _denormalize_imagenet(_as_numpy(batch[image_key]))
    gt_actions = _as_numpy(batch[action_key]) if action_key in batch else None
    pred_actions_raw = predictions.get(f"{embodiment_name}_{action_key}")
    pred_actions = _as_numpy(pred_actions_raw) if pred_actions_raw is not None else None

    b, h, w, _ = images.shape
    out_h, out_w = h * UP_SCALE, w * UP_SCALE
    scale = out_w / _WORLD_SIZE

    frames = []
    for i in range(b):
        frame = cv2.resize(images[i], (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        if gt_actions is not None:
            _draw_traj(frame, gt_actions[i], scale, _GT_COLOR)
        if pred_actions is not None:
            _draw_traj(frame, pred_actions[i], scale, _PRED_COLOR)
        frames.append(frame)
    return np.stack(frames, axis=0)
