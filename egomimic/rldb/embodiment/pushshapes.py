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
    """HNet-style keymap for pushshapes_sim: per-frame obs windows.

    Obs keys carry ``horizon=action_horizon`` so the dataloader returns
    ``(T, ...)`` per-frame windows — required for HNet's AdaLN per-token
    cond (one ``obs_t`` per action token).

    For HPT, which only consumes ``obs_t=0`` and would otherwise pay 32× the
    JPEG-decode cost, use ``get_keymap_hpt`` instead.

    Extra kwargs (e.g. ``norm_mode=True`` injected by trainHydra) are
    accepted and ignored so norm-stat collection doesn't crash.
    """
    return {
        "front_img_1": {
            "key_type": "camera_keys",
            "zarr_key": "observations.images.front_img_1",
            "horizon": int(action_horizon),
        },
        "state_agent_obj": {
            "key_type": "proprio_keys",
            "zarr_key": "observations.state",
            "horizon": int(action_horizon),
        },
        "actions": {
            "key_type": "action_keys",
            "zarr_key": "actions",
            "horizon": int(action_horizon),
        },
    }


def get_keymap_hpt(action_horizon: int = 32, **kwargs) -> dict:
    """HPT-style keymap for pushshapes_sim: single-frame obs.

    Obs keys carry no horizon, so the dataloader returns one frame of
    ``front_img_1`` + ``state_agent_obj`` per sample alongside an
    ``action_horizon``-long action chunk. Matches HPT's contract
    ("predict the next ``action_horizon`` actions given current obs") and
    avoids 32× the JPEG-decode work that the windowed HNet keymap incurs.

    Extra kwargs (e.g. ``norm_mode=True`` injected by trainHydra) are
    accepted and ignored.
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
    """Reverse ImageNet normalisation. Accepts a (B, C, H, W) or
    (B, T, C, H, W) float batch and returns matching shape with channel
    last, dtype uint8 ((B, H, W, C) or (B, T, H, W, C))."""
    mean = _IMAGENET_MEAN.reshape((1,) * (images.ndim - 3) + (3, 1, 1))
    std = _IMAGENET_STD.reshape((1,) * (images.ndim - 3) + (3, 1, 1))
    out = images * std + mean
    out = np.clip(out, 0.0, 1.0)
    # Move channels last (last axis = C).
    out = np.moveaxis(out, -3, -1)
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
    seq_lens: np.ndarray | None = None,
    **_unused_kwargs: Any,
) -> np.ndarray:
    """Render validation frames with GT (green) and predicted (red) trajectories.

    Two input modes:
      * Single-frame per episode (legacy): ``batch[image_key]`` is
        ``(B, C, H, W)``. Output ``(B, H', W', 3)`` — one frame per
        episode with full-episode trajectories overlaid.
      * Per-frame video (full episode): ``batch[image_key]`` is
        ``(B, T, C, H, W)``. Output is the concatenated per-episode
        video ``(sum(T_b), H', W', 3)`` so the saved .mp4 plays each
        episode in turn. ``seq_lens (B,)`` masks zero-padded tail frames.

    Returns:
        (N, H', W', 3) uint8 array, ``H'/W' = image_size * UP_SCALE``.
    """
    embodiment_name = get_embodiment(int(batch["embodiment"][0].item())).lower()

    images = _denormalize_imagenet(_as_numpy(batch[image_key]))
    gt_actions = _as_numpy(batch[action_key]) if action_key in batch else None
    pred_actions_raw = predictions.get(f"{embodiment_name}_{action_key}")
    pred_actions = _as_numpy(pred_actions_raw) if pred_actions_raw is not None else None
    if seq_lens is None and "seq_lens" in batch:
        seq_lens = _as_numpy(batch["seq_lens"])

    # Per-frame mode: images is (B, T, H, W, C).
    if images.ndim == 5:
        B, T_max, h, w, _ = images.shape
        out_h, out_w = h * UP_SCALE, w * UP_SCALE
        scale = out_w / _WORLD_SIZE
        if seq_lens is None:
            seq_lens = np.full(B, T_max, dtype=np.int64)
        else:
            seq_lens = np.asarray(seq_lens).astype(np.int64)

        frames: list[np.ndarray] = []
        for b in range(B):
            T_b = int(seq_lens[b])
            for t in range(T_b):
                frame = cv2.resize(
                    images[b, t], (out_w, out_h), interpolation=cv2.INTER_LINEAR
                )
                if gt_actions is not None:
                    _draw_traj(frame, gt_actions[b, :T_b], scale, _GT_COLOR)
                if pred_actions is not None:
                    _draw_traj(frame, pred_actions[b, :T_b], scale, _PRED_COLOR)
                frames.append(frame)
            # 5-frame black separator between episodes for readability.
            if b < B - 1 and frames:
                sep = np.zeros((5, out_h, out_w, 3), dtype=np.uint8)
                frames.extend(list(sep))
        return np.stack(frames, axis=0)

    # Single-frame legacy path: (B, H, W, C).
    b_count, h, w, _ = images.shape
    out_h, out_w = h * UP_SCALE, w * UP_SCALE
    scale = out_w / _WORLD_SIZE

    frames = []
    for i in range(b_count):
        frame = cv2.resize(images[i], (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        if gt_actions is not None:
            _draw_traj(frame, gt_actions[i], scale, _GT_COLOR)
        if pred_actions is not None:
            _draw_traj(frame, pred_actions[i], scale, _PRED_COLOR)
        frames.append(frame)
    return np.stack(frames, axis=0)
