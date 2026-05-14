"""Keymap helper and visualization for the pushshapes_sim embodiment."""

from __future__ import annotations

import cv2
import numpy as np
import torch

# ImageNet normalization constants used by eval_image_augs
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Pushshapes world is 512×512; observations are rendered at image_size (default 96)
_WORLD_SIZE = 512.0


def get_keymap(**kwargs) -> dict:
    """Return the key_map for pushshapes_sim ZarrDataset.

    Extra kwargs (e.g. norm_mode=True injected by trainHydra) are accepted
    and ignored so that norm-stat collection doesn't crash.
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
            "horizon": 32,
        },
    }


def viz_gt_preds(
    predictions,
    batch,
    image_key: str = "front_img_1",
    action_key: str = "actions",
    mode: str = "traj",
) -> np.ndarray:
    """Render validation frames with predicted (red) and GT (green) trajectories.

    Returns (B, H, W, 3) uint8 numpy array suitable for the video buffer.
    """
    embodiment_id = batch["embodiment"][0].item()
    from egomimic.rldb.embodiment.embodiment import get_embodiment
    embodiment_name = get_embodiment(embodiment_id).lower()

    pred_key = f"{embodiment_name}_{action_key}"
    pred_actions = predictions.get(pred_key)  # (B, T, 2) or None

    images = batch[image_key]  # (B, C, H, W), ImageNet-normalised
    if isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()

    # Reverse ImageNet normalisation → [0, 1] → uint8
    images = images * _IMAGENET_STD[:, None, None] + _IMAGENET_MEAN[:, None, None]
    images = np.clip(images, 0.0, 1.0)
    images = np.transpose(images, (0, 2, 3, 1))  # (B, H, W, C)
    images = (images * 255).astype(np.uint8)

    gt_actions = batch.get(action_key)
    if isinstance(gt_actions, torch.Tensor):
        gt_actions = gt_actions.cpu().float().numpy()
    if pred_actions is not None and isinstance(pred_actions, torch.Tensor):
        pred_actions = pred_actions.cpu().float().numpy()

    B, H, W, _ = images.shape

    # Upscale factor: render trajectories on a larger canvas for readable video
    UP = 4
    out_h, out_w = H * UP, W * UP
    scale = out_w / _WORLD_SIZE

    frames = []
    for i in range(B):
        frame = cv2.resize(images[i], (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        if gt_actions is not None:
            _draw_traj(frame, gt_actions[i], scale, color=(0, 200, 0))
        if pred_actions is not None:
            _draw_traj(frame, pred_actions[i], scale, color=(220, 50, 50))
        frames.append(frame)

    return np.stack(frames, axis=0)


def _draw_traj(frame: np.ndarray, traj: np.ndarray, scale: float, color: tuple) -> None:
    """Overlay a 2-D trajectory as dots+lines on frame in-place."""
    H, W = frame.shape[:2]
    pts = []
    for xy in traj:
        px = int(xy[0] * scale)
        py = int(xy[1] * scale)
        px = max(0, min(W - 1, px))
        py = max(0, min(H - 1, py))
        pts.append((px, py))

    for j in range(1, len(pts)):
        cv2.line(frame, pts[j - 1], pts[j], color, thickness=2, lineType=cv2.LINE_AA)
    for px, py in pts[::2]:
        cv2.circle(frame, (px, py), radius=3, color=color, thickness=-1)
