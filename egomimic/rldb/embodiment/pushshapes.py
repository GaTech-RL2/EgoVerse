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
from egomimic.utils.egomimicUtils import draw_dot_on_frame

# ImageNet normalization constants applied by eval_image_augs; we invert them
# before rendering so the image looks correct in the viz frame.
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# PushShapes world is always 512x512 px; observations are rendered at
# image_size (default 96). We upscale the obs image by UP_SCALE before
# drawing trajectories so the video is readable.
_WORLD_SIZE = 512.0
UP_SCALE = 4

# Trajectory styling. Match the main-branch ``draw_actions`` pattern:
# render each chunk as palette-graded dots (light→dark = early→late) on a
# single frame, GT in Greens and pred in Reds.
_GT_PALETTE = "Greens"
_PRED_PALETTE = "Reds"
_DOT_RADIUS = 3
# Per-chunk we render ONE frame (not T frames repeated). Repeat that frame
# this many times so each chunk gets ~1 sec of dwell at 30 fps in the mp4.
_CHUNK_DWELL_FRAMES = 30


def _draw_chunk(
    frame: np.ndarray,
    actions: np.ndarray,
    scale: float,
    palette: str,
    dot_size: int = _DOT_RADIUS,
) -> np.ndarray:
    """Draw an action chunk as palette-graded dots on ``frame``.

    ``actions`` is (N, 2) in world (0–512) coords; ``scale`` maps world→pixel.
    Uses ``draw_dot_on_frame`` (main-branch pattern) so consecutive dots are
    colored along a perceptual gradient — viewer can see chunk ordering at a
    glance. Returns the modified frame (draw_dot_on_frame returns a copy).
    """
    pix = np.asarray(actions, dtype=np.float32).reshape(-1, 2) * float(scale)
    return draw_dot_on_frame(
        frame, pix.tolist(), show=False, palette=palette, dot_size=dot_size
    )


def get_keymap_hpt(action_horizon: int = 32, **kwargs) -> dict:
    """HPT-specific keymap: SINGLE-FRAME obs + action_horizon-long action chunk.

    HPT's contract is "one current obs -> predict next action_horizon actions".
    Unlike H-Net (which sees a per-token obs window aligned with actions and
    is meant to encode trajectories), HPT must condition on the CURRENT obs
    only. Returning windowed obs would make training trivial (model just reads
    the future obs out of its context) while inference, which only has the
    current obs, becomes wildly OOD. Diagnosed 2026-05-19: closed-loop sim
    rollout was failing despite low training loss because obs windowing
    let the model cheat at train time.

    Extra kwargs (e.g. ``norm_mode=True`` from trainHydra) are accepted and
    ignored so norm-stat collection doesn't crash.
    """
    return {
        "front_img_1": {
            "key_type": "camera_keys",
            "zarr_key": "observations.images.front_img_1",
            # no horizon -> single-frame obs (B, C, H, W) per sample
        },
        "state_agent_obj": {
            "key_type": "proprio_keys",
            "zarr_key": "observations.state",
            # no horizon -> single-frame (B, D) per sample
        },
        "actions": {
            "key_type": "action_keys",
            "zarr_key": "actions",
            "horizon": int(action_horizon),
        },
    }


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
    # Per-frame obs (obs_t aligned with action_t): give the obs keys the same
    # horizon as actions so the dataloader returns (T, ...) windows rather than
    # a single broadcast frame. CondEncoderModule.encode then skips its
    # unsqueeze-and-expand branch (kicks in only when x.dim()==2 for state /
    # ==4 for images), and AdaLN sees a true per-token cond.
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


def get_keymap_eval(action_horizon: int = 32, **kwargs) -> dict:
    """``get_keymap`` plus a ``goal_pose`` passthrough for closed-loop sim eval.

    ``PackedSimEval.batch_to_env_init`` (``egomimic/eval/core/eval_sim.py``)
    reads ``batch["goal_pose"]`` to set the PushShapes env goal when the rollout
    inits from a replayed val episode. The training keymap omits it (training
    never uses the goal). ``goal_pose`` is declared with ``key_type:
    "goal_keys"``, which is **not** in
    ``MultiDataset.NORMALIZE_KEY_TYPES = ("proprio_keys", "action_keys")``, so
    NormStats reads it into the packed batch and passes it straight through —
    raw, un-normalized — to the evaluator. Same map serves both circle proxies.

    NOTE: this intentionally omits EgoVerse2's extra ``init_action`` passthrough
    (a raw-actions seed for delta-rollout integration). pact-2's eval stack uses
    ``init_mode: "replay"`` and never reads ``init_action`` (verified: no
    reference in ``egomimic/eval`` or ``egomimic/algo``), so adding it would be
    dead weight. If a delta-rollout path lands later, re-add it then.
    """
    km = get_keymap(action_horizon=action_horizon)
    km["goal_pose"] = {
        "key_type": "goal_keys",
        "zarr_key": "goal_pose",
        "horizon": int(action_horizon),
    }
    return km


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

    # Per-frame mode: images is (B, T, H, W, C). Emit ONE output panel-frame
    # per REAL timestep so this panel is length-aligned with PCA + boundary
    # strip panels (both also per-timestep). The composite then has all
    # panels temporally in sync — no chunk-dwell, no padding-to-black.
    if images.ndim == 5:
        B, T_max, h, w, _ = images.shape
        out_h, out_w = h * UP_SCALE, w * UP_SCALE
        scale = out_w / _WORLD_SIZE
        if seq_lens is None:
            seq_lens = np.full(B, T_max, dtype=np.int64)
        else:
            seq_lens = np.asarray(seq_lens).astype(np.int64)

        # Allow callers (e.g. HNetEvalVideo via **_unused_kwargs) to cap
        # episodes via max_videos; default = no cap. Keeps composite
        # panels short and aligned with other evals that cap the same way.
        max_videos = _unused_kwargs.get("max_videos")
        B_render = min(B, int(max_videos)) if max_videos is not None else B
        frames: list[np.ndarray] = []
        for b in range(B_render):
            T_b = max(1, int(seq_lens[b]))
            for t in range(T_b):
                base = cv2.resize(
                    images[b, t], (out_w, out_h), interpolation=cv2.INTER_LINEAR
                )
                # Overlay full GT + pred trajectories on every frame (static
                # trail; the moving obs background is what differentiates
                # frames). Palette-graded dots: light=early, dark=late.
                if gt_actions is not None:
                    base = _draw_chunk(base, gt_actions[b, :T_b], scale, _GT_PALETTE)
                if pred_actions is not None:
                    base = _draw_chunk(
                        base, pred_actions[b, :T_b], scale, _PRED_PALETTE
                    )
                frames.append(base)
            # 5-frame black separator between episodes — matches PCA +
            # boundary strip separator behaviour for clean per-episode breaks.
            if b < B_render - 1:
                sep = np.zeros((5, out_h, out_w, 3), dtype=np.uint8)
                frames.extend(list(sep))
        return np.stack(frames, axis=0)

    # Single-frame legacy path: (B, H, W, C). One image per sample with the
    # same palette-graded chunk overlay.
    b_count, h, w, _ = images.shape
    out_h, out_w = h * UP_SCALE, w * UP_SCALE
    scale = out_w / _WORLD_SIZE

    frames = []
    for i in range(b_count):
        frame = cv2.resize(images[i], (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        if gt_actions is not None:
            frame = _draw_chunk(frame, gt_actions[i], scale, _GT_PALETTE)
        if pred_actions is not None:
            frame = _draw_chunk(frame, pred_actions[i], scale, _PRED_PALETTE)
        frames.append(frame)
    return np.stack(frames, axis=0)
