"""Arc-token action visualization (matches ArcTokEvalVideo viz path).

Training/eval keep ``actions_cartesian`` as ``(M+1, 8)`` arc tokens. For
overlays we detokenize to a time-parameterized ``(H, 14)`` cartesian chunk
(zero rotation) and draw with the embodiment traj/axes helpers — the same
algebra used by ``egomimic.eval.eval_arctok.ArcTokEvalVideo``.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch

from egomimic.rldb.embodiment.embodiment import Embodiment
from egomimic.rldb.embodiment.human import Human
from egomimic.rldb.zarr.arc_length_tokenizer import (
    ARC_TOK_BIMANUAL_DIM,
    TokenizeBimanualArcLengthCartesian,
)

# Defaults match ``train_zarr_cartesian.yaml`` / ``eval_arctok_cartesian``.
DEFAULT_MIN_DISTANCE_UNIT = 0.20
DEFAULT_RESAMPLED_VECTOR_LENGTH = 15
DEFAULT_ROLLOUT_HORIZON = 100


def detokenize_arc_actions(
    arc_actions: np.ndarray | torch.Tensor,
    *,
    min_distance_unit: float = DEFAULT_MIN_DISTANCE_UNIT,
    resampled_vector_length: int = DEFAULT_RESAMPLED_VECTOR_LENGTH,
    action_horizon: int = DEFAULT_ROLLOUT_HORIZON,
) -> np.ndarray:
    """``(M+1, 8)`` arc tokens -> ``(H, 14)`` cartesian (rotation cols = 0)."""
    if isinstance(arc_actions, torch.Tensor):
        arc_actions = arc_actions.detach().cpu().numpy()
    arc = np.asarray(arc_actions, dtype=np.float64)
    if arc.ndim != 2 or arc.shape[1] != ARC_TOK_BIMANUAL_DIM:
        raise ValueError(
            f"Expected arc actions shaped (M+1, {ARC_TOK_BIMANUAL_DIM}), got {arc.shape}"
        )
    if arc.shape[0] != int(resampled_vector_length) + 1:
        raise ValueError(
            f"Expected M+1={int(resampled_vector_length) + 1} tokens "
            f"(M={int(resampled_vector_length)}), got {arc.shape[0]}"
        )

    tok = TokenizeBimanualArcLengthCartesian(
        min_distance_unit=float(min_distance_unit),
        resampled_vector_length=int(resampled_vector_length),
    )
    det = tok.detokenize(arc, action_horizon=int(action_horizon))  # (H, 8)
    out = np.zeros((int(action_horizon), 14), dtype=np.float64)
    out[:, 0:3] = det[:, 0:3]  # L xyz
    out[:, 6:7] = det[:, 3:4]  # L gripper
    out[:, 7:10] = det[:, 4:7]  # R xyz
    out[:, 13:14] = det[:, 7:8]  # R gripper
    return out


def visualize_arc_tokens(
    batch: dict,
    *,
    embodiment_cls: type[Embodiment] = Human,
    mode: Literal["traj", "traj+rotation", "axes"] = "traj",
    action_key: str = "actions_cartesian",
    image_key: str | None = None,
    min_distance_unit: float = DEFAULT_MIN_DISTANCE_UNIT,
    resampled_vector_length: int = DEFAULT_RESAMPLED_VECTOR_LENGTH,
    action_horizon: int = DEFAULT_ROLLOUT_HORIZON,
    **viz_kwargs,
) -> np.ndarray:
    """Detokenize batched arc actions and draw via ``embodiment_cls.viz``.

    Args:
        batch: dataloader batch with ``(B, M+1, 8)`` under ``action_key``.
        embodiment_cls: viz backend (default ``Human``).
        mode: passed through to ``viz_transformed_batch``.
        action_key / image_key: batch keys for actions and front image.
        min_distance_unit / resampled_vector_length / action_horizon: must
            match the tokenizer used when the batch was produced.

    Returns:
        RGB uint8 image for batch index 0.
    """
    arc = batch[action_key][0]
    det14 = detokenize_arc_actions(
        arc,
        min_distance_unit=min_distance_unit,
        resampled_vector_length=resampled_vector_length,
        action_horizon=action_horizon,
    )
    batch_viz = dict(batch)
    batch_viz[action_key] = torch.as_tensor(det14)[None]
    return embodiment_cls.viz_transformed_batch(
        batch_viz,
        mode=mode,
        viz_batch_key=action_key,
        image_key=image_key,
        **viz_kwargs,
    )
