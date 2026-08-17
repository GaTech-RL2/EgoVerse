"""Eval for arc-tokenized 21-keypoint hand-pose policies.

These models emit ``(B, M+1, 138)``: M waypoints plus one velocity row, in the
bimanual keypoint layout ``[L wrist(6) | L kp(63) | R wrist(6) | R kp(63)]``.

Three things downstream want a *time-indexed* chunk instead, and all three
break on a raw token:

  * the arc-matched metric, which resamples a cartesian path;
  * the wrist-frame -> camera-frame revert transform, whose reshape is written
    against a fixed ``(H, 21, 3)`` layout, so a 31-row token fails with
    ``cannot reshape array of size 1953 into shape (100, 21, 3)``;
  * the viz overlay, which projects one dot per timestep.

So this evaluator detokenizes once and feeds the reconstruction to all three,
via the ``_arc_match_source`` / ``_transform_source`` hooks on the base class.
That is the same thing an open-loop rollout would send to the controller, so
the resulting videos and cam-frame metrics are faithful to deploy.

Handles both keypoint tokenizers, which share the ``(M+1, 138)`` output:
  * ``TokenizeBimanualArcLengthKeypoints`` — one clock for the whole hand.
  * ``TokenizeBimanualHybridArcKeypoints`` — independent eef / keypoint
    clocks, reconstructed over their common in-token window.
"""

from __future__ import annotations

import numpy as np
import torch

from egomimic.eval.eval_hpt import HPTEvalVideo
from egomimic.rldb.zarr.keypoint_arc_tokenizer import (
    BIMANUAL_DIM,
    TokenizeBimanualArcLengthKeypoints,
    TokenizeBimanualHybridArcKeypoints,
)


class KeypointArcTokEvalVideo(HPTEvalVideo):
    """HPT evaluator for keypoint arc-token heads.

    Config knobs (yaml):
      * ``hybrid``: if true, build the two-stream tokenizer, whose rollout
        window is ``min_s(D_s / v_s)``; otherwise the single-clock one.
      * ``min_distance_unit`` / ``min_distance_unit_eef`` /
        ``min_distance_unit_kp``: D per stream. MUST match the data pipeline —
        detokenizing with a different D silently rescales the reconstruction.
      * ``resampled_vector_length``: M. Model output must be ``(B, M+1, 138)``.
      * ``rollout_horizon``: H frames to reconstruct per sample.
      * ``kp_distance_mode``: which norm the keypoint stream integrates.
      * ``dt``: control period (default 1/30).
    """

    def __init__(
        self,
        limit_val_batches: int = 400,
        viz_func: dict = None,
        transform_lists: dict | None = None,
        hybrid: bool = False,
        min_distance_unit: float = 0.45,
        min_distance_unit_eef: float = 0.45,
        min_distance_unit_kp: float = 0.41,
        resampled_vector_length: int = 30,
        rollout_horizon: int = 100,
        kp_distance_mode: str = "linf",
        dt: float = 1.0 / 30.0,
        **kwargs,
    ):
        super().__init__(
            limit_val_batches=limit_val_batches,
            viz_func=viz_func,
            transform_lists=transform_lists,
            **kwargs,
        )
        if hybrid:
            self._detokenizer = TokenizeBimanualHybridArcKeypoints(
                min_distance_unit_eef=float(min_distance_unit_eef),
                min_distance_unit_kp=float(min_distance_unit_kp),
                resampled_vector_length=int(resampled_vector_length),
                kp_distance_mode=kp_distance_mode,
                dt=float(dt),
            )
        else:
            self._detokenizer = TokenizeBimanualArcLengthKeypoints(
                min_distance_unit=float(min_distance_unit),
                resampled_vector_length=int(resampled_vector_length),
                distance_mode=kp_distance_mode,
                dt=float(dt),
            )
        self.hybrid = bool(hybrid)
        self.rollout_horizon = int(rollout_horizon)
        self._M = int(resampled_vector_length)

    # ------------------------------------------------------------------

    def _detokenize_batch(self, tokens) -> torch.Tensor:
        """(B, M+1, 138) tokens -> (B, H, 138) time-indexed chunks."""
        t = tokens if isinstance(tokens, torch.Tensor) else torch.as_tensor(tokens)
        arr = t.detach().cpu().numpy().astype(np.float64)

        if arr.ndim != 3 or arr.shape[-1] != BIMANUAL_DIM:
            raise ValueError(
                f"expected keypoint arc tokens (B, M+1, {BIMANUAL_DIM}), got {arr.shape}"
            )
        if arr.shape[1] != self._M + 1:
            raise ValueError(
                f"configured for M={self._M} (M+1={self._M + 1} rows), got "
                f"{arr.shape[1]} rows. The evaluator's resampled_vector_length "
                f"must match the tokenizer that produced the training data."
            )

        H = self.rollout_horizon
        out = np.stack(
            [self._detokenizer.detokenize(arr[b], action_horizon=H) for b in range(len(arr))]
        )
        return torch.from_numpy(out).to(t.device, dtype=t.dtype)

    def _is_token(self, tensor) -> bool:
        return (
            tensor is not None
            and getattr(tensor, "ndim", 0) == 3
            and tensor.shape[-1] == BIMANUAL_DIM
            and tensor.shape[1] == self._M + 1
        )

    # ------------------------------------------------------------------
    # Base-class hooks. Both reconstruct; guarded so an already-detokenized
    # tensor passes through untouched.

    def _arc_match_source(self, tensor):
        if tensor is None:
            return None
        t = tensor if isinstance(tensor, torch.Tensor) else torch.as_tensor(tensor)
        return self._detokenize_batch(t) if self._is_token(t) else t

    def _transform_source(self, tensor):
        if tensor is None:
            return None
        t = tensor if isinstance(tensor, torch.Tensor) else torch.as_tensor(tensor)
        return self._detokenize_batch(t) if self._is_token(t) else t
