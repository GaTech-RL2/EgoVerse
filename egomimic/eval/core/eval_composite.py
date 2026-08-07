"""
Composite evaluators.

* :class:`EvalList` (inherits from :class:`Eval`): pure runner — calls
  each sub-eval's lifecycle hooks in turn and merges their metrics.
  Has NO video machinery of its own. Use this as the top-level
  orchestrator combining heterogeneous sub-evals (e.g. one teacher-forced
  composite video + one closed-loop sim eval), each of which writes its
  own outputs independently.

* :class:`EvalVideoList` (inherits from :class:`EvalVideo`): composite
  video — runs N ``EvalVideo`` sub-evals and concatenates their per-step
  panels along width into ONE side-by-side mp4. Sub-evals MUST inherit
  from EvalVideo (the contract is that they produce ``(N, H, W, 3)``
  uint8 arrays per embodiment).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from egomimic.eval.core.eval import Eval
from egomimic.eval.core.eval_video import EvalVideo


class EvalList(Eval):
    """Pure list-runner. No own video output; each sub-eval writes its
    own outputs. Sub-evals can be any ``Eval`` (video or not)."""

    def __init__(self, evals: list | None = None):
        super().__init__()
        self.evals: list = list(evals or [])

    def _wire_subevals(self):
        for ev in self.evals:
            ev.trainer = self.trainer
            ev.model = self.model

    def on_validation_start(self):
        self._wire_subevals()
        for ev in self.evals:
            ev.on_validation_start()

    def on_validation_end(self):
        for ev in self.evals:
            ev.on_validation_end()

    def on_validation_step(self, batch, batch_idx, dataloader_idx=0):
        self._wire_subevals()
        for ev in self.evals:
            ev.on_validation_step(batch, batch_idx, dataloader_idx)

    def compute_metrics_and_viz(
        self, batch: Dict[int, Dict[str, Any]]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[int, np.ndarray]]:
        """Run each sub-eval, merge metrics. Returns empty image dict —
        each sub-eval owns its own video buffer / output directory."""
        self._wire_subevals()
        all_metrics: Dict[str, torch.Tensor] = {}
        for ev in self.evals:
            m, _ = ev.compute_metrics_and_viz(batch)
            all_metrics.update(m)
        return all_metrics, {}


class EvalVideoList(EvalVideo):
    """Composite video evaluator.

    Runs N EvalVideo sub-evals on the same batch and concatenates their
    ``(N, H, W, 3)`` panels along WIDTH (per embodiment) into one mp4.
    All sub-evals MUST inherit from EvalVideo and produce the same
    number of frames ``N`` per embodiment (or shorter panels get
    zero-padded to ``max`` N).

    pad_h: how to reconcile differing panel heights.
      min: crop each panel to the shortest height.
      max: zero-pad each to the tallest. Use this when sub-evals
      legitimately have different heights (e.g. PCA fig is taller than
      boundary strip).
    """

    def __init__(
        self,
        evals: list[EvalVideo] | None = None,
        pad_h: str = "min",
        below_indices: list | None = None,
        limit_val_batches: int = 4,
        viz_func: dict | None = None,
        transform_lists: dict | None = None,
        max_videos: int | None = None,
    ):
        super().__init__(
            limit_val_batches=limit_val_batches,
            viz_func=viz_func,
            transform_lists=transform_lists,
            max_videos=max_videos,
        )
        if pad_h not in {"min", "max"}:
            raise ValueError(f"pad_h must be 'min' or 'max', got {pad_h!r}")
        self.pad_h = pad_h
        # Sub-eval indices whose panels are vstacked BELOW the hstacked top
        # row (rather than concatenated into it along width). Used for the
        # boundary strip so it sits under the traj+PCA row, full-width.
        self.below_indices = set(int(i) for i in (below_indices or []))

        evals_list: list[EvalVideo] = list(evals or [])
        # Enforce the EvalVideo-only contract: every sub-eval must emit
        # frames (otherwise the width-concat has nothing to stack).
        for ev in evals_list:
            if not isinstance(ev, EvalVideo):
                raise TypeError(
                    f"EvalVideoList sub-evals must inherit from EvalVideo "
                    f"(got {type(ev).__name__!r}). Use EvalList for non-video sub-evals."
                )
        self.evals: list[EvalVideo] = evals_list

    def _wire_subevals(self):
        for ev in self.evals:
            ev.trainer = self.trainer
            ev.model = self.model

    def on_validation_start(self):
        self._wire_subevals()
        for ev in self.evals:
            ev.on_validation_start()
        # ALSO set up our own image buffer + video dir — the composite
        # mp4 is written through THIS class's buffer.
        super().on_validation_start()

    def on_validation_end(self):
        for ev in self.evals:
            ev.on_validation_end()
        super().on_validation_end()

    def on_validation_step(self, batch, batch_idx, dataloader_idx=0):
        """Standard EvalVideo step path: compute_metrics_and_viz (below)
        returns the merged side-by-side frames, EvalVideo's machinery
        then buffers + flushes them to ONE mp4."""
        self._wire_subevals()
        super().on_validation_step(batch, batch_idx, dataloader_idx)

    def compute_metrics_and_viz(
        self, batch: Dict[int, Dict[str, Any]]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[int, np.ndarray]]:
        self._wire_subevals()
        all_metrics: Dict[str, torch.Tensor] = {}
        per_eval_images: List[Dict[int, np.ndarray]] = []
        for ev in self.evals:
            m, ims = ev.compute_metrics_and_viz(batch)
            all_metrics.update(m)
            per_eval_images.append(ims)

        # Concatenate sub-eval panels along width per embodiment.
        merged: Dict[int, np.ndarray] = {}
        if not per_eval_images:
            return all_metrics, merged

        def _frame_pad(panels):
            # Pad N (frame count) so all panels share the same time axis.
            target_N = max(p.shape[0] for p in panels)
            out = []
            for p in panels:
                if p.shape[0] < target_N:
                    pad = np.zeros(
                        (target_N - p.shape[0], p.shape[1], p.shape[2], p.shape[3]),
                        dtype=p.dtype,
                    )
                    out.append(np.concatenate([p, pad], axis=0))
                else:
                    out.append(p)
            return out

        all_emb_ids = sorted({eid for d in per_eval_images for eid in d.keys()})
        for emb_id in all_emb_ids:
            # Split panels into "right" (hstacked into the top row) and
            # "below" (vstacked beneath the top row, full-width).
            right_panels = [
                per_eval_images[i][emb_id]
                for i in range(len(per_eval_images))
                if i not in self.below_indices and emb_id in per_eval_images[i]
            ]
            below_panels = [
                per_eval_images[i][emb_id]
                for i in range(len(per_eval_images))
                if i in self.below_indices and emb_id in per_eval_images[i]
            ]
            if not right_panels and not below_panels:
                continue

            # ----- top row: hstack right_panels, heights reconciled via pad_h -----
            top_row = None
            if right_panels:
                heights = [p.shape[1] for p in right_panels]
                target_h = min(heights) if self.pad_h == "min" else max(heights)
                aligned = []
                for p in right_panels:
                    N, H, W, C = p.shape
                    if H == target_h:
                        aligned.append(p)
                    elif H > target_h:
                        aligned.append(p[:, :target_h])
                    else:
                        pad = np.zeros((N, target_h - H, W, C), dtype=p.dtype)
                        aligned.append(np.concatenate([p, pad], axis=1))
                aligned = _frame_pad(aligned)
                top_row = np.concatenate(aligned, axis=2)

            # ----- below: vstack each below panel under the top row -----
            if below_panels:
                widths = [p.shape[2] for p in below_panels]
                if top_row is not None:
                    widths.append(top_row.shape[2])
                target_w = max(widths)

                def _pad_w(p):
                    N, H, W, C = p.shape
                    if W < target_w:
                        pad = np.zeros((N, H, target_w - W, C), dtype=p.dtype)
                        return np.concatenate([p, pad], axis=2)
                    return p

                all_rows = []
                if top_row is not None:
                    all_rows.append(_pad_w(top_row))
                for bp_ in below_panels:
                    all_rows.append(_pad_w(bp_))
                all_rows = _frame_pad(all_rows)
                stacked = np.concatenate(all_rows, axis=1)
            else:
                stacked = top_row

            # libx264 needs both spatial dims even — pad odd by one row/col.
            N, H, W, C = stacked.shape
            if H % 2 == 1:
                stacked = np.concatenate(
                    [stacked, np.zeros((N, 1, W, C), dtype=stacked.dtype)], axis=1
                )
                H += 1
            if W % 2 == 1:
                stacked = np.concatenate(
                    [stacked, np.zeros((N, H, 1, C), dtype=stacked.dtype)], axis=2
                )
            merged[emb_id] = stacked
        return all_metrics, merged


# Backwards-compat alias — old yamls / scripts may still reference this name.
# Will be removed in a follow-up after all yamls are updated.
EvalListSideBySide = EvalVideoList
