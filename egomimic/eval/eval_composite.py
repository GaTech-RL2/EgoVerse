"""
EvalList — composite evaluator that runs multiple sub-evals on the same
validation batch.

Two flavours:

* :class:`EvalList` (base): runs each sub-eval in turn, merges their
  metrics, and returns the **union** of their image dicts. Each sub-eval
  emits its own .mp4 to its own buffer / output directory, so the saved
  validation videos are independent per sub-eval. Use this when the
  sub-evals don't share a common timeline or you just want separate
  diagnostics videos.

* :class:`EvalListSideBySide` (subclass): in addition to running each
  sub-eval, concatenates per-embodiment image arrays along width so the
  saved composite .mp4 plays the panels side-by-side. Requires all
  sub-evals to emit the same number of frames per embodiment and uses
  ``pad_h`` to reconcile differing panel heights.

Each sub-eval must satisfy the ``EvalVideo.compute_metrics_and_viz``
contract: ``(metrics_dict, {emb_id: (N, H, W, 3) uint8})``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from egomimic.eval.eval_video import EvalVideo


class EvalList(EvalVideo):
    """Run a list of sub-evals on each val batch. Metrics merge; each
    sub-eval's images get forwarded unmodified.

    Args (yaml):
      evals: list of pre-instantiated sub-evals (each is an EvalVideo).
      limit_val_batches: inherited from EvalVideo — caps how many val
        batches the parent strategy processes.
    """

    def __init__(
        self,
        evals: list[EvalVideo] | None = None,
        limit_val_batches: int = 4,
        viz_func: dict | None = None,
        transform_lists: dict | None = None,
    ):
        super().__init__(
            limit_val_batches=limit_val_batches,
            viz_func=viz_func,
            transform_lists=transform_lists,
        )
        self.evals: list[EvalVideo] = list(evals or [])

    # ------------------------------------------------------------------ #

    def _wire_subevals(self):
        for ev in self.evals:
            ev.trainer = self.trainer
            ev.model = self.model

    def on_validation_start(self):
        self._wire_subevals()
        for ev in self.evals:
            ev.on_validation_start()
        # Skip the parent's video-dir mkdir; this composite has no buffer
        # of its own. Each sub-eval owns its directory.

    def on_validation_end(self):
        for ev in self.evals:
            ev.on_validation_end()

    def on_validation_step(self, batch, batch_idx, dataloader_idx=0):
        """Delegate to each sub-eval's own ``on_validation_step``.

        Each sub-eval owns its own ``val_image_buffer`` (inherited from
        ``EvalVideo``), writes its own ``.mp4``, and logs its own metrics.
        The composite class doesn't keep a buffer or write any video —
        if you want the panels combined into one .mp4, use
        :class:`EvalListSideBySide` instead.
        """
        self._wire_subevals()
        for ev in self.evals:
            ev.on_validation_step(batch, batch_idx, dataloader_idx)

    def compute_metrics_and_viz(
        self, batch: Dict[int, Dict[str, Any]]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[int, np.ndarray]]:
        """Plain pass-through: run each sub-eval, merge metrics, collect
        their image dicts into ``self.last_per_eval_images`` for the
        side-by-side subclass. Returns an empty image dict so the
        ``EvalVideo`` parent path doesn't try to write a composite video.
        """
        self._wire_subevals()
        all_metrics: Dict[str, torch.Tensor] = {}
        self.last_per_eval_images: List[Dict[int, np.ndarray]] = []
        for ev in self.evals:
            m, ims = ev.compute_metrics_and_viz(batch)
            all_metrics.update(m)
            self.last_per_eval_images.append(ims)
        return all_metrics, {}


class EvalListSideBySide(EvalList):
    """Side-by-side composite. Each sub-eval emits ``(N, H, W, 3)`` for the
    same ``N`` (frame count); we concatenate the panels along ``W``.

    ``pad_h``: ``"min"`` crops each panel to the shortest height;
    ``"max"`` zero-pads each to the tallest. Default ``"min"`` produces
    a clean composite frame.
    """

    def __init__(self, pad_h: str = "min", **kwargs):
        super().__init__(**kwargs)
        if pad_h not in {"min", "max"}:
            raise ValueError(f"pad_h must be 'min' or 'max', got {pad_h!r}")
        self.pad_h = pad_h

    def on_validation_start(self):
        # Re-enable the parent's video-dir mkdir; this composite DOES own
        # an output buffer (the merged side-by-side .mp4).
        self._wire_subevals()
        for ev in self.evals:
            ev.on_validation_start()
        # Skip parent (EvalList) implementation that suppresses mkdir;
        # walk up to EvalVideo for the buffer setup.
        EvalVideo.on_validation_start(self)

    def on_validation_end(self):
        # Flush the composite buffer (merged side-by-side mp4), then
        # let each sub-eval finalize its own state.
        EvalVideo.on_validation_end(self)
        for ev in self.evals:
            ev.on_validation_end()

    def on_validation_step(self, batch, batch_idx, dataloader_idx=0):
        # Use the parent ``EvalVideo`` path so the merged images_dict
        # produced by ``compute_metrics_and_viz`` below ends up in
        # ``self.val_image_buffer`` and flushes as ONE composite .mp4.
        EvalVideo.on_validation_step(self, batch, batch_idx, dataloader_idx)

    def compute_metrics_and_viz(
        self, batch: Dict[int, Dict[str, Any]]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[int, np.ndarray]]:
        all_metrics, _ = super().compute_metrics_and_viz(batch)
        per_eval_images = self.last_per_eval_images
        merged: Dict[int, np.ndarray] = {}
        if not per_eval_images:
            return all_metrics, merged

        all_emb_ids = sorted({eid for d in per_eval_images for eid in d.keys()})
        for emb_id in all_emb_ids:
            panels = [d[emb_id] for d in per_eval_images if emb_id in d]
            if not panels:
                continue
            heights = [p.shape[1] for p in panels]
            target_h = min(heights) if self.pad_h == "min" else max(heights)
            aligned = []
            for p in panels:
                N, H, W, C = p.shape
                if H == target_h:
                    aligned.append(p)
                elif H > target_h:
                    aligned.append(p[:, :target_h])
                else:
                    pad = np.zeros((N, target_h - H, W, C), dtype=p.dtype)
                    aligned.append(np.concatenate([p, pad], axis=1))
            target_N = max(a.shape[0] for a in aligned)
            for i, a in enumerate(aligned):
                if a.shape[0] < target_N:
                    pad = np.zeros(
                        (target_N - a.shape[0], a.shape[1], a.shape[2], a.shape[3]),
                        dtype=a.dtype,
                    )
                    aligned[i] = np.concatenate([a, pad], axis=0)
            stacked = np.concatenate(aligned, axis=2)
            # libx264 requires both spatial dims to be divisible by 2.
            # Right-pad / bottom-pad a single black row/col if needed.
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
