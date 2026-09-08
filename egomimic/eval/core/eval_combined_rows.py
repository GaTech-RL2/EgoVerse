"""CombinedRowsEval — wraps an EvalListSideBySide and vstacks per-embodiment
composites into a single video with one row per embodiment.

Row order follows ``embodiment_order`` (list of emb name strings, e.g.
["pushshapes_sim", "pushshapes_sim_stick"]). The first name is the top row.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torchvision.io as tvio

from egomimic.eval.core.eval_video import EvalVideo
from egomimic.rldb.embodiment.embodiment import get_embodiment_id


class CombinedRowsEval(EvalVideo):

    def __init__(
        self,
        inner_eval: EvalVideo,
        embodiment_order: list[str],
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
        self.inner_eval = inner_eval
        self.emb_id_order = [get_embodiment_id(name) for name in embodiment_order]

    def _wire(self):
        self.inner_eval.trainer = self.trainer
        self.inner_eval.model = self.model

    def on_validation_start(self):
        self._wire()
        self.inner_eval.on_validation_start()
        if self.trainer.is_global_zero:
            os.makedirs(
                os.path.join(self.video_dir(), f"epoch_{self.trainer.current_epoch}"),
                exist_ok=True,
            )

    def on_validation_step(self, batch, batch_idx, dataloader_idx=0):
        self._wire()
        self.inner_eval.on_validation_step(batch, batch_idx, dataloader_idx)

    def compute_metrics_and_viz(
        self, batch: Dict[int, Dict[str, Any]]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[int, np.ndarray]]:
        return {}, {}

    def on_validation_end(self):
        self._wire()
        buffers = self.inner_eval.val_image_buffer
        rows = []
        for emb_id in self.emb_id_order:
            buf = buffers.get(emb_id, [])
            if buf:
                rows.append(torch.stack(buf).numpy())

        if len(rows) >= 2:
            max_n = max(r.shape[0] for r in rows)
            max_w = max(r.shape[2] for r in rows)
            padded = []
            for r in rows:
                n, h, w, c = r.shape
                if n < max_n:
                    last = r[-1:] if n > 0 else np.zeros((1, h, w, c), dtype=r.dtype)
                    r = np.concatenate([r, np.repeat(last, max_n - n, axis=0)], axis=0)
                    n = max_n
                if w < max_w:
                    r = np.concatenate(
                        [r, np.zeros((n, h, max_w - w, c), dtype=r.dtype)], axis=2
                    )
                padded.append(r)
            combined = np.concatenate(padded, axis=1)
            # Ensure even dims for h264
            n, h, w, c = combined.shape
            if h % 2 == 1:
                combined = np.concatenate(
                    [combined, np.zeros((n, 1, w, c), dtype=combined.dtype)], axis=1
                )
            if w % 2 == 1:
                combined = np.concatenate(
                    [combined, np.zeros((n, h + (h % 2), 1, c), dtype=combined.dtype)],
                    axis=2,
                )
            path = os.path.join(
                self.video_dir(),
                f"epoch_{self.trainer.current_epoch}",
                "combined_rows.mp4",
            )
            tvio.write_video(path, torch.from_numpy(combined), fps=30, video_codec="h264")

        self.inner_eval.on_validation_end()
