"""
BoundaryStripEval — renders the chunker's boundary probability over time
as a thin vertical strip.

For each validation episode, we run the algo's teacher-forced
``forward_packed`` and pull ``boundary_prob`` out of ``ctx.aux`` (each
chunker stage emits a ``bpred`` with shape ``(T_total, 2)`` where column
1 is ``P(boundary)`` per token). Then for each frame ``t`` of the video,
we render a centered ``window`` of the surrounding ``[t - W/2, t + W/2]``
boundary probs as a vertical column of colored squares — bright when
the chunker would fire a boundary, dark when it wouldn't.

The strip's height is fixed (one square per timestep in the window) and
its width is configurable. Each frame's strip is independent so a
viewer scrubbing through the video can see the chunker's recent /
upcoming boundary decisions.

If multiple chunkers are present, the strip column groups them
vertically (one sub-strip per chunker, top-most first).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from egomimic.eval.eval_video import EvalVideo


def _color_for_prob(p: float) -> tuple[int, int, int]:
    """RGB colour for a boundary probability ``p`` in [0, 1].

    Sweeps cool→warm: 0 = dark blue, 0.5 = grey-green, 1 = bright red. Easy
    to read in a strip without a colorbar.
    """
    p = float(np.clip(p, 0.0, 1.0))
    r = int(255 * p)
    g = int(255 * (1 - abs(2 * p - 1)))
    b = int(255 * (1 - p))
    return (r, g, b)


class BoundaryStripEval(EvalVideo):
    """Per-frame centered window of chunker P(boundary), rendered as
    coloured squares.

    Args (yaml):
      window: number of timesteps in the centered window (default 64).
        The strip is ``window`` squares tall.
      strip_width: pixel width of each square (default 16).
      square_height: pixel height of each square (default 8).
        Strip height = ``window * square_height``.
      future_pad: how to fill the window's tail past the episode's last
        frame: ``"black"`` (default) or ``"clamp"`` (repeat last value).
      chunker_idx: which chunker's bpred to render. ``None`` = stack all
        chunkers' strips top-to-bottom.
    """

    def __init__(
        self,
        window: int = 64,
        strip_width: int = 16,
        square_height: int = 8,
        future_pad: str = "black",
        chunker_idx: int | None = None,
        limit_val_batches: int = 4,
        viz_func: dict | None = None,
        transform_lists: dict | None = None,
    ):
        super().__init__(
            limit_val_batches=limit_val_batches,
            viz_func=viz_func,
            transform_lists=transform_lists,
        )
        self.window = int(window)
        self.strip_width = int(strip_width)
        self.square_height = int(square_height)
        if future_pad not in {"black", "clamp"}:
            raise ValueError("future_pad must be 'black' or 'clamp'")
        self.future_pad = future_pad
        self.chunker_idx = chunker_idx

    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _run_forward_and_collect_bprobs(
        self, batch: Dict[int, Dict[str, Any]]
    ) -> Dict[int, List[torch.Tensor]]:
        """Run ``policy.forward_packed`` for each emb and collect the list
        of per-chunker boundary_prob tensors. Returns
        ``{emb_id: [boundary_prob_packed_chunker_0, …]}``.
        """
        algo = self.model
        out: Dict[int, List[torch.Tensor]] = {}
        for emb_id, _batch in batch.items():
            if not _batch.get("_packed", False):
                continue
            ac_key = algo.resolved_ac_keys[emb_id]
            obs = algo._build_obs(_batch, emb_id)
            actions = _batch[ac_key]
            cu = _batch["cu_seqlens"]
            max_seqlen = int(_batch["max_seq_len"])
            policy = algo.nets["policy"]
            _pred, aux = policy.forward_packed(
                actions,
                obs,
                cu,
                max_seqlen,
            )
            # ``aux`` is list[dict]; each chunker entry has 'bpred'.
            bprobs: List[torch.Tensor] = []
            for entry in aux:
                bp = entry.get("bpred") if isinstance(entry, dict) else None
                if bp is None:
                    continue
                # bp.boundary_prob: packed (T_total, 2)
                bprobs.append(bp.boundary_prob[..., 1].detach().cpu())
            out[emb_id] = bprobs
        return out

    def _render_strip_for_episode(
        self,
        bprob_packed: torch.Tensor,
        T_ep: int,
    ) -> np.ndarray:
        """Render the per-frame centered window strip for one chunker on
        one episode. Returns ``(T_ep + sep, strip_H, strip_W, 3)`` uint8.
        """
        T_ep = int(T_ep)
        W = self.window
        sq_h = self.square_height
        sq_w = self.strip_width
        strip_H = W * sq_h
        strip_W = sq_w

        # bprob_packed is (T_ep,) for this episode.
        bp = bprob_packed.numpy()
        if bp.shape[0] < T_ep:
            # Defensive: pad with zeros so indexing is safe.
            bp = np.concatenate([bp, np.zeros(T_ep - bp.shape[0])])

        frames = np.zeros((T_ep, strip_H, strip_W, 3), dtype=np.uint8)
        for t in range(T_ep):
            half = W // 2
            for i in range(W):
                idx = t - half + i  # global index into bp
                if 0 <= idx < bp.shape[0]:
                    color = _color_for_prob(bp[idx])
                elif self.future_pad == "clamp":
                    clamped = max(0, min(bp.shape[0] - 1, idx))
                    color = _color_for_prob(bp[clamped])
                else:
                    color = (0, 0, 0)
                y0 = i * sq_h
                frames[t, y0 : y0 + sq_h, :] = color
            # Mark current step with a thin yellow horizontal line at center.
            ymid = half * sq_h + sq_h // 2
            frames[t, ymid - 1 : ymid + 1, :] = (255, 220, 0)
        return frames

    # ------------------------------------------------------------------ #

    def compute_metrics_and_viz(
        self, batch: Dict[int, Dict[str, Any]]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[int, np.ndarray]]:
        metrics: Dict[str, torch.Tensor] = {}
        images_dict: Dict[int, np.ndarray] = {}
        bprob_by_emb = self._run_forward_and_collect_bprobs(batch)

        for emb_id, _batch in batch.items():
            bprobs = bprob_by_emb.get(emb_id, [])
            if not bprobs:
                continue
            chosen = bprobs if self.chunker_idx is None else [bprobs[self.chunker_idx]]
            cu = _batch["cu_seqlens"]
            seq_lens = _batch["seq_lens"]
            B = int(seq_lens.shape[0])

            ep_panels: List[np.ndarray] = []
            for b in range(B):
                s = int(cu[b].item())
                e = int(cu[b + 1].item())
                T_ep = e - s
                # Stack per-chunker strips vertically (one row per chunker).
                per_chunker_panels = []
                for bp in chosen:
                    panel = self._render_strip_for_episode(bp[s:e], T_ep)
                    per_chunker_panels.append(panel)
                # All have same T_ep; concat along height (axis=1).
                stacked = np.concatenate(per_chunker_panels, axis=1)
                ep_panels.append(stacked)
                # Black separator (5 frames) between episodes.
                if b < B - 1:
                    sep = np.zeros(
                        (5, stacked.shape[1], stacked.shape[2], 3), dtype=np.uint8
                    )
                    ep_panels.append(sep)

            images_dict[emb_id] = np.concatenate(ep_panels, axis=0)

            # Aggregate metric: fraction of tokens where P(boundary) > 0.5
            # (one per chunker).
            for ci, bp in enumerate(bprobs):
                rate = float((bp > 0.5).float().mean().item())
                metrics[f"Valid/emb{emb_id}_chunker{ci}_pboundary_gt05"] = torch.tensor(
                    rate, device=self.trainer.lightning_module.device
                )

        return metrics, images_dict
