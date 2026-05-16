"""
BoundaryStripEval — renders the chunker's boundary probability AND its
committed boundary decisions over time as a pair of thin vertical strips.

For each validation episode we run the algo's teacher-forced
``forward_packed`` and pull ``boundary_prob`` plus ``boundary_mask`` out
of ``ctx.aux`` (each chunker stage emits a ``bpred`` with
``boundary_prob: (T_total, 2)`` — column 1 = ``P(boundary)`` — and
``boundary_mask: (T_total,)`` — the chunker's committed fire decisions
after the STE / argmax).

Per video frame ``t``, two side-by-side strips of the centered window
``[t - W/2, t + W/2]`` are rendered:

  * **Gradient strip** — continuous greyscale of soft ``P(boundary)``.
    White = quiet, black = about to fire. Use to read confidence and
    spot step-to-step jitter.
  * **Discrete strip** — binary red/white of ``boundary_mask``. Red rows
    = committed chunk dividers. Use to read decisions: chunk sizes,
    rhythm, alignment with motion.

Each timestep occupies ``pixels_per_step`` pixel rows (default 1 → smooth
gradient; bump for a chunkier "square" look). A yellow 1-pixel row
through both strips marks the current frame.

If multiple chunkers are present, the two-strip panels are stacked
vertically (one pair per chunker, top-most chunker first).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as _cm
import numpy as np
import torch

from egomimic.eval.eval_video import EvalVideo

# Continuous greyscale: ``P(boundary) = 0`` → white (chunker quiet),
# ``P(boundary) = 1`` → black (chunker fires a boundary here). Matches
# the "mark a boundary as a dark tick on a timeline" intuition. Step-to-
# step instability shows up as speckled grey noise in the strip.
_CMAP = _cm.get_cmap("gray_r")


def _colors_for_probs(probs: np.ndarray) -> np.ndarray:
    """Vectorised RGB lookup for an array of probabilities in [0, 1].

    Returns ``(N, 3)`` uint8.
    """
    p = np.clip(probs.astype(np.float32), 0.0, 1.0)
    rgba = _CMAP(p)  # (N, 4) float in [0, 1]
    return (rgba[..., :3] * 255).astype(np.uint8)


class BoundaryStripEval(EvalVideo):
    """Per-frame centered window of chunker P(boundary), rendered as
    coloured squares.

    Args (yaml):
      window: number of timesteps in the centered window (default 256).
        Strip height = ``window * pixels_per_step``.
      strip_width: pixel width of the strip (default 24).
      pixels_per_step: how many pixel rows each timestep occupies. 1
        (default) gives a smooth gradient; higher values produce a
        chunkier "square" look.
      future_pad: how to fill the window's tail past the episode's last
        frame: ``"black"`` (default) or ``"clamp"`` (repeat last value).
      chunker_idx: which chunker's bpred to render. ``None`` = stack all
        chunkers' strips top-to-bottom.
    """

    def __init__(
        self,
        window: int = 256,
        strip_width: int = 24,
        pixels_per_step: int = 1,
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
        self.pixels_per_step = int(pixels_per_step)
        if future_pad not in {"black", "clamp"}:
            raise ValueError("future_pad must be 'black' or 'clamp'")
        self.future_pad = future_pad
        self.chunker_idx = chunker_idx

    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _run_forward_and_collect_bprobs(
        self, batch: Dict[int, Dict[str, Any]]
    ) -> Dict[int, List[tuple[torch.Tensor, torch.Tensor]]]:
        """Run ``policy.forward_packed`` for each emb and collect per-chunker
        ``(boundary_prob_packed, boundary_mask_packed)`` tuples.

        Returns ``{emb_id: [(prob_chunker_0, mask_chunker_0), …]}``.
        """
        algo = self.model
        out: Dict[int, List[tuple[torch.Tensor, torch.Tensor]]] = {}
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
            bprobs: List[tuple[torch.Tensor, torch.Tensor]] = []
            for entry in aux:
                bp = entry.get("bpred") if isinstance(entry, dict) else None
                if bp is None:
                    continue
                # bp.boundary_prob: packed (T_total, 2); column 1 = P(boundary).
                # bp.boundary_mask: packed (T_total,) bool — the chunker's
                # committed boundary decisions (argmax / STE output).
                prob = bp.boundary_prob[..., 1].detach().cpu()
                mask = bp.boundary_mask.detach().cpu().to(torch.bool)
                bprobs.append((prob, mask))
            out[emb_id] = bprobs
        return out

    def _render_strip_for_episode(
        self,
        bprob_packed: torch.Tensor,
        bmask_packed: torch.Tensor,
        T_ep: int,
    ) -> np.ndarray:
        """Render the per-frame centered window strip for one chunker on
        one episode. Returns ``(T_ep, strip_H, 2*strip_W + 1, 3)`` uint8.

        Two strips, side-by-side, separated by a 1-pixel black divider:

          A) **Gradient strip** — continuous greyscale (``gray_r``) of the
             soft ``P(boundary)`` in [0, 1]. White = quiet, black = chunker
             about to fire. Use this to read the chunker's *confidence*
             and to spot step-to-step jitter.
          B) **Discrete strip** — binary red/white from ``boundary_mask``.
             Red where the chunker actually fired a boundary (committed
             chunk divider), white otherwise. Use this to read the
             chunker's *decisions* — chunk sizes, regularity, etc.

        A yellow 1-pixel line through both strips marks the current frame
        at the centre of the window.
        """
        T_ep = int(T_ep)
        W = self.window
        pps = self.pixels_per_step
        sw = self.strip_width
        strip_H = W * pps

        bp = bprob_packed.numpy().astype(np.float32)
        bm = bmask_packed.numpy().astype(bool)
        if bp.shape[0] < T_ep:
            bp = np.concatenate([bp, np.zeros(T_ep - bp.shape[0], dtype=np.float32)])
            bm = np.concatenate([bm, np.zeros(T_ep - bm.shape[0], dtype=bool)])

        half = W // 2
        t_idx = np.arange(T_ep)[:, None]
        offsets = np.arange(W)[None, :] - half
        gidx = t_idx + offsets
        in_range = (gidx >= 0) & (gidx < bp.shape[0])
        if self.future_pad == "clamp":
            clamped = np.clip(gidx, 0, bp.shape[0] - 1)
            prob_grid = bp[clamped]
            mask_grid = bm[clamped]
        else:
            clamped = np.clip(gidx, 0, bp.shape[0] - 1)
            prob_grid = np.where(in_range, bp[clamped], 0.0)
            mask_grid = np.where(in_range, bm[clamped], False)

        # Gradient strip = greyscale P(boundary).
        grad_colors = _colors_for_probs(prob_grid.reshape(-1)).reshape(T_ep, W, 3)
        # Discrete strip = red where boundary_mask is True, white else.
        red = np.array([220, 30, 30], dtype=np.uint8)
        white = np.array([255, 255, 255], dtype=np.uint8)
        disc_colors = np.where(mask_grid[..., None], red, white).astype(np.uint8)

        # Stretch each row to ``pps`` pixel-rows → (T_ep, strip_H, 3).
        if pps > 1:
            grad_colors = np.repeat(grad_colors, pps, axis=1)
            disc_colors = np.repeat(disc_colors, pps, axis=1)

        # Tile each across the strip width → (T_ep, strip_H, sw, 3).
        grad_frames = np.broadcast_to(
            grad_colors[:, :, None, :], (T_ep, strip_H, sw, 3)
        ).copy()
        disc_frames = np.broadcast_to(
            disc_colors[:, :, None, :], (T_ep, strip_H, sw, 3)
        ).copy()

        # Yellow current-step marker (1px line at the centre of the window)
        # painted on both strips.
        ymid = half * pps + pps // 2
        yellow = (255, 220, 0)
        grad_frames[:, max(0, ymid - 1) : ymid + 1, :, :] = yellow
        disc_frames[:, max(0, ymid - 1) : ymid + 1, :, :] = yellow

        # 2-px black divider column between the two strips. (Width = even so
        # the eventual composite stays divisible-by-2 for x264 encoding.)
        divider = np.zeros((T_ep, strip_H, 2, 3), dtype=np.uint8)
        return np.concatenate([grad_frames, divider, disc_frames], axis=2)

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
                for prob, mask in chosen:
                    panel = self._render_strip_for_episode(prob[s:e], mask[s:e], T_ep)
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

            # Aggregate metrics per chunker.
            for ci, (prob, mask) in enumerate(bprobs):
                rate_prob = float((prob > 0.5).float().mean().item())
                rate_mask = float(mask.float().mean().item())
                metrics[f"Valid/emb{emb_id}_chunker{ci}_pboundary_gt05"] = torch.tensor(
                    rate_prob, device=self.trainer.lightning_module.device
                )
                metrics[f"Valid/emb{emb_id}_chunker{ci}_boundary_mask_rate"] = (
                    torch.tensor(rate_mask, device=self.trainer.lightning_module.device)
                )

        return metrics, images_dict
