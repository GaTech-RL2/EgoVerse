"""
BoundaryStripEval — renders the chunker's boundary probability over time
as a thin vertical strip.

For each validation episode, we run the algo's teacher-forced
``forward_packed`` and pull ``boundary_prob`` out of ``ctx.aux`` (each
chunker stage emits a ``bpred`` with shape ``(T_total, 2)`` where column
1 is ``P(boundary)`` per token). Then for each frame ``t`` of the video,
we render a centered ``window`` of the surrounding ``[t - W/2, t + W/2]``
boundary probs as a vertical gradient column — bright when the chunker
would fire a boundary, dark when it wouldn't.

By default each timestep occupies ONE pixel row so consecutive probs
blend into a smooth gradient (good for spotting instability — sharp
banding between near-by frames indicates jumpy chunker decisions). Bump
``pixels_per_step`` for a chunkier "square" look.

If multiple chunkers are present, the strip column groups them
vertically (one sub-strip per chunker, top-most first).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as _cm
import numpy as np
import torch

from egomimic.eval.eval_video import EvalVideo

# Continuous colormap: matplotlib's ``magma`` runs black → purple → orange
# → near-white as the value rises. Smooth, perceptually uniform, and
# distinguishes neighbouring values well so you can see step-to-step
# instability as visible banding/noise in the strip.
_CMAP = _cm.get_cmap("magma")


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
        one episode. Returns ``(T_ep, strip_H, strip_W, 3)`` uint8.

        Vectorised: builds the full (T_ep, window) probability matrix in
        one shot, looks up colors, then stretches each row to
        ``pixels_per_step`` and tiles horizontally to ``strip_width``.
        """
        T_ep = int(T_ep)
        W = self.window
        pps = self.pixels_per_step
        sw = self.strip_width
        strip_H = W * pps

        bp = bprob_packed.numpy().astype(np.float32)
        if bp.shape[0] < T_ep:
            bp = np.concatenate([bp, np.zeros(T_ep - bp.shape[0], dtype=np.float32)])

        # Build (T_ep, W) matrix of probabilities for the centered window
        # around each ``t``. ``idx_grid[t, i] = t - W//2 + i``.
        half = W // 2
        t_idx = np.arange(T_ep)[:, None]  # (T_ep, 1)
        offsets = np.arange(W)[None, :] - half  # (1, W)
        gidx = t_idx + offsets  # (T_ep, W)
        in_range = (gidx >= 0) & (gidx < bp.shape[0])
        if self.future_pad == "clamp":
            clamped = np.clip(gidx, 0, bp.shape[0] - 1)
            prob_grid = bp[clamped]
        else:
            prob_grid = np.where(in_range, bp[np.clip(gidx, 0, bp.shape[0] - 1)], 0.0)
            # ``bp[np.clip(...)]`` produces a real lookup; we only keep it
            # where in_range was True. The else branch ("black" pad) is
            # already (0, 0, 0) in the colormap at p=0, but we explicitly
            # zero so the actual probability data isn't seen as in-range.

        # (T_ep, W, 3) uint8
        colors = _colors_for_probs(prob_grid.reshape(-1)).reshape(T_ep, W, 3)
        # Stretch each row to ``pps`` pixel-rows → (T_ep, W*pps, 3)
        if pps > 1:
            colors = np.repeat(colors, pps, axis=1)
        # Tile across width → (T_ep, strip_H, strip_W, 3)
        frames = np.broadcast_to(colors[:, :, None, :], (T_ep, strip_H, sw, 3)).copy()
        # Yellow current-step marker (1px line at the centre of the window).
        ymid = half * pps + pps // 2
        frames[:, max(0, ymid - 1) : ymid + 1, :, :] = (255, 220, 0)
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
