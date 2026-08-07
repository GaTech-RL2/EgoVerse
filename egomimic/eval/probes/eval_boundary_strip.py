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

from egomimic.eval.core.eval_video import EvalVideo

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



def _compose_bprobs_to_frame_level(bprobs, T_total):
    """Upsample inner chunker (prob, mask) to frame resolution. bprobs is in
    forward_packed order (innermost first). Walks outer -> inner composing
    cumsums so each level's per-token signal is replicated across the frames
    that map to it."""
    import numpy as _np
    import torch as _torch
    if not bprobs:
        return []
    out = list(bprobs)
    outer_idx = None
    for i, (_p, _m) in enumerate(bprobs):
        if _m.shape[0] == T_total:
            outer_idx = i
            break
    if outer_idx is None:
        return out
    outer_mask = bprobs[outer_idx][1].numpy().astype(bool)
    cur_level_map = _np.maximum(_np.cumsum(outer_mask) - 1, 0)
    for inner_i in range(outer_idx - 1, -1, -1):
        prob_i, mask_i = bprobs[inner_i]
        prob_np = prob_i.numpy()
        mask_np = mask_i.numpy().astype(bool)
        if mask_np.shape[0] == 0:
            continue
        idx = _np.clip(cur_level_map, 0, mask_np.shape[0] - 1)
        out[inner_i] = (
            _torch.from_numpy(prob_np[idx]),
            _torch.from_numpy(mask_np[idx]),
        )
        local_map = _np.maximum(_np.cumsum(mask_np) - 1, 0)
        cur_level_map = local_map[idx]
    return out


def _compose_top_boundary_frames(bprobs, T_total):
    """Crisp frame-level segmentation by the TOP (innermost) chunker.

    Unlike ``_compose_bprobs_to_frame_level`` (which REPLICATES each inner
    token's signal across its whole frame span — good for confidence reading,
    bad for seeing where high-level chunks actually start), this returns a
    frame-level ``(prob, mask)`` where ``mask[t]`` is True iff frame ``t``
    STARTS a top-level chunk: t starts a token at every stage going up AND
    the top stage fires at that token. ``prob[t]`` carries the top stage's
    P(boundary) on chunk-start frames, 0 elsewhere.

    Returns None when there is only one (frame-level) stage — nothing to
    compose.
    """
    import numpy as _np
    import torch as _torch
    if not bprobs:
        return None
    outer_idx = None
    for i, (_p, m) in enumerate(bprobs):
        if m.shape[0] == T_total:
            outer_idx = i
            break
    if outer_idx is None or outer_idx == 0:
        return None
    crisp = bprobs[outer_idx][1].numpy().astype(bool)
    cur_map = _np.maximum(_np.cumsum(crisp) - 1, 0)
    top_prob = None
    for inner_i in range(outer_idx - 1, -1, -1):
        prob_i, mask_i = bprobs[inner_i]
        m = mask_i.numpy().astype(bool)
        if m.shape[0] == 0:
            return None
        idx = _np.clip(cur_map, 0, m.shape[0] - 1)
        crisp = crisp & m[idx]
        local_map = _np.maximum(_np.cumsum(m) - 1, 0)
        cur_map = local_map[idx]
        top_prob = prob_i.numpy()[idx]
    prob_out = _np.where(crisp, _np.clip(top_prob, 0.0, 1.0), 0.0).astype(
        _np.float32
    )
    return (_torch.from_numpy(prob_out), _torch.from_numpy(crisp))


def _render_stage_label(stage_idx, h, w):
    """White label panel with 'Stage k' text centred (or a custom string if
    ``stage_idx`` is a str). Returns (h, w, 3) uint8."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return np.full((h, w, 3), 200, dtype=np.uint8)
    img = Image.new("RGB", (w, h), color=(245, 245, 245))
    draw = ImageDraw.Draw(img)
    font = None
    for fp in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial.ttf",
    ]:
        try:
            font = ImageFont.truetype(fp, 14)
            break
        except IOError:
            continue
    if font is None:
        font = ImageFont.load_default()
    text = stage_idx if isinstance(stage_idx, str) else f"Stage {stage_idx}"
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((w - tw) // 2, (h - th) // 2), text, fill=(20, 20, 20), font=font)
    return np.array(img, dtype=np.uint8)


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
      composed_strip: when True (default) and the model has >1 chunker
        stage, append a "Top→frame" row: crisp red dividers at exactly the
        frames where a TOP-level chunk starts (composed through all stages)
        — the highest-level segmentation projected to frame resolution.
    """

    def __init__(
        self,
        window: int = 256,
        strip_width: int = 24,
        pixels_per_step: int = 1,
        future_pad: str = "black",
        chunker_idx: int | None = None,
        composed_strip: bool = True,
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
        self.window = int(window)
        self.strip_width = int(strip_width)
        self.pixels_per_step = int(pixels_per_step)
        if future_pad not in {"black", "clamp"}:
            raise ValueError("future_pad must be 'black' or 'clamp'")
        self.future_pad = future_pad
        self.chunker_idx = chunker_idx
        self.composed_strip = bool(composed_strip)

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
            policy = algo.policy
            domain_name = getattr(algo, "domain_by_id", {}).get(emb_id)
            _pred, aux = policy.forward_packed(
                actions,
                obs,
                cu,
                max_seqlen,
                embodiment_id=domain_name,
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
            cu = _batch["cu_seqlens"]
            seq_lens = _batch["seq_lens"]
            B = int(seq_lens.shape[0])
            T_total = int(cu[-1].item())
            # Crisp top-level chunk starts at frame level (from the RAW
            # per-stage masks, before the replicating upsample below).
            composed = (
                _compose_top_boundary_frames(bprobs, T_total)
                if self.composed_strip and self.chunker_idx is None
                else None
            )
            # Upsample inner chunkers (prob, mask) to frame resolution.
            bprobs = _compose_bprobs_to_frame_level(bprobs, T_total)
            # Display order: OUTER chunker (frame-level natively) on top as
            # "Stage 0", inner chunkers below. aux list is innermost-first,
            # so reverse for the display. Each display item: (label, prob, mask).
            if self.chunker_idx is None:
                display_chunkers = [
                    (ci, p, m) for ci, (p, m) in enumerate(reversed(bprobs))
                ]
            else:
                _p, _m = bprobs[self.chunker_idx]
                display_chunkers = [(self.chunker_idx, _p, _m)]
            if composed is not None:
                display_chunkers.append(("Top→frm", composed[0], composed[1]))

            B_render = min(B, self.max_videos) if self.max_videos is not None else B
            ep_widths = []
            for b in range(B_render):
                ep_widths.append(int(cu[b + 1].item()) - int(cu[b].item()))
            ep_sep_w = 4
            label_w = 64
            sw = self.strip_width
            red = np.array([220, 30, 30], dtype=np.uint8)
            white = np.array([255, 255, 255], dtype=np.uint8)
            row_h = 2 * sw + 2

            yellow = (255, 220, 0)
            per_ep_imgs = []  # one (T_ep_b, H, W_b, 3) per episode
            for b in range(B_render):
                s_ = int(cu[b].item())
                e_ = int(cu[b + 1].item())
                T_ep_b = e_ - s_
                # Build per-chunker rows for THIS episode only.
                chunker_rows = []
                for ci, prob, mask in display_chunkers:
                    bp_arr = prob[s_:e_].numpy().astype(np.float32)
                    bm_arr = mask[s_:e_].numpy().astype(bool)
                    if bp_arr.shape[0] < T_ep_b:
                        bp_arr = np.concatenate([
                            bp_arr,
                            np.zeros(T_ep_b - bp_arr.shape[0], dtype=np.float32),
                        ])
                        bm_arr = np.concatenate([
                            bm_arr,
                            np.zeros(T_ep_b - bm_arr.shape[0], dtype=bool),
                        ])
                    grad = _colors_for_probs(bp_arr).reshape(T_ep_b, 3)
                    grad_row = np.broadcast_to(
                        grad[None, :, :], (sw, T_ep_b, 3)
                    ).copy()
                    disc = np.where(bm_arr[:, None], red, white).astype(np.uint8)
                    disc_row = np.broadcast_to(
                        disc[None, :, :], (sw, T_ep_b, 3)
                    ).copy()
                    div = np.zeros((2, T_ep_b, 3), dtype=np.uint8)
                    seg = np.concatenate([grad_row, div, disc_row], axis=0)
                    label = _render_stage_label(ci, row_h, label_w)
                    chunker_rows.append(np.concatenate([label, seg], axis=1))
                base_ep = np.concatenate(chunker_rows, axis=0)  # (H, label_w+T_ep_b, 3)
                base_H_ep, base_W_ep, _ = base_ep.shape
                panel_ep = np.broadcast_to(
                    base_ep[None, :, :, :], (T_ep_b, base_H_ep, base_W_ep, 3)
                ).copy()
                # Stamp marker per frame within this episode.
                for t_in_ep in range(T_ep_b):
                    col = label_w + t_in_ep
                    panel_ep[t_in_ep, :, max(0, col - 1) : col + 1, :] = yellow
                per_ep_imgs.append(panel_ep)

            # Episodes can have different widths; pad to max width before
            # concatenating along time (axis=0).
            target_W = max(p.shape[2] for p in per_ep_imgs)
            target_H = per_ep_imgs[0].shape[1]
            padded = []
            for p_ in per_ep_imgs:
                if p_.shape[2] < target_W:
                    pad = np.zeros(
                        (p_.shape[0], target_H, target_W - p_.shape[2], 3),
                        dtype=p_.dtype,
                    )
                    p_ = np.concatenate([p_, pad], axis=2)
                padded.append(p_)
            images_dict[emb_id] = np.concatenate(padded, axis=0)

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
            # Composed top-level metrics: chunks-per-frame and its inverse,
            # the average TOP-level chunk length in FRAMES — the single most
            # interpretable router-health number (compare to the configured
            # per-stage targets multiplied out, e.g. 2x2 looks x 8 = 32f).
            if composed is not None:
                top_rate = float(composed[1].float().mean().item())
                metrics[f"Valid/emb{emb_id}_top_chunk_rate"] = torch.tensor(
                    top_rate, device=self.trainer.lightning_module.device
                )
                metrics[f"Valid/emb{emb_id}_top_chunk_len_frames"] = torch.tensor(
                    (1.0 / top_rate) if top_rate > 0 else float(T_total),
                    device=self.trainer.lightning_module.device,
                )

        return metrics, images_dict
