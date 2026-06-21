"""
KeyframeProjectionEval — for each chunker stage, project its committed
chunk-start tokens down to FRAME level and render the actual image
keyframes that stage selects.

A frame t is a *stage-s keyframe* iff t starts a token at EVERY level up
to and including s (crisp composition — same math as BoundaryStripEval's
"Top→frm" row, kept for every stage). Stage 0 = the outer (frame-level)
chunker; the deepest stage is the model's high-level segmentation.

Visualization per episode: one filmstrip row per stage — the obs image at
each of that stage's keyframes, labeled with the true env frame number
(token_index * obs_stride). Read the deepest row to judge whether the
learned segmentation is SEMANTIC (contact / direction change / goal
approach) or arbitrary.

Emitted metrics per stage: ``Valid/emb{e}_stage{s}_keyframe_rate`` (frame-
level) and ``Valid/emb{e}_stage{s}_frames_per_keyframe`` (in env frames,
i.e. rate^-1 * obs_stride) — compare against the per-stage compression
targets multiplied out.

Yaml usage (inside an EvalVideoList, like BoundaryStripEval):
    - _target_: egomimic.eval.probes.eval_keyframes.KeyframeProjectionEval
      max_videos: 4
      max_tiles: 48
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from egomimic.eval.core.eval_video import EvalVideo


def per_stage_keyframe_masks(bprobs, T_total):
    """``bprobs``: list of (prob, mask) in forward_packed order (innermost
    FIRST). Returns frame-level bool masks in DISPLAY order (outer stage
    first): masks[s][t] True iff token t starts a stage-s chunk, composed
    through all shallower stages."""
    outer_idx = None
    for i, (_p, m) in enumerate(bprobs):
        if m.shape[0] == T_total:
            outer_idx = i
            break
    if outer_idx is None:
        return []
    crisp = bprobs[outer_idx][1].cpu().numpy().astype(bool)
    masks = [crisp.copy()]
    cur_map = np.maximum(np.cumsum(crisp) - 1, 0)
    for inner_i in range(outer_idx - 1, -1, -1):
        m = bprobs[inner_i][1].cpu().numpy().astype(bool)
        if m.shape[0] == 0:
            break
        idx = np.clip(cur_map, 0, m.shape[0] - 1)
        crisp = crisp & m[idx]
        masks.append(crisp.copy())
        local_map = np.maximum(np.cumsum(m) - 1, 0)
        cur_map = local_map[idx]
    return masks


def _filmstrip(imgs, masks, obs_stride, max_tiles):
    """imgs: (T, 3, H, W) float [0,1] (token granularity); masks: per-stage
    bool masks for ONE episode. Returns (H_canvas, W_canvas, 3) uint8."""
    from PIL import Image, ImageDraw, ImageFont
    T, _, H, W = imgs.shape
    pad, label_h, title_w = 4, 14, 118
    font = None
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11
        )
    except IOError:
        font = ImageFont.load_default()
    rows = []
    for s, mask in enumerate(masks):
        idxs = np.where(mask[:T])[0][:max_tiles]
        n = max(len(idxs), 1)
        row = Image.new(
            "RGB", (n * (W + pad) + pad, H + label_h + pad), (245, 245, 245)
        )
        d = ImageDraw.Draw(row)
        for j, t in enumerate(idxs):
            arr = (np.clip(imgs[t].transpose(1, 2, 0), 0, 1) * 255).astype(
                np.uint8
            )
            row.paste(Image.fromarray(arr), (pad + j * (W + pad), label_h))
            d.text(
                (pad + j * (W + pad), 1),
                f"f{int(t) * obs_stride}",
                fill=(10, 10, 10),
                font=font,
            )
        capped = mask[:T].sum() > max_tiles
        rows.append(
            (f"stage {s} ({len(idxs)} kf{', capped' if capped else ''})", row)
        )
    width = max(r.size[0] for _t, r in rows) + title_w
    height = sum(r.size[1] + 18 for _t, r in rows) + 10
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    y = 5
    for title, row in rows:
        d.text((4, y + 2), title, fill=(0, 0, 0), font=font)
        canvas.paste(row, (title_w, y))
        y += row.size[1] + 18
    return np.array(canvas, dtype=np.uint8)


class KeyframeProjectionEval(EvalVideo):
    """Per-stage keyframe projection filmstrips + rate metrics.

    Args (yaml):
      max_tiles: cap on keyframe tiles per stage row (default 48).
      img_key: obs image key in the packed batch (default front_img_1).
      hold_frames: each episode's filmstrip is emitted as this many
        identical video frames so the standard video writer renders a
        readable still (default 40).
    """

    def __init__(
        self,
        max_tiles: int = 48,
        img_key: str = "front_img_1",
        hold_frames: int = 40,
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
        self.max_tiles = int(max_tiles)
        self.img_key = str(img_key)
        self.hold_frames = int(hold_frames)
        # Filled on each compute_metrics_and_viz call: {emb_id: [canvas]}
        # so offline runners (scripts/keyframe_projection.py) can save PNGs.
        self.last_canvases: Dict[int, List[np.ndarray]] = {}

    @torch.no_grad()
    def compute_metrics_and_viz(
        self, batch: Dict[int, Dict[str, Any]]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[int, np.ndarray]]:
        algo = self.model
        device = self.trainer.lightning_module.device
        metrics: Dict[str, torch.Tensor] = {}
        images_dict: Dict[int, np.ndarray] = {}
        self.last_canvases = {}

        for emb_id, _b in batch.items():
            if not _b.get("_packed", False):
                continue
            ac_key = algo.resolved_ac_keys[emb_id]
            obs = algo._build_obs(_b, emb_id)
            cu = _b["cu_seqlens"]
            _pred, aux = algo.policy.forward_packed(
                _b[ac_key],
                obs,
                cu,
                int(_b["max_seq_len"]),
                embodiment_id=getattr(algo, "domain_by_id", {}).get(emb_id),
            )
            bprobs = []
            for entry in aux:
                bp = entry.get("bpred") if isinstance(entry, dict) else None
                if bp is not None:
                    bprobs.append(
                        (
                            bp.boundary_prob[..., 1].detach().cpu(),
                            bp.boundary_mask.detach().cpu().bool(),
                        )
                    )
            T_total = int(cu[-1].item())
            masks = per_stage_keyframe_masks(bprobs, T_total)
            if not masks:
                continue
            obs_stride = int(getattr(algo, "obs_stride", 1))
            imgs_packed = _b[self.img_key].detach().cpu().numpy()

            B = int(_b["seq_lens"].shape[0])
            B_render = (
                min(B, self.max_videos) if self.max_videos is not None else B
            )
            canvases = []
            for b in range(B_render):
                s_, e_ = int(cu[b].item()), int(cu[b + 1].item())
                canvases.append(
                    _filmstrip(
                        imgs_packed[s_:e_],
                        [m[s_:e_] for m in masks],
                        obs_stride,
                        self.max_tiles,
                    )
                )
            self.last_canvases[emb_id] = canvases
            # Pad canvases to a common size, hold each as a still segment.
            Hm = max(c.shape[0] for c in canvases)
            Wm = max(c.shape[1] for c in canvases)
            Hm += Hm % 2
            Wm += Wm % 2
            frames = []
            for c in canvases:
                padded = np.full((Hm, Wm, 3), 255, dtype=np.uint8)
                padded[: c.shape[0], : c.shape[1]] = c
                frames.extend([padded] * self.hold_frames)
            images_dict[emb_id] = np.stack(frames, axis=0)

            for s, m in enumerate(masks):
                rate = float(m.mean())
                metrics[f"Valid/emb{emb_id}_stage{s}_keyframe_rate"] = (
                    torch.tensor(rate, device=device)
                )
                metrics[f"Valid/emb{emb_id}_stage{s}_frames_per_keyframe"] = (
                    torch.tensor(
                        obs_stride / max(rate, 1e-9), device=device
                    )
                )
        return metrics, images_dict
