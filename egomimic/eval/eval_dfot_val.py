"""DFoT val-data evaluator: teacher-forced action prediction with viz.

Runs two off-env prediction modes per val episode and renders a single mp4
overlaying both predictions on top of the GT environment frames:

  * **full_chunk**: one-shot denoising of the entire episode's action
    sequence. Single schedule with uniform per-token noise levels driven
    from 1.0 -> 0.0 over ``n_chunk_steps`` DDIM steps. Tests how well the
    model can recover a full trajectory from a single chunk of obs.
  * **ar (staircase)**: configurable causal-AR rolling-staircase denoising.
    The buffer holds ``chunk_size * step_size`` in-flight tokens at
    staircase noise levels; at each env-frame step the front-most token
    commits and a new fully-noisy token enters the back. The two knobs
    expose staircase geometry — ``ar_chunk_size`` = how many tokens share
    a rung ("width"), ``ar_step_size`` = how many denoise steps per rung
    ("height").

For each val episode the evaluator emits one mp4: the GT env image stream
(from ``front_img_1``) overlaid with three coloured action dots per frame
— GT (green), full_chunk pred (blue), AR pred (yellow). Per-mode per-
frame MSE is reported as the validation metric.

This evaluator does NOT touch a simulator (cf. ``HNetSimEval`` /
``HPTSimEval``). All obs come straight from the packed val batch — no
closed-loop drift between predicted action and next obs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch

from egomimic.algo.dfot.continuous_diffusion import ContinuousDiffusion
from egomimic.algo.dfot.discrete_diffusion import DiscreteDiffusion
from egomimic.algo.dfot.sampling import (
    sample,
    staircase_ar_schedule,
    vanilla_schedule,
)
from egomimic.eval.eval_video import EvalVideo


def _to_xy_pix(action_world: np.ndarray, img_hw: tuple) -> tuple:
    """Map a world-frame xy action into pixel coords for a (H, W) image.

    PushShapesEnv renders at ``image_size`` pixels per world-unit. Caller
    passes ``img_hw = (H, W)``; we assume world coords are in
    ``[0, image_size]`` and scale linearly. Out-of-bounds returns None.
    """
    h, w = img_hw
    x, y = float(action_world[0]), float(action_world[1])
    # PushShapes world frame uses image_size as the world extent; pull
    # the scale from the image dims (H==W==image_size in the configs we use).
    scale = max(h, w) / float(max(h, w))  # = 1.0 for square; safe default
    cx, cy = int(round(x * scale)), int(round(y * scale))
    if 0 <= cx < w and 0 <= cy < h:
        return cx, cy
    return None


def _draw_overlay(
    img_chw_uint8: np.ndarray,
    actions_world: Dict[str, np.ndarray],
    palette: Dict[str, Tuple[int, int, int]],
    radius: int = 4,
) -> np.ndarray:
    """Overlay one dot per mode on a (C, H, W) uint8 image. Returns (H, W, C)
    uint8 ready for video stacking."""
    if img_chw_uint8.dtype != np.uint8:
        img_chw_uint8 = (img_chw_uint8 * 255.0).clip(0, 255).astype(np.uint8)
    img_hwc = np.transpose(img_chw_uint8, (1, 2, 0))
    H, W, _ = img_hwc.shape
    out = np.ascontiguousarray(img_hwc.copy())
    for name, a in actions_world.items():
        pt = _to_xy_pix(a, (H, W))
        if pt is None:
            continue
        cv2.circle(out, pt, radius + 2, (0, 0, 0), thickness=-1)
        cv2.circle(out, pt, radius, palette[name], thickness=-1)
    return out


# ---------------------------------------------------------------------- #
# Eval class
# ---------------------------------------------------------------------- #


class DFoTValEval(EvalVideo):
    """Teacher-forced val-data evaluator for DFoT.

    Args:
        do_full_chunk: run the one-shot full-episode denoising mode.
        do_ar: run the causal-AR staircase mode.
        n_chunk_steps: number of DDIM steps for the full-chunk mode.
        ar_chunk_size: staircase width (tokens per rung). 1 = vanilla
            causal-AR.
        ar_step_size: staircase height (denoising steps per rung).
        limit_val_batches: cap on val batches per validation pass.
        max_videos: cap on number of episodes rendered per val pass.
        coverage_threshold: unused here (kept for config-shape parity with
            sim-eval configs); accepted and ignored.
        env_kwargs: unused (no env); accepted for config parity.
        embodiment_name: which embodiment to expect in the packed batch
            (default ``"pushshapes_sim"``).
        viz_func / transform_lists: forwarded to ``EvalVideo`` base.
    """

    def __init__(
        self,
        do_full_chunk: bool = True,
        do_ar: bool = True,
        n_chunk_steps: int = 50,
        ar_chunk_size: int = 1,
        ar_step_size: int = 1,
        limit_val_batches: int = 4,
        max_videos: int = 2,
        coverage_threshold: float = 0.7,
        env_kwargs: dict | None = None,
        embodiment_name: str = "pushshapes_sim",
        viz_func=None,
        transform_lists=None,
    ):
        super().__init__(
            limit_val_batches=limit_val_batches,
            viz_func=viz_func,
            transform_lists=transform_lists,
            max_videos=max_videos,
        )
        self.do_full_chunk = bool(do_full_chunk)
        self.do_ar = bool(do_ar)
        self.n_chunk_steps = int(n_chunk_steps)
        self.ar_chunk_size = int(ar_chunk_size)
        self.ar_step_size = int(ar_step_size)
        self.embodiment_name = embodiment_name
        # Kept for config-shape parity with eval_hnet_sim; not used.
        _ = coverage_threshold, env_kwargs

    # ---- env-image extraction ---- #

    def _img_uint8(self, img_chw_norm: torch.Tensor) -> np.ndarray:
        """Convert one (C, H, W) image tensor (any float range) to uint8
        (C, H, W) numpy. Heuristic: if max <= 1.5 we treat as [0,1] floats."""
        x = img_chw_norm.detach().cpu().float().numpy()
        if x.max() <= 1.5:
            x = x * 255.0
        return x.clip(0, 255).astype(np.uint8)

    # ---- per-episode prediction passes ---- #

    @torch.no_grad()
    def _full_chunk_pred(
        self,
        algo,
        actions_norm: torch.Tensor,  # (T, A)
        cond_per_frame: torch.Tensor | None,  # (T, d_cond) or None
        emb_id: int,
    ) -> torch.Tensor:
        """One-shot DDIM denoise over the full episode. Returns predicted
        actions in WORLD frame (un-normalized), shape (T, A)."""
        T, A = actions_norm.shape
        device = actions_norm.device
        diff = algo.diffusion
        backbone = algo.nets["backbone"]
        discrete_ts = (
            int(diff.timesteps) if isinstance(diff, DiscreteDiffusion) else None
        )
        schedule = vanilla_schedule(
            n_steps=self.n_chunk_steps, T=T, discrete_timesteps=discrete_ts
        ).to(device)
        ec = cond_per_frame.unsqueeze(0) if cond_per_frame is not None else None
        pred_norm = sample(
            diff,
            backbone,
            schedule_matrix=schedule,
            action_dim=A,
            batch_size=1,
            external_cond=ec,
            device=device,
        ).squeeze(0)  # (T, A)
        # Un-normalize.
        ac_key = algo.resolved_ac_keys[emb_id]
        pred_world = algo.norm_stats.unnormalize({ac_key: pred_norm}, emb_id)[ac_key]
        return pred_world  # (T, A)

    @torch.no_grad()
    def _ar_pred(
        self,
        algo,
        cond_per_frame: torch.Tensor,  # (T, d_cond)
        emb_id: int,
        T: int,
    ) -> torch.Tensor:
        """Single-call staircase causal-AR denoising over the whole episode.

        The schedule_matrix is the rolling-staircase geometry produced by
        ``staircase_ar_schedule(T, chunk_size=ar_chunk_size,
        step_size=ar_step_size)``. ``sample()`` then walks all
        ``ceil(T / ar_chunk_size) * ar_step_size`` denoising steps, with
        every token at every step taking its noise level from the matrix.
        Earlier tokens reach 0 first; later tokens still at high noise
        — the canonical DFoT-paper causal-AR pattern.
        """
        device = cond_per_frame.device
        diff = algo.diffusion
        backbone = algo.nets["backbone"]
        discrete_ts = (
            int(diff.timesteps) if isinstance(diff, DiscreteDiffusion) else None
        )
        ac_key = algo.resolved_ac_keys[emb_id]
        A = int(algo.action_dim)
        schedule = staircase_ar_schedule(
            T=T,
            chunk_size=int(self.ar_chunk_size),
            step_size=int(self.ar_step_size),
            discrete_timesteps=discrete_ts,
        ).to(device)
        ec = cond_per_frame.unsqueeze(0)  # (1, T, d_cond)
        pred_norm = sample(
            diff,
            backbone,
            schedule_matrix=schedule,
            action_dim=A,
            batch_size=1,
            external_cond=ec,
            device=device,
        ).squeeze(0)  # (T, A)
        pred_world = algo.norm_stats.unnormalize({ac_key: pred_norm}, emb_id)[ac_key]
        return pred_world

    # ---- main eval entrypoint ---- #

    def compute_metrics_and_viz(self, batch):
        device = self.trainer.lightning_module.device
        algo = self.model
        metrics: Dict[str, torch.Tensor] = {}
        images_dict: Dict[int, np.ndarray] = {}

        palette = {
            "gt": (0, 255, 0),
            "chunk": (0, 128, 255),
            "ar": (0, 255, 255),
        }

        for emb_id, _batch in batch.items():
            if not _batch.get("_packed", False):
                # Unsupported in v1; skip silently.
                continue
            cu = _batch["cu_seqlens"]
            if not torch.is_tensor(cu):
                continue
            B = int(cu.shape[0]) - 1
            if B <= 0:
                continue
            B_render = (
                min(B, self.max_videos) if self.max_videos is not None else B
            )

            ac_key = algo.resolved_ac_keys[emb_id]
            actions_packed = _batch[ac_key]  # (T_total, A)
            obs = algo._build_obs(_batch, emb_id)
            cond_packed = algo._encode_cond_packed(obs)  # (T_total, d_cond) or None
            img_key = next(iter(algo.camera_keys[emb_id]), None)
            imgs_packed = _batch.get(img_key) if img_key is not None else None

            ep_chunk_mse: List[float] = []
            ep_ar_mse: List[float] = []
            ep_frames: List[np.ndarray] = []

            for b in range(B_render):
                s = int(cu[b].item())
                e = int(cu[b + 1].item())
                T = e - s
                if T <= 1:
                    continue

                # Slice this episode's tensors.
                acts_norm_ep = actions_packed[s:e]  # (T, A)
                cond_ep = cond_packed[s:e] if cond_packed is not None else None
                imgs_ep = imgs_packed[s:e] if imgs_packed is not None else None

                # GT in world frame.
                gt_world = algo.norm_stats.unnormalize({ac_key: acts_norm_ep}, emb_id)[
                    ac_key
                ]  # (T, A)

                chunk_world = None
                ar_world = None

                if self.do_full_chunk:
                    chunk_world = self._full_chunk_pred(
                        algo, acts_norm_ep, cond_ep, emb_id
                    )
                    ep_chunk_mse.append(
                        float(torch.mean((chunk_world - gt_world) ** 2).item())
                    )

                if self.do_ar:
                    if cond_ep is None:
                        ar_world = None
                    else:
                        ar_world = self._ar_pred(algo, cond_ep, emb_id, T=T)
                        ep_ar_mse.append(
                            float(torch.mean((ar_world - gt_world) ** 2).item())
                        )

                # Build viz video for this episode.
                if imgs_ep is not None:
                    frames_ep: List[np.ndarray] = []
                    for t in range(T):
                        img_chw = self._img_uint8(imgs_ep[t])
                        overlay_actions = {"gt": gt_world[t].detach().cpu().numpy()}
                        if chunk_world is not None:
                            overlay_actions["chunk"] = (
                                chunk_world[t].detach().cpu().numpy()
                            )
                        if ar_world is not None:
                            overlay_actions["ar"] = ar_world[t].detach().cpu().numpy()
                        out = _draw_overlay(img_chw, overlay_actions, palette)
                        frames_ep.append(np.ascontiguousarray(out))
                    ep_frames.extend(frames_ep)
                    if b < B_render - 1 and frames_ep:
                        H, W, _ = frames_ep[0].shape
                        sep = np.zeros((5, W, 3), dtype=np.uint8)
                        # 5-row black separator between consecutive episodes.
                        ep_frames.append(np.broadcast_to(sep, (5, W, 3)).copy())

            # Aggregate metrics.
            if ep_chunk_mse:
                metrics[f"emb{emb_id}_chunk_action_mse"] = torch.tensor(
                    float(np.mean(ep_chunk_mse)), device=device
                )
            if ep_ar_mse:
                metrics[f"emb{emb_id}_ar_action_mse"] = torch.tensor(
                    float(np.mean(ep_ar_mse)), device=device
                )
            if ep_frames:
                images_dict[emb_id] = np.stack(ep_frames, axis=0)

        return metrics, images_dict
