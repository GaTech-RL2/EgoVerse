"""DFoT self-rollout (world-model) evaluator.

For the joint-obs+action DFoT variant (``ObsActionDFoTOuterStage``), the
model has been trained to denoise ``concat([state, action])`` jointly per
step. That makes the backbone both a policy and a world model — given
just one starting image, it can hallucinate a full trajectory of
(state, action) tokens with NO ground-truth states / actions from the
dataset and NO environment stepping.

This evaluator exercises that world-model capability:

  1. Pick the first ``max_videos`` episodes from each val batch.
  2. For each, take the **t=0 image only** (state from the dataset is
     NOT used; the bundle is initialized from pure noise).
  3. Encode the image through ``algo.cond_encoder`` and broadcast it
     across the rollout horizon T as the (frozen) AdaLN cond.
  4. Run chunk-mode DDIM sampling: ``algo._sample_chunk`` over
     T = ``rollout_steps`` tokens, producing a (1, T, bundle_dim)
     bundle.
  5. Split the bundle via ``algo.outer_stage.action_slice`` — pre-slice
     dims are the predicted obs trajectory; post-slice are the predicted
     actions.
  6. Render an mp4: at each frame t, show the starting image with the
     predicted agent path drawn so far (trail up to t) and a current-pos
     dot. Result is a "vision of the next T ticks the model believes
     should happen given this initial observation".

Notes:
  * The starting image is held constant across all T rollout ticks —
    this is the world-model test, not closed-loop env stepping.
  * Predicted obs and predicted action both come from the diffusion
    bundle. The state portion is the model's "imagined" future state.
  * If the algo's ``outer_stage.action_slice`` covers the full bundle
    (vanilla DFoT — no obs in bundle), this eval falls back to drawing
    only the predicted action trajectory.
"""

from __future__ import annotations

from typing import Dict

import cv2
import numpy as np
import torch

from egomimic.eval.eval_video import EvalVideo
from egomimic.rldb.embodiment.embodiment import get_embodiment_id


def _img_chw_to_uint8(img_chw: torch.Tensor) -> np.ndarray:
    """(C, H, W) float -> (C, H, W) uint8 ready for the drawing helper."""
    x = img_chw.detach().cpu().float().numpy()
    if x.max() <= 1.5:
        x = x * 255.0
    return x.clip(0, 255).astype(np.uint8)


def _draw_trajectory_frame(
    img_chw_uint8: np.ndarray,
    path_xy: np.ndarray,  # (t+1, 2) world-frame positions
    cur_idx: int,
    world_size: float,
    upscale_to: int = 512,
    trail_color: tuple = (180, 220, 255),
    head_color: tuple = (255, 255, 255),
    radius: int = 6,
) -> np.ndarray:
    """Draw a path overlay on the starting image. Returns (H, W, C) uint8."""
    img_hwc = np.transpose(img_chw_uint8, (1, 2, 0))
    img_hwc = cv2.resize(
        img_hwc, (upscale_to, upscale_to), interpolation=cv2.INTER_NEAREST
    )
    H, W, _ = img_hwc.shape
    out = np.ascontiguousarray(img_hwc.copy())

    sx = W / float(world_size)
    sy = H / float(world_size)

    def _to_pix(xy: np.ndarray) -> tuple | None:
        cx = int(round(float(xy[0]) * sx))
        cy = int(round(float(xy[1]) * sy))
        if 0 <= cx < W and 0 <= cy < H:
            return cx, cy
        return None

    pts = [_to_pix(p) for p in path_xy[: cur_idx + 1]]
    pts = [p for p in pts if p is not None]
    for i in range(1, len(pts)):
        cv2.line(out, pts[i - 1], pts[i], trail_color, thickness=2)
    if pts:
        cv2.circle(out, pts[-1], radius + 2, (0, 0, 0), thickness=-1)
        cv2.circle(out, pts[-1], radius, head_color, thickness=-1)
    return out


class DFoTSelfRolloutEval(EvalVideo):
    """World-model self-rollout for joint obs+action DFoT.

    Args:
        rollout_steps: number of bundle tokens to generate from the
            single starting image. The mp4 will have this many frames.
        world_size: physics world extent for pixel scaling (PushShapes
            uses 512). Pixels = world_xy * (image_dim / world_size).
        image_key: which obs key carries the env image (default
            ``"front_img_1"``).
        agent_state_slice: slice into the predicted state portion of the
            bundle that holds (agent_x, agent_y) for the trajectory
            overlay. Default ``slice(0, 2)`` (matches
            ``state_agent_obj`` layout where the first two dims are the
            agent xy).
        cfg_scale: classifier-free-guidance scale at sampling. 1.0 = no
            CFG. Mirrors ``DFoTValEval``.
        upscale_to: upscale the starting image to this side before
            drawing. Default 512.
        max_videos: per-pass cap on rendered episodes.
        limit_val_batches: per-pass cap on val batches.
        embodiment_name: which embodiment the packed val batches carry.
            For PushShapes that's ``"pushshapes_sim"``.
        viz_func / transform_lists: forwarded to ``EvalVideo`` base
            (kept for config-shape parity; the eval doesn't use them).
    """

    def __init__(
        self,
        rollout_steps: int = 64,
        world_size: float = 512.0,
        image_key: str = "front_img_1",
        agent_state_slice: slice | tuple = (0, 2),
        cfg_scale: float = 1.0,
        upscale_to: int = 512,
        max_videos: int = 2,
        limit_val_batches: int = 4,
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
        self.rollout_steps = int(rollout_steps)
        self.world_size = float(world_size)
        self.image_key = image_key
        if isinstance(agent_state_slice, slice):
            self.agent_state_slice = agent_state_slice
        else:
            start, stop = agent_state_slice
            self.agent_state_slice = slice(int(start), int(stop))
        self.cfg_scale = float(cfg_scale)
        self.upscale_to = int(upscale_to)
        self.embodiment_name = embodiment_name

    # ------------------------------------------------------------------
    # Sampler call. Reuses the algo's own _sample_chunk so warmup /
    # CFG / scheduler choices stay consistent with closed-loop infer.
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _rollout_from_image(
        self, algo, img_chw_norm: torch.Tensor, emb_id: int
    ) -> torch.Tensor:
        """Run T-step chunk-mode sampling from one image. Returns the
        full predicted bundle ``(T, bundle_dim)``."""
        device = img_chw_norm.device
        T = self.rollout_steps

        # Encode the single image into per-frame cond. CondEncoderModule's
        # image branch checks dim() == 4 to fire the (B, C, H, W) -> (B,
        # T, C, H, W) broadcast over T_action. Passing a 5-D tensor with
        # a leading T=1 would be interpreted as already-temporal and
        # would NOT broadcast — cond would come out (1, 1, d_cond) and
        # break the backbone's noise/cond concat at T=rollout_steps.
        img_bchw = img_chw_norm.unsqueeze(0)  # (1, C, H, W)
        obs_for_cond = {self.image_key: img_bchw}
        cond_dict = algo.cond_encoder.encode(obs_for_cond, T_action=T)
        cond = cond_dict.get(algo.cond_output_key)  # (1, T, d_cond) or None
        if cond is None:
            raise RuntimeError(
                "cond_encoder produced no fused_cond — DFoTSelfRolloutEval "
                "requires image conditioning."
            )

        bundle = algo._sample_chunk(B=1, T=T, cond=cond, device=device)
        return bundle.squeeze(0)  # (T, bundle_dim)

    # ------------------------------------------------------------------
    # Per-batch entry point. Picks first ``max_videos`` episodes; per
    # episode emits ``rollout_steps`` frames into the buffer.
    # ------------------------------------------------------------------

    def compute_metrics_and_viz(self, batch):
        algo = self.model
        metrics: Dict[str, torch.Tensor] = {}
        images_dict: Dict[int, np.ndarray] = {}

        emb_id = get_embodiment_id(self.embodiment_name)
        if emb_id not in batch:
            return metrics, images_dict
        _batch = batch[emb_id]
        ac_key = algo.resolved_ac_keys[emb_id]

        # Get the image tensor. Packed-mode obs is (T_total, C, H, W);
        # cu_seqlens tells us episode boundaries. We take the first
        # frame of each of the first ``max_videos`` episodes.
        if self.image_key not in _batch:
            return metrics, images_dict
        imgs = _batch[self.image_key]  # (T_total, C, H, W) packed
        if not _batch.get("_packed", False):
            # Padded fallback: (B, T, C, H, W). Pick (i, 0) for each i.
            B = imgs.shape[0]
            n = min(B, self.max_videos or B)
            starting = [imgs[i, 0] for i in range(n)]
        else:
            cu = _batch["cu_seqlens"].to(imgs.device, dtype=torch.long)
            B_eps = int(cu.shape[0] - 1)
            n = min(B_eps, self.max_videos or B_eps)
            starting = [imgs[int(cu[i].item())] for i in range(n)]

        # Run rollout per episode + render frames.
        action_slice = algo.outer_stage.action_slice
        # The state portion is everything BEFORE action_slice. For the
        # vanilla DFoT case (no obs in bundle), action_slice spans the
        # full trailing dim and the state slice below is empty; in that
        # fallback we draw the predicted action xy as the path.
        obs_slice = slice(0, action_slice.start or 0)
        has_obs_in_bundle = (obs_slice.stop > obs_slice.start)

        frames_for_emb = []
        for img_chw in starting:
            bundle_pred = self._rollout_from_image(algo, img_chw, emb_id)
            # Un-normalize: we need world-frame xy for the overlay.
            # Build a fake "actions" dict and run norm_stats.unnormalize
            # — this only works for the action portion. For the obs
            # portion, we'd need a per-obs-key unnormalize; for v1 we
            # use the action xy when obs slice is empty, otherwise the
            # state agent xy slice.
            if has_obs_in_bundle:
                # Take the configured agent-xy slice from the obs
                # portion of the bundle. State norm stats may differ
                # from action stats; we run unnormalize using the
                # bundle's leading dims under a synthetic "state_agent_obj"
                # key, then take only the agent xy.
                obs_pred = bundle_pred[..., obs_slice]  # (T, obs_dim)
                obs_keys = getattr(
                    algo.outer_stage, "bundle_obs_keys", ["state_agent_obj"]
                )
                obs_dims = getattr(
                    algo.outer_stage, "bundle_obs_dims", [obs_pred.shape[-1]]
                )
                # Split per-key obs prediction along last dim.
                splits = list(torch.split(obs_pred, obs_dims, dim=-1))
                obs_unnorm_dict = {k: v for k, v in zip(obs_keys, splits)}
                obs_unnorm = algo.norm_stats.unnormalize(obs_unnorm_dict, emb_id)
                agent_state = obs_unnorm[obs_keys[0]]  # (T, dim_0)
                xy = agent_state[..., self.agent_state_slice]  # (T, 2)
            else:
                # Fallback: vanilla DFoT — bundle is action only. Use
                # the predicted action as the trajectory.
                act_pred = bundle_pred[..., action_slice]  # (T, A)
                act_unnorm = algo.norm_stats.unnormalize(
                    {ac_key: act_pred}, emb_id
                )[ac_key]
                xy = act_unnorm[..., :2]

            xy_np = xy.detach().cpu().numpy()
            img_uint8 = _img_chw_to_uint8(img_chw)
            for t in range(self.rollout_steps):
                frame = _draw_trajectory_frame(
                    img_uint8,
                    path_xy=xy_np,
                    cur_idx=t,
                    world_size=self.world_size,
                    upscale_to=self.upscale_to,
                )
                frames_for_emb.append(frame)

        if frames_for_emb:
            images_dict[emb_id] = np.stack(frames_for_emb, axis=0)
        return metrics, images_dict
