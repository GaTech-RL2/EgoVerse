"""``ImageSpatialDFoTOuterStage`` — DFoT world-model with spatial latent.

Plan-A variant of the obs+action+image diffusion-forcing pipeline:
**diffuses ONLY the image latent** (kept in its full ``(C, H, W)``
spatial form, NOT flattened). State + action enter the model as
``external_cond`` via a small MLP, broadcast across patches by the
``DFoTSpatialBackbone``'s AdaLN.

Sacrifice vs ``ObsActionImageDFoTOuterStage`` (the flat-bundle variant):
this stage CANNOT produce predicted actions — action prediction is
deferred to a follow-up "action-as-extra-token" hybrid. The trade-off
is the spatial inductive bias that ``DiT3D``-style architectures use
to actually make image prediction work (see Optimize.md / the plan-A
write-up for the architectural reasoning).

Bundle shape: ``(latent_channels, latent_h, latent_w)`` per step. The
algo's sampler / AR code reads ``outer_stage.bundle_shape`` (a tuple)
to allocate tensors; vanilla 1D-bundle stages still use
``outer_stage.bundle_dim`` (an int). Both code paths are supported.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import List, Optional

import torch
import torch.nn as nn

from egomimic.algo.diffusion.outer_stages.outer_stage import DFoTOuterStage
from egomimic.algo.diffusion.vae_algo import load_pretrained_vae


class ImageSpatialDFoTOuterStage(DFoTOuterStage):
    """Spatial-image-only DFoT outer stage.

    Args:
        cond_encoder: kept on the class for ABI parity, but typically
            unused (pass an empty CondEncoderModule with no obs_specs
            and no img_encoders). state + action conditioning is done
            internally by ``_state_action_to_cond``.
        backbone: a ``DFoTSpatialBackbone``. Its ``cond_dim`` MUST equal
            ``state_action_proj_dim`` (this stage's MLP output width).
        diffusion: continuous / discrete diffusion module.
        bundle_obs_keys: list of obs keys to include in the external_cond
            (typically ``["state_agent_obj"]``).
        bundle_obs_dims: per-key feature widths matching ``bundle_obs_keys``.
        vae_checkpoint_path: path to the frozen ImageVAE Lightning ckpt.
        image_key: which obs key carries the image.
        action_dim: ACTION feature width (for the MLP input).
        state_action_proj_dim: width of the projected ``external_cond``
            fed to the backbone. Must match backbone's ``cond_dim``.
        cond_output_key: ABI parity; unused.
    """

    def __init__(
        self,
        action_dim: int,
        cond_encoder,
        backbone,
        diffusion,
        bundle_obs_keys: List[str],
        bundle_obs_dims: List[int],
        vae_checkpoint_path: str,
        image_key: str = "front_img_1",
        state_action_proj_dim: int = 128,
        cond_output_key: str = "fused_cond",
    ):
        # We don't call the ObsActionDFoTOuterStage constructor's
        # bundle-width sanity check (which assumes 1D bundle). Go
        # straight to DFoTOuterStage's __init__.
        super().__init__(
            action_dim=action_dim,
            cond_encoder=cond_encoder,
            backbone=backbone,
            diffusion=diffusion,
            cond_output_key=cond_output_key,
        )

        vae = load_pretrained_vae(vae_checkpoint_path)
        if not getattr(vae, "spatial_latent", False):
            raise NotImplementedError(
                "ImageSpatialDFoTOuterStage requires a VAE with spatial_latent=True."
            )
        self.vae = vae
        self.image_key = str(image_key)
        self._latent_shape = (
            int(vae.latent_channels),
            int(vae.bottleneck_size),
            int(vae.bottleneck_size),
        )
        self.bundle_obs_keys = list(bundle_obs_keys)
        self.bundle_obs_dims = [int(d) for d in bundle_obs_dims]
        self._state_dim_total = sum(self.bundle_obs_dims)
        self._action_dim = int(action_dim)
        self._state_action_proj_dim = int(state_action_proj_dim)

        # External-cond MLP: concat(state, action) -> state_action_proj_dim.
        self.state_action_proj = nn.Sequential(
            nn.Linear(self._state_dim_total + self._action_dim, self._state_action_proj_dim),
            nn.SiLU(),
            nn.Linear(self._state_action_proj_dim, self._state_action_proj_dim),
        )

        # Backbone sanity-check: its cond_dim should match the projector.
        bb_cond_dim = int(getattr(backbone, "cond_dim", -1))
        if bb_cond_dim != self._state_action_proj_dim:
            raise ValueError(
                f"backbone.cond_dim ({bb_cond_dim}) must equal "
                f"state_action_proj_dim ({self._state_action_proj_dim})."
            )

        # Latent normalization: center and scale VAE latents so the
        # diffusion prior is standard Gaussian. Without this, the VAE's
        # per-channel bias causes sampled latents to decode to near-white.
        latent_stats_path = getattr(vae, "_latent_stats_path", None)
        if latent_stats_path is None:
            # Default: try to load from the same dir as the VAE ckpt.
            import os
            stats_path = os.path.join(
                os.path.dirname(vae_checkpoint_path), "vae_latent_stats.json"
            )
            if os.path.exists(stats_path):
                latent_stats_path = stats_path
        if latent_stats_path is not None:
            import json
            with open(latent_stats_path) as f:
                ls = json.load(f)
            self.register_buffer(
                "_latent_mean",
                torch.tensor(ls["latent_mean"]).float().reshape(1, -1, 1, 1),
            )
            self.register_buffer(
                "_latent_std",
                torch.tensor(ls["latent_std"]).float().reshape(1, -1, 1, 1),
            )
        else:
            self._latent_mean = None
            self._latent_std = None

    # ------------------------------------------------------------------
    # Shape advertised to the sampler / AR allocators.
    # ------------------------------------------------------------------

    @property
    def bundle_shape(self) -> tuple:
        """Spatial trailing dims of the diffusion target. Sampler builds
        ``(B, T, *bundle_shape)``."""
        return self._latent_shape

    @property
    def bundle_dim(self) -> int:
        """Backwards compatibility: total numel of bundle_shape. The algo
        and sampler should prefer ``bundle_shape`` when present, but a
        few legacy code paths still read ``bundle_dim``."""
        c, h, w = self._latent_shape
        return c * h * w

    @property
    def action_slice(self) -> slice:
        """No action in this bundle — return an empty slice to signal
        downstream code that action extraction from the bundle is N/A.
        Callers that want predicted actions from this model should be
        upgraded to a different stage (the action-as-extra-token
        hybrid follow-up)."""
        return slice(0, 0)

    # ------------------------------------------------------------------
    # External cond from state + action.
    # ------------------------------------------------------------------

    def _state_action_to_cond(
        self, batch: dict, ctx: SimpleNamespace
    ) -> Optional[torch.Tensor]:
        """Per-step ``(state || action)`` -> projected ``external_cond``.

        Packed:  returns ``(T_total, state_action_proj_dim)``.
        Padded:  returns ``(B, T, state_action_proj_dim)``.
        """
        pieces = []
        for key in self.bundle_obs_keys:
            if key not in ctx.obs:
                raise KeyError(
                    f"obs key '{key}' required in external_cond but missing "
                    f"from ctx.obs (keys: {list(ctx.obs.keys())})."
                )
            pieces.append(ctx.obs[key])
        pieces.append(batch[ctx.action_key])
        concat = torch.cat(pieces, dim=-1)
        return self.state_action_proj(concat)

    # ------------------------------------------------------------------
    # Build VAE latent target (spatial).
    # ------------------------------------------------------------------

    def normalize_latent(self, z: torch.Tensor) -> torch.Tensor:
        if self._latent_mean is not None:
            return (z - self._latent_mean.to(z.device)) / self._latent_std.to(z.device)
        return z

    def denormalize_latent(self, z: torch.Tensor) -> torch.Tensor:
        if self._latent_mean is not None:
            return z * self._latent_std.to(z.device) + self._latent_mean.to(z.device)
        return z

    # ------------------------------------------------------------------
    # Video-rollout hook (COMBINE A — decode-on-outer-stage).
    #
    # This stage owns: building per-step (state,action) external_cond, the
    # CONDITIONAL chunk/AR sampler (with optional GT-context anchoring), and
    # the VAE decode of the predicted spatial latent. Moved byte-for-byte
    # from the old ``eval_dfot_spatial_video_rollout.DFoTSpatialVideoRolloutEval``
    # (``_rollout`` + ``_build_cond_seq`` + per-episode body). Side-by-side
    # [GT|pred] panel; per-step MSE vs the per-timestep-aligned GT frames.
    # ------------------------------------------------------------------

    video_metric_prefix = "spatial"
    video_panel = "sidebyside"
    video_has_extra_metrics = False

    @torch.no_grad()
    def _rollout_latent(
        self, ev, algo, cond_seq: torch.Tensor, device, context_latent=None,
        batch_size: int = 1,
    ) -> torch.Tensor:
        """Returns predicted latent ``(batch_size, T, C, H, W)``.

        If ``context_latent`` (1, n_ctx, C, H, W) is given, the first n_ctx
        latents are seeded from GT and pinned clean across all sampling steps
        (anchored conditional rollout — predict the continuation given the real
        first frame[s] + the action sequence). Else the whole sequence is
        generated from noise conditioned only on the actions."""
        from egomimic.models.diffusion.diffusion.discrete_diffusion import DiscreteDiffusion
        from egomimic.models.diffusion.sampling import (
            sample as _sample,
            staircase_ar_schedule,
            vanilla_schedule,
        )

        bundle_shape = self.bundle_shape
        T = cond_seq.shape[1]
        discrete_ts = (
            int(algo.diffusion.timesteps)
            if isinstance(algo.diffusion, DiscreteDiffusion)
            else None
        )
        if ev.mode == "chunk":
            schedule = vanilla_schedule(
                n_steps=ev.n_chunk_steps, T=T, discrete_timesteps=discrete_ts,
            ).to(device)
        else:
            schedule = staircase_ar_schedule(
                T=T, chunk_size=ev.ar_chunk_size, step_size=ev.ar_step_size,
                discrete_timesteps=discrete_ts,
            ).to(device)

        if context_latent is None:
            return _sample(
                algo.diffusion, algo.backbone, schedule_matrix=schedule,
                x_shape=bundle_shape, batch_size=batch_size,
                external_cond=cond_seq, cfg_scale=ev.cfg_scale, device=device,
            )

        from egomimic.models.diffusion.sampling import sample_step
        n_ctx = context_latent.shape[1]
        clean = -1 if discrete_ts is not None else 0.0
        schedule = schedule.clone()
        schedule[:, :n_ctx] = clean
        x = torch.randn(batch_size, T, *bundle_shape, device=device)
        x[:, :n_ctx] = context_latent
        for s in range(schedule.shape[0] - 1):
            x = sample_step(
                algo.diffusion, algo.backbone, x=x,
                current_levels=schedule[s], next_levels=schedule[s + 1],
                external_cond=cond_seq, eta=0.0, cfg_scale=ev.cfg_scale,
            )
            x[:, :n_ctx] = context_latent
        return x

    def _build_cond_seq(
        self, algo, _batch: dict, emb_id: int, start: int, T: int
    ) -> torch.Tensor:
        """``(1, T, state_action_proj_dim)`` projected cond over a slice
        of the val batch starting at frame ``start``."""
        ac_key = algo.resolved_ac_keys[emb_id]
        pieces = []
        for key in self.bundle_obs_keys:
            pieces.append(_batch[key][start : start + T])
        pieces.append(_batch[ac_key][start : start + T])
        concat = torch.cat(pieces, dim=-1)  # (T, state_dim + action_dim)
        cond = self.state_action_proj(concat)  # (T, cond_dim)
        return cond.unsqueeze(0)                # (1, T, cond_dim)

    @torch.no_grad()
    def rollout_video_episode(
        self, ev, algo, _batch, emb_id, ep_idx, ep_start, ep_len, device
    ) -> SimpleNamespace:
        """Per-episode world-model video rollout for the spatial stage."""
        imgs = _batch[ev.image_key]
        is_packed = _batch.get("_packed", False)
        T_rollout = min(ev.rollout_steps, ep_len)

        if is_packed:
            start = ep_start
            cond_seq = self._build_cond_seq(
                algo, _batch, emb_id, start, T_rollout,
            )
            gt_seq = imgs[start : start + T_rollout]
        else:
            # Padded: take row ep_idx, first T_rollout frames.
            ac_key = algo.resolved_ac_keys[emb_id]
            pieces = []
            for key in self.bundle_obs_keys:
                pieces.append(_batch[key][ep_idx, :T_rollout])
            pieces.append(_batch[ac_key][ep_idx, :T_rollout])
            concat = torch.cat(pieces, dim=-1)
            cond_seq = self.state_action_proj(concat).unsqueeze(0)
            gt_seq = imgs[ep_idx, :T_rollout]

        cond_seq = cond_seq.to(device).float()
        # Optional GT context anchor: VAE-encode the first n_context frames
        # to latents and pin them clean -> predict the continuation given
        # the real frame(s) + the action sequence (the in-distribution task).
        context_latent = None
        if ev.n_context_frames > 0:
            n_ctx = min(ev.n_context_frames, T_rollout)
            ctx_img = gt_seq[:n_ctx].to(device).float()
            if ctx_img.max() > 1.5:
                ctx_img = ctx_img / 255.0
            mu, _lv = self.vae.encode(ctx_img)
            context_latent = self.normalize_latent(mu).unsqueeze(0)
        # ---- Sampler returns (1, T, C, H, W) ----
        pred_latent = self._rollout_latent(
            ev, algo, cond_seq, device, context_latent=context_latent,
        ).squeeze(0)
        # ---- VAE decode -> pixel frames (T, 3, H, W) ----
        pred_frames = self.vae.decode(self.denormalize_latent(pred_latent))

        n_cmp = min(ev.recon_loss_n_frames, pred_frames.shape[0])
        gt_f = gt_seq[:n_cmp].to(device).float()
        if gt_f.max() > 1.5:
            gt_f = gt_f / 255.0

        return SimpleNamespace(
            pred_frames=pred_frames,
            gt_for_mse=gt_f,
            # Raw GT slice (NOT pre-normalized) — the unified eval's
            # side-by-side panel reproduces the original per-frame
            # ``gt_seq[t] / (255 if gt_seq.max()>1.5 else 1)`` exactly.
            gt_panel_raw=gt_seq,
            extra_metrics={},
        )

    def _build_latent(self, ctx: SimpleNamespace) -> torch.Tensor:
        """``image -> VAE.encode -> spatial latent``.

        Packed mode: image is ``(T_total, C, H, W)``; output is
        ``(T_total, latent_c, h, w)``.

        Padded mode: image is ``(B, T, C, H, W)``; output is
        ``(B, T, latent_c, h, w)``.
        """
        if self.image_key not in ctx.obs:
            raise KeyError(
                f"image key '{self.image_key}' required but missing from "
                f"ctx.obs (keys: {list(ctx.obs.keys())})."
            )
        img = ctx.obs[self.image_key]
        orig_dim = img.dim()
        if orig_dim == 5:
            B, T, C, H, W = img.shape
            img_flat = img.reshape(B * T, C, H, W)
        elif orig_dim == 4:
            img_flat = img
        else:
            raise ValueError(
                f"image has unexpected dim {orig_dim}; expected 4 (packed) or 5 (padded)."
            )
        with torch.no_grad():
            mu, _logvar = self.vae.encode(img_flat)
        # mu: (N, latent_c, h, w). Reshape back to padded form if needed.
        if orig_dim == 5:
            mu = mu.reshape(B, T, *mu.shape[1:])
        return self.normalize_latent(mu)

    # ------------------------------------------------------------------
    # encode -> q_sample on spatial latent. external_cond computed and
    # attached to ctx.
    # ------------------------------------------------------------------

    def forward(self, batch: dict, ctx: SimpleNamespace) -> torch.Tensor:
        """Override to handle packed data by running each episode
        independently through the backbone (no padding, no mask needed).
        """
        x_t = self.encode(batch, ctx)
        time_cond = ctx.q_state["time_cond"]

        if not ctx.is_packed:
            v_pred = self.inner_stage(
                x_t, time_cond, external_cond=ctx.external_cond,
            )
        else:
            cu = ctx.cu_seqlens
            B = cu.shape[0] - 1
            pieces = []
            for i in range(B):
                s, e = int(cu[i].item()), int(cu[i + 1].item())
                x_ep = x_t[s:e].unsqueeze(0)
                t_ep = time_cond[s:e].unsqueeze(0)
                c_ep = ctx.external_cond[s:e].unsqueeze(0)
                v_ep = self.inner_stage(x_ep, t_ep, external_cond=c_ep)
                pieces.append(v_ep.squeeze(0))
            v_pred = torch.cat(pieces, dim=0)

        self.decode(v_pred, batch, ctx)
        return v_pred

    def encode(self, batch: dict, ctx: SimpleNamespace) -> torch.Tensor:
        latent = self._build_latent(ctx)

        # Build external_cond from (state, action).
        cond = self._state_action_to_cond(batch, ctx)

        if ctx.is_packed:
            if latent.dim() != 4:
                raise ValueError(
                    f"packed latent must be (T_total, C, H, W); got "
                    f"{tuple(latent.shape)}"
                )
            T_total = latent.shape[0]
            t = self._sample_noise_levels((T_total,), latent.device)
        else:
            if latent.dim() != 5:
                raise ValueError(
                    f"padded latent must be (B, T, C, H, W); got "
                    f"{tuple(latent.shape)}"
                )
            B, T = latent.shape[:2]
            t = self._sample_noise_levels((B, T), latent.device)

        # For discrete diffusion, q_sample needs explicit noise so we can
        # store it in q_state for compute_loss.
        from egomimic.models.diffusion.diffusion.discrete_diffusion import DiscreteDiffusion as _DD
        if isinstance(self.diffusion, _DD):
            noise = torch.randn_like(latent).clamp_(
                -self.diffusion.clip_noise, self.diffusion.clip_noise
            )
            x_t = self.diffusion.q_sample(latent, t, noise=noise)
            ctx.q_state = {
                "x_t": x_t, "k": t, "time_cond": t,
                "noise": noise, "x_start": latent,
            }
        else:
            q = self.diffusion.q_sample(latent, t)
            ctx.q_state = q
            x_t = q["x_t"]
        ctx.external_cond = cond
        ctx.latent_clean = latent
        return x_t
