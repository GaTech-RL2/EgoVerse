"""``ObsActionImageDFoTOuterStage`` — joint obs+action+image DFoT.

Extends ``ObsActionDFoTOuterStage`` so the diffusion bundle per step is
``[state, vae_latent_flat, action]``, where ``vae_latent_flat`` is the
flattened spatial latent (e.g. 4x6x6 = 144D) produced by a *frozen*
pretrained ``ImageVAE`` on the input image.

This is what lets the diffusion model jointly predict actions AND
future video frames: at inference we sample the bundle, slice the
latent portion, and pass it through the VAE decoder to recover pixel
frames.

The image goes ONLY into the bundle now — the cond_encoder is image-
free (the original obs+action variant used image as AdaLN cond; here
that's redundant since the image is part of the diffused state and we
don't want to double-count).

Bundle layout (in concat order):
  [ bundle_obs_keys ... | vae_latent_flat | action ]

So ``action_slice`` is the trailing 2D, and ``image_latent_slice``
indexes the middle ``latent_flat_dim``-wide chunk.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import List

import torch

from egomimic.algo.diffusion.outer_stages.obs_action_outer_stage import ObsActionDFoTOuterStage
from egomimic.algo.diffusion.vae_algo import load_pretrained_vae


class ObsActionImageDFoTOuterStage(ObsActionDFoTOuterStage):
    """DFoT outer stage that diffuses ``[state, image_latent, action]``
    jointly per step, with a frozen pretrained ``ImageVAE`` providing
    the image latent.

    Args:
        action_dim: ACTION-only width (e.g. 2 for pushshapes).
        cond_encoder: ``CondEncoderModule``. Usually image-free for this
            variant -- the image is in the bundle, not in the cond path.
            Pass an empty-obs encoder if you really want no cond.
        backbone: ``DFoTBackbone`` with ``action_dim = bundle_dim``
            (= sum(bundle_obs_dims) + latent_flat_dim + action_dim).
        diffusion: continuous / discrete diffusion module.
        bundle_obs_keys: obs keys to include in the bundle (e.g.
            ``["state_agent_obj"]``).
        bundle_obs_dims: per-key feature width for each obs key.
        vae_checkpoint_path: path to the frozen ImageVAE Lightning ckpt.
            Loaded via ``algo.vae.algo.load_pretrained_vae``.
        image_key: which obs key carries the image stream. Default
            ``"front_img_1"``.
        cond_output_key: unused for this variant since cond_encoder is
            typically empty; kept for ABI parity with the base class.
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
        action_in_bundle: bool = True,
        cond_output_key: str = "fused_cond",
    ):
        # Load the VAE first so we know its latent_flat_dim — that
        # determines the bundle width the backbone has to expect.
        vae = load_pretrained_vae(vae_checkpoint_path)
        if not getattr(vae, "spatial_latent", False):
            raise NotImplementedError(
                "ObsActionImageDFoTOuterStage currently only supports "
                "VAEs with spatial latent (latent shape = "
                "(latent_channels, H', W')). Re-train the VAE with "
                "spatial_latent=True (e.g. config vae_pushshapes_v4)."
            )
        latent_c = int(vae.latent_channels)
        latent_h = int(vae.bottleneck_size)
        latent_w = int(vae.bottleneck_size)
        latent_flat_dim = latent_c * latent_h * latent_w

        # Treat the image latent as just another "obs" key in the bundle
        # construction. We append "_image_latent" to the obs key list +
        # latent_flat_dim to the dim list; ``_build_bundle`` (inherited)
        # will then concat in order: [bundle_obs_keys..., _image_latent,
        # action]. Override ``_build_bundle`` below to compute the
        # latent on the fly from the image via the frozen VAE.
        extended_keys = list(bundle_obs_keys) + ["_image_latent"]
        extended_dims = list(bundle_obs_dims) + [latent_flat_dim]

        super().__init__(
            action_dim=action_dim,
            cond_encoder=cond_encoder,
            backbone=backbone,
            diffusion=diffusion,
            bundle_obs_keys=extended_keys,
            bundle_obs_dims=extended_dims,
            action_in_bundle=action_in_bundle,
            cond_output_key=cond_output_key,
        )

        # Register the frozen VAE as a submodule so .to(device) moves
        # its weights. All its params are already requires_grad=False
        # from ``load_pretrained_vae``.
        self.vae = vae
        self.image_key = str(image_key)
        self._latent_shape = (latent_c, latent_h, latent_w)
        self._latent_flat_dim = int(latent_flat_dim)
        # Save the raw obs-key list (without the synthetic _image_latent)
        # so callers can introspect what real keys are in the bundle.
        self.real_bundle_obs_keys = list(bundle_obs_keys)
        self.real_bundle_obs_dims = list(bundle_obs_dims)

    @property
    def image_latent_slice(self) -> slice:
        """Slice into the bundle's trailing dim that holds the flat
        image latent. Useful for the self-rollout eval to pull out
        predicted latents before VAE-decoding them."""
        # bundle layout: [state(dims)... | latent(flat_dim) | action(action_dim)]
        start = sum(self.real_bundle_obs_dims)
        return slice(start, start + self._latent_flat_dim)

    @property
    def latent_shape(self) -> tuple[int, int, int]:
        """``(latent_channels, latent_h, latent_w)`` — the un-flattened
        spatial shape of the VAE's latent. The eval reshapes the bundle's
        latent slice to this before calling ``vae.decode``."""
        return self._latent_shape

    # ------------------------------------------------------------------
    # Video-rollout hook (COMBINE A — decode-on-outer-stage).
    #
    # The family-agnostic ``DFoTVideoRolloutEval`` calls this per episode.
    # This stage owns: the UNCONDITIONAL chunk/AR bundle sampler, the
    # latent-slice extraction, and the frozen-VAE decode that turn sampler
    # output into pixel frames. Code below is moved byte-for-byte from the
    # old ``eval_dfot_video_rollout.DFoTVideoRolloutEval._rollout`` +
    # ``_decode_latents_to_frames`` + the per-episode body of its
    # ``compute_metrics_and_viz`` (single-panel: t=0 GT prepended).
    # ------------------------------------------------------------------

    #: metric-key infix the unified eval uses (``Valid/embN_<prefix>_recon_mse_*``).
    video_metric_prefix = "video"
    #: panel layout the unified eval assembles for this family.
    video_panel = "single_t0prepend"
    #: this family compares preds against the GT episode's first-N frames
    #: (unconditioned rollout — no per-step GT alignment guarantee).
    video_has_extra_metrics = False

    @torch.no_grad()
    def _decode_latents_to_frames(
        self, latent_flat: torch.Tensor
    ) -> torch.Tensor:
        """``(T, latent_flat_dim) -> (T, 3, H, W)``."""
        c, h, w = self.latent_shape
        z = latent_flat.view(-1, c, h, w)
        # Frozen VAE on the right device.
        return self.vae.decode(z)

    @torch.no_grad()
    def _rollout_bundle(self, ev, algo, device, cond) -> torch.Tensor:
        """Sample a (T, bundle_dim) bundle. Mode = ``"chunk"`` uses
        ``algo._sample_chunk`` (uniform per-token noise + DDIM). Mode =
        ``"ar"`` builds a staircase schedule matrix and calls
        ``sample`` directly so the per-token noise pattern mirrors
        DFoT's training AR schedule."""
        from egomimic.models.diffusion.diffusion.discrete_diffusion import DiscreteDiffusion
        from egomimic.models.diffusion.sampling import (
            sample as _sample,
            staircase_ar_schedule,
        )

        T = ev.rollout_steps
        if ev.mode == "chunk":
            bundle = algo._sample_chunk(B=1, T=T, cond=cond, device=device)
            return bundle.squeeze(0)

        # mode == "ar"
        bundle_dim = int(algo.outer_stage.bundle_dim)
        discrete_ts = (
            int(algo.diffusion.timesteps)
            if isinstance(algo.diffusion, DiscreteDiffusion)
            else None
        )
        schedule = staircase_ar_schedule(
            T=T,
            chunk_size=ev.ar_chunk_size,
            step_size=ev.ar_step_size,
            discrete_timesteps=discrete_ts,
        ).to(device)
        ec = cond.unsqueeze(0) if cond is not None else None
        bundle = _sample(
            algo.diffusion,
            algo.backbone,
            schedule_matrix=schedule,
            action_dim=bundle_dim,
            batch_size=1,
            external_cond=ec,
            cfg_scale=ev.cfg_scale,
            device=device,
        ).squeeze(0)
        return bundle

    @torch.no_grad()
    def rollout_video_episode(
        self, ev, algo, _batch, emb_id, ep_idx, ep_start, ep_len, device
    ) -> SimpleNamespace:
        """Per-episode video rollout for the joint obs+action+image stage.

        Returns a namespace the unified eval consumes:
          * ``pred_frames`` ``(T,3,H,W)`` float in [0,1] — decoded preds.
          * ``gt_for_mse``  ``(T,3,H,W)`` float in [0,1] — GT first-N frames
            (this family's MSE is against the GT episode's leading frames;
            the rollout is unconditioned so alignment is only distributional).
          * ``gt_t0_chw``   the raw t=0 GT image to prepend as launch context.
          * ``extra_metrics`` empty for this family.
        """
        imgs = _batch[ev.image_key]
        is_packed = _batch.get("_packed", False)

        # t=0 launch image for this episode.
        if is_packed:
            img_chw = imgs[ep_start]
        else:
            img_chw = imgs[ep_idx, 0]

        # The model's _sample_chunk wants a cond tensor that matches the
        # backbone's external_cond shape. For the obs+action+image
        # variant the cond_encoder is usually empty (image is in the
        # bundle, not the cond), so cond should be None.
        cond = None
        bundle_pred = self._rollout_bundle(ev, algo, device, cond)  # (T, bundle_dim)
        latent_seq = bundle_pred[:, self.image_latent_slice]
        pred_frames = self._decode_latents_to_frames(latent_seq)

        n_cmp = min(ev.recon_loss_n_frames, pred_frames.shape[0])
        if is_packed:
            gt_seq = imgs[ep_start : ep_start + n_cmp]
        else:
            gt_seq = imgs[ep_idx, :n_cmp]
        # GT in [0,1] float; predicted also in [0,1] from VAE sigmoid.
        gt_f = gt_seq.to(device).float()
        if gt_f.max() > 1.5:
            gt_f = gt_f / 255.0

        return SimpleNamespace(
            pred_frames=pred_frames,
            gt_for_mse=gt_f,
            gt_t0_chw=img_chw,
            extra_metrics={},
        )

    # ------------------------------------------------------------------
    # Override the parent's _build_bundle so the "_image_latent" entry
    # is computed from the image via the frozen VAE, not pulled from
    # ctx.obs directly.
    # ------------------------------------------------------------------

    def _build_bundle(self, batch: dict, ctx: SimpleNamespace) -> torch.Tensor:
        actions = batch[ctx.action_key]
        pieces: List[torch.Tensor] = []

        # 1. Real obs keys (state, etc.).
        for key, dim in zip(self.real_bundle_obs_keys, self.real_bundle_obs_dims):
            if key not in ctx.obs:
                raise KeyError(
                    f"obs key '{key}' required in bundle but missing from "
                    f"ctx.obs (keys: {list(ctx.obs.keys())})."
                )
            v = ctx.obs[key]
            if v.shape[-1] != dim:
                raise ValueError(
                    f"obs '{key}' trailing dim {v.shape[-1]} != configured "
                    f"bundle_obs_dim {dim}."
                )
            pieces.append(v)

        # 2. Image latent via frozen VAE encoder. Always no_grad.
        if self.image_key not in ctx.obs:
            raise KeyError(
                f"image key '{self.image_key}' required but missing from "
                f"ctx.obs (keys: {list(ctx.obs.keys())})."
            )
        img = ctx.obs[self.image_key]
        # img can be packed (T_total, C, H, W) or padded (B, T, C, H, W).
        # Collapse to (N, C, H, W) for the VAE encoder, then reshape back.
        orig_dim = img.dim()
        if orig_dim == 5:
            B, T, C, H, W = img.shape
            img_flat = img.reshape(B * T, C, H, W)
        elif orig_dim == 4:
            img_flat = img
        else:
            raise ValueError(
                f"image '{self.image_key}' has unexpected dim {orig_dim}; "
                f"expected 4 (packed T_total,C,H,W) or 5 (padded B,T,C,H,W)."
            )
        with torch.no_grad():
            # Use posterior mean (mu) for the latent — deterministic so
            # the diffusion target is well-defined.
            mu, _logvar = self.vae.encode(img_flat)
        # mu shape: (N, latent_c, h', w'); flatten the spatial dims.
        latent_flat = mu.flatten(1)
        if orig_dim == 5:
            latent_flat = latent_flat.reshape(B, T, -1)
        pieces.append(latent_flat)

        # 3. Action (only when it is a diffusion target; in world-model mode
        # it is routed through external_cond by the base ``encode``).
        if self.action_in_bundle:
            pieces.append(actions)

        bundle = torch.cat(pieces, dim=-1)
        if bundle.shape[-1] != self._bundle_dim:
            raise RuntimeError(
                f"bundle width mismatch: built {bundle.shape[-1]}, expected "
                f"{self._bundle_dim}."
            )
        return bundle
