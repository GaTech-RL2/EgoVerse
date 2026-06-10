"""
DFoT Algo: per-token-noise-level diffusion over action chunks.

Supports both packed (full variable-length episodes; ``cu_seqlens``-driven
within-episode attention) and padded (fixed-T windows) batches. In packed
mode obs is per-frame (one obs per action timestep, aligned via the
``pushshapes.get_keymap`` per-frame keymap); in padded mode obs may be
single-frame and is broadcast across T at the per-token AdaLN. Loss is the
diffusion (epsilon / v / x0) MSE with the configured weighting strategy.

Two inference modes (see ``inference_step``):
  * "ar":    rolling causal-AR staircase. One ``sample_step`` per env tick.
             Matches training distribution. Default.
  * "chunk": vanilla DDIM over a fixed window with plan-and-execute. Legacy
             baseline; doesn't exercise DFoT's per-token-noise capability.

Teacher-forced offline val viz lives in ``egomimic/eval/eval_dfot_val.py``
(``DFoTValEval``).
"""

from collections import OrderedDict
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from overrides import override

from egomimic.algo.algo import Algo
from egomimic.models.diffusion.backbones.backbone import DFoTBackbone
from egomimic.models.diffusion.diffusion.continuous_diffusion import ContinuousDiffusion
from egomimic.models.diffusion.diffusion.discrete_diffusion import DiscreteDiffusion
from egomimic.algo.diffusion.outer_stages.outer_stage import DFoTOuterStage, make_dfot_ctx
from egomimic.models.diffusion.sampling import ddim_sample, ddpm_sample, sample_step
from egomimic.models.stems.cond_encoders import CondEncoderModule
from egomimic.rldb.embodiment.embodiment import get_embodiment, get_embodiment_id


class _PackedBackboneWrapper:
    """Closure-style wrapper so the diffusion module's 3-arg
    ``backbone(x, t, cond)`` call automatically threads cu_seqlens / max_seqlen
    into ``DFoTBackbone.forward`` for packed-mode within-episode attention."""

    def __init__(self, backbone: DFoTBackbone, cu_seqlens, max_seqlen):
        self.backbone = backbone
        self.cu_seqlens = cu_seqlens
        self.max_seqlen = max_seqlen

    def __call__(self, x, noise_levels, external_cond=None):
        return self.backbone(
            x,
            noise_levels,
            external_cond=external_cond,
            cu_seqlens=self.cu_seqlens,
            max_seqlen=self.max_seqlen,
        )


class DFoTLoss(nn.Module):
    """DFoT epsilon-MSE with sigmoid (SNR-style) weighting.

    Self-contained loss for the ``DFoT`` algo (each model owns its own loss).
    Implements the ``forward(batch, ctx) -> scalar`` contract the algo
    orchestrator calls.

    Reads:
      - ``batch["pred_v"]``: v-prediction emitted by the DFoT backbone
        (written by ``DFoTOuterStage.decode``).
      - ``ctx.q_state``: dict produced by ``diffusion.q_sample`` during
        ``DFoTOuterStage.encode``. Carries ``x_t``, ``noise``, ``alpha_t``,
        ``sigma_t``, ``logsnr``.

    The actual math lives in ``diffusion.compute_loss``; this class is the
    interface that fits the OuterStage-orchestrated training loop and
    reduces the per-token loss to a scalar.

    ``diffusion`` is the same instance held by the outer stage — it has
    no learnable params (in continuous mode) so this is a free reference,
    not a duplicate submodule.
    """

    def __init__(self, diffusion: nn.Module):
        super().__init__()
        self.diffusion = diffusion

    def forward(self, batch: dict, ctx) -> torch.Tensor:
        # Structured-target outer stages (e.g. the 2D spatial-image + action
        # policy) compute their own multi-term loss and stash it here, since a
        # single (image, action) target can't flow through the scalar v-MSE
        # path below. No-op for every existing 1D/spatial stage.
        precomputed = getattr(ctx, "precomputed_loss", None)
        if precomputed is not None:
            return precomputed
        v_pred = batch["pred_v"]
        q_state = ctx.q_state
        per_token = self.diffusion.compute_loss(v_pred, q_state)
        return per_token.mean()


class DFoT(Algo):
    """Diffusion Forcing Transformer (action-chunk denoising) Algo.

    Args:
        action_dim: action feature width.
        action_horizon: AR buffer / chunk length T. Also the planning window
            for legacy chunk-mode inference.
        cond_encoder: ``CondEncoderModule`` for obs -> ``(B, T, cond_dim)``
            (per-frame) or ``(B, cond_dim)`` (single-frame, broadcast).
        backbone: ``DFoTBackbone`` (built via Hydra). Owns x / cond / time
            projections and the ``Isotropic`` trunk.
        norm_stats: ``MultiDataset`` (injected by
            ``pl_model._instantiate_model``).
        diffusion_type: ``"discrete"`` or ``"continuous"``.
        diffusion_kwargs: dict forwarded to the chosen diffusion class.
        sampler: ``"ddpm"`` or ``"ddim"`` — used only by chunk-mode inference.
        sampler_n_steps: denoising step count for chunk-mode inference.
        sampler_eta: DDIM eta (0.0 = deterministic) for chunk-mode inference.
        inference_mode: ``"ar"`` (default) for rolling causal-AR staircase or
            ``"chunk"`` for legacy plan-and-execute.
        ar_inference_chunk_size: tokens committed per env tick in AR mode
            (1 = classic causal AR; >1 = chunked staircase rungs).
            ``action_horizon`` must be divisible by this.
        domains: list of embodiment names (single-element for v1).
        ac_keys: dict ``embodiment_name -> action zarr key``.
        cond_output_key: key under which the cond encoder exposes its fused
            cond.
    """

    def __init__(
        self,
        outer_stage: DFoTOuterStage,
        action_dim: int,
        action_horizon: int,
        norm_stats,
        loss: Optional[nn.Module] = None,
        sampler: str = "ddim",
        sampler_n_steps: int = 50,
        sampler_eta: float = 0.0,
        inference_mode: str = "ar",
        ar_inference_chunk_size: int = 1,
        ar_inference_step_size: int = 1,
        cfg_scale: float = 1.0,
        sp_n_context: int = 4,
        sp_commit: int = 1,
        sp_n_samples: int = 1,
        domains: Optional[list] = None,
        ac_keys: Optional[dict] = None,
        device=None,
        **kwargs,
    ):
        """Refactored DFoT algo.

        Args:
            outer_stage: ``DFoTOuterStage`` owning the cond_encoder + backbone
                + diffusion submodules and implementing the training-path
                encode -> q_sample -> backbone -> decode flow.
            loss: an ``nn.Module`` (typically ``DFoTLoss``) that consumes
                ``batch['pred_v']`` + ``ctx.q_state`` and emits the scalar
                training loss.
            sampler / sampler_n_steps / sampler_eta / inference_mode /
            ar_inference_chunk_size / ar_inference_step_size / cfg_scale:
                Inference knobs. Closed-loop AR + chunk-mode inference paths
                still live on this algo class and consume the outer_stage's
                submodules via the ``cond_encoder`` / ``backbone`` /
                ``diffusion`` properties below.
        """
        super().__init__()
        self.norm_stats = norm_stats
        self.domains = list(domains or [])
        self.ac_keys = dict(ac_keys or {})
        self.action_dim = int(action_dim)
        self.action_horizon = int(action_horizon)
        self.cond_output_key = outer_stage.cond_output_key
        self.sampler = sampler
        self.sampler_n_steps = int(sampler_n_steps)
        self.sampler_eta = float(sampler_eta)
        if inference_mode not in {"ar", "chunk", "spatial_rh", "spatial_decoupled", "pixel_policy", "pixel_regress", "pixel_decoupled"}:
            raise ValueError(
                f"inference_mode must be 'ar', 'chunk', 'spatial_rh', or "
                f"'spatial_decoupled', got {inference_mode!r}"
            )
        self.inference_mode = inference_mode
        self.ar_inference_chunk_size = int(ar_inference_chunk_size)
        # Number of `sample_step` calls per env tick. step_size>1 advances
        # noise levels by 1/(n_rungs*step_size) per sub-step. Mirrors the
        # offline staircase_ar_schedule(chunk, step) shape.
        self.ar_inference_step_size = int(ar_inference_step_size)
        # Classifier-free-guidance scale at inference. 1.0 disables CFG.
        self.cfg_scale = float(cfg_scale)
        # spatial_rh closed-loop controller: context window + actions committed
        # per replan (sp_commit>1 = predict a short chunk open-loop, re-plan
        # every sp_commit ticks -> sp_commit-fold fewer diffusion rollouts).
        self.sp_n_context = int(sp_n_context)
        self.sp_commit = int(sp_commit)
        self.sp_n_samples = int(sp_n_samples)
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # outer_stage owns: cond_encoder, inner_stage (backbone), diffusion.
        # loss reads ctx.q_state (populated by outer_stage.encode) + batch.
        # If no loss is provided, default to DFoTLoss(outer_stage.diffusion).
        if loss is None:
            loss = DFoTLoss(outer_stage.diffusion)
        self.nets = nn.ModuleDict({"outer_stage": outer_stage, "loss": loss})
        self.nets = self.nets.float().to(self.device)

        # Resolve per-embodiment keys via norm_stats (HNet-style). Shared
        # base helper (collapse c5): see Algo._resolve_embodiment_keys.
        self._resolve_embodiment_keys(norm_stats)

    # ----- Convenience accessors so the inference-path code paths
    # (forward_eval, _sample_chunk, _inference_step_ar, _inference_step_chunk)
    # can keep referring to ``self.backbone`` / ``self.cond_encoder`` /
    # ``self.diffusion`` without going through ``self.nets["outer_stage"]``
    # each call. All three are submodules of the outer_stage.

    @property
    def outer_stage(self) -> DFoTOuterStage:
        return self.nets["outer_stage"]

    @property
    def loss(self) -> nn.Module:
        return self.nets["loss"]

    @property
    def cond_encoder(self) -> CondEncoderModule:
        return self.outer_stage.cond_encoder

    @property
    def backbone(self) -> DFoTBackbone:
        return self.outer_stage.inner_stage

    @property
    def diffusion(self) -> nn.Module:
        return self.outer_stage.diffusion

    # Packed-mode metadata that must NOT go through zarr_key_to_keyname
    # resolution (these are bookkeeping, not feature tensors).
    _PACKED_META_KEYS = ("cu_seqlens", "max_seq_len", "seq_lens")

    # ---- Algo API -------------------------------------------------------- #

    @override
    def process_batch_for_training(self, batch):
        """Accept both padded ``(B, T, *)`` batches and packed
        ``(T_total, *)`` + ``cu_seqlens`` batches."""
        processed = {}
        for emb_name, _batch in batch.items():
            emb_id = get_embodiment_id(emb_name)
            processed[emb_id] = {}
            is_packed = "cu_seqlens" in _batch

            for key, value in _batch.items():
                if is_packed and key in self._PACKED_META_KEYS:
                    processed[emb_id][key] = value
                    continue
                key_name = self.norm_stats.zarr_key_to_keyname(key, emb_id)
                if key_name is not None:
                    processed[emb_id][key_name] = value
                else:
                    processed[emb_id][key] = value

            processed[emb_id]["_packed"] = is_packed
            # Synthesize seq_lens from cu_seqlens for packed batches if the
            # collator didn't emit it. Several downstream evaluators
            # (``PackedSimEval._infer_n_episodes``, etc.) key off seq_lens
            # to find episode boundaries — silently returning 0 episodes
            # when it's missing produces no metrics + no videos.
            if is_packed and "seq_lens" not in processed[emb_id]:
                cu = processed[emb_id].get("cu_seqlens")
                if cu is not None and torch.is_tensor(cu):
                    processed[emb_id]["seq_lens"] = (cu[1:] - cu[:-1]).to(torch.int64)
            processed[emb_id] = self.norm_stats.normalize(processed[emb_id], emb_id)
            processed[emb_id]["embodiment"] = torch.tensor(
                [emb_id], device=self.device, dtype=torch.int64
            )
            for key, value in processed[emb_id].items():
                if isinstance(value, torch.Tensor):
                    value = value.to(self.device)
                    if value.is_floating_point():
                        value = value.float()
                    processed[emb_id][key] = value
        return processed

    # _build_obs is inherited from Algo (collapse c5 — byte-identical).

    def _encode_cond(self, obs: dict, T: int) -> Optional[torch.Tensor]:
        """Encode obs to per-token cond. Honors per-frame obs (no reduction)."""
        cond_dict = self.cond_encoder.encode(obs, T)
        return cond_dict.get(self.cond_output_key)

    def _encode_cond_packed(self, obs: dict) -> Optional[torch.Tensor]:
        """Packed-mode cond. Obs values are (T_total, ...). We fake batch=1
        by ``unsqueeze(0)``-ing each so ``CondEncoderModule.encode`` runs in
        its already-per-frame branch (it doesn't broadcast when dim already
        matches). Output: (T_total, d_cond) or None."""
        obs_3d = {
            k: (v.unsqueeze(0) if torch.is_tensor(v) else v)
            for k, v in obs.items()
        }
        # T_action argument is unused when obs is already per-frame (dim==3
        # for state, dim==5 for image); pass any non-zero placeholder.
        cond_dict = self.cond_encoder.encode(obs_3d, T_action=1)
        c = cond_dict.get(self.cond_output_key)
        if c is None:
            return None
        if c.dim() == 3 and c.shape[0] == 1:
            c = c.squeeze(0)
        return c  # (T_total, d_cond)

    def _sample_noise_levels(self, shape, device) -> torch.Tensor:
        """Per-token random noise level. Discrete -> longs in [0, timesteps);
        continuous -> floats in (0, 1)."""
        if isinstance(self.diffusion, DiscreteDiffusion):
            return torch.randint(
                0, self.diffusion.timesteps, shape, device=device, dtype=torch.long
            )
        return torch.rand(shape, device=device).clamp_(1e-5, 1.0 - 1e-5)

    @override
    def forward_training(self, batch):
        """Refactored training forward: delegates encode/decode + loss to
        the outer_stage + loss submodules.

        For each embodiment:
          1. Build a DFoT context with packed/padded mode and obs.
          2. Call outer_stage(batch_emb, ctx) — runs encode -> q_sample,
             backbone, decode (writes batch[pred_v]).
          3. Call loss(batch_emb, ctx) — reads pred_v + ctx.q_state, returns
             scalar SNR-weighted eps-MSE.
        """
        predictions = OrderedDict()
        for emb_id, _batch in batch.items():
            ac_key = self.resolved_ac_keys.get(emb_id, "actions")
            is_packed = _batch.get("_packed", False)
            obs = self._build_obs(_batch, emb_id)
            ctx = make_dfot_ctx(
                is_packed=is_packed,
                action_key=ac_key,
                obs=obs,
                cu_seqlens=_batch.get("cu_seqlens") if is_packed else None,
                max_seqlen=(int(_batch.get("max_seq_len", 0)) or None) if is_packed else None,
            )
            self.outer_stage(_batch, ctx)
            mse = self.loss(_batch, ctx)
            predictions[f"{emb_id}_action_loss"] = mse
        return predictions

    @override
    def forward_eval(self, batch):
        """Returns val-loss + sampled chunks for each embodiment. Sampled
        chunks are always single-window (B=1 if packed) at length
        ``self.action_horizon`` — packed-mode val skips per-position chunk
        sampling for now (rollout drives closed-loop quality via
        ``inference_step``)."""
        unnorm = {}
        backbone = self.backbone
        for emb_id, _batch in batch.items():
            ac_key = self.resolved_ac_keys.get(emb_id, "actions")
            actions = _batch[ac_key]
            is_packed = _batch.get("_packed", False)
            obs = self._build_obs(_batch, emb_id)

            # Build the joint diffusion target (= the bundle for obs+action
            # outer stages, or just actions for vanilla DFoT). Previously this
            # path diffused raw `actions`, which is wrong for obs-action
            # variants whose backbone/diffusion are sized to the full bundle.
            _eval_ctx = make_dfot_ctx(
                is_packed=is_packed,
                action_key=ac_key,
                obs=obs,
                cu_seqlens=_batch.get("cu_seqlens") if is_packed else None,
                max_seqlen=(int(_batch.get("max_seq_len", 0)) or None) if is_packed else None,
            )
            _build_bundle = getattr(self.outer_stage, "_build_bundle", None)
            target = _build_bundle(_batch, _eval_ctx) if _build_bundle is not None else actions

            if is_packed:
                T_total = target.shape[0]
                cu = _batch["cu_seqlens"]
                msl = int(_batch.get("max_seq_len", 0)) or None
                cond = self._encode_cond_packed(obs)
                k = self._sample_noise_levels((T_total,), target.device)
                packed_backbone = _PackedBackboneWrapper(backbone, cu, msl)
                _, loss = self.diffusion(packed_backbone, target, k, external_cond=cond)
                unnorm[f"emb{emb_id}_loss"] = loss.mean()
                # No chunk sampling in packed val — too expensive per episode and
                # the closed-loop measure lives in inference_step / sim eval.
            else:
                B, T, _ = actions.shape
                cond = self._encode_cond(obs, T)
                k = self._sample_noise_levels((B, T), target.device)
                _, loss = self.diffusion(backbone, target, k, external_cond=cond)
                unnorm[f"emb{emb_id}_loss"] = loss.mean()
                sampled = self._sample_chunk(B, T, cond=cond, device=actions.device)
                # _sample_chunk returns the full bundle; pick out the action
                # slice (channel axis for 5-D pixel bundles — D02) so
                # unnormalize sees an action-shaped tensor.
                sampled = self._extract_bundle_actions(
                    sampled,
                    self.outer_stage.action_slice,
                    action_dim=int(
                        getattr(self.outer_stage, "action_dim", self.action_dim)
                    ),
                )
                preds = OrderedDict()
                preds[ac_key] = sampled
                unnorm_actions = self.norm_stats.unnormalize(preds, emb_id)
                for key, val in unnorm_actions.items():
                    unnorm[f"emb{emb_id}_{key}"] = val
        return unnorm

    def _sample_chunk(
        self, B: int, T: int, cond: Optional[torch.Tensor], device
    ) -> torch.Tensor:
        # Build the schedule + run sample() directly so we can plumb
        # cfg_scale through. (ddim_sample/ddpm_sample wrappers don't yet
        # expose cfg_scale; could be added if needed.)
        outer = self.outer_stage
        # Prefer ``bundle_shape`` (tuple) when the outer stage has it
        # (spatial backbones like ``DFoTSpatialBackbone``); fall back to
        # the legacy ``bundle_dim`` (int) for 1D-bundle outer stages.
        # ``hasattr(outer, "bundle_shape")`` is always True on the base
        # class, so check the tuple-ness explicitly.
        b_shape = getattr(outer, "bundle_shape", None)
        spatial_mode = isinstance(b_shape, tuple) and len(b_shape) > 1
        if self.sampler not in ("ddpm", "ddim"):
            raise ValueError(f"unknown sampler {self.sampler!r}")
        discrete_ts = (
            int(self.diffusion.timesteps)
            if isinstance(self.diffusion, DiscreteDiffusion)
            else None
        )
        n_steps = self.sampler_n_steps if self.sampler == "ddim" else (
            discrete_ts or self.sampler_n_steps
        )
        from egomimic.models.diffusion.sampling import vanilla_schedule, sample as _sample
        sm = vanilla_schedule(n_steps=n_steps, T=T, discrete_timesteps=discrete_ts)
        eta = 1.0 if self.sampler == "ddpm" else self.sampler_eta
        sample_kwargs = dict(
            schedule_matrix=sm,
            batch_size=B,
            external_cond=cond,
            eta=eta,
            cfg_scale=self.cfg_scale,
            device=device,
        )
        if spatial_mode:
            sample_kwargs["x_shape"] = b_shape
        else:
            sample_kwargs["action_dim"] = outer.bundle_dim
        return _sample(self.diffusion, self.backbone, **sample_kwargs)

    @staticmethod
    def _extract_bundle_actions(
        sampled: torch.Tensor,
        action_slice: slice,
        action_dim: Optional[int] = None,
    ) -> torch.Tensor:
        """Pick the action portion out of a sampled diffusion bundle.

        3-D ``(B, T, D)`` bundles keep the legacy trailing-axis slice
        (byte-identical to ``sampled[..., action_slice]``). 5-D pixel bundles
        ``(B, T, C, H, W)`` carry the action as broadcast channel planes, so
        the slice must hit the CHANNEL axis — ``sampled[..., action_slice]``
        lands on image width (D02) — and the planes decode to an action
        vector by global-avg-pool, mirroring the validated
        ``_inference_step_pixel_policy`` decode.
        """
        if sampled.dim() == 5:
            planes = sampled[:, :, action_slice]  # (B, T, C_a, H, W)
            actions = planes.mean(dim=(-2, -1))   # (B, T, C_a)
            if action_dim is not None:
                actions = actions[..., :action_dim]
            return actions
        return sampled[..., action_slice]

    # compute_losses is inherited from Algo (collapse c7 — the pure
    # sum-per-embodiment {emb}_action_loss reducer is now the Algo default).
    # log_info is inherited from Algo (collapse c5 — byte-identical default).

    # ---- Sim eval hook ---- #
    #
    # Two inference modes, selected by ``self.inference_mode`` (config):
    #   "ar"     — DFoT-flavored causal-AR rolling staircase (default).
    #              Each env step does ONE ``sample_step`` against the AR
    #              buffer; the front rung commits as the action(s).
    #              Matches training distribution (per-token random noise).
    #   "chunk"  — legacy chunk-replan: predict ``action_horizon`` actions
    #              with uniform-per-token DDIM, execute them one at a time,
    #              replan after the chunk is exhausted. Cheaper but doesn't
    #              exercise DFoT's per-token-noise capability.

    @torch.no_grad()
    def inference_step(
        self, obs_zarr: dict, t: int, emb_id: int, T_max=None
    ) -> "np.ndarray":
        if self.inference_mode == "ar":
            return self._inference_step_ar(obs_zarr, t, emb_id)
        if self.inference_mode == "chunk":
            return self._inference_step_chunk(obs_zarr, t, emb_id)
        if self.inference_mode == "spatial_rh":
            return self._inference_step_spatial_rh(obs_zarr, t, emb_id)
        if self.inference_mode == "spatial_decoupled":
            return self._inference_step_spatial_decoupled(obs_zarr, t, emb_id)
        if self.inference_mode == "pixel_policy":
            return self._inference_step_pixel_policy(obs_zarr, t, emb_id)
        if self.inference_mode == "pixel_regress":
            return self._inference_step_pixel_regress(obs_zarr, t, emb_id)
        if self.inference_mode == "pixel_decoupled":
            return self._inference_step_pixel_decoupled(obs_zarr, t, emb_id)
        raise ValueError(f"unknown inference_mode {self.inference_mode!r}")

    def _ar_state_init(self, device, external_cond):
        """Initialize the AR buffer + staircase geometry AND warm it up so
        every slot's actual noise level matches the schedule before any
        action is fired.

        Without warmup, the buffer is ``randn`` (all slots at noise 1.0) but
        the staircase schedule claims token 0 is at noise 1/K, token 1 at
        2/K, etc. — a mismatch that produces junk for the first K-1 env
        ticks. After ``n_rungs - 1`` ``sample_step`` calls, every slot has
        been denoised the right number of times and the buffer/schedule
        agree.
        """
        if self.action_horizon % self.ar_inference_chunk_size != 0:
            raise ValueError(
                f"action_horizon ({self.action_horizon}) must be divisible by "
                f"ar_inference_chunk_size ({self.ar_inference_chunk_size})."
            )
        n_rungs = self.action_horizon // self.ar_inference_chunk_size
        is_discrete = isinstance(self.diffusion, DiscreteDiffusion)
        if is_discrete:
            self._ar_unit = max(1, self.diffusion.timesteps // n_rungs)
        else:
            self._ar_unit = 1.0 / float(n_rungs)
        self._ar_discrete = is_discrete
        self._ar_buffer = torch.randn(
            1, self.action_horizon, self.outer_stage.bundle_dim,
            device=device, dtype=torch.float32,
        )
        self._sim_committed_queue = []

        # ---- Warmup: walk the staircase forward without committing ----
        # We need every slot at the noise level the schedule expects. The
        # buffer starts with all slots at noise 1.0 (fully noisy randn).
        # The staircase, when first queried, claims slot 0 is at 1/K. We
        # need to "demote" each slot's actual noise to match by running
        # n_rungs - 1 denoise steps where the schedule entries are shifted
        # back by one rung at the top.
        #
        # Concretely: at warmup step ``w`` (0-indexed, 0..n_rungs-2):
        #   declared current levels = (rung_idx + (n_rungs-1-w)) / n_rungs,
        #     clamp(<=1.0); slots beyond the declared front are still 1.0.
        #   declared next levels    = (rung_idx + (n_rungs-2-w)) / n_rungs,
        #     clamp(<=1.0).
        # This walks the schedule from "all slots at 1.0" down to the
        # canonical staircase [1/K, 2/K, ..., 1.0] over n_rungs-1 steps.
        # On the final canonical step (committing tick 0), the first
        # ``inference_step`` call then advances by one more unit, which is
        # the correct first-action commit.
        if n_rungs > 1:
            self._ar_warmup(device, external_cond=external_cond, n_rungs=n_rungs)

    @torch.no_grad()
    def _ar_warmup(self, device, external_cond, n_rungs: int):
        """Run ``n_rungs - 1`` denoise steps to fill the buffer to the
        canonical staircase [1/K, 2/K, ..., 1.0] starting from ``randn``."""
        tok_idx = torch.arange(self.action_horizon, device=device).float()
        rung_idx = tok_idx // self.ar_inference_chunk_size  # 0..n_rungs-1
        for w in range(n_rungs - 1):
            # At warmup step w (0-indexed, 0..n_rungs-2):
            #   slot i is at rung_idx[i] + (n_rungs - w) entering this step,
            #   clamped to n_rungs (= level 1.0 ceiling). So at w=0 every
            #   slot starts at the full-noise ceiling — exactly matching the
            #   ``randn`` buffer — and slot 0 then denoises by one rung.
            #   After this step, slot 0 has gone (n_rungs - w)/K -> (n_rungs - w - 1)/K
            #   (for w=0: 1.0 -> (K-1)/K).
            shift_cur = float(n_rungs - w)
            shift_nxt = float(n_rungs - 1 - w)
            cur_rung = (rung_idx + shift_cur).clamp(max=float(n_rungs))
            nxt_rung = (rung_idx + shift_nxt).clamp(max=float(n_rungs))
            if self._ar_discrete:
                cur_levels = (cur_rung * self._ar_unit).long().clamp(
                    -1, self.diffusion.timesteps - 1
                )
                nxt_levels = (nxt_rung * self._ar_unit).long().clamp(
                    -1, self.diffusion.timesteps - 1
                )
            else:
                cur_levels = (cur_rung * self._ar_unit).clamp(0.0, 1.0)
                nxt_levels = (nxt_rung * self._ar_unit).clamp(0.0, 1.0)
            self._ar_buffer = sample_step(
                self.diffusion,
                self.backbone,
                x=self._ar_buffer,
                current_levels=cur_levels.unsqueeze(0),
                next_levels=nxt_levels.unsqueeze(0),
                external_cond=external_cond,
                cfg_scale=self.cfg_scale,
            )

    def _ar_levels(self, offset: float, device) -> torch.Tensor:
        """Per-token noise levels for the staircase at rung-offset ``offset``.
        ``offset=0`` = current; ``offset=1`` = after one rung advance.
        Tokens within the same chunk share a rung's level."""
        tok_idx = torch.arange(self.action_horizon, device=device).float()
        rung_idx = (tok_idx // self.ar_inference_chunk_size) + 1.0 - offset
        if self._ar_discrete:
            levels = (rung_idx * self._ar_unit).long().clamp(
                -1, self.diffusion.timesteps - 1
            )
        else:
            levels = (rung_idx * self._ar_unit).clamp(0.0, 1.0)
        return levels.unsqueeze(0)  # (1, action_horizon)

    @torch.no_grad()
    @torch.no_grad()
    def _inference_step_pixel_policy(self, obs_zarr, t, emb_id):
        """Closed-loop controller for the PIXEL obs+action policy. The action
        rides as broadcast channels inside the diffused frame. Pin the last
        n_context OBSERVED frames (real RGB + executed-action planes) clean,
        denoise the next k frames' [RGB + action] jointly, read the committed
        frame's action planes by global-avg-pool -> action. Receding horizon."""
        from egomimic.models.diffusion.sampling import vanilla_schedule
        import numpy as _np

        outer = self.outer_stage
        diff = self.diffusion
        device = next(self.backbone.parameters()).device
        ac_key = self.ac_keys[get_embodiment(emb_id).lower()]
        n_ctx = max(1, int(getattr(self, "sp_n_context", 1)))
        k = max(1, int(getattr(self, "sp_commit", 1)))
        n_steps = int(self.sampler_n_steps)
        Ci = int(outer._image_channels)
        Ca = int(outer._action_channels)
        A = int(outer.action_dim)
        H = W = int(outer._image_size)
        C = Ci + Ca

        if t == 0 or not hasattr(self, "_pp_rgb"):
            self._pp_rgb = []     # observed RGB, each (Ci,H,W) in model range
            self._pp_act = []     # executed NORMALIZED actions, each (A,)
            self._pp_queue = []   # pending committed unnorm actions

        img = obs_zarr[outer.image_key].float().to(device)
        if img.max() > 1.5:
            img = img / 255.0
        if img.dim() == 4:
            img = img[0]
        # D03 seam: context pins must live in the stage's model range
        # ("01" = identity pass-through, "pm1" = [-1, 1]).
        img = outer.to_model_range(img)
        self._pp_rgb.append(img)

        if self._pp_queue:
            return self._pp_queue.pop(0)

        def act_plane(a):
            return a[:Ca].reshape(Ca, 1, 1).expand(Ca, H, W)

        T = n_ctx + k
        nrgb = len(self._pp_rgb)
        nact = len(self._pp_act)

        ctx_frames = []
        for j in range(n_ctx):
            ri = max(0, nrgb - n_ctx + j)               # most-recent n_ctx OBSERVED frames
            rgb = self._pp_rgb[ri]
            ai = nact - n_ctx + j                        # the action that produced that obs
            if 0 <= ai < nact:
                a = self._pp_act[ai]
            else:
                a = torch.zeros(A, device=device)
                try:
                    _st = obs_zarr["state_agent_obj"]
                    _st = _st[0] if _st.dim() == 2 else _st
                    _xy = _st[:A].float().to(device).unsqueeze(0)
                    a = self.norm_stats.normalize({ac_key: _xy}, emb_id)[ac_key][0][:A]
                except Exception:
                    pass
            ctx_frames.append(torch.cat([rgb, act_plane(a)], dim=0))
        ctx_stack = torch.stack(ctx_frames, dim=0).unsqueeze(0)   # (1,n_ctx,C,H,W)

        dts = int(diff.timesteps) if isinstance(diff, DiscreteDiffusion) else None
        clean = -1 if dts is not None else 0.0
        sched = vanilla_schedule(n_steps, T, discrete_timesteps=dts).to(device).clone()
        sched[:, :n_ctx] = clean

        x = torch.randn(1, T, C, H, W, device=device)
        x[:, :n_ctx] = ctx_stack
        for s in range(sched.shape[0] - 1):
            klev = sched[s].clamp_min(0).long().unsqueeze(0)
            v = self.backbone(x, klev, external_cond=None)
            x = self._struct_ddim_step(diff, x, v, sched[s], sched[s + 1], eta=float(getattr(self, "sampler_eta", 0.0)))
            x[:, :n_ctx] = ctx_stack

        pred_planes = x[0, n_ctx:n_ctx + k, Ci:Ci + Ca]          # (k,Ca,H,W)
        pred_norm = pred_planes.mean(dim=(2, 3))[:, :A]          # (k,A) global-avg-pool
        for j in range(k):
            self._pp_act.append(pred_norm[j].detach())
        unnorm = self.norm_stats.unnormalize({ac_key: pred_norm}, emb_id)[ac_key]
        unnorm_np = unnorm.detach().cpu().numpy()
        for row in unnorm_np[1:]:
            self._pp_queue.append(row.reshape(-1).astype(_np.float32))
        return unnorm_np[0].reshape(-1).astype(_np.float32)

    @torch.no_grad()
    def _inference_step_pixel_regress(self, obs_zarr, t, emb_id):
        """Closed-loop controller for Design B (regression). Pin the last
        n_context observed RGB frames clean, denoise the next k RGB frames,
        then read the action off each predicted frame via the outer stage's
        conv ``action_head``. Receding horizon."""
        from egomimic.models.diffusion.sampling import vanilla_schedule
        import numpy as _np

        outer = self.outer_stage
        diff = self.diffusion
        device = next(self.backbone.parameters()).device
        ac_key = self.ac_keys[get_embodiment(emb_id).lower()]
        n_ctx = max(1, int(getattr(self, "sp_n_context", 1)))
        k = max(1, int(getattr(self, "sp_commit", 1)))
        n_steps = int(self.sampler_n_steps)
        Ci = int(outer._image_channels)
        H = W = int(outer._image_size)

        if t == 0 or not hasattr(self, "_pr_rgb"):
            self._pr_rgb = []
            self._pr_queue = []

        img = obs_zarr[outer.image_key].float().to(device)
        if img.max() > 1.5:
            img = img / 255.0
        if img.dim() == 4:
            img = img[0]
        # D03 seam: context pins must live in the stage's model range
        # ("01" = identity pass-through, "pm1" = [-1, 1]).
        img = outer.to_model_range(img)
        self._pr_rgb.append(img)

        if self._pr_queue:
            return self._pr_queue.pop(0)

        T = n_ctx + k
        nrgb = len(self._pr_rgb)
        ctx_stack = torch.stack(
            [self._pr_rgb[max(0, nrgb - n_ctx + j)] for j in range(n_ctx)], dim=0
        ).unsqueeze(0)

        dts = int(diff.timesteps) if isinstance(diff, DiscreteDiffusion) else None
        clean = -1 if dts is not None else 0.0
        sched = vanilla_schedule(n_steps, T, discrete_timesteps=dts).to(device).clone()
        sched[:, :n_ctx] = clean

        x = torch.randn(1, T, Ci, H, W, device=device)
        x[:, :n_ctx] = ctx_stack
        for sidx in range(sched.shape[0] - 1):
            klev = sched[sidx].clamp_min(0).long().unsqueeze(0)
            v = self.backbone(x, klev, external_cond=None)
            x = self._struct_ddim_step(diff, x, v, sched[sidx], sched[sidx + 1], eta=float(getattr(self, "sampler_eta", 0.0)))
            x[:, :n_ctx] = ctx_stack

        pred_frames = x[0, n_ctx:n_ctx + k]               # (k, Ci, H, W) predicted clean frames
        pred_norm = outer.action_head(pred_frames)        # (k, A)
        unnorm = self.norm_stats.unnormalize({ac_key: pred_norm}, emb_id)[ac_key]
        unnorm_np = unnorm.detach().cpu().numpy()
        for row in unnorm_np[1:]:
            self._pr_queue.append(row.reshape(-1).astype(_np.float32))
        return unnorm_np[0].reshape(-1).astype(_np.float32)

    def _inference_step_ar(
        self, obs_zarr: dict, t: int, emb_id: int
    ) -> "np.ndarray":
        """Causal-AR rolling-staircase inference. One sample_step per env
        tick, one (or ``ar_inference_chunk_size``-many) action(s) committed
        per call. Buffer carries across calls."""
        embodiment_name = get_embodiment(emb_id).lower()
        device = next(self.backbone.parameters()).device
        ac_key = self.ac_keys[embodiment_name]

        # Encode current obs into per-call cond. Broadcast across buffer
        # tokens inside the backbone (per-token AdaLN).
        obs_norm = self.norm_stats.normalize(obs_zarr, emb_id)
        cond = self._encode_cond(obs_norm, self.action_horizon)
        if cond is not None and cond.dim() == 3:
            cond = cond[:, 0]  # (1, cond_dim) — backbone broadcasts to T

        # Reset + warm up on episode start. Warmup uses the t=0 obs cond
        # for all warmup denoise steps (no future obs available); this is
        # the correct online-AR semantics — every future env tick uses its
        # own obs.
        if t == 0 or not hasattr(self, "_ar_buffer"):
            self._ar_state_init(device, external_cond=cond)

        # If we already have committed actions ready, just pop one.
        if self._sim_committed_queue:
            return self._sim_committed_queue.pop(0)

        # One denoise step on the buffer.
        cur_levels = self._ar_levels(offset=0.0, device=device)
        nxt_levels = self._ar_levels(offset=1.0, device=device)
        self._ar_buffer = sample_step(
            self.diffusion,
            self.backbone,
            x=self._ar_buffer,
            current_levels=cur_levels,
            next_levels=nxt_levels,
            external_cond=cond,
            cfg_scale=self.cfg_scale,
        )

        # Commit front rung, slide buffer, push fresh noisy rung at the back.
        chunk = self.ar_inference_chunk_size
        committed_norm = self._ar_buffer[:, :chunk, :].clone()  # (1, chunk, bundle_dim)
        new_back = torch.randn(
            1, chunk, self.outer_stage.bundle_dim,
            device=device, dtype=torch.float32,
        )
        self._ar_buffer = torch.cat(
            [self._ar_buffer[:, chunk:, :], new_back], dim=1
        )

        # For joint obs+action bundles, slice out the action portion before
        # returning to the env. For vanilla DFoT this is a no-op (action_slice
        # spans the full trailing dim).
        committed_actions = committed_norm[..., self.outer_stage.action_slice]
        committed_world = self.norm_stats.unnormalize(
            {ac_key: committed_actions.squeeze(0)}, emb_id
        )[ac_key]
        committed_np = committed_world.detach().cpu().numpy()
        for row in committed_np[1:]:
            self._sim_committed_queue.append(row.reshape(-1).astype(np.float32))
        return committed_np[0].reshape(-1).astype(np.float32)

    @torch.no_grad()
    def _inference_step_chunk(
        self, obs_zarr: dict, t: int, emb_id: int
    ) -> "np.ndarray":
        """Legacy chunk-replan inference (uniform per-token noise, plan +
        execute action_horizon steps before replanning). Cheaper but does
        NOT exercise DFoT's per-token-noise capability."""
        embodiment_name = get_embodiment(emb_id).lower()
        device = next(self.backbone.parameters()).device
        ac_key = self.ac_keys[embodiment_name]
        if t == 0:
            self._sim_state = {"chunk": None, "chunk_idx": 0}
        state = self._sim_state

        if state["chunk"] is None or state["chunk_idx"] >= self.action_horizon:
            obs_norm = self.norm_stats.normalize(obs_zarr, emb_id)
            cond = self._encode_cond(obs_norm, self.action_horizon)
            sampled = self._sample_chunk(
                B=1, T=self.action_horizon, cond=cond, device=device
            )
            # Slice action portion out of the (potentially joint) bundle
            # before unnormalizing — channel axis + avg-pool decode for 5-D
            # pixel bundles (D02). For vanilla DFoT action_slice is full.
            sampled_actions = self._extract_bundle_actions(
                sampled,
                self.outer_stage.action_slice,
                action_dim=int(
                    getattr(self.outer_stage, "action_dim", self.action_dim)
                ),
            )
            chunk_world = self.norm_stats.unnormalize(
                {ac_key: sampled_actions.squeeze(0)}, emb_id
            )[ac_key]
            state["chunk"] = chunk_world.detach()
            state["chunk_idx"] = 0

        idx = state["chunk_idx"]
        action_world = state["chunk"][idx]
        state["chunk_idx"] = idx + 1
        return action_world.cpu().numpy().reshape(-1).astype(np.float32)

    # ------------------------------------------------------------------ #
    # Closed-loop controller for the 2D spatial obs+action policy.
    # ------------------------------------------------------------------ #
    def _struct_ddim_step(self, diff, x, v, cur, nxt, eta: float = 0.0):
        """One DDIM step from a v-prediction (discrete diffusion).
        ``cur``/``nxt`` are (T,) per-token levels; ``x`` is (1, T, *trailing).
        At ``eta=0`` (default) mirrors DFoTPolicyActionEval._ddim_from_v
        exactly; ``eta>0`` adds the stochastic DDIM noise term (D27 — the
        validated pureDF runs sample every rollout/val path at eta=1.0)."""
        T = x.shape[1]
        pad = x.dim() - 2
        kBT = cur.clamp_min(0).long().unsqueeze(0)
        x0 = diff.predict_start_from_v(x, kBT, v)
        eps = diff.predict_noise_from_v(x, kBT, v)
        an = diff.alphas_cumprod[nxt.clamp_min(0).long()]
        an = torch.where(nxt < 0, torch.ones_like(an), an)
        an = an.reshape(1, T, *([1] * pad))
        if eta <= 0.0:
            c = (1.0 - an).clamp_min(0.0).sqrt()
            return x0 * an.sqrt() + eps * c
        ac = diff.alphas_cumprod[cur.clamp_min(0).long()]
        ac = ac.reshape(1, T, *([1] * pad))
        sigma = eta * ((1.0 - an) / (1.0 - ac).clamp_min(1e-12)).sqrt() \
            * (1.0 - ac / an.clamp_min(1e-12)).clamp_min(0.0).sqrt()
        c = (1.0 - an - sigma**2).clamp_min(0.0).sqrt()
        z = torch.randn_like(x).clamp_(-diff.clip_noise, diff.clip_noise)
        return x0 * an.sqrt() + eps * c + sigma * z

    @torch.no_grad()
    def _inference_step_pixel_decoupled(self, obs_zarr, t, emb_id):
        """Closed-loop controller for the PIXEL DECOUPLED-action policy, with
        optional CHUNKING (sp_commit>1). Pin the current RGB (+ last n_context)
        CLEAN; for a chunk, append (sp_commit-1) FUTURE frames whose obs AND
        action are denoised (model predicts future obs as a world-model), read
        the action at the current frame + the future frames, commit the whole
        chunk open-loop. Action never a clean input (no copy), no offset.
        sp_n_samples averages K diffusion samples (variance reduction)."""
        from egomimic.models.diffusion.sampling import vanilla_schedule

        outer = self.outer_stage
        diff = self.diffusion
        device = next(self.backbone.parameters()).device
        ac_key = self.ac_keys[get_embodiment(emb_id).lower()]
        n_ctx = max(1, int(getattr(self, "sp_n_context", 1)))
        k = max(1, int(getattr(self, "sp_commit", 1)))
        n_steps = int(self.sampler_n_steps)
        n_samp = max(1, int(getattr(self, "sp_n_samples", 1)))
        A = int(outer.action_dim)
        Ci = int(outer._image_channels)
        H = W = int(outer._image_size)

        if t == 0 or not hasattr(self, "_pd_rgb"):
            self._pd_rgb = []
            self._pd_queue = []

        img = obs_zarr[outer.image_key].float().to(device)
        if img.max() > 1.5:
            img = img / 255.0
        if img.dim() == 4:
            img = img[0]
        # D03 seam: context pins must live in the stage's model range
        # ("01" = identity pass-through, "pm1" = [-1, 1]).
        img = outer.to_model_range(img)
        self._pd_rgb.append(img)

        if self._pd_queue:
            return self._pd_queue.pop(0)

        L = len(self._pd_rgb)
        idx = [max(0, L - n_ctx + i) for i in range(n_ctx)]
        rgb_ctx = torch.stack([self._pd_rgb[i] for i in idx], dim=0).unsqueeze(0)

        T = n_ctx + (k - 1)
        dts = int(diff.timesteps) if isinstance(diff, DiscreteDiffusion) else None
        sched = vanilla_schedule(n_steps, T, discrete_timesteps=dts).to(device)
        obs_sched = sched.clone()
        obs_sched[:, :n_ctx] = 0  # context obs CLEAN (re-pinned each step)

        ctx_b = rgb_ctx.expand(n_samp, -1, -1, -1, -1)
        x_obs = torch.randn(n_samp, T, Ci, H, W, device=device)
        x_obs[:, :n_ctx] = ctx_b
        x_act = torch.randn(n_samp, T, A, device=device)
        for st in range(sched.shape[0] - 1):
            o_lev = obs_sched[st].unsqueeze(0).expand(n_samp, -1)
            a_lev = sched[st].unsqueeze(0).expand(n_samp, -1)
            v_img, v_act = self.backbone(
                x_obs, o_lev, external_cond=None, action=x_act, action_noise_levels=a_lev,
            )
            x_obs = self._struct_ddim_step(diff, x_obs, v_img, obs_sched[st], obs_sched[st + 1], eta=float(getattr(self, "sampler_eta", 0.0)))
            x_obs[:, :n_ctx] = ctx_b
            x_act = self._struct_ddim_step(diff, x_act, v_act, sched[st], sched[st + 1], eta=float(getattr(self, "sampler_eta", 0.0)))

        chunk = x_act[:, n_ctx - 1:].mean(0)  # (k, A): current + (k-1) future actions
        unnorm = self.norm_stats.unnormalize({ac_key: chunk}, emb_id)[ac_key]
        unnorm_np = unnorm.detach().cpu().numpy()
        for row in unnorm_np[1:]:
            self._pd_queue.append(row.reshape(-1).astype(np.float32))
        return unnorm_np[0].reshape(-1).astype(np.float32)

    def _inference_step_spatial_rh(
        self, obs_zarr: dict, t: int, emb_id: int
    ) -> "np.ndarray":
        """Reactive receding-horizon controller for the 2D spatial obs+action
        policy (``SpatialObsActionPolicyDFoTOuterStage`` + dual-stream DiT3D).

        The backbone uses ONE noise level per frame (shared by the latent and
        its action token), so we cannot pin the current image clean while
        predicting its action. Instead: pin the last ``n_context`` OBSERVED
        frames (real latent + executed action) clean, condition on the REAL
        current state via external_cond, and predict the next frame's
        (imagined latent, action); commit the action. Real observations enter
        as clean context one tick later. This is the closed-loop form of the
        validated DFoTPolicyActionEval._rollout.
        """
        from egomimic.models.diffusion.sampling import vanilla_schedule

        outer = self.outer_stage
        diff = self.diffusion
        device = next(self.backbone.parameters()).device
        embodiment_name = get_embodiment(emb_id).lower()
        ac_key = self.ac_keys[embodiment_name]
        n_ctx = int(self.sp_n_context)
        k = max(1, int(self.sp_commit))                         # actions committed per replan
        n_steps = int(self.sampler_n_steps)

        if t == 0 or not hasattr(self, "_sp_lat"):
            self._sp_lat = []      # observed clean latents, each (1, C, H, W)
            self._sp_state = []    # observed states, each (1, sd) normalized
            self._sp_act = []      # executed NORMALIZED actions, each (A,)
            self._sp_queue = []    # pending committed unnorm actions (np, A)

        # ---- encode current obs (becomes a clean context frame next tick) ----
        obs_norm = self.norm_stats.normalize(obs_zarr, emb_id)
        state = torch.cat(
            [obs_norm[kk] for kk in outer.bundle_obs_keys], dim=-1
        ).float().to(device)                                    # (1, sd)
        img = obs_zarr[outer.image_key].float().to(device)
        if img.max() > 1.5:
            img = img / 255.0
        mu, _ = outer.vae.encode(img)                           # (1, C, H, W)
        lat = outer.normalize_latent(mu)                        # (1, C, H, W)
        self._sp_lat.append(lat)
        self._sp_state.append(state)

        # Mid-chunk: return the next already-committed action (no replan).
        if self._sp_queue:
            return self._sp_queue.pop(0)

        # ---- replan: predict the next k (latent, action) frames ----
        C, H, W = outer.bundle_shape
        A = outer._action_dim
        n_done = len(self._sp_act)                              # executed actions so far
        ctx_idx = [max(0, n_done - n_ctx + i) for i in range(n_ctx)]
        T = n_ctx + k

        latent_ctx = torch.cat([self._sp_lat[i] for i in ctx_idx], dim=0)  # (n_ctx, C, H, W)
        # cond: context-frame states + the REAL current state for each predicted frame
        state_seq = [self._sp_state[i] for i in ctx_idx] + [self._sp_state[-1]] * k
        cond = outer.state_only_proj(torch.cat(state_seq, dim=0)).unsqueeze(0)  # (1, T, proj)
        act_ctx = torch.stack(
            [self._sp_act[i] if i < len(self._sp_act) else torch.zeros(A, device=device)
             for i in ctx_idx], dim=0)                          # (n_ctx, A)

        dts = int(diff.timesteps) if isinstance(diff, DiscreteDiffusion) else None
        clean = -1 if dts is not None else 0.0
        sched = vanilla_schedule(n_steps, T, discrete_timesteps=dts).to(device).clone()
        sched[:, :n_ctx] = clean

        x_lat = torch.randn(1, T, C, H, W, device=device)
        x_lat[:, :n_ctx] = latent_ctx.unsqueeze(0)
        x_act = torch.randn(1, T, A, device=device)
        x_act[:, :n_ctx] = act_ctx.unsqueeze(0)
        for s in range(sched.shape[0] - 1):
            klev = sched[s].clamp_min(0).long().unsqueeze(0)
            v_lat, v_act = self.backbone(x_lat, klev, external_cond=cond, action=x_act)
            x_lat = self._struct_ddim_step(diff, x_lat, v_lat, sched[s], sched[s + 1], eta=float(getattr(self, "sampler_eta", 0.0)))
            x_act = self._struct_ddim_step(diff, x_act, v_act, sched[s], sched[s + 1], eta=float(getattr(self, "sampler_eta", 0.0)))
            x_lat[:, :n_ctx] = latent_ctx.unsqueeze(0)
            x_act[:, :n_ctx] = act_ctx.unsqueeze(0)

        pred_norm = x_act[0, n_ctx:n_ctx + k]                   # (k, A) normalized predicted actions
        for j in range(k):
            self._sp_act.append(pred_norm[j].detach())         # record executed (normalized)
        unnorm = self.norm_stats.unnormalize(
            {ac_key: pred_norm}, emb_id
        )[ac_key]                                               # (k, A) absolute frame
        unnorm_np = unnorm.detach().cpu().numpy()
        for row in unnorm_np[1:]:
            self._sp_queue.append(row.reshape(-1).astype(np.float32))
        return unnorm_np[0].reshape(-1).astype(np.float32)

    @torch.no_grad()
    def _inference_step_spatial_decoupled(
        self, obs_zarr: dict, t: int, emb_id: int
    ) -> "np.ndarray":
        """Closed-loop controller for the DECOUPLED-action policy
        (``decouple_action_noise=True``). The action token has its own noise
        level, so we CAN pin the real current image clean while predicting its
        action -> reactive, and the action is never a clean input (no copy).

        Each tick: pin the last ``n_context`` OBSERVED latents clean (incl. the
        current frame), keep ALL action tokens noised, denoise the action stream
        on its own schedule conditioned on the clean obs + state, and commit the
        action predicted at the current (last) frame. Causal: no future frames.
        """
        from egomimic.models.diffusion.sampling import vanilla_schedule

        outer = self.outer_stage
        diff = self.diffusion
        device = next(self.backbone.parameters()).device
        embodiment_name = get_embodiment(emb_id).lower()
        ac_key = self.ac_keys[embodiment_name]
        n_ctx = int(self.sp_n_context)
        n_steps = int(self.sampler_n_steps)

        if t == 0 or not hasattr(self, "_sd_lat"):
            self._sd_lat = []      # observed clean latents, each (1, C, H, W)
            self._sd_state = []    # observed states, each (1, sd) normalized

        obs_norm = self.norm_stats.normalize(obs_zarr, emb_id)
        state = torch.cat(
            [obs_norm[kk] for kk in outer.bundle_obs_keys], dim=-1
        ).float().to(device)
        img = obs_zarr[outer.image_key].float().to(device)
        if img.max() > 1.5:
            img = img / 255.0
        mu, _ = outer.vae.encode(img)
        self._sd_lat.append(outer.normalize_latent(mu))
        self._sd_state.append(state)

        C, H, W = outer.bundle_shape
        A = outer._action_dim
        L = len(self._sd_lat)
        idx = [max(0, L - n_ctx + i) for i in range(n_ctx)]   # current frame is last
        T = n_ctx
        latent_ctx = torch.cat([self._sd_lat[i] for i in idx], dim=0).unsqueeze(0)   # (1,T,C,H,W) CLEAN
        cond = outer.state_only_proj(
            torch.cat([self._sd_state[i] for i in idx], dim=0)
        ).unsqueeze(0)                                                                # (1,T,proj)

        dts = int(diff.timesteps) if isinstance(diff, DiscreteDiffusion) else None
        clean = -1 if dts is not None else 0.0
        # obs given clean for all frames; action denoised on its own schedule.
        obs_levels = torch.full(
            (1, T), clean, device=device,
            dtype=torch.long if dts is not None else torch.float32,
        )
        act_sched = vanilla_schedule(n_steps, T, discrete_timesteps=dts).to(device)

        x_lat = latent_ctx                                   # clean obs, never stepped
        x_act = torch.randn(1, T, A, device=device)
        for s in range(act_sched.shape[0] - 1):
            _, v_act = self.backbone(
                x_lat, obs_levels, external_cond=cond, action=x_act,
                action_noise_levels=act_sched[s].unsqueeze(0),
            )
            x_act = self._struct_ddim_step(diff, x_act, v_act, act_sched[s], act_sched[s + 1], eta=float(getattr(self, "sampler_eta", 0.0)))

        pred_norm = x_act[0, -1]                             # action at the current frame
        unnorm = self.norm_stats.unnormalize(
            {ac_key: pred_norm.unsqueeze(0)}, emb_id
        )[ac_key]
        return unnorm.squeeze(0).detach().cpu().numpy().reshape(-1).astype(np.float32)
