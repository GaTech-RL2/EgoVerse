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
from egomimic.algo.dfot.backbone import DFoTBackbone
from egomimic.algo.dfot.discrete_diffusion import DiscreteDiffusion
from egomimic.algo.dfot.outer_stage import DFoTOuterStage, make_dfot_ctx
from egomimic.algo.dfot.sampling import sample_step
from egomimic.algo.loss import DFoTLoss, Loss
from egomimic.models.hnet_nets.cond_encoders import CondEncoderModule
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
        loss: Optional[Loss] = None,
        sampler: str = "ddim",
        sampler_n_steps: int = 50,
        sampler_eta: float = 0.0,
        inference_mode: str = "ar",
        ar_inference_chunk_size: int = 1,
        ar_inference_step_size: int = 1,
        cfg_scale: float = 1.0,
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
            loss: ``Loss`` (typically ``DFoTLoss``) that consumes
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
        if inference_mode not in {"ar", "chunk"}:
            raise ValueError(
                f"inference_mode must be 'ar' or 'chunk', got {inference_mode!r}"
            )
        self.inference_mode = inference_mode
        self.ar_inference_chunk_size = int(ar_inference_chunk_size)
        # Number of `sample_step` calls per env tick. step_size>1 advances
        # noise levels by 1/(n_rungs*step_size) per sub-step. Mirrors the
        # offline staircase_ar_schedule(chunk, step) shape.
        self.ar_inference_step_size = int(ar_inference_step_size)
        # Classifier-free-guidance scale at inference. 1.0 disables CFG.
        self.cfg_scale = float(cfg_scale)
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

        # Resolve per-embodiment keys via norm_stats (HNet-style).
        self.embodiment_ids = {}
        self.proprio_keys = {}
        self.lang_keys = {}
        self.camera_keys = {}
        self.resolved_ac_keys = {}
        for emb in self.domains:
            emb_id = get_embodiment_id(emb)
            self.embodiment_ids[emb] = emb_id
            self.proprio_keys[emb_id] = []
            self.lang_keys[emb_id] = []
            self.camera_keys[emb_id] = []
            for key in norm_stats.keys_of_type("action_keys", emb_id):
                if (
                    norm_stats.is_key_with_embodiment(key, emb_id)
                    and key == self.ac_keys[emb]
                ):
                    self.resolved_ac_keys[emb_id] = key
            for key in norm_stats.keys_of_type("proprio_keys", emb_id):
                if norm_stats.is_key_with_embodiment(key, emb_id):
                    self.proprio_keys[emb_id].append(key)
            for key in norm_stats.keys_of_type("lang_keys", emb_id):
                if norm_stats.is_key_with_embodiment(key, emb_id):
                    self.lang_keys[emb_id].append(key)
            for key in norm_stats.keys_of_type("camera_keys", emb_id):
                if norm_stats.is_key_with_embodiment(key, emb_id):
                    self.camera_keys[emb_id].append(key)

    # ----- Convenience accessors so the inference-path code paths
    # (forward_eval, _sample_chunk, _inference_step_ar, _inference_step_chunk)
    # can keep referring to ``self.backbone`` / ``self.cond_encoder`` /
    # ``self.diffusion`` without going through ``self.nets["outer_stage"]``
    # each call. All three are submodules of the outer_stage.

    @property
    def outer_stage(self) -> DFoTOuterStage:
        return self.nets["outer_stage"]

    @property
    def loss(self) -> Loss:
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

    def _build_obs(self, _batch, emb_id):
        obs = {}
        for key in (
            self.proprio_keys[emb_id]
            + self.lang_keys[emb_id]
            + self.camera_keys[emb_id]
        ):
            if key in _batch:
                obs[key] = _batch[key]
        return obs

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
            k: (v.unsqueeze(0) if torch.is_tensor(v) else v) for k, v in obs.items()
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
            ac_key = self.resolved_ac_keys[emb_id]
            is_packed = _batch.get("_packed", False)
            obs = self._build_obs(_batch, emb_id)
            ctx = make_dfot_ctx(
                is_packed=is_packed,
                action_key=ac_key,
                obs=obs,
                cu_seqlens=_batch.get("cu_seqlens") if is_packed else None,
                max_seqlen=(int(_batch.get("max_seq_len", 0)) or None)
                if is_packed
                else None,
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
            ac_key = self.resolved_ac_keys[emb_id]
            actions = _batch[ac_key]
            is_packed = _batch.get("_packed", False)
            obs = self._build_obs(_batch, emb_id)

            if is_packed:
                T_total = actions.shape[0]
                cu = _batch["cu_seqlens"]
                msl = int(_batch.get("max_seq_len", 0)) or None
                cond = self._encode_cond_packed(obs)
                k = self._sample_noise_levels((T_total,), actions.device)
                packed_backbone = _PackedBackboneWrapper(backbone, cu, msl)
                _, loss = self.diffusion(
                    packed_backbone, actions, k, external_cond=cond
                )
                unnorm[f"emb{emb_id}_loss"] = loss.mean()
                # No chunk sampling in packed val — too expensive per episode and
                # the closed-loop measure lives in inference_step / sim eval.
            else:
                B, T, _ = actions.shape
                cond = self._encode_cond(obs, T)
                k = self._sample_noise_levels((B, T), actions.device)
                _, loss = self.diffusion(backbone, actions, k, external_cond=cond)
                unnorm[f"emb{emb_id}_loss"] = loss.mean()
                sampled = self._sample_chunk(B, T, cond=cond, device=actions.device)
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
        bundle_dim = self.outer_stage.bundle_dim
        if self.sampler not in ("ddpm", "ddim"):
            raise ValueError(f"unknown sampler {self.sampler!r}")
        discrete_ts = (
            int(self.diffusion.timesteps)
            if isinstance(self.diffusion, DiscreteDiffusion)
            else None
        )
        n_steps = (
            self.sampler_n_steps
            if self.sampler == "ddim"
            else (discrete_ts or self.sampler_n_steps)
        )
        from egomimic.algo.dfot.sampling import sample as _sample
        from egomimic.algo.dfot.sampling import vanilla_schedule

        sm = vanilla_schedule(n_steps=n_steps, T=T, discrete_timesteps=discrete_ts)
        eta = 1.0 if self.sampler == "ddpm" else self.sampler_eta
        return _sample(
            self.diffusion,
            self.backbone,
            schedule_matrix=sm,
            action_dim=bundle_dim,
            batch_size=B,
            external_cond=cond,
            eta=eta,
            cfg_scale=self.cfg_scale,
            device=device,
        )

    @override
    def compute_losses(self, predictions, batch):
        total = torch.tensor(0.0, device=self.device)
        loss_dict = OrderedDict()
        for emb_id in batch.keys():
            a = predictions[f"{emb_id}_action_loss"]
            loss_dict[f"emb{emb_id}_action_loss"] = a
            total = total + a
        loss_dict["action_loss"] = total / max(len(batch), 1)
        return loss_dict

    @override
    def log_info(self, info):
        log = OrderedDict()
        log["Loss"] = info["losses"]["action_loss"].item()
        for k, v in info["losses"].items():
            log[k] = v.item()
        return log

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
            1,
            self.action_horizon,
            self.outer_stage.bundle_dim,
            device=device,
            dtype=torch.float32,
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
                cur_levels = (
                    (cur_rung * self._ar_unit)
                    .long()
                    .clamp(-1, self.diffusion.timesteps - 1)
                )
                nxt_levels = (
                    (nxt_rung * self._ar_unit)
                    .long()
                    .clamp(-1, self.diffusion.timesteps - 1)
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
            levels = (
                (rung_idx * self._ar_unit)
                .long()
                .clamp(-1, self.diffusion.timesteps - 1)
            )
        else:
            levels = (rung_idx * self._ar_unit).clamp(0.0, 1.0)
        return levels.unsqueeze(0)  # (1, action_horizon)

    @torch.no_grad()
    def _inference_step_ar(self, obs_zarr: dict, t: int, emb_id: int) -> "np.ndarray":
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
            1,
            chunk,
            self.outer_stage.bundle_dim,
            device=device,
            dtype=torch.float32,
        )
        self._ar_buffer = torch.cat([self._ar_buffer[:, chunk:, :], new_back], dim=1)

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
            # before unnormalizing. For vanilla DFoT action_slice is full.
            sampled_actions = sampled[..., self.outer_stage.action_slice]
            chunk_world = self.norm_stats.unnormalize(
                {ac_key: sampled_actions.squeeze(0)}, emb_id
            )[ac_key]
            state["chunk"] = chunk_world.detach()
            state["chunk_idx"] = 0

        idx = state["chunk_idx"]
        action_world = state["chunk"][idx]
        state["chunk_idx"] = idx + 1
        return action_world.cpu().numpy().reshape(-1).astype(np.float32)
