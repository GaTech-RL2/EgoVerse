"""
DFoT Algo: per-token-noise-level diffusion over action chunks.

Single-domain (PushShapes by default), per-frame loader. Each sample is a
``(B, T, action_dim)`` action chunk with single-frame obs; obs is encoded
into ``(B, cond_dim)`` via the existing ``CondEncoderModule`` and broadcast
to per-token inside the backbone. Loss is the diffusion (epsilon/v/x0)
MSE with the chosen loss-weighting strategy.
"""

from collections import OrderedDict
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from overrides import override

from egomimic.algo.algo import Algo
from egomimic.algo.dfot.backbone import DFoTBackbone
from egomimic.algo.dfot.continuous_diffusion import ContinuousDiffusion
from egomimic.algo.dfot.discrete_diffusion import DiscreteDiffusion
from egomimic.algo.dfot.sampling import ddim_sample, ddpm_sample
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
        action_horizon: chunk length T.
        cond_encoder: ``CondEncoderModule`` for obs -> ``(B, cond_dim)``.
        backbone: ``DFoTBackbone`` (built via Hydra). Owns x/cond/time
            projections and the Isotropic trunk.
        diffusion_type: ``"discrete"`` or ``"continuous"``.
        diffusion_kwargs: dict forwarded to the chosen diffusion class.
        sampler: ``"ddpm"`` or ``"ddim"``.
        sampler_n_steps: number of denoising steps at inference time.
        sampler_eta: DDIM eta (0.0 = deterministic).
        norm_stats: MultiDataset (injected by ``pl_model._instantiate_model``).
        domains: list of embodiment names (single-element for v1).
        ac_keys: dict ``embodiment_name -> action zarr key``.
        cond_output_key: key under which the cond encoder exposes its fused cond.
    """

    def __init__(
        self,
        action_dim: int,
        action_horizon: int,
        cond_encoder: CondEncoderModule,
        backbone: DFoTBackbone,
        norm_stats,
        diffusion_type: str = "continuous",
        diffusion_kwargs: Optional[dict] = None,
        sampler: str = "ddim",
        sampler_n_steps: int = 50,
        sampler_eta: float = 0.0,
        domains: Optional[list] = None,
        ac_keys: Optional[dict] = None,
        cond_output_key: str = "fused_cond",
        device=None,
        **kwargs,
    ):
        super().__init__()
        self.norm_stats = norm_stats
        self.domains = list(domains or [])
        self.ac_keys = dict(ac_keys or {})
        self.action_dim = int(action_dim)
        self.action_horizon = int(action_horizon)
        self.cond_output_key = cond_output_key
        self.sampler = sampler
        self.sampler_n_steps = int(sampler_n_steps)
        self.sampler_eta = float(sampler_eta)
        self.diffusion_type = diffusion_type
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        diffusion_kwargs = dict(diffusion_kwargs or {})
        diffusion_kwargs.setdefault("action_dim", self.action_dim)
        if diffusion_type == "discrete":
            self.diffusion = DiscreteDiffusion(**diffusion_kwargs)
        elif diffusion_type == "continuous":
            self.diffusion = ContinuousDiffusion(**diffusion_kwargs)
        else:
            raise ValueError(f"unknown diffusion_type {diffusion_type!r}")

        self.nets = nn.ModuleDict(
            {
                "backbone": backbone,
                "cond_encoder": cond_encoder,
                "diffusion": self.diffusion,
            }
        )
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
        cond_dict = self.nets["cond_encoder"].encode(obs, T)
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
        cond_dict = self.nets["cond_encoder"].encode(obs_3d, T_action=1)
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
        predictions = OrderedDict()
        backbone = self.nets["backbone"]
        for emb_id, _batch in batch.items():
            ac_key = self.resolved_ac_keys[emb_id]
            actions = _batch[ac_key]
            is_packed = _batch.get("_packed", False)
            obs = self._build_obs(_batch, emb_id)

            if is_packed:
                # actions: (T_total, action_dim)
                if actions.dim() != 2:
                    raise ValueError(
                        f"packed actions must be (T_total, action_dim); "
                        f"got {tuple(actions.shape)}"
                    )
                T_total = actions.shape[0]
                cu = _batch["cu_seqlens"]
                msl = int(_batch.get("max_seq_len", 0)) or None

                cond = self._encode_cond_packed(obs)
                k = self._sample_noise_levels((T_total,), actions.device)
                # Wrap backbone so the diffusion module's 3-arg call carries
                # the cu_seqlens/max_seqlen kwargs through to packed attention.
                packed_backbone = _PackedBackboneWrapper(backbone, cu, msl)
                _, loss = self.diffusion(packed_backbone, actions, k, external_cond=cond)
            else:
                if actions.dim() != 3:
                    raise ValueError(
                        f"padded actions must be (B, T, action_dim); got "
                        f"{tuple(actions.shape)}"
                    )
                B, T, _ = actions.shape
                cond = self._encode_cond(obs, T)
                k = self._sample_noise_levels((B, T), actions.device)
                _, loss = self.diffusion(backbone, actions, k, external_cond=cond)

            mse = loss.mean()
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
        backbone = self.nets["backbone"]
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
                _, loss = self.diffusion(packed_backbone, actions, k, external_cond=cond)
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
        shape = (B, T, self.action_dim)
        if self.sampler == "ddpm":
            return ddpm_sample(
                self.diffusion,
                self.nets["backbone"],
                shape,
                external_cond=cond,
                n_steps=self.sampler_n_steps,
                device=device,
            )
        if self.sampler == "ddim":
            return ddim_sample(
                self.diffusion,
                self.nets["backbone"],
                shape,
                external_cond=cond,
                n_steps=self.sampler_n_steps,
                eta=self.sampler_eta,
                device=device,
            )
        raise ValueError(f"unknown sampler {self.sampler!r}")

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

    # ---- Sim eval hook (HPT-style chunk sampler) ---- #

    @torch.no_grad()
    def inference_step(
        self, obs_zarr: dict, t: int, emb_id: int, T_max=None
    ) -> "np.ndarray":
        embodiment_name = get_embodiment(emb_id).lower()
        device = next(self.nets["backbone"].parameters()).device
        ac_key = self.ac_keys[embodiment_name]
        if t == 0:
            self._sim_state = {"chunk": None, "chunk_idx": 0}
        state = self._sim_state

        if state["chunk"] is None or state["chunk_idx"] >= self.action_horizon:
            # obs_zarr is already shaped per the SimRolloutEval contract:
            # state_agent_obj (1, D), front_img_1 (1, C, H, W). CondEncoderModule
            # handles the per-T_action broadcast internally — don't unsqueeze.
            obs_norm = self.norm_stats.normalize(obs_zarr, emb_id)
            cond = self._encode_cond(obs_norm, self.action_horizon)
            sampled = self._sample_chunk(
                B=1, T=self.action_horizon, cond=cond, device=device
            )
            chunk_world = self.norm_stats.unnormalize(
                {ac_key: sampled.squeeze(0)}, emb_id
            )[ac_key]
            state["chunk"] = chunk_world.detach()
            state["chunk_idx"] = 0

        idx = state["chunk_idx"]
        action_world = state["chunk"][idx]
        state["chunk_idx"] = idx + 1
        return action_world.cpu().numpy().reshape(-1).astype(np.float32)
