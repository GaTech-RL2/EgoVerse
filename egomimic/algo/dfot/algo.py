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

    # ---- Algo API -------------------------------------------------------- #

    @override
    def process_batch_for_training(self, batch):
        """Per-frame batches only (no packed mode for v1)."""
        processed = {}
        for emb_name, _batch in batch.items():
            emb_id = get_embodiment_id(emb_name)
            processed[emb_id] = {}
            for key, value in _batch.items():
                key_name = self.norm_stats.zarr_key_to_keyname(key, emb_id)
                if key_name is not None:
                    processed[emb_id][key_name] = value
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
        cond_dict = self.nets["cond_encoder"].encode(obs, T)
        if self.cond_output_key not in cond_dict:
            return None
        # Cond encoder emits (B, T, d_cond); reduce to (B, d_cond) by taking
        # the first frame since obs is single-frame for DFoT v1. The backbone
        # then broadcasts back to per-token inside.
        c = cond_dict[self.cond_output_key]
        if c.dim() == 3:
            c = c[:, 0]
        return c

    def _sample_noise_levels(self, B: int, T: int, device) -> torch.Tensor:
        """Per-token random noise level. Discrete -> longs in [0, timesteps);
        continuous -> floats in (0, 1)."""
        if isinstance(self.diffusion, DiscreteDiffusion):
            return torch.randint(
                0, self.diffusion.timesteps, (B, T), device=device, dtype=torch.long
            )
        # continuous
        return torch.rand((B, T), device=device).clamp_(1e-5, 1.0 - 1e-5)

    @override
    def forward_training(self, batch):
        predictions = OrderedDict()
        backbone = self.nets["backbone"]
        for emb_id, _batch in batch.items():
            ac_key = self.resolved_ac_keys[emb_id]
            actions = _batch[ac_key]  # (B, T, action_dim)
            if actions.dim() != 3:
                raise ValueError(
                    f"DFoT expects per-frame action chunks (B, T, action_dim); "
                    f"got shape {tuple(actions.shape)}"
                )
            B, T, _ = actions.shape
            obs = self._build_obs(_batch, emb_id)
            cond = self._encode_cond(obs, T)
            k = self._sample_noise_levels(B, T, actions.device)
            _, loss = self.diffusion(backbone, actions, k, external_cond=cond)
            mse = loss.mean()
            predictions[f"{emb_id}_action_loss"] = mse
        return predictions

    @override
    def forward_eval(self, batch):
        unnorm = {}
        backbone = self.nets["backbone"]
        for emb_id, _batch in batch.items():
            ac_key = self.resolved_ac_keys[emb_id]
            actions = _batch[ac_key]
            B, T, _ = actions.shape
            obs = self._build_obs(_batch, emb_id)
            cond = self._encode_cond(obs, T)
            # Val loss at a random noise level (same path as training).
            k = self._sample_noise_levels(B, T, actions.device)
            _, loss = self.diffusion(backbone, actions, k, external_cond=cond)
            unnorm[f"emb{emb_id}_loss"] = loss.mean()
            # Sampled actions.
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
        ac_key = (
            self.ac_keys[embodiment_name]
            if embodiment_name in self.ac_keys
            else self.ac_keys[emb_id]
        )
        if t == 0:
            self._sim_state = {"chunk": None, "chunk_idx": 0}
        state = self._sim_state

        if state["chunk"] is None or state["chunk_idx"] >= self.action_horizon:
            obs_norm = self.norm_stats.normalize(obs_zarr, emb_id)
            # Add batch dim where missing.
            obs_b = {
                k: (v.unsqueeze(0) if torch.is_tensor(v) and v.dim() < 4 else v)
                for k, v in obs_norm.items()
            }
            cond = self._encode_cond(obs_b, self.action_horizon)
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
