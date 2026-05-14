"""
H-Net policy for EgoVerse.

The policy treats the action sequence as the input modality (autoregressive,
causal). Observations (proprio + optional pre-encoded image features) are
fused into a single conditioning vector that modulates every boundary token
inside the H-Net via AdaLN.

Loss = action MSE (next-action prediction) + ratio_loss_weight * ratio_loss.
"""
from collections import OrderedDict
from typing import List, Optional

import torch
import torch.nn as nn
from overrides import override

from egomimic.algo.algo import Algo
from egomimic.models.hnet_nets import HNet as HNetCore
from egomimic.models.hnet_nets.config import HNetConfig
from egomimic.models.hnet_nets.hnet import ratio_loss
from egomimic.rldb.embodiment.embodiment import get_embodiment_id


def _mlp(in_dim: int, out_dim: int, widths: Optional[List[int]] = None) -> nn.Sequential:
    widths = widths or []
    layers: List[nn.Module] = []
    prev = in_dim
    for w in widths:
        layers += [nn.Linear(prev, w), nn.GELU()]
        prev = w
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class HNetPolicy(nn.Module):
    """
    action-tokenizer -> H-Net (causal, AdaLN on encoder) -> action-detokenizer.

    Per-frame conditioning: each obs/image key is encoded to (B, T, embed_dim),
    concatenated across keys, then fused to (B, T, d_cond). The H-Net encoder's
    AdaLN modulates action token t with the corresponding obs at time t.

    Obs encoders (`obs_specs`) are MLPs along the last dim of (B, T, D).
    Image encoders (`img_encoders`) are arbitrary nn.Modules instantiated via
    Hydra; they must accept (..., C, H, W) and expose `.embed_dim`. The bundled
    egomimic.models.hnet_nets.image_encoders.SimpleConv satisfies both.

    Shape contract for `obs` at runtime: every key must be (B, T, ...) where
    T == action_horizon. Single-frame obs (B, ...) is broadcast across T.
    """

    def __init__(
        self,
        action_dim: int,
        action_horizon: int,
        d_model: int,
        d_cond: int,
        hnet_config: HNetConfig,
        obs_specs: Optional[dict] = None,
        img_encoders: Optional[dict] = None,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.d_model = d_model
        self.d_cond = d_cond

        self.action_in = nn.Linear(action_dim, d_model)
        self.action_out = nn.Linear(d_model, action_dim)
        self.bos = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.bos, std=0.02)
        self.pos_emb = nn.Parameter(torch.zeros(1, action_horizon, d_model))
        nn.init.normal_(self.pos_emb, std=0.02)

        # MLP-based obs encoders (proprio, language features, etc.).
        obs_specs = obs_specs or {}
        self.obs_keys = sorted(obs_specs.keys())
        self.obs_encoders = nn.ModuleDict()
        fused_dim = 0
        for key in self.obs_keys:
            spec = obs_specs[key]
            self.obs_encoders[key] = _mlp(
                spec["input_dim"], spec["embed_dim"], spec.get("widths", [])
            )
            fused_dim += spec["embed_dim"]

        # Image encoders: Hydra-instantiated modules. Each must expose embed_dim.
        img_encoders = img_encoders or {}
        self.img_keys = sorted(img_encoders.keys())
        self.img_encoders = nn.ModuleDict({k: img_encoders[k] for k in self.img_keys})
        for key in self.img_keys:
            if not hasattr(self.img_encoders[key], "embed_dim"):
                raise AttributeError(
                    f"img_encoders['{key}'] must expose `.embed_dim` (saw "
                    f"{type(self.img_encoders[key]).__name__})."
                )
            fused_dim += self.img_encoders[key].embed_dim

        if fused_dim == 0:
            self.cond_proj = None
        else:
            self.cond_proj = _mlp(fused_dim, d_cond, widths=[max(d_cond, fused_dim)])

        self.hnet = HNetCore(hnet_config, stage_idx=0, d_cond=d_cond, causal=True)

    def _encode_cond(self, obs: dict) -> Optional[torch.Tensor]:
        """Returns per-frame cond of shape (B, T, d_cond), or None if no obs."""
        if self.cond_proj is None:
            return None
        T = self.action_horizon
        feats: List[torch.Tensor] = []

        for key in self.obs_keys:
            if key not in obs:
                continue
            x = obs[key]                          # expected (B, T, D) or (B, D)
            if x.dim() == 2:                      # broadcast single-frame across T
                x = x.unsqueeze(1).expand(-1, T, -1)
            feats.append(self.obs_encoders[key](x))     # (B, T, embed_dim)

        for key in self.img_keys:
            if key not in obs:
                continue
            x = obs[key]                          # expected (B, T, C, H, W) or (B, C, H, W)
            if x.dim() == 4:                      # broadcast single-frame across T
                x = x.unsqueeze(1).expand(-1, T, -1, -1, -1)
            feats.append(self.img_encoders[key](x))     # (B, T, embed_dim)

        if not feats:
            return None
        fused = torch.cat(feats, dim=-1)          # (B, T, sum_embed)
        return self.cond_proj(fused)              # (B, T, d_cond)

    def forward(self, actions: torch.Tensor, obs: dict):
        """
        actions: (B, T, action_dim) ground-truth actions for teacher-forcing.
                 Pass zeros at eval-time if you want unconditional rollout (or
                 use `generate` for autoregressive sampling).
        obs:     dict of (B, ...) obs tensors keyed by name.
        Returns: (pred_actions (B, T, action_dim), routing_outputs, masks)
        """
        B, T, _ = actions.shape
        # Teacher-forcing: input = [BOS, a_0, ..., a_{T-1}], length T
        x = self.action_in(actions)
        x = torch.cat([self.bos.expand(B, -1, -1), x[:, :-1]], dim=1)
        x = x + self.pos_emb[:, :T]

        cond = self._encode_cond(obs)
        mask = torch.ones(B, T, dtype=torch.bool, device=actions.device)
        h, bpred_outputs, structs = self.hnet(x, mask=mask, cond=cond)
        return self.action_out(h), bpred_outputs, structs

    @torch.no_grad()
    def generate(self, obs: dict, batch_size: int, device) -> torch.Tensor:
        """Autoregressive rollout from BOS for `action_horizon` steps using
        the H-Net's KV-cached single-token `step` path (O(T * model) instead
        of O(T^2 * model))."""
        T = self.action_horizon
        cond_full = self._encode_cond(obs)        # (B, T, d_cond) or None
        actions = torch.zeros(batch_size, T, self.action_dim, device=device)

        dtype = next(self.parameters()).dtype
        inference_params = self.hnet.allocate_inference_cache(
            batch_size=batch_size, max_seqlen=T, device=device, dtype=dtype
        )

        # First token = BOS
        cur = self.bos.expand(batch_size, -1, -1) + self.pos_emb[:, 0:1]
        for t in range(T):
            # Per-step cond slice (B, d_cond) -- AdaLN unsqueezes to broadcast
            # over the single-token sequence dim inside the encoder.
            cond_t = cond_full[:, t] if cond_full is not None else None
            h, _ = self.hnet.step(cur, inference_params, cond=cond_t)
            a_t = self.action_out(h)              # (B, 1, action_dim)
            actions[:, t : t + 1] = a_t
            if t < T - 1:
                cur = self.action_in(a_t) + self.pos_emb[:, t + 1 : t + 2]
        return actions


class HNet(Algo):
    """
    H-Net policy Algo. Single-domain action-sequence model with **per-frame**
    obs conditioning -- each action token sees the obs at its own timestep.

    YAML knobs:
        action_dim, action_horizon, d_model, d_cond
        hnet_config:     {arch_layout, d_model, d_intermediate, attn_cfg,
                          ssm_cfg, target_compression_ratios}
        obs_specs:       {<key>: {input_dim, embed_dim, widths}}  -- MLP encoders
        img_encoders:    {<key>: nn.Module with .embed_dim}        -- Hydra-instantiated
        ratio_loss_weight
        domains, ac_keys
    """

    def __init__(
        self,
        data_schematic,
        action_dim: int,
        action_horizon: int,
        d_model: int,
        d_cond: int,
        hnet_config: dict,
        obs_specs: dict = None,
        img_encoders: dict = None,
        ratio_loss_weight: float = 0.03,
        domains: list = None,
        ac_keys: dict = None,
        device=None,
        **kwargs,
    ):
        super().__init__()
        self.data_schematic = data_schematic
        self.domains = list(domains or [])
        self.ac_keys = dict(ac_keys or {})
        self.ratio_loss_weight = ratio_loss_weight
        self.action_horizon = action_horizon
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        cfg = HNetConfig.from_dict(hnet_config)
        self.target_ratios = list(cfg.target_compression_ratios)
        policy = HNetPolicy(
            action_dim=action_dim,
            action_horizon=action_horizon,
            d_model=d_model,
            d_cond=d_cond,
            hnet_config=cfg,
            obs_specs=obs_specs or {},
            img_encoders=img_encoders or {},
        )
        self.nets = nn.ModuleDict({"policy": policy})
        self.nets = self.nets.float().to(self.device)

        # Resolve per-embodiment keys via data_schematic (like HPT)
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
            for key in data_schematic.keys_of_type("action_keys", emb_id):
                if (
                    data_schematic.is_key_with_embodiment(key, emb_id)
                    and key == self.ac_keys[emb]
                ):
                    self.resolved_ac_keys[emb_id] = key
            for key in data_schematic.keys_of_type("proprio_keys", emb_id):
                if data_schematic.is_key_with_embodiment(key, emb_id):
                    self.proprio_keys[emb_id].append(key)
            for key in data_schematic.keys_of_type("lang_keys", emb_id):
                if data_schematic.is_key_with_embodiment(key, emb_id):
                    self.lang_keys[emb_id].append(key)
            for key in data_schematic.keys_of_type("camera_keys", emb_id):
                if data_schematic.is_key_with_embodiment(key, emb_id):
                    self.camera_keys[emb_id].append(key)

    # ---- Algo API --------------------------------------------------------

    @override
    def process_batch_for_training(self, batch):
        processed = {}
        for emb_name, _batch in batch.items():
            emb_id = get_embodiment_id(emb_name)
            processed[emb_id] = {}
            for key, value in _batch.items():
                key_name = self.data_schematic.zarr_key_to_keyname(key, emb_id)
                if key is not None:
                    processed[emb_id][key_name] = value

            ac_key = self.resolved_ac_keys[emb_id]
            B, S, _ = processed[emb_id][ac_key].shape
            processed[emb_id]["pad_mask"] = torch.ones(
                B, S, 1, device=processed[emb_id][ac_key].device
            )
            processed[emb_id] = self.data_schematic.normalize_data(processed[emb_id], emb_id)
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

    @override
    def forward_training(self, batch):
        predictions = OrderedDict()
        policy = self.nets["policy"]
        for emb_id, _batch in batch.items():
            ac_key = self.resolved_ac_keys[emb_id]
            actions = _batch[ac_key]  # (B, T, action_dim)
            obs = self._build_obs(_batch, emb_id)

            pred, bpred_outputs, structs = policy(actions, obs)
            mse = nn.functional.mse_loss(pred, actions)
            rloss = ratio_loss(bpred_outputs, structs, self.target_ratios).to(mse.device)
            predictions[f"{emb_id}_pred"] = pred
            predictions[f"{emb_id}_action_loss"] = mse
            predictions[f"{emb_id}_ratio_loss"] = rloss
        return predictions

    @override
    def forward_eval(self, batch):
        unnorm = {}
        policy = self.nets["policy"]
        for emb_id, _batch in batch.items():
            ac_key = self.resolved_ac_keys[emb_id]
            obs = self._build_obs(_batch, emb_id)
            B = next(iter(obs.values())).shape[0] if obs else _batch[ac_key].shape[0]
            actions = policy.generate(obs, batch_size=B, device=self.device)
            preds = OrderedDict()
            preds[ac_key] = actions
            unnorm_actions = self.data_schematic.unnormalize_data(preds, emb_id)
            for key, val in unnorm_actions.items():
                unnorm[f"emb{emb_id}_{key}"] = val
        return unnorm

    @override
    def compute_losses(self, predictions, batch):
        total = torch.tensor(0.0, device=self.device)
        loss_dict = OrderedDict()
        for emb_id in batch.keys():
            a = predictions[f"{emb_id}_action_loss"]
            r = predictions[f"{emb_id}_ratio_loss"]
            loss_dict[f"emb{emb_id}_action_loss"] = a
            loss_dict[f"emb{emb_id}_ratio_loss"] = r
            total = total + a + self.ratio_loss_weight * r
        loss_dict["action_loss"] = total / max(len(batch), 1)
        return loss_dict

    @override
    def log_info(self, info):
        log = OrderedDict()
        log["Loss"] = info["losses"]["action_loss"].item()
        for k, v in info["losses"].items():
            log[k] = v.item()
        return log


