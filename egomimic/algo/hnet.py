"""
H-Net policy for EgoVerse — stage-based architecture.

The policy treats the action sequence as the input modality (autoregressive,
causal). Observations are encoded by a ``CondEncoderModule`` into a
``cond_dict`` carried by an ``HNetContext`` that is threaded through the
stage tree. Each stage reads whichever cond key it wants (or ignores cond
entirely).

Loss = action MSE (next-action prediction) +
       sum_over_chunkers( weight * ratio_loss(boundary_predictions) ).

The per-chunker ratio-loss weights live inside the chunker stages themselves;
this algo just calls ``ratio_loss_from_aux(ctx.aux)`` after forward.
"""
from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn
from overrides import override

from egomimic.algo.algo import Algo
from egomimic.models.hnet_nets.cond_encoders import CondEncoderModule
from egomimic.models.hnet_nets.context import HNetContext
from egomimic.models.hnet_nets.hnet import HNet as HNetCore
from egomimic.models.hnet_nets.hnet import ratio_loss_from_aux
from egomimic.rldb.embodiment.embodiment import get_embodiment_id


class HNetPolicy(nn.Module):
    """
    action-tokenizer → stage-based H-Net → action-detokenizer.

    Owns action_in / action_out projections, BOS token, positional embedding,
    the ``CondEncoderModule``, and the ``HNetCore`` (stage tree).
    """

    def __init__(
        self,
        action_dim: int,
        action_horizon: int,
        d_model: int,
        cond_encoder: CondEncoderModule,
        hnet: HNetCore,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.d_model = d_model

        self.action_in = nn.Linear(action_dim, d_model)
        self.action_out = nn.Linear(d_model, action_dim)
        self.bos = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.bos, std=0.02)
        self.pos_emb = nn.Parameter(torch.zeros(1, action_horizon, d_model))
        nn.init.normal_(self.pos_emb, std=0.02)

        self.cond_encoder = cond_encoder
        self.hnet = hnet

        # Sanity-check that the stage tree's outer hidden dim matches d_model.
        if self.hnet.input_hidden_dim != d_model:
            raise ValueError(
                f"hnet.input_hidden_dim ({self.hnet.input_hidden_dim}) "
                f"must equal d_model ({d_model})."
            )
        if self.hnet.output_hidden_dim != d_model:
            raise ValueError(
                f"hnet.output_hidden_dim ({self.hnet.output_hidden_dim}) "
                f"must equal d_model ({d_model})."
            )

    def _build_ctx(self, obs: dict) -> HNetContext:
        cond_dict = self.cond_encoder.encode(obs, self.action_horizon)
        return HNetContext(cond_dict=cond_dict, aux=[], inference_params=None)

    def forward(self, actions: torch.Tensor, obs: dict):
        """
        actions: (B, T, action_dim) ground-truth actions for teacher-forcing.
        obs:     dict of (B, ...) obs tensors.

        Returns: (pred_actions (B, T, action_dim), aux list).
        """
        B, T, _ = actions.shape
        x = self.action_in(actions)
        x = torch.cat([self.bos.expand(B, -1, -1), x[:, :-1]], dim=1)
        x = x + self.pos_emb[:, :T]

        ctx = self._build_ctx(obs)
        h = self.hnet(x, ctx)
        return self.action_out(h), ctx.aux

    @torch.no_grad()
    def generate(self, obs: dict, batch_size: int, device) -> torch.Tensor:
        """Autoregressive rollout from BOS for ``action_horizon`` steps."""
        T = self.action_horizon
        cond_dict = self.cond_encoder.encode(obs, T)
        actions = torch.zeros(batch_size, T, self.action_dim, device=device)
        dtype = next(self.parameters()).dtype

        inference_params = self.hnet.allocate_inference_cache(
            batch_size=batch_size, max_seqlen=T, device=device, dtype=dtype,
        )

        # Per-step cond_dict slice (B, d_cond) — AdaLN broadcasts over the
        # single-token sequence dim inside the encoder.
        def slice_cond(t: int) -> dict:
            return {k: v[:, t] if v.dim() == 3 else v for k, v in cond_dict.items()}

        cur = self.bos.expand(batch_size, -1, -1) + self.pos_emb[:, 0:1]
        for t in range(T):
            ctx = HNetContext(
                cond_dict=slice_cond(t), aux=[], inference_params=inference_params,
            )
            h = self.hnet.step(cur, ctx)
            a_t = self.action_out(h)
            actions[:, t : t + 1] = a_t
            if t < T - 1:
                cur = self.action_in(a_t) + self.pos_emb[:, t + 1 : t + 2]
        return actions


class HNet(Algo):
    """
    H-Net policy Algo. Single-domain action-sequence model with per-frame
    obs conditioning -- each action token sees the obs at its own timestep.
    """

    def __init__(
        self,
        data_schematic,
        action_dim: int,
        action_horizon: int,
        d_model: int,
        d_cond: int,
        cond_encoder: CondEncoderModule,
        hnet: HNetCore,
        domains: list = None,
        ac_keys: dict = None,
        device=None,
        **kwargs,
    ):
        super().__init__()
        self.data_schematic = data_schematic
        self.domains = list(domains or [])
        self.ac_keys = dict(ac_keys or {})
        self.action_horizon = action_horizon
        self.d_cond = d_cond
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        policy = HNetPolicy(
            action_dim=action_dim,
            action_horizon=action_horizon,
            d_model=d_model,
            cond_encoder=cond_encoder,
            hnet=hnet,
        )
        self.nets = nn.ModuleDict({"policy": policy})
        self.nets = self.nets.float().to(self.device)

        # Resolve per-embodiment keys via data_schematic (like HPT).
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
            actions = _batch[ac_key]
            obs = self._build_obs(_batch, emb_id)

            pred, aux = policy(actions, obs)
            mse = nn.functional.mse_loss(pred, actions)
            rloss = ratio_loss_from_aux(aux, device=mse.device)
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
            # Ratio-loss weights are baked into each chunker stage, so r
            # is already a properly-weighted sum.
            total = total + a + r
        loss_dict["action_loss"] = total / max(len(batch), 1)
        return loss_dict

    @override
    def log_info(self, info):
        log = OrderedDict()
        log["Loss"] = info["losses"]["action_loss"].item()
        for k, v in info["losses"].items():
            log[k] = v.item()
        return log
