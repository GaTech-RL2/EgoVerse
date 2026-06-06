"""BC-RNN algo. Mirrors robomimic's ``BC_RNN``.

Training: per-step MSE between predicted and ground-truth actions.
Inference: persistent LSTM hidden state across env ticks, reset on ``t=0``
or every ``rnn_horizon`` ticks (matches robomimic ``rnn.horizon``).
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from overrides import override

from egomimic.algo.algo import Algo
from egomimic.algo.bcrnn.outer_stage import BCRNNOuterStage
from egomimic.algo.loss import Loss, MSELoss
from egomimic.models.hnet_nets.cond_encoders import CondEncoderModule
from egomimic.rldb.embodiment.embodiment import get_embodiment, get_embodiment_id


class _BCRNNCtx:
    """Minimal context object for BC-RNN forward.

    Carries ``cu_seqlens`` / ``max_seqlen`` for packed mode, and ``rnn_state``
    that gets populated by the LSTM trunk's forward.
    """

    def __init__(
        self,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ):
        self.cu_seqlens = cu_seqlens
        self.max_seqlen = max_seqlen
        self.rnn_state = None


class BCRNN(Algo):
    """BC training with a recurrent (LSTM/GRU) policy.

    Args:
        outer_stage: ``BCRNNOuterStage`` — owns the obs encoder, LSTM trunk,
            and per-step action head.
        action_dim: action feature width.
        norm_stats: ``MultiDataset`` (injected by
            ``pl_model._instantiate_model``).
        loss: optional ``Loss``. Defaults to per-step MSE on ``pred_action``
            vs ``actions``.
        rnn_horizon: # env ticks before the rollout hidden state is reset
            (robomimic ``rnn.horizon``; default 10). Set <= 0 to never reset.
        rnn_open_loop: if True, the first obs is reused for every rollout
            tick within a horizon, and only the hidden state evolves.
            Matches robomimic ``rnn.open_loop``.
        domains: list of embodiment names.
        ac_keys: dict ``embodiment_name -> action zarr key``.
    """

    def __init__(
        self,
        outer_stage: BCRNNOuterStage,
        action_dim: int,
        norm_stats,
        loss: Optional[Loss] = None,
        rnn_horizon: int = 10,
        rnn_open_loop: bool = False,
        domains: Optional[list] = None,
        ac_keys: Optional[dict] = None,
        device=None,
        **kwargs,
    ):
        super().__init__()
        self.norm_stats = norm_stats
        self.domains = list(domains or [])
        self.ac_keys = dict(ac_keys or {})
        self.action_dim = int(action_dim)
        self.rnn_horizon = int(rnn_horizon)
        self.rnn_open_loop = bool(rnn_open_loop)
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        if loss is None:
            loss = MSELoss(pred_key="pred_action", target_key="actions")
        self.nets = nn.ModuleDict({"outer_stage": outer_stage, "loss": loss})
        self.nets = self.nets.float().to(self.device)

        # Embodiment / key resolution (same shape as DFoT/HNet).
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

        self._sim_state = None
        self._open_loop_obs = None
        self._rnn_counter = 0

    @property
    def outer_stage(self) -> BCRNNOuterStage:
        return self.nets["outer_stage"]

    @property
    def loss(self) -> Loss:
        return self.nets["loss"]

    @property
    def cond_encoder(self) -> CondEncoderModule:
        return self.outer_stage.cond_encoder

    @property
    def policy(self) -> BCRNNOuterStage:
        # back-compat alias; some eval code reads ``algo.policy``.
        return self.outer_stage

    _PACKED_META_KEYS = ("cu_seqlens", "max_seq_len", "seq_lens")

    # ---- Algo API -------------------------------------------------------- #

    @override
    def process_batch_for_training(self, batch):
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

    @override
    def forward_training(self, batch):
        predictions = OrderedDict()
        for emb_id, _batch in batch.items():
            ac_key = self.resolved_ac_keys[emb_id]
            actions = _batch[ac_key]
            obs = self._build_obs(_batch, emb_id)
            is_packed = _batch.get("_packed", False)
            new_batch = {"actions": actions, "__obs": obs}
            ctx = _BCRNNCtx(
                cu_seqlens=_batch["cu_seqlens"] if is_packed else None,
                max_seqlen=int(_batch["max_seq_len"]) if is_packed else None,
            )
            self.outer_stage(new_batch, ctx)
            mse = self.loss(new_batch, ctx)
            predictions[f"{emb_id}_pred"] = new_batch["pred_action"]
            predictions[f"{emb_id}_action_loss"] = mse
        return predictions

    @override
    def forward_eval(self, batch):
        """Teacher-forced single forward pass per embodiment. Same shape
        as HNet.forward_eval — returns unnormalized per-frame predictions
        plus ``seq_lens`` for packed mode so downstream metric code can
        mask the padding.
        """
        unnorm = {}
        for emb_id, _batch in batch.items():
            ac_key = self.resolved_ac_keys[emb_id]
            actions = _batch[ac_key]
            obs = self._build_obs(_batch, emb_id)
            is_packed = _batch.get("_packed", False)
            new_batch = {"actions": actions, "__obs": obs}
            ctx = _BCRNNCtx(
                cu_seqlens=_batch["cu_seqlens"] if is_packed else None,
                max_seqlen=int(_batch["max_seq_len"]) if is_packed else None,
            )
            self.outer_stage(new_batch, ctx)
            pred = new_batch["pred_action"]
            if is_packed:
                # Re-pad the (T_total, A) prediction to (B_eps, T_max, A) for
                # the downstream evaluator's expected shape.
                cu = _batch["cu_seqlens"].to(pred.device, dtype=torch.long)
                seq_lens = (cu[1:] - cu[:-1]).to(torch.long)
                B_eps = seq_lens.shape[0]
                T_max = int(seq_lens.max().item())
                padded = pred.new_zeros(B_eps, T_max, self.action_dim)
                for i in range(B_eps):
                    L = int(seq_lens[i].item())
                    start = int(cu[i].item())
                    padded[i, :L] = pred[start : start + L]
                preds = OrderedDict({ac_key: padded})
                unnorm_actions = self.norm_stats.unnormalize(preds, emb_id)
                for key, val in unnorm_actions.items():
                    unnorm[f"emb{emb_id}_{key}"] = val
                unnorm[f"emb{emb_id}_seq_lens"] = seq_lens.cpu()
            else:
                preds = OrderedDict({ac_key: pred})
                unnorm_actions = self.norm_stats.unnormalize(preds, emb_id)
                for key, val in unnorm_actions.items():
                    unnorm[f"emb{emb_id}_{key}"] = val
        return unnorm

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

    # ----- Sim eval hook (mirrors HNet.inference_step contract) ----- #

    @torch.no_grad()
    def inference_step(
        self, obs_zarr: dict, t: int, emb_id: int, T_max=None
    ) -> "np.ndarray":
        policy = self.outer_stage
        device = next(policy.parameters()).device
        # Reset on t=0 OR every rnn_horizon ticks (robomimic behaviour).
        need_reset = (
            t == 0
            or self._sim_state is None
            or (self.rnn_horizon > 0 and self._rnn_counter % self.rnn_horizon == 0)
        )
        if need_reset:
            self._sim_state = policy.init_step_state(batch_size=1, device=device)
            self._rnn_counter = 0
            if self.rnn_open_loop:
                # remember the initial obs to re-feed during this horizon
                self._open_loop_obs = {
                    k: (v.clone().detach() if torch.is_tensor(v) else v)
                    for k, v in obs_zarr.items()
                }

        obs_to_use = self._open_loop_obs if self.rnn_open_loop else obs_zarr
        obs_norm = self.norm_stats.normalize(obs_to_use, emb_id)
        # Promote each obs tensor to a (B=1, T=1, ...) shape so CondEncoder
        # treats it as per-frame for this single tick.
        obs_step = {}
        for k, v in obs_norm.items():
            if torch.is_tensor(v):
                if v.dim() == 1:  # (D,) -> (1, 1, D)
                    obs_step[k] = v.unsqueeze(0).unsqueeze(0)
                elif v.dim() == 3:  # (C, H, W) -> (1, 1, C, H, W)
                    obs_step[k] = v.unsqueeze(0).unsqueeze(0)
                elif v.dim() == 2:  # (1, D) or (B, D)
                    obs_step[k] = v.unsqueeze(1)
                elif v.dim() == 4:  # (1, C, H, W) or (B, C, H, W)
                    obs_step[k] = v.unsqueeze(1)
                else:
                    obs_step[k] = v
            else:
                obs_step[k] = v

        action_norm = policy.step(self._sim_state, obs_step, t)  # (1, 1, A)
        embodiment_name = get_embodiment(emb_id).lower()
        ac_key = (
            self.ac_keys[embodiment_name]
            if embodiment_name in self.ac_keys
            else self.ac_keys[emb_id]
        )
        action_unnorm = self.norm_stats.unnormalize(
            {ac_key: action_norm.squeeze(0).squeeze(0)}, emb_id
        )[ac_key]
        self._rnn_counter += 1
        return action_unnorm.detach().cpu().numpy().reshape(-1).astype(np.float32)
