"""PipelineAlgo — the BATCHFLOW runner.

Satisfies the exact three-method contract pl_model.ModelWrapper expects
(process_batch_for_training / forward_training / compute_losses) so the
lightning glue is untouched. The model is ONE Pipeline of stages; losses are
stages; the runner:

  * seeds a FLAT batch dict per embodiment (obs/* keys, actions, packing
    meta, "embodiment", the aux/* accumulators),
  * runs the pipeline,
  * sums loss/* into the training loss,
  * prefixes log/* into per-emb metric keys (one naming scheme for every
    module: Train/{emb}_{name}).

Rollout (inference_step) runs the SAME pipeline on an obs-only dict resolved
by Pipeline.plan() — posterior/loss stages are provably excluded.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import List, Optional

import torch
import torch.nn as nn

from egomimic.algo.algo import Algo
from egomimic.rldb.embodiment.embodiment import get_embodiment_id
from egomimic.pipeline.core import Pipeline, Stage, sum_losses

_PACKED_META_KEYS = (
    "cu_seqlens", "max_seq_len", "seq_lens", "batch_size",
    "embodiment", "episode_idx", "chunk_offset",
)


class PipelineAlgo(Algo):
    def __init__(
        self,
        stages: List[Stage],
        norm_stats,
        domains: list = None,
        ac_keys: dict = None,
        device=None,
        action_horizon: int = 2560,
        train_obs_transforms: list | None = None,
        episode_level_transforms: list | None = None,
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            # no dead knobs: unknown config keys fail loudly.
            raise TypeError(f"PipelineAlgo got unknown config keys: {sorted(kwargs)}")
        self.norm_stats = norm_stats
        self.domains = list(domains or [])
        self.domain_by_id = {get_embodiment_id(e): e for e in self.domains}
        self.ac_keys = dict(ac_keys or {})
        self.action_horizon = int(action_horizon)
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.train_obs_transforms = list(train_obs_transforms or [])
        self.episode_level_transforms = list(episode_level_transforms or [])
        self._resolve_embodiment_keys(norm_stats)
        self.nets = nn.ModuleDict({"policy": Pipeline(list(stages))})
        self.nets.to(self.device)
        # replan cadence for closed-loop AR = the TargetBuilder's stride
        # (introspected -> train/inference cadence can never desync).
        self.replan_stride = 1
        for s in self.nets["policy"].stages:
            if type(s).__name__ == "TargetBuilder":
                self.replan_stride = int(s.stride)

    # ------------------------------------------------------------------ #
    @property
    def policy(self) -> Pipeline:
        return self.nets["policy"]

    def _seed(self, emb_id: int, _batch: dict) -> dict:
        """Flat batch dict for one embodiment (the ONE carrier)."""
        obs = self._build_obs(_batch, emb_id)
        b = {
            "actions": _batch[self.resolved_ac_keys[emb_id]],
            "cu_seqlens": _batch["cu_seqlens"],
            "max_seq_len": int(_batch["max_seq_len"]),
            "embodiment": self.domain_by_id.get(emb_id),
            "aux/chunker": [],
        }
        for k, v in obs.items():
            b[f"obs/{k}"] = v
        return b

    # ------------------------------------------------------------------ #
    # pl_model contract
    # ------------------------------------------------------------------ #
    def process_batch_for_training(self, batch):
        processed = {}
        for emb_name, _batch in batch.items():
            emb_id = get_embodiment_id(emb_name)
            if (self.episode_level_transforms and self.policy.training
                    and "cu_seqlens" in _batch):
                from egomimic.algo.hnet.episode_transforms import (
                    apply_episode_level_transforms,
                )
                _batch = apply_episode_level_transforms(
                    _batch, self.episode_level_transforms)
            out = {}
            is_packed = "cu_seqlens" in _batch
            for key, value in _batch.items():
                if is_packed and key in _PACKED_META_KEYS:
                    out[key] = value
                    continue
                key_name = self.norm_stats.zarr_key_to_keyname(key, emb_id)
                if key_name is not None:
                    out[key_name] = value
            out["_packed"] = is_packed
            out = self.norm_stats.normalize(out, emb_id)
            if self.train_obs_transforms and self.policy.training:
                for t in self.train_obs_transforms:
                    out = t(out)
            for key, value in out.items():
                if isinstance(value, torch.Tensor):
                    value = value.to(self.device)
                    if value.is_floating_point():
                        value = value.float()
                    out[key] = value
            processed[emb_id] = out
        return processed

    def forward_training(self, batch):
        predictions = OrderedDict()
        for emb_id, _batch in batch.items():
            b = self.policy(self._seed(emb_id, _batch))
            total = sum_losses(b)
            dev = total.device
            predictions[f"{emb_id}_action_loss"] = total
            for k, v in b.items():
                if k.startswith("loss/"):
                    predictions[f"{emb_id}_{k[5:].replace('/', '_')}_loss"] = (
                        v.detach() if torch.is_tensor(v) else torch.tensor(float(v), device=dev))
                elif k.startswith("log/"):
                    predictions[f"{emb_id}_{k[4:].replace('/', '_')}"] = torch.tensor(
                        float(v), device=dev)
        return predictions

    def compute_losses(self, predictions, batch):
        losses = OrderedDict()
        embs = list(batch.keys())
        total = None
        for e in embs:
            t = predictions[f"{e}_action_loss"]
            total = t if total is None else total + t
        losses["action_loss"] = total / max(len(embs), 1)
        for k, v in predictions.items():
            if torch.is_tensor(v) and v.dim() == 0:
                losses[k] = v
        return losses

    def forward_eval(self, batch):
        """Teacher-forced eval: same pipeline; pred_action is already decoded."""
        predictions = OrderedDict()
        for emb_id, _batch in batch.items():
            b = self.policy(self._seed(emb_id, _batch))
            predictions[self.resolved_ac_keys[emb_id]] = b["pred_action"]
        return predictions

    # ------------------------------------------------------------------ #
    # Rollout — the persistent dict IS the state. plan() resolves the
    # deployed subset once (obs-only seeds exclude posterior/loss stages).
    # ------------------------------------------------------------------ #
    def init_step_state(self, batch_size, T_max, device, dtype):
        assert batch_size == 1, "rollout is batch_size=1 (recompute-over-prefix)"
        return {"obs_prefix": [], "device": device, "dtype": dtype, "plan": None}

    @torch.no_grad()
    def step(self, state: dict, obs_norm: dict, t: int, embodiment_id=None):
        state["obs_prefix"].append({k: v for k, v in obs_norm.items()})
        prefix = state["obs_prefix"]
        T = len(prefix)
        dev, dt = state["device"], state["dtype"]
        b = {
            # NO "actions" key: plan() then provably excludes TargetBuilder,
            # posterior and every loss stage -> DENSE full-rate prefix, exactly
            # the old rollout semantics (stride cadence lives in the queue).
            "cu_seqlens": torch.tensor([0, T], device=dev, dtype=torch.long),
            "max_seq_len": T,
            "embodiment": embodiment_id,
            "aux/chunker": [],
        }
        for k in prefix[0].keys():
            vs = [f[k] for f in prefix]
            b[f"obs/{k}"] = (torch.stack(vs, 0).to(dev)
                             if torch.is_tensor(vs[0]) else vs[-1])
        if state["plan"] is None:
            runnable, excluded = self.policy.plan(list(b.keys()))
            state["plan"] = runnable
            if excluded:
                names = [(type(s).__name__, miss) for s, miss in excluded]
                print(f"[PipelineAlgo.step] plan excluded (train-only): {names}")
        for stage in state["plan"]:
            b = stage(b)
        return b["pred_action"][T - 1]  # (C, D) decoded chunk at the last token

    # ------------------------------------------------------------------ #
    # Sim-eval entry (PackedSimEval calls this every env frame).
    # ------------------------------------------------------------------ #
    def inference_step(self, obs_zarr: dict, t: int, emb_id: int, T_max=None):
        import numpy as np
        from egomimic.rldb.embodiment.embodiment import get_embodiment

        if t == 0:
            device = next(self.nets.parameters()).device
            self._sim_state = self.init_step_state(
                batch_size=1, T_max=int(T_max or self.action_horizon),
                device=device, dtype=next(self.nets.parameters()).dtype)
            self._sim_action_queue: list = []
        embodiment_name = get_embodiment(emb_id).lower()
        ac_key = (self.ac_keys[embodiment_name] if embodiment_name in self.ac_keys
                  else self.ac_keys[emb_id])
        if self._sim_action_queue:
            a_norm = self._sim_action_queue.pop(0)
        else:
            obs_norm = self.norm_stats.normalize(obs_zarr, emb_id)
            chunk = self.step(self._sim_state, obs_norm, t,
                              embodiment_id=self.domain_by_id.get(emb_id))
            if chunk.dim() == 2:  # (C, D): keep replan_stride actions
                n_keep = max(1, min(self.replan_stride, chunk.shape[0]))
                self._sim_action_queue = [chunk[j] for j in range(n_keep)]
            else:  # (D,)
                self._sim_action_queue = [chunk]
            a_norm = self._sim_action_queue.pop(0)
        out = self.norm_stats.unnormalize({ac_key: a_norm}, emb_id)[ac_key]
        return out.detach().cpu().numpy().reshape(-1).astype(np.float32)
