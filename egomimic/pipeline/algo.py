"""Training and validation adapter for dependency-aware policy pipelines."""

from __future__ import annotations

from collections import OrderedDict
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn

from egomimic.algo.algo import Algo
from egomimic.eval.inference_graph import (
    ActionCacheState,
    InferenceGraph,
    KeyedNode,
)
from egomimic.pipeline.core import Pipeline, Stage, sum_losses
from egomimic.rldb.embodiment.embodiment import get_embodiment_id

_PACKED_META_KEYS = frozenset(
    {"cu_seqlens", "max_seq_len", "seq_lens", "batch_size", "embodiment"}
)
_VIZ_PASSTHROUGH_KEYS = frozenset(
    {
        "embodiment",
        "front_intrinsics",
        "left_camera_extrinsics",
        "right_camera_extrinsics",
        "viz_current_wrist_poses",
    }
)


class PipelineAlgo(Algo):
    """Expose a :class:`Pipeline` through the repository's ``Algo`` contract.

    The adapter deliberately owns no policy logic. It maps loader keys into the
    normalized model namespace, seeds one flat dictionary per embodiment, runs
    the same stage list for training and teacher-forced validation, and reduces
    every ``loss/*`` value produced by explicit loss nodes.
    """

    def __init__(
        self,
        stages: Iterable[Stage],
        norm_stats,
        domains: list[str],
        ac_keys: dict[str, str],
        auxiliary_ac_keys: dict | None = None,
        action_horizon: int = 1,
        rollout_adapter=None,
        rollout_transform_mode: str | None = None,
        inference_stages: dict | None = None,
        device=None,
    ):
        super().__init__()
        self.norm_stats = norm_stats
        self.domains = list(domains)
        self.ac_keys = dict(ac_keys)
        self.auxiliary_ac_keys = dict(auxiliary_ac_keys or {})
        self.action_horizon = int(action_horizon)
        self.rollout_adapter = rollout_adapter
        self.rollout_transform_mode = rollout_transform_mode
        self.inference_stages = inference_stages
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.domain_by_id = {
            get_embodiment_id(domain): domain for domain in self.domains
        }
        self._resolve_keys()
        self.nets = nn.ModuleDict({"policy": Pipeline(list(stages))})
        self.nets.to(self.device)
        self._inference_action_cache = ActionCacheState()
        self._inference_graph = self._build_inference_graph()

    @property
    def policy(self) -> Pipeline:
        return self.nets["policy"]

    def _resolve_keys(self) -> None:
        self.proprio_keys = {}
        self.lang_keys = {}
        self.camera_keys = {}
        self.resolved_ac_keys = {}
        for domain in self.domains:
            emb_id = get_embodiment_id(domain)
            for key_type, destination in (
                ("proprio_keys", self.proprio_keys),
                ("lang_keys", self.lang_keys),
                ("camera_keys", self.camera_keys),
            ):
                destination[emb_id] = [
                    key
                    for key in self.norm_stats.keys_of_type(key_type, emb_id)
                    if self.norm_stats.is_key_with_embodiment(key, emb_id)
                ]
            available_actions = {
                key
                for key in self.norm_stats.keys_of_type("action_keys", emb_id)
                if self.norm_stats.is_key_with_embodiment(key, emb_id)
            }
            requested = self.ac_keys[domain]
            if requested not in available_actions:
                raise KeyError(
                    f"Action key {requested!r} is unavailable for {domain!r}; "
                    f"available={sorted(available_actions)}"
                )
            self.resolved_ac_keys[emb_id] = requested

    def _prepare_loader_batch(self, batch: dict, emb_id: int) -> dict:
        """Map zarr keys once, normalizing only packed/raw batches.

        Standard ``MultiDataset`` samples are normalized in ``__getitem__``.
        Packed loaders carry ``cu_seqlens`` and intentionally defer
        normalization until after collation. Keeping that distinction here is
        the double-normalization guard for every Pipeline policy.
        """
        is_packed = "cu_seqlens" in batch
        out = {}
        for key, value in batch.items():
            if is_packed and key in _PACKED_META_KEYS:
                out[key] = value
                continue
            mapped = self.norm_stats.zarr_key_to_keyname(key, emb_id)
            if mapped is not None:
                out[mapped] = value
            elif key in _VIZ_PASSTHROUGH_KEYS:
                out[key] = value
        if is_packed:
            out = self.norm_stats.normalize(out, emb_id)
        return out

    def _build_obs(self, batch: dict, emb_id: int) -> dict:
        keys = (
            self.proprio_keys[emb_id]
            + self.lang_keys[emb_id]
            + self.camera_keys[emb_id]
        )
        return {key: batch[key] for key in keys if key in batch}

    def _seed(self, emb_id: int, batch: dict, include_actions: bool = True) -> dict:
        seed = {"embodiment": self.domain_by_id[emb_id]}
        if include_actions:
            action_key = self.resolved_ac_keys[emb_id]
            seed["actions"] = batch[action_key]
        seed.update(
            {
                f"obs/{key}": value
                for key, value in self._build_obs(batch, emb_id).items()
            }
        )
        return seed

    def _move_to_device(self, batch: dict) -> dict:
        for key, value in batch.items():
            if torch.is_tensor(value):
                value = value.to(self.device)
                if value.is_floating_point():
                    value = value.float()
                batch[key] = value
        return batch

    def process_batch_for_training(self, batch: dict) -> dict:
        processed = {}
        for domain, loader_batch in batch.items():
            emb_id = get_embodiment_id(domain)
            if emb_id not in self.domain_by_id:
                raise KeyError(f"Unexpected embodiment batch {domain!r}")
            out = self._prepare_loader_batch(loader_batch, emb_id)
            processed[emb_id] = self._move_to_device(out)
        return processed

    def process_batch_for_rollout(self, batch: dict) -> dict:
        """Map and normalize observations assembled outside ``MultiDataset``.

        Training samples from the standard loader are already normalized in
        ``MultiDataset.__getitem__``.  Robot/simulator rollout observations are
        built directly, so this entry point normalizes the raw keys before
        mapping them into the model namespace.
        """
        processed = {}
        for domain, loader_batch in batch.items():
            emb_id = get_embodiment_id(domain)
            if emb_id not in self.domain_by_id:
                raise KeyError(f"Unexpected embodiment batch {domain!r}")
            rollout_input = dict(loader_batch)
            for action_key in self.norm_stats.keys_of_type("action_keys", emb_id):
                rollout_input.pop(action_key, None)
                zarr_key = self.norm_stats.keyname_to_zarr_key(action_key, emb_id)
                if zarr_key is not None:
                    rollout_input.pop(zarr_key, None)
            normalized = self.norm_stats.normalize(rollout_input, emb_id)
            out = self._prepare_loader_batch(normalized, emb_id)
            processed[emb_id] = self._move_to_device(out)
        return processed

    def _rollout_seed(self, emb_id: int, batch: dict, rollout_t: int) -> dict:
        seed = self._seed(emb_id, batch, include_actions=False)
        obs_steps = {
            int(stage.rollout_obs_steps)
            for stage in self.policy.stages
            if hasattr(stage, "rollout_obs_steps")
        }
        if len(obs_steps) != 1:
            raise RuntimeError(
                "Pipeline rollout requires exactly one declared observation horizon; "
                f"found {sorted(obs_steps)}"
            )
        n_obs_steps = obs_steps.pop()
        for key, value in list(seed.items()):
            if not key.startswith("obs/") or not torch.is_tensor(value):
                continue
            if n_obs_steps == 1:
                value = value.unsqueeze(1)
            elif value.ndim < 2 or value.shape[1] != n_obs_steps:
                raise ValueError(
                    f"Rollout {key} must carry {n_obs_steps} observations, got "
                    f"shape {tuple(value.shape)}"
                )
            seed[key] = value
        seed["rollout_t"] = int(rollout_t)
        return seed

    def _run_rollout_policy(self, seed: dict) -> dict:
        runnable, excluded = self.policy.plan(seed.keys(), mode="rollout")
        blocked = [
            (type(stage).__name__, missing)
            for stage, missing in excluded
            if missing != ["<train-only>"]
        ]
        if blocked:
            raise RuntimeError(f"Pipeline rollout graph has blocked stages: {blocked}")
        result = seed
        for stage in runnable:
            result = stage(result)
        return result

    @torch.inference_mode()
    def forward_rollout(self, batch: dict, rollout_t: int = 0) -> OrderedDict:
        """Run the observation-only graph and return deployable actions.

        The sampler predicts normalized arc tokens.  They are unnormalized
        before the configured rollout adapter reconstructs fixed-rate actions.
        """
        predictions = OrderedDict()
        for emb_id, loader_batch in batch.items():
            result = self._run_rollout_policy(
                self._rollout_seed(emb_id, loader_batch, rollout_t)
            )
            action_key = self.resolved_ac_keys[emb_id]
            normalized_tokens = result["pred_action"]
            tokens = self.norm_stats.unnormalize(
                {action_key: normalized_tokens}, emb_id
            )[action_key]
            context = self.norm_stats.unnormalize(loader_batch, emb_id)
            actions = (
                self.rollout_adapter.decode(tokens, context)
                if self.rollout_adapter is not None
                else tokens
            )
            domain = self.domain_by_id[emb_id]
            predictions[f"emb{emb_id}_{action_key}"] = actions
            predictions[f"{domain}_{action_key}"] = actions
            predictions[f"{domain}_{action_key}_tokens"] = tokens
            predictions.setdefault(action_key, actions)
        return predictions

    # ------------------------------------------------------------------ #
    # Shared deployment graph. Cache hits terminate before observation
    # transforms and model execution; cache misses use the exact current
    # process_batch_for_rollout -> forward_rollout path above.
    # ------------------------------------------------------------------ #
    def _build_inference_graph(self) -> InferenceGraph:
        defaults = {
            "check_cache": {
                "in": {"obs": "obs"},
                "out": {"action": "policy.action"},
            },
            "inference_preprocess": {
                "in": {"obs": "obs"},
                "out": {"request": "model.request"},
            },
            "model": {
                "in": {"request": "model.request"},
                "out": {"plan": "model.plan"},
            },
            "update_cache": {
                "in": {"plan": "model.plan", "obs": "obs"},
                "out": {"action": "policy.action"},
            },
        }
        cfg = self.inference_stages or {}
        nodes = cfg.get("nodes", cfg) if hasattr(cfg, "get") else {}

        def ports(name: str) -> dict:
            node = (
                cfg.get("model", defaults[name])
                if name == "model"
                else nodes.get(name, defaults[name])
            )
            return {
                "in": dict(node.get("in", defaults[name]["in"])),
                "out": dict(node.get("out", defaults[name]["out"])),
            }

        return InferenceGraph(
            check_cache=KeyedNode(
                self._graph_check_cache, **ports("check_cache")
            ),
            inference_preprocess=KeyedNode(
                self._graph_preprocess, **ports("inference_preprocess")
            ),
            model=KeyedNode(self._graph_model, **ports("model")),
            update_cache=KeyedNode(
                self._graph_update_cache, **ports("update_cache")
            ),
            terminal_key=str(cfg.get("terminal", "policy.action")),
        )

    def _inference_cache_value(self, key: str, default):
        overrides = getattr(self, "inference_cache_overrides", {}) or {}
        if key in overrides:
            return overrides[key]
        cfg = self.inference_stages or {}
        cache = cfg.get("cache", {}) if hasattr(cfg, "get") else {}
        value = cache.get(key, default) if hasattr(cache, "get") else default
        return default if value is None else value

    def reset_inference(self) -> None:
        """Reset episode-scoped deployment state without changing the model."""
        self._inference_action_cache.reset()
        self._inference_started = False
        adapter = self.rollout_adapter
        if adapter is not None and hasattr(adapter, "reset"):
            adapter.reset()

    def _graph_check_cache(self, obs: dict):
        del obs
        if not self._inference_action_cache:
            return None
        return self._inference_action_cache.pop()

    def _graph_preprocess(self, obs: dict) -> dict:
        adapter = getattr(self, "inference_obs_adapter", None)
        model_obs = adapter(obs) if adapter is not None else obs
        domain = self.domain_by_id[self._inference_emb_id]
        return self.process_batch_for_rollout({domain: model_obs})

    def _graph_model(self, request: dict) -> dict:
        predictions = self.forward_rollout(
            request, rollout_t=self._inference_t
        )
        action_key = self.resolved_ac_keys[self._inference_emb_id]
        actions = predictions[f"emb{self._inference_emb_id}_{action_key}"]
        if torch.is_tensor(actions):
            actions = actions.detach().cpu().numpy()
        actions = np.asarray(actions, dtype=np.float32)
        while actions.ndim > 2 and actions.shape[0] == 1:
            actions = actions[0]
        if actions.ndim == 1:
            actions = actions[None, :]
        if actions.ndim != 2:
            raise ValueError(
                "rollout prediction must be an action chunk with shape "
                f"(T, D), got {actions.shape}"
            )
        configured_keep = int(self._inference_cache_value("n_keep", 1))
        n_keep = max(1, min(configured_keep, actions.shape[0]))
        return {"actions": [actions[index].copy() for index in range(n_keep)]}

    def _graph_update_cache(self, plan: dict, obs: dict):
        del obs
        self._inference_action_cache.replace(plan["actions"])
        return self._inference_action_cache.pop()

    def inference_step(
        self,
        obs_zarr: dict,
        t: int,
        emb_id: int,
        T_max=None,
    ) -> np.ndarray:
        """Return one committed action from the shared deployment graph.

        ``t == 0`` is the episode reset signal. ``T_max`` remains in the
        cross-model public contract but this stateless rollout path does not
        allocate a fixed-size history buffer.
        """
        del T_max
        if int(t) == 0:
            self.reset_inference()
            self._inference_started = True
        elif not getattr(self, "_inference_started", False):
            raise RuntimeError("inference_step must begin with t == 0")
        emb_id = int(emb_id)
        if emb_id not in self.domain_by_id:
            raise KeyError(
                f"checkpoint does not support embodiment id {emb_id}; "
                f"available={sorted(self.domain_by_id)}"
            )
        self._inference_t = int(t)
        self._inference_emb_id = emb_id
        action = self._inference_graph(obs=obs_zarr)
        return np.asarray(action, dtype=np.float32).reshape(-1)

    def forward_training(self, batch: dict) -> OrderedDict:
        predictions = OrderedDict()
        for emb_id, loader_batch in batch.items():
            result = self.policy(self._seed(emb_id, loader_batch))
            total = sum_losses(result)
            predictions[f"{emb_id}_action_loss"] = total
            for key, value in result.items():
                if not (key.startswith("loss/") or key.startswith("log/")):
                    continue
                metric = key.replace("/", "_")
                if not torch.is_tensor(value):
                    value = torch.tensor(float(value), device=total.device)
                predictions[f"{emb_id}_{metric}"] = value
        return predictions

    def compute_losses(self, predictions: dict, batch: dict) -> OrderedDict:
        per_domain = [predictions[f"{emb_id}_action_loss"] for emb_id in batch]
        if not per_domain:
            raise RuntimeError("PipelineAlgo received an empty multi-dataset batch")
        losses = OrderedDict(action_loss=torch.stack(per_domain).mean())
        losses.update(
            (key, value)
            for key, value in predictions.items()
            if torch.is_tensor(value) and value.ndim == 0
        )
        return losses

    def forward_eval(self, batch: dict) -> OrderedDict:
        predictions = OrderedDict()
        for emb_id, loader_batch in batch.items():
            result = self.policy(self._seed(emb_id, loader_batch))
            action_key = self.resolved_ac_keys[emb_id]
            prediction = result["pred_action"]
            predictions[f"emb{emb_id}_{action_key}"] = prediction
            predictions[f"{self.domain_by_id[emb_id]}_{action_key}"] = prediction
            predictions.setdefault(action_key, prediction)
        return predictions

    def log_info(self, info: dict) -> OrderedDict:
        losses = info["losses"]
        logged = OrderedDict(Loss=losses["action_loss"].item())
        logged.update((key, value.item()) for key, value in losses.items())
        return logged
