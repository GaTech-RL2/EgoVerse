"""Training and validation adapter for dependency-aware policy pipelines."""

from __future__ import annotations

from collections import OrderedDict
from typing import Iterable

import torch
import torch.nn as nn

from egomimic.algo.algo import Algo
from egomimic.pipeline.core import Pipeline, Stage, sum_losses
from egomimic.rldb.embodiment.embodiment import get_embodiment_id

_PACKED_META_KEYS = frozenset(
    {"cu_seqlens", "max_seq_len", "seq_lens", "batch_size", "embodiment"}
)
_VIZ_PASSTHROUGH_KEYS = frozenset(
    {
        "embodiment",
        "pad_mask",
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
        rollout_adapters: dict | None = None,
        rollout_observation_adapters: dict | None = None,
        rollout_transform_mode: str | None = None,
        device=None,
    ):
        super().__init__()
        stages = list(stages)
        self.norm_stats = norm_stats
        self.domains = list(domains)
        self.ac_keys = dict(ac_keys)
        self.auxiliary_ac_keys = dict(auxiliary_ac_keys or {})
        self.action_horizon = int(action_horizon)
        # Keep the historical singleton as a fallback so old configs and
        # checkpoints reconstruct unchanged. A configured domain entry wins.
        self.rollout_adapter = rollout_adapter
        self.rollout_adapters = dict(rollout_adapters or {})
        unknown_adapter_domains = set(self.rollout_adapters) - set(self.domains)
        if unknown_adapter_domains:
            raise ValueError(
                "Rollout adapters configured for unknown domains: "
                f"{sorted(unknown_adapter_domains)}"
            )
        self.rollout_observation_adapters = dict(rollout_observation_adapters or {})
        unknown_observation_adapter_domains = set(
            self.rollout_observation_adapters
        ) - set(self.domains)
        if unknown_observation_adapter_domains:
            raise ValueError(
                "Rollout observation adapters configured for unknown domains: "
                f"{sorted(unknown_observation_adapter_domains)}"
            )
        self.rollout_transform_mode = rollout_transform_mode
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.domain_by_id = {
            get_embodiment_id(domain): domain for domain in self.domains
        }
        self._resolve_keys()
        for stage in stages:
            bind_normalization = getattr(stage, "bind_action_normalization", None)
            if bind_normalization is None:
                continue
            for domain in self.domains:
                emb_id = get_embodiment_id(domain)
                action_key = self.resolved_ac_keys[emb_id]
                stats = self.norm_stats.norm_stats[emb_id][action_key]
                bind_normalization(
                    domain,
                    norm_mode=self.norm_stats.norm_mode,
                    stats=stats,
                )
        self.nets = nn.ModuleDict({"policy": Pipeline(stages)})
        self.nets.to(self.device)

    @property
    def policy(self) -> Pipeline:
        return self.nets["policy"]

    def rollout_adapter_for(self, domain: str | int):
        """Return the domain adapter, falling back to the legacy singleton."""
        if isinstance(domain, int):
            if domain not in self.domain_by_id:
                raise KeyError(f"Unknown rollout embodiment id {domain}")
            domain = self.domain_by_id[domain]
        domain = str(domain)
        if domain not in self.domains:
            raise KeyError(
                f"Unknown rollout domain {domain!r}; configured={self.domains}"
            )
        if domain in self.rollout_adapters:
            return self.rollout_adapters[domain]
        return self.rollout_adapter

    def rollout_observation_adapter_for(self, domain: str | int):
        """Return the optional pre-normalization observation adapter."""
        if isinstance(domain, int):
            if domain not in self.domain_by_id:
                raise KeyError(f"Unknown rollout embodiment id {domain}")
            domain = self.domain_by_id[domain]
        domain = str(domain)
        if domain not in self.domains:
            raise KeyError(
                f"Unknown rollout domain {domain!r}; configured={self.domains}"
            )
        return self.rollout_observation_adapters.get(domain)

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
        if "pad_mask" in batch:
            seed["pad_mask"] = batch["pad_mask"]
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
            observation_adapter = self.rollout_observation_adapter_for(domain)
            if observation_adapter is not None:
                rollout_input = observation_adapter.encode(rollout_input)
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
        n_obs_steps = self._rollout_observation_horizon()
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

    def _rollout_observation_horizon(self) -> int:
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
        return obs_steps.pop()

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
            adapter = self.rollout_adapter_for(emb_id)
            actions = adapter.decode(tokens, context) if adapter is not None else tokens
            domain = self.domain_by_id[emb_id]
            predictions[f"emb{emb_id}_{action_key}"] = actions
            predictions[f"{domain}_{action_key}"] = actions
            predictions[f"{domain}_{action_key}_tokens"] = tokens
            predictions.setdefault(action_key, actions)
        return predictions

    def _append_sim_observation(self, obs_zarr: dict) -> dict:
        """Retain only the model's most recent rollout observation window."""
        n_obs_steps = self._rollout_observation_horizon()
        self._sim_obs_history.append(dict(obs_zarr))
        self._sim_obs_history = self._sim_obs_history[-n_obs_steps:]
        history = list(self._sim_obs_history)
        if len(history) < n_obs_steps:
            history = [history[0]] * (n_obs_steps - len(history)) + history
        if n_obs_steps == 1:
            return history[-1]

        window = {}
        for key in history[-1]:
            values = [entry[key] for entry in history]
            if torch.is_tensor(values[0]):
                window[key] = torch.stack(values, dim=1)
            else:
                window[key] = values[-1]
        return window

    @torch.inference_mode()
    def inference_step(
        self,
        obs_zarr: dict,
        t: int,
        emb_id: int,
        T_max: int | None = None,
    ):
        """Closed-loop simulator entry point with decoded native-action queues.

        ``forward_rollout`` remains the single model/decode path. This wrapper
        only keeps the observation window and consumes a predicted native
        action chunk. Setting ``algo.replan_every`` (the canonical evaluator's
        ``--replan-every`` flag) limits how many actions are executed before a
        fresh observation is queried; otherwise the complete decoded chunk is
        consumed open loop.
        """
        del T_max
        if emb_id not in self.domain_by_id:
            raise KeyError(
                f"Unknown rollout embodiment id {emb_id}; "
                f"configured={sorted(self.domain_by_id)}"
            )
        if t == 0 or not hasattr(self, "_sim_action_queue"):
            self._sim_action_queue = []
            self._sim_obs_history = []

        obs_window = self._append_sim_observation(obs_zarr)
        if not self._sim_action_queue:
            domain = self.domain_by_id[emb_id]
            processed = self.process_batch_for_rollout({domain: obs_window})
            action_key = self.resolved_ac_keys[emb_id]
            actions = self.forward_rollout(processed, rollout_t=t)[
                f"emb{emb_id}_{action_key}"
            ]
            if torch.is_tensor(actions):
                if actions.ndim >= 3 and actions.shape[0] == 1:
                    actions = actions[0]
                if actions.ndim == 1:
                    actions = actions.unsqueeze(0)
                if actions.ndim != 2:
                    raise ValueError(
                        "Pipeline rollout must decode to (H, D) or (1, H, D), "
                        f"got {tuple(actions.shape)}"
                    )
                chunk = [actions[index] for index in range(actions.shape[0])]
            else:
                import numpy as np

                actions = np.asarray(actions)
                if actions.ndim >= 3 and actions.shape[0] == 1:
                    actions = actions[0]
                if actions.ndim == 1:
                    actions = actions[None]
                if actions.ndim != 2:
                    raise ValueError(
                        "Pipeline rollout must decode to (H, D) or (1, H, D), "
                        f"got {actions.shape}"
                    )
                chunk = [actions[index] for index in range(actions.shape[0])]

            replan_every = getattr(self, "replan_every", None)
            n_keep = len(chunk) if replan_every is None else int(replan_every)
            if n_keep <= 0:
                raise ValueError(f"replan_every must be positive, got {n_keep}")
            self._sim_action_queue = chunk[: min(n_keep, len(chunk))]
            if not self._sim_action_queue:
                raise RuntimeError("Pipeline rollout decoded an empty action chunk")

        action = self._sim_action_queue.pop(0)
        if torch.is_tensor(action):
            # NumPy has no bfloat16 dtype. The explicit FP32 boundary preserves
            # the model's BF16 compute while keeping simulator I/O portable.
            return action.detach().float().cpu().numpy().reshape(-1)

        import numpy as np

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
            if "pred_action_samples" in result:
                samples = result["pred_action_samples"]
                predictions[f"emb{emb_id}_{action_key}_samples"] = samples
                predictions[f"{self.domain_by_id[emb_id]}_{action_key}_samples"] = (
                    samples
                )
            if "raw_pred_action" in result:
                predictions[f"emb{emb_id}_{action_key}_raw"] = result["raw_pred_action"]
            if "raw_pred_action_samples" in result:
                predictions[f"emb{emb_id}_{action_key}_raw_samples"] = result[
                    "raw_pred_action_samples"
                ]
        return predictions

    def log_info(self, info: dict) -> OrderedDict:
        losses = info["losses"]
        logged = OrderedDict(Loss=losses["action_loss"].item())
        logged.update((key, value.item()) for key, value in losses.items())

        # Stable W&B aliases for normalized native-action MSE. The loss keys
        # retain their historical embodiment-id names, while these aliases are
        # readable and consistent across single-domain and cotrain runs.
        per_domain_mse = []
        per_domain_raw_mse = []
        per_domain_canonical_mse = []
        per_domain_energy_score = []
        gauge_metric_keys = {
            "Loss": "loss_latent_endpoint_gauge",
            "Active": "log_latent_endpoint_gauge_active",
            "Endpoint_RMS": "log_latent_endpoint_total_rms",
            "GroupMean_RMS": "log_latent_endpoint_group_mean_rms",
            "WithinK_RMS": "log_latent_endpoint_centered_within_k_rms",
            "MaxAbs": "log_latent_endpoint_max_abs",
            "Decoder_FirstLinear_Frobenius": (
                "log_decoder_first_linear_weight_frobenius_norm"
            ),
            "Latent_Decoder_Scale_Product": (
                "log_latent_decoder_scale_product"
            ),
        }
        per_domain_gauge = {name: [] for name in gauge_metric_keys}
        for emb_id, domain in self.domain_by_id.items():
            mse_key = f"{emb_id}_loss_native_action"
            if mse_key not in losses:
                mse_key = f"{emb_id}_log_native_action"
            if mse_key in losses:
                value = losses[mse_key].item()
                logged[f"MSE/{domain}"] = value
                per_domain_mse.append(value)

            raw_mse_key = f"{emb_id}_log_raw_action_mse"
            if raw_mse_key in losses:
                value = losses[raw_mse_key].item()
                logged[f"Raw_MSE/{domain}"] = value
                per_domain_raw_mse.append(value)

            canonical_mse_key = f"{emb_id}_log_canonical_action_mse"
            if canonical_mse_key in losses:
                value = losses[canonical_mse_key].item()
                logged[f"Canonical_MSE/{domain}"] = value
                per_domain_canonical_mse.append(value)

            score_key = f"{emb_id}_loss_conditional_energy_score"
            if score_key not in losses:
                score_key = f"{emb_id}_log_conditional_energy_score"
            if score_key in losses:
                value = losses[score_key].item()
                logged[f"EnergyScore/{domain}"] = value
                per_domain_energy_score.append(value)

            for name, suffix in gauge_metric_keys.items():
                key = f"{emb_id}_{suffix}"
                if key not in losses:
                    continue
                value = losses[key].item()
                logged[f"LatentGauge/{name}/{domain}"] = value
                per_domain_gauge[name].append(value)
        if per_domain_mse:
            logged["MSE"] = sum(per_domain_mse) / len(per_domain_mse)
        if per_domain_raw_mse:
            logged["Raw_MSE"] = sum(per_domain_raw_mse) / len(per_domain_raw_mse)
        if per_domain_canonical_mse:
            logged["Canonical_MSE"] = sum(per_domain_canonical_mse) / len(
                per_domain_canonical_mse
            )
        if per_domain_energy_score:
            logged["EnergyScore"] = sum(per_domain_energy_score) / len(
                per_domain_energy_score
            )
        for name, values in per_domain_gauge.items():
            if values:
                logged[f"LatentGauge/{name}"] = sum(values) / len(values)
        return logged
