"""Pipeline-native standard Diffusion Policy stage.

This module is deliberately an adapter, not a second diffusion
implementation.  It delegates denoising, the epsilon objective, and sampling
to :class:`egomimic.models.diffusion_policy.DiffusionPolicy` instances.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn

from egomimic.models.diffusion_policy import DiffusionPolicy
from egomimic.pipeline.core import Stage


def _scheduler_value(scheduler, name: str):
    """Read a scheduler setting from repo-local or diffusers schedulers."""
    value = getattr(scheduler, name, None)
    if value is not None:
        return value
    config = getattr(scheduler, "config", None)
    if config is None:
        return None
    if isinstance(config, Mapping):
        return config.get(name)
    return getattr(config, name, None)


class MultiDomainDiffusionPolicyStage(Stage):
    """Run one genuine noise-prediction Diffusion Policy per action domain.

    A separate ``DiffusionPolicy`` is required for each domain because a
    ConditionalUnet1D has a fixed action channel width.  This is what permits
    ChainGripper points6 and U-Socket rotvec4 to share the observation encoder
    while retaining correct, unpadded DP objectives.

    Training performs one randomly noised epsilon-prediction step and writes
    ``loss/diffusion_noise``.  Evaluation and rollout perform the scheduler's
    complete reverse process and write ``pred_action``.  The mode split avoids
    paying for a full reverse chain on every optimizer step.
    """

    # Training consumes a target and emits only the epsilon objective. Rollout
    # has no target and emits a sampled action instead. Keeping those contracts
    # separate prevents dependency audits from inventing keys in either graph.
    reads = ["condition", "target", "embodiment"]
    writes = ["loss/diffusion_noise", "log/*"]
    reads_by_mode = {"rollout": ["condition", "embodiment"]}
    writes_by_mode = {"rollout": ["pred_action", "log/*"]}
    objective = "epsilon"

    def __init__(
        self,
        policies: Mapping[str, DiffusionPolicy],
        action_horizon: int,
        condition_input_dim: int,
    ):
        super().__init__()
        self.action_horizon = int(action_horizon)
        self.condition_input_dim = int(condition_input_dim)
        if self.action_horizon <= 0 or self.condition_input_dim <= 0:
            raise ValueError("action_horizon and condition_input_dim must be positive")
        if not policies:
            raise ValueError("policies must contain at least one domain")

        normalized = {}
        action_dims = {}
        for raw_domain, policy in dict(policies).items():
            domain = str(raw_domain)
            if not isinstance(policy, DiffusionPolicy):
                raise TypeError(
                    f"Policy for {domain!r} must be DiffusionPolicy, got "
                    f"{type(policy).__name__}"
                )
            if int(policy.action_horizon) != self.action_horizon:
                raise ValueError(
                    f"Policy {domain!r} horizon={policy.action_horizon}, expected "
                    f"{self.action_horizon}"
                )
            if set(policy.infer_ac_dims) != {domain}:
                raise ValueError(
                    f"Policy {domain!r} infer_ac_dims must contain only that domain, "
                    f"got {sorted(policy.infer_ac_dims)}"
                )
            action_dim = int(policy.infer_ac_dims[domain])
            if action_dim <= 0:
                raise ValueError(
                    f"Policy {domain!r} has invalid action width {action_dim}"
                )
            prediction_type = _scheduler_value(
                policy.noise_scheduler, "prediction_type"
            )
            if prediction_type != "epsilon":
                raise ValueError(
                    f"Policy {domain!r} must use the epsilon noise objective; "
                    f"scheduler prediction_type={prediction_type!r}"
                )
            train_steps = _scheduler_value(
                policy.noise_scheduler, "num_train_timesteps"
            )
            if train_steps is None or int(train_steps) <= 0:
                raise ValueError(
                    f"Policy {domain!r} scheduler has no positive train timestep count"
                )
            if (
                policy.num_inference_steps is None
                or int(policy.num_inference_steps) <= 0
            ):
                raise ValueError(
                    f"Policy {domain!r} must declare positive num_inference_steps"
                )

            proj_cond = getattr(policy.model, "proj_cond", None)
            if (
                proj_cond is not None
                and int(proj_cond.in_features) != self.condition_input_dim
            ):
                raise ValueError(
                    f"Policy {domain!r} expects {proj_cond.in_features} condition "
                    f"features, configured Pipeline condition has {self.condition_input_dim}"
                )
            normalized[domain] = policy
            action_dims[domain] = action_dim

        self.policies = nn.ModuleDict(normalized)
        self.action_dims = action_dims

    def policy_for(self, embodiment: str) -> DiffusionPolicy:
        embodiment = str(embodiment)
        if embodiment not in self.policies:
            raise KeyError(
                f"Unknown diffusion domain {embodiment!r}; "
                f"configured={sorted(self.policies)}"
            )
        return self.policies[embodiment]

    def _validate_condition(self, condition: torch.Tensor) -> None:
        if condition.ndim != 2 or condition.shape[-1] != self.condition_input_dim:
            raise ValueError(
                "Diffusion condition must have shape "
                f"(B, {self.condition_input_dim}), got {tuple(condition.shape)}"
            )

    def _validate_target(
        self, target: torch.Tensor, condition: torch.Tensor, embodiment: str
    ) -> None:
        expected = (
            int(condition.shape[0]),
            self.action_horizon,
            self.action_dims[embodiment],
        )
        if tuple(target.shape) != expected:
            raise ValueError(
                f"Diffusion target for {embodiment!r} must be {expected}, "
                f"got {tuple(target.shape)}"
            )

    def forward(self, batch: dict) -> dict:
        embodiment = str(batch["embodiment"])
        policy = self.policy_for(embodiment)
        condition = batch["condition"]
        self._validate_condition(condition)

        if self.training:
            target = batch.get("target")
            if target is None:
                raise ValueError("Diffusion Policy training requires target actions")
            self._validate_target(target, condition, embodiment)
            predicted_noise, sampled_noise = policy.predict(target, condition)
            if predicted_noise.shape != sampled_noise.shape:
                raise ValueError(
                    "Diffusion epsilon prediction shape mismatch: "
                    f"prediction={tuple(predicted_noise.shape)} "
                    f"noise={tuple(sampled_noise.shape)}"
                )
            loss = policy.loss_fn(predicted_noise, sampled_noise)
            batch["loss/diffusion_noise"] = loss
            batch["log/diffusion_noise"] = loss.detach()
            batch["log/diffusion_target_rms"] = target.detach().square().mean().sqrt()
            return batch

        prediction = policy.sample_action(condition, embodiment)
        expected = (
            int(condition.shape[0]),
            self.action_horizon,
            self.action_dims[embodiment],
        )
        if tuple(prediction.shape) != expected:
            raise ValueError(
                f"Diffusion sample for {embodiment!r} must be {expected}, "
                f"got {tuple(prediction.shape)}"
            )
        batch["pred_action"] = prediction
        batch["log/diffusion_inference_steps"] = float(policy.num_inference_steps)
        batch["log/diffusion_prediction_rms"] = (
            prediction.detach().square().mean().sqrt()
        )
        return batch
