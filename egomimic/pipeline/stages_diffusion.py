"""Factorized Pipeline stages for epsilon-prediction Diffusion Policy."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F

from egomimic.models.diffusion_policy import DiffusionPolicy
from egomimic.pipeline.core import Stage

_SCHEDULER_FIELDS = (
    "num_train_timesteps",
    "beta_start",
    "beta_end",
    "beta_schedule",
    "variance_type",
    "clip_sample",
    "set_alpha_to_one",
    "steps_offset",
    "prediction_type",
)


def _scheduler_value(scheduler, name: str):
    """Read a setting from either a repository-local or diffusers scheduler."""
    value = getattr(scheduler, name, None)
    if value is not None:
        return value
    config = getattr(scheduler, "config", None)
    if config is None:
        return None
    if isinstance(config, Mapping):
        return config.get(name)
    return getattr(config, name, None)


def _scheduler_signature(scheduler) -> tuple:
    """Return the forward/reverse-process identity shared across DP nodes."""
    return (
        f"{type(scheduler).__module__}.{type(scheduler).__qualname__}",
        tuple(
            (name, repr(_scheduler_value(scheduler, name)))
            for name in _SCHEDULER_FIELDS
        ),
    )


def _validate_epsilon_scheduler(scheduler, label: str) -> None:
    prediction_type = _scheduler_value(scheduler, "prediction_type")
    if prediction_type != "epsilon":
        raise ValueError(
            f"Scheduler for {label!r} must use the epsilon noise objective; "
            f"prediction_type={prediction_type!r}"
        )
    train_steps = _scheduler_value(scheduler, "num_train_timesteps")
    if train_steps is None or int(train_steps) <= 0:
        raise ValueError(
            f"Scheduler for {label!r} has no positive train timestep count"
        )


def _validate_action_shape(
    tensor: torch.Tensor,
    *,
    batch_size: int,
    action_horizon: int,
    action_dim: int,
    label: str,
) -> None:
    expected = (batch_size, action_horizon, action_dim)
    if tuple(tensor.shape) != expected:
        raise ValueError(f"{label} must be {expected}, got {tuple(tensor.shape)}")


class DiffusionNoisingStage(Stage):
    """Apply the forward diffusion process without route-specific behavior."""

    train_only = True
    reads = ("target",)
    writes = (
        "diffusion/noisy_action",
        "diffusion/noise_target",
        "diffusion/timestep",
        "diffusion/scheduler_signature",
    )
    objective = "epsilon"

    def __init__(self, noise_scheduler, action_horizon: int, action_dim: int):
        super().__init__()
        self.noise_scheduler = noise_scheduler
        self.action_horizon = int(action_horizon)
        self.action_dim = int(action_dim)
        if self.action_horizon <= 0 or self.action_dim <= 0:
            raise ValueError("action_horizon and action_dim must be positive")
        _validate_epsilon_scheduler(self.noise_scheduler, "diffusion")

    def forward(self, batch: dict) -> dict:
        target = batch["target"]
        _validate_action_shape(
            target,
            batch_size=int(target.shape[0]),
            action_horizon=self.action_horizon,
            action_dim=self.action_dim,
            label="Diffusion target",
        )
        noise = torch.randn(target.shape, device=target.device)
        timesteps = torch.randint(
            0,
            int(_scheduler_value(self.noise_scheduler, "num_train_timesteps")),
            (int(target.shape[0]),),
            device=target.device,
        ).long()
        batch["diffusion/noisy_action"] = self.noise_scheduler.add_noise(
            target, noise, timesteps
        )
        batch["diffusion/noise_target"] = noise
        batch["diffusion/timestep"] = timesteps
        batch["diffusion/scheduler_signature"] = _scheduler_signature(
            self.noise_scheduler
        )
        return batch


class DiffusionDenoiserStage(Stage):
    """Predict epsilon or sample actions with one route-free policy."""

    reads = (
        "condition",
        "diffusion/noisy_action",
        "diffusion/timestep",
        "diffusion/scheduler_signature",
    )
    writes = ("diffusion/predicted_noise",)
    reads_by_mode = {"inference": ("condition",)}
    writes_by_mode = {"inference": ("pred_action", "log/*")}
    objective = "epsilon"

    def __init__(
        self,
        policy: DiffusionPolicy,
        action_horizon: int,
        action_dim: int,
        condition_input_dim: int,
    ):
        super().__init__()
        self.action_horizon = int(action_horizon)
        self.action_dim = int(action_dim)
        self.condition_input_dim = int(condition_input_dim)
        if (
            self.action_horizon <= 0
            or self.action_dim <= 0
            or self.condition_input_dim <= 0
        ):
            raise ValueError(
                "action_horizon, action_dim, and condition_input_dim must be positive"
            )
        if not isinstance(policy, DiffusionPolicy):
            raise TypeError(
                f"policy must be DiffusionPolicy, got {type(policy).__name__}"
            )
        if int(policy.action_horizon) != self.action_horizon:
            raise ValueError(
                f"Policy horizon={policy.action_horizon}, expected {self.action_horizon}"
            )
        _validate_epsilon_scheduler(policy.noise_scheduler, "diffusion")
        if policy.num_inference_steps is None or int(policy.num_inference_steps) <= 0:
            raise ValueError("Policy must declare positive num_inference_steps")
        self._validate_model_condition(policy)
        self.policy = policy

    def _validate_model_condition(self, policy: DiffusionPolicy) -> None:
        proj_cond = getattr(policy.model, "proj_cond", None)
        if (
            proj_cond is not None
            and int(proj_cond.in_features) != self.condition_input_dim
        ):
            raise ValueError(
                f"Policy expects {proj_cond.in_features} condition features, "
                f"configured Pipeline condition has {self.condition_input_dim}"
            )
        global_cond_dim = getattr(policy.model, "global_cond_dim", None)
        if (
            global_cond_dim is not None
            and int(global_cond_dim) != self.condition_input_dim
        ):
            raise ValueError(
                f"Policy expects {global_cond_dim} condition features, configured "
                f"Pipeline condition has {self.condition_input_dim}"
            )

    def _validate_condition(self, condition: torch.Tensor) -> None:
        if condition.ndim != 2 or int(condition.shape[-1]) != self.condition_input_dim:
            raise ValueError(
                "Diffusion condition must have shape "
                f"(B, {self.condition_input_dim}), got {tuple(condition.shape)}"
            )

    def _forward_train(self, batch: dict) -> dict:
        policy = self.policy
        condition = batch["condition"]
        self._validate_condition(condition)

        expected_signature = _scheduler_signature(policy.noise_scheduler)
        if batch["diffusion/scheduler_signature"] != expected_signature:
            raise ValueError("Forward and reverse diffusion schedulers differ")
        noisy_action = batch["diffusion/noisy_action"]
        _validate_action_shape(
            noisy_action,
            batch_size=int(condition.shape[0]),
            action_horizon=self.action_horizon,
            action_dim=self.action_dim,
            label="Noisy diffusion action",
        )
        prediction = policy.model(noisy_action, batch["diffusion/timestep"], condition)
        if prediction.shape != noisy_action.shape:
            raise ValueError(
                "Diffusion epsilon prediction shape mismatch: "
                f"prediction={tuple(prediction.shape)} "
                f"noisy_action={tuple(noisy_action.shape)}"
            )
        batch["diffusion/predicted_noise"] = prediction
        return batch

    def _forward_inference(self, batch: dict) -> dict:
        policy = self.policy
        condition = batch["condition"]
        self._validate_condition(condition)
        prediction = policy.sample_action(condition, action_dim=self.action_dim)
        _validate_action_shape(
            prediction,
            batch_size=int(condition.shape[0]),
            action_horizon=self.action_horizon,
            action_dim=self.action_dim,
            label="Diffusion sample",
        )
        batch["pred_action"] = prediction
        batch["log/diffusion_inference_steps"] = float(policy.num_inference_steps)
        batch["log/diffusion_prediction_rms"] = (
            prediction.detach().square().mean().sqrt()
        )
        return batch

    def execute(self, batch: dict, *, mode: str) -> dict:
        if mode == "train":
            return self._forward_train(batch)
        if mode == "inference":
            return self._forward_inference(batch)
        raise ValueError(f"Unsupported diffusion execution mode {mode!r}")

    def forward(self, batch: dict) -> dict:
        """Retain direct-call training behavior; graph mode is always explicit."""
        return self._forward_train(batch)


class DiffusionEpsilonLossStage(Stage):
    """Apply epsilon-prediction MSE as an explicit leaf node."""

    train_only = True
    reads = ("diffusion/predicted_noise", "diffusion/noise_target", "target")
    writes = ("loss/diffusion_noise", "log/*")
    objective = "epsilon"

    def forward(self, batch: dict) -> dict:
        prediction = batch["diffusion/predicted_noise"]
        noise_target = batch["diffusion/noise_target"]
        if prediction.shape != noise_target.shape:
            raise ValueError(
                "Diffusion epsilon prediction shape mismatch: "
                f"prediction={tuple(prediction.shape)} "
                f"target={tuple(noise_target.shape)}"
            )
        loss = F.mse_loss(prediction, noise_target)
        batch["loss/diffusion_noise"] = loss
        batch["log/diffusion_noise"] = loss.detach()
        batch["log/diffusion_target_rms"] = (
            batch["target"].detach().square().mean().sqrt()
        )
        return batch
