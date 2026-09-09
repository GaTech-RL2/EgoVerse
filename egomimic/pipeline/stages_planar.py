"""Compact Planar-only stages layered on the generic Pipeline core."""

from __future__ import annotations

import math
from fractions import Fraction

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from egomimic.pipeline.core import Stage, resolve_homogeneous_scalar
from egomimic.rldb.embodiment.embodiment import get_embodiment


class TokenwiseActionDecoder(nn.Sequential):
    """Decode every latent token independently into a common Planar action."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        action_dim: int = 5,
        num_layers: int = 3,
    ):
        latent_dim = int(latent_dim)
        hidden_dim = int(hidden_dim)
        action_dim = int(action_dim)
        if min(latent_dim, hidden_dim, action_dim) <= 0:
            raise ValueError("decoder dimensions must be positive")
        if num_layers < 2:
            raise ValueError("num_layers must be at least two")
        layers = []
        for index in range(int(num_layers)):
            input_dim = latent_dim if index == 0 else hidden_dim
            output_dim = action_dim if index == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(input_dim, output_dim))
            if index < num_layers - 1:
                layers.append(nn.SiLU())
        # Keep the registered parameter tree compact: ``0.weight``, ``2.weight``, ...
        super().__init__(*layers)
        self.latent_dim = latent_dim
        self.action_dim = action_dim

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 3 or latent.shape[-1] != self.latent_dim:
            raise ValueError(
                f"expected latent shape (B, T, {self.latent_dim}), got {latent.shape}"
            )
        return super().forward(latent)


class PlanarFlowSampler(Stage):
    """Integrate a learned latent vector field and decode a Planar action chunk."""

    reads = ("sampler/noise", "condition", "embodiment")
    writes = ("sampler/endpoint", "pred_action", "log/*")

    def __init__(
        self,
        denoising_module: nn.Module,
        condition_input_dim: int,
        action_horizon: int,
        action_dims: dict[str, int],
        latent_dim: int = 96,
        condition_dim: int = 256,
        decoder_hidden_dim: int = 256,
        decoder_num_layers: int = 3,
        denoiser_hidden_dim: int = 384,
        num_inference_steps: int = 8,
        sampling_schedule: dict | None = None,
        gradient_checkpointing: bool = True,
        gradient_accumulation_steps: int = 1,
    ):
        super().__init__()
        self.denoising_module = denoising_module
        self.condition_input_dim = int(condition_input_dim)
        self.action_horizon = int(action_horizon)
        self.action_dims = {str(key): int(value) for key, value in action_dims.items()}
        self.latent_dim = int(latent_dim)
        self.condition_dim = int(condition_dim)
        self.denoiser_hidden_dim = int(denoiser_hidden_dim)
        self.num_inference_steps = int(num_inference_steps)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.gradient_accumulation_steps = int(gradient_accumulation_steps)
        if (
            min(
                self.condition_input_dim,
                self.action_horizon,
                self.latent_dim,
                self.condition_dim,
                self.num_inference_steps,
                self.gradient_accumulation_steps,
            )
            <= 0
        ):
            raise ValueError("sampler dimensions and step counts must be positive")
        if not self.action_dims or any(
            value <= 0 for value in self.action_dims.values()
        ):
            raise ValueError("action_dims must contain positive widths")

        self.condition_projection = nn.Linear(
            self.condition_input_dim, self.condition_dim
        )
        self.domain_embeddings = nn.ParameterDict(
            {
                domain: nn.Parameter(torch.empty(self.condition_dim).normal_(std=0.02))
                for domain in self.action_dims
            }
        )
        self.decoders = nn.ModuleDict(
            {
                domain: TokenwiseActionDecoder(
                    latent_dim=self.latent_dim,
                    hidden_dim=decoder_hidden_dim,
                    action_dim=action_dim,
                    num_layers=decoder_num_layers,
                )
                for domain, action_dim in self.action_dims.items()
            }
        )
        default_schedule = {
            1: {1: 0.5, 2: 0.5},
            2001: {2: 0.8, 4: 0.15, 8: 0.05},
        }
        self.schedule = self._compile_schedule(sampling_schedule or default_schedule)
        self.register_buffer("training_batches_seen", torch.zeros((), dtype=torch.long))
        self.last_integration_step_sizes = None
        self._validate_denoiser()

    @staticmethod
    def _compile_schedule(schedule: dict) -> dict[int, tuple[int, ...]]:
        cycles = {}
        for raw_start, raw_weights in schedule.items():
            start = int(raw_start)
            weights = {int(key): float(value) for key, value in raw_weights.items()}
            if start < 1 or not weights or any(key <= 0 for key in weights):
                raise ValueError("sampling schedule keys must be positive")
            if any(value <= 0 for value in weights.values()) or not math.isclose(
                sum(weights.values()), 1.0, abs_tol=1e-8
            ):
                raise ValueError(
                    "sampling schedule weights must be positive and sum to one"
                )
            fractions = {
                key: Fraction(str(value)).limit_denominator(1000)
                for key, value in weights.items()
            }
            denominator = math.lcm(*(value.denominator for value in fractions.values()))
            counts = {
                key: value.numerator * (denominator // value.denominator)
                for key, value in fractions.items()
            }
            divisor = math.gcd(*counts.values())
            cycles[start] = tuple(
                step for step in sorted(counts) for _ in range(counts[step] // divisor)
            )
        if not cycles or min(cycles) != 1:
            raise ValueError("sampling schedule must begin at optimizer step one")
        return dict(sorted(cycles.items()))

    def _validate_denoiser(self):
        input_projection = getattr(self.denoising_module, "proj_u", None)
        output_projection = getattr(self.denoising_module, "proj_d", None)
        if (
            input_projection is not None
            and input_projection.in_features != self.latent_dim
        ):
            raise ValueError("denoiser input width does not match latent_dim")
        if (
            output_projection is not None
            and output_projection.out_features != self.latent_dim
        ):
            raise ValueError("denoiser output width does not match latent_dim")
        if (
            output_projection is not None
            and output_projection.in_features != self.denoiser_hidden_dim
        ):
            raise ValueError("denoiser hidden width does not match denoiser_hidden_dim")

    def unroll_steps_at(self, optimizer_step: int) -> int:
        optimizer_step = max(1, int(optimizer_step))
        start = max(value for value in self.schedule if value <= optimizer_step)
        cycle = self.schedule[start]
        return cycle[(optimizer_step - start) % len(cycle)]

    @staticmethod
    def sample_step_sizes(
        batch_size: int,
        num_steps: int,
        reference: torch.Tensor,
        generator=None,
    ) -> torch.Tensor:
        if num_steps == 1:
            return torch.ones(batch_size, 1, device=reference.device)
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        interior = (
            torch.rand(
                batch_size,
                num_steps - 1,
                dtype=torch.float64,
                device=reference.device,
                generator=generator,
            )
            .sort(dim=-1)
            .values
        )
        endpoints = torch.cat(
            (
                torch.zeros(
                    batch_size, 1, dtype=torch.float64, device=reference.device
                ),
                interior,
                torch.ones(batch_size, 1, dtype=torch.float64, device=reference.device),
            ),
            dim=-1,
        )
        return endpoints.diff(dim=-1).float()

    def _velocity(self, latent, time, condition):
        if self.training and self.gradient_checkpointing and torch.is_grad_enabled():
            return checkpoint(
                self.denoising_module,
                latent,
                time,
                condition,
                use_reentrant=False,
            )
        return self.denoising_module(latent, time, condition)

    def _resolve_branch(self, selector) -> str:
        branch = resolve_homogeneous_scalar(selector, label="Planar branch selector")
        if isinstance(branch, int):
            branch = get_embodiment(branch)
            if branch is None:
                raise KeyError("unknown Planar branch selector")
            branch = branch.lower()
        else:
            branch = str(branch)
        if branch not in self.decoders:
            raise KeyError(f"unknown Planar branch {branch!r}")
        return branch

    def integrate(self, latent, condition, num_steps, step_sizes=None):
        batch_size = latent.shape[0]
        if step_sizes is None:
            step_sizes = torch.full(
                (batch_size, num_steps),
                1.0 / num_steps,
                device=latent.device,
                dtype=torch.float32,
            )
        if step_sizes.shape != (batch_size, num_steps):
            raise ValueError("integration grid has the wrong shape")
        if not bool(torch.all(step_sizes > 0)) or not torch.allclose(
            step_sizes.sum(dim=-1),
            torch.ones(batch_size, device=latent.device),
            atol=1e-6,
            rtol=1e-6,
        ):
            raise ValueError("each integration grid must be positive and sum to one")
        time = torch.zeros(batch_size, device=latent.device, dtype=torch.float32)
        for index in range(num_steps):
            delta = step_sizes[:, index]
            latent = latent + delta[:, None, None] * self._velocity(
                latent, time, condition
            )
            time = time + delta
        self.last_integration_step_sizes = step_sizes.detach()
        return latent

    def _execute_mode(self, batch: dict, mode: str) -> dict:
        branch = self._resolve_branch(batch["embodiment"])
        noise = batch["sampler/noise"]
        condition = batch["condition"]
        expected = (condition.shape[0], self.action_horizon, self.latent_dim)
        if tuple(noise.shape) != expected:
            raise ValueError(
                f"expected sampler/noise shape {expected}, got {noise.shape}"
            )
        if condition.shape[-1] != self.condition_input_dim:
            raise ValueError(
                f"expected condition width {self.condition_input_dim}, got {condition.shape[-1]}"
            )
        condition = self.condition_projection(condition).unsqueeze(1)
        condition = condition + self.domain_embeddings[branch].view(1, 1, -1)
        if mode == "train":
            self.training_batches_seen.add_(1)
            optimizer_step = (
                int(self.training_batches_seen.item()) - 1
            ) // self.gradient_accumulation_steps + 1
            num_steps = self.unroll_steps_at(optimizer_step)
            step_sizes = self.sample_step_sizes(noise.shape[0], num_steps, noise)
            batch["log/optimizer_step"] = float(optimizer_step)
        elif mode == "inference":
            num_steps = self.num_inference_steps
            step_sizes = None
        else:
            raise ValueError(f"Unsupported Planar execution mode {mode!r}")
        endpoint = self.integrate(noise, condition, num_steps, step_sizes)
        prediction = self.decoders[branch](endpoint)
        batch["sampler/endpoint"] = endpoint
        batch["pred_action"] = prediction
        batch["log/sampler_unroll_steps"] = float(num_steps)
        batch["log/sampler_endpoint_rms"] = endpoint.detach().square().mean().sqrt()
        return batch

    def execute(self, batch: dict, *, mode: str) -> dict:
        return self._execute_mode(batch, mode)

    def forward(self, batch: dict) -> dict:
        """Retain direct-call behavior while graph execution passes mode explicitly."""
        return self._execute_mode(
            batch, "train" if self.training else "inference"
        )


class PlanarActionMSELoss(Stage):
    """Strict train-only normalized common-action MSE."""

    train_only = True
    reads = ("pred_action", "target")
    writes = ("loss/action", "log/MSE")

    def forward(self, batch: dict) -> dict:
        prediction, target = batch["pred_action"], batch["target"]
        if prediction.shape != target.shape:
            raise ValueError(
                f"action shape mismatch: {prediction.shape} versus {target.shape}"
            )
        error = (prediction - target).square()
        mask = batch.get("pad_mask")
        if mask is None:
            loss = error.mean()
        else:
            mask = mask.to(device=error.device, dtype=error.dtype)
            while mask.ndim < error.ndim:
                mask = mask.unsqueeze(-1)
            loss = (error * mask).sum() / mask.expand_as(error).sum().clamp_min(1)
        batch["loss/action"] = loss
        batch["log/MSE"] = loss.detach()
        return batch
