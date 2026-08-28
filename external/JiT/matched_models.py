"""Matched ImageNet objectives for JiT and decoder-only latent denoising.

The JiT path preserves the upstream clean-image prediction objective. The
endpoint-latent path never reads the target image before terminal decoding.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from egomimic.models.denoising_nets import CrossBlock, posemb_sincos
from model_jit import JiT_models


def _unpatchify(tokens: torch.Tensor, patch_size: int, channels: int = 3) -> torch.Tensor:
    batch, count, width = tokens.shape
    side = math.isqrt(count)
    if side * side != count:
        raise ValueError(f"Token count {count} is not a square grid")
    expected = patch_size * patch_size * channels
    if width != expected:
        raise ValueError(f"Patch width {width} does not match expected {expected}")
    x = tokens.reshape(batch, side, side, patch_size, patch_size, channels)
    x = torch.einsum("nhwpqc->nchpwq", x)
    return x.reshape(batch, channels, side * patch_size, side * patch_size)


class JiTObjective(nn.Module):
    """Official JiT-B/16 network and direct clean-image prediction loss."""

    architecture = "jit_b16"

    def __init__(
        self,
        image_size: int = 256,
        num_classes: int = 1000,
        label_drop_prob: float = 0.1,
        p_mean: float = -0.8,
        p_std: float = 0.8,
        noise_scale: float = 1.0,
        t_eps: float = 0.05,
    ) -> None:
        super().__init__()
        self.net = JiT_models["JiT-B/16"](
            input_size=image_size,
            in_channels=3,
            num_classes=num_classes,
            attn_drop=0.0,
            proj_drop=0.0,
        )
        self.image_size = int(image_size)
        self.num_classes = int(num_classes)
        self.label_drop_prob = float(label_drop_prob)
        self.p_mean = float(p_mean)
        self.p_std = float(p_std)
        self.noise_scale = float(noise_scale)
        self.t_eps = float(t_eps)

    def _drop_labels(self, labels: torch.Tensor) -> torch.Tensor:
        if not self.training or self.label_drop_prob <= 0:
            return labels
        drop = torch.rand(labels.shape[0], device=labels.device) < self.label_drop_prob
        return torch.where(drop, torch.full_like(labels, self.num_classes), labels)

    def forward(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        optimizer_step: int,
        force_steps: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        del optimizer_step, force_steps
        labels = self._drop_labels(labels)
        logit_t = torch.randn(images.shape[0], device=images.device) * self.p_std + self.p_mean
        t = torch.sigmoid(logit_t).reshape(-1, 1, 1, 1)
        noise = torch.randn_like(images) * self.noise_scale
        state = t * images + (1.0 - t) * noise
        target_velocity = (images - state) / (1.0 - t).clamp_min(self.t_eps)
        prediction = self.net(state, t.flatten(), labels)
        predicted_velocity = (prediction - state) / (1.0 - t).clamp_min(self.t_eps)
        loss = (target_velocity - predicted_velocity).square().mean()
        return {
            "loss": loss,
            "prediction_rms": prediction.detach().square().mean().sqrt(),
            "noise_rms": noise.detach().square().mean().sqrt(),
            "endpoint_rms": prediction.detach().square().mean().sqrt(),
            "latent_delta_rms": (prediction.detach() - state.detach()).square().mean().sqrt(),
            "unroll_steps": torch.ones((), device=images.device),
        }

    @torch.no_grad()
    def sample(
        self,
        labels: torch.Tensor,
        num_steps: int = 16,
        cfg_scale: float = 1.0,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        state = (
            torch.randn(
                labels.shape[0], 3, self.image_size, self.image_size,
                device=labels.device,
            )
            * self.noise_scale
            if noise is None
            else noise.clone()
        )
        for index in range(num_steps):
            t_value = index / num_steps
            t = torch.full(
                (labels.shape[0],), t_value, device=labels.device, dtype=torch.float32
            )
            shaped_t = t.reshape(-1, 1, 1, 1)
            conditional = self.net(state, t, labels)
            velocity = (conditional - state) / (1.0 - shaped_t).clamp_min(self.t_eps)
            if cfg_scale != 1.0:
                null_labels = torch.full_like(labels, self.num_classes)
                unconditional = self.net(state, t, null_labels)
                uncond_velocity = (unconditional - state) / (
                    1.0 - shaped_t
                ).clamp_min(self.t_eps)
                velocity = uncond_velocity + cfg_scale * (velocity - uncond_velocity)
            state = state + velocity / num_steps
        return state


class ImageCrossTransformer(nn.Module):
    """Existing Pipeline cross-transformer with explicit factorized 2-D positions."""

    def __init__(
        self,
        grid_size: int,
        latent_dim: int,
        hidden_dim: int,
        depth: int,
        num_heads: int,
        dropout: float,
        mlp_layers: int,
        mlp_ratio: float,
    ) -> None:
        super().__init__()
        if hidden_dim < latent_dim:
            raise ValueError("hidden_dim must not bottleneck latent_dim")
        self.grid_size = int(grid_size)
        self.hidden_dim = int(hidden_dim)
        self.proj_u = nn.Linear(latent_dim, hidden_dim)
        self.proj_d = nn.Linear(hidden_dim, latent_dim)
        self.row_position = nn.Parameter(torch.zeros(1, grid_size, 1, hidden_dim))
        self.column_position = nn.Parameter(torch.zeros(1, 1, grid_size, hidden_dim))
        nn.init.normal_(self.row_position, std=0.02)
        nn.init.normal_(self.column_position, std=0.02)
        self.layers = nn.ModuleList(
            [
                CrossBlock(
                    cond_dim=hidden_dim,
                    hidden_dim=hidden_dim,
                    n_heads=num_heads,
                    dropout=dropout,
                    mlp_layers=mlp_layers,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(depth)
            ]
        )

    def forward(
        self, latent: torch.Tensor, timesteps: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        batch, count, _ = latent.shape
        expected = self.grid_size * self.grid_size
        if count != expected:
            raise ValueError(f"Expected {expected} latent tokens, got {count}")
        hidden = self.proj_u(latent).reshape(
            batch, self.grid_size, self.grid_size, self.hidden_dim
        )
        hidden = hidden + self.row_position + self.column_position
        hidden = hidden.reshape(batch, count, self.hidden_dim)
        time_embedding = posemb_sincos(
            timesteps, self.hidden_dim, min_period=4e-3, max_period=4.0
        ).to(device=hidden.device, dtype=hidden.dtype)
        hidden = hidden + time_embedding.unsqueeze(1)
        for layer in self.layers:
            hidden = layer(hidden, condition)
        return self.proj_d(hidden)


@dataclass(frozen=True)
class IntegrationResult:
    endpoint: torch.Tensor
    delta_rms: torch.Tensor
    step_sizes: torch.Tensor


class EndpointLatentObjective(nn.Module):
    """Strict decoder-only endpoint-trained iterative latent image generator."""

    architecture = "endpoint_latent"

    def __init__(
        self,
        image_size: int = 256,
        patch_size: int = 16,
        latent_dim: int = 96,
        hidden_dim: int = 352,
        depth: int = 16,
        num_heads: int = 8,
        dropout: float = 0.1,
        mlp_layers: int = 4,
        mlp_ratio: float = 4.0,
        decoder_hidden_dim: int = 512,
        num_classes: int = 1000,
        label_drop_prob: float = 0.1,
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        if image_size % patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.grid_size = image_size // patch_size
        self.num_tokens = self.grid_size * self.grid_size
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.label_drop_prob = float(label_drop_prob)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.label_embedding = nn.Embedding(num_classes + 1, hidden_dim)
        nn.init.normal_(self.label_embedding.weight, std=0.02)
        self.field = ImageCrossTransformer(
            grid_size=self.grid_size,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            num_heads=num_heads,
            dropout=dropout,
            mlp_layers=mlp_layers,
            mlp_ratio=mlp_ratio,
        )
        patch_width = patch_size * patch_size * 3
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, decoder_hidden_dim),
            nn.SiLU(),
            nn.Linear(decoder_hidden_dim, decoder_hidden_dim),
            nn.SiLU(),
            nn.Linear(decoder_hidden_dim, patch_width),
        )

    @staticmethod
    def unroll_steps_at(optimizer_step: int) -> int:
        step = max(int(optimizer_step), 1)
        if step <= 2000:
            return 1 if step % 2 else 2
        cycle = (2,) * 16 + (4,) * 3 + (8,)
        return cycle[(step - 2001) % len(cycle)]

    @staticmethod
    def sample_step_sizes(
        batch_size: int,
        num_steps: int,
        device: torch.device,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if num_steps == 1:
            return torch.ones(batch_size, 1, device=device, dtype=torch.float32)
        interior = torch.rand(
            batch_size,
            num_steps - 1,
            device=device,
            dtype=torch.float64,
            generator=generator,
        ).sort(dim=-1).values
        endpoints = torch.cat(
            [
                torch.zeros(batch_size, 1, device=device, dtype=torch.float64),
                interior,
                torch.ones(batch_size, 1, device=device, dtype=torch.float64),
            ],
            dim=-1,
        )
        steps = endpoints.diff(dim=-1).to(torch.float32)
        if not bool(torch.all(steps > 0)):
            raise RuntimeError("Integration grid contains a non-positive step")
        if not torch.allclose(
            steps.sum(dim=-1), torch.ones(batch_size, device=device), atol=1e-6, rtol=1e-6
        ):
            raise RuntimeError("Integration grid does not sum to one")
        return steps

    def _condition(self, labels: torch.Tensor, drop: bool) -> torch.Tensor:
        if drop and self.label_drop_prob > 0:
            mask = torch.rand(labels.shape[0], device=labels.device) < self.label_drop_prob
            labels = torch.where(mask, torch.full_like(labels, self.num_classes), labels)
        return self.label_embedding(labels).unsqueeze(1)

    def _velocity(
        self, latent: torch.Tensor, time: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(self.field, latent, time, condition, use_reentrant=False)
        return self.field(latent, time, condition)

    def integrate(
        self,
        initial_latent: torch.Tensor,
        condition: torch.Tensor,
        num_steps: int,
        step_sizes: Optional[torch.Tensor] = None,
    ) -> IntegrationResult:
        batch = initial_latent.shape[0]
        if step_sizes is None:
            step_sizes = torch.full(
                (batch, num_steps), 1.0 / num_steps,
                device=initial_latent.device, dtype=torch.float32,
            )
        if step_sizes.shape != (batch, num_steps):
            raise ValueError(
                f"Expected step_sizes {(batch, num_steps)}, got {tuple(step_sizes.shape)}"
            )
        if not bool(torch.all(step_sizes > 0)):
            raise ValueError("All integration steps must be positive")
        if not torch.allclose(
            step_sizes.sum(-1), torch.ones(batch, device=step_sizes.device),
            atol=1e-6, rtol=1e-6,
        ):
            raise ValueError("Each integration grid must sum to one")
        latent = initial_latent
        time = torch.zeros(batch, device=latent.device, dtype=torch.float32)
        deltas = []
        for index in range(num_steps):
            velocity = self._velocity(latent, time, condition)
            dt = step_sizes[:, index].reshape(batch, 1, 1)
            delta = dt * velocity
            latent = latent + delta
            deltas.append(delta.detach().square().mean())
            time = time + step_sizes[:, index]
        delta_rms = torch.stack(deltas).mean().sqrt()
        return IntegrationResult(latent, delta_rms, step_sizes)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        patches = self.decoder(latent)
        return _unpatchify(patches, self.patch_size, channels=3)

    def predict(
        self,
        labels: torch.Tensor,
        optimizer_step: int,
        force_steps: Optional[int] = None,
        noise: Optional[torch.Tensor] = None,
        step_sizes: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, IntegrationResult]:
        batch = labels.shape[0]
        if noise is None:
            noise = torch.randn(
                batch, self.num_tokens, self.latent_dim, device=labels.device
            )
        condition = self._condition(labels, drop=self.training)
        num_steps = int(force_steps or self.unroll_steps_at(optimizer_step))
        if step_sizes is None:
            step_sizes = self.sample_step_sizes(batch, num_steps, labels.device)
        result = self.integrate(noise, condition, num_steps, step_sizes)
        return self.decode(result.endpoint), result

    def forward(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        optimizer_step: int,
        force_steps: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        # The target image is deliberately consumed only after latent generation
        # and terminal decoding have completed.
        prediction, result = self.predict(labels, optimizer_step, force_steps=force_steps)
        loss = (prediction - images).square().mean()
        return {
            "loss": loss,
            "prediction_rms": prediction.detach().square().mean().sqrt(),
            "noise_rms": result.endpoint.new_tensor(1.0),
            "endpoint_rms": result.endpoint.detach().square().mean().sqrt(),
            "latent_delta_rms": result.delta_rms.detach(),
            "unroll_steps": result.endpoint.new_tensor(result.step_sizes.shape[1]),
        }

    @torch.no_grad()
    def sample(
        self,
        labels: torch.Tensor,
        num_steps: int = 16,
        cfg_scale: float = 1.0,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch = labels.shape[0]
        latent = (
            torch.randn(batch, self.num_tokens, self.latent_dim, device=labels.device)
            if noise is None else noise.clone()
        )
        conditional = self._condition(labels, drop=False)
        unconditional = self._condition(
            torch.full_like(labels, self.num_classes), drop=False
        ) if cfg_scale != 1.0 else None
        time = torch.zeros(batch, device=labels.device, dtype=torch.float32)
        for _ in range(num_steps):
            velocity = self.field(latent, time, conditional)
            if unconditional is not None:
                uncond_velocity = self.field(latent, time, unconditional)
                velocity = uncond_velocity + cfg_scale * (velocity - uncond_velocity)
            latent = latent + velocity / num_steps
            time = time + 1.0 / num_steps
        return self.decode(latent)


def build_model(architecture: str, image_size: int = 256, num_classes: int = 1000) -> nn.Module:
    if architecture == "jit_b16":
        return JiTObjective(image_size=image_size, num_classes=num_classes)
    if architecture == "endpoint_latent":
        return EndpointLatentObjective(image_size=image_size, num_classes=num_classes)
    raise ValueError(f"Unknown architecture: {architecture}")


def trainable_parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
