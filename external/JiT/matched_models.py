"""Matched ImageNet objectives for JiT and end-to-end latent denoising.

The JiT path preserves the upstream clean-image prediction objective. The
endpoint-latent path never reads the target image before terminal decoding.
The unified-latent path uses the same field to tokenize images and denoise the
learned latent distribution, with no pretrained image encoder or decoder.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F
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


def _group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    raise RuntimeError(f"No valid GroupNorm divisor for {channels} channels")


class ResidualConvBlock(nn.Module):
    """Small spatial decoder block trained from scratch with the generator."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = _group_count(channels)
        self.norm_1 = nn.GroupNorm(groups, channels)
        self.conv_1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm_2 = nn.GroupNorm(groups, channels)
        self.conv_2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.conv_1(F.silu(self.norm_1(inputs)))
        hidden = self.conv_2(F.silu(self.norm_2(hidden)))
        return inputs + hidden


class SpatialImageDecoder(nn.Module):
    """Decode a latent grid with neighborhood mixing at every image scale."""

    def __init__(
        self,
        grid_size: int,
        latent_dim: int,
        patch_size: int,
        channels: Tuple[int, ...] = (192, 128, 96, 64, 32),
    ) -> None:
        super().__init__()
        levels = int(math.log2(patch_size))
        if 2**levels != patch_size:
            raise ValueError("Spatial decoder requires a power-of-two patch size")
        if len(channels) != levels + 1:
            raise ValueError(
                f"Expected {levels + 1} decoder channel widths, got {len(channels)}"
            )
        self.grid_size = int(grid_size)
        self.latent_dim = int(latent_dim)
        self.input_projection = nn.Conv2d(latent_dim, channels[0], kernel_size=1)
        self.blocks = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        for source, target in zip(channels[:-1], channels[1:]):
            self.blocks.append(ResidualConvBlock(source))
            self.upsamplers.append(
                nn.Conv2d(source, target, kernel_size=3, padding=1)
            )
        self.final_block = ResidualConvBlock(channels[-1])
        self.output = nn.Conv2d(channels[-1], 3, kernel_size=3, padding=1)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        batch, count, width = latent.shape
        expected = self.grid_size * self.grid_size
        if count != expected or width != self.latent_dim:
            raise ValueError(
                f"Expected latent {(batch, expected, self.latent_dim)}, got {tuple(latent.shape)}"
            )
        hidden = latent.transpose(1, 2).reshape(
            batch, self.latent_dim, self.grid_size, self.grid_size
        )
        hidden = self.input_projection(hidden)
        for block, upsampler in zip(self.blocks, self.upsamplers):
            hidden = block(hidden)
            hidden = F.interpolate(hidden, scale_factor=2.0, mode="nearest")
            hidden = upsampler(hidden)
        return self.output(self.final_block(hidden))


class UnifiedLatentObjective(nn.Module):
    """One shared field for image tokenization and class-conditional denoising.

    Image tokenization supplies patch tokens as cross-attention context. Latent
    generation supplies a class token instead. The learned tokenizer target is
    detached before the denoising loss, preventing the flow objective from
    shrinking the latent representation, while both modes still update the
    shared field in the same end-to-end training step.
    """

    architecture = "unified_latent"

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
        decoder_channels: Tuple[int, ...] = (192, 128, 96, 64, 32),
        num_classes: int = 1000,
        label_drop_prob: float = 0.1,
        gradient_checkpointing: bool = True,
        p_mean: float = -0.8,
        p_std: float = 0.8,
        t_eps: float = 0.05,
        reconstruction_weight: float = 1.0,
        tokenizer_t_max: float = 0.01,
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
        self.p_mean = float(p_mean)
        self.p_std = float(p_std)
        self.t_eps = float(t_eps)
        self.reconstruction_weight = float(reconstruction_weight)
        self.tokenizer_t_max = float(tokenizer_t_max)

        self.label_embedding = nn.Embedding(num_classes + 1, hidden_dim)
        self.image_patch_embedding = nn.Conv2d(
            3, hidden_dim, kernel_size=patch_size, stride=patch_size
        )
        self.image_row_position = nn.Parameter(
            torch.zeros(1, self.grid_size, 1, hidden_dim)
        )
        self.image_column_position = nn.Parameter(
            torch.zeros(1, 1, self.grid_size, hidden_dim)
        )
        self.tokenizer_mode_embedding = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.label_embedding.weight, std=0.02)
        nn.init.normal_(self.image_row_position, std=0.02)
        nn.init.normal_(self.image_column_position, std=0.02)
        nn.init.normal_(self.tokenizer_mode_embedding, std=0.02)

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
        self.latent_norm = nn.LayerNorm(latent_dim)
        self.decoder = SpatialImageDecoder(
            grid_size=self.grid_size,
            latent_dim=latent_dim,
            patch_size=patch_size,
            channels=decoder_channels,
        )

    def _image_condition(self, images: torch.Tensor) -> torch.Tensor:
        hidden = self.image_patch_embedding(images).permute(0, 2, 3, 1)
        hidden = hidden + self.image_row_position + self.image_column_position
        hidden = hidden.reshape(images.shape[0], self.num_tokens, self.hidden_dim)
        mode = self.tokenizer_mode_embedding.expand(images.shape[0], -1, -1)
        return torch.cat([mode, hidden], dim=1)

    def _class_condition(self, labels: torch.Tensor, drop: bool) -> torch.Tensor:
        if drop and self.label_drop_prob > 0:
            mask = torch.rand(labels.shape[0], device=labels.device) < self.label_drop_prob
            labels = torch.where(mask, torch.full_like(labels, self.num_classes), labels)
        return self.label_embedding(labels).unsqueeze(1)

    def _predict_clean(
        self, latent: torch.Tensor, time: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            prediction = checkpoint(
                self.field, latent, time, condition, use_reentrant=False
            )
        else:
            prediction = self.field(latent, time, condition)
        return self.latent_norm(prediction)

    def encode(
        self, images: torch.Tensor, noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch = images.shape[0]
        if noise is None:
            noise = torch.randn(
                batch, self.num_tokens, self.latent_dim, device=images.device
            )
        time = torch.rand(batch, device=images.device) * self.tokenizer_t_max
        return self._predict_clean(noise, time, self._image_condition(images))

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)

    def reconstruct(
        self, images: torch.Tensor, noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.decode(self.encode(images, noise=noise))

    @staticmethod
    def _multiscale_l1(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        losses = []
        for scale in (2, 4):
            losses.append(
                F.l1_loss(
                    F.avg_pool2d(prediction, kernel_size=scale),
                    F.avg_pool2d(target, kernel_size=scale),
                )
            )
        return torch.stack(losses).mean()

    def forward(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        optimizer_step: int,
        force_steps: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        del optimizer_step, force_steps
        target_latent = self.encode(images)
        reconstruction = self.decode(target_latent)
        reconstruction_l1 = F.l1_loss(reconstruction, images)
        reconstruction_loss = reconstruction_l1 + 0.5 * self._multiscale_l1(
            reconstruction, images
        )

        detached_target = target_latent.detach()
        logit_t = (
            torch.randn(images.shape[0], device=images.device) * self.p_std
            + self.p_mean
        )
        time = torch.sigmoid(logit_t).reshape(-1, 1, 1)
        noise = torch.randn_like(detached_target)
        state = time * detached_target + (1.0 - time) * noise
        predicted_clean = self._predict_clean(
            state, time.flatten(), self._class_condition(labels, drop=self.training)
        )
        flow_loss = F.mse_loss(predicted_clean, detached_target)
        loss = flow_loss + self.reconstruction_weight * reconstruction_loss
        flat_latent = target_latent.detach().float().reshape(-1, self.latent_dim)
        metrics = {
            "loss": loss,
            "flow_loss": flow_loss.detach(),
            "reconstruction_loss": reconstruction_loss.detach(),
            "reconstruction_l1": reconstruction_l1.detach(),
            "latent_mean": target_latent.detach().mean(),
            "latent_std": target_latent.detach().std(),
            "latent_feature_std": flat_latent.std(dim=0).mean().to(target_latent),
            "prediction_rms": predicted_clean.detach().square().mean().sqrt(),
            "noise_rms": noise.detach().square().mean().sqrt(),
            "endpoint_rms": detached_target.square().mean().sqrt(),
            "latent_delta_rms": (predicted_clean.detach() - state.detach())
            .square()
            .mean()
            .sqrt(),
            "unroll_steps": target_latent.new_tensor(1.0),
        }
        if not self.training:
            centered = flat_latent - flat_latent.mean(dim=0, keepdim=True)
            covariance = centered.T @ centered / max(centered.shape[0] - 1, 1)
            eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
            probabilities = eigenvalues / eigenvalues.sum().clamp_min(1e-12)
            effective_rank = torch.exp(
                -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
            )
            metrics["latent_effective_rank"] = effective_rank.to(target_latent)
        return metrics

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
        batch = labels.shape[0]
        latent = (
            torch.randn(batch, self.num_tokens, self.latent_dim, device=labels.device)
            if noise is None
            else noise.clone()
        )
        conditional = self._class_condition(labels, drop=False)
        unconditional = (
            self._class_condition(torch.full_like(labels, self.num_classes), drop=False)
            if cfg_scale != 1.0
            else None
        )
        for index in range(num_steps):
            time_value = index / num_steps
            time = torch.full(
                (batch,), time_value, device=labels.device, dtype=torch.float32
            )
            predicted_clean = self._predict_clean(latent, time, conditional)
            velocity = (predicted_clean - latent) / max(1.0 - time_value, self.t_eps)
            if unconditional is not None:
                uncond_clean = self._predict_clean(latent, time, unconditional)
                uncond_velocity = (uncond_clean - latent) / max(
                    1.0 - time_value, self.t_eps
                )
                velocity = uncond_velocity + cfg_scale * (velocity - uncond_velocity)
            latent = latent + velocity / num_steps
        return self.decode(latent)


def build_model(architecture: str, image_size: int = 256, num_classes: int = 1000) -> nn.Module:
    if architecture == "jit_b16":
        return JiTObjective(image_size=image_size, num_classes=num_classes)
    if architecture == "endpoint_latent":
        return EndpointLatentObjective(image_size=image_size, num_classes=num_classes)
    if architecture == "unified_latent":
        return UnifiedLatentObjective(image_size=image_size, num_classes=num_classes)
    raise ValueError(f"Unknown architecture: {architecture}")


def trainable_parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
