"""Released UNITE stages for compact action-register latents."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from egomimic.pipeline.core import Stage


def _mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError(
            "Released UNITE tensor shape mismatch: "
            f"prediction={tuple(prediction.shape)} target={tuple(target.shape)}"
        )
    return (prediction - target).square().mean()


class _PerEmbodimentDecoder(nn.Module):
    """Small decoder router that preserves released checkpoint key names."""

    def __init__(self, decoders: Dict[str, nn.Module]):
        super().__init__()
        configured = {str(domain): decoder for domain, decoder in decoders.items()}
        if not configured or any(
            not isinstance(decoder, nn.Module) for decoder in configured.values()
        ):
            raise ValueError("decoders must contain at least one nn.Module")
        self.decoders = nn.ModuleDict(configured)
        self.domains = tuple(configured)
        latent_dims = {
            int(getattr(decoder, "latent_dim", -1)) for decoder in configured.values()
        }
        if len(latent_dims) != 1 or min(latent_dims) <= 0:
            raise ValueError("All decoders must expose the same positive latent_dim")
        self.latent_dim = latent_dims.pop()
        self.action_dims = {
            domain: int(getattr(decoder, "action_dim", -1))
            for domain, decoder in configured.items()
        }
        if any(action_dim <= 0 for action_dim in self.action_dims.values()):
            raise ValueError("Every decoder must expose a positive action_dim")

    def decoder_for(self, embodiment: str) -> nn.Module:
        embodiment = str(embodiment)
        if embodiment not in self.decoders:
            raise KeyError(
                f"Unknown UNITE embodiment {embodiment!r}; configured={self.domains}"
            )
        return self.decoders[embodiment]


class ReleasedRecipeUniteLatentPolicy(Stage):
    """Joint reconstruction/flow UNITE policy with Dopri5 and CFG."""

    reads = ["sampler/noise", "condition", "target", "embodiment"]
    writes = [
        "unite/clean_latent",
        "unite/reconstructed_action",
        "unite/flow_loss",
    ]
    reads_by_mode = {
        "inference": ["sampler/noise", "condition", "embodiment"],
    }
    writes_by_mode = {
        "inference": ["sampler/endpoint", "pred_action"],
    }

    def __init__(
        self,
        generative_encoder: nn.Module,
        decoders: Dict[str, nn.Module],
        timestep_shift_alpha: float = 0.5,
        flow_steps_per_reconstruction: int = 14,
        flow_mini_batch: int = 14,
        train_eps: float = 0.05,
        sample_eps: float = 0.05,
        lognorm_mu: float = 0.0,
        lognorm_sigma: float = 1.0,
        reconstruction_noising_start: float = 0.7,
        reconstruction_noising_probability: float = 0.5,
        condition_dropout_probability: float = 0.1,
        cfg_scale: float = 4.0,
        cfg_interval: tuple[float, float] = (0.0, 1.0),
        dopri5_output_points: int = 50,
        dopri5_atol: float = 1.0e-6,
        dopri5_rtol: float = 1.0e-3,
    ):
        super().__init__()
        self.generative_encoder = generative_encoder
        self.action_decoder = _PerEmbodimentDecoder(decoders)
        self.timestep_shift_alpha = float(timestep_shift_alpha)
        self.flow_steps_per_reconstruction = int(flow_steps_per_reconstruction)
        self.flow_mini_batch = int(flow_mini_batch)
        self.train_eps = float(train_eps)
        self.sample_eps = float(sample_eps)
        self.lognorm_mu = float(lognorm_mu)
        self.lognorm_sigma = float(lognorm_sigma)
        self.reconstruction_noising_start = float(reconstruction_noising_start)
        self.reconstruction_noising_probability = float(
            reconstruction_noising_probability
        )
        self.condition_dropout_probability = float(condition_dropout_probability)
        self.cfg_scale = float(cfg_scale)
        self.cfg_interval = tuple(float(value) for value in cfg_interval)
        self.dopri5_output_points = int(dopri5_output_points)
        self.dopri5_atol = float(dopri5_atol)
        self.dopri5_rtol = float(dopri5_rtol)

        domains = tuple(getattr(self.generative_encoder, "domains", ()))
        if not domains or set(domains) != set(self.action_decoder.domains):
            raise ValueError("UNITE encoder and decoder domains must match")
        latent_dim = int(getattr(self.generative_encoder, "latent_dim", -1))
        if latent_dim != self.action_decoder.latent_dim:
            raise ValueError("UNITE encoder and decoder latent dimensions must match")
        if dict(getattr(self.generative_encoder, "action_dims", {})) != (
            self.action_decoder.action_dims
        ):
            raise ValueError("UNITE encoder and decoder action dimensions must match")
        if not torch.isfinite(torch.tensor(self.timestep_shift_alpha)):
            raise ValueError("timestep_shift_alpha must be finite")
        if self.timestep_shift_alpha <= 0.0:
            raise ValueError("timestep_shift_alpha must be positive")
        if self.flow_steps_per_reconstruction <= 0 or self.flow_mini_batch <= 0:
            raise ValueError("UNITE flow sample counts must be positive")
        if not 0.0 <= self.train_eps < 0.5 or not 0.0 <= self.sample_eps < 0.5:
            raise ValueError("UNITE epsilon values must be in [0, 0.5)")
        if self.lognorm_sigma <= 0.0:
            raise ValueError("lognorm_sigma must be positive")
        if not 0.0 <= self.reconstruction_noising_start <= 1.0:
            raise ValueError("reconstruction_noising_start must be in [0, 1]")
        if not 0.0 <= self.reconstruction_noising_probability <= 1.0:
            raise ValueError("reconstruction_noising_probability must be in [0, 1]")
        if not 0.0 <= self.condition_dropout_probability <= 1.0:
            raise ValueError("condition_dropout_probability must be in [0, 1]")
        if not torch.isfinite(torch.tensor(self.cfg_scale)) or self.cfg_scale < 0.0:
            raise ValueError("cfg_scale must be finite and non-negative")
        if len(self.cfg_interval) != 2 or not (
            0.0 <= self.cfg_interval[0] < self.cfg_interval[1] <= 1.0
        ):
            raise ValueError("cfg_interval must satisfy 0 <= start < end <= 1")
        if self.dopri5_output_points < 2:
            raise ValueError("dopri5_output_points must be at least 2")
        if self.dopri5_atol <= 0.0 or self.dopri5_rtol <= 0.0:
            raise ValueError("dopri5 tolerances must be positive")

    def _resolve_domain(self, embodiment: str | None = None) -> str:
        resolver = getattr(self.generative_encoder, "_resolve_domain", None)
        if resolver is None:
            raise TypeError("UNITE generative encoder lacks domain resolution")
        return resolver(embodiment)

    def _decode(self, latent: torch.Tensor, embodiment: str) -> torch.Tensor:
        return self.action_decoder.decoder_for(embodiment)(latent)

    def _validate_noise(self, noise: torch.Tensor) -> None:
        expected = (
            int(getattr(self.generative_encoder, "num_latent_tokens", -1)),
            int(getattr(self.generative_encoder, "latent_dim", -1)),
        )
        if noise.ndim != 3 or tuple(noise.shape[1:]) != expected:
            raise ValueError(
                f"UNITE sampler/noise must have shape (B, {expected[0]}, "
                f"{expected[1]}), got {tuple(noise.shape)}"
            )

    def shift_time(self, time: torch.Tensor) -> torch.Tensor:
        alpha = torch.as_tensor(
            self.timestep_shift_alpha,
            device=time.device,
            dtype=time.dtype,
        )
        return alpha * time / (1.0 + (alpha - 1.0) * time)

    def _null_condition_like(
        self, condition: torch.Tensor, embodiment: str
    ) -> torch.Tensor:
        return self.generative_encoder.null_condition_like(condition, embodiment)

    def _condition_with_dropout(
        self, condition: torch.Tensor, embodiment: str
    ) -> torch.Tensor:
        if not self.training or self.condition_dropout_probability == 0.0:
            return condition
        mask_shape = (int(condition.shape[0]),) + (1,) * (condition.ndim - 1)
        mask = (
            torch.rand(int(condition.shape[0]), device=condition.device)
            < self.condition_dropout_probability
        ).reshape(mask_shape)
        return torch.where(
            mask, self._null_condition_like(condition, embodiment), condition
        )

    def _guided_clean_prediction(
        self,
        latent: torch.Tensor,
        time: torch.Tensor,
        condition: torch.Tensor,
        cfg_scale: float,
        embodiment: str,
    ) -> torch.Tensor:
        if float(cfg_scale) <= 1.0:
            return self.generative_encoder.denoise(latent, time, condition, embodiment)
        doubled_latent = torch.cat((latent, latent), dim=0)
        doubled_time = torch.cat((time, time), dim=0)
        doubled_condition = torch.cat(
            (condition, self._null_condition_like(condition, embodiment)), dim=0
        )
        conditioned, unconditional = self.generative_encoder.denoise(
            doubled_latent, doubled_time, doubled_condition, embodiment
        ).chunk(2, dim=0)
        guided = unconditional + float(cfg_scale) * (conditioned - unconditional)
        start, end = self.cfg_interval
        active = ((time < end) & ((start == 0.0) | (time > start))).reshape(
            int(time.shape[0]), *([1] * (latent.ndim - 1))
        )
        return torch.where(active, guided, conditioned)

    def _sample_flow_time(self, batch_size: int, device: torch.device) -> torch.Tensor:
        normal = torch.randn(batch_size, device=device, dtype=torch.float32)
        normal = self.lognorm_mu + self.lognorm_sigma * normal
        return self.shift_time(torch.sigmoid(normal))

    def _noisy_reconstruction_latent(self, clean_latent: torch.Tensor) -> torch.Tensor:
        if not self.training or self.reconstruction_noising_probability == 0.0:
            return clean_latent
        batch_size = int(clean_latent.shape[0])
        time = self.reconstruction_noising_start + (
            1.0 - self.reconstruction_noising_start
        ) * torch.rand(batch_size, device=clean_latent.device, dtype=torch.float32)
        time = time.reshape(batch_size, 1, 1).to(clean_latent)
        noised = time * clean_latent + (1.0 - time) * torch.randn_like(clean_latent)
        mask = (
            torch.rand(batch_size, device=clean_latent.device)
            < self.reconstruction_noising_probability
        ).reshape(batch_size, 1, 1)
        return torch.where(mask, noised, clean_latent)

    @staticmethod
    def _flow_chunks(total: int, size: int) -> tuple[int, ...]:
        full, remainder = divmod(int(total), int(size))
        return (*([int(size)] * full), *((int(remainder),) if remainder else ()))

    def _released_flow_loss(
        self,
        clean_latent: torch.Tensor,
        condition: torch.Tensor,
        embodiment: str,
    ) -> torch.Tensor:
        detached_clean = clean_latent.detach()
        batch_size = int(detached_clean.shape[0])
        loss = None
        for repeats in self._flow_chunks(
            self.flow_steps_per_reconstruction, self.flow_mini_batch
        ):
            repeated_clean = detached_clean.repeat(repeats, 1, 1)
            repeated_condition = condition.repeat(
                repeats, *([1] * (condition.ndim - 1))
            )
            repeated_condition = self._condition_with_dropout(
                repeated_condition, embodiment
            )
            time = self._sample_flow_time(batch_size * repeats, detached_clean.device)
            time_view = time.reshape(batch_size * repeats, 1, 1).to(detached_clean)
            noise = torch.randn_like(repeated_clean)
            corrupted = time_view * repeated_clean + (1.0 - time_view) * noise
            prediction = self.generative_encoder.denoise(
                corrupted, time, repeated_condition, embodiment
            )
            denominator = (1.0 - time_view).clamp_min(self.train_eps)
            chunk_loss = ((repeated_clean - prediction) / denominator).square().mean()
            weighted = chunk_loss * repeats
            loss = weighted if loss is None else loss + weighted
        if loss is None:
            raise RuntimeError("Released UNITE flow loop produced no samples")
        return loss

    def _integrate_dopri5_trajectory(
        self,
        noise: torch.Tensor,
        condition: torch.Tensor,
        cfg_scale: float,
        embodiment: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Integrate once and return every released Dopri5 output-grid state."""

        try:
            from torchdiffeq import odeint
        except ImportError as exc:
            raise RuntimeError("Released UNITE sampling requires torchdiffeq") from exc

        batch_size = int(noise.shape[0])
        raw_grid = torch.linspace(
            0.0,
            1.0,
            self.dopri5_output_points,
            device=noise.device,
            dtype=torch.float32,
        )
        grid = self.shift_time(raw_grid)

        def velocity(time_scalar: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
            clean_prediction = self._guided_clean_prediction(
                latent,
                time_scalar.expand(batch_size),
                condition,
                cfg_scale,
                embodiment,
            )
            denominator = (1.0 - time_scalar).clamp_min(self.sample_eps)
            return (clean_prediction.float() - latent.float()) / denominator.float()

        trajectory = odeint(
            velocity,
            noise.float(),
            grid,
            method="dopri5",
            atol=self.dopri5_atol,
            rtol=self.dopri5_rtol,
        )
        if tuple(trajectory.shape[1:]) != tuple(noise.shape):
            raise RuntimeError("Released UNITE Dopri5 trajectory shape is invalid")
        if not bool(torch.isfinite(trajectory).all()):
            raise RuntimeError("Released UNITE Dopri5 trajectory is non-finite")
        return trajectory, raw_grid, grid

    def _integrate_dopri5(
        self,
        noise: torch.Tensor,
        condition: torch.Tensor,
        cfg_scale: float,
        embodiment: str,
    ) -> torch.Tensor:
        """Integrate once over the released shifted grid and return its endpoint."""

        trajectory, _, _ = self._integrate_dopri5_trajectory(
            noise, condition, cfg_scale, embodiment
        )
        return trajectory[-1]

    def sample(
        self,
        noise: torch.Tensor,
        condition: torch.Tensor,
        embodiment: str,
        *,
        cfg_scale: float | None = None,
    ) -> torch.Tensor:
        self._validate_noise(noise)
        scale = self.cfg_scale if cfg_scale is None else float(cfg_scale)
        return self._integrate_dopri5(noise, condition, scale, embodiment)

    def _capture_backbone_activations(self, operation):
        blocks = getattr(self.generative_encoder.denoising_module, "blocks", None)
        if not isinstance(blocks, nn.ModuleList) or not blocks:
            raise RuntimeError(
                "Released UNITE diagnostics require a denoiser ModuleList named "
                "'blocks'"
            )
        captured: dict[str, torch.Tensor] = {}
        handles = []

        def hook_for(name: str):
            def capture(_module, _inputs, output):
                if not torch.is_tensor(output) or output.ndim != 3:
                    raise RuntimeError(
                        f"Released UNITE diagnostic layer {name} returned an "
                        "invalid activation"
                    )
                if name in captured:
                    raise RuntimeError(
                        f"Released UNITE diagnostic layer {name} ran more than once"
                    )
                captured[name] = output.detach()

            return capture

        for index, block in enumerate(blocks):
            name = f"block_{index:02d}"
            handles.append(block.register_forward_hook(hook_for(name)))
        try:
            output = operation()
        finally:
            for handle in handles:
                handle.remove()
        expected = {f"block_{index:02d}" for index in range(len(blocks))}
        if set(captured) != expected:
            raise RuntimeError(
                "Released UNITE diagnostic hooks missed backbone blocks: "
                f"expected={sorted(expected)} observed={sorted(captured)}"
            )
        return output, captured

    @torch.inference_mode()
    def validation_diagnostics(
        self,
        *,
        noise: torch.Tensor,
        condition: torch.Tensor,
        target: torch.Tensor,
        embodiment: str,
        raw_noise_levels: tuple[float, ...] | list[float],
    ) -> dict:
        """Capture one exact Dopri5 trajectory and paired pathway activations."""

        if self.training:
            raise RuntimeError("Released UNITE diagnostics require evaluation mode")
        embodiment = self._resolve_domain(embodiment)
        self._validate_noise(noise)
        levels = torch.as_tensor(
            raw_noise_levels, device=noise.device, dtype=torch.float32
        )
        if (
            levels.ndim != 1
            or levels.numel() < 2
            or not bool(torch.isfinite(levels).all())
            or bool((levels < 0.0).any())
            or bool((levels > 1.0).any())
            or not bool(torch.all(levels[1:] > levels[:-1]))
        ):
            raise ValueError(
                "raw_noise_levels must be finite, strictly increasing values in [0, 1]"
            )

        clean_latent, tokenization_activations = self._capture_backbone_activations(
            lambda: self.generative_encoder.tokenize(target, embodiment, noise)
        )
        clean_target = clean_latent.detach()
        trajectory, raw_grid, shifted_grid = self._integrate_dopri5_trajectory(
            noise, condition, self.cfg_scale, embodiment
        )
        decoded_trajectory = torch.stack(
            [self._decode(state, embodiment).detach() for state in trajectory], dim=0
        )

        shifted_levels = self.shift_time(levels)
        final_predictions = []
        denoising_by_layer = {name: [] for name in tokenization_activations}
        batch_size = int(clean_target.shape[0])
        for shifted_level in shifted_levels:
            coefficient = shifted_level.reshape(1, 1, 1).to(clean_target)
            corrupted = coefficient * clean_target + (1.0 - coefficient) * noise.to(
                clean_target
            )
            time = shifted_level.expand(batch_size)
            prediction, activations = self._capture_backbone_activations(
                lambda corrupted=corrupted, time=time: self.generative_encoder.denoise(
                    corrupted, time, condition, embodiment
                )
            )
            final_predictions.append(prediction.detach())
            for name, activation in activations.items():
                denoising_by_layer[name].append(activation)

        return {
            "embodiment": embodiment,
            "clean_latent": clean_target,
            "initial_noise": noise.detach(),
            "sampler_raw_grid": raw_grid.detach(),
            "sampler_shifted_grid": shifted_grid.detach(),
            "sampler_latents": trajectory.detach(),
            "decoded_actions_normalized": decoded_trajectory,
            "raw_noise_levels": levels.detach(),
            "shifted_noise_levels": shifted_levels.detach(),
            "noise_level_final_predictions": torch.stack(final_predictions, dim=0),
            "tokenization_activations": tokenization_activations,
            "denoising_activations": {
                name: torch.stack(values, dim=0)
                for name, values in denoising_by_layer.items()
            },
        }

    def _forward_training(self, batch: dict, embodiment: str) -> dict:
        target = batch["target"]
        clean_latent = self.generative_encoder.tokenize(
            target,
            embodiment,
            batch["sampler/noise"],
        )
        reconstructed_action = self._decode(
            self._noisy_reconstruction_latent(clean_latent), embodiment
        )
        flow_loss = self._released_flow_loss(
            clean_latent, batch["condition"], embodiment
        )
        batch.update(
            {
                "unite/clean_latent": clean_latent,
                "unite/reconstructed_action": reconstructed_action,
                "unite/flow_loss": flow_loss,
            }
        )
        return batch

    def _forward_rollout(self, batch: dict, embodiment: str) -> dict:
        endpoint = self.sample(batch["sampler/noise"], batch["condition"], embodiment)
        batch["sampler/endpoint"] = endpoint
        batch["pred_action"] = self._decode(endpoint, embodiment)
        return batch

    def _execute_mode(self, batch: dict, mode: str) -> dict:
        embodiment = self._resolve_domain(batch["embodiment"])
        noise = batch["sampler/noise"]
        self._validate_noise(noise)
        if int(noise.shape[0]) != int(batch["condition"].shape[0]):
            raise ValueError("UNITE noise and condition batch sizes must match")
        if mode == "train":
            batch = self._forward_training(batch, embodiment)
        elif mode == "inference":
            batch = self._forward_rollout(batch, embodiment)
        else:
            raise ValueError(f"Unsupported UNITE execution mode {mode!r}")
        return batch

    def execute(self, batch: dict, *, mode: str) -> dict:
        return self._execute_mode(batch, mode)

    def forward(self, batch: dict) -> dict:
        """Retain direct-call behavior while graph execution passes mode explicitly."""
        return self._execute_mode(batch, "train" if self.training else "inference")


class ReleasedRecipeUniteObjective(Stage):
    """Action-space translation of the released reconstruction/flow objective."""

    train_only = True
    reads = [
        "target",
        "unite/reconstructed_action",
        "unite/flow_loss",
    ]
    writes = ["loss/*", "log/*"]

    def __init__(self, reconstruction_weight: float = 1.0, flow_weight: float = 1.0):
        super().__init__()
        self.reconstruction_weight = float(reconstruction_weight)
        self.flow_weight = float(flow_weight)
        if self.reconstruction_weight <= 0.0 or self.flow_weight <= 0.0:
            raise ValueError("Released UNITE loss weights must be positive")

    def forward(self, batch: dict) -> dict:
        reconstruction = _mse(batch["unite/reconstructed_action"], batch["target"])
        reconstruction_l1 = (
            (batch["unite/reconstructed_action"] - batch["target"]).abs().mean()
        )
        flow = batch["unite/flow_loss"]
        if flow.ndim != 0 or not bool(torch.isfinite(flow)):
            raise RuntimeError("Released UNITE flow loss must be a finite scalar")
        batch["loss/unite_reconstruction"] = self.reconstruction_weight * reconstruction
        batch["loss/unite_latent"] = self.flow_weight * flow
        batch["log/unite_reconstruction"] = reconstruction.detach()
        batch["log/unite_reconstruction_l1"] = reconstruction_l1.detach()
        batch["log/unite_latent"] = flow.detach()
        return batch
