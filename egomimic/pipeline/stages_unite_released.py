"""Released-recipe UNITE training mechanics for robot-action latents.

This keeps the official joint-per-optimizer-step structure: one noisy
reconstruction pass plus 14 independent logit-normal flow samples, accumulated
in bounded mini-batches.  Image L1+LPIPS is necessarily translated to normalized
native-action MSE; the tokenizer/denoiser mechanism remains unchanged.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from egomimic.pipeline.core import Stage
from egomimic.pipeline.stages_unite import UniteLatentPolicy


def _mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError(
            "Released UNITE tensor shape mismatch: "
            f"prediction={tuple(prediction.shape)} target={tuple(target.shape)}"
        )
    return (prediction - target).square().mean()


class ReleasedRecipeUniteLatentPolicy(UniteLatentPolicy):
    """UNITE x-start policy with the authors' released training mechanics."""

    writes = UniteLatentPolicy.writes + ["unite/flow_loss"]

    def __init__(
        self,
        *args,
        flow_steps_per_reconstruction: int = 14,
        flow_mini_batch: int = 4,
        train_eps: float = 0.05,
        sample_eps: float = 0.05,
        lognorm_mu: float = 0.0,
        lognorm_sigma: float = 1.0,
        reconstruction_noising_start: float = 0.7,
        reconstruction_noising_probability: float = 0.5,
        condition_dropout_probability: float = 0.1,
        sampling_method: Literal["euler", "dopri5"] = "euler",
        cfg_scale: float = 4.0,
        cfg_interval: tuple[float, float] = (0.0, 1.0),
        cfg_norm_order: Literal["norm_first"] = "norm_first",
        dopri5_num_steps: int = 50,
        dopri5_atol: float = 1.0e-6,
        dopri5_rtol: float = 1.0e-3,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
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
        self.sampling_method = str(sampling_method).lower()
        self.cfg_scale = float(cfg_scale)
        self.cfg_interval = tuple(float(value) for value in cfg_interval)
        self.cfg_norm_order = str(cfg_norm_order)
        self.dopri5_num_steps = int(dopri5_num_steps)
        self.dopri5_atol = float(dopri5_atol)
        self.dopri5_rtol = float(dopri5_rtol)
        self.null_observation_condition = nn.Parameter(
            torch.empty(1, 1, self.generative_encoder.condition_input_dim)
        )
        nn.init.normal_(self.null_observation_condition, mean=0.0, std=0.02)
        self._last_sampler_nfe = 0
        if self.flow_steps_per_reconstruction <= 0:
            raise ValueError("flow_steps_per_reconstruction must be positive")
        if self.flow_mini_batch <= 0:
            raise ValueError("flow_mini_batch must be positive")
        if not 0.0 <= self.train_eps < 0.5:
            raise ValueError("train_eps must be in [0, 0.5)")
        if not 0.0 <= self.sample_eps < 0.5:
            raise ValueError("sample_eps must be in [0, 0.5)")
        if self.lognorm_sigma <= 0.0:
            raise ValueError("lognorm_sigma must be positive")
        if not 0.0 <= self.reconstruction_noising_start <= 1.0:
            raise ValueError("reconstruction_noising_start must be in [0, 1]")
        if not 0.0 <= self.reconstruction_noising_probability <= 1.0:
            raise ValueError("reconstruction_noising_probability must be in [0, 1]")
        if not 0.0 <= self.condition_dropout_probability <= 1.0:
            raise ValueError("condition_dropout_probability must be in [0, 1]")
        if self.sampling_method not in {"euler", "dopri5"}:
            raise ValueError("sampling_method must be 'euler' or 'dopri5'")
        if not torch.isfinite(torch.tensor(self.cfg_scale)) or self.cfg_scale < 0.0:
            raise ValueError("cfg_scale must be finite and non-negative")
        if len(self.cfg_interval) != 2 or not (
            0.0 <= self.cfg_interval[0] < self.cfg_interval[1] <= 1.0
        ):
            raise ValueError("cfg_interval must satisfy 0 <= start < end <= 1")
        if self.cfg_norm_order != "norm_first":
            raise ValueError(
                "Only UNITE's released default cfg_norm_order='norm_first' is supported"
            )
        if self.dopri5_num_steps < 2:
            raise ValueError("dopri5_num_steps must be at least 2")
        if self.dopri5_atol <= 0.0 or self.dopri5_rtol <= 0.0:
            raise ValueError("dopri5 tolerances must be positive")

    def _null_condition_like(self, condition: torch.Tensor) -> torch.Tensor:
        """Expand the learned null condition without changing token geometry."""

        if condition.ndim == 2:
            expected = self.generative_encoder.condition_input_dim
            if int(condition.shape[-1]) != expected:
                raise ValueError("UNITE condition width does not match the encoder")
            return (
                self.null_observation_condition[:, 0].to(condition).expand_as(condition)
            )
        if condition.ndim == 3:
            expected = self.generative_encoder.condition_input_dim
            if int(condition.shape[-1]) != expected:
                raise ValueError("UNITE condition width does not match the encoder")
            return self.null_observation_condition.to(condition).expand_as(condition)
        raise ValueError("UNITE condition must have shape (B, D) or (B, C, D)")

    def _condition_with_dropout(
        self, condition: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply UNITE's per-example classifier-free training dropout."""

        if not self.training or self.condition_dropout_probability == 0.0:
            return condition, condition.new_zeros(())
        mask_shape = (int(condition.shape[0]),) + (1,) * (condition.ndim - 1)
        mask = (
            torch.rand(int(condition.shape[0]), device=condition.device)
            < self.condition_dropout_probability
        ).reshape(mask_shape)
        dropped = torch.where(mask, self._null_condition_like(condition), condition)
        return dropped, mask.float().mean()

    def _guided_clean_prediction(
        self,
        latent: torch.Tensor,
        time: torch.Tensor,
        condition: torch.Tensor,
        cfg_scale: float,
    ) -> torch.Tensor:
        """Evaluate conditional/unconditional branches and apply released CFG."""

        if float(cfg_scale) <= 1.0:
            return self.generative_encoder.denoise(latent, time, condition)
        doubled_latent = torch.cat((latent, latent), dim=0)
        doubled_time = torch.cat((time, time), dim=0)
        doubled_condition = torch.cat(
            (condition, self._null_condition_like(condition)), dim=0
        )
        conditioned, unconditional = self.generative_encoder.denoise(
            doubled_latent, doubled_time, doubled_condition
        ).chunk(2, dim=0)
        guided = unconditional + float(cfg_scale) * (conditioned - unconditional)
        start, end = self.cfg_interval
        active = ((time < end) & ((start == 0.0) | (time > start))).reshape(
            int(time.shape[0]), *([1] * (latent.ndim - 1))
        )
        return torch.where(active, guided, conditioned)

    def _sample_flow_time(self, batch_size: int, device: torch.device) -> torch.Tensor:
        # Official Transport samples a logit-normal, maps it to [eps, 1-eps],
        # then applies the rational timestep shift.  Torch RNG is used here so
        # DDP seed/provenance is controlled by the robot training stack.
        normal = torch.randn(batch_size, device=device, dtype=torch.float32)
        normal = self.lognorm_mu + self.lognorm_sigma * normal
        unit_time = torch.sigmoid(normal)
        raw_time = self.train_eps + (1.0 - 2.0 * self.train_eps) * unit_time
        return self.shift_time(raw_time)

    def separate_reconstruction_denoising_named_parameters(self, embodiments=None):
        """Expose disjoint parameter sets for the released separate ablation."""

        method = getattr(
            self.generative_encoder,
            "separate_reconstruction_denoising_named_parameters",
            None,
        )
        if method is None:
            raise RuntimeError("This released UNITE policy uses a shared GE module")
        return method(embodiments)

    def _noisy_reconstruction_latent(self, clean_latent: torch.Tensor) -> torch.Tensor:
        if not self.training or self.reconstruction_noising_probability == 0.0:
            return clean_latent
        batch_size = int(clean_latent.shape[0])
        time = self.reconstruction_noising_start + (
            1.0 - self.reconstruction_noising_start
        ) * torch.rand(batch_size, device=clean_latent.device, dtype=torch.float32)
        time = time.reshape(batch_size, 1, 1).to(clean_latent)
        noise = torch.randn_like(clean_latent)
        noised = time * clean_latent + (1.0 - time) * noise
        mask = (
            torch.rand(batch_size, device=clean_latent.device)
            < self.reconstruction_noising_probability
        ).reshape(batch_size, 1, 1)
        return torch.where(mask, noised, clean_latent)

    @staticmethod
    def _flow_chunks(total: int, size: int) -> tuple[int, ...]:
        full, remainder = divmod(int(total), int(size))
        chunks = [int(size)] * full
        if remainder:
            chunks.append(remainder)
        return tuple(chunks)

    def _released_flow_loss(
        self,
        clean_latent: torch.Tensor,
        condition: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        detached_clean = clean_latent.detach()
        batch_size = int(detached_clean.shape[0])
        loss = None
        representative_prediction = None
        representative_time = None
        dropout_fraction = detached_clean.new_zeros(())
        for repeats in self._flow_chunks(
            self.flow_steps_per_reconstruction, self.flow_mini_batch
        ):
            repeated_clean = detached_clean.repeat(repeats, 1, 1)
            repeated_condition = condition.repeat(
                repeats, *([1] * (condition.ndim - 1))
            )
            repeated_condition, chunk_dropout_fraction = self._condition_with_dropout(
                repeated_condition
            )
            dropout_fraction = dropout_fraction + repeats * chunk_dropout_fraction
            time = self._sample_flow_time(batch_size * repeats, detached_clean.device)
            time_view = time.reshape(batch_size * repeats, 1, 1).to(detached_clean)
            noise = torch.randn_like(repeated_clean)
            corrupted = time_view * repeated_clean + (1.0 - time_view) * noise
            prediction = self.generative_encoder.denoise(
                corrupted, time, repeated_condition
            )
            # Released _compute_flow_loss parameterizes x-start through
            # velocity, yielding this exact (1-t)^-2 weighted x-start error.
            denominator = (1.0 - time_view).clamp_min(self.train_eps)
            chunk_loss = ((repeated_clean - prediction) / denominator).square()
            chunk_loss = chunk_loss.mean()
            weighted = chunk_loss * repeats
            loss = weighted if loss is None else loss + weighted
            if representative_prediction is None:
                representative_prediction = prediction[:batch_size]
                representative_time = time[:batch_size]
        if loss is None or representative_prediction is None:
            raise RuntimeError("Released UNITE flow loop produced no samples")
        dropout_fraction = dropout_fraction / self.flow_steps_per_reconstruction
        return loss, representative_prediction, representative_time, dropout_fraction

    def _forward_training(self, batch: dict, embodiment: str) -> dict:
        target = batch["target"]
        clean_latent = self.generative_encoder.tokenize(target, embodiment)
        reconstructed_action = self._decode(
            self._noisy_reconstruction_latent(clean_latent), embodiment
        )
        flow_loss, predicted_clean, sampled_time, dropout_fraction = (
            self._released_flow_loss(clean_latent, batch["condition"])
        )
        endpoint = predicted_clean
        if not self.training:
            endpoint = self.sample(batch["sampler/noise"], batch["condition"])
        predicted_action = self._decode(endpoint, embodiment)

        batch["sampler/endpoint"] = endpoint
        batch["pred_action"] = predicted_action
        batch["unite/clean_latent"] = clean_latent
        batch["unite/predicted_clean_latent"] = predicted_clean
        batch["unite/reconstructed_action"] = reconstructed_action
        batch["unite/flow_loss"] = flow_loss
        batch["log/sampler_unroll_steps"] = float(
            1 if self.training else self._last_sampler_nfe
        )
        batch["log/unite_time_mean"] = sampled_time.detach().mean()
        batch["log/unite_condition_dropout_fraction"] = dropout_fraction.detach()
        batch["log/unite_cfg_scale"] = self.cfg_scale
        batch["log/unite_flow_samples_per_reconstruction"] = float(
            self.flow_steps_per_reconstruction
        )
        return batch

    def _forward_rollout(self, batch: dict, embodiment: str) -> dict:
        endpoint = self.sample(batch["sampler/noise"], batch["condition"])
        batch["sampler/endpoint"] = endpoint
        batch["pred_action"] = self._decode(endpoint, embodiment)
        batch["log/sampler_unroll_steps"] = float(self._last_sampler_nfe)
        batch["log/unite_cfg_scale"] = self.cfg_scale
        return batch

    def sample(
        self,
        noise: torch.Tensor,
        condition: torch.Tensor,
        *,
        sampling_method: str | None = None,
        cfg_scale: float | None = None,
    ) -> torch.Tensor:
        """Sample with released dopri5+CFG or the canonical fixed-J fallback."""

        self._validate_noise(noise)
        method = (
            self.sampling_method if sampling_method is None else sampling_method.lower()
        )
        scale = self.cfg_scale if cfg_scale is None else float(cfg_scale)
        if method == "dopri5":
            return self._sample_dopri5(noise, condition, scale)
        if method != "euler":
            raise ValueError("sampling_method must be 'euler' or 'dopri5'")
        return self._sample_euler(noise, condition, scale)

    def _sample_euler(
        self, noise: torch.Tensor, condition: torch.Tensor, cfg_scale: float
    ) -> torch.Tensor:
        """Preserve the fixed J sampler used by the canonical robot protocol."""

        batch_size = int(noise.shape[0])
        latent = noise
        raw_grid = torch.linspace(
            self.sample_eps,
            1.0 - self.sample_eps,
            self.num_inference_steps + 1,
            device=noise.device,
            dtype=torch.float32,
        )
        grid = self.shift_time(raw_grid)
        for index in range(self.num_inference_steps):
            current_t = grid[index]
            next_t = grid[index + 1]
            time = current_t.expand(batch_size)
            clean_prediction = self._guided_clean_prediction(
                latent, time, condition, cfg_scale
            )
            denominator = (1.0 - current_t).clamp_min(self.sample_eps)
            velocity = (clean_prediction - latent) / denominator.to(latent)
            latent = latent + (next_t - current_t).to(latent) * velocity
        self._last_sampler_nfe = self.num_inference_steps
        return latent

    def _sample_dopri5(
        self, noise: torch.Tensor, condition: torch.Tensor, cfg_scale: float
    ) -> torch.Tensor:
        """Match released UNITE: torchdiffeq dopri5 over 50 shifted times."""

        try:
            from torchdiffeq import odeint
        except ImportError as exc:
            raise RuntimeError(
                "Paper-compatible UNITE sampling requires torchdiffeq"
            ) from exc

        batch_size = int(noise.shape[0])
        raw_grid = torch.linspace(
            0.0,
            1.0,
            self.dopri5_num_steps,
            device=noise.device,
            dtype=torch.float32,
        )
        grid = self.shift_time(raw_grid)
        nfe = 0

        def velocity(time_scalar: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
            nonlocal nfe
            nfe += 1
            time = time_scalar.expand(batch_size)
            clean_prediction = self._guided_clean_prediction(
                latent, time, condition, cfg_scale
            )
            denominator = (1.0 - time_scalar).clamp_min(self.sample_eps)
            return (clean_prediction - latent) / denominator.to(latent)

        trajectory = odeint(
            velocity,
            noise,
            grid,
            method="dopri5",
            atol=self.dopri5_atol,
            rtol=self.dopri5_rtol,
        )
        self._last_sampler_nfe = nfe
        return trajectory[-1]


class ReleasedRecipeUniteObjective(Stage):
    """Robot-action translation of released reconstruction + flow objective."""

    train_only = True
    reads = [
        "pred_action",
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
        generated_action = _mse(batch["pred_action"], batch["target"])
        batch["loss/unite_reconstruction"] = self.reconstruction_weight * reconstruction
        batch["loss/unite_latent"] = self.flow_weight * flow
        batch["log/unite_reconstruction"] = reconstruction.detach()
        batch["log/unite_reconstruction_l1"] = reconstruction_l1.detach()
        batch["log/unite_latent"] = flow.detach()
        batch["log/native_action"] = generated_action.detach()
        return batch
