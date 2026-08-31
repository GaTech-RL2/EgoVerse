"""UNITE-style action tokenization and latent denoising stages.

The existing Latent Dense policy learns an action-space objective through an
unconstrained latent endpoint.  This module adds an independent policy head
whose latent endpoint is anchored by two passes through one shared generative
encoder:

* tokenization maps a native action chunk to a clean, reconstructable latent;
* denoising maps a corrupted latent back to that detached clean target.

The implementation follows the repository's Pipeline convention and leaves the
existing Latent Dense and Diffusion Policy stages untouched.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from egomimic.pipeline.core import Stage
from egomimic.pipeline.stages_sampler import PerEmbodimentActionDecoder


def _mean_squared_error(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError(
            "UNITE tensor shape mismatch: "
            f"prediction={tuple(prediction.shape)} target={tuple(target.shape)}"
        )
    return (prediction - target).square().mean()


class UniteGenerativeEncoder(nn.Module):
    """Share one Transformer between action tokenization and latent denoising.

    Embodiment-specific linear stems only reconcile the U-Socket and
    ChainGripper action widths.  Both modes then use the exact same denoising
    module and final LayerNorm, which is the weight-sharing mechanism central
    to UNITE.  The observation is a condition only in denoising mode; learned
    tokenization conditions provide the compatible cross-attention input for
    the clean-action pass.
    """

    def __init__(
        self,
        denoising_module: nn.Module,
        action_dims: Dict[str, int],
        condition_input_dim: int,
        latent_dim: int,
        condition_dim: int,
        denoiser_hidden_dim: int,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()
        self.denoising_module = denoising_module
        self.action_dims = {
            str(domain): int(action_dim)
            for domain, action_dim in dict(action_dims).items()
        }
        self.domains = tuple(self.action_dims)
        self.condition_input_dim = int(condition_input_dim)
        self.latent_dim = int(latent_dim)
        self.condition_dim = int(condition_dim)
        self.denoiser_hidden_dim = int(denoiser_hidden_dim)
        self.gradient_checkpointing = bool(gradient_checkpointing)

        if not self.domains or len(set(self.domains)) != len(self.domains):
            raise ValueError("action_dims must configure unique embodiment names")
        if any(action_dim <= 0 for action_dim in self.action_dims.values()):
            raise ValueError("Every UNITE action dimension must be positive")
        if (
            min(
                self.condition_input_dim,
                self.latent_dim,
                self.condition_dim,
                self.denoiser_hidden_dim,
            )
            <= 0
        ):
            raise ValueError("UNITE model dimensions must be positive")

        self.action_projections = nn.ModuleDict(
            {
                domain: nn.Linear(action_dim, self.latent_dim)
                for domain, action_dim in self.action_dims.items()
            }
        )
        self.condition_projection = nn.Linear(
            self.condition_input_dim, self.condition_dim
        )
        self.domain_embeddings = nn.ParameterDict(
            {
                domain: nn.Parameter(torch.empty(self.condition_dim).normal_(std=0.02))
                for domain in self.domains
            }
        )
        self.tokenization_conditions = nn.ParameterDict(
            {
                domain: nn.Parameter(torch.empty(self.condition_dim).normal_(std=0.02))
                for domain in self.domains
            }
        )
        self.output_norm = nn.LayerNorm(self.latent_dim)
        self._validate_denoiser_contract()

    def _validate_denoiser_contract(self) -> None:
        proj_u = getattr(self.denoising_module, "proj_u", None)
        proj_d = getattr(self.denoising_module, "proj_d", None)
        if proj_u is None or proj_d is None:
            raise ValueError(
                "UNITE generative encoder requires denoiser proj_u and proj_d"
            )
        if int(proj_u.in_features) != self.latent_dim:
            raise ValueError(
                f"Denoiser input is {proj_u.in_features}, expected "
                f"latent_dim={self.latent_dim}"
            )
        if int(proj_d.out_features) != self.latent_dim:
            raise ValueError(
                f"Denoiser output is {proj_d.out_features}, expected "
                f"latent_dim={self.latent_dim}"
            )
        time_mode = getattr(self.denoising_module, "time_conditioning", "concat")
        actual_hidden = (
            int(proj_u.out_features) * 2
            if time_mode == "concat"
            else int(proj_u.out_features)
        )
        if actual_hidden != self.denoiser_hidden_dim:
            raise ValueError(
                f"Denoiser produces hidden_dim={actual_hidden}, expected "
                f"{self.denoiser_hidden_dim}"
            )
        if int(proj_d.in_features) != self.denoiser_hidden_dim:
            raise ValueError(
                f"Denoiser proj_d input is {proj_d.in_features}, expected "
                f"{self.denoiser_hidden_dim}"
            )

    def _validate_domain(self, embodiment: str) -> str:
        embodiment = str(embodiment)
        if embodiment not in self.action_dims:
            raise KeyError(
                f"Unknown UNITE embodiment {embodiment!r}; "
                f"configured={list(self.domains)}"
            )
        return embodiment

    def _validate_latent_horizon(self, value: torch.Tensor) -> None:
        pos_emb = getattr(self.denoising_module, "pos_emb", None)
        if pos_emb is not None and (
            pos_emb.ndim < 2 or int(pos_emb.shape[-2]) != int(value.shape[1])
        ):
            raise ValueError(
                "UNITE denoiser positional horizon is "
                f"{tuple(pos_emb.shape)}, but input has {int(value.shape[1])} tokens"
            )

    def _run_shared_transformer(
        self,
        latent: torch.Tensor,
        time: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            output = checkpoint(
                self.denoising_module,
                latent,
                time,
                condition,
                use_reentrant=False,
            )
        else:
            output = self.denoising_module(latent, time, condition)
        return self.output_norm(output)

    def tokenize(self, actions: torch.Tensor, embodiment: str) -> torch.Tensor:
        embodiment = self._validate_domain(embodiment)
        if actions.ndim != 3 or int(actions.shape[-1]) != self.action_dims[embodiment]:
            raise ValueError(
                f"UNITE tokenizer for {embodiment!r} expected (B, T, "
                f"{self.action_dims[embodiment]}), got {tuple(actions.shape)}"
            )
        self._validate_latent_horizon(actions)
        projected_actions = self.action_projections[embodiment](actions)
        batch_size = int(actions.shape[0])
        time = torch.ones(batch_size, device=actions.device, dtype=torch.float32)
        token_condition = (
            self.tokenization_conditions[embodiment]
            + self.domain_embeddings[embodiment]
        ).to(projected_actions)
        condition = token_condition.reshape(1, 1, -1).expand(batch_size, 1, -1)
        return self._run_shared_transformer(projected_actions, time, condition)

    def denoise(
        self,
        latent: torch.Tensor,
        time: torch.Tensor,
        observation_condition: torch.Tensor,
        embodiment: str,
    ) -> torch.Tensor:
        embodiment = self._validate_domain(embodiment)
        if latent.ndim != 3 or int(latent.shape[-1]) != self.latent_dim:
            raise ValueError(
                f"UNITE denoiser expected (B, T, {self.latent_dim}), "
                f"got {tuple(latent.shape)}"
            )
        self._validate_latent_horizon(latent)
        if (
            observation_condition.ndim != 2
            or int(observation_condition.shape[-1]) != self.condition_input_dim
        ):
            raise ValueError(
                "UNITE observation condition must have shape "
                f"(B, {self.condition_input_dim}), got "
                f"{tuple(observation_condition.shape)}"
            )
        if time.ndim != 1 or int(time.shape[0]) != int(latent.shape[0]):
            raise ValueError(
                f"UNITE time must have shape ({int(latent.shape[0])},), "
                f"got {tuple(time.shape)}"
            )
        projected_condition = self.condition_projection(
            observation_condition
        ).unsqueeze(1)
        domain_embedding = self.domain_embeddings[embodiment].to(projected_condition)
        condition = projected_condition + domain_embedding.reshape(1, 1, -1)
        return self._run_shared_transformer(latent, time, condition)


class UniteLatentPolicy(Stage):
    """Train UNITE's two passes and sample its x-start latent flow at rollout."""

    reads = ["sampler/noise", "condition", "target", "embodiment"]
    writes = [
        "sampler/endpoint",
        "pred_action",
        "unite/clean_latent",
        "unite/predicted_clean_latent",
        "unite/reconstructed_action",
        "log/*",
    ]
    reads_by_mode = {
        "rollout": ["sampler/noise", "condition", "embodiment"],
    }
    writes_by_mode = {
        "rollout": ["sampler/endpoint", "pred_action", "log/*"],
    }

    def __init__(
        self,
        generative_encoder: UniteGenerativeEncoder,
        decoders: Dict[str, nn.Module],
        num_inference_steps: int = 8,
        reconstruction_noise_std: float = 0.1,
    ):
        super().__init__()
        if not isinstance(generative_encoder, UniteGenerativeEncoder):
            raise TypeError("generative_encoder must be UniteGenerativeEncoder")
        self.generative_encoder = generative_encoder
        self.action_decoder = PerEmbodimentActionDecoder(decoders)
        self.num_inference_steps = int(num_inference_steps)
        self.reconstruction_noise_std = float(reconstruction_noise_std)
        if self.num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive")
        if self.reconstruction_noise_std < 0.0:
            raise ValueError("reconstruction_noise_std must be non-negative")
        if set(self.generative_encoder.domains) != set(self.action_decoder.domains):
            raise ValueError("UNITE encoder and decoder embodiments must match exactly")
        if self.generative_encoder.latent_dim != self.action_decoder.latent_dim:
            raise ValueError("UNITE encoder and decoder latent dimensions must match")
        if self.generative_encoder.action_dims != self.action_decoder.action_dims:
            raise ValueError("UNITE encoder and decoder action dimensions must match")

    def _decode(self, latent: torch.Tensor, embodiment: str) -> torch.Tensor:
        return self.action_decoder.decoder_for(embodiment)(latent)

    def _validate_noise(self, noise: torch.Tensor) -> None:
        if (
            noise.ndim != 3
            or int(noise.shape[-1]) != self.generative_encoder.latent_dim
        ):
            raise ValueError(
                "UNITE expected sampler/noise shape "
                f"(B, T, {self.generative_encoder.latent_dim}), got "
                f"{tuple(noise.shape)}"
            )

    def sample(
        self,
        noise: torch.Tensor,
        condition: torch.Tensor,
        embodiment: str,
    ) -> torch.Tensor:
        """Integrate x-start predictions from pure noise at t=0 to data at t=1."""

        self._validate_noise(noise)
        batch_size = int(noise.shape[0])
        latent = noise
        grid = torch.linspace(
            0.0,
            1.0,
            self.num_inference_steps + 1,
            device=noise.device,
            dtype=torch.float32,
        )
        for index in range(self.num_inference_steps):
            current_t = grid[index]
            next_t = grid[index + 1]
            time = current_t.expand(batch_size)
            clean_prediction = self.generative_encoder.denoise(
                latent, time, condition, embodiment
            )
            if index == self.num_inference_steps - 1:
                latent = clean_prediction
                continue
            denominator = (1.0 - current_t).clamp_min(1.0e-6)
            noise_prediction = (
                latent - current_t.to(latent) * clean_prediction
            ) / denominator.to(latent)
            latent = (
                next_t.to(latent) * clean_prediction
                + (1.0 - next_t).to(latent) * noise_prediction
            )
        return latent

    def _forward_training(self, batch: dict, embodiment: str) -> dict:
        target = batch["target"]
        clean_latent = self.generative_encoder.tokenize(target, embodiment)
        detached_clean = clean_latent.detach()
        flow_noise = batch["sampler/noise"].to(detached_clean)
        time = torch.rand(
            int(target.shape[0]), device=target.device, dtype=torch.float32
        )
        latent_time = time.reshape(int(target.shape[0]), 1, 1).to(detached_clean)
        corrupted_latent = (
            latent_time * detached_clean + (1.0 - latent_time) * flow_noise
        )
        predicted_clean = self.generative_encoder.denoise(
            corrupted_latent, time, batch["condition"], embodiment
        )

        reconstruction_latent = clean_latent
        if self.reconstruction_noise_std > 0.0:
            reconstruction_latent = reconstruction_latent + (
                self.reconstruction_noise_std * torch.randn_like(clean_latent)
            )
        reconstructed_action = self._decode(reconstruction_latent, embodiment)
        endpoint = predicted_clean
        if not self.training:
            # Teacher-forced validation still computes the reconstruction and
            # random-t latent objectives, but its reported native-action MSE
            # must come from the complete inference sampler.
            endpoint = self.sample(
                batch["sampler/noise"], batch["condition"], embodiment
            )
        predicted_action = self._decode(endpoint, embodiment)

        batch["sampler/endpoint"] = endpoint
        batch["pred_action"] = predicted_action
        batch["unite/clean_latent"] = clean_latent
        batch["unite/predicted_clean_latent"] = predicted_clean
        batch["unite/reconstructed_action"] = reconstructed_action
        batch["log/sampler_unroll_steps"] = float(
            1 if self.training else self.num_inference_steps
        )
        batch["log/unite_time_mean"] = time.detach().mean()
        return batch

    def _forward_rollout(self, batch: dict, embodiment: str) -> dict:
        noise = batch["sampler/noise"]
        endpoint = self.sample(noise, batch["condition"], embodiment)
        batch["sampler/endpoint"] = endpoint
        batch["pred_action"] = self._decode(endpoint, embodiment)
        batch["log/sampler_unroll_steps"] = float(self.num_inference_steps)
        return batch

    def forward(self, batch: dict) -> dict:
        embodiment = self.generative_encoder._validate_domain(batch["embodiment"])
        noise = batch["sampler/noise"]
        self._validate_noise(noise)
        if int(noise.shape[0]) != int(batch["condition"].shape[0]):
            raise ValueError("UNITE noise and condition batch sizes must match")
        if "target" in batch:
            batch = self._forward_training(batch, embodiment)
        else:
            batch = self._forward_rollout(batch, embodiment)

        endpoint = batch["sampler/endpoint"]
        prediction = batch["pred_action"]
        batch["log/sampler_noise_rms"] = noise.detach().square().mean().sqrt()
        batch["log/sampler_endpoint_rms"] = endpoint.detach().square().mean().sqrt()
        batch["log/sampler_prediction_rms"] = prediction.detach().square().mean().sqrt()
        return batch


class UniteObjective(Stage):
    """UNITE reconstruction and detached clean-latent denoising objective."""

    train_only = True
    reads = [
        "pred_action",
        "target",
        "unite/clean_latent",
        "unite/predicted_clean_latent",
        "unite/reconstructed_action",
    ]
    writes = ["loss/*", "log/*"]

    def __init__(
        self,
        reconstruction_weight: float = 1.0,
        latent_weight: float = 1.0,
        generated_action_weight: float = 0.0,
    ):
        super().__init__()
        self.reconstruction_weight = float(reconstruction_weight)
        self.latent_weight = float(latent_weight)
        self.generated_action_weight = float(generated_action_weight)
        if self.reconstruction_weight <= 0.0 or self.latent_weight <= 0.0:
            raise ValueError("UNITE reconstruction and latent weights must be positive")
        if self.generated_action_weight < 0.0:
            raise ValueError("generated_action_weight must be non-negative")

    def forward(self, batch: dict) -> dict:
        target = batch["target"]
        reconstruction = _mean_squared_error(
            batch["unite/reconstructed_action"], target
        )
        latent = _mean_squared_error(
            batch["unite/predicted_clean_latent"],
            batch["unite/clean_latent"].detach(),
        )
        generated_action = _mean_squared_error(batch["pred_action"], target)

        batch["loss/unite_reconstruction"] = self.reconstruction_weight * reconstruction
        batch["loss/unite_latent"] = self.latent_weight * latent
        if self.generated_action_weight > 0.0:
            batch["loss/unite_generated_action"] = (
                self.generated_action_weight * generated_action
            )

        batch["log/unite_reconstruction"] = reconstruction.detach()
        batch["log/unite_latent"] = latent.detach()
        batch["log/native_action"] = generated_action.detach()
        batch["log/unite_clean_latent_std"] = (
            batch["unite/clean_latent"].detach().float().std()
        )
        return batch
