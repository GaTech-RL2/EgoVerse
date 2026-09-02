"""Clean-action/register UNITE Generative Encoders.

This module generalizes the original H-for-H action implementation to a true
UNITE bottleneck. Clean action tokens are fully observed tokenizer content;
Gaussian register queries become the compact latent. Observations are supplied
only to the denoising path. The shared arm reuses one AdaLN backbone, while the
separate control owns two disjoint copies.
"""

from __future__ import annotations

import copy
from typing import Dict, Iterable, Mapping

import torch
import torch.nn as nn

from egomimic.pipeline.stages_unite import UniteGenerativeEncoder


class ConfigurableUniteGenerativeEncoder(UniteGenerativeEncoder):
    """One shared backbone for action tokenization and latent denoising."""

    def __init__(
        self,
        denoising_module: nn.Module,
        action_dims: Dict[str, int],
        condition_input_dim: int,
        latent_dim: int,
        num_latent_tokens: int,
        condition_dim: int,
        denoiser_hidden_dim: int,
        gradient_checkpointing: bool = True,
        tokenization_time_max: float = 0.01,
    ):
        super().__init__(
            denoising_module=denoising_module,
            action_dims=action_dims,
            condition_input_dim=condition_input_dim,
            latent_dim=latent_dim,
            condition_dim=condition_dim,
            denoiser_hidden_dim=denoiser_hidden_dim,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.num_latent_tokens = int(num_latent_tokens)
        self.tokenization_time_max = float(tokenization_time_max)
        if self.num_latent_tokens <= 0:
            raise ValueError("num_latent_tokens must be positive")
        if not 0.0 <= self.tokenization_time_max <= 1.0:
            raise ValueError("tokenization_time_max must be in [0, 1]")
        self._validate_backbone_contract(self.denoising_module)

        # The legacy class embeds actions directly into latent width. In the
        # register architecture actions are context/content tokens instead.
        del self.action_projections
        del self.tokenization_conditions
        self.action_context_projections = nn.ModuleDict(
            {
                domain: nn.Linear(action_dim, self.condition_dim)
                for domain, action_dim in self.action_dims.items()
            }
        )

    def _validate_backbone_contract(self, module: nn.Module) -> None:
        attributes = {
            "input_dim": self.latent_dim,
            "output_dim": self.latent_dim,
            "horizon": self.num_latent_tokens,
            "condition_dim": self.condition_dim,
            "hidden_dim": self.denoiser_hidden_dim,
        }
        for name, expected in attributes.items():
            actual = int(getattr(module, name, -1))
            if actual != int(expected):
                raise ValueError(
                    f"UNITE backbone {name}={actual}, expected {int(expected)}"
                )
        actual_checkpointing = bool(getattr(module, "gradient_checkpointing", False))
        if actual_checkpointing != self.gradient_checkpointing:
            raise ValueError(
                "UNITE backbone gradient_checkpointing="
                f"{actual_checkpointing}, expected {self.gradient_checkpointing}"
            )

    def _resolve_domain(self, embodiment: str | None) -> str:
        if embodiment is not None:
            return self._validate_domain(embodiment)
        if len(self.domains) != 1:
            raise ValueError(
                "embodiment is required when multiple UNITE domains are configured"
            )
        return self.domains[0]

    def _tokenization_backbone(self) -> nn.Module:
        return self.denoising_module

    def _tokenization_norm(self) -> nn.Module:
        return self.output_norm

    def _tokenization_domain_embedding(self, embodiment: str) -> torch.Tensor:
        return self.domain_embeddings[embodiment]

    def _denoising_norm(self) -> nn.Module:
        return self.output_norm

    def _denoising_domain_embedding(self, embodiment: str) -> torch.Tensor:
        return self.domain_embeddings[embodiment]

    def _register_noise(self, actions: torch.Tensor) -> torch.Tensor:
        return torch.randn(
            int(actions.shape[0]),
            self.num_latent_tokens,
            self.latent_dim,
            device=actions.device,
            dtype=actions.dtype,
        )

    def tokenize(self, actions: torch.Tensor, embodiment: str) -> torch.Tensor:
        embodiment = self._resolve_domain(embodiment)
        action_dim = self.action_dims[embodiment]
        if actions.ndim != 3 or int(actions.shape[-1]) != action_dim:
            raise ValueError(
                f"UNITE tokenizer for {embodiment!r} expected "
                f"(B, H, {action_dim}), got {tuple(actions.shape)}"
            )
        if int(actions.shape[1]) <= 0:
            raise ValueError("clean action content must contain at least one token")
        content = self.action_context_projections[embodiment](actions)
        domain = self._tokenization_domain_embedding(embodiment).to(content)
        content = content + domain.reshape(1, 1, -1)
        registers = self._register_noise(actions)
        batch_size = int(actions.shape[0])
        time = (
            torch.rand(batch_size, device=actions.device, dtype=torch.float32)
            * self.tokenization_time_max
        )
        encoded = self._tokenization_backbone()(
            registers,
            time,
            content_tokens=content,
        )
        expected = (batch_size, self.num_latent_tokens, self.latent_dim)
        if tuple(encoded.shape) != expected:
            raise RuntimeError(
                f"UNITE tokenizer produced {tuple(encoded.shape)}, expected {expected}"
            )
        return self._tokenization_norm()(encoded)

    def denoise(
        self,
        latent: torch.Tensor,
        time: torch.Tensor,
        observation_condition: torch.Tensor,
        embodiment: str | None = None,
    ) -> torch.Tensor:
        embodiment = self._resolve_domain(embodiment)
        expected_latent = (self.num_latent_tokens, self.latent_dim)
        if latent.ndim != 3 or tuple(latent.shape[1:]) != expected_latent:
            raise ValueError(
                "UNITE denoiser expected latent shape "
                f"(B, {self.num_latent_tokens}, {self.latent_dim}), got "
                f"{tuple(latent.shape)}"
            )
        if (
            observation_condition.ndim not in {2, 3}
            or int(observation_condition.shape[-1]) != self.condition_input_dim
        ):
            raise ValueError(
                "UNITE observation condition must have shape "
                f"(B, D) or (B, C, {self.condition_input_dim}), got "
                f"{tuple(observation_condition.shape)}"
            )
        batch_size = int(latent.shape[0])
        if int(observation_condition.shape[0]) != batch_size:
            raise ValueError("observation-condition batch does not match latent")
        if time.ndim == 0:
            time = time.expand(batch_size)
        if time.ndim != 1 or int(time.shape[0]) != batch_size:
            raise ValueError(
                f"UNITE time must have shape ({batch_size},), got {tuple(time.shape)}"
            )
        condition = self.condition_projection(observation_condition)
        domain = self._denoising_domain_embedding(embodiment).to(condition)
        condition = condition + domain.reshape(*((1,) * (condition.ndim - 1)), -1)
        predicted = self.denoising_module(latent, time, condition)
        expected = (batch_size, self.num_latent_tokens, self.latent_dim)
        if tuple(predicted.shape) != expected:
            raise RuntimeError(
                f"UNITE denoiser produced {tuple(predicted.shape)}, expected {expected}"
            )
        return self._denoising_norm()(predicted)

    def shared_reconstruction_denoising_named_parameters(
        self,
        embodiments: Iterable[str] | None = None,
    ) -> tuple[tuple[str, nn.Parameter], ...]:
        selected_domains = (
            self.domains
            if embodiments is None
            else tuple(self._validate_domain(domain) for domain in embodiments)
        )
        core_method = getattr(
            self.denoising_module, "shared_core_named_parameters", None
        )
        core = (
            core_method()
            if core_method is not None
            else tuple(self.denoising_module.named_parameters())
        )
        named: list[tuple[str, nn.Parameter]] = [
            (f"denoising_module.{name}", parameter) for name, parameter in core
        ]
        named.extend(
            (f"output_norm.{name}", parameter)
            for name, parameter in self.output_norm.named_parameters()
        )
        named.extend(
            (f"domain_embeddings.{domain}", self.domain_embeddings[domain])
            for domain in selected_domains
        )
        if not named or len({id(parameter) for _, parameter in named}) != len(named):
            raise RuntimeError(
                "UNITE shared parameter selection is empty or duplicated"
            )
        return tuple(named)


class SeparateUniteGenerativeEncoder(ConfigurableUniteGenerativeEncoder):
    """Capacity control with disjoint tokenizer and denoiser backbones."""

    def __init__(
        self,
        tokenization_module: nn.Module,
        denoising_module: nn.Module,
        action_dims: Dict[str, int],
        condition_input_dim: int,
        latent_dim: int,
        num_latent_tokens: int,
        condition_dim: int,
        denoiser_hidden_dim: int,
        gradient_checkpointing: bool = True,
        tokenization_time_max: float = 0.01,
    ):
        if tokenization_module is denoising_module:
            raise ValueError("separate UNITE requires distinct backbone objects")
        super().__init__(
            denoising_module=tokenization_module,
            action_dims=action_dims,
            condition_input_dim=condition_input_dim,
            latent_dim=latent_dim,
            num_latent_tokens=num_latent_tokens,
            condition_dim=condition_dim,
            denoiser_hidden_dim=denoiser_hidden_dim,
            gradient_checkpointing=gradient_checkpointing,
            tokenization_time_max=tokenization_time_max,
        )
        tokenizer = self.denoising_module
        del self.denoising_module
        self.tokenization_module = tokenizer
        self.denoising_module = denoising_module
        self._validate_backbone_contract(self.denoising_module)
        self.denoising_output_norm = nn.LayerNorm(self.latent_dim)
        self.denoising_domain_embeddings = nn.ParameterDict(
            {
                domain: nn.Parameter(torch.empty(self.condition_dim).normal_(std=0.02))
                for domain in self.domains
            }
        )

    def _tokenization_backbone(self) -> nn.Module:
        return self.tokenization_module

    def _denoising_norm(self) -> nn.Module:
        return self.denoising_output_norm

    def _denoising_domain_embedding(self, embodiment: str) -> torch.Tensor:
        return self.denoising_domain_embeddings[embodiment]

    def shared_reconstruction_denoising_named_parameters(
        self, embodiments: Iterable[str] | None = None
    ) -> tuple[tuple[str, nn.Parameter], ...]:
        raise RuntimeError("Separate UNITE has no shared GE parameters")

    def separate_reconstruction_denoising_named_parameters(
        self,
        embodiments: Iterable[str] | None = None,
    ) -> tuple[
        tuple[tuple[str, nn.Parameter], ...],
        tuple[tuple[str, nn.Parameter], ...],
    ]:
        selected_domains = (
            self.domains
            if embodiments is None
            else tuple(self._validate_domain(domain) for domain in embodiments)
        )
        tokenizer: list[tuple[str, nn.Parameter]] = list(
            (f"tokenization_module.{name}", parameter)
            for name, parameter in self.tokenization_module.named_parameters()
        )
        tokenizer.extend(
            (f"output_norm.{name}", parameter)
            for name, parameter in self.output_norm.named_parameters()
        )
        tokenizer.extend(
            (f"action_context_projections.{name}", parameter)
            for name, parameter in self.action_context_projections.named_parameters()
        )
        tokenizer.extend(
            (f"domain_embeddings.{domain}", self.domain_embeddings[domain])
            for domain in selected_domains
        )

        denoiser: list[tuple[str, nn.Parameter]] = list(
            (f"denoising_module.{name}", parameter)
            for name, parameter in self.denoising_module.named_parameters()
        )
        denoiser.extend(
            (f"denoising_output_norm.{name}", parameter)
            for name, parameter in self.denoising_output_norm.named_parameters()
        )
        denoiser.extend(
            (f"condition_projection.{name}", parameter)
            for name, parameter in self.condition_projection.named_parameters()
        )
        denoiser.extend(
            (
                f"denoising_domain_embeddings.{domain}",
                self.denoising_domain_embeddings[domain],
            )
            for domain in selected_domains
        )
        tokenizer_ids = {id(parameter) for _, parameter in tokenizer}
        denoiser_ids = {id(parameter) for _, parameter in denoiser}
        if not tokenizer or not denoiser or tokenizer_ids & denoiser_ids:
            raise RuntimeError("Separate UNITE parameter sets are empty or overlap")
        return tuple(tokenizer), tuple(denoiser)


def _instantiate_backbone(backbone_config) -> nn.Module:
    if isinstance(backbone_config, nn.Module):
        return backbone_config
    try:
        from hydra.utils import instantiate
    except ImportError as exc:
        raise RuntimeError("Hydra is required to instantiate UNITE backbones") from exc
    module = instantiate(backbone_config)
    if not isinstance(module, nn.Module):
        raise TypeError("backbone_config did not instantiate an nn.Module")
    return module


def _configure_gradient_checkpointing(module: nn.Module, enabled: bool) -> None:
    if not hasattr(module, "gradient_checkpointing"):
        raise ValueError("UNITE backbone must expose a gradient_checkpointing setting")
    module.gradient_checkpointing = bool(enabled)


def build_configurable_unite_generative_encoder(
    backbone_config: Mapping,
    share_encoder_denoiser: bool,
    action_dims: Dict[str, int],
    condition_input_dim: int,
    latent_dim: int,
    num_latent_tokens: int,
    condition_dim: int,
    denoiser_hidden_dim: int,
    gradient_checkpointing: bool = True,
    tokenization_time_max: float = 0.01,
) -> ConfigurableUniteGenerativeEncoder:
    """Hydra factory used by every row of the 2x3 register sweep."""

    first = _instantiate_backbone(backbone_config)
    _configure_gradient_checkpointing(first, gradient_checkpointing)
    common = dict(
        action_dims=action_dims,
        condition_input_dim=condition_input_dim,
        latent_dim=latent_dim,
        num_latent_tokens=num_latent_tokens,
        condition_dim=condition_dim,
        denoiser_hidden_dim=denoiser_hidden_dim,
        gradient_checkpointing=gradient_checkpointing,
        tokenization_time_max=tokenization_time_max,
    )
    if bool(share_encoder_denoiser):
        return ConfigurableUniteGenerativeEncoder(
            denoising_module=first,
            **common,
        )
    second = (
        copy.deepcopy(backbone_config)
        if isinstance(backbone_config, nn.Module)
        else _instantiate_backbone(backbone_config)
    )
    if isinstance(second, Mapping):
        second = _instantiate_backbone(second)
    _configure_gradient_checkpointing(second, gradient_checkpointing)
    return SeparateUniteGenerativeEncoder(
        tokenization_module=first,
        denoising_module=second,
        **common,
    )
