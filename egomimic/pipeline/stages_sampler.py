"""Input and noise stages shared by configured Pipeline graphs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Dict

import torch
import torch.nn as nn

from egomimic.pipeline.core import Stage, resolve_homogeneous_scalar


class DPStyleObsEncoder(nn.Module):
    """Concatenate low-dimensional state with encoded image observations."""

    def __init__(self, obs_specs: Dict[str, dict], img_encoders: Dict[str, nn.Module]):
        super().__init__()
        if not obs_specs or not img_encoders:
            raise ValueError("DPStyleObsEncoder needs low-dim and image inputs")
        self.obs_specs = {str(key): dict(spec) for key, spec in obs_specs.items()}
        self.img_encoders = nn.ModuleDict(img_encoders)

    def forward_packed(self, *, obs_packed: dict, T_total: int, **_context):
        features = []
        for key in sorted(self.obs_specs):
            spec = self.obs_specs[key]
            value = obs_packed[key]
            start, end = spec.get("input_slice", [0, value.shape[-1]])
            value = value[..., int(start) : int(end)]
            expected = int(spec.get("input_dim", value.shape[-1]))
            if value.shape[-1] != expected:
                raise ValueError(
                    f"DPStyleObsEncoder {key}: expected {expected} dims after "
                    f"slice, got {value.shape[-1]}"
                )
            features.append(value)
        for key in sorted(self.img_encoders):
            features.append(self.img_encoders[key](obs_packed[key]))
        if any(int(feature.shape[0]) != int(T_total) for feature in features):
            raise ValueError("DPStyleObsEncoder inputs do not share T_total")
        return torch.cat(features, dim=-1)


class FusedObsEncoder(Stage):
    """Encode explicitly selected batch inputs into one conditioning vector."""

    writes = ["condition"]

    def __init__(
        self,
        encoder: nn.Module,
        inputs: Mapping[str, str],
        n_obs_steps: int = 2,
        forward_context: Dict[str, str] | None = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.inputs = {
            str(packed_key): str(batch_key)
            for packed_key, batch_key in dict(inputs).items()
        }
        if not self.inputs or any(
            not packed_key or not batch_key
            for packed_key, batch_key in self.inputs.items()
        ):
            raise ValueError("inputs must map packed names to non-empty batch keys")
        self.n_obs_steps = int(n_obs_steps)
        self.forward_context = {
            str(batch_key): str(argument)
            for batch_key, argument in dict(forward_context or {}).items()
        }
        if any(not key or not value for key, value in self.forward_context.items()):
            raise ValueError("forward_context keys and arguments must be non-empty")
        arguments = tuple(self.forward_context.values())
        if len(set(arguments)) != len(arguments):
            raise ValueError("forward_context arguments must be unique")
        reserved_arguments = {
            "obs_packed",
            "cu_seqlens",
            "T_total",
            "device",
            "dtype",
        }
        conflicts = sorted(set(arguments) & reserved_arguments)
        if conflicts:
            raise ValueError(
                "forward_context cannot replace encoder inputs: " f"{conflicts}"
            )
        if self.n_obs_steps <= 0:
            raise ValueError("n_obs_steps must be positive")
        self.reads = list(dict.fromkeys([*self.inputs.values(), *self.forward_context]))

    def forward(self, batch: dict) -> dict:
        obs_packed = {
            packed_key: batch[batch_key]
            for packed_key, batch_key in self.inputs.items()
        }
        try:
            reference = next(
                value for value in obs_packed.values() if torch.is_tensor(value)
            )
        except StopIteration as exc:
            raise ValueError("FusedObsEncoder requires tensor observations") from exc

        batch_size, n_obs = int(reference.shape[0]), self.n_obs_steps
        if n_obs == 1:
            for key, value in list(obs_packed.items()):
                if not torch.is_tensor(value):
                    continue
                if int(value.shape[0]) != batch_size:
                    raise ValueError(
                        f"FusedObsEncoder: {self.inputs[key]} has batch "
                        f"{value.shape[0]}; "
                        f"expected {batch_size}"
                    )
        else:
            if reference.ndim < 2 or reference.shape[1] != n_obs:
                raise ValueError(
                    f"FusedObsEncoder got shape {tuple(reference.shape)}; expected "
                    f"(B, {n_obs}, ...)"
                )
            for key, value in list(obs_packed.items()):
                if not torch.is_tensor(value):
                    continue
                if value.shape[:2] != (batch_size, n_obs):
                    raise ValueError(
                        f"FusedObsEncoder: {self.inputs[key]} has shape "
                        f"{tuple(value.shape)}; "
                        f"expected leading dimensions {(batch_size, n_obs)}"
                    )
                obs_packed[key] = value.reshape(batch_size * n_obs, *value.shape[2:])

        total = batch_size * n_obs
        device = reference.device
        dtype = reference.dtype if reference.is_floating_point() else torch.float32
        cu_seqlens = torch.arange(0, total + 1, n_obs, device=device, dtype=torch.long)
        for module in self.modules():
            if getattr(module, "crop_scope", None) == "episode":
                module._episode_cu = cu_seqlens
        encoder_context = {
            argument: batch[batch_key]
            for batch_key, argument in self.forward_context.items()
        }
        encoded = self.encoder.forward_packed(
            obs_packed=obs_packed,
            cu_seqlens=cu_seqlens,
            T_total=total,
            device=device,
            dtype=dtype,
            **encoder_context,
        )
        expected_rows = batch_size * n_obs
        if (
            not torch.is_tensor(encoded)
            or encoded.ndim != 2
            or int(encoded.shape[0]) != expected_rows
        ):
            shape = tuple(encoded.shape) if torch.is_tensor(encoded) else None
            raise ValueError(
                "FusedObsEncoder output must have shape "
                f"({expected_rows}, feature_dim), got {shape}"
            )
        batch["condition"] = encoded.reshape(batch_size, n_obs * encoded.shape[-1])

        return batch


class GaussianLatentNoise(Stage):
    """Create Gaussian latent tokens without reading the action target."""

    reads = ["condition"]
    writes = ["sampler/noise"]

    def __init__(
        self,
        num_tokens: int,
        latent_dim: int = 128,
    ):
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.latent_dim = int(latent_dim)
        if self.num_tokens <= 0 or self.latent_dim <= 0:
            raise ValueError("num_tokens and latent_dim must be positive")

    def forward(self, batch: dict) -> dict:
        condition = batch["condition"]
        batch_size = int(condition.shape[0])
        if batch_size <= 0:
            raise ValueError("condition must describe at least one sample")
        device = condition.device
        dtype = (
            torch.get_autocast_dtype(device.type)
            if torch.is_autocast_enabled(device.type)
            else torch.get_default_dtype()
        )
        batch["sampler/noise"] = torch.randn(
            batch_size,
            self.num_tokens,
            self.latent_dim,
            dtype=dtype,
            device=device,
        )
        return batch


class KeyedFeatureProjection(Stage):
    """Select a feature projection using ordinary batch-dictionary keys."""

    def __init__(
        self,
        selector_key: str,
        input_key: str,
        output_key: str,
        projections: Dict[str, dict],
        output_dim: int = 64,
        selector_aliases: Mapping | None = None,
    ):
        super().__init__()
        self.selector_key = str(selector_key)
        self.input_key = str(input_key)
        self.output_key = str(output_key)
        if not self.selector_key or not self.input_key or not self.output_key:
            raise ValueError(
                "selector_key, input_key, and output_key must be non-empty"
            )
        self.reads = (self.input_key, self.selector_key)
        self.writes = (self.output_key,)
        self.output_dim = int(output_dim)
        if self.output_dim <= 0 or not projections:
            raise ValueError("positive output_dim and at least one projection required")

        self.projection_specs = {}
        branches = {}
        for selector, raw_spec in projections.items():
            spec = dict(raw_spec)
            input_dim = int(spec["input_dim"])
            hidden_dim = int(spec.get("hidden_dim", self.output_dim))
            if input_dim <= 0 or hidden_dim <= 0:
                raise ValueError("input_dim and hidden_dim must be positive")
            selector = str(selector)
            if not selector:
                raise ValueError("projection selector names must be non-empty")
            self.projection_specs[selector] = {
                "input_dim": input_dim,
                "semantic": str(spec.get("semantic", "")),
            }
            branches[selector] = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, self.output_dim),
            )
        self.projections = nn.ModuleDict(branches)

        self.selector_aliases = {
            str(resolve_homogeneous_scalar(source, label="selector alias")): str(target)
            for source, target in dict(selector_aliases or {}).items()
        }
        unknown_targets = set(self.selector_aliases.values()) - set(self.projections)
        if unknown_targets:
            raise ValueError(
                "selector_aliases reference unknown projections: "
                f"{sorted(unknown_targets)}"
            )

    def forward(self, batch: dict) -> dict:
        selector = str(
            resolve_homogeneous_scalar(
                batch[self.selector_key], label=self.selector_key
            )
        )
        selector = self.selector_aliases.get(selector, selector)
        if selector not in self.projections:
            raise KeyError(
                f"No projection for {self.selector_key}={selector!r}; "
                f"configured={list(self.projections)}"
            )
        value = batch[self.input_key]
        spec = self.projection_specs[selector]
        if value.shape[-1] != spec["input_dim"]:
            raise ValueError(
                f"{self.input_key} width for {selector!r} is {value.shape[-1]}, "
                f"expected {spec['input_dim']} ({spec['semantic']})"
            )
        batch[self.output_key] = self.projections[selector](value)
        return batch
