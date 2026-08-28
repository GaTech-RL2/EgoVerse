"""Pipeline-native nodes for the rank-safe end-to-end latent sampler."""

import math
from fractions import Fraction
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from egomimic.pipeline.core import Stage


class DPStyleObsEncoder(nn.Module):
    """Diffusion-Policy-style visual feature plus raw low-dimensional state.

    DP's PushT observation encoder is a shared ResNet18 + SpatialSoftmax(32)
    producing 64 image features, concatenated directly with the normalized 2-D
    agent position.  This adapter preserves that 66-D contract while exposing
    the ``forward_packed`` surface used by ``FusedObsEncoder``.
    """

    def __init__(self, obs_specs: Dict[str, dict], img_encoders: Dict[str, nn.Module]):
        super().__init__()
        if not obs_specs or not img_encoders:
            raise ValueError("DPStyleObsEncoder needs low-dim and image inputs")
        self.obs_specs = {str(k): dict(v) for k, v in obs_specs.items()}
        self.img_encoders = nn.ModuleDict(img_encoders)

    def forward_packed(self, *, obs_packed: dict, T_total: int, **kwargs):
        features = []
        # Match robomimic ObservationEncoder ordering: low-dimensional keys,
        # then RGB keys, each lexicographically stable within its modality.
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
        if any(int(x.shape[0]) != int(T_total) for x in features):
            raise ValueError("DPStyleObsEncoder inputs do not share T_total")
        return torch.cat(features, dim=-1)


class SharedAdapterObsEncoder(nn.Module):
    """One shared observation representation plus one embodiment adapter.

    Both branches are fused immediately into a single condition tensor. This
    preserves real front-camera weight sharing without exposing A/S streams to
    the sampler graph. The adapter owns only modalities that cannot be shared
    because their keys or dimensionalities differ by embodiment.
    """

    def __init__(self, shared_encoder: nn.Module, embodiment_adapter: nn.Module):
        super().__init__()
        self.shared_encoder = shared_encoder
        self.embodiment_adapter = embodiment_adapter

    def forward_packed(
        self, *, obs_packed: dict, embodiment_id: str, T_total: int, **kwargs
    ) -> torch.Tensor:
        call = {
            "obs_packed": obs_packed,
            "embodiment_id": embodiment_id,
            "T_total": T_total,
            **kwargs,
        }
        shared = self.shared_encoder.forward_packed(**call)
        adapter = self.embodiment_adapter.forward_packed(**call)
        if shared.shape[:-1] != adapter.shape[:-1]:
            raise ValueError(
                "Shared and embodiment-adapter features have incompatible "
                f"shapes: {tuple(shared.shape)} vs {tuple(adapter.shape)}"
            )
        return torch.cat((shared, adapter), dim=-1)


class FusedObsEncoder(Stage):
    """Encode and stack a standard MultiDataset observation window.

    The public batch remains batch-native: obs tensors enter as ``(B, N, ...)``
    and ``condition`` leaves as ``(B, N * d_model)``.  The legacy stem API is
    packed internally only for the duration of the encoder call; packing
    metadata never leaks into the Pipeline graph.
    """

    # The historical declarations remain the truthful training contract. At
    # rollout there is no action target, so only the condition is emitted.
    reads = ["obs/*", "embodiment", "actions"]
    writes = ["condition", "target"]
    reads_by_mode = {"rollout": ["obs/*", "embodiment"]}
    writes_by_mode = {"rollout": ["condition"]}

    def __init__(self, encoder: nn.Module, n_obs_steps: int = 2):
        super().__init__()
        self.encoder = encoder
        self.n_obs_steps = int(n_obs_steps)
        self.rollout_obs_steps = self.n_obs_steps
        if self.n_obs_steps <= 0:
            raise ValueError("n_obs_steps must be positive")

    def forward(self, batch: dict) -> dict:
        obs_packed = {
            key.split("/", 1)[1]: value
            for key, value in batch.items()
            if key.startswith("obs/")
        }
        reference = next(
            value for value in obs_packed.values() if torch.is_tensor(value)
        )
        batch_size, n_obs = int(reference.shape[0]), self.n_obs_steps
        if n_obs == 1:
            # PushT's canonical current-observation keymap returns (B, ...),
            # without a redundant singleton history axis.  Rollout assembly
            # does add that axis, so remove it only under the explicit marker.
            for key, value in list(obs_packed.items()):
                if not torch.is_tensor(value):
                    continue
                if int(value.shape[0]) != batch_size:
                    raise ValueError(
                        f"FusedObsEncoder: obs/{key} has batch {value.shape[0]}; "
                        f"expected {batch_size}"
                    )
                if "rollout_t" in batch:
                    if value.ndim < 2 or value.shape[1] != 1:
                        raise ValueError(
                            f"FusedObsEncoder rollout obs/{key} must have an "
                            f"explicit singleton history axis, got {tuple(value.shape)}"
                        )
                    obs_packed[key] = value[:, 0]
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
                        f"FusedObsEncoder: obs/{key} has shape {tuple(value.shape)}; "
                        f"expected leading dimensions {(batch_size, n_obs)}"
                    )
                obs_packed[key] = value.reshape(batch_size * n_obs, *value.shape[2:])
        total, device = batch_size * n_obs, reference.device
        dtype = reference.dtype if reference.is_floating_point() else torch.float32
        donor = torch.zeros((total, 1), device=device, dtype=dtype)
        cu = torch.arange(0, total + 1, n_obs, device=device, dtype=torch.long)
        for module in self.modules():
            if getattr(module, "crop_scope", None) == "episode":
                module._episode_cu = cu
        encoded = self.encoder.forward_packed(
            actions_packed=donor,
            obs_packed=obs_packed,
            cu_seqlens=cu,
            T_total=total,
            device=device,
            dtype=dtype,
            embodiment_id=str(batch["embodiment"]),
        )
        batch["condition"] = encoded.reshape(batch_size, n_obs * encoded.shape[-1])
        target = batch.pop("actions", None)
        if target is None and "rollout_t" not in batch:
            raise ValueError("FusedObsEncoder: training batch has no actions target")
        if target is not None:
            batch["target"] = target
        return batch


class GaussianLatentNoise(Stage):
    """Create independent Gaussian initial state without reading the target."""

    reads = ["condition"]
    writes = ["sampler/noise"]

    def __init__(self, action_horizon: int, latent_dim: int):
        super().__init__()
        self.action_horizon = int(action_horizon)
        self.latent_dim = int(latent_dim)
        if self.action_horizon <= 0 or self.latent_dim <= 0:
            raise ValueError("action_horizon and latent_dim must be positive")

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
            self.action_horizon,
            self.latent_dim,
            dtype=dtype,
            device=device,
        )
        return batch


class TemporalConvActionDecoder(nn.Module):
    """Decode a short latent sequence into a 2x-longer action sequence.

    The channel MLP first lifts each latent token into the decoder width. A
    transposed 1-D convolution then learns the temporal interpolation while
    projecting decoder channels into the embodiment's transformed training
    action channels.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        action_dim: int,
        latent_horizon: int,
        action_horizon: int,
        extra_hidden_layers: int = 0,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.action_dim = int(action_dim)
        self.latent_horizon = int(latent_horizon)
        self.action_horizon = int(action_horizon)
        self.extra_hidden_layers = int(extra_hidden_layers)
        if (
            min(
                self.latent_dim,
                self.hidden_dim,
                self.action_dim,
                self.latent_horizon,
                self.action_horizon,
            )
            <= 0
        ):
            raise ValueError(
                "Temporal decoder dimensions and horizons must be positive"
            )
        if self.extra_hidden_layers < 0:
            raise ValueError("extra_hidden_layers must be non-negative")
        if self.action_horizon != 2 * self.latent_horizon:
            raise ValueError(
                "TemporalConvActionDecoder currently requires action_horizon == "
                f"2 * latent_horizon, got {self.action_horizon} and "
                f"{self.latent_horizon}"
            )

        projection_layers: List[nn.Module] = [
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
        ]
        for _ in range(self.extra_hidden_layers):
            projection_layers.extend(
                [nn.Linear(self.hidden_dim, self.hidden_dim), nn.SiLU()]
            )
        self.channel_projection = nn.Sequential(*projection_layers)
        # (T - 1) * stride - 2 * padding + kernel_size = 2T.
        self.temporal_upsampler = nn.ConvTranspose1d(
            self.hidden_dim,
            self.action_dim,
            kernel_size=4,
            stride=2,
            padding=1,
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        expected = (self.latent_horizon, self.latent_dim)
        if latent.ndim != 3 or tuple(latent.shape[1:]) != expected:
            raise ValueError(
                "TemporalConvActionDecoder expected latent shape "
                f"(B, {self.latent_horizon}, {self.latent_dim}), got "
                f"{tuple(latent.shape)}"
            )
        hidden = self.channel_projection(latent).transpose(1, 2)
        action = self.temporal_upsampler(hidden).transpose(1, 2)
        expected_output = (
            int(latent.shape[0]),
            self.action_horizon,
            self.action_dim,
        )
        if tuple(action.shape) != expected_output:
            raise RuntimeError(
                "TemporalConvActionDecoder produced shape "
                f"{tuple(action.shape)}, expected {expected_output}"
            )
        return action


class MultiJActionSampler(Stage):
    """Pipeline sampler node with an explicitly injected denoising module."""

    reads = ["sampler/noise", "condition", "embodiment"]
    writes = ["pred_action", "log/*"]

    def __init__(
        self,
        denoising_module: nn.Module,
        condition_input_dim: int,
        action_horizon: int,
        action_dims: Dict[str, int],
        latent_dim: int = 128,
        condition_dim: int = 384,
        decoder_hidden_dim: int = 512,
        decoder_extra_hidden_layers_by_domain: Dict[str, int] = None,
        denoiser_hidden_dim: int = 512,
        num_inference_steps: int = 16,
        sampling_schedule: Dict[int, Dict[int, float]] = None,
        gradient_checkpointing: bool = True,
        gradient_accumulation_steps: int = 1,
        schedule_anchor_domain: str = "eva_bimanual",
        latent_horizon: int = None,
        decoder_type: str = "token_mlp",
    ):
        super().__init__()
        self.denoising_module = denoising_module
        self.condition_input_dim = int(condition_input_dim)
        self.action_horizon = int(action_horizon)
        # Historically the latent and output horizons were identical. Keep
        # action_horizon as the produced policy horizon and make only the
        # compressed vector-field/noise length independently configurable.
        self.latent_horizon = int(
            self.action_horizon if latent_horizon is None else latent_horizon
        )
        self.decoder_type = str(decoder_type)
        self.action_dims = {str(k): int(v) for k, v in dict(action_dims).items()}
        self.latent_dim = int(latent_dim)
        self.condition_dim = int(condition_dim)
        self.decoder_hidden_dim = int(decoder_hidden_dim)
        self.denoiser_hidden_dim = int(denoiser_hidden_dim)
        self.num_inference_steps = int(num_inference_steps)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.gradient_accumulation_steps = int(gradient_accumulation_steps)
        self.schedule_anchor_domain = str(schedule_anchor_domain)
        if decoder_extra_hidden_layers_by_domain is None:
            decoder_extra_hidden_layers_by_domain = {}
        unknown = set(decoder_extra_hidden_layers_by_domain) - set(self.action_dims)
        if unknown:
            raise ValueError(
                f"Decoder depth configured for unknown domains: {sorted(unknown)}"
            )
        self.decoder_extra_hidden_layers_by_domain = {
            domain: int(decoder_extra_hidden_layers_by_domain.get(domain, 0))
            for domain in self.action_dims
        }
        if any(v < 0 for v in self.decoder_extra_hidden_layers_by_domain.values()):
            raise ValueError("Decoder extra hidden-layer counts must be non-negative")
        if self.action_horizon <= 0 or self.latent_horizon <= 0 or self.latent_dim <= 0:
            raise ValueError(
                "action_horizon, latent_horizon, and latent_dim must be positive"
            )
        if self.decoder_hidden_dim <= 0 or self.num_inference_steps <= 0:
            raise ValueError(
                "decoder_hidden_dim and num_inference_steps must be positive"
            )
        if sampling_schedule is None:
            sampling_schedule = {
                1: {1: 0.5, 2: 0.5},
                2001: {2: 0.8, 4: 0.15, 8: 0.05},
            }
        self.sampling_schedule, self._sampling_cycles = self._compile_schedule(
            sampling_schedule
        )

        if self.decoder_type not in {"token_mlp", "temporal_conv"}:
            raise ValueError(
                "decoder_type must be one of {'token_mlp', 'temporal_conv'}, got "
                f"{self.decoder_type!r}"
            )
        if (
            self.decoder_type == "token_mlp"
            and self.latent_horizon != self.action_horizon
        ):
            raise ValueError(
                "token_mlp preserves sequence length, so latent_horizon must "
                "equal action_horizon"
            )

        decoders = {}
        for domain, action_dim in self.action_dims.items():
            if self.decoder_type == "temporal_conv":
                decoders[domain] = TemporalConvActionDecoder(
                    latent_dim=self.latent_dim,
                    hidden_dim=self.decoder_hidden_dim,
                    action_dim=action_dim,
                    latent_horizon=self.latent_horizon,
                    action_horizon=self.action_horizon,
                    extra_hidden_layers=self.decoder_extra_hidden_layers_by_domain[
                        domain
                    ],
                )
            else:
                layers: List[nn.Module] = [
                    nn.Linear(self.latent_dim, self.decoder_hidden_dim),
                    nn.SiLU(),
                    nn.Linear(self.decoder_hidden_dim, self.decoder_hidden_dim),
                    nn.SiLU(),
                ]
                for _ in range(self.decoder_extra_hidden_layers_by_domain[domain]):
                    layers.extend(
                        [
                            nn.Linear(self.decoder_hidden_dim, self.decoder_hidden_dim),
                            nn.SiLU(),
                        ]
                    )
                layers.append(nn.Linear(self.decoder_hidden_dim, action_dim))
                decoders[domain] = nn.Sequential(*layers)
        self.decoders = nn.ModuleDict(decoders)
        self.domain_embeddings = nn.ParameterDict(
            {
                domain: nn.Parameter(torch.empty(self.condition_dim).normal_(std=0.02))
                for domain in self.action_dims
            }
        )
        self.last_integration_step_sizes = None
        self.condition_projection = nn.Linear(
            self.condition_input_dim, self.condition_dim
        )
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.schedule_anchor_domain not in self.action_dims:
            raise ValueError("schedule_anchor_domain must name a configured domain")
        self.register_buffer("training_batches_seen", torch.zeros((), dtype=torch.long))
        self._validate_denoiser_contract()

    def _validate_denoiser_contract(self) -> None:
        proj_u = getattr(self.denoising_module, "proj_u", None)
        proj_d = getattr(self.denoising_module, "proj_d", None)
        if proj_u is not None and proj_u.in_features != self.latent_dim:
            raise ValueError(
                f"Denoiser input is {proj_u.in_features}, expected latent_dim={self.latent_dim}"
            )
        if proj_u is not None and proj_u.out_features < self.latent_dim:
            raise ValueError(
                "Rank-deficient denoiser input: "
                f"latent_dim={self.latent_dim} projects to {proj_u.out_features} features"
            )
        if proj_d is not None and proj_d.out_features != self.latent_dim:
            raise ValueError(
                f"Denoiser output is {proj_d.out_features}, expected {self.latent_dim}"
            )
        time_mode = getattr(self.denoising_module, "time_conditioning", "concat")
        if proj_u is not None:
            actual_hidden = (
                proj_u.out_features * 2
                if time_mode == "concat"
                else proj_u.out_features
            )
            if actual_hidden != self.denoiser_hidden_dim:
                raise ValueError(
                    f"Denoiser produces hidden_dim={actual_hidden}, expected "
                    f"{self.denoiser_hidden_dim}"
                )
        if proj_d is not None and proj_d.in_features != self.denoiser_hidden_dim:
            raise ValueError(
                f"Denoiser proj_d input is {proj_d.in_features}, expected "
                f"{self.denoiser_hidden_dim}"
            )
        pos_emb = getattr(self.denoising_module, "pos_emb", None)
        if pos_emb is not None and (
            pos_emb.ndim < 2 or int(pos_emb.shape[-2]) != self.latent_horizon
        ):
            raise ValueError(
                "Denoiser positional horizon is "
                f"{tuple(pos_emb.shape)}, expected sequence length "
                f"{self.latent_horizon}"
            )

    def condition_for_domain(
        self, condition: torch.Tensor, embodiment: str
    ) -> torch.Tensor:
        if embodiment not in self.domain_embeddings:
            raise KeyError(f"Unknown embodiment {embodiment!r}")
        if condition.shape[-1] != self.condition_dim:
            raise ValueError(
                f"Expected {self.condition_dim}-D sampler condition, got "
                f"{condition.shape[-1]}"
            )
        embedding = self.domain_embeddings[embodiment].to(condition)
        shape = [1] * (condition.ndim - 1) + [self.condition_dim]
        return condition + embedding.view(*shape)

    def _condition_from_batch(self, batch: dict, embodiment: str) -> torch.Tensor:
        condition = batch["condition"]
        if condition.shape[-1] != self.condition_input_dim:
            raise ValueError(
                f"Expected {self.condition_input_dim}-D pipeline condition, got "
                f"{condition.shape[-1]}"
            )
        projected = self.condition_projection(condition).unsqueeze(1)
        return self.condition_for_domain(projected, embodiment)

    def _optimizer_step(self, embodiment: str) -> int:
        if self.training and embodiment == self.schedule_anchor_domain:
            self.training_batches_seen.add_(1)
        batch_step = max(int(self.training_batches_seen.item()), 1)
        return (batch_step - 1) // self.gradient_accumulation_steps + 1

    def decoder(self, embodiment: str) -> nn.Module:
        if embodiment not in self.decoders:
            raise KeyError(f"Unknown embodiment {embodiment!r}")
        return self.decoders[embodiment]

    @staticmethod
    def _compile_schedule(schedule):
        normalized, cycles = {}, {}
        for raw_start, raw_split in dict(schedule).items():
            start = int(raw_start)
            split = {int(j): float(weight) for j, weight in dict(raw_split).items()}
            if start < 1:
                raise ValueError("sampling_schedule start steps must be positive")
            if not split or any(j not in {1, 2, 4, 8, 16, 128} for j in split):
                raise ValueError("sampling_schedule supports J in {1,2,4,8,16,128}")
            if any(weight <= 0.0 for weight in split.values()):
                raise ValueError("sampling_schedule weights must be positive")
            if not math.isclose(sum(split.values()), 1.0, rel_tol=0.0, abs_tol=1e-8):
                raise ValueError(
                    f"sampling_schedule[{start}] weights must sum to 1, got "
                    f"{sum(split.values())}"
                )
            fractions = {
                j: Fraction(str(weight)).limit_denominator(1000)
                for j, weight in split.items()
            }
            denominator = math.lcm(*(value.denominator for value in fractions.values()))
            counts = {
                j: value.numerator * (denominator // value.denominator)
                for j, value in fractions.items()
            }
            common = math.gcd(*counts.values())
            cycle = tuple(j for j in sorted(counts) for _ in range(counts[j] // common))
            normalized[start] = split
            cycles[start] = cycle
        starts = sorted(normalized)
        if not starts or starts[0] != 1:
            raise ValueError("sampling_schedule must define a phase starting at step 1")
        return (
            {start: normalized[start] for start in starts},
            {start: cycles[start] for start in starts},
        )

    def unroll_steps_at(self, optimizer_step: int) -> int:
        optimizer_step = max(int(optimizer_step), 1)
        start = max(step for step in self._sampling_cycles if step <= optimizer_step)
        cycle = self._sampling_cycles[start]
        return cycle[(optimizer_step - start) % len(cycle)]

    def sample_step_sizes(
        self, batch_size: int, num_steps: int, reference: torch.Tensor, generator=None
    ) -> torch.Tensor:
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if num_steps == 1:
            return torch.ones(
                batch_size, 1, device=reference.device, dtype=torch.float32
            )
        internal = (
            torch.rand(
                batch_size,
                num_steps - 1,
                device=reference.device,
                dtype=torch.float64,
                generator=generator,
            )
            .sort(dim=-1)
            .values
        )
        endpoints = torch.cat(
            (
                torch.zeros(
                    batch_size, 1, device=reference.device, dtype=torch.float64
                ),
                internal,
                torch.ones(batch_size, 1, device=reference.device, dtype=torch.float64),
            ),
            dim=-1,
        )
        # Keep the integration grid in fp32 even under bf16 autocast. Casting
        # sorted-uniform gaps to bf16 can round a small positive gap to zero;
        # accumulating bf16 time can also reach 1 early and make the final
        # residual non-positive.
        return endpoints.diff(dim=-1).to(device=reference.device, dtype=torch.float32)

    def _velocity(
        self, latent: torch.Tensor, time: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(
                self.denoising_module, latent, time, condition, use_reentrant=False
            )
        return self.denoising_module(latent, time, condition)

    def integrate(
        self,
        initial_latent: torch.Tensor,
        condition: torch.Tensor,
        num_steps: int,
        step_sizes: torch.Tensor = None,
    ) -> torch.Tensor:
        batch_size = initial_latent.shape[0]
        if step_sizes is None:
            step_sizes = torch.full(
                (batch_size, num_steps),
                1.0 / num_steps,
                device=initial_latent.device,
                dtype=torch.float32,
            )
        elif step_sizes.shape != (batch_size, num_steps):
            raise ValueError(
                f"step_sizes must be ({batch_size}, {num_steps}), got "
                f"{tuple(step_sizes.shape)}"
            )
        step_sizes = step_sizes.to(device=initial_latent.device, dtype=torch.float32)
        if not bool(torch.all(step_sizes > 0.0)):
            raise ValueError("Every integration step must be positive")
        grid_sum = step_sizes.sum(dim=-1)
        if not torch.allclose(
            grid_sum, torch.ones_like(grid_sum), rtol=1e-6, atol=1e-6
        ):
            raise ValueError("Every integration grid must sum to one")
        latent = initial_latent
        time = torch.zeros(
            batch_size, device=initial_latent.device, dtype=torch.float32
        )
        actual_steps = []
        for index in range(num_steps):
            dt = step_sizes[:, index]
            if not bool(torch.all(dt > 0.0)):
                raise ValueError("An integration step became non-positive")
            latent = latent + dt.reshape(
                batch_size, *([1] * (latent.ndim - 1))
            ) * self._velocity(latent, time, condition)
            actual_steps.append(dt.detach())
            time = time + dt
        self.last_integration_step_sizes = torch.stack(actual_steps, dim=-1)
        return latent

    def forward(self, batch: dict) -> dict:
        embodiment = str(batch["embodiment"])
        noise = batch["sampler/noise"]
        expected_noise_shape = (
            int(batch["condition"].shape[0]),
            self.latent_horizon,
            self.latent_dim,
        )
        if tuple(noise.shape) != expected_noise_shape:
            raise ValueError(
                f"Sampler noise has shape {tuple(noise.shape)}, expected "
                f"{expected_noise_shape}"
            )
        condition = self._condition_from_batch(batch, embodiment)
        if self.training:
            optimizer_step = self._optimizer_step(embodiment)
            num_steps = self.unroll_steps_at(optimizer_step)
            step_sizes = self.sample_step_sizes(noise.shape[0], num_steps, noise)
            batch["log/optimizer_step"] = float(optimizer_step)
        else:
            num_steps = self.num_inference_steps
            step_sizes = None
        endpoint = self.integrate(
            noise, condition, num_steps=num_steps, step_sizes=step_sizes
        )
        prediction = self.decoder(embodiment)(endpoint)
        expected_prediction_shape = (
            int(noise.shape[0]),
            self.action_horizon,
            self.action_dims[embodiment],
        )
        if tuple(prediction.shape) != expected_prediction_shape:
            raise RuntimeError(
                f"Decoder produced shape {tuple(prediction.shape)}, expected "
                f"{expected_prediction_shape}"
            )
        batch["pred_action"] = prediction
        batch["log/sampler_unroll_steps"] = float(num_steps)
        batch["log/sampler_noise_rms"] = noise.detach().square().mean().sqrt()
        batch["log/sampler_endpoint_rms"] = endpoint.detach().square().mean().sqrt()
        batch["log/sampler_prediction_rms"] = prediction.detach().square().mean().sqrt()
        return batch


class NativeActionMSELoss(Stage):
    """Strict train-only normalized native-action MSE node."""

    train_only = True
    reads = ["pred_action", "target"]
    writes = ["loss/native_action", "log/native_action"]

    def forward(self, batch: dict) -> dict:
        prediction, target = batch["pred_action"], batch["target"]
        if prediction.shape != target.shape:
            raise ValueError(
                "NativeActionMSELoss shape mismatch: "
                f"prediction={tuple(prediction.shape)} target={tuple(target.shape)}"
            )
        error = (prediction - target).square()
        pad_mask = batch.get("pad_mask")
        if pad_mask is None:
            loss = error.mean()
        else:
            mask = pad_mask.to(device=error.device, dtype=error.dtype)
            while mask.ndim < error.ndim:
                mask = mask.unsqueeze(-1)
            mask = mask.expand_as(error)
            loss = (error * mask).sum() / mask.sum().clamp_min(1.0)
        batch["loss/native_action"] = loss
        batch["log/native_action"] = loss.detach()
        return batch
