"""Pipeline-native nodes for the rank-safe end-to-end latent sampler."""

import math
from fractions import Fraction
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from egomimic.pipeline.core import Stage
from egomimic.pipeline.losses import conditional_energy_score


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


class EmbodimentProprioProjection(Stage):
    """Project U4 and Chain6 proprio through separate MLP branches."""

    reads = ["obs/state_agent_model", "embodiment"]
    writes = ["obs/proprio_condition"]

    def __init__(
        self,
        projections: Dict[str, dict],
        output_dim: int = 64,
    ):
        super().__init__()
        self.output_dim = int(output_dim)
        if self.output_dim <= 0:
            raise ValueError("output_dim must be positive")
        if not projections:
            raise ValueError("projections must configure at least one embodiment")

        self.projection_specs = {}
        branches = {}
        for domain, raw_spec in projections.items():
            spec = dict(raw_spec)
            source_dim = int(spec["source_dim"])
            hidden_dim = int(spec.get("hidden_dim", self.output_dim))
            if source_dim <= 0 or hidden_dim <= 0:
                raise ValueError("source_dim and hidden_dim must be positive")
            self.projection_specs[str(domain)] = {
                "source_dim": source_dim,
                "semantic": str(spec.get("semantic", "")),
            }
            branches[str(domain)] = nn.Sequential(
                nn.Linear(source_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, self.output_dim),
            )
        self.projections = nn.ModuleDict(branches)

    def forward(self, batch: dict) -> dict:
        domain = str(batch["embodiment"])
        if domain not in self.projections:
            raise KeyError(
                f"No proprio projection for {domain!r}; "
                f"configured={list(self.projections)}"
            )
        value = batch["obs/state_agent_model"]
        spec = self.projection_specs[domain]
        if value.shape[-1] != spec["source_dim"]:
            raise ValueError(
                f"{domain} state_agent_model width is {value.shape[-1]}, "
                f"expected {spec['source_dim']} ({spec['semantic']})"
            )
        batch["obs/proprio_condition"] = self.projections[domain](value)
        return batch


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

    def __init__(
        self,
        encoder: nn.Module,
        n_obs_steps: int = 2,
        required_obs_keys: List[str] | None = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.n_obs_steps = int(n_obs_steps)
        self.rollout_obs_steps = self.n_obs_steps
        if self.n_obs_steps <= 0:
            raise ValueError("n_obs_steps must be positive")
        if required_obs_keys is not None:
            required = tuple(
                key if str(key).startswith("obs/") else f"obs/{key}"
                for key in required_obs_keys
            )
            if not required:
                raise ValueError("required_obs_keys cannot be empty")
            self.reads = [*required, "embodiment", "actions"]
            self.reads_by_mode = {
                "rollout": [*required, "embodiment"],
            }

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
    """Create Gaussian initial states without reading target values.

    ``num_samples`` groups multiple independent latent draws under one encoded
    training or teacher-forced-validation condition. Observation-only rollout
    deliberately retains the historical single-sample tensor contract.
    """

    reads = ["condition"]
    writes = ["sampler/noise"]

    def __init__(
        self,
        num_tokens: int = None,
        latent_dim: int = 128,
        action_horizon: int = None,
        num_samples: int = 1,
    ):
        super().__init__()
        if num_tokens is None and action_horizon is None:
            raise ValueError("num_tokens must be configured")
        if num_tokens is not None and action_horizon is not None:
            raise ValueError("Configure num_tokens or legacy action_horizon, not both")
        self.num_tokens = int(action_horizon if num_tokens is None else num_tokens)
        self.latent_dim = int(latent_dim)
        self.num_samples = int(num_samples)
        if self.num_tokens <= 0 or self.latent_dim <= 0 or self.num_samples <= 0:
            raise ValueError("num_tokens, latent_dim, and num_samples must be positive")

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
        shape = (batch_size, self.num_tokens, self.latent_dim)
        if self.num_samples > 1 and "target" in batch:
            shape = (
                batch_size,
                self.num_samples,
                self.num_tokens,
                self.latent_dim,
            )
        batch["sampler/noise"] = torch.randn(*shape, dtype=dtype, device=device)
        return batch

    @property
    def action_horizon(self) -> int:
        """Read-only compatibility alias for legacy recipe audits."""

        return self.num_tokens


class TokenwiseMLPActionDecoder(nn.Sequential):
    """Decode every latent token independently without owning a horizon."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        action_dim: int,
        num_layers: int = None,
        extra_hidden_layers: int = None,
    ):
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.action_dim = int(action_dim)
        self.temporal_factor = 1
        if num_layers is None:
            num_layers = 3 + int(extra_hidden_layers or 0)
        elif extra_hidden_layers is not None:
            legacy_num_layers = 3 + int(extra_hidden_layers)
            if int(num_layers) != legacy_num_layers:
                raise ValueError(
                    "num_layers disagrees with deprecated extra_hidden_layers"
                )
        self.num_layers = int(num_layers)
        self.extra_hidden_layers = self.num_layers - 3
        if min(self.latent_dim, self.hidden_dim, self.action_dim) <= 0:
            raise ValueError("Tokenwise MLP decoder dimensions must be positive")
        if self.num_layers < 2:
            raise ValueError("Tokenwise MLP decoder num_layers must be at least 2")

        layers: List[nn.Module] = []
        for layer_index in range(self.num_layers):
            input_dim = self.latent_dim if layer_index == 0 else self.hidden_dim
            is_output = layer_index == self.num_layers - 1
            output_dim = self.action_dim if is_output else self.hidden_dim
            layers.append(nn.Linear(input_dim, output_dim))
            if not is_output:
                layers.append(nn.SiLU())
        super().__init__(*layers)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 3 or int(latent.shape[-1]) != self.latent_dim:
            raise ValueError(
                "TokenwiseMLPActionDecoder expected latent shape "
                f"(B, T, {self.latent_dim}), got {tuple(latent.shape)}"
            )
        return super().forward(latent)

    def output_num_tokens(self, input_num_tokens: int) -> int:
        input_num_tokens = int(input_num_tokens)
        if input_num_tokens <= 0:
            raise ValueError("input_num_tokens must be positive")
        return input_num_tokens


class TemporalConvActionDecoder(nn.Module):
    """Decode a latent sequence into a 2x-longer action sequence.

    The tokenwise MLP first projects every latent token into the embodiment's
    action space. A transposed 1-D convolution then learns the temporal
    interpolation entirely in that action space.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        action_dim: int,
        num_layers: int = None,
        extra_hidden_layers: int = None,
        project_to_action_before_temporal: bool = True,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.action_dim = int(action_dim)
        self.temporal_factor = 2
        if num_layers is None:
            num_layers = 3 + int(extra_hidden_layers or 0)
        elif extra_hidden_layers is not None:
            legacy_num_layers = 3 + int(extra_hidden_layers)
            if int(num_layers) != legacy_num_layers:
                raise ValueError(
                    "num_layers disagrees with deprecated extra_hidden_layers"
                )
        self.num_layers = int(num_layers)
        self.extra_hidden_layers = self.num_layers - 3
        self.project_to_action_before_temporal = bool(project_to_action_before_temporal)
        if min(self.latent_dim, self.hidden_dim, self.action_dim) <= 0:
            raise ValueError("Temporal decoder dimensions must be positive")
        if self.num_layers < 2:
            raise ValueError("Temporal decoder num_layers must be at least 2")

        projection_layers: List[nn.Module] = []
        projection_num_layers = self.num_layers - 1
        for layer_index in range(projection_num_layers):
            input_dim = self.latent_dim if layer_index == 0 else self.hidden_dim
            is_projection_output = layer_index == projection_num_layers - 1
            output_dim = (
                self.action_dim
                if self.project_to_action_before_temporal and is_projection_output
                else self.hidden_dim
            )
            projection_layers.append(nn.Linear(input_dim, output_dim))
            if not (self.project_to_action_before_temporal and is_projection_output):
                projection_layers.append(nn.SiLU())
        self.channel_projection = nn.Sequential(*projection_layers)
        temporal_channels = (
            self.action_dim
            if self.project_to_action_before_temporal
            else self.hidden_dim
        )
        # (T - 1) * stride - 2 * padding + kernel_size = 2T.
        self.temporal_upsampler = nn.ConvTranspose1d(
            temporal_channels,
            self.action_dim,
            kernel_size=4,
            stride=2,
            padding=1,
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 3 or int(latent.shape[-1]) != self.latent_dim:
            raise ValueError(
                "TemporalConvActionDecoder expected latent shape "
                f"(B, T, {self.latent_dim}), got {tuple(latent.shape)}"
            )
        hidden = self.channel_projection(latent).transpose(1, 2)
        action = self.temporal_upsampler(hidden).transpose(1, 2)
        expected_output = (
            int(latent.shape[0]),
            2 * int(latent.shape[1]),
            self.action_dim,
        )
        if tuple(action.shape) != expected_output:
            raise RuntimeError(
                "TemporalConvActionDecoder produced shape "
                f"{tuple(action.shape)}, expected {expected_output}"
            )
        return action

    def output_num_tokens(self, input_num_tokens: int) -> int:
        input_num_tokens = int(input_num_tokens)
        if input_num_tokens <= 0:
            raise ValueError("input_num_tokens must be positive")
        return 2 * input_num_tokens


class LatentFlowSampler(Stage):
    """Integrate an explicitly injected vector field to a latent endpoint."""

    reads = ["sampler/noise", "condition", "embodiment"]
    writes = ["sampler/endpoint", "log/*"]

    def __init__(
        self,
        denoising_module: nn.Module,
        condition_input_dim: int,
        domains: List[str],
        latent_dim: int = 128,
        condition_dim: int = 384,
        denoiser_hidden_dim: int = 512,
        num_inference_steps: int = 16,
        sampling_schedule: Dict[int, Dict[int, float]] = None,
        gradient_checkpointing: bool = True,
        gradient_accumulation_steps: int = 1,
        schedule_anchor_domain: str = "eva_bimanual",
        _legacy_decoders: nn.ModuleDict = None,
    ):
        super().__init__()
        self.denoising_module = denoising_module
        if _legacy_decoders is not None:
            self.decoders = _legacy_decoders
        self.condition_input_dim = int(condition_input_dim)
        self.domains = tuple(str(domain) for domain in domains)
        if not self.domains or len(set(self.domains)) != len(self.domains):
            raise ValueError("domains must contain unique embodiment names")
        self.latent_dim = int(latent_dim)
        self.condition_dim = int(condition_dim)
        self.denoiser_hidden_dim = int(denoiser_hidden_dim)
        self.num_inference_steps = int(num_inference_steps)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.gradient_accumulation_steps = int(gradient_accumulation_steps)
        self.schedule_anchor_domain = str(schedule_anchor_domain)
        if self.latent_dim <= 0 or self.num_inference_steps <= 0:
            raise ValueError("latent_dim and num_inference_steps must be positive")
        if sampling_schedule is None:
            sampling_schedule = {
                1: {1: 0.5, 2: 0.5},
                2001: {2: 0.8, 4: 0.15, 8: 0.05},
            }
        self.sampling_schedule, self._sampling_cycles = self._compile_schedule(
            sampling_schedule
        )

        self.domain_embeddings = nn.ParameterDict(
            {
                domain: nn.Parameter(torch.empty(self.condition_dim).normal_(std=0.02))
                for domain in self.domains
            }
        )
        self.last_integration_step_sizes = None
        self.condition_projection = nn.Linear(
            self.condition_input_dim, self.condition_dim
        )
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.schedule_anchor_domain not in self.domains:
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

    def _validate_noise_contract(self, noise: torch.Tensor) -> None:
        if noise.ndim not in {3, 4} or int(noise.shape[-1]) != self.latent_dim:
            raise ValueError(
                "LatentFlowSampler expected sampler/noise shape "
                f"(B, T, {self.latent_dim}) or (B, K, T, {self.latent_dim}), "
                f"got {tuple(noise.shape)}"
            )
        pos_emb = getattr(self.denoising_module, "pos_emb", None)
        if pos_emb is not None and (
            pos_emb.ndim < 2 or int(pos_emb.shape[-2]) != int(noise.shape[-2])
        ):
            raise ValueError(
                "Denoiser positional horizon is "
                f"{tuple(pos_emb.shape)}, but sampler/noise has "
                f"{int(noise.shape[-2])} tokens"
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
        self._validate_noise_contract(noise)
        expected_batch_size = int(batch["condition"].shape[0])
        if int(noise.shape[0]) != expected_batch_size:
            raise ValueError(
                f"sampler/noise batch is {int(noise.shape[0])}, expected "
                f"{expected_batch_size} from condition"
            )
        condition = self._condition_from_batch(batch, embodiment)
        grouped = noise.ndim == 4
        if grouped:
            batch_size, num_samples, num_tokens, latent_dim = noise.shape
            flat_noise = noise.reshape(batch_size * num_samples, num_tokens, latent_dim)
            condition = condition.repeat_interleave(num_samples, dim=0)
        else:
            batch_size, num_samples = int(noise.shape[0]), 1
            flat_noise = noise
        if self.training:
            optimizer_step = self._optimizer_step(embodiment)
            num_steps = self.unroll_steps_at(optimizer_step)
            # One integration grid is drawn per condition and shared by all K
            # latent seeds. Gaussian noise is therefore the only within-group
            # generator randomness when configured dropout is zero.
            step_sizes = self.sample_step_sizes(batch_size, num_steps, flat_noise)
            if grouped:
                step_sizes = step_sizes.repeat_interleave(num_samples, dim=0)
            batch["log/optimizer_step"] = float(optimizer_step)
        else:
            num_steps = self.num_inference_steps
            step_sizes = None
        flat_endpoint = self.integrate(
            flat_noise, condition, num_steps=num_steps, step_sizes=step_sizes
        )
        endpoint = (
            flat_endpoint.reshape(
                batch_size, num_samples, flat_endpoint.shape[-2], latent_dim
            )
            if grouped
            else flat_endpoint
        )
        batch["sampler/endpoint"] = endpoint
        batch["log/sampler_unroll_steps"] = float(num_steps)
        batch["log/sampler_noise_rms"] = noise.detach().square().mean().sqrt()
        batch["log/sampler_endpoint_rms"] = endpoint.detach().square().mean().sqrt()
        return batch


class PerEmbodimentActionDecoder(Stage):
    """Route one latent endpoint to an embodiment-specific action decoder."""

    reads = ["sampler/endpoint", "embodiment"]
    writes = ["pred_action", "pred_action_samples", "log/*"]

    def __init__(self, decoders: Dict[str, nn.Module]):
        super().__init__()
        configured = {
            str(domain): decoder for domain, decoder in dict(decoders).items()
        }
        if not configured:
            raise ValueError("decoders must configure at least one embodiment")
        if any(not isinstance(decoder, nn.Module) for decoder in configured.values()):
            raise TypeError("Every per-embodiment decoder must be an nn.Module")

        self.decoders = nn.ModuleDict(configured)
        self.domains = tuple(configured)
        latent_dims = {
            int(getattr(decoder, "latent_dim", -1)) for decoder in configured.values()
        }
        temporal_factors = {
            int(getattr(decoder, "temporal_factor", -1))
            for decoder in configured.values()
        }
        if len(latent_dims) != 1 or min(latent_dims) <= 0:
            raise ValueError("All decoders must expose the same positive latent_dim")
        if len(temporal_factors) != 1 or min(temporal_factors) <= 0:
            raise ValueError(
                "All decoders must expose the same positive temporal_factor"
            )
        self.latent_dim = latent_dims.pop()
        self.temporal_factor = temporal_factors.pop()
        self.action_dims = {
            domain: int(getattr(decoder, "action_dim", -1))
            for domain, decoder in configured.items()
        }
        if any(action_dim <= 0 for action_dim in self.action_dims.values()):
            raise ValueError("Every decoder must expose a positive action_dim")
        if any(
            not callable(getattr(decoder, "output_num_tokens", None))
            for decoder in configured.values()
        ):
            raise ValueError("Every decoder must expose output_num_tokens()")

    def decoder_for(self, embodiment: str) -> nn.Module:
        embodiment = str(embodiment)
        if embodiment not in self.decoders:
            raise KeyError(
                f"Unknown embodiment {embodiment!r}; configured={list(self.domains)}"
            )
        return self.decoders[embodiment]

    def output_num_tokens(self, input_num_tokens: int) -> int:
        outputs = {
            int(decoder.output_num_tokens(input_num_tokens))
            for decoder in self.decoders.values()
        }
        if len(outputs) != 1:
            raise RuntimeError(
                "Per-embodiment decoders disagree on temporal output length"
            )
        return outputs.pop()

    def forward(self, batch: dict) -> dict:
        embodiment = str(batch["embodiment"])
        endpoint = batch["sampler/endpoint"]
        if endpoint.ndim not in {3, 4} or int(endpoint.shape[-1]) != self.latent_dim:
            raise ValueError(
                "PerEmbodimentActionDecoder expected sampler/endpoint shape "
                f"(B, T, {self.latent_dim}) or (B, K, T, {self.latent_dim}), "
                f"got {tuple(endpoint.shape)}"
            )
        grouped = endpoint.ndim == 4
        if grouped:
            batch_size, num_samples, num_tokens, latent_dim = endpoint.shape
            decoder_input = endpoint.reshape(
                batch_size * num_samples, num_tokens, latent_dim
            )
        else:
            batch_size, num_samples, num_tokens = (
                int(endpoint.shape[0]),
                1,
                int(endpoint.shape[1]),
            )
            decoder_input = endpoint
        flat_prediction = self.decoder_for(embodiment)(decoder_input)
        expected_flat = (
            batch_size * num_samples,
            self.output_num_tokens(num_tokens),
            self.action_dims[embodiment],
        )
        if tuple(flat_prediction.shape) != expected_flat:
            raise RuntimeError(
                f"Decoder for {embodiment!r} produced "
                f"{tuple(flat_prediction.shape)}, expected {expected_flat}"
            )
        prediction_samples = flat_prediction.reshape(
            batch_size,
            num_samples,
            expected_flat[1],
            expected_flat[2],
        )
        # Existing rollout/evaluator consumers retain their rank-3 contract.
        # The complete grouped tensor is explicit for distributional losses.
        batch["pred_action_samples"] = prediction_samples
        batch["pred_action"] = prediction_samples[:, 0]
        batch["log/sampler_prediction_rms"] = (
            prediction_samples.detach().square().mean().sqrt()
        )
        return batch


class MultiJActionSampler(LatentFlowSampler):
    """Legacy sampler-plus-decoder composite kept for existing recipes."""

    reads = ["sampler/noise", "condition", "embodiment"]
    writes = ["sampler/endpoint", "pred_action", "log/*"]

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
        action_horizon = int(action_horizon)
        latent_horizon = int(
            action_horizon if latent_horizon is None else latent_horizon
        )
        action_dims = {str(k): int(v) for k, v in dict(action_dims).items()}
        decoder_hidden_dim = int(decoder_hidden_dim)
        decoder_type = str(decoder_type)
        if decoder_extra_hidden_layers_by_domain is None:
            decoder_extra_hidden_layers_by_domain = {}
        unknown = set(decoder_extra_hidden_layers_by_domain) - set(action_dims)
        if unknown:
            raise ValueError(
                f"Decoder depth configured for unknown domains: {sorted(unknown)}"
            )
        decoder_depths = {
            domain: int(decoder_extra_hidden_layers_by_domain.get(domain, 0))
            for domain in action_dims
        }
        if any(depth < 0 for depth in decoder_depths.values()):
            raise ValueError("Decoder extra hidden-layer counts must be non-negative")
        if action_horizon <= 0 or latent_horizon <= 0:
            raise ValueError("action_horizon and latent_horizon must be positive")
        if decoder_hidden_dim <= 0:
            raise ValueError("decoder_hidden_dim must be positive")
        if decoder_type not in {"token_mlp", "temporal_conv"}:
            raise ValueError(
                "decoder_type must be one of {'token_mlp', 'temporal_conv'}, got "
                f"{decoder_type!r}"
            )
        if decoder_type == "token_mlp" and latent_horizon != action_horizon:
            raise ValueError(
                "token_mlp preserves sequence length, so latent_horizon must "
                "equal action_horizon"
            )
        if decoder_type == "temporal_conv" and action_horizon != 2 * latent_horizon:
            raise ValueError(
                "temporal_conv requires action_horizon == 2 * latent_horizon"
            )

        decoder_cls = (
            TemporalConvActionDecoder
            if decoder_type == "temporal_conv"
            else TokenwiseMLPActionDecoder
        )
        decoders = nn.ModuleDict(
            {
                domain: decoder_cls(
                    latent_dim=int(latent_dim),
                    hidden_dim=decoder_hidden_dim,
                    action_dim=action_dim,
                    extra_hidden_layers=decoder_depths[domain],
                    **(
                        {"project_to_action_before_temporal": False}
                        if decoder_type == "temporal_conv"
                        else {}
                    ),
                )
                for domain, action_dim in action_dims.items()
            }
        )

        super().__init__(
            denoising_module=denoising_module,
            condition_input_dim=condition_input_dim,
            domains=list(action_dims),
            latent_dim=latent_dim,
            condition_dim=condition_dim,
            denoiser_hidden_dim=denoiser_hidden_dim,
            num_inference_steps=num_inference_steps,
            sampling_schedule=sampling_schedule,
            gradient_checkpointing=gradient_checkpointing,
            gradient_accumulation_steps=gradient_accumulation_steps,
            schedule_anchor_domain=schedule_anchor_domain,
            _legacy_decoders=decoders,
        )
        self.action_horizon = action_horizon
        self.latent_horizon = latent_horizon
        self.action_dims = action_dims
        self.decoder_hidden_dim = decoder_hidden_dim
        self.decoder_type = decoder_type
        self.decoder_extra_hidden_layers_by_domain = decoder_depths

        pos_emb = getattr(self.denoising_module, "pos_emb", None)
        if pos_emb is not None and (
            pos_emb.ndim < 2 or int(pos_emb.shape[-2]) != self.latent_horizon
        ):
            raise ValueError(
                "Denoiser positional horizon is "
                f"{tuple(pos_emb.shape)}, expected sequence length "
                f"{self.latent_horizon}"
            )

    def decoder(self, embodiment: str) -> nn.Module:
        embodiment = str(embodiment)
        if embodiment not in self.decoders:
            raise KeyError(f"Unknown embodiment {embodiment!r}")
        return self.decoders[embodiment]

    def forward(self, batch: dict) -> dict:
        batch = super().forward(batch)
        embodiment = str(batch["embodiment"])
        endpoint = batch["sampler/endpoint"]
        prediction = self.decoder(embodiment)(endpoint)
        expected = (
            int(endpoint.shape[0]),
            self.action_horizon,
            self.action_dims[embodiment],
        )
        if tuple(prediction.shape) != expected:
            raise RuntimeError(
                f"Decoder produced shape {tuple(prediction.shape)}, expected {expected}"
            )
        batch["pred_action"] = prediction
        batch["log/sampler_prediction_rms"] = prediction.detach().square().mean().sqrt()
        return batch


class PerEmbodimentActionCanonicalizer(Stage):
    """Apply the deployed action equivalence before any loss or rollout.

    Decoder outputs are normalized tokens. This node uses the exact dataset
    normalization to enter physical units, applies one embodiment-specific
    differentiable canonicalizer, then returns to normalized model units. Raw
    outputs are retained only for diagnostics. Because the canonicalized tensor
    replaces ``pred_action``, training, teacher-forced evaluation, and rollout
    all consume the same representation.
    """

    reads = ["pred_action", "embodiment"]
    writes = [
        "pred_action",
        "pred_action_samples",
        "raw_pred_action",
        "raw_pred_action_samples",
        "raw_target",
        "target",
        "log/canonicalization_rmse",
        "log/action_representation_mse",
        "log/raw_action_mse",
        "loss/action_representation",
    ]

    def __init__(
        self,
        canonicalizers: Dict[str, nn.Module],
        representation_loss_weight: float = 0.0,
    ):
        super().__init__()
        self.canonicalizers = nn.ModuleDict(dict(canonicalizers))
        self.representation_loss_weight = float(representation_loss_weight)
        if not self.canonicalizers:
            raise ValueError("At least one action canonicalizer is required")
        if self.representation_loss_weight < 0.0:
            raise ValueError("representation_loss_weight must be non-negative")
        self._normalization_buffers: dict[str, tuple[str, str]] = {}

    def bind_action_normalization(
        self,
        domain: str,
        *,
        norm_mode: str,
        stats: dict,
    ) -> None:
        """Bind the exact affine action normalization used by the dataset."""

        domain = str(domain)
        if domain not in self.canonicalizers:
            return
        if domain in self._normalization_buffers:
            raise RuntimeError(f"Action normalization already bound for {domain!r}")
        norm_mode = str(norm_mode)
        if norm_mode == "zscore":
            offset = torch.as_tensor(stats["mean"], dtype=torch.float32)
            scale = torch.as_tensor(stats["std"], dtype=torch.float32) + 1e-6
        elif norm_mode == "minmax":
            minimum = torch.as_tensor(stats["min"], dtype=torch.float32)
            maximum = torch.as_tensor(stats["max"], dtype=torch.float32)
            scale = 0.5 * (maximum - minimum + 1e-6)
            offset = minimum + scale
        elif norm_mode == "quantile":
            minimum = torch.as_tensor(stats["quantile_1"], dtype=torch.float32)
            maximum = torch.as_tensor(stats["quantile_99"], dtype=torch.float32)
            scale = 0.5 * (maximum - minimum + 1e-6)
            offset = minimum + scale
        else:
            raise ValueError(f"Unsupported action normalization mode {norm_mode!r}")
        expected_dim = getattr(self.canonicalizers[domain], "action_dim", None)
        if expected_dim is not None and int(offset.shape[-1]) != int(expected_dim):
            raise ValueError(
                f"Canonicalizer for {domain!r} expects action_dim={expected_dim}, "
                f"but normalization has shape {tuple(offset.shape)}"
            )
        index = len(self._normalization_buffers)
        offset_name = f"_action_norm_offset_{index}"
        scale_name = f"_action_norm_scale_{index}"
        self.register_buffer(offset_name, offset, persistent=True)
        self.register_buffer(scale_name, scale, persistent=True)
        self._normalization_buffers[domain] = (offset_name, scale_name)

    def canonicalize_normalized_actions(
        self, actions: torch.Tensor, domain: str
    ) -> torch.Tensor:
        domain = str(domain)
        if domain not in self.canonicalizers:
            return actions.float()
        if domain not in self._normalization_buffers:
            raise RuntimeError(
                f"Action normalization was not bound for canonicalized domain {domain!r}"
            )
        offset_name, scale_name = self._normalization_buffers[domain]
        offset = getattr(self, offset_name)
        scale = getattr(self, scale_name)
        physical = actions.float() * scale + offset
        canonical = self.canonicalizers[domain](physical)
        return (canonical - offset) / scale

    def unnormalize_actions(self, actions: torch.Tensor, domain: str) -> torch.Tensor:
        domain = str(domain)
        if domain not in self._normalization_buffers:
            raise RuntimeError(f"Action normalization was not bound for {domain!r}")
        offset_name, scale_name = self._normalization_buffers[domain]
        return actions.float() * getattr(self, scale_name) + getattr(self, offset_name)

    def forward(self, batch: dict) -> dict:
        domain = str(batch["embodiment"])
        raw_prediction = batch["pred_action"]
        batch["raw_pred_action"] = raw_prediction
        batch["pred_action"] = self.canonicalize_normalized_actions(
            raw_prediction, domain
        )
        residuals = [
            (batch["pred_action"].detach() - raw_prediction.detach().float()).square()
        ]
        if "pred_action_samples" in batch:
            raw_samples = batch["pred_action_samples"]
            canonical_samples = self.canonicalize_normalized_actions(
                raw_samples, domain
            )
            batch["raw_pred_action_samples"] = raw_samples
            batch["pred_action_samples"] = canonical_samples
            # Keep the rank-3 compatibility output exactly equal to member zero.
            batch["pred_action"] = canonical_samples[:, 0]
            residuals = [
                (canonical_samples.detach() - raw_samples.detach().float()).square()
            ]
            diagnostic_input = raw_samples
        else:
            diagnostic_input = raw_prediction
        if "target" in batch:
            raw_target = batch["target"]
            batch["raw_target"] = raw_target
            batch["target"] = self.canonicalize_normalized_actions(raw_target, domain)
            if "raw_pred_action_samples" in batch:
                raw_error = (
                    batch["raw_pred_action_samples"].float()
                    - raw_target.float()[:, None]
                ).square()
            else:
                raw_error = (raw_prediction.float() - raw_target.float()).square()
            batch["log/raw_action_mse"] = raw_error.mean().detach()
        representation_mse = torch.cat(
            [residual.reshape(-1) for residual in residuals]
        ).mean()
        batch["log/canonicalization_rmse"] = representation_mse.sqrt()
        batch["log/action_representation_mse"] = representation_mse
        if "target" in batch and self.representation_loss_weight > 0.0:
            # This selects a stable raw representative of each executed action
            # equivalence class; both scientific arms use the identical term.
            live_residual = (
                batch["pred_action_samples"] - batch["raw_pred_action_samples"].float()
                if "pred_action_samples" in batch
                else batch["pred_action"] - batch["raw_pred_action"].float()
            )
            batch["loss/action_representation"] = (
                self.representation_loss_weight * live_residual.square().mean()
            )
        diagnostics = getattr(self.canonicalizers[domain], "diagnostics", None)
        if diagnostics is not None:
            physical = self.unnormalize_actions(diagnostic_input, domain)
            for name, value in diagnostics(physical).items():
                batch[f"log/{name}"] = value.detach()
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


class ConditionalEnergyScoreLoss(Stage):
    """Train-only conditional energy score over grouped action chunks.

    This is a replacement endpoint objective, not an auxiliary term. Native
    MSE remains a detached diagnostic and contributes no gradient to
    ``sum_losses``.
    """

    train_only = True
    reads = ["pred_action_samples", "target"]
    writes = [
        "loss/conditional_energy_score",
        "log/conditional_energy_score",
        "log/energy_attraction",
        "log/energy_repulsion",
        "log/energy_pairwise_distance",
        "log/native_action",
        "log/canonical_action_mse",
        "log/ensemble_mean_mse",
        "log/best_of_k_mse",
    ]

    def __init__(
        self,
        beta: float = 1.0,
        normalize_by_dimension: bool = True,
        expected_num_samples: int | None = None,
    ):
        super().__init__()
        self.beta = float(beta)
        self.normalize_by_dimension = bool(normalize_by_dimension)
        self.expected_num_samples = (
            None if expected_num_samples is None else int(expected_num_samples)
        )
        if not math.isfinite(self.beta) or not 0.0 < self.beta < 2.0:
            raise ValueError(f"beta must satisfy 0 < beta < 2, got {self.beta}")
        if self.expected_num_samples is not None and self.expected_num_samples < 2:
            raise ValueError("expected_num_samples must be at least two")

    def forward(self, batch: dict) -> dict:
        prediction_samples = batch["pred_action_samples"]
        if (
            self.expected_num_samples is not None
            and prediction_samples.ndim >= 2
            and int(prediction_samples.shape[1]) != self.expected_num_samples
        ):
            raise ValueError(
                "ConditionalEnergyScoreLoss expected "
                f"K={self.expected_num_samples}, got "
                f"shape {tuple(prediction_samples.shape)}"
            )
        target = batch["target"]
        metrics = conditional_energy_score(
            prediction_samples,
            target,
            beta=self.beta,
            normalize_by_dimension=self.normalize_by_dimension,
            pad_mask=batch.get("pad_mask"),
        )
        batch["loss/conditional_energy_score"] = metrics["score"]
        for destination, source in (
            ("log/conditional_energy_score", "score"),
            ("log/energy_attraction", "attraction"),
            ("log/energy_repulsion", "repulsion"),
            ("log/energy_pairwise_distance", "pairwise_distance"),
            ("log/native_action", "mse"),
            ("log/canonical_action_mse", "mse"),
            ("log/ensemble_mean_mse", "ensemble_mean_mse"),
            ("log/best_of_k_mse", "best_of_k_mse"),
        ):
            batch[destination] = metrics[source].detach()
        return batch


class GroupedActionMSELoss(Stage):
    """Matched K-sample MSE control with detached energy-score diagnostics."""

    train_only = True
    reads = ["pred_action_samples", "target"]
    writes = [
        "loss/grouped_action_mse",
        "log/native_action",
        "log/canonical_action_mse",
        "log/conditional_energy_score",
        "log/energy_attraction",
        "log/energy_repulsion",
        "log/energy_pairwise_distance",
        "log/ensemble_mean_mse",
        "log/best_of_k_mse",
    ]

    def __init__(
        self,
        beta: float = 1.0,
        normalize_by_dimension: bool = True,
        expected_num_samples: int | None = None,
    ):
        super().__init__()
        self.beta = float(beta)
        self.normalize_by_dimension = bool(normalize_by_dimension)
        self.expected_num_samples = (
            None if expected_num_samples is None else int(expected_num_samples)
        )
        if not math.isfinite(self.beta) or not 0.0 < self.beta < 2.0:
            raise ValueError(f"beta must satisfy 0 < beta < 2, got {self.beta}")
        if self.expected_num_samples is not None and self.expected_num_samples < 2:
            raise ValueError("expected_num_samples must be at least two")

    def forward(self, batch: dict) -> dict:
        prediction_samples = batch["pred_action_samples"]
        target = batch["target"]
        if (
            self.expected_num_samples is not None
            and prediction_samples.ndim >= 2
            and int(prediction_samples.shape[1]) != self.expected_num_samples
        ):
            raise ValueError(
                "GroupedActionMSELoss expected "
                f"K={self.expected_num_samples}, got "
                f"shape {tuple(prediction_samples.shape)}"
            )
        error = (prediction_samples.float() - target.float()[:, None]).square()
        pad_mask = batch.get("pad_mask")
        if pad_mask is None:
            loss = error.mean()
        else:
            mask = pad_mask.to(device=error.device, dtype=torch.float32)
            if mask.ndim == 2:
                mask = mask[:, None, :, None]
            elif mask.ndim == 3:
                mask = mask[:, None]
            else:
                raise ValueError(
                    "pad_mask must have shape (B,H) or (B,H,D), got "
                    f"{tuple(mask.shape)}"
                )
            mask = mask.expand_as(error)
            loss = (error * mask).sum() / mask.sum().clamp_min(1.0)
        metrics = conditional_energy_score(
            prediction_samples,
            target,
            beta=self.beta,
            normalize_by_dimension=self.normalize_by_dimension,
            pad_mask=pad_mask,
        )
        batch["loss/grouped_action_mse"] = loss
        batch["log/native_action"] = loss.detach()
        batch["log/canonical_action_mse"] = loss.detach()
        for destination, source in (
            ("log/conditional_energy_score", "score"),
            ("log/energy_attraction", "attraction"),
            ("log/energy_repulsion", "repulsion"),
            ("log/energy_pairwise_distance", "pairwise_distance"),
            ("log/ensemble_mean_mse", "ensemble_mean_mse"),
            ("log/best_of_k_mse", "best_of_k_mse"),
        ):
            batch[destination] = metrics[source].detach()
        return batch
