"""Lightning wrapper and telemetry for the released UNITE register policy."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import Any

import hydra
import torch
import torch.nn as nn

from egomimic.pl_utils.pl_model import ModelWrapper
from egomimic.pipeline.stages_unite_released import ReleasedRecipeUniteLatentPolicy


class ReleasedUniteModelWrapper(ModelWrapper):
    """Optimize the joint UNITE objective and report normalized components."""

    gradient_telemetry_cadence = 100
    _component_keys = (
        ("ReconstructionLoss", "loss/unite_reconstruction"),
        ("FlowLoss", "loss/unite_latent"),
        ("ReconstructionL1", "log/unite_reconstruction_l1"),
    )
    _content_only = ("content_projection.", "content_pos_emb")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        config_tree = getattr(self.hparams, "config_tree", None)
        cfg = self._as_config(config_tree) if config_tree is not None else None
        configured = (
            None if cfg is None else cfg.model.get("share_encoder_denoiser", None)
        )
        if configured is not None and not isinstance(configured, bool):
            raise TypeError("model.share_encoder_denoiser must be a boolean")
        self._configured_share_encoder_denoiser = configured
        self._unite_validation_sums = OrderedDict()
        self._unite_validation_count = 0

    @torch.inference_mode()
    def forward_eval(self, batch: Mapping) -> OrderedDict:
        """Expose ordinary inference alongside the UNITE diagnostic boundary."""

        return self.model.forward_eval(batch)

    @torch.inference_mode()
    def forward_unite_diagnostics(
        self,
        batch: Mapping,
        *,
        raw_noise_levels: tuple[float, ...] | list[float],
    ) -> OrderedDict:
        """Run the pipeline prefix and capture released-UNITE diagnostics."""

        if not isinstance(batch, Mapping):
            raise TypeError("UNITE diagnostics input must be a source mapping")
        stages = self.model.pipeline.stages
        policies = [
            stage for stage in stages if isinstance(stage, ReleasedRecipeUniteLatentPolicy)
        ]
        if len(policies) != 1:
            raise RuntimeError(
                "Released UNITE diagnostics require exactly one latent policy; "
                f"found {len(policies)}"
            )
        policy = policies[0]
        diagnostics = OrderedDict()
        for source, source_batch in batch.items():
            if not isinstance(source_batch, Mapping):
                raise TypeError(f"UNITE diagnostic source {source!r} must be a mapping")
            result = dict(source_batch)
            for stage in stages:
                if stage is policy:
                    break
                result = stage.execute(result, mode="inference")
            required = {"sampler/noise", "condition", "target", "embodiment"}
            missing = required - set(result)
            if missing:
                raise RuntimeError(
                    f"Released UNITE diagnostic prefix for {source!r} is missing "
                    f"{sorted(missing)}"
                )
            diagnostics[source] = policy.validation_diagnostics(
                noise=result["sampler/noise"],
                condition=result["condition"],
                target=result["target"],
                embodiment=result["embodiment"],
                raw_noise_levels=raw_noise_levels,
            )
        return diagnostics

    @staticmethod
    def _finite_scalar(value: Any, label: str) -> torch.Tensor:
        if not torch.is_tensor(value) or value.ndim != 0:
            raise TypeError(f"{label} must be a scalar tensor")
        if not bool(torch.isfinite(value.detach())):
            raise RuntimeError(f"Non-finite UNITE metric {label}")
        return value

    def _unite_topology(self) -> tuple[nn.Module, nn.Module, bool]:
        stages = getattr(getattr(self.model, "pipeline", None), "stages", None)
        if stages is None:
            raise RuntimeError("Released UNITE requires a registered Pipeline")
        matches = [
            (stage, stage.generative_encoder)
            for stage in stages
            if isinstance(getattr(stage, "generative_encoder", None), nn.Module)
            and hasattr(stage.generative_encoder, "denoising_module")
        ]
        if len(matches) != 1:
            raise RuntimeError(
                "UNITE telemetry requires exactly one generative encoder; "
                f"found {len(matches)}"
            )
        stage, encoder = matches[0]
        shared = not hasattr(encoder, "tokenization_module")
        if not shared and encoder.tokenization_module is encoder.denoising_module:
            raise RuntimeError("Separate UNITE topology aliases its two backbones")
        configured = self._configured_share_encoder_denoiser
        if configured is not None and configured is not shared:
            raise RuntimeError(
                "Resolved UNITE sharing flag disagrees with the runtime topology"
            )
        return stage, encoder, shared

    @staticmethod
    def _active_domains(predictions: Mapping, encoder: nn.Module) -> tuple[str, ...]:
        resolver = getattr(encoder, "_resolve_domain", None)
        if resolver is None:
            raise RuntimeError("UNITE encoder does not expose branch resolution")
        domains = []
        for source, result in predictions.items():
            if not isinstance(result, Mapping) or "embodiment" not in result:
                raise RuntimeError(
                    f"UNITE source {source!r} is missing embodiment metadata"
                )
            domain = str(resolver(result["embodiment"]))
            if domain not in domains:
                domains.append(domain)
        if not domains:
            raise RuntimeError("UNITE telemetry received no active branches")
        configured = tuple(str(domain) for domain in getattr(encoder, "domains", ()))
        if configured and set(domains) != set(configured):
            raise RuntimeError(
                "UNITE telemetry batches must cover every configured branch"
            )
        return configured or tuple(domains)

    @staticmethod
    def _branch_parameter(name: str, prefix: str, domains: Sequence[str]) -> bool:
        return any(
            name == f"{prefix}.{domain}" or name.startswith(f"{prefix}.{domain}.")
            for domain in domains
        )

    @classmethod
    def _backbone_parameter(cls, name: str, prefix: str) -> bool:
        marker = f"{prefix}."
        return name.startswith(marker) and not name[len(marker) :].startswith(
            cls._content_only
        )

    @staticmethod
    def _select_named(encoder: nn.Module, label: str, predicate):
        named = tuple(
            (name, parameter)
            for name, parameter in encoder.named_parameters()
            if predicate(name)
        )
        identities = [id(parameter) for _, parameter in named]
        if not named or len(set(identities)) != len(identities):
            raise RuntimeError(
                f"UNITE {label} parameter selection is empty or duplicated"
            )
        if any(not parameter.requires_grad for _, parameter in named):
            raise RuntimeError(f"UNITE {label} telemetry requires trainable parameters")
        return named

    @classmethod
    def _shared_named_parameters(
        cls,
        encoder: nn.Module,
        domains: Sequence[str],
        *,
        include_null_input: bool,
    ) -> tuple[tuple[str, nn.Parameter], ...]:
        # Action/content projections are reconstruction-only and cannot enter the
        # shared-gradient cosine. The null input enters only when dropout uses it.
        return cls._select_named(
            encoder,
            "shared",
            lambda name: (
                cls._backbone_parameter(name, "denoising_module")
                or name.startswith(("output_norm.", "condition_projection."))
                or cls._branch_parameter(name, "domain_embeddings", domains)
                or (
                    include_null_input
                    and cls._branch_parameter(name, "null_condition_inputs", domains)
                )
            ),
        )

    @classmethod
    def _separate_named_parameters(
        cls,
        encoder: nn.Module,
        domains: Sequence[str],
        *,
        include_denoiser_null_input: bool,
    ) -> tuple[
        tuple[tuple[str, nn.Parameter], ...],
        tuple[tuple[str, nn.Parameter], ...],
    ]:
        tokenizer = cls._select_named(
            encoder,
            "tokenizer",
            lambda name: (
                name.startswith(
                    (
                        "tokenization_module.",
                        "output_norm.",
                        "tokenization_condition_projection.",
                    )
                )
                or cls._branch_parameter(name, "action_context_projections", domains)
                or cls._branch_parameter(
                    name, "tokenization_null_condition_inputs", domains
                )
                or cls._branch_parameter(name, "domain_embeddings", domains)
            ),
        )
        denoiser = cls._select_named(
            encoder,
            "denoiser",
            lambda name: (
                cls._backbone_parameter(name, "denoising_module")
                or name.startswith(("denoising_output_norm.", "condition_projection."))
                or cls._branch_parameter(name, "denoising_domain_embeddings", domains)
                or (
                    include_denoiser_null_input
                    and cls._branch_parameter(name, "null_condition_inputs", domains)
                )
            ),
        )
        if {id(parameter) for _, parameter in tokenizer} & {
            id(parameter) for _, parameter in denoiser
        }:
            raise RuntimeError("Separate UNITE telemetry parameter sets overlap")
        return tokenizer, denoiser

    @staticmethod
    def _distributed_gradient(gradient: torch.Tensor) -> torch.Tensor:
        # The copy keeps collectives read-only with respect to graph and .grad state.
        value = gradient.detach().float().clone()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)
            value.div_(torch.distributed.get_world_size())
        return value

    @classmethod
    def _gradient_norm(cls, gradients: Sequence[torch.Tensor]) -> torch.Tensor:
        values = [cls._distributed_gradient(gradient) for gradient in gradients]
        if not values:
            raise RuntimeError("UNITE telemetry received no gradients")
        norm = sum(
            (value.square().sum() for value in values), values[0].new_zeros(())
        ).sqrt()
        if not bool(torch.isfinite(norm)) or float(norm) <= 0.0:
            raise RuntimeError("UNITE telemetry gradient norm is zero or non-finite")
        return norm

    def _log_telemetry(self, name: str, value: Any) -> None:
        value = torch.as_tensor(value, device=self.device, dtype=torch.float32)
        self._finite_scalar(value, name)
        self.log(name, value, on_step=True, on_epoch=False, sync_dist=False)

    @staticmethod
    def _autograd(
        loss: torch.Tensor, named: Sequence[tuple[str, nn.Parameter]]
    ) -> tuple[tuple[nn.Parameter, ...], tuple[torch.Tensor, ...]]:
        parameters = tuple(parameter for _, parameter in named)
        gradients = torch.autograd.grad(
            loss,
            parameters,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )
        return parameters, gradients

    def _measure_shared_gradients(
        self,
        reconstruction_loss: torch.Tensor,
        flow_loss: torch.Tensor,
        named: Sequence[tuple[str, nn.Parameter]],
    ) -> None:
        parameters, reconstruction_gradients = self._autograd(
            reconstruction_loss, named
        )
        _, flow_gradients = self._autograd(flow_loss, named)
        pairs = [
            (
                self._distributed_gradient(reconstruction),
                self._distributed_gradient(flow),
            )
            for reconstruction, flow in zip(reconstruction_gradients, flow_gradients)
        ]
        if not pairs:
            raise RuntimeError("UNITE shared telemetry received no gradients")
        zero = pairs[0][0].new_zeros(())
        dot = sum((reconstruction * flow).sum() for reconstruction, flow in pairs)
        reconstruction_norm = sum(
            (reconstruction.square().sum() for reconstruction, _ in pairs), zero
        ).sqrt()
        flow_norm = sum((flow.square().sum() for _, flow in pairs), zero).sqrt()
        values = torch.stack((dot, reconstruction_norm, flow_norm))
        if not bool(torch.isfinite(values).all()):
            raise RuntimeError("Non-finite UNITE shared-gradient telemetry")
        if float(reconstruction_norm) <= 0.0 or float(flow_norm) <= 0.0:
            raise RuntimeError("UNITE shared-gradient telemetry has a zero norm")
        cosine = (dot / (reconstruction_norm * flow_norm)).clamp(-1.0, 1.0)
        for name, value in (
            ("log/unite_gradient_cosine", cosine),
            ("log/unite_recon_grad_norm", reconstruction_norm),
            ("log/unite_denoise_grad_norm", flow_norm),
            (
                "log/unite_gradient_parameter_count",
                sum(parameter.numel() for parameter in parameters),
            ),
            ("log/unite_gradient_tensor_count", len(parameters)),
        ):
            self._log_telemetry(name, value)

    def _measure_separate_gradients(
        self,
        reconstruction_loss: torch.Tensor,
        flow_loss: torch.Tensor,
        tokenizer_named: Sequence[tuple[str, nn.Parameter]],
        denoiser_named: Sequence[tuple[str, nn.Parameter]],
    ) -> None:
        _, reconstruction_gradients = self._autograd(
            reconstruction_loss, tokenizer_named
        )
        _, flow_gradients = self._autograd(flow_loss, denoiser_named)
        self._log_telemetry(
            "log/unite_tokenizer_recon_grad_norm",
            self._gradient_norm(reconstruction_gradients),
        )
        self._log_telemetry(
            "log/unite_denoiser_flow_grad_norm",
            self._gradient_norm(flow_gradients),
        )

    def _measure_topology_gradients(
        self,
        reconstruction_loss: torch.Tensor,
        flow_loss: torch.Tensor,
        predictions: Mapping,
    ) -> None:
        stage, encoder, shared = self._unite_topology()
        domains = self._active_domains(predictions, encoder)
        use_null = (
            self.training
            and float(getattr(stage, "condition_dropout_probability", 0.0)) > 0.0
        )
        if shared:
            named = self._shared_named_parameters(
                encoder, domains, include_null_input=use_null
            )
            self._measure_shared_gradients(reconstruction_loss, flow_loss, named)
        else:
            tokenizer, denoiser = self._separate_named_parameters(
                encoder, domains, include_denoiser_null_input=use_null
            )
            self._measure_separate_gradients(
                reconstruction_loss, flow_loss, tokenizer, denoiser
            )

    def _weighted_components(
        self, predictions: Mapping
    ) -> tuple[OrderedDict[str, torch.Tensor], int]:
        if not isinstance(predictions, Mapping) or not predictions:
            raise RuntimeError("UNITE received no source predictions")
        sums = OrderedDict((name, None) for name, _ in self._component_keys)
        count = 0
        for source, result in predictions.items():
            if not isinstance(result, Mapping):
                raise TypeError(f"UNITE source {source!r} result must be a mapping")
            target = result.get("target")
            if not torch.is_tensor(target) or target.ndim == 0:
                raise TypeError(f"UNITE source {source!r} target must be batched")
            source_count = int(target.shape[0])
            if source_count <= 0:
                raise RuntimeError(f"UNITE source {source!r} has no samples")
            count += source_count
            for name, key in self._component_keys:
                value = self._finite_scalar(result.get(key), f"{source!r}/{key}")
                weighted = value * source_count
                sums[name] = weighted if sums[name] is None else sums[name] + weighted
        components = OrderedDict((name, value / count) for name, value in sums.items())
        components["TotalLoss"] = (
            components["ReconstructionLoss"] + components["FlowLoss"]
        )
        components.move_to_end("TotalLoss", last=False)
        for name, value in components.items():
            self._finite_scalar(value, name)
        return components, count

    @staticmethod
    def _reduce_sums(
        sums: Mapping[str, torch.Tensor], count: int
    ) -> tuple[OrderedDict[str, torch.Tensor], int]:
        if count <= 0:
            raise RuntimeError("UNITE metric reduction has no samples")
        names = tuple(sums)
        first = sums[names[0]]
        payload = torch.stack(
            (
                *(sums[name].detach().double() for name in names),
                torch.tensor(float(count), device=first.device, dtype=torch.float64),
            )
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(payload, op=torch.distributed.ReduceOp.SUM)
        if not bool(torch.isfinite(payload).all()) or float(payload[-1]) <= 0.0:
            raise RuntimeError("Non-finite UNITE distributed metric reduction")
        means = OrderedDict(
            (name, (payload[index] / payload[-1]).to(first.dtype))
            for index, name in enumerate(names)
        )
        return means, int(payload[-1].item())

    @classmethod
    def _distributed_weighted_components(
        cls, components: Mapping[str, torch.Tensor], count: int
    ) -> tuple[OrderedDict[str, torch.Tensor], int]:
        return cls._reduce_sums(
            OrderedDict(
                (name, value.detach().double() * count)
                for name, value in components.items()
            ),
            count,
        )

    def training_step(self, batch, batch_idx):
        del batch_idx
        self.train()
        batch = self.model.process_batch_for_training(batch)
        predictions = self.model.forward_training(batch)
        components, count = self._weighted_components(predictions)
        self._unite_topology()
        logged, global_count = self._distributed_weighted_components(components, count)
        for name, value in logged.items():
            self.log(
                f"Train/UNITE/{name}",
                self._finite_scalar(value, f"Train/UNITE/{name}"),
                on_step=True,
                on_epoch=True,
                sync_dist=False,
                batch_size=global_count,
            )

        # During training_step, global_step counts updates completed before the
        # current batch. Measure on the batch whose optimizer update will reach
        # the requested cadence; otherwise max_steps=100 stops at global_step 99
        # without ever emitting cadence-100 telemetry.
        next_step = int(self.global_step) + 1
        if next_step % self.gradient_telemetry_cadence == 0:
            self._measure_topology_gradients(
                components["ReconstructionLoss"],
                components["FlowLoss"],
                predictions,
            )
        return components["TotalLoss"]

    def on_validation_start(self):
        self._unite_validation_sums = OrderedDict()
        self._unite_validation_count = 0
        if self.evaluator is not None:
            self.model.device = self.device
            self.evaluator.model = self
            self.evaluator.on_validation_start()

    @torch.no_grad()
    def _measure_validation_components(self, batch, batch_idx: int) -> None:
        devices = []
        if self.device.type == "cuda":
            devices = [self.device.index or torch.cuda.current_device()]
        seed = 420_042 + int(batch_idx) + int(self.global_rank) * 1_000_003
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed(seed)
            predictions = self.model.forward_training(batch)
        self._unite_topology()
        components, count = self._weighted_components(predictions)
        for name, value in components.items():
            weighted = value.detach().double() * count
            self._unite_validation_sums[name] = (
                self._unite_validation_sums.get(name, 0.0) + weighted
            )
        self._unite_validation_count += count

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if isinstance(batch, Mapping):
            batch = OrderedDict(
                (source, value) for source, value in batch.items() if value is not None
            )
        if not batch:
            return
        batch = self.model.process_batch_for_training(batch)
        self._measure_validation_components(batch, batch_idx)
        # Native action errors and EnergyScore belong to the bound evaluator.
        if self.evaluator is not None:
            self.evaluator.on_validation_step(batch, batch_idx, dataloader_idx)

    def on_validation_epoch_end(self):
        if self._unite_validation_count:
            metrics, _ = self._reduce_sums(
                self._unite_validation_sums, self._unite_validation_count
            )
            self.log_dict(
                OrderedDict(
                    (f"Valid/UNITE/{name}", value) for name, value in metrics.items()
                ),
                on_step=False,
                on_epoch=True,
                sync_dist=False,
            )

    def on_validation_end(self):
        if self.evaluator is not None:
            self.evaluator.on_validation_end()

    def configure_optimizers(self) -> dict[str, Any]:
        """Instantiate Muon/AdamW from stable model-local parameter names."""

        config_tree = getattr(self.hparams, "config_tree", None)
        if config_tree is None:
            raise RuntimeError("Released UNITE optimizer requires config_tree")
        cfg = self._as_config(config_tree)
        optimizer = hydra.utils.instantiate(
            cfg.model.optimizer,
            named_params=tuple(
                self.nets.named_parameters(prefix="nets", remove_duplicate=True)
            ),
        )
        if callable(optimizer):
            optimizer = optimizer()
        scheduler_cfg = cfg.model.get("scheduler")
        scheduler = (
            hydra.utils.instantiate(scheduler_cfg, optimizer=optimizer)
            if scheduler_cfg is not None
            else None
        )
        if callable(scheduler):
            scheduler = scheduler()
        if scheduler is None:
            return {"optimizer": optimizer}
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": self.hparams.scheduler_interval,
                "frequency": self.hparams.scheduler_frequency,
            },
        }
