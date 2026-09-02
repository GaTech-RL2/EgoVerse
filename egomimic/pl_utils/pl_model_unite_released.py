"""Lightning wrapper for the released-mechanism UNITE register sweep."""

from __future__ import annotations

import random
import time
from collections import OrderedDict
from typing import Any, Dict

import hydra
import torch

import egomimic.utils.tensor_utils as TensorUtils
from egomimic.pl_utils.pl_model import ModelWrapper


class ReleasedUniteModelWrapper(ModelWrapper):
    """Keep the joint UNITE loss while exposing topology-aware telemetry."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.unite_flow_updates_per_reconstruction != 0:
            raise ValueError(
                "Released UNITE register rows require joint reconstruction+flow "
                "updates (unite_flow_updates_per_reconstruction=0)"
            )
        config_tree = getattr(self.hparams, "config_tree", None)
        cfg = self._as_config(config_tree) if config_tree is not None else None
        configured_sharing = (
            None
            if cfg is None
            else cfg.model.get("share_encoder_denoiser", None)
        )
        self.share_encoder_denoiser = (
            None if configured_sharing is None else bool(configured_sharing)
        )

    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        checkpoint.setdefault("hyper_parameters", {})[
            "share_encoder_denoiser"
        ] = self.share_encoder_denoiser

    def _unite_latent_policy(self):
        matches = [
            stage
            for stage in self.model.policy.stages
            if hasattr(stage, "shared_reconstruction_denoising_named_parameters")
        ]
        if len(matches) != 1:
            raise RuntimeError(
                "UNITE gradient telemetry requires exactly one latent policy; "
                f"found {len(matches)}"
            )
        return matches[0]

    def _unite_separate_parameters(self):
        policy = self._unite_latent_policy()
        encoder = getattr(policy, "generative_encoder", None)
        method = getattr(
            encoder,
            "separate_reconstruction_denoising_named_parameters",
            None,
        )
        if method is None:
            raise RuntimeError("Shared UNITE has no separate parameter sets")
        return method(self.model.domains)

    @staticmethod
    def _distributed_norm(gradients) -> torch.Tensor:
        square = None
        for gradient in gradients:
            value = gradient.float()
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)
                value.div_(torch.distributed.get_world_size())
            term = value.square().sum()
            square = term if square is None else square + term
        if square is None:
            raise RuntimeError("UNITE telemetry received no gradients")
        norm = square.sqrt()
        if not bool(torch.isfinite(norm)) or float(norm) <= 0.0:
            raise RuntimeError("UNITE telemetry gradient norm is zero/non-finite")
        return norm

    def _measure_unite_separate_gradients(self, reconstruction_loss, flow_loss):
        tokenizer_named, denoiser_named = self._unite_separate_parameters()
        tokenizer_parameters = tuple(parameter for _, parameter in tokenizer_named)
        denoiser_parameters = tuple(parameter for _, parameter in denoiser_named)
        if not tokenizer_parameters or not denoiser_parameters:
            raise RuntimeError("Separate UNITE telemetry parameter set is empty")
        if {id(parameter) for parameter in tokenizer_parameters} & {
            id(parameter) for parameter in denoiser_parameters
        }:
            raise RuntimeError("Separate UNITE telemetry parameter sets overlap")
        tokenizer_gradients = torch.autograd.grad(
            reconstruction_loss,
            tokenizer_parameters,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )
        denoiser_gradients = torch.autograd.grad(
            flow_loss,
            denoiser_parameters,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )
        tokenizer_norm = self._distributed_norm(tokenizer_gradients)
        denoiser_norm = self._distributed_norm(denoiser_gradients)
        self._log_train_metric(
            "log/unite_tokenizer_recon_grad_norm",
            tokenizer_norm.detach(),
            sync_dist=False,
        )
        self._log_train_metric(
            "log/unite_denoiser_flow_grad_norm",
            denoiser_norm.detach(),
            sync_dist=False,
        )

    def _measure_topology_gradients(self, reconstruction_loss, flow_loss):
        policy = self._unite_latent_policy()
        encoder = getattr(policy, "generative_encoder", None)
        runtime_shared = not hasattr(
            encoder, "separate_reconstruction_denoising_named_parameters"
        )
        if (
            self.share_encoder_denoiser is not None
            and self.share_encoder_denoiser is not runtime_shared
        ):
            raise RuntimeError(
                "Resolved UNITE sharing flag disagrees with the instantiated topology"
            )
        if runtime_shared:
            self._measure_unite_shared_gradients(reconstruction_loss, flow_loss)
        else:
            self._measure_unite_separate_gradients(reconstruction_loss, flow_loss)

    def training_step(self, batch, batch_idx):
        """Use the same joint graph for optimization and diagnostic gradients."""

        self.train()
        t0 = time.time()
        batch = self.model.process_batch_for_training(batch)
        t1 = time.time()
        predictions = self.model.forward_training(batch)
        t2 = time.time()
        source_losses = self.model.compute_losses(predictions, batch)
        t3 = time.time()
        self._log_train_metric("Timing/Process_Batch_Sec", t1 - t0)
        self._log_train_metric("Timing/Forward_Pass_Sec", t2 - t1)
        self._log_train_metric("Timing/Compute_Losses_Sec", t3 - t2)

        losses = OrderedDict(
            (key, value.mean() if torch.is_tensor(value) else value)
            for key, value in source_losses.items()
        )
        reconstruction_loss = self._mean_loss_terms(
            losses, "_loss_unite_reconstruction"
        )
        flow_loss = self._mean_loss_terms(losses, "_loss_unite_latent")
        # This is the optimized loss. The diagnostic autograd calls below are
        # read-only with respect to parameter.grad and cannot replace either term.
        losses["action_loss"] = reconstruction_loss + flow_loss
        reconstruction_l1 = self._mean_loss_terms(
            losses, "_log_unite_reconstruction_l1"
        )
        self._log_train_metric("Train/UNITE/TotalLoss", losses["action_loss"])
        self._log_train_metric(
            "Train/UNITE/ReconstructionLoss", reconstruction_loss
        )
        self._log_train_metric("Train/UNITE/FlowLoss", flow_loss)
        self._log_train_metric("Train/UNITE/ReconstructionL1", reconstruction_l1)

        cadence = self.unite_gradient_telemetry_every_n_steps
        # AdaLN-Zero starts with a degenerate flow prediction. Measure after at
        # least one real optimizer update, on the same graph as the next joint loss.
        if cadence > 0 and (int(self.global_step) + 1) % cadence == 0:
            self._measure_topology_gradients(reconstruction_loss, flow_loss)

        if (
            self.debug_loss_spike
            and random.random() < self.debug_loss_spike_prob
            and self.global_step > 100
        ):
            losses["action_loss"] = (
                losses["action_loss"] * self.debug_loss_spike_factor
            )

        info = {"losses": TensorUtils.detach(losses)}
        for key, value in self.model.log_info(info).items():
            self._log_train_metric("Train/" + key, value)
        return losses["action_loss"]

    def on_train_batch_end(self, outputs, batch, batch_idx):
        optimizer = self.optimizers(use_pl_optimizer=False)
        if not hasattr(optimizer, "adamw") or not hasattr(optimizer, "muon"):
            raise RuntimeError("Released UNITE optimizer groups are unavailable")
        self._log_train_metric(
            "Optimizer/LR/AdamW", optimizer.adamw.param_groups[0]["lr"]
        )
        self._log_train_metric(
            "Optimizer/LR/Muon", optimizer.muon.param_groups[0]["lr"]
        )
        return super().on_train_batch_end(outputs, batch, batch_idx)

    def on_validation_start(self):
        self._unite_validation_sums = {}
        return super().on_validation_start()

    def _accumulate_unite_validation(self, name, value, weight):
        weighted = value.detach().double() * int(weight)
        if name not in self._unite_validation_sums:
            self._unite_validation_sums[name] = [weighted, int(weight)]
        else:
            self._unite_validation_sums[name][0] += weighted
            self._unite_validation_sums[name][1] += int(weight)

    @torch.no_grad()
    def _measure_unite_validation_components(self, batch, batch_idx):
        devices = []
        if self.device.type == "cuda":
            devices = [
                self.device.index
                if self.device.index is not None
                else torch.cuda.current_device()
            ]
        rank = int(getattr(self, "global_rank", 0))
        seed = 420_042 + int(batch_idx) + rank * 1_000_003
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed(seed)
            for emb_id, loader_batch in batch.items():
                result = self.model.policy(self.model._seed(emb_id, loader_batch))
                reconstruction = result["loss/unite_reconstruction"]
                flow = result["loss/unite_latent"]
                normalized_prediction = result["unite/reconstructed_action"]
                normalized_target = result["target"]
                normalized_l1 = (
                    normalized_prediction - normalized_target
                ).abs().mean()
                action_key = self.model.resolved_ac_keys[emb_id]
                native_prediction = self.model.norm_stats.unnormalize(
                    {action_key: normalized_prediction}, emb_id
                )[action_key]
                native_target = self.model.norm_stats.unnormalize(
                    {action_key: normalized_target}, emb_id
                )[action_key]
                if int(native_prediction.shape[-1]) != 4:
                    raise RuntimeError(
                        "U-Socket UNITE validation requires x_y_cos_theta_sin_theta"
                    )
                translation_error = native_prediction[..., :2] - native_target[..., :2]
                predicted_angle = torch.atan2(
                    native_prediction[..., 3], native_prediction[..., 2]
                )
                target_angle = torch.atan2(
                    native_target[..., 3], native_target[..., 2]
                )
                angle_error = torch.atan2(
                    torch.sin(predicted_angle - target_angle),
                    torch.cos(predicted_angle - target_angle),
                )
                native_mse = 0.5 * (
                    translation_error.square().mean() + angle_error.square().mean()
                )
                native_l1 = 0.5 * (
                    translation_error.abs().mean() + angle_error.abs().mean()
                )
                batch_size = int(normalized_target.shape[0])
                domain = self.model.domain_by_id[emb_id]
                values = {
                    "TotalLoss": reconstruction + flow,
                    "ReconstructionLoss": reconstruction,
                    "FlowLoss": flow,
                    "ReconstructionL1": normalized_l1,
                    "ReconstructionNativeMSE": native_mse,
                    "ReconstructionNativeL1": native_l1,
                }
                for suffix, value in values.items():
                    self._accumulate_unite_validation(suffix, value, batch_size)
                    self._accumulate_unite_validation(
                        f"{suffix}/{domain}", value, batch_size
                    )

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if self.evaluator is None:
            return
        if isinstance(batch, dict):
            batch = {
                name: value for name, value in batch.items() if value is not None
            }
        if not batch:
            return
        batch = self.model.process_batch_for_training(batch)
        self._measure_unite_validation_components(batch, batch_idx)
        self.evaluator.on_validation_step(batch, batch_idx, dataloader_idx)

    def on_validation_end(self):
        metrics = {}
        for suffix, (total, count) in self._unite_validation_sums.items():
            pair = torch.stack(
                (
                    total.to(self.device),
                    torch.tensor(
                        float(count), device=self.device, dtype=torch.float64
                    ),
                )
            )
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(pair, op=torch.distributed.ReduceOp.SUM)
            value = (pair[0] / pair[1]).float()
            if not bool(torch.isfinite(value)):
                raise RuntimeError(f"Non-finite UNITE validation metric {suffix}")
            metrics[f"Valid/UNITE/{suffix}"] = value
        if metrics:
            self.log_dict(metrics, on_step=False, on_epoch=True, sync_dist=False)
        return super().on_validation_end()

    def configure_optimizers(self) -> Dict[str, Any]:
        """Instantiate Muon/AdamW from stable model-local parameter names."""

        # PipelineAlgo is intentionally not an nn.Module. ModelWrapper registers
        # its policy ModuleDict as ``self.nets`` so Lightning owns the exact
        # trainable graph. Keep the historical ``nets.policy...`` names used by
        # the released optimizer contract while removing shared aliases once.
        named_parameters = tuple(
            self.nets.named_parameters(prefix="nets", remove_duplicate=True)
        )
        config_tree = getattr(self.hparams, "config_tree", None)
        if config_tree is not None:
            cfg = self._as_config(config_tree)
            optimizer = hydra.utils.instantiate(
                cfg.model.optimizer,
                named_params=named_parameters,
            )
            if callable(optimizer):
                optimizer = optimizer()
            scheduler_cfg = cfg.model.get("scheduler")
            if scheduler_cfg is not None:
                scheduler = hydra.utils.instantiate(
                    scheduler_cfg,
                    optimizer=optimizer,
                )
                if callable(scheduler):
                    scheduler = scheduler()
            else:
                scheduler = None
        else:
            optimizer = self.hparams.optimizer(named_params=named_parameters)
            scheduler = (
                self.hparams.scheduler(optimizer=optimizer)
                if self.hparams.scheduler is not None
                else None
            )

        if scheduler is not None:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": self.hparams.scheduler_interval,
                    "frequency": self.hparams.scheduler_frequency,
                },
            }
        return {"optimizer": optimizer}
