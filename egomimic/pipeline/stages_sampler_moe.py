"""MoE-aware extension of the Pipeline multi-J action sampler."""

import torch

from egomimic.models.moe_ffn import MoEFFN
from egomimic.pipeline.stages_sampler import MultiJActionSampler


class MoEMultiJActionSampler(MultiJActionSampler):
    """Collect router loss and diagnostics without changing sampler behavior."""

    writes = ["pred_action", "loss/moe_lb", "log/*"]

    def _velocity(self, latent, time, condition):
        velocity = super()._velocity(latent, time, condition)
        modules = [
            module
            for module in self.denoising_module.modules()
            if isinstance(module, MoEFFN) and module.last_aux_loss is not None
        ]
        if modules:
            self._moe_aux_calls.append(
                torch.stack([module.last_aux_loss for module in modules]).mean()
            )
            self._moe_frac_calls.append(
                torch.stack([module.last_expert_frac for module in modules]).mean(0)
            )
            self._moe_entropy_calls.append(
                torch.stack([module.last_gate_entropy for module in modules]).mean()
            )
        return velocity

    def forward(self, batch):
        self._moe_aux_calls = []
        self._moe_frac_calls = []
        self._moe_entropy_calls = []
        batch = super().forward(batch)
        if self._moe_aux_calls:
            auxiliary_loss = torch.stack(self._moe_aux_calls).mean()
            batch["loss/moe_lb"] = auxiliary_loss
            batch["log/moe_lb"] = auxiliary_loss.detach()
            fractions = torch.stack(self._moe_frac_calls).mean(0)
            for expert, fraction in enumerate(fractions):
                batch[f"log/moe_expert_frac_e{expert}"] = fraction.detach()
            batch["log/moe_gate_entropy"] = (
                torch.stack(self._moe_entropy_calls).mean().detach()
            )
        return batch
