"""Sparse feed-forward experts used by Pipeline denoisers."""

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


class SwiGLU(nn.Module):
    """A compact SwiGLU MLP with an explicit per-expert inner width."""

    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()
        self.gate_value = nn.Linear(d_model, 2 * d_hidden)
        self.output = nn.Linear(d_hidden, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = self.gate_value(x).chunk(2, dim=-1)
        return self.output(F.silu(gate) * value)


class MoEFFN(nn.Module):
    """Content-routed top-k experts with batched expert execution."""

    def __init__(
        self,
        d_model: int,
        d_intermediate: int,
        num_experts: int = 8,
        top_k: int = 4,
        aux_weight: float = 1.0e-3,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.d_expert = int(d_intermediate)
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.aux_weight = float(aux_weight)
        if self.num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if not 0 < self.top_k <= self.num_experts:
            raise ValueError("top_k must be in [1, num_experts]")
        if self.d_expert <= 0:
            raise ValueError("d_intermediate must be positive")

        self.gate = nn.Linear(self.d_model, self.num_experts, bias=False)
        self.experts = nn.ModuleList(
            SwiGLU(self.d_model, self.d_expert) for _ in range(self.num_experts)
        )
        self.last_aux_loss: Optional[torch.Tensor] = None
        self.last_expert_frac: Optional[torch.Tensor] = None
        self.last_gate_entropy: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        tokens = x.reshape(-1, original_shape[-1])
        probabilities = F.softmax(self.gate(tokens).float(), dim=-1)
        top_values, top_indices = torch.topk(probabilities, self.top_k, dim=-1)
        top_weights = top_values / top_values.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        routed_weights = torch.zeros_like(probabilities)
        routed_weights.scatter_(1, top_indices, top_weights)

        # Evaluate the experts as two batched matrix products. Computing every
        # expert and gathering the routed top-k outputs is intentionally dense:
        # it avoids a Python expert loop, per-expert CUDA synchronizations, and
        # DDP-only parameter anchors while preserving the routed result and the
        # checkpoint's ModuleList parameter names.
        gate_value_weight = torch.stack(
            [expert.gate_value.weight for expert in self.experts]
        )
        gate_value_bias = torch.stack(
            [expert.gate_value.bias for expert in self.experts]
        )
        gate_value = torch.einsum("nd,ehd->neh", tokens, gate_value_weight)
        gate_value = gate_value + gate_value_bias.unsqueeze(0)
        gate, value = gate_value.chunk(2, dim=-1)
        hidden = F.silu(gate) * value

        output_weight = torch.stack([expert.output.weight for expert in self.experts])
        output_bias = torch.stack([expert.output.bias for expert in self.experts])
        expert_outputs = torch.einsum("neh,edh->ned", hidden, output_weight)
        expert_outputs = expert_outputs + output_bias.unsqueeze(0)
        selected_outputs = expert_outputs.gather(
            1,
            top_indices.unsqueeze(-1).expand(-1, -1, self.d_model),
        )
        output = torch.sum(
            selected_outputs * top_weights.to(selected_outputs).unsqueeze(-1),
            dim=1,
        ).to(tokens)

        dispatch = (routed_weights > 0).to(probabilities.dtype)
        denominator = max(tokens.shape[0] * self.top_k, 1)
        fractions = dispatch.sum(dim=0) / denominator
        mean_probability = probabilities.mean(dim=0)
        self.last_aux_loss = (
            self.aux_weight
            * self.num_experts
            * torch.sum(fractions.detach() * mean_probability)
        )
        self.last_expert_frac = fractions.detach()
        detached_probability = mean_probability.detach()
        self.last_gate_entropy = -torch.sum(
            detached_probability * detached_probability.clamp_min(1e-9).log()
        )
        return output.reshape(original_shape)
