"""
Mixture-of-Experts FFN — drop-in replacement for the dense ``SwiGLU`` MLP in a
:class:`~egomimic.models.hnet.blocks.TransformerBlock`.

Design (single-stream H-Net MoE ablation, ``hnet_dualstream_sym_w1_moe``):

* ``num_experts`` independent SwiGLU experts, each with inner width
  ``d_intermediate`` (the PER-EXPERT width). Routing picks ``top_k`` experts
  per token and combines their outputs with renormalised softmax gate weights.
* The gate is a single ``Linear(d_model -> num_experts)`` on TOKEN CONTENT ONLY
  (no embodiment id) — the whole point of the single-stream ablation is that
  specialisation, if any, is learned by the router rather than hard-wired
  per-embodiment.
* Attention is untouched (stays dense/shared in the block); only the FFN is
  swapped.

Sizing / active-param matching:
    A dense ``SwiGLU(d_model, base)`` has ``3 * d_model * base`` params. Setting
    the per-expert width ``d_intermediate = base / top_k`` makes the ``top_k``
    ACTIVE experts sum to ``3 * d_model * base`` — i.e. the per-token active FFN
    equals the dense block it replaces. TOTAL FFN params grow by
    ``num_experts / top_k`` (the extra, non-activated capacity).

Dispatch:
    True sparse dispatch — each expert runs only on the tokens routed to it, so
    the FFN FLOPs per token equal ``top_k`` experts (== the dense FFN), NOT
    ``num_experts``. Works on packed ``(T_total, D)`` and padded ``(B, L, D)``
    inputs (flattened to ``(N, D)`` token rows). Single-GPU here (no DDP), so an
    expert receiving zero tokens in a step is harmless.

Load-balance aux loss (Switch-Transformer style, normalised so ~1.0 at perfect
balance and ~num_experts at full collapse):

    f_e = (#assignments to expert e) / (N * top_k)        # sums to 1 over e
    P_e = mean_token softmax_gate_prob(e)                 # sums to 1 over e
    L_aux = aux_weight * num_experts * sum_e (f_e * P_e)

The gradient flows through ``P_e`` (soft); ``f_e`` uses the hard top-k counts
(non-differentiable, treated as constant) — standard Switch formulation.

The aux loss + per-expert routing fractions + gate entropy are stashed on the
module each forward (``last_aux_loss`` / ``last_expert_frac`` / ``last_gate_entropy``)
and collected by ``ComputeStage.forward`` into ``ctx.extras["moe_aux"]`` — the
same "aux registered on the context, summed by the Loss" mechanism the chunker
ratio loss uses.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from egomimic.models.hnet.blocks import SwiGLU


class MoEFFN(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_intermediate: int,
        num_experts: int = 8,
        top_k: int = 3,
        aux_weight: float = 0.01,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.d_expert = int(d_intermediate)  # PER-EXPERT SwiGLU inner width
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.aux_weight = float(aux_weight)
        if self.top_k > self.num_experts:
            raise ValueError(
                f"top_k ({self.top_k}) must be <= num_experts ({self.num_experts})"
            )
        # Content-only gate. bias=False + standard 0.02 init => near-uniform
        # logits at init (tiny per-token noise breaks the top-k tie), so routing
        # starts balanced rather than collapsed.
        self.gate = nn.Linear(self.d_model, self.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [SwiGLU(self.d_model, self.d_expert) for _ in range(self.num_experts)]
        )
        # Per-forward diagnostics (NOT parameters / buffers — plain attributes,
        # so they never enter the state_dict). Collected by ComputeStage.
        self.last_aux_loss: Optional[torch.Tensor] = None
        self.last_expert_frac: Optional[torch.Tensor] = None
        self.last_gate_entropy: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        d = orig_shape[-1]
        tokens = x.reshape(-1, d)  # (N, D) — one row per token (packed or padded)
        N = tokens.shape[0]

        logits = self.gate(tokens)                     # (N, E)
        probs = F.softmax(logits.float(), dim=-1)      # (N, E), fp32 for stability

        top_val, top_idx = torch.topk(probs, self.top_k, dim=-1)   # (N, k)
        # Renormalise the selected weights so the top-k combine sums to 1.
        top_w = top_val / (top_val.sum(dim=-1, keepdim=True) + 1e-9)  # (N, k)
        gate_full = torch.zeros_like(probs)                          # (N, E)
        gate_full.scatter_(1, top_idx, top_w)                        # (N, E)

        # Sparse dispatch: run each expert only on its routed tokens.
        out = torch.zeros_like(tokens)
        for e, expert in enumerate(self.experts):
            sel = gate_full[:, e] > 0                                # (N,) bool
            if not bool(sel.any()):
                continue
            idx = sel.nonzero(as_tuple=True)[0]
            y_e = expert(tokens.index_select(0, idx))                # (n_e, D)
            w_e = gate_full[idx, e].unsqueeze(-1).to(y_e.dtype)      # (n_e, 1)
            # Cast to out.dtype: under bf16 autocast the expert output is bf16
            # while out (the FFN residual-stream input dtype) may be fp32;
            # index_add_ requires matching scalar types.
            out.index_add_(0, idx, (w_e * y_e).to(out.dtype))

        # ---- load-balance aux + diagnostics ----
        dispatch = (gate_full > 0).to(probs.dtype)     # (N, E) hard top-k mask
        counts = dispatch.sum(dim=0)                   # (E,)
        f = counts / max(N * self.top_k, 1)            # (E,), sums to 1
        P = probs.mean(dim=0)                          # (E,), sums to 1, differentiable
        aux = self.aux_weight * self.num_experts * torch.sum(f.detach() * P)

        self.last_aux_loss = aux
        self.last_expert_frac = f.detach()
        self.last_gate_entropy = -(P.detach() * (P.detach() + 1e-9).log()).sum()

        return out.reshape(orig_shape)
