"""TTT-Linear across-time layer (RoboTTT-style, arXiv/NVIDIA GEAR 2026).

Fast weights W parameterize a linear map f_W(x) = W x. Per chunk of H
timesteps within an episode: APPLY the current fast weights to the queries
(strictly causal at chunk granularity), then UPDATE them with one closed-form
SGD step on the key->value regression loss  L = mean ||W K_t - V_t||^2:

    grad_W = (2/H) (K W^T - V)^T K
    W     <- W - eta * grad_W

Everything is plain differentiable tensor algebra: the outer task loss
backpropagates through the scan, meta-learning W0, the projections and the
inner learning rate (no inner autograd, no optimizer objects, no hooks).

Batchflow contract: pure function of (x, cu_seqlens) — fast weights are
forward-local and reset to the learned W0 at every episode start; nothing
persists on the module between calls. The tanh gate (alpha ~ 1e-3 at init)
makes the layer ~identity at initialization (adaLN-Zero convention), so
adding it to an existing config is a true single-delta.
"""
from typing import Optional

import math

import torch
import torch.nn as nn


class TTTLinearLayer(nn.Module):
    def __init__(self, d_model: int, chunk_size: int = 8,
                 eta_init: float = 0.1, gate_init: float = 1e-3):
        super().__init__()
        d = int(d_model)
        self.chunk_size = int(chunk_size)
        self.norm = nn.LayerNorm(d)
        self.wq = nn.Linear(d, d, bias=False)
        self.wk = nn.Linear(d, d, bias=False)
        self.wv = nn.Linear(d, d, bias=False)
        for m in (self.wq, self.wk, self.wv):
            nn.init.normal_(m.weight, std=0.02)
        self.W0 = nn.Parameter(torch.zeros(d, d))          # meta-learned init
        self.log_eta = nn.Parameter(torch.tensor(math.log(float(eta_init))))
        self.alpha = nn.Parameter(torch.full((d,), float(gate_init)))

    def forward(self, x: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        """x (T, d) packed; cu_seqlens (E+1,) episode boundaries."""
        xn = self.norm(x)
        q, k, v = self.wq(xn), self.wk(xn), self.wv(xn)
        eta = self.log_eta.exp()
        out = torch.zeros_like(x)
        H = self.chunk_size
        for b in range(len(cu_seqlens) - 1):
            s, e = int(cu_seqlens[b]), int(cu_seqlens[b + 1])
            W = self.W0
            for c0 in range(s, e, H):
                c1 = min(c0 + H, e)
                Kc, Vc, Qc = k[c0:c1], v[c0:c1], q[c0:c1]
                # apply-then-update: queries see fast weights trained on
                # PREVIOUS chunks only (strictly causal at chunk granularity)
                out[c0:c1] = Qc @ W.transpose(0, 1)
                pred = Kc @ W.transpose(0, 1)
                grad = 2.0 * (pred - Vc).transpose(0, 1) @ Kc / max(c1 - c0, 1)
                W = W - eta * grad
        return x + torch.tanh(self.alpha) * out
