"""Rectified-flow (flow-matching) action head for the dual-stream H-Net.

Drop-in replacement for :class:`DualPartitionedGMMHead` behind the SAME
plumbing (``action_head_type='gmm'`` + the ``gmm_head`` kwarg): the outer
stage calls ``forward(a_top, s, emb_id)`` and stashes the result in
``batch["pred_action"]``; ``GMMLoss`` then calls ``head.nll(raw, target)``
and the eval/rollout paths call ``head.decode(raw)``. Here:

  * ``forward`` returns the CONDITIONING latent ``z = cat([a_top, s], -1)``
    (shape ``(..., d_model_a + d_model_s)``) — an opaque "raw" tensor. It is
    differentiable, so gradients flow back through both trunk streams.
  * ``nll(z, target)`` is the rectified-flow velocity-matching loss:
    ``t ~ U(0,1)`` per token, ``x_t = (1-t)*eps + t*a`` (t=1 is DATA, t=0 is
    NOISE), predict ``v = net(z, x_t, t)`` with target ``a - eps``; MSE.
    (Named ``nll`` only so the existing ``GMMLoss`` wrapper works unchanged —
    it is a plain MSE, NOT a log-likelihood; it has a nonzero floor even at
    perfect overfit, so judge overfit by sampled-action MSE, not this.)
  * ``decode(z)`` integrates the learned ODE with ``num_inference_steps``
    Euler steps from ``x_0 ~ N(0, I)`` at t=0 to t=1 and returns the action
    chunk ``(..., chunk_len, action_dim)`` clamped to [-1, 1] (actions are
    minmax-normalized to [-1, 1] upstream, mirroring the GMM tanh bound).

Velocity net: MLP (``n_layers`` x Linear+GELU at ``hidden_dim``) over
``cat([z, x_t.flatten, sincos(t)])`` -> ``chunk_len * action_dim``.

The head is SHARED across embodiments (no per-emb partitioning) — this class
exists to prove the flow objective trains through the dual-stream trunk;
``is_emb_partitioned = True`` only so the outer stage passes ``emb_id``
(accepted and ignored, keeping call sites identical to the GMM head).
"""

import math

import torch
import torch.nn as nn


def sincos_time_embedding(t: torch.Tensor, dim: int, max_freq: float = 1000.0) -> torch.Tensor:
    """``t (...,)`` in [0, 1] -> ``(..., dim)`` sinusoidal features.

    Frequencies are geometric in [1, max_freq] (t spans a single unit
    interval, so max_freq=1000 gives ~3 decades of resolution).
    """
    half = dim // 2
    freqs = torch.exp(
        torch.linspace(0.0, math.log(max_freq), half, device=t.device, dtype=t.dtype)
    )
    args = t[..., None] * freqs  # (..., half)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class FlowMatchingActionHead(nn.Module):
    """Rectified-flow action head. See module docstring for the contract."""

    # Outer stage passes emb_id when this is set (kept for call-site parity
    # with DualPartitionedGMMHead; the head itself is shared across embs).
    is_emb_partitioned = True

    def __init__(
        self,
        d_model_a: int = 256,
        d_model_s: int = 64,
        action_dim: int = 2,
        chunk_len: int = 4,
        hidden_dim: int = 1024,
        n_layers: int = 4,
        time_embed_dim: int = 64,
        num_inference_steps: int = 10,
        embodiments=None,  # accepted for yaml parity; unused (shared head)
    ):
        super().__init__()
        self.d_model_a = int(d_model_a)
        self.d_model_s = int(d_model_s)
        self.action_dim = int(action_dim)
        self.chunk_len = int(chunk_len)
        self.time_embed_dim = int(time_embed_dim)
        self.num_inference_steps = int(num_inference_steps)
        self.d_cond = self.d_model_a + self.d_model_s

        in_dim = self.d_cond + self.chunk_len * self.action_dim + self.time_embed_dim
        layers, inn = [], in_dim
        for _ in range(int(n_layers)):
            layers += [nn.Linear(inn, int(hidden_dim)), nn.GELU()]
            inn = int(hidden_dim)
        layers += [nn.Linear(inn, self.chunk_len * self.action_dim)]
        self.net = nn.Sequential(*layers)
        # Zero-init the output projection: v(x,t) == 0 at init -> the initial
        # loss equals E||a - eps||^2 (a clean, arch-independent reference) and
        # early ODE integration is a no-op instead of a random drift.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    # ------------------------------------------------------------------
    # forward: (a_top, s[, emb_id]) -> conditioning latent z (the "raw").
    # ------------------------------------------------------------------
    def forward(self, a_top: torch.Tensor, s: torch.Tensor, emb_id=None) -> torch.Tensor:
        return torch.cat([a_top, s], dim=-1)  # (..., d_cond), differentiable

    # ------------------------------------------------------------------
    # velocity net
    # ------------------------------------------------------------------
    def _velocity(self, z: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """``z (..., d_cond)``, ``x_t (..., C, D)``, ``t (...,)`` -> v ``(..., C, D)``."""
        lead = z.shape[:-1]
        temb = sincos_time_embedding(t.to(z.dtype), self.time_embed_dim)
        x_flat = x_t.reshape(*lead, self.chunk_len * self.action_dim).to(z.dtype)
        v = self.net(torch.cat([z, x_flat, temb], dim=-1))
        return v.reshape(*lead, self.chunk_len, self.action_dim)

    # ------------------------------------------------------------------
    # loss (named nll so the existing GMMLoss wrapper works unchanged)
    # ------------------------------------------------------------------
    def nll(self, raw: torch.Tensor, target: torch.Tensor, mask=None) -> torch.Tensor:
        """Rectified-flow velocity MSE. ``raw`` = z ``(..., d_cond)``;
        ``target`` = GT action chunk ``(..., C, D)`` (chunk_targets)."""
        if self.chunk_len == 1 and target.dim() == raw.dim():
            target = target.unsqueeze(-2)  # (..., D) -> (..., 1, D)
        lead = raw.shape[:-1]
        a = target.float()
        eps = torch.randn_like(a)
        t = torch.rand(lead, device=raw.device, dtype=a.dtype)
        tb = t[..., None, None]
        x_t = (1.0 - tb) * eps + tb * a  # t=0 noise, t=1 data
        v_target = a - eps
        v = self._velocity(raw, x_t, t).float()
        err = ((v - v_target) ** 2).mean(dim=(-1, -2))  # (...) per token
        if mask is None:
            return err.mean()
        mask = mask.to(err.dtype)
        return (err * mask).sum() / (mask.sum() + 1e-8)

    # ------------------------------------------------------------------
    # decode: Euler-integrate the ODE, return the sampled action chunk.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def decode(self, raw: torch.Tensor) -> torch.Tensor:
        """``raw`` = z ``(..., d_cond)`` -> actions ``(..., C, D)`` in [-1, 1]."""
        lead = raw.shape[:-1]
        k = self.num_inference_steps
        x = torch.randn(
            *lead, self.chunk_len, self.action_dim, device=raw.device, dtype=torch.float32
        )
        dt = 1.0 / k
        for i in range(k):
            t = torch.full(lead, i * dt, device=raw.device, dtype=torch.float32)
            v = self._velocity(raw, x, t).float()
            x = x + dt * v
        return x.clamp(-1.0, 1.0)
