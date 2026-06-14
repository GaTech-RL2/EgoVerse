"""Action-conditioned JEPA (Joint-Embedding Predictive Architecture) auxiliary.

This is the *model-agnostic core* of the JEPA world-model auxiliary loss described
in the design notes. It is shared by every policy family (HPT flow, BC-RNN
Transformer, BC-RNN H-Net); each algo supplies only a thin adapter (see
``egomimic/algo/jepa_mixin.py``) that knows how to (a) hand over its observation
encoder and (b) embed an obs dict into a pooled latent.

The mechanism (Flavor B, the "world-model" JEPA):

    z_ctx   = online_encoder(o_t)                     # grad flows -> pressures the encoder
    z_tgt   = sg( EMA_encoder(o_{t+k}) )              # stop-grad target, EMA of the online encoder
    z_pred  = predictor(z_ctx, a_{t:t+k})            # action-conditioned prediction
    L_jepa  = dist(z_pred, z_tgt)                     # latent-space loss

Collapse prevention (the one thing that *must* be right): the target encoder is a
frozen EMA copy of the online encoder (no gradient), and the targets are detached
and (optionally) L2-normalized. The predictor + online encoder are the only
trained parts; the EMA encoder is updated by ``update_target`` outside autograd.

Default-OFF: nothing here runs unless a model config sets ``jepa.enabled=true``.
"""

from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class JEPAPredictor(nn.Module):
    """Predicts the future obs-latent from the current obs-latent + action chunk.

    The action chunk ``a_{t:t+k}`` (B, k, action_dim) is flattened and concatenated
    with the context latent; an MLP maps the result to a predicted target latent.
    A small MLP is deliberately the v1 choice — it is the cheapest thing that
    conditions on the action; a transformer-over-action-tokens predictor can be
    swapped in later behind the same forward signature.
    """

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        action_horizon: int,
        hidden_dim: int = 512,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        in_dim = latent_dim + action_dim * action_horizon
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(max(1, n_layers)):
            layers += [nn.Linear(d, hidden_dim), nn.GELU()]
            d = hidden_dim
        layers += [nn.Linear(d, latent_dim)]
        self.net = nn.Sequential(*layers)
        self.action_dim = action_dim
        self.action_horizon = action_horizon

    def forward(self, z_ctx: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        # z_ctx: (B, latent_dim); action_chunk: (B, k, action_dim)
        a = action_chunk.reshape(action_chunk.shape[0], -1)  # (B, k*action_dim)
        return self.net(torch.cat([z_ctx, a], dim=-1))


class JEPAModule(nn.Module):
    """Owns the trained predictor + the frozen EMA target-encoder.

    The *online* encoder is owned by the policy (so its gradients flow into the
    representation the BC head reads); this module only deep-copies it once to seed
    the EMA target and references it again at ``update_target`` time. Registering
    this module under the policy means the predictor is picked up by the optimizer
    automatically, while the EMA encoder (``requires_grad_(False)``) is skipped
    (its grads stay ``None``) and is instead updated by the EMA step.
    """

    def __init__(
        self,
        online_encoder: nn.Module,
        latent_dim: int,
        action_dim: int,
        action_horizon: int,
        hidden_dim: int = 512,
        n_layers: int = 2,
        ema_decay: float = 0.996,
        loss_type: str = "smooth_l1",
        normalize_targets: bool = True,
        var_coef: float = 1.0,
        cov_coef: float = 0.04,
    ) -> None:
        super().__init__()
        self.predictor = JEPAPredictor(
            latent_dim, action_dim, action_horizon, hidden_dim, n_layers
        )
        # EMA target = frozen deep copy of the online encoder.
        self.target_encoder = copy.deepcopy(online_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        self.target_encoder.eval()
        self.ema_decay = float(ema_decay)
        self.loss_type = str(loss_type)
        self.normalize_targets = bool(normalize_targets)
        # VICReg anti-collapse coefficients (variance + covariance on z_ctx).
        # var_coef pushes each embedding dim's std toward >=1 (kills constant /
        # directional collapse that the normalized invariance term alone allows);
        # cov_coef decorrelates dims. Set both to 0 to disable.
        self.var_coef = float(var_coef)
        self.cov_coef = float(cov_coef)

    @staticmethod
    def _variance_loss(z: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
        std = torch.sqrt(z.var(dim=0) + eps)  # (D,)
        return torch.mean(F.relu(gamma - std))

    @staticmethod
    def _covariance_loss(z: torch.Tensor) -> torch.Tensor:
        B, D = z.shape
        if B < 2:
            return z.new_zeros(())
        zc = z - z.mean(dim=0, keepdim=True)
        cov = (zc.T @ zc) / (B - 1)  # (D, D)
        off = cov - torch.diag(torch.diag(cov))
        return (off.pow(2).sum()) / D

    @torch.no_grad()
    def update_target(self, online_encoder: nn.Module) -> None:
        """EMA step: target <- decay*target + (1-decay)*online. Call once per batch."""
        d = self.ema_decay
        for pt, po in zip(self.target_encoder.parameters(), online_encoder.parameters()):
            pt.mul_(d).add_(po.detach().to(pt.dtype), alpha=1.0 - d)
        # Buffers (e.g. BatchNorm running stats) are copied, not EMA'd.
        for bt, bo in zip(self.target_encoder.buffers(), online_encoder.buffers()):
            bt.copy_(bo)

    def loss(
        self,
        z_ctx: torch.Tensor,
        z_tgt: torch.Tensor,
        action_chunk: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Latent prediction loss. ``z_ctx`` carries grad (online encoder); ``z_tgt``
        is detached (EMA encoder). ``valid_mask`` (B,) drops samples whose future
        frame is invalid (e.g. clamped past the episode end)."""
        z_pred = self.predictor(z_ctx, action_chunk)
        zt = z_tgt.detach()
        zp = z_pred
        if self.normalize_targets:
            zp = F.normalize(zp, dim=-1)
            zt = F.normalize(zt, dim=-1)

        if self.loss_type == "smooth_l1":
            per = F.smooth_l1_loss(zp, zt, reduction="none").mean(-1)
        elif self.loss_type == "mse":
            per = F.mse_loss(zp, zt, reduction="none").mean(-1)
        elif self.loss_type == "cosine":
            per = 1.0 - (F.normalize(zp, dim=-1) * F.normalize(zt, dim=-1)).sum(-1)
        else:
            raise ValueError(f"Unknown jepa loss_type={self.loss_type!r}")

        if valid_mask is not None:
            valid = valid_mask.to(per.dtype)
            inv = (per * valid).sum() / valid.sum().clamp_min(1.0)
        else:
            inv = per.mean()

        # Anti-collapse: VICReg variance + covariance on the online embedding.
        # Without these the normalized invariance term collapses to ~0 (all
        # embeddings -> one direction). The variance hinge keeps per-dim std >= 1.
        var = self._variance_loss(z_ctx) if self.var_coef > 0 else z_ctx.new_zeros(())
        cov = self._covariance_loss(z_ctx) if self.cov_coef > 0 else z_ctx.new_zeros(())
        total = inv + self.var_coef * var + self.cov_coef * cov
        # Stash components for logging/diagnosis (read by the algo if it wants).
        self.last_components = {
            "inv": inv.detach(),
            "var": var.detach(),
            "cov": cov.detach(),
            "z_std": z_ctx.detach().std(dim=0).mean(),
        }
        return total
