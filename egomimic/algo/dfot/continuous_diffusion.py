"""
Continuous-time diffusion (v-prediction, sigmoid loss weighting) for
action-chunk denoising. Per-token noise levels are sampled in ``[0, 1]``
and converted to log-SNR via the schedule. The model is fed
``precond_scale * logSNR`` as its time conditioning (rather than a
discrete index).
"""

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from .noise_schedule import CosineNoiseSchedule


class ContinuousDiffusion(nn.Module):
    """Per-token continuous-time DDPM (v-pred + sigmoid weighting only).

    The backbone signature is ``backbone(x_t, time_cond, external_cond)``
    where ``time_cond`` is the per-token ``precond_scale * logSNR``
    (shape ``(B, T)``, dtype float).
    """

    def __init__(
        self,
        action_dim: int,
        precond_scale: float = 0.25,
        sigmoid_bias: float = -1.0,
        clip_noise: float = 20.0,
        logsnr_min: float = -15.0,
        logsnr_max: float = 15.0,
        shift: float = 1.0,
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.precond_scale = float(precond_scale)
        self.sigmoid_bias = float(sigmoid_bias)
        self.clip_noise = float(clip_noise)
        self.schedule = CosineNoiseSchedule(
            logsnr_min=logsnr_min, logsnr_max=logsnr_max, shift=shift
        )

    @property
    def max_logsnr(self) -> torch.Tensor:
        return self.schedule.max_logsnr

    @property
    def min_logsnr(self) -> torch.Tensor:
        return self.schedule.min_logsnr

    def _broadcast(self, scalar_per_token: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Reshape ``(B, T)`` to broadcast against ``(B, T, action_dim)``."""
        return scalar_per_token.reshape(
            *scalar_per_token.shape, *([1] * (x.dim() - scalar_per_token.dim()))
        )

    def q_sample(self, x: torch.Tensor, t: torch.Tensor) -> dict:
        """Forward-noising step. Adds per-token noise to ``x`` at time
        ``t``; returns all state needed both to call the backbone (``x_t``,
        ``time_cond``) and to compute the SNR-weighted MSE loss later
        (``alpha_t``, ``sigma_t``, ``noise``, ``logsnr``).

        Split out from ``forward`` so the new ``OuterStage`` (encode-time)
        and ``Loss`` (loss-time) can call them independently. ``forward``
        below stays as a back-compat wrapper.
        """
        logsnr = self.schedule(t)
        noise = torch.randn_like(x).clamp_(-self.clip_noise, self.clip_noise)
        alpha_t = self._broadcast(torch.sigmoid(logsnr).sqrt(), x)
        sigma_t = self._broadcast(torch.sigmoid(-logsnr).sqrt(), x)
        x_t = alpha_t * x + sigma_t * noise
        return {
            "x_t": x_t,
            "noise": noise,
            "alpha_t": alpha_t,
            "sigma_t": sigma_t,
            "logsnr": logsnr,
            "time_cond": self.precond_scale * logsnr,
        }

    def compute_loss(self, v_pred: torch.Tensor, q_state: dict) -> torch.Tensor:
        """Per-token SNR-weighted epsilon-MSE between predicted noise
        (derived from ``v_pred``) and the noise stored in ``q_state``.
        Caller is responsible for reducing (``.mean()``).
        """
        x_t = q_state["x_t"]
        alpha_t = q_state["alpha_t"]
        sigma_t = q_state["sigma_t"]
        noise = q_state["noise"]
        logsnr = q_state["logsnr"]
        noise_pred = alpha_t * v_pred + sigma_t * x_t
        loss = F.mse_loss(noise_pred, noise.detach(), reduction="none")
        loss_weight = self._broadcast(
            torch.sigmoid(self.sigmoid_bias - logsnr), v_pred
        )
        return loss * loss_weight

    def forward(
        self,
        backbone,
        x: torch.Tensor,
        t: torch.Tensor,
        external_cond: Optional[torch.Tensor] = None,
    ):
        """Train-time forward — back-compat wrapper around
        ``q_sample`` + backbone call + ``compute_loss``.

        Args:
            backbone: ``(x_t, time_cond, external_cond) -> v_pred``.
            x: clean ``(B, T, action_dim)``.
            t: per-token diffusion time in ``[0, 1]``, shape ``(B, T)``.
            external_cond: ``(B, T, cond_dim)`` or ``(B, cond_dim)``.
        """
        q = self.q_sample(x, t)
        v_pred = backbone(q["x_t"], q["time_cond"], external_cond)
        loss = self.compute_loss(v_pred, q)
        x_pred = q["alpha_t"] * q["x_t"] - q["sigma_t"] * v_pred
        return x_pred, loss

    def model_v(
        self,
        backbone,
        x_t: torch.Tensor,
        t: torch.Tensor,
        external_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sampling helper: evaluate v at the given (x_t, t)."""
        logsnr = self.schedule(t)
        return backbone(x_t, self.precond_scale * logsnr, external_cond)

    def v_to_x0_and_noise(self, x_t, t, v):
        logsnr = self.schedule(t)
        alpha_t = self._broadcast(torch.sigmoid(logsnr).sqrt(), x_t)
        sigma_t = self._broadcast(torch.sigmoid(-logsnr).sqrt(), x_t)
        x0 = alpha_t * x_t - sigma_t * v
        eps = alpha_t * v + sigma_t * x_t
        return x0, eps
