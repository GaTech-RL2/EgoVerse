"""
Discrete-time DDPM for action-chunk denoising (per-token noise levels).

Slimmed from the DFoT video implementation: drops video-specific shape
machinery (channels/HxW), DDIM logic is provided separately in
``sampling.py``. ``forward(x, k, external_cond)`` runs the standard q-sample
+ model-prediction + weighted-MSE loss path with per-token ``k`` of shape
``(B, T)`` and action tensor ``x`` of shape ``(B, T, action_dim)``.
"""

from collections import namedtuple
from typing import Literal, Optional

import torch
from torch import nn
from torch.nn import functional as F

from .noise_schedule import make_beta_schedule


ModelPrediction = namedtuple(
    "ModelPrediction", ["pred_noise", "pred_x_start", "model_out"]
)


def _extract(a: torch.Tensor, k: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    """Gather schedule values ``a[k]`` and reshape to broadcast over ``x``.

    ``a`` is (timesteps,); ``k`` is (B,) or (B, T) (per-token). ``x_shape``
    is the action tensor shape, e.g. ``(B, T, action_dim)``.
    """
    shape = k.shape
    out = a[k]
    return out.reshape(*shape, *((1,) * (len(x_shape) - len(shape))))


class DiscreteDiffusion(nn.Module):
    """Per-token discrete-time DDPM training+loss core.

    Args:
        action_dim: dim of the trailing feature axis on ``x``.
        timesteps: number of discrete noise levels.
        beta_schedule: 'cosine' | 'linear' | 'alphas_cumprod_linear'.
        schedule_fn_kwargs: extra kwargs to ``make_beta_schedule``.
        objective: 'pred_noise' | 'pred_x0' | 'pred_v'.
        loss_weighting_strategy: 'uniform' | 'min_snr' | 'sigmoid'.
        snr_clip: SNR clip used by 'min_snr' weighting.
        sigmoid_bias: bias used by 'sigmoid' weighting.
        clip_noise: clip ε samples to ``[-clip_noise, clip_noise]``.
    """

    def __init__(
        self,
        action_dim: int,
        timesteps: int = 1000,
        beta_schedule: str = "cosine",
        schedule_fn_kwargs: Optional[dict] = None,
        objective: Literal["pred_noise", "pred_x0", "pred_v"] = "pred_v",
        loss_weighting_strategy: Literal["uniform", "min_snr", "sigmoid"] = "min_snr",
        snr_clip: float = 5.0,
        sigmoid_bias: float = -1.0,
        clip_noise: float = 20.0,
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.timesteps = int(timesteps)
        self.objective = objective
        self.loss_weighting_strategy = loss_weighting_strategy
        self.snr_clip = float(snr_clip)
        self.sigmoid_bias = float(sigmoid_bias)
        self.clip_noise = float(clip_noise)

        betas = make_beta_schedule(
            schedule=beta_schedule,
            timesteps=self.timesteps,
            zero_terminal_snr=self.objective != "pred_noise",
            **(schedule_fn_kwargs or {}),
        )

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        def reg(name: str, val: torch.Tensor):
            self.register_buffer(name, val.to(torch.float32), persistent=False)

        reg("betas", betas)
        reg("alphas_cumprod", alphas_cumprod)
        reg("alphas_cumprod_prev", alphas_cumprod_prev)
        reg("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        reg("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        reg("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        reg("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        reg("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        reg("posterior_variance", posterior_variance)
        reg(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        reg(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        reg(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        snr = alphas_cumprod / (1 - alphas_cumprod)
        reg("snr", snr)
        if loss_weighting_strategy == "min_snr":
            clipped_snr = snr.clone().clamp_(max=self.snr_clip)
            reg("clipped_snr", clipped_snr)
        if loss_weighting_strategy == "sigmoid":
            reg("logsnr", torch.log(snr))

    # ---- closed-form predictions ---- #

    def predict_start_from_noise(self, x_k, k, noise):
        return (
            _extract(self.sqrt_recip_alphas_cumprod, k, x_k.shape) * x_k
            - _extract(self.sqrt_recipm1_alphas_cumprod, k, x_k.shape) * noise
        )

    def predict_noise_from_start(self, x_k, k, x0):
        return (
            x_k - _extract(self.sqrt_alphas_cumprod, k, x_k.shape) * x0
        ) / _extract(self.sqrt_one_minus_alphas_cumprod, k, x_k.shape)

    def predict_v(self, x_start, k, noise):
        return (
            _extract(self.sqrt_alphas_cumprod, k, x_start.shape) * noise
            - _extract(self.sqrt_one_minus_alphas_cumprod, k, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_k, k, v):
        return (
            _extract(self.sqrt_alphas_cumprod, k, x_k.shape) * x_k
            - _extract(self.sqrt_one_minus_alphas_cumprod, k, x_k.shape) * v
        )

    def predict_noise_from_v(self, x_k, k, v):
        return (
            _extract(self.sqrt_alphas_cumprod, k, x_k.shape) * v
            + _extract(self.sqrt_one_minus_alphas_cumprod, k, x_k.shape) * x_k
        )

    def q_sample(self, x_start, k, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start).clamp_(-self.clip_noise, self.clip_noise)
        return (
            _extract(self.sqrt_alphas_cumprod, k, x_start.shape) * x_start
            + _extract(self.sqrt_one_minus_alphas_cumprod, k, x_start.shape) * noise
        )

    def q_posterior(self, x_start, x_k, k):
        mean = (
            _extract(self.posterior_mean_coef1, k, x_k.shape) * x_start
            + _extract(self.posterior_mean_coef2, k, x_k.shape) * x_k
        )
        var = _extract(self.posterior_variance, k, x_k.shape)
        log_var = _extract(self.posterior_log_variance_clipped, k, x_k.shape)
        return mean, var, log_var

    # ---- model wrapper ---- #

    def model_predictions(
        self,
        backbone,
        x: torch.Tensor,
        k: torch.Tensor,
        external_cond: Optional[torch.Tensor] = None,
    ) -> ModelPrediction:
        model_out = backbone(x, k, external_cond)
        if self.objective == "pred_noise":
            pred_noise = torch.clamp(model_out, -self.clip_noise, self.clip_noise)
            x_start = self.predict_start_from_noise(x, k, pred_noise)
        elif self.objective == "pred_x0":
            x_start = model_out
            pred_noise = self.predict_noise_from_start(x, k, x_start)
        elif self.objective == "pred_v":
            v = model_out
            x_start = self.predict_start_from_v(x, k, v)
            pred_noise = self.predict_noise_from_v(x, k, v)
        else:
            raise ValueError(f"unknown objective {self.objective}")
        return ModelPrediction(pred_noise, x_start, model_out)

    def compute_loss_weights(self, k: torch.Tensor) -> torch.Tensor:
        strategy = self.loss_weighting_strategy
        if strategy == "uniform":
            return torch.ones_like(k, dtype=torch.float32)
        snr = self.snr[k]
        if strategy == "min_snr":
            clipped_snr = self.clipped_snr[k]
            epsilon_weighting = clipped_snr / snr.clamp(min=1e-8)
        elif strategy == "sigmoid":
            logsnr = self.logsnr[k]
            epsilon_weighting = torch.sigmoid(self.sigmoid_bias - logsnr)
        else:
            raise ValueError(f"unknown loss weighting strategy {strategy}")
        if self.objective == "pred_noise":
            return epsilon_weighting
        if self.objective == "pred_x0":
            return epsilon_weighting * snr
        if self.objective == "pred_v":
            return epsilon_weighting * snr / (snr + 1)
        raise ValueError(f"unknown objective {self.objective}")

    def forward(
        self,
        backbone,
        x: torch.Tensor,
        k: torch.Tensor,
        external_cond: Optional[torch.Tensor] = None,
    ):
        """Train-time forward.

        Args:
            backbone: callable ``(x_k, k, external_cond) -> model_out`` where
                ``x_k`` is (B, T, action_dim), ``k`` is (B, T).
            x: clean target ``(B, T, action_dim)``.
            k: per-token noise levels ``(B, T)`` (long, in ``[0, timesteps)``).
            external_cond: per-token obs cond ``(B, T, cond_dim)`` or
                ``(B, cond_dim)`` (broadcast inside ``backbone``).
        """
        noise = torch.randn_like(x).clamp_(-self.clip_noise, self.clip_noise)
        x_k = self.q_sample(x, k, noise=noise)
        pred = self.model_predictions(backbone, x_k, k, external_cond)

        if self.objective == "pred_noise":
            target = noise
        elif self.objective == "pred_x0":
            target = x
        elif self.objective == "pred_v":
            target = self.predict_v(x, k, noise)
        else:
            raise ValueError(f"unknown objective {self.objective}")

        loss = F.mse_loss(pred.model_out, target.detach(), reduction="none")
        w = self.compute_loss_weights(k)
        # Broadcast (B, T) weights against (B, T, action_dim) loss.
        loss = loss * w.unsqueeze(-1)
        return pred.pred_x_start, loss
