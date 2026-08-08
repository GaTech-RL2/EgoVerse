"""
Noise / beta schedules for DFoT. Ports the cosine + linear schedules from
the upstream repo, plus zero-terminal-SNR enforcement. Continuous-time
``CosineNoiseSchedule`` returns a log-SNR for a t in [0, 1].
"""

import math
from typing import Literal

import torch
from torch import nn


def make_beta_schedule(
    schedule: Literal["cosine", "linear", "alphas_cumprod_linear"],
    timesteps: int,
    zero_terminal_snr: bool = True,
    clip_min: float = 1e-9,
    **kwargs,
) -> torch.Tensor:
    """Build a discrete beta schedule of length ``timesteps``.

    Returns betas (length ``timesteps``) suitable for the discrete DDPM
    derivation: ``alphas_cumprod = (1 - betas).cumprod(0)``.
    """
    if schedule == "cosine":
        alphas_cumprod = cosine_schedule(timesteps=timesteps, **kwargs)
    elif schedule == "linear":
        alphas_cumprod = beta_linear_schedule(timesteps=timesteps, **kwargs)
    elif schedule == "alphas_cumprod_linear":
        alphas_cumprod = alphas_cumprod_linear_schedule(timesteps=timesteps)
    else:
        raise ValueError(f"Unknown beta schedule {schedule!r}")
    if schedule != "cosine" and zero_terminal_snr:
        alphas_cumprod = enforce_zero_terminal_snr(alphas_cumprod)
    alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
    alphas = torch.cat([alphas_cumprod[0:1], alphas])
    betas = 1 - alphas
    return torch.clip(betas, clip_min, 1.0)


def cosine_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine schedule from Nichol & Dhariwal 2021. Built-in zero terminal SNR."""
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    return alphas_cumprod[1:]


def alphas_cumprod_linear_schedule(timesteps: int) -> torch.Tensor:
    """Linear-in-alphas_cumprod schedule (Lin 2023)."""
    t = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float64) / timesteps
    return (1 - t)[1:]


def beta_linear_schedule(
    timesteps: int, start: float = 0.0001, end: float = 0.02
) -> torch.Tensor:
    """Original DDPM linear-in-betas schedule."""
    betas = torch.linspace(start, end, timesteps, dtype=torch.float64)
    return (1 - betas).cumprod(dim=0)


def enforce_zero_terminal_snr(alphas_cumprod: torch.Tensor) -> torch.Tensor:
    """Lin et al. 2023 — shift/scale alphas_cumprod so the terminal SNR is 0."""
    alphas_cumprod_sqrt = torch.sqrt(alphas_cumprod)
    alphas_cumprod_sqrt_0 = alphas_cumprod_sqrt[0].clone()
    alphas_cumprod_sqrt_T = alphas_cumprod_sqrt[-1].clone()
    alphas_cumprod_sqrt -= alphas_cumprod_sqrt_T
    alphas_cumprod_sqrt *= alphas_cumprod_sqrt_0 / alphas_cumprod_sqrt[0]
    return alphas_cumprod_sqrt ** 2


class CosineNoiseSchedule(nn.Module):
    """Continuous-time cosine noise schedule (Simple Diffusion, Kingma 2023).

    ``forward(t)`` returns log-SNR for ``t`` in ``[0, 1]``. ``shift`` adjusts
    the schedule by ``2 * log(shift)`` in log-SNR space.
    """

    def __init__(
        self,
        logsnr_min: float = -15.0,
        logsnr_max: float = 15.0,
        shift: float = 1.0,
    ):
        super().__init__()
        self.register_buffer(
            "t_min",
            torch.atan(torch.exp(-0.5 * torch.tensor(logsnr_max, dtype=torch.float32))),
            persistent=False,
        )
        self.register_buffer(
            "t_max",
            torch.atan(torch.exp(-0.5 * torch.tensor(logsnr_min, dtype=torch.float32))),
            persistent=False,
        )
        self.register_buffer(
            "shift",
            2 * torch.log(torch.tensor(shift, dtype=torch.float32)),
            persistent=False,
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return (
            -2 * torch.log(torch.tan(self.t_min + t * (self.t_max - self.t_min)))
            + self.shift
        )

    @property
    def max_logsnr(self) -> torch.Tensor:
        return self.forward(torch.tensor(0.0, device=self.shift.device))

    @property
    def min_logsnr(self) -> torch.Tensor:
        return self.forward(torch.tensor(1.0, device=self.shift.device))
