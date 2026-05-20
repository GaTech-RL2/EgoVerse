"""
DDPM + DDIM samplers for both discrete and continuous DFoT diffusion.

All samplers use per-token noise levels (the noise level tensor is broadcast
to every token in the chunk for v1 — DFoT's per-token-rolling sampler is a
future addition). External cond is forwarded to the backbone untouched.
"""

from typing import Optional, Union

import torch

from .continuous_diffusion import ContinuousDiffusion
from .discrete_diffusion import DiscreteDiffusion


@torch.no_grad()
def ddpm_sample(
    diffusion: Union[DiscreteDiffusion, ContinuousDiffusion],
    backbone,
    shape,
    external_cond: Optional[torch.Tensor] = None,
    n_steps: Optional[int] = None,
    device=None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """DDPM ancestral sampler.

    Args:
        diffusion: a ``DiscreteDiffusion`` or ``ContinuousDiffusion`` instance.
        backbone: ``DFoTBackbone`` (or compatible) — takes
            ``(x, noise_levels, external_cond)``.
        shape: target ``(B, T, action_dim)``.
        external_cond: obs conditioning, forwarded to ``backbone``.
        n_steps: number of denoising steps. For discrete diffusion defaults
            to ``diffusion.timesteps``; for continuous defaults to 50.
    """
    device = device or (
        external_cond.device if external_cond is not None else torch.device("cpu")
    )
    B, T, _ = shape
    x = torch.randn(*shape, device=device, dtype=dtype)

    if isinstance(diffusion, DiscreteDiffusion):
        steps = n_steps or diffusion.timesteps
        assert steps == diffusion.timesteps, (
            "DDPM ancestral sampling requires n_steps == diffusion.timesteps; "
            "use ddim_sample for fewer steps."
        )
        for i in reversed(range(diffusion.timesteps)):
            k = torch.full((B, T), i, device=device, dtype=torch.long)
            pred = diffusion.model_predictions(backbone, x, k, external_cond)
            mean, _, log_var = diffusion.q_posterior(pred.pred_x_start, x, k)
            noise = torch.randn_like(x) if i > 0 else torch.zeros_like(x)
            x = mean + torch.exp(0.5 * log_var) * noise
        return x

    if isinstance(diffusion, ContinuousDiffusion):
        # Discretize [0, 1] into n_steps intervals; ancestral step at each.
        steps = n_steps or 50
        ts = torch.linspace(1.0, 0.0, steps + 1, device=device)
        for i in range(steps):
            t_cur = torch.full((B, T), ts[i].item(), device=device)
            t_next = torch.full((B, T), ts[i + 1].item(), device=device)
            v = diffusion.model_v(backbone, x, t_cur, external_cond)
            x0, eps = diffusion.v_to_x0_and_noise(x, t_cur, v)
            # Convert continuous-time to alpha factors at cur/next.
            logsnr_cur = diffusion.schedule(t_cur)
            logsnr_next = diffusion.schedule(t_next)
            alpha_cur = torch.sigmoid(logsnr_cur).sqrt().reshape(B, T, 1)
            sigma_cur = torch.sigmoid(-logsnr_cur).sqrt().reshape(B, T, 1)
            alpha_next = torch.sigmoid(logsnr_next).sqrt().reshape(B, T, 1)
            sigma_next = torch.sigmoid(-logsnr_next).sqrt().reshape(B, T, 1)
            # Posterior mean (ancestral). Use the DDPM-style q(x_{t-1}|x_t,x_0)
            # formula adapted to continuous time:
            #   mean = (alpha_next / alpha_cur) * (x - sigma_cur * eps_adj)
            # where eps_adj keeps eps consistent with the (alpha,sigma)
            # reparam. Variance ~ sigma_next^2 in the final step.
            x = alpha_next * x0 + sigma_next * eps
            if i < steps - 1:
                # Add small ancestral noise (continuous-time analogue).
                x = x + 0.0 * torch.randn_like(x)
        return x

    raise TypeError(f"unsupported diffusion type {type(diffusion).__name__}")


@torch.no_grad()
def ddim_sample(
    diffusion: Union[DiscreteDiffusion, ContinuousDiffusion],
    backbone,
    shape,
    external_cond: Optional[torch.Tensor] = None,
    n_steps: int = 50,
    eta: float = 0.0,
    device=None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """DDIM sampler (deterministic when ``eta=0``)."""
    device = device or (
        external_cond.device if external_cond is not None else torch.device("cpu")
    )
    B, T, _ = shape
    x = torch.randn(*shape, device=device, dtype=dtype)

    if isinstance(diffusion, DiscreteDiffusion):
        # Pick n_steps timesteps evenly across [0, timesteps).
        idx = torch.linspace(-1, diffusion.timesteps - 1, n_steps + 1).long().to(device)
        # Walk from high noise -> low noise.
        for i in reversed(range(n_steps)):
            k_cur = idx[i + 1]
            k_next = idx[i]
            k_cur_b = torch.full((B, T), k_cur.item(), device=device, dtype=torch.long)
            k_cur_clip = k_cur_b.clamp_(min=0)
            pred = diffusion.model_predictions(backbone, x, k_cur_clip, external_cond)
            x0 = pred.pred_x_start
            eps = pred.pred_noise

            alpha = diffusion.alphas_cumprod[k_cur_clip].unsqueeze(-1)
            if k_next < 0:
                alpha_next = torch.ones_like(alpha)
            else:
                k_next_b = torch.full(
                    (B, T), k_next.item(), device=device, dtype=torch.long
                )
                alpha_next = diffusion.alphas_cumprod[k_next_b].unsqueeze(-1)
            sigma = (
                eta
                * (
                    (1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)
                ).clamp_min(0.0).sqrt()
                if k_next >= 0
                else torch.zeros_like(alpha)
            )
            c = (1 - alpha_next - sigma ** 2).clamp_min(0.0).sqrt()
            noise = torch.randn_like(x) if eta > 0 and k_next >= 0 else torch.zeros_like(x)
            x = x0 * alpha_next.sqrt() + eps * c + sigma * noise
        return x

    if isinstance(diffusion, ContinuousDiffusion):
        ts = torch.linspace(1.0, 0.0, n_steps + 1, device=device)
        for i in range(n_steps):
            t_cur = torch.full((B, T), ts[i].item(), device=device)
            t_next = torch.full((B, T), ts[i + 1].item(), device=device)
            v = diffusion.model_v(backbone, x, t_cur, external_cond)
            x0, eps = diffusion.v_to_x0_and_noise(x, t_cur, v)
            logsnr_next = diffusion.schedule(t_next)
            alpha_next = torch.sigmoid(logsnr_next).sqrt().reshape(B, T, 1)
            sigma_next = torch.sigmoid(-logsnr_next).sqrt().reshape(B, T, 1)
            # DDIM (eta=0): x_{t_next} = alpha_next * x0 + sigma_next * eps.
            # eta > 0 adds Gaussian noise scaled by eta * sigma_next.
            if eta > 0:
                noise = torch.randn_like(x)
                x = alpha_next * x0 + sigma_next * (
                    (1 - eta ** 2).clamp_min(0.0).sqrt() * eps + eta * noise
                )
            else:
                x = alpha_next * x0 + sigma_next * eps
        return x

    raise TypeError(f"unsupported diffusion type {type(diffusion).__name__}")
