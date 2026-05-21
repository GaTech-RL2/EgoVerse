"""
DFoT sampling: one generic primitive + named schedule constructors.

Design contract
---------------
A *schedule_matrix* of shape ``(n_steps + 1, T)`` fully specifies an inference
pattern. Entry ``schedule_matrix[s, i]`` is the noise level of token ``i`` at
denoising step ``s``. The generic ``sample`` function walks ``s = 0 → n_steps``,
running the backbone with per-token noise levels read directly from the matrix
and doing a DDIM-style step from ``schedule_matrix[s]`` to
``schedule_matrix[s + 1]``.

  * **Continuous diffusion**: entries are floats in ``[0, 1]`` (consumed by
    ``ContinuousDiffusion.schedule`` to produce logSNR).
  * **Discrete diffusion**: entries are integer step indices in
    ``[0, timesteps)``. ``-1`` is treated as "fully clean" (alpha = 1).

Anything more complex than vanilla denoising — full-chunk, causal-AR
staircase, chunked-staircase — is just a different schedule_matrix built by
one of the named constructors below.

The convenience wrappers ``ddim_sample`` / ``ddpm_sample`` build a vanilla
(uniform-per-token) schedule and call ``sample``; they exist for backward
compatibility with the original API and as a smoke baseline.
"""

from typing import Optional, Union

import torch

from .continuous_diffusion import ContinuousDiffusion
from .discrete_diffusion import DiscreteDiffusion


# --------------------------------------------------------------------------- #
# Schedule constructors. Each returns a ``(n_steps + 1, T)`` matrix.
# --------------------------------------------------------------------------- #


def vanilla_schedule(
    n_steps: int, T: int, *, discrete_timesteps: Optional[int] = None
) -> torch.Tensor:
    """Standard uniform schedule — every token denoises by the same amount
    each step. Equivalent to vanilla DDIM/DDPM.

    Continuous: linearly anneals 1.0 → 0.0 over ``n_steps + 1`` points.
    Discrete: walks ``timesteps - 1 → 0``, then -1 (fully clean) at the end.
    """
    if discrete_timesteps is None:
        levels = torch.linspace(1.0, 0.0, n_steps + 1)
    else:
        levels = torch.linspace(
            float(discrete_timesteps - 1), -1.0, n_steps + 1
        ).long().clamp_min(-1)
    return levels[:, None].expand(n_steps + 1, T).contiguous()


def staircase_ar_schedule(
    T: int,
    *,
    chunk_size: int = 1,
    step_size: int = 1,
    discrete_timesteps: Optional[int] = None,
) -> torch.Tensor:
    """Causal-AR rolling staircase schedule.

    Two knobs control staircase geometry:
      * ``chunk_size`` (staircase "width"): number of tokens that share a
        rung's noise level. ``chunk_size=1`` reduces to one token per rung
        (classic causal AR); larger values let multiple tokens denoise
        together.
      * ``step_size`` (staircase "height"): number of denoising steps each
        rung takes before sliding forward.

    Noise level for token ``i`` at step ``s``:

        rung   = i // chunk_size
        level  = clamp(1 - (s - rung*step_size) / step_size, 0, 1)

    Total denoising steps = ``ceil(T / chunk_size) * step_size``.
    Matrix shape: ``(total_steps + 1, T)``.
    """
    n_chunks = (T + chunk_size - 1) // chunk_size
    n_total = n_chunks * step_size
    step_idx = torch.arange(n_total + 1).float()  # (n_total + 1,)
    tok_idx = torch.arange(T).float()
    rung = (tok_idx // chunk_size)
    raw = 1.0 - (step_idx[:, None] - rung[None, :] * step_size) / float(step_size)
    levels = raw.clamp(0.0, 1.0)
    if discrete_timesteps is not None:
        levels = (levels * (discrete_timesteps - 1)).round().long().clamp(
            0, discrete_timesteps - 1
        )
    return levels.contiguous()


def causal_ar_schedule(
    T: int,
    *,
    n_steps_per_token: int = 1,
    discrete_timesteps: Optional[int] = None,
) -> torch.Tensor:
    """Backwards-compatible thin wrapper: classic causal-AR (one token per
    rung). Equivalent to ``staircase_ar_schedule(T, chunk_size=1,
    step_size=n_steps_per_token, ...)``.
    """
    return staircase_ar_schedule(
        T=T,
        chunk_size=1,
        step_size=n_steps_per_token,
        discrete_timesteps=discrete_timesteps,
    )


# --------------------------------------------------------------------------- #
# Generic sample primitive.
# --------------------------------------------------------------------------- #


@torch.no_grad()
def sample(
    diffusion: Union[ContinuousDiffusion, DiscreteDiffusion],
    backbone,
    *,
    schedule_matrix: torch.Tensor,
    action_dim: int,
    batch_size: int = 1,
    external_cond: Optional[torch.Tensor] = None,
    eta: float = 0.0,
    device=None,
    dtype: torch.dtype = torch.float32,
    x_init: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Denoise an action chunk under an arbitrary per-token-per-step
    schedule.

    Args:
        diffusion: ``ContinuousDiffusion`` or ``DiscreteDiffusion``.
        backbone: callable ``(x, noise_levels, external_cond) -> pred``.
        schedule_matrix: ``(n_steps + 1, T)``. Row 0 = initial per-token
            noise levels; row -1 = final. Floats for continuous, longs for
            discrete (``-1`` denotes "fully clean" sentinel).
        action_dim: feature dim of the action.
        batch_size: batch dim for output; schedule is broadcast across batch.
        external_cond: optional obs cond ``(B, T, cond_dim)`` or
            ``(B, cond_dim)``.
        eta: DDIM stochasticity (0.0 = deterministic).
        x_init: optional initial sample. If None, samples standard normal.
            Useful for stateful online rollouts that want to carry a buffer
            across calls.

    Returns:
        ``(B, T, action_dim)`` denoised actions.
    """
    n_steps_plus_one, T = schedule_matrix.shape
    n_steps = n_steps_plus_one - 1
    device = device or (
        external_cond.device if external_cond is not None else schedule_matrix.device
    )
    schedule_matrix = schedule_matrix.to(device)
    B = batch_size

    if x_init is None:
        x = torch.randn(B, T, action_dim, device=device, dtype=dtype)
    else:
        x = x_init.to(device=device, dtype=dtype)
        if x.shape != (B, T, action_dim):
            raise ValueError(
                f"x_init shape {tuple(x.shape)} != ({B}, {T}, {action_dim})"
            )

    if isinstance(diffusion, ContinuousDiffusion):
        for s in range(n_steps):
            t_cur = schedule_matrix[s].to(dtype).unsqueeze(0).expand(B, T)
            t_next = schedule_matrix[s + 1].to(dtype).unsqueeze(0).expand(B, T)
            v = diffusion.model_v(backbone, x, t_cur, external_cond)
            x0, eps = diffusion.v_to_x0_and_noise(x, t_cur, v)
            logsnr_next = diffusion.schedule(t_next)
            alpha_next = torch.sigmoid(logsnr_next).sqrt().unsqueeze(-1)
            sigma_next = torch.sigmoid(-logsnr_next).sqrt().unsqueeze(-1)
            if eta > 0:
                noise = torch.randn_like(x)
                x = alpha_next * x0 + sigma_next * (
                    (1 - eta ** 2).clamp_min(0.0).sqrt() * eps + eta * noise
                )
            else:
                x = alpha_next * x0 + sigma_next * eps
        return x

    if isinstance(diffusion, DiscreteDiffusion):
        for s in range(n_steps):
            k_cur = schedule_matrix[s].long().unsqueeze(0).expand(B, T)
            k_next = schedule_matrix[s + 1].long().unsqueeze(0).expand(B, T)
            pred = diffusion.model_predictions(
                backbone, x, k_cur.clamp_min(0), external_cond
            )
            x0 = pred.pred_x_start
            eps = pred.pred_noise
            alpha = diffusion.alphas_cumprod[k_cur.clamp_min(0)].unsqueeze(-1)
            k_next_clamp = k_next.clamp_min(0)
            alpha_next = diffusion.alphas_cumprod[k_next_clamp].unsqueeze(-1)
            fully_clean = (k_next < 0).unsqueeze(-1)
            alpha_next = torch.where(
                fully_clean, torch.ones_like(alpha_next), alpha_next
            )
            sigma_sq = (
                eta ** 2
                * (1 - alpha / alpha_next.clamp_min(1e-8)).clamp_min(0.0)
                * (1 - alpha_next).clamp_min(0.0)
                / (1 - alpha).clamp_min(1e-8)
            )
            sigma = sigma_sq.clamp_min(0.0).sqrt()
            c = (1 - alpha_next - sigma ** 2).clamp_min(0.0).sqrt()
            noise_term = torch.randn_like(x) if eta > 0 else torch.zeros_like(x)
            x = x0 * alpha_next.sqrt() + eps * c + sigma * noise_term
        return x

    raise TypeError(f"unsupported diffusion type {type(diffusion).__name__}")


# --------------------------------------------------------------------------- #
# Convenience wrappers (vanilla DDIM / DDPM).
# --------------------------------------------------------------------------- #


def ddim_sample(
    diffusion: Union[ContinuousDiffusion, DiscreteDiffusion],
    backbone,
    shape,
    external_cond: Optional[torch.Tensor] = None,
    n_steps: int = 50,
    eta: float = 0.0,
    device=None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Vanilla DDIM — ``sample`` called with a uniform-per-token schedule."""
    B, T, A = shape
    discrete_ts = (
        int(diffusion.timesteps) if isinstance(diffusion, DiscreteDiffusion) else None
    )
    sm = vanilla_schedule(n_steps=n_steps, T=T, discrete_timesteps=discrete_ts)
    return sample(
        diffusion,
        backbone,
        schedule_matrix=sm,
        action_dim=A,
        batch_size=B,
        external_cond=external_cond,
        eta=eta,
        device=device,
        dtype=dtype,
    )


def ddpm_sample(
    diffusion: DiscreteDiffusion,
    backbone,
    shape,
    external_cond: Optional[torch.Tensor] = None,
    n_steps: Optional[int] = None,
    device=None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """DDPM ancestral — only meaningful for ``DiscreteDiffusion``. For
    continuous-time DFoT, use ``ddim_sample`` (true continuous-time DDPM
    requires an SDE solver, out of scope here)."""
    if not isinstance(diffusion, DiscreteDiffusion):
        raise TypeError(
            "ddpm_sample only supports DiscreteDiffusion; use ddim_sample "
            "for continuous-time DFoT."
        )
    B, T, A = shape
    steps = n_steps or diffusion.timesteps
    if steps != diffusion.timesteps:
        raise ValueError(
            f"DDPM ancestral requires n_steps == diffusion.timesteps "
            f"(got {steps} vs {diffusion.timesteps}); use ddim_sample "
            "for fewer steps."
        )
    sm = vanilla_schedule(
        n_steps=steps, T=T, discrete_timesteps=diffusion.timesteps
    )
    return sample(
        diffusion,
        backbone,
        schedule_matrix=sm,
        action_dim=A,
        batch_size=B,
        external_cond=external_cond,
        eta=1.0,  # ancestral noise on every step
        device=device,
        dtype=dtype,
    )


# --------------------------------------------------------------------------- #
# Online causal-AR rollout helper.
# --------------------------------------------------------------------------- #


class CausalARRollout:
    """Online causal-AR sampler for closed-loop env interaction.

    Maintains a token buffer of length ``buffer_size``. On each ``step()``:
      1. Build a 2-row schedule slice ``[current_levels, next_levels]``
         describing the rolling staircase advance.
      2. Call ``sample`` with that mini-schedule, feeding the current buffer
         in via ``x_init`` so we keep state across env ticks.
      3. Commit the ``chunk_size`` front tokens (now at noise=0) and slide
         a fresh fully-noisy slot into the back.

    Args:
        diffusion: continuous or discrete diffusion module.
        backbone: ``(x, noise_levels, cond) -> pred`` callable.
        action_dim: feature dim of the action.
        buffer_size: number of in-flight tokens.
        chunk_size: tokens per rung. With ``chunk_size=1`` and
            ``buffer_size=k``, each ``step()`` commits one action and pushes
            one new noisy token.
        device: torch device for buffer state.
        dtype: float dtype.

    Notes:
        Designed for closed-loop sim eval where each env step provides a
        fresh ``external_cond`` (the current obs's encoding). Pass the
        per-tick cond into ``step(external_cond)``.
    """

    def __init__(
        self,
        diffusion: Union[ContinuousDiffusion, DiscreteDiffusion],
        backbone,
        *,
        action_dim: int,
        buffer_size: int,
        device,
        chunk_size: int = 1,
        dtype: torch.dtype = torch.float32,
    ):
        self.diffusion = diffusion
        self.backbone = backbone
        self.action_dim = int(action_dim)
        self.buffer_size = int(buffer_size)
        self.chunk_size = int(chunk_size)
        self.device = device
        self.dtype = dtype
        self._discrete = isinstance(diffusion, DiscreteDiffusion)
        if self._discrete:
            self._unit = max(1, diffusion.timesteps // self.buffer_size)
        else:
            self._unit = 1.0 / float(self.buffer_size)

        # Fully-noisy initial buffer.
        self.x = torch.randn(
            1, self.buffer_size, self.action_dim, device=device, dtype=dtype
        )

    def _levels(self, offset: float) -> torch.Tensor:
        """Per-token noise levels at staircase offset ``offset`` (in unit
        rungs). ``offset=0`` means token 0 is one unit above clean, the
        front-most one-step-to-commit configuration. After one rung advance
        (``offset=1``), token 0 is fully clean."""
        idx = torch.arange(self.buffer_size, device=self.device).float() + 1.0
        idx = idx - offset
        if self._discrete:
            levels = (idx * self._unit).long().clamp(-1, self.diffusion.timesteps - 1)
        else:
            levels = (idx * self._unit).clamp(0.0, 1.0)
        return levels.unsqueeze(0)  # (1, buffer_size)

    @torch.no_grad()
    def step(self, external_cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Advance the staircase by one unit, commit the front ``chunk_size``
        tokens, push fresh fully-noisy slots. Returns the committed actions
        of shape ``(chunk_size, action_dim)``."""
        cur = self._levels(offset=0.0)
        nxt = self._levels(offset=1.0)
        schedule = torch.stack([cur.squeeze(0), nxt.squeeze(0)], dim=0)
        x_next = sample(
            self.diffusion,
            self.backbone,
            schedule_matrix=schedule,
            action_dim=self.action_dim,
            batch_size=1,
            external_cond=external_cond,
            device=self.device,
            dtype=self.dtype,
            x_init=self.x,
        )
        self.x = x_next
        # Pop the chunk_size front tokens (committed), push fresh noisy tokens.
        front = self.x[:, : self.chunk_size, :].clone()  # (1, chunk_size, A)
        new_back = torch.randn(
            1, self.chunk_size, self.action_dim, device=self.device, dtype=self.dtype
        )
        self.x = torch.cat([self.x[:, self.chunk_size :, :], new_back], dim=1)
        return front.squeeze(0)  # (chunk_size, A)
