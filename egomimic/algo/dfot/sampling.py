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


class _CFGBackbone:
    """Wrap a backbone for classifier-free-guidance inference.

    Each call runs TWO backbone passes (with cond and with cond zeroed
    post-embedding to match training-time dropout exactly) and returns
    the blended model output:

        out_guided = out_uncond + cfg_scale * (out_cond - out_uncond)

    For continuous diffusion (v-pred) the model output is ``v``; for
    discrete it's the configured ``objective`` output (v / x0 / noise) —
    the blend is correct on any of them because the diffusion class then
    converts the blended output through the same closed-form.

    ``cfg_scale=1.0`` short-circuits to a single pass (zero overhead).
    Requires the backbone to accept ``force_uncond=True`` (DFoTBackbone).
    """

    def __init__(self, backbone, cfg_scale: float):
        self.backbone = backbone
        self.cfg_scale = float(cfg_scale)

    def __call__(self, x, noise_levels, external_cond=None):
        if self.cfg_scale == 1.0 or external_cond is None:
            return self.backbone(x, noise_levels, external_cond=external_cond)
        out_cond = self.backbone(x, noise_levels, external_cond=external_cond)
        out_uncond = self.backbone(
            x, noise_levels, external_cond=external_cond, force_uncond=True
        )
        return out_uncond + self.cfg_scale * (out_cond - out_uncond)


@torch.no_grad()
def sample_step(
    diffusion: Union[ContinuousDiffusion, DiscreteDiffusion],
    backbone,
    *,
    x: torch.Tensor,
    current_levels: torch.Tensor,
    next_levels: torch.Tensor,
    external_cond: Optional[torch.Tensor] = None,
    eta: float = 0.0,
    cfg_scale: float = 1.0,
) -> torch.Tensor:
    """One denoise step. Per-token noise levels are arbitrary.

    This is the primitive — every other inference path is a loop over
    ``sample_step``. Use it directly when the schedule isn't known in
    advance (e.g. online AR rollout where the buffer grows over env ticks);
    use ``sample(schedule_matrix, ...)`` when you have the full plan up
    front.

    Args:
        diffusion: ``ContinuousDiffusion`` or ``DiscreteDiffusion``.
        backbone: callable ``(x, noise_levels, external_cond) -> pred``.
        x: current per-token state, ``(B, T, action_dim)``.
        current_levels: per-token noise levels at the current step, shape
            ``(B, T)`` or ``(T,)`` (broadcast over batch). Floats in
            ``[0, 1]`` for continuous diffusion, longs in ``[0, timesteps)``
            for discrete (``-1`` = fully clean sentinel).
        next_levels: per-token noise levels at the next step, same shape
            convention as ``current_levels``.
        external_cond: ``(B, T, cond_dim)`` or ``(B, cond_dim)`` or ``None``.
        eta: DDIM stochasticity (0.0 = deterministic).

    Returns:
        Updated ``x`` of the same shape ``(B, T, action_dim)``.

    Note:
        ``x.shape[1]`` (the T dimension) must match ``current_levels`` and
        ``next_levels``. If you want to *grow* T between calls (push new
        noisy tokens at the back of an AR buffer), append the new slot to
        ``x`` before calling ``sample_step`` and provide noise levels for
        the new tokens at the appropriate "fully noisy" entry.
    """
    # Support both 1D-bundle x of shape (B, T, action_dim) and spatial
    # x of shape (B, T, C, H, W). All per-token broadcasting below adds
    # singleton dims to match ``x.dim()`` instead of hardcoding unsqueeze(-1).
    B, T = x.shape[:2]
    dtype = x.dtype
    pad_dims = x.dim() - 2  # number of trailing dims after (B, T)

    def _to_x_shape(t: torch.Tensor) -> torch.Tensor:
        """Reshape ``(B, T)`` to broadcast against ``x``."""
        return t.reshape(B, T, *([1] * pad_dims))

    # Broadcast (T,) -> (B, T).
    if current_levels.dim() == 1:
        current_levels = current_levels.unsqueeze(0).expand(B, T)
    if next_levels.dim() == 1:
        next_levels = next_levels.unsqueeze(0).expand(B, T)

    # CFG: wrap backbone for two-pass cond/uncond blend when cfg_scale > 1.
    bb = _CFGBackbone(backbone, cfg_scale) if cfg_scale != 1.0 else backbone

    if isinstance(diffusion, ContinuousDiffusion):
        t_cur = current_levels.to(dtype)
        t_next = next_levels.to(dtype)
        v = diffusion.model_v(bb, x, t_cur, external_cond)
        x0, eps = diffusion.v_to_x0_and_noise(x, t_cur, v)
        logsnr_next = diffusion.schedule(t_next)
        alpha_next = _to_x_shape(torch.sigmoid(logsnr_next).sqrt())
        sigma_next = _to_x_shape(torch.sigmoid(-logsnr_next).sqrt())
        if eta > 0:
            noise = torch.randn_like(x)
            return alpha_next * x0 + sigma_next * (
                (1 - eta ** 2).clamp_min(0.0).sqrt() * eps + eta * noise
            )
        return alpha_next * x0 + sigma_next * eps

    if isinstance(diffusion, DiscreteDiffusion):
        k_cur = current_levels.long()
        k_next = next_levels.long()
        pred = diffusion.model_predictions(
            bb, x, k_cur.clamp_min(0), external_cond
        )
        x0 = pred.pred_x_start
        eps = pred.pred_noise
        alpha = _to_x_shape(diffusion.alphas_cumprod[k_cur.clamp_min(0)])
        k_next_clamp = k_next.clamp_min(0)
        alpha_next = _to_x_shape(diffusion.alphas_cumprod[k_next_clamp])
        fully_clean = _to_x_shape((k_next < 0).to(dtype=torch.bool).long()).bool()
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
        return x0 * alpha_next.sqrt() + eps * c + sigma * noise_term

    raise TypeError(f"unsupported diffusion type {type(diffusion).__name__}")


@torch.no_grad()
def sample(
    diffusion: Union[ContinuousDiffusion, DiscreteDiffusion],
    backbone,
    *,
    schedule_matrix: torch.Tensor,
    action_dim: Optional[int] = None,
    x_shape: Optional[tuple] = None,
    batch_size: int = 1,
    external_cond: Optional[torch.Tensor] = None,
    eta: float = 0.0,
    cfg_scale: float = 1.0,
    device=None,
    dtype: torch.dtype = torch.float32,
    x_init: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Denoise under a known static schedule. Thin loop over ``sample_step``.

    Use this when the full ``schedule_matrix`` is known up front (e.g.
    offline teacher-forced eval, full-chunk denoising, fixed-length
    staircase). For online AR where the schedule grows over time, call
    ``sample_step`` directly per env tick.

    Args:
        diffusion: ``ContinuousDiffusion`` or ``DiscreteDiffusion``.
        backbone: callable ``(x, noise_levels, external_cond) -> pred``.
        schedule_matrix: ``(n_steps + 1, T)``. Row 0 = initial per-token
            noise levels; row -1 = final.
        action_dim: trailing feature dim for 1D-bundle backbones; output
            shape is ``(B, T, action_dim)``. Exactly one of ``action_dim``
            / ``x_shape`` must be set.
        x_shape: trailing shape for spatial backbones (e.g. ``(C, H, W)``
            for ``DFoTSpatialBackbone``); output shape is
            ``(B, T, *x_shape)``.
        batch_size: batch dim for output; schedule is broadcast across batch.
        external_cond: ``(B, T, cond_dim)`` or ``(B, cond_dim)`` or ``None``.
        eta: DDIM stochasticity (0.0 = deterministic).
        x_init: optional initial sample; defaults to standard normal.

    Returns:
        Denoised tensor of shape ``(B, T, action_dim)`` (if ``action_dim``
        was given) or ``(B, T, *x_shape)`` (if ``x_shape`` was given).
    """
    if (action_dim is None) == (x_shape is None):
        raise ValueError(
            "exactly one of action_dim / x_shape must be provided to sample()."
        )
    n_steps_plus_one, T = schedule_matrix.shape
    n_steps = n_steps_plus_one - 1
    device = device or (
        external_cond.device if external_cond is not None else schedule_matrix.device
    )
    schedule_matrix = schedule_matrix.to(device)
    B = batch_size
    out_shape = (
        (B, T, int(action_dim)) if action_dim is not None
        else (B, T, *tuple(int(d) for d in x_shape))
    )

    if x_init is None:
        x = torch.randn(*out_shape, device=device, dtype=dtype)
    else:
        x = x_init.to(device=device, dtype=dtype)
        if tuple(x.shape) != out_shape:
            raise ValueError(
                f"x_init shape {tuple(x.shape)} != {out_shape}"
            )

    for s in range(n_steps):
        x = sample_step(
            diffusion,
            backbone,
            x=x,
            current_levels=schedule_matrix[s],
            next_levels=schedule_matrix[s + 1],
            external_cond=external_cond,
            eta=eta,
            cfg_scale=cfg_scale,
        )
    return x


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


# Online causal-AR rollout state is managed directly by ``DFoT._inference_step_ar``;
# see ``egomimic/algo/dfot/algo.py``. There's no separate class — the AR buffer
# advance is just one ``sample_step`` call plus front-pop/back-push bookkeeping.
