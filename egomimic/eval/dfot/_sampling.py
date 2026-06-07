"""Shared anchored-DDIM rollout helper for the DFoT evaluators (COMBINE B).

The "anchored clean-history" rollout pattern is structurally identical across
two single-tensor DFoT samplers:

  * :meth:`DFoTBundleAnchoredEval._rollout` — flat bundle ``(1, T, bundle_dim)``.
  * :meth:`ImageSpatialDFoTOuterStage._rollout_latent` (anchored branch) —
    spatial latent ``(B, T, C, H, W)``.

Both clone a per-token noise schedule, PIN the first ``n_ctx`` tokens CLEAN
(``-1`` for discrete diffusion, ``0.0`` for continuous), seed those tokens from
a context tensor, then loop :func:`sample_step` re-pinning the context after
every denoise step. They differ ONLY in the trailing tensor shape, the batch
size, and whether they pass a non-default ``cfg_scale`` — all of which this
helper takes as explicit arguments, so the two call sites stay byte-identical.

NOTE — the 2D-policy ``DFoTPolicyActionEval._rollout`` is a DIFFERENT sampler:
it co-denoises TWO streams (obs-latent ``x_lat`` + action ``x_act``) through a
dual-output backbone ``(v_lat, v_act)`` with a hand-rolled v-prediction DDIM
step, not the single-tensor ``sample_step`` primitive. It is intentionally NOT
folded into this helper (its loop body has no single-tensor equivalent).
"""

from __future__ import annotations

import torch

from egomimic.models.diffusion.sampling import sample_step


@torch.no_grad()
def anchored_ddim_rollout(
    diffusion,
    backbone,
    *,
    schedule: torch.Tensor,
    context: torch.Tensor,
    total_T: int,
    trailing_shape,
    device,
    batch_size: int = 1,
    external_cond: torch.Tensor | None = None,
    discrete_ts: int | None = None,
    cfg_scale: float = 1.0,
) -> torch.Tensor:
    """Anchored clean-history DDIM rollout over a single per-token tensor.

    Seeds the first ``n_ctx`` (= ``context.shape[1]``) tokens from ``context``,
    pins them CLEAN across every step of ``schedule``, and denoises the rest via
    repeated :func:`sample_step` (eta=0). Returns the final ``x`` of shape
    ``(batch_size, total_T, *trailing_shape)``.

    Args:
        diffusion: ``ContinuousDiffusion`` or ``DiscreteDiffusion``.
        backbone: callable passed straight to :func:`sample_step`.
        schedule: per-step per-token noise levels ``(n_steps, total_T)``. NOT
            mutated — cloned internally before the context columns are pinned.
        context: clean prefix ``(batch_size, n_ctx, *trailing_shape)`` seeded
            into and re-pinned over the first ``n_ctx`` tokens each step.
        total_T: full token length ``T`` of the rollout.
        trailing_shape: per-token trailing dims (e.g. ``(bundle_dim,)`` or
            ``(C, H, W)``); ``x`` is ``(batch_size, total_T, *trailing_shape)``.
        device: device for the noise init.
        batch_size: leading batch dim of ``x`` (default 1).
        external_cond: optional ``(B, T, cond_dim)`` / ``(B, cond_dim)`` cond.
        discrete_ts: ``int`` discrete-timestep count -> clean sentinel ``-1``;
            ``None`` (continuous) -> clean sentinel ``0.0``.
        cfg_scale: classifier-free-guidance scale forwarded to ``sample_step``.

    Returns:
        ``x`` of shape ``(batch_size, total_T, *trailing_shape)``.
    """
    n_ctx = context.shape[1]
    clean = -1 if discrete_ts is not None else 0.0
    schedule = schedule.clone()
    schedule[:, :n_ctx] = clean
    x = torch.randn(batch_size, total_T, *trailing_shape, device=device)
    x[:, :n_ctx] = context
    for s in range(schedule.shape[0] - 1):
        x = sample_step(
            diffusion, backbone, x=x,
            current_levels=schedule[s], next_levels=schedule[s + 1],
            external_cond=external_cond, eta=0.0, cfg_scale=cfg_scale,
        )
        x[:, :n_ctx] = context
    return x
