"""Distributional losses for Pipeline-native action generators."""

from __future__ import annotations

import math

import torch


def _masked_action_view(
    prediction_samples: torch.Tensor,
    target: torch.Tensor,
    pad_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return FP32 flattened actions and the active dimension count.

    ``prediction_samples`` is grouped by condition. Each complete ``H x D``
    action chunk is one event for the energy score; coordinates and timesteps
    are not treated as independent samples.
    """

    if prediction_samples.ndim != 4:
        raise ValueError(
            "prediction_samples must have shape (B, K, H, D), got "
            f"{tuple(prediction_samples.shape)}"
        )
    if target.ndim != 3:
        raise ValueError(f"target must have shape (B, H, D), got {tuple(target.shape)}")
    if (
        prediction_samples.shape[0] != target.shape[0]
        or prediction_samples.shape[2:] != target.shape[1:]
    ):
        raise ValueError(
            "prediction/target event shape mismatch: "
            f"prediction={tuple(prediction_samples.shape)} target={tuple(target.shape)}"
        )

    prediction = prediction_samples.float()
    target_fp32 = target.float()
    batch_size, num_samples = prediction.shape[:2]
    event_size = int(target_fp32.shape[1] * target_fp32.shape[2])

    if pad_mask is None:
        active = torch.full(
            (batch_size, 1),
            float(event_size),
            device=prediction.device,
            dtype=torch.float32,
        )
        return (
            prediction.reshape(batch_size, num_samples, event_size),
            target_fp32.reshape(batch_size, 1, event_size),
            active,
        )

    mask = pad_mask.to(device=prediction.device, dtype=torch.float32)
    if mask.ndim == 2:
        mask = mask.unsqueeze(-1)
    if mask.ndim != 3:
        raise ValueError(
            f"pad_mask must have shape (B, H) or (B, H, D), got {tuple(mask.shape)}"
        )
    try:
        mask = mask.expand_as(target_fp32)
    except RuntimeError as error:
        raise ValueError(
            f"pad_mask {tuple(mask.shape)} cannot cover target {tuple(target.shape)}"
        ) from error
    mask = mask.reshape(batch_size, 1, event_size)
    active = mask.sum(dim=-1)
    if not bool(torch.all(active > 0.0)):
        raise ValueError("every condition must retain at least one action coordinate")
    return (
        prediction.reshape(batch_size, num_samples, event_size) * mask,
        target_fp32.reshape(batch_size, 1, event_size) * mask,
        active,
    )


def _distance_power_from_squared(
    squared_distance: torch.Tensor,
    active_dimensions: torch.Tensor,
    *,
    beta: float,
    normalize_by_dimension: bool,
) -> torch.Tensor:
    """Compute ``(||x-y||_2 / sqrt(d))**beta`` with a safe zero gradient."""

    scaled = squared_distance
    if normalize_by_dimension:
        while active_dimensions.ndim < scaled.ndim:
            active_dimensions = active_dimensions.unsqueeze(-1)
        scaled = scaled / active_dimensions

    # The exact distance is zero at collisions. Clamp only the backward path
    # at floating-point zero so beta < 1 does not create NaN/Inf gradients.
    # Every positive normal FP32 value retains the exact power.
    tiny = torch.finfo(scaled.dtype).tiny
    powered = scaled.clamp_min(tiny).pow(0.5 * beta)
    return torch.where(scaled > 0.0, powered, torch.zeros_like(powered))


def conditional_energy_score(
    prediction_samples: torch.Tensor,
    target: torch.Tensor,
    *,
    beta: float = 1.0,
    normalize_by_dimension: bool = True,
    pad_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Unbiased grouped-sample conditional energy-score estimator.

    Args:
        prediction_samples: ``(B, K, H, D)`` samples. The ``K`` samples in a
            group must share one condition and differ only through generator
            randomness.
        target: One observed ``(B, H, D)`` action chunk per condition.
        beta: Energy-distance exponent. Strict propriety holds for
            ``0 < beta < 2`` under the usual finite-moment assumptions.
        normalize_by_dimension: Use RMS Euclidean distance so action4 and
            point6 retain comparable per-embodiment scale before equal domain
            averaging. This positive domain-specific rescaling preserves the
            population optimum.
        pad_mask: Optional ``(B,H)`` or ``(B,H,D)`` validity mask.

    The estimator is

    ``mean_i ||X_i-y||^beta - sum_{i!=j}||X_i-X_j||^beta/(2K(K-1))``.

    All distances and returned metrics are computed in FP32 even when model
    activations are BF16.
    """

    beta = float(beta)
    if not math.isfinite(beta) or not 0.0 < beta < 2.0:
        raise ValueError(f"beta must satisfy 0 < beta < 2, got {beta}")

    prediction, target_fp32, active = _masked_action_view(
        prediction_samples, target, pad_mask
    )
    batch_size, num_samples, _ = prediction.shape
    if num_samples < 2:
        raise ValueError(
            "conditional energy score needs at least two samples per condition, "
            f"got K={num_samples}"
        )

    target_squared = (prediction - target_fp32).square().sum(dim=-1)
    pairwise_squared = (
        (prediction[:, :, None, :] - prediction[:, None, :, :]).square().sum(dim=-1)
    )
    attraction_values = _distance_power_from_squared(
        target_squared,
        active,
        beta=beta,
        normalize_by_dimension=normalize_by_dimension,
    )
    pairwise_values = _distance_power_from_squared(
        pairwise_squared,
        active,
        beta=beta,
        normalize_by_dimension=normalize_by_dimension,
    )

    diagonal = torch.eye(
        num_samples, device=prediction.device, dtype=torch.bool
    ).unsqueeze(0)
    off_diagonal_sum = pairwise_values.masked_fill(diagonal, 0.0).sum(dim=(1, 2))
    per_condition_attraction = attraction_values.mean(dim=1)
    per_condition_pairwise = off_diagonal_sum / (num_samples * (num_samples - 1))
    per_condition_repulsion = 0.5 * per_condition_pairwise
    per_condition_score = per_condition_attraction - per_condition_repulsion

    per_sample_mse = target_squared / active
    ensemble_mean_squared = (
        prediction.mean(dim=1, keepdim=True) - target_fp32
    ).square().sum(dim=-1) / active

    return {
        "score": per_condition_score.mean(),
        "attraction": per_condition_attraction.mean(),
        "repulsion": per_condition_repulsion.mean(),
        "pairwise_distance": per_condition_pairwise.mean(),
        "mse": per_sample_mse.mean(),
        "ensemble_mean_mse": ensemble_mean_squared.mean(),
        "best_of_k_mse": per_sample_mse.min(dim=1).values.mean(),
        "per_condition_score": per_condition_score,
        "per_condition_attraction": per_condition_attraction,
        "per_condition_repulsion": per_condition_repulsion,
        "per_condition_pairwise_distance": per_condition_pairwise,
        "per_condition_mse": per_sample_mse.mean(dim=1),
    }
