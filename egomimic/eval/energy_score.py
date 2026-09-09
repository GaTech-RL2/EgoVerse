"""Deterministic, block-balanced Energy Score for stochastic action chunks."""

from __future__ import annotations

import torch

DEFAULT_PLANAR_BLOCKS = ((0, 2), (2, 4), (4, 5))


def semantic_chunk_distance(
    left: torch.Tensor,
    right: torch.Tensor,
    blocks=DEFAULT_PLANAR_BLOCKS,
) -> torch.Tensor:
    """Average equal-weight RMS distances over declared semantic blocks."""
    if left.shape[-2:] != right.shape[-2:]:
        raise ValueError(
            f"action chunk shapes disagree: {left.shape[-2:]} vs {right.shape[-2:]}"
        )
    width = left.shape[-1]
    normalized_blocks = tuple(tuple(map(int, block)) for block in blocks)
    covered = [index for start, end in normalized_blocks for index in range(start, end)]
    if len(covered) != width or set(covered) != set(range(width)):
        raise ValueError(
            "semantic blocks must partition every action channel exactly once; "
            f"got {normalized_blocks} for width {width}"
        )
    terms = []
    for start, end in normalized_blocks:
        if not 0 <= int(start) < int(end) <= width:
            raise ValueError(f"invalid semantic block {(start, end)} for width {width}")
        error = (
            left[..., :, int(start) : int(end)] - right[..., :, int(start) : int(end)]
        )
        terms.append(error.square().mean(dim=(-2, -1)).sqrt())
    if not terms:
        raise ValueError("at least one semantic block is required")
    return torch.stack(terms).mean(dim=0)


def energy_score(
    samples: torch.Tensor,
    target: torch.Tensor,
    blocks=DEFAULT_PLANAR_BLOCKS,
) -> dict[str, torch.Tensor]:
    """Compute per-condition Energy Score using ordered distinct sample pairs.

    ``samples`` is ``(K, B, H, D)`` and ``target`` is ``(B, H, D)``. Returned
    scalar values are macro-averages across the B frozen validation conditions;
    ``*_by_condition`` values retain one diagnostic per condition.
    """
    if samples.ndim != 4 or target.ndim != 3 or samples.shape[1:] != target.shape:
        raise ValueError(
            f"expected samples (K,B,H,D) and target (B,H,D), got "
            f"{samples.shape} and {target.shape}"
        )
    sample_count = samples.shape[0]
    if sample_count < 2:
        raise ValueError("Energy Score requires at least two samples")
    accuracy_by_condition = semantic_chunk_distance(
        samples, target.unsqueeze(0), blocks
    ).mean(dim=0)
    pair_distance = semantic_chunk_distance(samples[:, None], samples[None, :], blocks)
    mask = ~torch.eye(sample_count, dtype=torch.bool, device=samples.device)
    diversity_by_condition = (
        pair_distance[mask].reshape(-1, target.shape[0]).mean(dim=0)
    )
    score_by_condition = accuracy_by_condition - 0.5 * diversity_by_condition
    return {
        "score": score_by_condition.mean(),
        "accuracy": accuracy_by_condition.mean(),
        "diversity": diversity_by_condition.mean(),
        "score_by_condition": score_by_condition,
        "accuracy_by_condition": accuracy_by_condition,
        "diversity_by_condition": diversity_by_condition,
    }
