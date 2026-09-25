"""Numerical boundaries for an actual language-embedding intervention."""

import math

import pytest
import torch

from astra_reversal.intervention_conditioning import (
    TextEmbeddingIntervention,
    pooled_text_residual,
)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"guidance_prompt": ""},
        {"alpha": True},
        {"alpha": -0.01},
        {"alpha": 1.01},
        {"alpha": math.nan},
        {"max_relative_norm": math.inf},
        {"max_relative_norm": 0.26},
    ],
)
def test_invalid_intervention_is_rejected(kwargs):
    values = {"guidance_prompt": "put the cup on the plate", "alpha": 0.5}
    values.update(kwargs)
    with pytest.raises(ValueError):
        TextEmbeddingIntervention(**values)


def test_disabled_intervention_is_identity_without_reading_guidance():
    base = torch.tensor([[[2.0, 1.0], [999.0, -999.0]]])
    mask = torch.tensor([[True, False]])
    result, metrics = pooled_text_residual(base, mask, object(), object(), alpha=0)
    assert result is base
    assert metrics["delta_frobenius"] == metrics["relative_rms"] == 0


def test_bound_uses_actual_stored_delta_and_keeps_padding_unchanged():
    base = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [999.0, -999.0]]])
    mask = torch.tensor([[True, True, False]])
    guidance = torch.tensor([[[100.0, -100.0], [float(2**20), float(2**20)]]])
    guidance_mask = torch.tensor([[True, False]])
    before = base.clone()
    edited, metrics = pooled_text_residual(
        base, mask, guidance, guidance_mask, alpha=1.0
    )
    assert torch.equal(base, before)
    assert torch.equal(edited[~mask], base[~mask])
    delta = edited[mask].double() - base[mask].double()
    actual = torch.linalg.vector_norm(delta).item()
    original_norm = torch.linalg.vector_norm(base[mask].double()).item()
    assert 0 < actual <= 0.25 * original_norm
    assert metrics["bound_applied"]
    assert metrics["delta_frobenius"] == actual
    assert metrics["relative_rms"] == actual / original_norm
    # The enormous masked guidance slot must not change the result.
    other = guidance.clone()
    other[:, 1] = -123.0
    again, _ = pooled_text_residual(base, mask, other, guidance_mask, alpha=1.0)
    assert torch.equal(edited, again)


def test_guidance_length_does_not_create_or_remove_text_slots():
    base = torch.tensor([[[4.0, 4.0], [6.0, 6.0], [20.0, 30.0]]])
    mask = torch.tensor([[True, True, False]])
    guide = torch.tensor([[[6.0, 6.0], [8.0, 8.0], [10.0, 10.0], [12.0, 12.0]]])
    guide_mask = torch.ones((1, 4), dtype=torch.bool)
    edited, metrics = pooled_text_residual(base, mask, guide, guide_mask, alpha=0.1)
    assert edited.shape == base.shape
    torch.testing.assert_close(edited[mask], base[mask] + 0.4)
    assert torch.equal(edited[~mask], base[~mask])
    assert not metrics["bound_applied"]


def test_zero_original_norm_has_zero_budget_and_finite_metrics():
    base = torch.zeros((1, 2, 3))
    mask = torch.ones((1, 2), dtype=torch.bool)
    edited, metrics = pooled_text_residual(
        base, mask, torch.ones_like(base), mask, alpha=1
    )
    assert torch.equal(edited, base)
    assert metrics["relative_rms"] == 0
    assert all(
        math.isfinite(value) for value in metrics.values() if isinstance(value, float)
    )


@pytest.mark.parametrize("bad", ["mask", "empty", "nonfinite", "width", "dtype"])
def test_malformed_guidance_is_rejected(bad):
    base = torch.ones((1, 2, 3))
    mask = torch.ones((1, 2), dtype=torch.bool)
    guide, guide_mask = base.clone(), mask.clone()
    if bad == "mask":
        guide_mask = guide_mask.float()
    elif bad == "empty":
        guide_mask[:] = False
    elif bad == "nonfinite":
        guide[0, 0, 0] = math.nan
    elif bad == "width":
        guide = guide[:, :, :2]
    else:
        guide = guide.double()
    with pytest.raises(ValueError):
        pooled_text_residual(base, mask, guide, guide_mask, alpha=0.5)
