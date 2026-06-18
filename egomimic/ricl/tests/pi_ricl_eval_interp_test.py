"""CPU unit tests for PIRiclEval's distance-weighted action interpolation.

The blend (`_interpolate_toward_neighbor`) and the per-batch distance scale
(`_interp_scale`) are pure torch (they never touch ``self``), so we call them
unbound with a dummy ``self`` — no model, no openpi, no GPU. This pins the
correctness-critical math behind the inference-only interpolation lever ported
from ``egomimic/ricl/droid_eval.py``:

  p <- w*a_nn + (1-w)*p  over the overlapping chunk,  w = exp(-lamda*dist/dist_max)

masked (invalid neighbor -> w=0) and finite-guarded (inf distance -> w=0).
"""

from __future__ import annotations

import math

import torch

from egomimic.eval.pi_ricl_eval import PIRiclEval


# Both methods are pure (never touch ``self``); call them unbound with a dummy self.
def _blend(*args, **kwargs):
    return PIRiclEval._interpolate_toward_neighbor(None, *args, **kwargs)


def _scale(proc):
    return PIRiclEval._interp_scale(None, proc)


def _inputs(B=2, T=4, D=3, dist=0.0, mask=1.0):
    p = torch.arange(B * T * D, dtype=torch.float32).reshape(B, T, D)
    a0 = -torch.ones(B, T, D)  # neighbor chunk, distinct from p
    d = torch.full((B, 1), float(dist))
    m = torch.full((B, 1), float(mask))
    return p, a0, d, m


def test_w1_returns_neighbor_on_overlap():
    # dist=0 -> w=exp(0)=1 -> output is the neighbor chunk exactly.
    p, a0, d, m = _inputs(dist=0.0)
    out, wmean = _blend(p, a0, d, m, dist_max=1.0, lamda=1.0)
    assert torch.allclose(out, a0)
    assert math.isclose(wmean, 1.0, rel_tol=1e-6)


def test_large_lambda_returns_p():
    # huge lamda -> w=exp(-inf)~0 -> output is the model prediction unchanged.
    p, a0, d, m = _inputs(dist=1.0)
    out, wmean = _blend(p, a0, d, m, dist_max=1.0, lamda=1e6)
    assert torch.allclose(out, p)
    assert wmean < 1e-6


def test_masked_neighbor_gives_w0():
    # invalid neighbor (mask=0) -> w forced to 0 -> prediction unchanged even at dist=0.
    p, a0, d, m = _inputs(dist=0.0, mask=0.0)
    out, wmean = _blend(p, a0, d, m, dist_max=1.0, lamda=1.0)
    assert torch.allclose(out, p)
    assert wmean == 0.0


def test_nonfinite_distance_gives_w0():
    p, a0, d, m = _inputs(dist=0.0)
    d[:, 0] = float("inf")
    out, wmean = _blend(p, a0, d, m, dist_max=1.0, lamda=1.0)
    assert torch.allclose(out, p)
    assert wmean == 0.0


def test_intermediate_weight_is_convex_blend():
    # known w: dist=dist_max, lamda=1 -> w=exp(-1).
    p, a0, d, m = _inputs(dist=2.0)
    out, _ = _blend(p, a0, d, m, dist_max=2.0, lamda=1.0)
    w = math.exp(-1.0)
    expected = w * a0 + (1.0 - w) * p
    assert torch.allclose(out, expected, atol=1e-6)


def test_blend_only_touches_overlapping_steps():
    # neighbor chunk shorter than the prediction -> only the first H steps blend.
    p, _, d, m = _inputs(B=1, T=5, D=2, dist=0.0)
    a0 = -torch.ones(1, 3, 2)  # H = min(5, 3) = 3
    out, _ = _blend(p, a0, d, m, dist_max=1.0, lamda=1.0)
    assert torch.allclose(out[:, :3], a0)  # blended
    assert torch.allclose(out[:, 3:], p[:, 3:])  # untouched tail


def test_interp_scale_max_finite_distance():
    proc = {0: {"ricl_retrieved_dist": torch.tensor([[0.5, 1.5], [float("inf"), 2.0]])}}
    assert _scale(proc) == 2.0


def test_interp_scale_none_when_no_positive_distance():
    # all-zero (cache without distances) -> None so the sweep is skipped, not w=1.
    proc = {0: {"ricl_retrieved_dist": torch.zeros(3, 4)}}
    assert _scale(proc) is None
    assert _scale({0: {}}) is None
