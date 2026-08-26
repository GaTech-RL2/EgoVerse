"""Unit tests for ``egomimic.models.dp3_encoder``.

    source emimic/bin/activate
    python -m pytest egomimic/tests/test_dp3_encoder.py -v
"""

from __future__ import annotations

import pytest
import torch

from egomimic.models.dp3_encoder import DP3PointNetEncoder

N = 256  # small cloud for fast tests


@pytest.fixture(scope="module")
def enc():
    torch.manual_seed(0)
    return DP3PointNetEncoder(output_dim=64, num_points=N)


def _cloud(B=2, T=1, I=1, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(B, T, I, N * 3, generator=g) * 2.0  # metres-ish


def test_shape_contract_hpt_layout(enc):
    out = enc(_cloud(B=3))
    assert out.shape == (3, 1, 64)


def test_shape_contract_TI_flatten(enc):
    out = enc(_cloud(B=2, T=1, I=2))
    assert out.shape == (2, 2, 64)


def test_accepts_BN3_and_flat_layouts(enc):
    x4 = _cloud(B=2)
    xBN3 = x4.reshape(2, N, 3)
    xflat = x4.reshape(2, N * 3)
    enc.eval()
    with torch.no_grad():
        o4, o3, o2 = enc(x4), enc(xBN3), enc(xflat)
    assert torch.allclose(o4, o3) and torch.allclose(o4, o2)


def test_wrong_point_count_raises(enc):
    with pytest.raises(ValueError, match="Expected"):
        enc(torch.rand(2, 1, 1, (N + 1) * 3))


def test_permutation_invariance(enc):
    """Max-pool over per-point MLP => output must be invariant to point order."""
    enc.eval()
    x = _cloud(B=2).reshape(2, N, 3)
    perm = torch.randperm(N)
    with torch.no_grad():
        assert torch.allclose(enc(x), enc(x[:, perm]), atol=1e-6)


def test_gradient_flows(enc):
    enc.train()
    x = _cloud(B=2).requires_grad_(True)
    enc(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_aug_off_by_default_is_deterministic():
    enc = DP3PointNetEncoder(output_dim=32, num_points=N)
    enc.train()
    x = _cloud(B=2)
    torch.manual_seed(1); o1 = enc(x)
    torch.manual_seed(2); o2 = enc(x)
    assert torch.allclose(o1, o2), "no-aug train forward must not depend on RNG"


def test_aug_on_varies_and_finite():
    enc = DP3PointNetEncoder(
        output_dim=32, num_points=N,
        pose_jitter_deg=5.0, pose_jitter_m=0.02,
        point_noise_std_m=0.01, point_dropout=0.1,
    )
    enc.train()
    x = _cloud(B=2)
    torch.manual_seed(1); o1 = enc(x)
    torch.manual_seed(2); o2 = enc(x)
    assert torch.isfinite(o1).all() and torch.isfinite(o2).all()
    assert not torch.allclose(o1, o2)


def test_aug_disabled_in_eval():
    enc = DP3PointNetEncoder(
        output_dim=32, num_points=N,
        pose_jitter_deg=5.0, pose_jitter_m=0.02,
        point_noise_std_m=0.01, point_dropout=0.1,
    )
    enc.eval()
    x = _cloud(B=2)
    with torch.no_grad():
        torch.manual_seed(1); o1 = enc(x)
        torch.manual_seed(2); o2 = enc(x)
    assert torch.allclose(o1, o2), "eval() must be deterministic with aug configured"


def test_matches_dp3_reference_math():
    """Hand-rolled reference of DP3's PointNetEncoderXYZ forward — guards the port."""
    torch.manual_seed(0)
    enc = DP3PointNetEncoder(output_dim=16, num_points=8)
    enc.eval()
    x = torch.rand(1, 8, 3)
    with torch.no_grad():
        ref = enc.final_projection(torch.max(enc.mlp(x), dim=1)[0])
        out = enc(x)
    assert torch.allclose(out.squeeze(1), ref)
