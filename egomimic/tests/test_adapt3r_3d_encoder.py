"""Unit tests for ``egomimic.models.adapt3r_3d_encoder``.

Run from EgoVerse repo root with:

    source emimic/bin/activate
    python -m pytest egomimic/tests/test_adapt3r_3d_encoder.py -v

Or, to skip the slow DINO download tests:

    python -m pytest egomimic/tests/test_adapt3r_3d_encoder.py -v -m "not slow"
"""

from __future__ import annotations

import math
import os
import sys
import warnings

import pytest
import torch

from egomimic.models.adapt3r_3d_encoder import (
    Adapt3R3DEncoder,
    AttentionExtractor,
    NeRFSinusoidalPosEmb,
    depth2fgpcd_batch,
    fps_pytorch,
    lift_point_cloud_batch,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _identity_extrinsics():
    return [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]


def _aria_intrinsics_orig():
    """Aria RGB-camera intrinsics at native 2016×1512 (from user-provided calib)."""
    return [
        [877.96474921, 0.0, 989.30439345],
        [0.0, 877.96474921, 744.17180392],
        [0.0, 0.0, 1.0],
    ]


def _make_rgbd(B=2, T=1, I=1, H=64, W=64, depth=1.0, seed=0):
    """4-channel RGBD tensor with a non-zero, non-trivial pattern."""
    g = torch.Generator().manual_seed(seed)
    rgb = torch.rand(B, T, I, 3, H, W, generator=g)
    d = torch.full((B, T, I, 1, H, W), float(depth)) + 0.05 * torch.randn(
        B, T, I, 1, H, W, generator=g
    )
    return torch.cat([rgb, d.clamp_min(0.1)], dim=3)


# ---------------------------------------------------------------------------
# 1. NeRFSinusoidalPosEmb
# ---------------------------------------------------------------------------

def test_nerf_dim_constraint():
    with pytest.raises(AssertionError):
        NeRFSinusoidalPosEmb(dim=10)            # not divisible by 6
    NeRFSinusoidalPosEmb(dim=60)               # ok


def test_nerf_shape_and_determinism():
    enc = NeRFSinusoidalPosEmb(dim=60)
    x = torch.randn(4, 16, 3)
    a = enc(x)
    b = enc(x)
    assert a.shape == (4, 16, 60)
    assert torch.allclose(a, b)


def test_nerf_distinct_outputs():
    enc = NeRFSinusoidalPosEmb(dim=60)
    x = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    out = enc(x)
    # Distinct inputs → distinct embeddings.
    assert not torch.allclose(out[0, 0], out[0, 1])
    assert not torch.allclose(out[0, 1], out[0, 2])


def test_nerf_matches_original_adapt3r():
    """Bitwise compare against the original Adapt3R implementation."""
    adapt3r_path = "/home/droid_robot/zhenyang/Adapt3R"
    if not os.path.exists(adapt3r_path):
        pytest.skip("Adapt3R repo not available for differential test")
    sys.path.insert(0, adapt3r_path)
    try:
        from adapt3r.algos.utils.position_encodings import NeRFSinusoidalPosEmb as Orig
    finally:
        sys.path.remove(adapt3r_path)

    torch.manual_seed(0)
    a = Orig(60)
    b = NeRFSinusoidalPosEmb(60)
    x = torch.randn(2, 8, 3)
    out_a = a(x)
    out_b = b(x)
    # Both are no_grad / parameter-free; outputs should match exactly.
    assert torch.allclose(out_a, out_b, atol=1e-6), (
        f"max diff = {(out_a - out_b).abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# 2. AttentionExtractor
# ---------------------------------------------------------------------------

def test_attention_extractor_shape():
    ex = AttentionExtractor(in_shape=120, out_channels=256, hidden_dim=256, num_heads=4)
    pc = torch.randn(3, 1, 64, 120)
    out = ex(pc)
    assert out.shape == (3, 1, 256)


def test_attention_extractor_return_attn():
    ex = AttentionExtractor(in_shape=120, out_channels=256, hidden_dim=256, num_heads=4)
    pc = torch.randn(2, 1, 32, 120)
    out, attn = ex(pc, return_attn=True)
    assert out.shape == (2, 1, 256)
    assert attn.shape == (2, 1, 4, 32)
    # Softmax sums to 1 along N
    assert torch.allclose(attn.sum(dim=-1), torch.ones_like(attn.sum(dim=-1)), atol=1e-5)


# ---------------------------------------------------------------------------
# 3. Pinhole math: round-trip
# ---------------------------------------------------------------------------

def test_pinhole_roundtrip():
    """Project synthetic 3D points → pixel coords + depth → unproject. Recover XYZ."""
    K = torch.tensor([[100.0, 0.0, 32.0], [0.0, 100.0, 32.0], [0.0, 0.0, 1.0]]).reshape(
        1, 1, 3, 3
    )
    H, W = 64, 64

    # Build synthetic depth: a plane at z=2 with a square step at z=1.
    depth = torch.full((1, 1, H, W), 2.0)
    depth[..., 20:40, 20:40] = 1.0

    pcd = depth2fgpcd_batch(depth, K, keepdims=True)         # [1,1,H,W,3]
    # Re-project pcd → pixel coords via K and check Z = depth.
    XYZ = pcd[0, 0]                                          # [H, W, 3]
    fx, fy = K[0, 0, 0, 0], K[0, 0, 1, 1]
    cx, cy = K[0, 0, 0, 2], K[0, 0, 1, 2]
    u = XYZ[..., 0] * fx / XYZ[..., 2] + cx
    v = XYZ[..., 1] * fy / XYZ[..., 2] + cy
    ug, vg = torch.meshgrid(
        torch.arange(W, dtype=torch.float32),
        torch.arange(H, dtype=torch.float32),
        indexing="xy",
    )
    assert torch.allclose(u, ug, atol=1e-4)
    assert torch.allclose(v, vg, atol=1e-4)
    assert torch.allclose(XYZ[..., 2], depth[0, 0])


def test_lift_with_identity_extrinsics_equals_depth2fgpcd():
    K = torch.tensor([[100.0, 0.0, 32.0], [0.0, 100.0, 32.0], [0.0, 0.0, 1.0]]).reshape(
        1, 1, 3, 3
    )
    E = torch.eye(4).reshape(1, 1, 4, 4)
    depth = torch.rand(1, 1, 32, 32) + 0.5
    a = depth2fgpcd_batch(depth, K, keepdims=True)
    b = lift_point_cloud_batch(depth, K, E, keepdims=True)
    assert torch.allclose(a, b)


# ---------------------------------------------------------------------------
# 4. FPS
# ---------------------------------------------------------------------------

def test_fps_picks_endpoints_on_a_line():
    # Points evenly spaced on x-axis. With seed at index 0 (leftmost) and
    # n_samples=2, the second pick should be the rightmost (index N-1).
    N = 20
    pcd = torch.zeros(1, N, 3)
    pcd[0, :, 0] = torch.linspace(0, 1, N)
    idx = fps_pytorch(pcd, n_samples=2)
    assert idx[0, 0].item() == 0
    assert idx[0, 1].item() == N - 1


def test_fps_unique_indices():
    pcd = torch.randn(2, 100, 3)
    idx = fps_pytorch(pcd, n_samples=10)
    for b in range(2):
        assert len(torch.unique(idx[b])) == 10


# ---------------------------------------------------------------------------
# 5. Adapt3R3DEncoder — shape contract (ResNet18, fast / no-download tests)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def encoder_resnet18():
    return Adapt3R3DEncoder(
        output_dim=256, hidden_dim=60, num_points=64,
        intrinsics=_aria_intrinsics_orig(), extrinsics=_identity_extrinsics(),
        image_size_orig=[2016, 1512], image_size_input=[64, 64],
        backbone="resnet18", dummy_depth=1.0,
    )


def test_encoder_shape_contract(encoder_resnet18):
    x = _make_rgbd(B=2, T=1, I=1, H=64, W=64, depth=1.0)
    out = encoder_resnet18(x)
    assert out.shape == (2, 1, 256)


def test_encoder_T_I_flatten():
    """T*I should flatten into the second output dim."""
    enc = Adapt3R3DEncoder(
        output_dim=128, hidden_dim=60, num_points=16,
        intrinsics=_aria_intrinsics_orig(), extrinsics=_identity_extrinsics(),
        image_size_orig=[2016, 1512], image_size_input=[64, 64],
        backbone="resnet18", dummy_depth=1.0,
    )
    x = _make_rgbd(B=2, T=2, I=3, H=64, W=64)
    out = enc(x)
    assert out.shape == (2, 6, 128)


def test_encoder_intrinsics_rescale():
    """K should be rescaled by image_size_input / image_size_orig."""
    enc = Adapt3R3DEncoder(
        output_dim=64, hidden_dim=60, num_points=4,
        intrinsics=_aria_intrinsics_orig(), extrinsics=_identity_extrinsics(),
        image_size_orig=[2016, 1512], image_size_input=[360, 360],
        backbone="resnet18", dummy_depth=1.0,
    )
    K = enc.K.squeeze().numpy()
    assert math.isclose(K[0, 0], 877.96474921 * 360 / 2016, rel_tol=1e-5)
    assert math.isclose(K[1, 1], 877.96474921 * 360 / 1512, rel_tol=1e-5)
    assert math.isclose(K[0, 2], 989.30439345 * 360 / 2016, rel_tol=1e-5)
    assert math.isclose(K[1, 2], 744.17180392 * 360 / 1512, rel_tol=1e-5)


def test_encoder_freezes_backbone(encoder_resnet18):
    for p in encoder_resnet18.backbone.parameters():
        assert not p.requires_grad
    # Other modules should be trainable.
    assert any(p.requires_grad for p in encoder_resnet18.rgb_proj.parameters())
    assert any(p.requires_grad for p in encoder_resnet18.attention_extractor.parameters())


def test_encoder_finetune_param_groups():
    enc = Adapt3R3DEncoder(
        output_dim=64, hidden_dim=60, num_points=4,
        intrinsics=_aria_intrinsics_orig(), extrinsics=_identity_extrinsics(),
        image_size_orig=[2016, 1512], image_size_input=[64, 64],
        backbone="resnet18", finetune_backbone=True, dummy_depth=1.0,
    )
    groups = enc.get_finetune_param_groups(base_lr=1e-4, backbone_lr_scale=0.1)
    assert len(groups) == 2
    assert math.isclose(groups[0]["lr"], 1e-5)
    assert math.isclose(groups[1]["lr"], 1e-4)
    assert groups[0]["name"].endswith("backbone")
    assert len(groups[0]["params"]) > 0
    assert len(groups[1]["params"]) > 0


def test_encoder_fps_clamp_warns():
    """num_points > spatial-grid should warn and not crash."""
    enc = Adapt3R3DEncoder(
        output_dim=64, hidden_dim=60, num_points=10000,
        intrinsics=_aria_intrinsics_orig(), extrinsics=_identity_extrinsics(),
        image_size_orig=[2016, 1512], image_size_input=[64, 64],
        backbone="resnet18", dummy_depth=1.0,
    )
    x = _make_rgbd(B=1, H=64, W=64)
    with pytest.warns(UserWarning, match="clamping"):
        out = enc(x)
    assert out.shape == (1, 1, 64)


def test_encoder_gradient_flow():
    """Loss.backward() on the output should produce grads on the trainable parts."""
    enc = Adapt3R3DEncoder(
        output_dim=64, hidden_dim=60, num_points=16,
        intrinsics=_aria_intrinsics_orig(), extrinsics=_identity_extrinsics(),
        image_size_orig=[2016, 1512], image_size_input=[64, 64],
        backbone="resnet18", dummy_depth=1.0,
    )
    x = _make_rgbd(B=2, H=64, W=64)
    enc.train()
    out = enc(x)
    out.sum().backward()
    # rgb_proj and attention_extractor weights must have grads.
    assert enc.rgb_proj.weight.grad is not None
    assert enc.rgb_proj.weight.grad.abs().sum().item() > 0
    # Backbone is frozen → no grad.
    for p in enc.backbone.parameters():
        assert p.grad is None or p.grad.abs().sum().item() == 0


def test_encoder_return_internals(encoder_resnet18):
    # ResNet18 at H=W=64 has stride 32 → 2x2 = 4 spatial points; FPS will clamp to 4.
    x = _make_rgbd(B=2, H=64, W=64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out, internals = encoder_resnet18(x, return_internals=True)
    assert out.shape == (2, 1, 256)
    for key in ("token", "feat_grid", "pcd_full", "pcd_ds", "pcd_fps", "feat_fps", "fps_idx", "attn"):
        assert key in internals
    assert internals["token"].shape == out.shape
    assert internals["pcd_full"].shape[:2] == (2, 1)
    assert internals["pcd_full"].shape[-1] == 3
    # FPS clamped to spatial-grid size (4) in this small-input case.
    assert internals["pcd_fps"].shape == (2, 1, 4, 3)
    assert internals["attn"].shape == (2, 1, 4, 4)   # B, F, num_heads, N_points


# ---------------------------------------------------------------------------
# 6. Adapt3R3DEncoder — DINO backbone (downloads ~85 MB on first run)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_encoder_dino_shape_contract():
    enc = Adapt3R3DEncoder(
        output_dim=256, hidden_dim=60, num_points=64,
        intrinsics=_aria_intrinsics_orig(), extrinsics=_identity_extrinsics(),
        image_size_orig=[2016, 1512], image_size_input=[360, 360],
        backbone="dino", dino_img_size=224, dummy_depth=1.0,
    )
    x = _make_rgbd(B=1, H=360, W=360)
    out = enc(x)
    assert out.shape == (1, 1, 256)
    # DINO ViT-S/14 at 224 input → 16x16 = 256 spatial points.
    assert enc.feat_dim == 384
