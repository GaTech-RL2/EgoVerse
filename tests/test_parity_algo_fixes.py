"""
Unit tests for the D02 parity fixes in ``egomimic/algo/diffusion/algo.py``
(DFoT port-parity checklist).

D02: ``forward_eval`` and ``_inference_step_chunk`` applied
``sampled[..., action_slice]`` to the sampled diffusion bundle. For 5-D pixel
bundles ``(B, T, C, H, W)`` the ellipsis lands the slice on the LAST axis
(image width), so the "actions" handed to unnormalize / the env were denoised
pixel columns. The fix (``DFoT._extract_bundle_actions``) slices the channel
axis explicitly and decodes the action planes to an action vector by
global-avg-pool, mirroring the validated ``_inference_step_pixel_policy``
decode; the legacy 3-D ``(B, T, D)`` trailing-axis path stays byte-identical.

Also guards the new ``trainer/default.yaml`` ``gradient_clip_val`` knob (D09):
the key must exist (so the parity recipe can override it) and must default to
null = Lightning's current no-clipping behavior.

Pure torch, CPU-only, no datasets.
"""

from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import torch
import yaml

from egomimic.algo.diffusion.algo import DFoT

REPO_ROOT = Path(__file__).resolve().parents[1]

# 3 RGB + 2 broadcast action planes, as in dfot_pushshapes_pixel_policy.yaml.
ACTION_SLICE = slice(3, 5)


def _pixel_bundle(B=2, T=9, C=5, H=8, W=8):
    """Bundle whose action planes are constant per (b, t, channel) — the
    broadcast-plane training encoding — so the avg-pool decode is exact."""
    torch.manual_seed(0)
    bundle = torch.randn(B, T, C, H, W)
    acts = torch.randn(B, T, C - 3)
    bundle[:, :, 3:] = acts[..., None, None].expand(B, T, C - 3, H, W)
    return bundle, acts


# ---- _extract_bundle_actions: 5-D pixel bundles ------------------------- #


def test_5d_bundle_slices_channel_axis_and_pools():
    bundle, acts = _pixel_bundle()
    out = DFoT._extract_bundle_actions(bundle, ACTION_SLICE, action_dim=2)
    assert out.shape == (2, 9, 2)
    assert torch.allclose(out, acts, atol=1e-6)


def test_5d_old_ellipsis_expression_sliced_width():
    # Regression doc for the D02 repro: the pre-fix expression hits the
    # LAST axis (W), not the channel axis.
    bundle, _ = _pixel_bundle(H=8, W=8)
    old = bundle[..., ACTION_SLICE]
    assert old.shape == (2, 9, 5, 8, 2)  # width columns 3:5 — wrong
    new = DFoT._extract_bundle_actions(bundle, ACTION_SLICE, action_dim=2)
    assert new.shape != old.shape


def test_5d_truncates_to_action_dim():
    # C_a may exceed action_dim; decode keeps the leading action_dim
    # components, matching _inference_step_pixel_policy's `[:, :A]`.
    bundle, acts = _pixel_bundle(C=6)  # 3 RGB + 3 action planes
    out = DFoT._extract_bundle_actions(bundle, slice(3, 6), action_dim=2)
    assert out.shape == (2, 9, 2)
    assert torch.allclose(out, acts[..., :2], atol=1e-6)


# ---- _extract_bundle_actions: legacy 3-D bundles ------------------------ #


def test_3d_bundle_byte_identical_to_legacy_slice():
    torch.manual_seed(1)
    bundle = torch.randn(3, 9, 7)
    sl = slice(5, 7)
    out = DFoT._extract_bundle_actions(bundle, sl)
    assert torch.equal(out, bundle[..., sl])


def test_3d_bundle_ignores_action_dim():
    # action_dim must never alter the legacy 1-D-bundle path: action_slice
    # already defines the action span there.
    torch.manual_seed(2)
    bundle = torch.randn(2, 4, 6)
    sl = slice(2, 6)
    out = DFoT._extract_bundle_actions(bundle, sl, action_dim=1)
    assert torch.equal(out, bundle[..., sl])


# ---- call-site wiring: _inference_step_chunk ---------------------------- #


class _IdentityNormStats:
    def normalize(self, data, emb_id):
        return data

    def unnormalize(self, data, emb_id):
        return data


def _make_chunk_algo(bundle: torch.Tensor) -> DFoT:
    """Minimal DFoT for exercising _inference_step_chunk on CPU: real class,
    fake outer_stage / norm_stats, patched _sample_chunk."""
    algo = object.__new__(DFoT)
    algo.nets = {
        "outer_stage": types.SimpleNamespace(
            action_slice=ACTION_SLICE,
            action_dim=2,
            inner_stage=torch.nn.Linear(1, 1),  # backbone device probe only
        )
    }
    algo.norm_stats = _IdentityNormStats()
    algo.ac_keys = {"pushshapes_sim": "actions"}
    algo.action_dim = 2
    algo.action_horizon = bundle.shape[1]
    algo._encode_cond = lambda obs, T: None
    algo._sample_chunk = lambda B, T, cond, device: bundle
    return algo


def test_chunk_inference_returns_pooled_action_for_pixel_bundle():
    bundle, acts = _pixel_bundle(B=1, T=4)
    algo = _make_chunk_algo(bundle)
    emb_id = 15  # PUSHSHAPES_SIM
    for t in range(3):
        action = algo._inference_step_chunk({}, t, emb_id)
        assert action.shape == (2,)
        assert action.dtype == np.float32
        # action = avg-pool of frame t's action planes, not pixel columns
        assert np.allclose(action, acts[0, t].numpy(), atol=1e-6)


# ---- trainer/default.yaml: gradient_clip_val knob (D09) ----------------- #


def test_trainer_default_yaml_exposes_gradient_clip_val_null():
    cfg = yaml.safe_load(
        (REPO_ROOT / "egomimic/hydra_configs/trainer/default.yaml").read_text()
    )
    assert "gradient_clip_val" in cfg
    assert cfg["gradient_clip_val"] is None
