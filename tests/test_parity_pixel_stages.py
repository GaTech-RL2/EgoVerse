"""Parity-checklist tests for the pixel outer stages (D03 / D06 / D16).

Pins, on CPU with no data:

1. ``image_range`` knob: ``"01"`` (default) is a literal no-op on the
   historic /255 extraction path; ``"pm1"`` maps to [-1, 1] right after
   extraction and round-trips back through ``from_model_range``.
2. Per-group loss split: all knobs ``None`` leaves ``ctx.precomputed_loss``
   unset (DFoTLoss single-mean path, same RNG draws); setting weights
   produces exactly ``w_img * img_mean + w_act * act_mean`` computed via
   channel-sliced ``diffusion.compute_loss`` calls.
3. D06 rollout fix: the 5-channel policy bundle no longer crashes the
   sliding-window video rollout (3-ch GT context is lifted to a bundle with
   GT-action planes), while the 3-ch video case stays byte-identical to the
   pre-fix algorithm.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from egomimic.algo.diffusion.algo import DFoTLoss
from egomimic.algo.diffusion.outer_stages.outer_stage import make_dfot_ctx
from egomimic.algo.diffusion.outer_stages.pixel_obs_action_outer_stage import (
    PixelObsActionDFoTOuterStage,
)
from egomimic.algo.diffusion.outer_stages.pixel_spatial_outer_stage import (
    PixelSpatialDFoTOuterStage,
)
from egomimic.models.diffusion.diffusion.discrete_diffusion import DiscreteDiffusion

A = 2          # action_dim == action_channels
C_IMG = 3
HW = 16        # >= 11 so torchmetrics SSIM (kernel 11) works in the rollout
T_EP = 8
TIMESTEPS = 16
AC_KEY = "actions"
IMG_KEY = "front_img_1"
EMB_ID = 0


class _TinyBackbone(nn.Module):
    """Shape-preserving deterministic stand-in for the DiT3D."""

    def forward(self, x, t, external_cond=None, cu_seqlens=None, max_seqlen=None):
        return x * 0.5


def _make_diffusion() -> DiscreteDiffusion:
    return DiscreteDiffusion(action_dim=A, timesteps=TIMESTEPS)


def _make_policy_stage(**kw) -> PixelObsActionDFoTOuterStage:
    return PixelObsActionDFoTOuterStage(
        action_dim=A,
        cond_encoder=None,
        backbone=_TinyBackbone(),
        diffusion=_make_diffusion(),
        image_key=IMG_KEY,
        image_channels=C_IMG,
        image_size=HW,
        frame_sampling="full",
        pixel_mode="policy",
        action_channels=A,
        **kw,
    )


def _make_video_stage(**kw) -> PixelSpatialDFoTOuterStage:
    return PixelSpatialDFoTOuterStage(
        action_dim=A,
        cond_encoder=None,
        backbone=_TinyBackbone(),
        diffusion=_make_diffusion(),
        image_key=IMG_KEY,
        image_channels=C_IMG,
        image_size=HW,
        frame_sampling="full",
        **kw,
    )


def _packed_batch(seed: int = 0):
    torch.manual_seed(seed)
    imgs = torch.randint(0, 256, (T_EP, C_IMG, HW, HW), dtype=torch.uint8)
    actions = torch.randn(T_EP, A)
    batch = {IMG_KEY: imgs, AC_KEY: actions, "_packed": True}
    cu = torch.tensor([0, T_EP], dtype=torch.long)
    ctx = make_dfot_ctx(
        is_packed=True,
        action_key=AC_KEY,
        obs={IMG_KEY: imgs},
        cu_seqlens=cu,
        max_seqlen=T_EP,
    )
    return batch, ctx


def _core_has_strategy_override() -> bool:
    sig = inspect.signature(DiscreteDiffusion.compute_loss)
    return "weighting_strategy" in sig.parameters


# --------------------------------------------------------------------------- #
# 1. image_range                                                               #
# --------------------------------------------------------------------------- #
def test_image_range_01_default_is_identity():
    stage = _make_video_stage()  # image_range defaults to "01"
    assert stage.image_range == "01"
    assert stage.model_range_bounds == (0.0, 1.0)

    img = torch.randint(0, 256, (T_EP, C_IMG, HW, HW), dtype=torch.uint8)
    ctx = SimpleNamespace(obs={IMG_KEY: img})
    out = stage._extract_images(ctx)
    assert torch.equal(out, img.float() / 255.0)

    # "01" mode helpers are literal pass-throughs (same tensor object).
    x = torch.rand(4, C_IMG, HW, HW)
    assert stage.to_model_range(x) is x
    assert stage.from_model_range(x) is x


def test_image_range_pm1_roundtrip():
    stage = _make_video_stage(image_range="pm1")
    assert stage.model_range_bounds == (-1.0, 1.0)

    img = torch.randint(0, 256, (T_EP, C_IMG, HW, HW), dtype=torch.uint8)
    ctx = SimpleNamespace(obs={IMG_KEY: img})
    out = stage._extract_images(ctx)
    assert torch.equal(out, (img.float() / 255.0) * 2.0 - 1.0)
    assert out.min() >= -1.0 and out.max() <= 1.0

    # encode -> decode identity (both directions).
    x01 = torch.rand(4, C_IMG, HW, HW)
    assert torch.allclose(stage.from_model_range(stage.to_model_range(x01)), x01,
                          atol=1e-6)
    xpm = torch.rand(4, C_IMG, HW, HW) * 2.0 - 1.0
    assert torch.allclose(stage.to_model_range(stage.from_model_range(xpm)), xpm,
                          atol=1e-6)


def test_image_range_rejects_unknown():
    with pytest.raises(ValueError):
        _make_video_stage(image_range="zscore")


# --------------------------------------------------------------------------- #
# 2. per-group loss split                                                      #
# --------------------------------------------------------------------------- #
def test_group_loss_disabled_by_default_single_mean_path():
    stage = _make_policy_stage()
    assert not stage._group_loss_enabled
    batch, ctx = _packed_batch()

    torch.manual_seed(7)
    v_pred = stage(batch, ctx)

    # Default knobs: the outer stage must NOT precompute a loss — DFoTLoss
    # runs the historic single-mean path on pred_v + q_state.
    assert getattr(ctx, "precomputed_loss", None) is None
    loss = DFoTLoss(stage.diffusion)(batch, ctx)
    expected = stage.diffusion.compute_loss(v_pred, ctx.q_state).mean()
    assert torch.equal(loss, expected)


def test_group_loss_knobs_do_not_change_rng_draws():
    """The split only post-processes v_pred — encode-time RNG (noise levels,
    epsilon) must be identical with and without the knobs."""
    base = _make_policy_stage()
    split = _make_policy_stage(image_loss_weight=0.5, action_loss_weight_group=3.0)

    batch_a, ctx_a = _packed_batch()
    torch.manual_seed(123)
    base(batch_a, ctx_a)

    batch_b, ctx_b = _packed_batch()
    torch.manual_seed(123)
    split(batch_b, ctx_b)

    assert torch.equal(ctx_a.q_state["k"], ctx_b.q_state["k"])
    assert torch.equal(ctx_a.q_state["noise"], ctx_b.q_state["noise"])
    assert torch.equal(batch_a["pred_v"], batch_b["pred_v"])
    assert ctx_b.precomputed_loss is not None


def test_group_loss_split_matches_manual():
    w_img, w_act = 0.7, 2.0
    stage = _make_policy_stage(
        image_loss_weight=w_img, action_loss_weight_group=w_act
    )
    assert stage._group_loss_enabled
    batch, ctx = _packed_batch()

    torch.manual_seed(11)
    v_pred = stage(batch, ctx)
    got = ctx.precomputed_loss

    q = ctx.q_state
    def _sliced(sl):
        qq = dict(q)
        for key in ("x_t", "noise", "x_start"):
            qq[key] = q[key][..., sl, :, :]
        return qq

    img_sl, act_sl = slice(0, C_IMG), slice(C_IMG, C_IMG + A)
    img_loss = stage.diffusion.compute_loss(
        v_pred[..., img_sl, :, :], _sliced(img_sl)).mean()
    act_loss = stage.diffusion.compute_loss(
        v_pred[..., act_sl, :, :], _sliced(act_sl)).mean()
    expected = w_img * img_loss + w_act * act_loss
    assert torch.allclose(got, expected, atol=0, rtol=0)

    # Sanity: the split genuinely differs from the historic single mean.
    single = stage.diffusion.compute_loss(v_pred, q).mean()
    assert not torch.allclose(got, single)

    # DFoTLoss returns the precomputed value untouched.
    assert torch.equal(DFoTLoss(stage.diffusion)(batch, ctx), got)


@pytest.mark.skipif(
    not _core_has_strategy_override(),
    reason="DiscreteDiffusion.compute_loss lacks weighting_strategy override "
           "(cores-agent change not merged yet)",
)
def test_group_loss_strategy_override_passthrough():
    stage = _make_policy_stage(
        image_loss_weighting="uniform", action_loss_weighting="uniform"
    )
    batch, ctx = _packed_batch()
    torch.manual_seed(13)
    v_pred = stage(batch, ctx)

    q = ctx.q_state
    def _sliced(sl):
        qq = dict(q)
        for key in ("x_t", "noise", "x_start"):
            qq[key] = q[key][..., sl, :, :]
        return qq

    img_sl, act_sl = slice(0, C_IMG), slice(C_IMG, C_IMG + A)
    expected = (
        stage.diffusion.compute_loss(
            v_pred[..., img_sl, :, :], _sliced(img_sl),
            weighting_strategy="uniform").mean()
        + stage.diffusion.compute_loss(
            v_pred[..., act_sl, :, :], _sliced(act_sl),
            weighting_strategy="uniform").mean()
    )
    assert torch.allclose(ctx.precomputed_loss, expected, atol=0, rtol=0)


def test_group_loss_rejects_fused_min_snr_override():
    with pytest.raises(ValueError):
        _make_policy_stage(image_loss_weighting="fused_min_snr")
    with pytest.raises(ValueError):
        _make_policy_stage(action_loss_weighting="fused_min_snr")


# --------------------------------------------------------------------------- #
# 3. D06 rollout: 5-ch policy bundle + 3-ch bit-compat                         #
# --------------------------------------------------------------------------- #
def _passthrough_sample_step(diffusion, backbone, *, x, current_levels,
                             next_levels, external_cond=None, eta=0.0,
                             cfg_scale=1.0):
    return x


def _fake_ev():
    return SimpleNamespace(
        image_key=IMG_KEY,
        rollout_steps=6,
        n_context_frames=2,
        rollout_window=4,
        n_chunk_steps=3,
        recon_loss_n_frames=4,
    )


def _fake_algo(stage):
    return SimpleNamespace(
        diffusion=stage.diffusion,
        backbone=stage.inner_stage,
        resolved_ac_keys={EMB_ID: AC_KEY},
    )


def test_rollout_policy_bundle_5ch(monkeypatch):
    from egomimic.models.diffusion import sampling as sampling_mod

    monkeypatch.setattr(sampling_mod, "sample_step", _passthrough_sample_step)

    stage = _make_policy_stage()
    assert stage.bundle_shape == (C_IMG + A, HW, HW)
    batch, _ = _packed_batch()
    ev, algo = _fake_ev(), _fake_algo(stage)

    torch.manual_seed(3)
    res = stage.rollout_video_episode(
        ev, algo, batch, EMB_ID, 0, 0, T_EP, torch.device("cpu")
    )

    # Image-channel-only frames for metrics/panels, decoded to [0, 1].
    assert res.pred_frames.shape == (ev.rollout_steps, C_IMG, HW, HW)
    assert res.pred_frames.min() >= 0.0 and res.pred_frames.max() <= 1.0
    assert res.gt_for_mse.shape == (ev.recon_loss_n_frames, C_IMG, HW, HW)

    # Context frames pass through the (mocked) sampler untouched ->
    # decoded predictions must reproduce the GT context exactly.
    gt01 = batch[IMG_KEY].float() / 255.0
    assert torch.equal(res.pred_frames[:2], gt01[:2])


def test_rollout_policy_context_uses_gt_action_planes():
    stage = _make_policy_stage()
    batch, _ = _packed_batch()
    gt01 = batch[IMG_KEY].float() / 255.0
    ctx_imgs = gt01[:2]

    bundle = stage._make_rollout_context(
        ctx_imgs, batch, _fake_algo(stage), EMB_ID, 0, 0, True
    )
    assert bundle.shape == (2, C_IMG + A, HW, HW)
    assert torch.equal(bundle[:, :C_IMG], ctx_imgs)
    # Action planes broadcast actions[t] (training pairing frame t <-> a_t).
    acts = batch[AC_KEY]
    for t in range(2):
        for c in range(A):
            assert torch.all(bundle[t, C_IMG + c] == acts[t, c])

    # No action key available -> zero planes (still bundle-shaped, no crash).
    bundle0 = stage._make_rollout_context(
        ctx_imgs, {IMG_KEY: batch[IMG_KEY]}, _fake_algo(stage), EMB_ID, 0, 0, True
    )
    assert torch.all(bundle0[:, C_IMG:] == 0)


def test_rollout_3ch_video_bit_compat(monkeypatch):
    """The pre-fix sliding-window algorithm, replicated inline, must produce
    byte-identical frames to the patched one for the 3-ch / image_range="01"
    video case (same RNG draw order, same clamps)."""
    from egomimic.models.diffusion import sampling as sampling_mod

    monkeypatch.setattr(sampling_mod, "sample_step", _passthrough_sample_step)

    stage = _make_video_stage()
    batch, _ = _packed_batch()
    ev, algo = _fake_ev(), _fake_algo(stage)
    device = torch.device("cpu")
    gt01 = batch[IMG_KEY].float() / 255.0

    def _orig_rollout(total_frames, context_frames, window_size):
        from egomimic.models.diffusion.sampling import sample_step, vanilla_schedule
        bundle_shape = stage.bundle_shape
        discrete_ts = int(algo.diffusion.timesteps)
        generated = context_frames.clone()
        while generated.shape[0] < total_frames:
            c = min(generated.shape[0], window_size - 1)
            h = min(total_frames - generated.shape[0], window_size - c)
            T_window = c + h
            context_part = generated[-c:]
            noise_part = torch.randn(h, *bundle_shape, device=device)
            x = torch.cat([context_part, noise_part], dim=0).unsqueeze(0)
            schedule = vanilla_schedule(
                n_steps=ev.n_chunk_steps, T=T_window,
                discrete_timesteps=discrete_ts,
            ).to(device)
            for step_idx in range(schedule.shape[0]):
                schedule[step_idx, :c] = -1
            for s in range(schedule.shape[0] - 1):
                x = sample_step(
                    algo.diffusion, algo.backbone, x=x,
                    current_levels=schedule[s], next_levels=schedule[s + 1],
                    external_cond=None, eta=0.0,
                )
                x[0, :c] = context_part
            new_frames = x[0, c:c + h]
            generated = torch.cat([generated, new_frames.clamp(0, 1)], dim=0)
        return generated[:total_frames]

    T_rollout = min(ev.rollout_steps, T_EP)
    n_ctx = max(1, min(ev.n_context_frames, T_rollout))
    window = min(ev.rollout_window, T_rollout)

    torch.manual_seed(99)
    ref = _orig_rollout(T_rollout, gt01[:n_ctx], window).clamp(0.0, 1.0)

    torch.manual_seed(99)
    res = stage.rollout_video_episode(
        ev, algo, batch, EMB_ID, 0, 0, T_EP, device
    )
    assert torch.equal(res.pred_frames, ref)


def test_rollout_pm1_decodes_to_unit_range(monkeypatch):
    from egomimic.models.diffusion import sampling as sampling_mod

    monkeypatch.setattr(sampling_mod, "sample_step", _passthrough_sample_step)

    stage = _make_policy_stage(image_range="pm1")
    batch, _ = _packed_batch()
    ev, algo = _fake_ev(), _fake_algo(stage)

    torch.manual_seed(5)
    res = stage.rollout_video_episode(
        ev, algo, batch, EMB_ID, 0, 0, T_EP, torch.device("cpu")
    )
    assert res.pred_frames.shape == (ev.rollout_steps, C_IMG, HW, HW)
    assert res.pred_frames.min() >= 0.0 and res.pred_frames.max() <= 1.0
    # gt_for_mse stays in [0, 1] -> metrics compare pred and GT in the SAME range.
    assert res.gt_for_mse.min() >= 0.0 and res.gt_for_mse.max() <= 1.0
    # Context frames round-trip [0,1] -> [-1,1] -> [0,1] through the rollout.
    gt01 = batch[IMG_KEY].float() / 255.0
    assert torch.allclose(res.pred_frames[:2], gt01[:2], atol=1e-6)


if __name__ == "__main__":
    test_image_range_01_default_is_identity()
    test_image_range_pm1_roundtrip()
    test_image_range_rejects_unknown()
    test_group_loss_disabled_by_default_single_mean_path()
    test_group_loss_knobs_do_not_change_rng_draws()
    test_group_loss_split_matches_manual()
    test_group_loss_rejects_fused_min_snr_override()
    print("non-fixture tests: PASS (run pytest for the rollout tests)")
