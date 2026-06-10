"""K action-chunk stacking (D07) — stage + controller side.

Pins, on CPU with no data:

1. K=1 bit-identity: ``action_chunk_k=1`` (and the omitted default) is the
   SAME code path as the pre-knob stage/controller — identical state_dict,
   identical RNG draws, identical tensors out of training forward and the
   ``_inference_step_pixel_*`` controllers.
2. K>1 policy layout: bundle = ``image_channels + K*action_dim`` planes,
   time-major ``[a0_x, a0_y, a1_x, ...]`` (upstream ``pusht_dataset.py``),
   plus ctor/encode validation of the incoming chunk width.
3. Decode roundtrip: actions -> broadcast planes -> ``decode_action_planes``
   recovers ``(T, K, action_dim)`` exactly (upstream ``unpack_channels``).
4. Controllers commit the full sp_commit*K chunk open-loop per model tick
   (grounded AR, upstream ``df_pusht.py:107-131``): one backbone plan per
   replan, queue pops in between, keyframe obs recorded at env-stride K.
5. Norm-stat tiling: per-step ``(A,)`` stats apply to ``(.., K*A)`` stacked
   tensors via the reshape seam (no K*A stats json) in
   ``process_batch_for_training`` and the chunk-mode plan path.
"""

from __future__ import annotations

import types

import numpy as np
import pytest
import torch
import torch.nn as nn

from egomimic.algo.diffusion.algo import DFoT
from egomimic.algo.diffusion.outer_stages.outer_stage import make_dfot_ctx
from egomimic.algo.diffusion.outer_stages.pixel_obs_action_outer_stage import (
    PixelObsActionDFoTOuterStage,
)
from egomimic.models.diffusion.diffusion.discrete_diffusion import DiscreteDiffusion
from egomimic.rldb.embodiment.embodiment import get_embodiment_id

A = 2
C_IMG = 3
HW = 8
T_EP = 6
TIMESTEPS = 8
IMG_KEY = "front_img_1"
AC_KEY = "actions"
EMB_ID = get_embodiment_id("pushshapes_sim")


class _TinyBackbone(nn.Module):
    """Shape-preserving DiT3D stand-in; parameter feeds the device probe.
    Counts calls so tests can pin one-plan-per-replan."""

    def __init__(self, action_token_dim: int = A):
        super().__init__()
        self.p = nn.Parameter(torch.zeros(1))
        self.action_token_dim = action_token_dim
        self.calls = 0

    def forward(self, x, k, external_cond=None, action=None,
                action_noise_levels=None, **_):
        self.calls += 1
        if action is not None:
            return x * 0.5, action * 0.5
        return x * 0.5


class _IdentityNormStats:
    def normalize(self, data, emb_id):
        return data

    def unnormalize(self, data, emb_id):
        return data


class _PerStepNormStats:
    """Affine stats shaped (A,) per underlying action step. Asserts every
    application sees last dim == A — proving K-stacked tensors reach the
    stats only through the unstack/reshape tiling seam."""

    def __init__(self, mean=(1.0, -2.0), std=(2.0, 4.0)):
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

    def _apply(self, data, fn):
        out = dict(data)
        for kk, v in data.items():
            if kk == AC_KEY and torch.is_tensor(v):
                assert v.shape[-1] == A, (
                    f"(A,)-shaped stats applied to last dim {v.shape[-1]}"
                )
                out[kk] = fn(v)
        return out

    def normalize(self, data, emb_id):
        return self._apply(data, lambda v: (v - self.mean) / self.std)

    def unnormalize(self, data, emb_id):
        return self._apply(data, lambda v: v * self.std + self.mean)

    # process_batch_for_training surface
    def zarr_key_to_keyname(self, key, emb_id):
        return None


def _make_stage(pixel_mode: str = "policy", k: int | None = None, **kw):
    kwargs = dict(
        action_dim=A,
        cond_encoder=None,
        backbone=kw.pop(
            "backbone",
            _TinyBackbone(A * (k or 1) if pixel_mode == "decoupled" else A),
        ),
        diffusion=DiscreteDiffusion(action_dim=A, timesteps=TIMESTEPS),
        image_key=IMG_KEY,
        image_channels=C_IMG,
        image_size=HW,
        frame_sampling="full",
        pixel_mode=pixel_mode,
        head_width=16,
    )
    if k is not None:
        kwargs["action_chunk_k"] = k
    kwargs.update(kw)
    return PixelObsActionDFoTOuterStage(**kwargs)


def _make_algo(stage, norm_stats=None, sp_commit: int = 1) -> DFoT:
    algo = object.__new__(DFoT)
    algo.nets = {"outer_stage": stage}
    algo.norm_stats = norm_stats or _IdentityNormStats()
    algo.ac_keys = {"pushshapes_sim": AC_KEY}
    algo.resolved_ac_keys = {EMB_ID: AC_KEY}
    algo.action_dim = A
    algo.sampler_n_steps = 2
    algo.sp_n_context = 2
    algo.sp_commit = sp_commit
    algo.sp_n_samples = 1
    return algo


def _packed_batch(k: int = 1, seed: int = 0):
    torch.manual_seed(seed)
    imgs = torch.randint(0, 256, (T_EP, C_IMG, HW, HW), dtype=torch.uint8)
    actions = torch.randn(T_EP, A * k)
    batch = {IMG_KEY: imgs, AC_KEY: actions, "_packed": True}
    ctx = make_dfot_ctx(
        is_packed=True,
        action_key=AC_KEY,
        obs={IMG_KEY: imgs},
        cu_seqlens=torch.tensor([0, T_EP], dtype=torch.long),
        max_seqlen=T_EP,
    )
    return batch, ctx


def _obs(seed: int = 0) -> dict:
    torch.manual_seed(seed)
    return {
        IMG_KEY: torch.randint(0, 256, (C_IMG, HW, HW), dtype=torch.uint8),
        "state_agent_obj": torch.tensor([0.25, -0.5, 0.0, 0.0]),
    }


# --------------------------------------------------------------------------- #
# 1. K=1 bit-identity                                                          #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode", ["policy", "regress", "decoupled"])
def test_k1_state_dict_bit_identical(mode):
    torch.manual_seed(0)
    base = _make_stage(mode)            # knob omitted (default 1)
    torch.manual_seed(0)
    explicit = _make_stage(mode, k=1)   # knob explicit
    sd_a, sd_b = base.state_dict(), explicit.state_dict()
    assert sd_a.keys() == sd_b.keys()
    for kk in sd_a:
        assert torch.equal(sd_a[kk], sd_b[kk]), kk
    assert base.action_chunk_k == 1 and explicit.chunk_action_width == A


@pytest.mark.parametrize("mode", ["policy", "regress", "decoupled"])
def test_k1_training_forward_bit_identical(mode):
    outs = []
    for k in (None, 1):
        torch.manual_seed(0)
        stage = _make_stage(mode, k=k)
        batch, ctx = _packed_batch()
        torch.manual_seed(123)
        v = stage.forward(batch, ctx)
        v = v[0] if isinstance(v, tuple) else v
        outs.append((v.detach(), ctx.q_state["x_t"].detach()))
    assert torch.equal(outs[0][0], outs[1][0])
    assert torch.equal(outs[0][1], outs[1][1])


_CONTROLLERS = [
    ("policy", "_inference_step_pixel_policy"),
    ("regress", "_inference_step_pixel_regress"),
    ("decoupled", "_inference_step_pixel_decoupled"),
]


@pytest.mark.parametrize("mode,method", _CONTROLLERS)
def test_k1_controller_bit_identical(mode, method):
    acts = []
    for k in (None, 1):
        torch.manual_seed(0)
        algo = _make_algo(_make_stage(mode, k=k))
        torch.manual_seed(42)
        a0 = getattr(algo, method)(_obs(), 0, EMB_ID)
        torch.manual_seed(43)
        a1 = getattr(algo, method)(_obs(1), 1, EMB_ID)
        acts.append((a0, a1))
    assert np.array_equal(acts[0][0], acts[1][0])
    assert np.array_equal(acts[0][1], acts[1][1])


# --------------------------------------------------------------------------- #
# 2. K>1 policy layout + validation                                            #
# --------------------------------------------------------------------------- #
def test_k_policy_bundle_and_plane_layout():
    K = 3
    stage = _make_stage("policy", k=K)
    assert stage._action_channels == K * A
    assert stage.bundle_shape == (C_IMG + K * A, HW, HW)
    assert stage.action_slice == slice(C_IMG, C_IMG + K * A)
    acts = torch.arange(2 * K * A, dtype=torch.float32).reshape(2, K * A)
    planes = stage._action_to_planes(acts, HW, HW)
    assert planes.shape == (2, K * A, HW, HW)
    # time-major [a0_x, a0_y, a1_x, ...]: channel c broadcasts actions[:, c]
    for c in range(K * A):
        assert torch.equal(planes[:, c], acts[:, c, None, None].expand(2, HW, HW))


def test_k_action_channels_mismatch_raises():
    with pytest.raises(ValueError, match="action_chunk_k"):
        _make_stage("policy", k=3, action_channels=4)


def test_k_encode_validates_incoming_action_width():
    stage = _make_stage("policy", k=3)
    batch, ctx = _packed_batch(k=1)  # actions (T, A): wrong width for K=3
    with pytest.raises(ValueError, match="K-stacked"):
        stage._encode_policy(batch, ctx)


def test_k_encode_policy_builds_2k_plane_bundle():
    K = 3
    stage = _make_stage("policy", k=K)
    batch, ctx = _packed_batch(k=K)
    x_t = stage._encode_policy(batch, ctx)
    assert x_t.shape == (T_EP, C_IMG + K * A, HW, HW)
    x_start = ctx.q_state["x_start"]
    # action planes of the clean bundle == broadcast of the stacked actions
    assert torch.equal(
        x_start[:, C_IMG:],
        stage._action_to_planes(batch[AC_KEY], HW, HW),
    )


def test_k_regress_head_and_decoupled_token_width():
    K = 3
    st = _make_stage("regress", k=K)
    assert st.action_head[-1].out_features == K * A
    # decoupled requires backbone.action_token_dim == K*A
    _make_stage("decoupled", k=K, backbone=_TinyBackbone(K * A))
    with pytest.raises(ValueError, match="action_token_dim"):
        _make_stage("decoupled", k=K, backbone=_TinyBackbone(A))


# --------------------------------------------------------------------------- #
# 3. decode roundtrip                                                          #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("K", [1, 2, 4])
def test_decode_roundtrip_stack_planes_decode(K):
    stage = _make_stage("policy", k=K)
    torch.manual_seed(0)
    acts = torch.randn(T_EP, K * A)
    planes = stage._action_to_planes(acts, HW, HW)
    dec = stage.decode_action_planes(planes)
    assert dec.shape == (T_EP, K, A)
    assert torch.allclose(dec, acts.reshape(T_EP, K, A), atol=1e-6)
    # flattening back recovers the time-major layout exactly
    assert torch.allclose(dec.reshape(T_EP, K * A), acts, atol=1e-6)


# --------------------------------------------------------------------------- #
# 4. controllers commit the full K-chunk open-loop                             #
# --------------------------------------------------------------------------- #
def test_policy_controller_commits_k_chunk_grounded_ar():
    K, COMMIT = 3, 2
    stage = _make_stage("policy", k=K)
    norm = _PerStepNormStats()
    algo = _make_algo(stage, norm_stats=norm, sp_commit=COMMIT)
    bb = stage.inner_stage

    torch.manual_seed(0)
    a0 = algo._inference_step_pixel_policy(_obs(0), 0, EMB_ID)
    assert a0.shape == (A,)
    assert len(algo._pp_queue) == COMMIT * K - 1
    assert len(algo._pp_act) == COMMIT          # one (K*A,) entry per keyframe
    assert all(v.shape == (K * A,) for v in algo._pp_act)
    plan_calls = bb.calls
    assert plan_calls > 0

    # committed sequence == row-wise unnorm of the recorded keyframe chunks,
    # time-major (chunk committed open-loop, no replan mid-chunk)
    flat = torch.stack(algo._pp_act).reshape(COMMIT * K, A)
    expected = norm.unnormalize({AC_KEY: flat}, EMB_ID)[AC_KEY].numpy()
    got = [a0] + list(algo._pp_queue)
    assert np.allclose(np.stack(got), expected, atol=1e-6)

    # mid-chunk ticks pop the queue without touching the model
    for t in range(1, COMMIT * K):
        algo._inference_step_pixel_policy(_obs(t), t, EMB_ID)
    assert bb.calls == plan_calls
    assert len(algo._pp_queue) == 0
    # keyframe obs recorded at env-stride K only: t = 0, 3 (t=6 not seen yet)
    assert len(algo._pp_rgb) == COMMIT

    # next tick replans (and is a keyframe boundary: t == COMMIT*K)
    algo._inference_step_pixel_policy(_obs(99), COMMIT * K, EMB_ID)
    assert bb.calls > plan_calls
    assert len(algo._pp_rgb) == COMMIT + 1
    assert len(algo._pp_act) == 2 * COMMIT


@pytest.mark.parametrize(
    "mode,method,queue",
    [
        ("regress", "_inference_step_pixel_regress", "_pr_queue"),
        ("decoupled", "_inference_step_pixel_decoupled", "_pd_queue"),
    ],
)
def test_regress_decoupled_controllers_commit_k_chunk(mode, method, queue):
    K, COMMIT = 2, 2
    stage = _make_stage(mode, k=K)
    algo = _make_algo(stage, norm_stats=_PerStepNormStats(), sp_commit=COMMIT)
    bb = stage.inner_stage
    torch.manual_seed(0)
    a0 = getattr(algo, method)(_obs(0), 0, EMB_ID)
    assert a0.shape == (A,)
    assert len(getattr(algo, queue)) == COMMIT * K - 1
    plan_calls = bb.calls
    for t in range(1, COMMIT * K):
        a = getattr(algo, method)(_obs(t), t, EMB_ID)
        assert a.shape == (A,)
    assert bb.calls == plan_calls  # whole chunk served from the queue


def test_chunk_inference_unstacks_k_per_frame():
    """_inference_step_chunk: pooled (T, K*A) planes -> T*K env actions in
    time-major order; one plan serves action_horizon*K ticks."""
    K, T = 2, 3
    stage = _make_stage("policy", k=K)
    norm = _PerStepNormStats()
    algo = _make_algo(stage, norm_stats=norm)
    algo.action_horizon = T
    algo._encode_cond = lambda obs, T_: None

    # known constant action planes: frame f, plane c -> 10*f + c
    bundle = torch.zeros(1, T, C_IMG + K * A, HW, HW)
    for f in range(T):
        for c in range(K * A):
            bundle[0, f, C_IMG + c] = 10.0 * f + c
    calls = {"n": 0}

    def _fake_sample_chunk(B, T, cond, device):
        calls["n"] += 1
        return bundle

    algo._sample_chunk = _fake_sample_chunk

    expected_rows = bundle[0, :, C_IMG:].mean(dim=(-2, -1)).reshape(T * K, A)
    expected = norm.unnormalize({AC_KEY: expected_rows}, EMB_ID)[AC_KEY].numpy()
    for t in range(T * K):
        a = algo._inference_step_chunk({}, t, EMB_ID)
        assert a.shape == (A,)
        assert np.allclose(a, expected[t], atol=1e-5)
    assert calls["n"] == 1
    algo._inference_step_chunk({}, T * K, EMB_ID)
    assert calls["n"] == 2  # replan only after all T*K actions are spent


# --------------------------------------------------------------------------- #
# 5. norm-stat tiling across K                                                 #
# --------------------------------------------------------------------------- #
def test_process_batch_tiles_per_step_stats_across_k():
    K = 4
    stage = _make_stage("policy", k=K)
    norm = _PerStepNormStats()
    algo = _make_algo(stage, norm_stats=norm)
    algo.device = torch.device("cpu")

    torch.manual_seed(0)
    acts = torch.randn(T_EP, K * A)
    batch = {
        "pushshapes_sim": {
            AC_KEY: acts.clone(),
            "cu_seqlens": torch.tensor([0, T_EP], dtype=torch.long),
        }
    }
    out = algo.process_batch_for_training(batch)[EMB_ID][AC_KEY]
    expected = (
        (acts.reshape(T_EP, K, A) - norm.mean) / norm.std
    ).reshape(T_EP, K * A)
    assert torch.allclose(out, expected, atol=1e-6)


def test_k1_process_batch_uses_direct_normalize_path():
    """K=1 must NOT route through the reshape seam (same code path)."""
    stage = _make_stage("policy", k=1)
    algo = _make_algo(stage)
    seen = {}

    class _Spy(_PerStepNormStats):
        def normalize(self, data, emb_id):
            seen["last_dim"] = data[AC_KEY].shape[-1]
            return super().normalize(data, emb_id)

    algo.norm_stats = _Spy()
    algo.device = torch.device("cpu")
    batch = {
        "pushshapes_sim": {
            AC_KEY: torch.randn(T_EP, A),
            "cu_seqlens": torch.tensor([0, T_EP], dtype=torch.long),
        }
    }
    algo.process_batch_for_training(batch)
    # direct path: the stats see the raw (T, A) tensor, no (T, 1, A) reshape
    assert seen["last_dim"] == A


def test_chunked_norm_seam_roundtrip():
    K = 3
    stage = _make_stage("policy", k=K)
    norm = _PerStepNormStats()
    algo = _make_algo(stage, norm_stats=norm)
    torch.manual_seed(0)
    acts = torch.randn(5, K * A)
    normed = algo._chunked_action_norm_seam({AC_KEY: acts}, EMB_ID, inverse=False)
    out = algo._chunked_action_norm_seam(normed, EMB_ID, inverse=True)[AC_KEY]
    assert normed[AC_KEY].shape == acts.shape
    assert torch.allclose(out, acts, atol=1e-5)
