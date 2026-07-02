"""
Pure-torch unit tests for the H-Net state-ownership refactor
(StateRowMixin.narrow / merge_ on the inference-state dataclasses).

Covers:
  (a) narrow(mask) reproduces the OLD stages._slice_* semantics (reference
      implementations inlined below, incl. the dim-1 gru_h cases and the
      IsotropicInferenceParams batch_size/max_seqlen metadata rules);
  (b) merge_ round-trip: narrow -> mutate sub -> merge_ -> exactly the masked
      rows changed, unmasked rows untouched (and narrow returns real copies);
  (c) the prev_x rollout scenario: eagerly-allocated prev_x/prev_x_valid,
      sub-level IN-PLACE write, merge_ -> persistent rows updated + valid;
  (d) materialize-on-merge: persistent prev_x None, sub writes a tensor,
      merge_ materializes zeros and writes the rows;
  (e) the always-on post-merge assertion fires on a hand-built violation, and
      the None-persistent + nested-sub-state case raises loudly.

Run:
    python3 -m pytest tests_state_roundtrip.py -x -q
or (no pytest installed):
    python3 tests_state_roundtrip.py
"""

import importlib.util
import os
import sys
import types
from dataclasses import dataclass, fields as dc_fields, is_dataclass
from typing import Optional

import torch

torch.manual_seed(0)

# --------------------------------------------------------------------------- #
# Module loading.
#
# This working copy contains the full egomimic/models/hnet package (all real
# modules: config, blocks, routing, scan_interface, register_interface,
# transformer_router, context, isotropic_builder, residual_mixer, stages) but
# NO egomimic package __init__ scaffolding, so build the package chain by hand
# and load each module by file path under its canonical dotted name. External
# optional deps (flash_attn / mamba_ssm / einops) are try/except'd inside the
# modules themselves — nothing needs stubbing. These tests exercise ONLY the
# state dataclasses, never the nn.Modules.
# --------------------------------------------------------------------------- #

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, "egomimic")


def _ensure_pkg(name):
    if name not in sys.modules:
        mod = types.ModuleType(name)
        mod.__path__ = []  # package marker
        sys.modules[name] = mod


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_ROOT, relpath))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


for _pkg in ("egomimic", "egomimic.models", "egomimic.models.hnet"):
    _ensure_pkg(_pkg)

_load("egomimic.models.hnet.config", "models/hnet/config.py")
blocks = _load("egomimic.models.hnet.blocks", "models/hnet/blocks.py")
routing = _load("egomimic.models.hnet.routing", "models/hnet/routing.py")
scan_interface = _load(
    "egomimic.models.hnet.scan_interface", "models/hnet/scan_interface.py"
)
register_interface = _load(
    "egomimic.models.hnet.register_interface", "models/hnet/register_interface.py"
)
transformer_router = _load(
    "egomimic.models.hnet.transformer_router", "models/hnet/transformer_router.py"
)
_load("egomimic.models.hnet.context", "models/hnet/context.py")
_load("egomimic.models.hnet.isotropic_builder", "models/hnet/isotropic_builder.py")
_load("egomimic.models.hnet.residual_mixer", "models/hnet/residual_mixer.py")
stages = _load("egomimic.models.hnet.stages", "models/hnet/stages.py")

KVCache = blocks.KVCache
MambaCache = blocks.MambaCache
IsotropicInferenceParams = blocks.IsotropicInferenceParams
StateRowMixin = blocks.StateRowMixin
RoutingModuleState = routing.RoutingModuleState
DeChunkState = routing.DeChunkState
ScanRouterState = scan_interface.ScanRouterState
ChunkScanEncoderState = scan_interface.ChunkScanEncoderState
SplineUpsamplerState = scan_interface.SplineUpsamplerState
MSublatentDeChunkerState = scan_interface.MSublatentDeChunkerState
RegisterTrunkState = register_interface.RegisterTrunkState
PhaseSummaryRouterState = transformer_router.PhaseSummaryRouterState
EncoderDecoderStageState = stages.EncoderDecoderStageState
ChunkerStageState = stages.ChunkerStageState
ComputeStageState = stages.ComputeStageState

try:
    import pytest

    _raises = pytest.raises
except ImportError:  # no-pytest fallback
    from contextlib import contextmanager

    @contextmanager
    def _raises(exc):
        try:
            yield
        except exc:
            return
        raise AssertionError(f"expected {exc.__name__} to be raised")


# --------------------------------------------------------------------------- #
# Synthetic-state builders. Every field is filled with distinguishable random
# tensors so a dropped/mixed-up field cannot cancel out.
# --------------------------------------------------------------------------- #

B, D, M_SUB = 5, 8, 4
MASK = torch.tensor([True, False, True, True, False])
K = int(MASK.sum())


def _f(*shape):
    return torch.randn(*shape)


def _b(*shape):
    return torch.rand(*shape) > 0.5


def _l(*shape):
    return torch.randint(0, 97, shape)


def make_kv(b=B):
    return KVCache(k=_f(b, 2, 6, 4), v=_f(b, 2, 6, 4), offsets=_l(b))


def make_mamba(b=B):
    return MambaCache(conv_state=_f(b, 6, 3), ssm_state=_f(b, 2, 4, 4))


def make_iso(b=B):
    return IsotropicInferenceParams(
        layer_caches={0: make_kv(b), 3: make_mamba(b)}, max_seqlen=64, batch_size=b
    )


def make_routing(b=B):
    return RoutingModuleState(has_seen_tokens=_b(b), last_hidden_state=_f(b, D))


def make_scan_router(b=B):
    return ScanRouterState(
        has_seen_tokens=_b(b), gru_h=_f(2, b, 7), mamba_cache=make_mamba(b), h_prev=_f(b, 7)
    )


def make_pser(b=B):
    return PhaseSummaryRouterState(
        has_seen_tokens=_b(b),
        kv=[make_kv(b), make_kv(b)],
        last_cut_pos=_l(b),
        t_idx=_l(b),
        seg_sum=_f(b, D),
        seg_count=_l(b),
    )


def make_scan_enc(b=B):
    return ChunkScanEncoderState(gru_h=_f(1, b, 7), offset=_l(b), buf=_f(b, 10, 7))


def make_spline(b=B):
    return SplineUpsamplerState(prev_inner_sub=_f(b, M_SUB, 7), have_prev=_b(b), offset=_l(b))


def make_msublatent(b=B):
    return MSublatentDeChunkerState(
        prev_inner_sub=_f(b, M_SUB, 7),
        ema_prev=_f(b, M_SUB * 7),
        offset=_l(b),
        have_prev=_b(b),
    )


def make_reg_trunk(b=B):
    return RegisterTrunkState(
        reg_k=[_f(b, 12, 4), _f(b, 12, 4)],
        reg_v=[_f(b, 12, 4), _f(b, 12, 4)],
        reg_fill=_l(b),
        mem_k=[_f(b, 6, 4), _f(b, 6, 4)],
        mem_v=[_f(b, 6, 4), _f(b, 6, 4)],
        mem_fill=_l(b),
        mem_x=_f(b, 6, D),
        offset=_l(b),
        has_seen=_b(b),
    )


def make_compute(b=B):
    return ComputeStageState(main_state=make_iso(b), inner_state=None)


def make_chunker(b=B, router="scan", with_prev_x=True):
    routers = {"scan": make_scan_router, "cossim": make_routing, "pser": make_pser}
    return ChunkerStageState(
        routing_state=routers[router](b),
        inner_state=make_compute(b),
        dechunk_state=DeChunkState(last_value=_f(b, D)),
        scan_enc_state=make_scan_enc(b),
        spline_state=make_spline(b),
        reg_trunk_state=make_reg_trunk(b),
        prev_x=(_f(b, D) if with_prev_x else None),
        prev_x_valid=(_b(b) if with_prev_x else None),
        msublatent_state=make_msublatent(b),
    )


def make_encdec(b=B):
    return EncoderDecoderStageState(
        encoder_state=make_iso(b), inner_state=make_chunker(b), decoder_state=make_iso(b)
    )


ALL_BUILDERS = {
    "KVCache": make_kv,
    "MambaCache": make_mamba,
    "IsotropicInferenceParams": make_iso,
    "RoutingModuleState": make_routing,
    "ScanRouterState": make_scan_router,
    "PhaseSummaryRouterState": make_pser,
    "ChunkScanEncoderState": make_scan_enc,
    "SplineUpsamplerState": make_spline,
    "MSublatentDeChunkerState": make_msublatent,
    "RegisterTrunkState": make_reg_trunk,
    "ComputeStageState": make_compute,
    "ChunkerStageState": make_chunker,
    "EncoderDecoderStageState": make_encdec,
}


# --------------------------------------------------------------------------- #
# Introspection helpers: flatten any state into {path: tensor}.
# --------------------------------------------------------------------------- #


def flatten(x, prefix=""):
    out = {}
    if x is None:
        return out
    if isinstance(x, torch.Tensor):
        out[prefix] = x
        return out
    if isinstance(x, dict):
        for k, v in x.items():
            out.update(flatten(v, f"{prefix}[{k!r}]"))
        return out
    if isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            out.update(flatten(v, f"{prefix}[{i}]"))
        return out
    if is_dataclass(x):
        for f in dc_fields(x):
            out.update(flatten(getattr(x, f.name), f"{prefix}.{f.name}"))
        return out
    if isinstance(x, (int, float, bool, str)):
        return out  # metadata scalars: compared separately
    raise TypeError(f"flatten: unexpected type {type(x)} at {prefix!r}")


def snapshot(x):
    return {p: t.clone() for p, t in flatten(x).items()}


# The dim-1 exception set, straight from the OLD stages helpers:
# _slice_scan_enc_state / _slice_routing_state sliced gru_h as [:, mask].
def _is_dim1(path):
    return path.endswith(".gru_h")


def ref_narrow_flat(state, mask):
    """OLD-slicing reference on the flattened view: every tensor is sliced on
    dim 0 with .clone(), EXCEPT GRU hiddens (dim 1)."""
    return {
        p: (t[:, mask].clone() if _is_dim1(p) else t[mask].clone())
        for p, t in flatten(state).items()
    }


# Explicit reference impls (verbatim ports of the OLD stages helpers) for the
# trickiest classes, so the generic flat rule above is itself cross-checked.


def ref_slice_scan_router(rs, mask):
    return dict(
        has_seen_tokens=rs.has_seen_tokens[mask].clone(),
        gru_h=(rs.gru_h[:, mask].clone() if rs.gru_h is not None else None),
        mamba_conv=(rs.mamba_cache.conv_state[mask].clone()),
        mamba_ssm=(rs.mamba_cache.ssm_state[mask].clone()),
        h_prev=(rs.h_prev[mask].clone() if rs.h_prev is not None else None),
    )


def ref_slice_scan_enc(s, mask):
    return dict(
        gru_h=s.gru_h[:, mask].clone(),
        offset=s.offset[mask].clone(),
        buf=s.buf[mask].clone(),
    )


def ref_slice_iso(params, mask):
    out = {}
    for k, c in params.layer_caches.items():
        if isinstance(c, KVCache):
            out[k] = (c.k[mask].clone(), c.v[mask].clone(), c.offsets[mask].clone())
        else:
            out[k] = (c.conv_state[mask].clone(), c.ssm_state[mask].clone())
    return out


# --------------------------------------------------------------------------- #
# (a) narrow == old slicing semantics
# --------------------------------------------------------------------------- #


def test_narrow_matches_old_slicing_all_classes():
    for name, build in ALL_BUILDERS.items():
        st = build()
        sub = st.narrow(MASK)
        got = flatten(sub)
        want = ref_narrow_flat(st, MASK)
        assert got.keys() == want.keys(), (
            f"{name}: narrow dropped/added fields: "
            f"{set(got) ^ set(want)}"
        )
        for p in want:
            assert torch.equal(got[p], want[p]), f"{name}: mismatch at {p}"
            assert got[p].dtype == want[p].dtype, f"{name}: dtype changed at {p}"


def test_narrow_scan_router_explicit_reference():
    rs = make_scan_router()
    sub = rs.narrow(MASK)
    ref = ref_slice_scan_router(rs, MASK)
    assert torch.equal(sub.has_seen_tokens, ref["has_seen_tokens"])
    assert torch.equal(sub.gru_h, ref["gru_h"])  # (num_layers, K, H): dim-1 slice
    assert sub.gru_h.shape == (2, K, 7)
    assert torch.equal(sub.mamba_cache.conv_state, ref["mamba_conv"])
    assert torch.equal(sub.mamba_cache.ssm_state, ref["mamba_ssm"])
    assert torch.equal(sub.h_prev, ref["h_prev"])
    # Optional fields None on both sides stay None.
    rs2 = ScanRouterState(has_seen_tokens=_b(B), gru_h=None, mamba_cache=None, h_prev=None)
    sub2 = rs2.narrow(MASK)
    assert sub2.gru_h is None and sub2.mamba_cache is None and sub2.h_prev is None


def test_narrow_scan_encoder_explicit_reference():
    s = make_scan_enc()
    sub = s.narrow(MASK)
    ref = ref_slice_scan_enc(s, MASK)
    assert torch.equal(sub.gru_h, ref["gru_h"]) and sub.gru_h.shape == (1, K, 7)
    assert torch.equal(sub.offset, ref["offset"])
    assert torch.equal(sub.buf, ref["buf"])


def test_narrow_iso_params_metadata():
    iso = make_iso()
    sub = iso.narrow(MASK)
    # OLD _slice_iso_params: max_seqlen copied, batch_size = mask.sum().
    assert sub.max_seqlen == iso.max_seqlen == 64
    assert sub.batch_size == K
    assert set(sub.layer_caches) == set(iso.layer_caches)
    ref = ref_slice_iso(iso, MASK)
    assert torch.equal(sub.layer_caches[0].k, ref[0][0])
    assert torch.equal(sub.layer_caches[0].v, ref[0][1])
    assert torch.equal(sub.layer_caches[0].offsets, ref[0][2])
    assert torch.equal(sub.layer_caches[3].conv_state, ref[3][0])
    assert torch.equal(sub.layer_caches[3].ssm_state, ref[3][1])


def test_narrow_is_deep_copy():
    for name, build in ALL_BUILDERS.items():
        st = build()
        before = snapshot(st)
        sub = st.narrow(MASK)
        for t in flatten(sub).values():  # mutate every sub leaf
            if t.dtype == torch.bool:
                t.logical_not_()
            elif t.is_floating_point():
                t.add_(1.0)
            else:
                t.add_(1)
        after = flatten(st)
        for p in before:
            assert torch.equal(after[p], before[p]), (
                f"{name}: narrow aliased persistent tensor at {p}"
            )


# --------------------------------------------------------------------------- #
# (b) merge_ round-trip
# --------------------------------------------------------------------------- #


def _mutate_all(state):
    for t in flatten(state).values():
        if t.dtype == torch.bool:
            t.logical_not_()
        elif t.is_floating_point():
            t.add_(1.0)
        else:
            t.add_(1)


def test_merge_roundtrip_all_classes():
    for name, build in ALL_BUILDERS.items():
        st = build()
        before = snapshot(st)
        sub = st.narrow(MASK)
        _mutate_all(sub)
        st.merge_(sub, MASK)
        after = flatten(st)
        sub_flat = flatten(sub)
        for p in before:
            if _is_dim1(p):
                assert torch.equal(after[p][:, MASK], sub_flat[p]), f"{name} {p}: masked cols not written"
                assert torch.equal(after[p][:, ~MASK], before[p][:, ~MASK]), f"{name} {p}: unmasked cols changed"
            else:
                assert torch.equal(after[p][MASK], sub_flat[p]), f"{name} {p}: masked rows not written"
                assert torch.equal(after[p][~MASK], before[p][~MASK]), f"{name} {p}: unmasked rows changed"
            assert after[p].dtype == before[p].dtype, f"{name} {p}: dtype changed by merge_"
        # metadata untouched by merge_ (old _scatter_iso_params behavior)
        if name == "IsotropicInferenceParams":
            assert st.max_seqlen == 64 and st.batch_size == B


def test_merge_is_in_place():
    st = make_chunker()
    ids_before = {p: id(t) for p, t in flatten(st).items()}
    sub = st.narrow(MASK)
    _mutate_all(sub)
    st.merge_(sub, MASK)
    ids_after = {p: id(t) for p, t in flatten(st).items()}
    assert ids_before == ids_after, "merge_ replaced persistent tensors instead of writing in place"


def test_wrapper_dict_state_roundtrip():
    # PerEmbodimentStage-style dict state through the thin wrappers.
    state = {"human": make_compute(), "robot": make_compute()}
    before = snapshot(state)
    sub = stages._slice_state(state, MASK)
    assert set(sub) == {"human", "robot"}
    _mutate_all(sub)
    stages._scatter_state(state, sub, MASK)
    after = flatten(state)
    sub_flat = flatten(sub)
    for p in before:
        assert torch.equal(after[p][MASK], sub_flat[p])
        assert torch.equal(after[p][~MASK], before[p][~MASK])
    # None passthrough
    assert stages._slice_state(None, MASK) is None
    stages._scatter_state(None, None, MASK)  # no-op, no raise
    # Unknown types still fail loudly.
    with _raises(TypeError):
        stages._slice_state(object(), MASK)
    with _raises(TypeError):
        stages._scatter_state(object(), object(), MASK)


# --------------------------------------------------------------------------- #
# (c) the prev_x scenario (the bug this refactor fixes)
# --------------------------------------------------------------------------- #


def test_prev_x_eager_roundtrip():
    st = make_chunker(with_prev_x=True)
    # Eager-allocation state as _allocate now builds it: zeros + all-invalid.
    st.prev_x = torch.zeros(B, D)
    st.prev_x_valid = torch.zeros(B, dtype=torch.bool)

    sub = st.narrow(MASK)
    assert torch.equal(sub.prev_x, torch.zeros(K, D))
    assert not sub.prev_x_valid.any()

    # What _step_first_token now does at the sub level: strictly IN PLACE.
    x = torch.randn(K, D)
    sub.prev_x.copy_(x)
    sub.prev_x_valid.fill_(True)

    st.merge_(sub, MASK)
    assert st.prev_x_valid[MASK].all(), "write-back of prev_x_valid dropped"
    assert not st.prev_x_valid[~MASK].any()
    assert torch.equal(st.prev_x[MASK], x), "write-back of prev_x dropped"
    assert torch.equal(st.prev_x[~MASK], torch.zeros(B - K, D))

    # Second rollout round: the sub now sees the bootstrapped carry.
    sub2 = st.narrow(MASK)
    assert sub2.prev_x_valid.all()
    assert torch.equal(sub2.prev_x, x)


def test_prev_x_read_semantics_torch_where():
    # Mirrors the _step_first_token read: invalid rows take no_prev.
    prev_x = torch.randn(K, D)
    valid = torch.tensor([True, False, True]).view(-1, 1)
    no_prev = torch.randn(D)
    got = torch.where(valid, prev_x, no_prev.expand_as(prev_x))
    assert torch.equal(got[0], prev_x[0])
    assert torch.equal(got[1], no_prev)
    assert torch.equal(got[2], prev_x[2])


# --------------------------------------------------------------------------- #
# (d) materialize-on-merge (legacy states with prev_x=None)
# --------------------------------------------------------------------------- #


def test_prev_x_materialize_on_merge():
    st = make_chunker(with_prev_x=False)
    assert st.prev_x is None and st.prev_x_valid is None
    sub = st.narrow(MASK)
    assert sub.prev_x is None  # None-on-both-sides passthrough

    # Legacy sub-level write path (state.prev_x was None): fresh tensors.
    x = torch.randn(K, D)
    sub.prev_x = x.clone()
    sub.prev_x_valid = torch.ones(K, dtype=torch.bool)

    st.merge_(sub, MASK)  # old code silently DROPPED this write-back
    assert st.prev_x is not None and st.prev_x.shape == (B, D)
    assert st.prev_x.dtype == x.dtype
    assert torch.equal(st.prev_x[MASK], x)
    assert torch.equal(st.prev_x[~MASK], torch.zeros(B - K, D))
    assert st.prev_x_valid is not None and st.prev_x_valid.dtype == torch.bool
    assert st.prev_x_valid[MASK].all() and not st.prev_x_valid[~MASK].any()


def test_materialize_dim1_field():
    # dim-1 analogue: gru_h lazily created on the sub side.
    rs = ScanRouterState(has_seen_tokens=_b(B), gru_h=None, mamba_cache=None, h_prev=None)
    sub = rs.narrow(MASK)
    sub.gru_h = torch.randn(2, K, 7)
    rs.merge_(sub, MASK)
    assert rs.gru_h is not None and rs.gru_h.shape == (2, B, 7)
    assert torch.equal(rs.gru_h[:, MASK], sub.gru_h)
    assert torch.equal(rs.gru_h[:, ~MASK], torch.zeros(2, B - K, 7))


def test_none_persistent_nested_state_raises():
    # None persistent + nested sub-state must fail loudly (naming the field),
    # not silently drop (old behavior) nor materialize garbage.
    rs = ScanRouterState(has_seen_tokens=_b(B), gru_h=None, mamba_cache=None, h_prev=None)
    sub = rs.narrow(MASK)
    sub.mamba_cache = make_mamba(K)
    with _raises(AssertionError):
        rs.merge_(sub, MASK)
    try:
        rs2 = ScanRouterState(has_seen_tokens=_b(B), gru_h=None, mamba_cache=None, h_prev=None)
        sub2 = rs2.narrow(MASK)
        sub2.mamba_cache = make_mamba(K)
        rs2.merge_(sub2, MASK)
    except AssertionError as e:
        assert "mamba_cache" in str(e)


# --------------------------------------------------------------------------- #
# (e) post-merge assertion fires on a hand-built violation
# --------------------------------------------------------------------------- #


@dataclass
class _MetaHole(StateRowMixin):
    """merge_ skips _META_FIELDS by contract, so a sub-only meta value is a
    hand-built violation the post-merge invariant must catch."""

    _META_FIELDS = ("meta",)

    a: Optional[torch.Tensor] = None
    meta: Optional[int] = None


def test_post_merge_assertion_fires():
    st = _MetaHole(a=torch.randn(B, 3), meta=None)
    sub = st.narrow(MASK)
    sub.meta = 7  # non-None on sub, None (and untouched) on persistent
    with _raises(AssertionError):
        st.merge_(sub, MASK)


def test_unknown_field_type_fails_loudly():
    @dataclass
    class _Odd(StateRowMixin):
        weird: object = None

    st = _Odd(weird=object())
    with _raises(TypeError):
        st.narrow(MASK)


# --------------------------------------------------------------------------- #
# no-pytest fallback runner
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import traceback

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    total = len(fns)
    print(f"{total - failed}/{total} passed")
    sys.exit(1 if failed else 0)
