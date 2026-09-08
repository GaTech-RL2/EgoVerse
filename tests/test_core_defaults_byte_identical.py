"""Byte-identical default-construction proof for the three BC cores.

Distilled from the session's fixed-seed reproducibility proofs (DESIGN.md
amendment 6.5). Guards the DESIGN.md step-6 relocation of the cores to their
role homes (``egomimic.models.cores.{lstm_core,transformer_core,hnet_core}``)
and any future edit: each core's DEFAULT constructor must be RNG-deterministic
(two fixed-seed builds produce ``torch.equal`` parameter tensors) and its
structure + a deterministic forward must match the committed reference
fingerprints captured from the pre-move code.

All construction + forward is forced onto CPU so the fingerprints are
device-independent and the asserts hold identically whether pytest runs on a
GPU node or a CPU node.
"""
from __future__ import annotations

import torch

from egomimic.models.cores.hnet_core import HNetCore
from egomimic.models.cores.lstm_core import LSTMCore
from egomimic.models.cores.transformer_core import TransformerCore

INPUT_DIM = 66
B, T = 2, 10

# Reference fingerprints captured (CPU, fixed seeds) from the code at the time
# of the DESIGN.md step-6 relocation. param_count + first_param_sum are exact
# (RNG-deterministic init); out_sum is a deterministic forward checksum.
REFS = {
    "lstm": {
        "param_count": 12280000,
        "first_param_sum": 2.926531295090129,
        "out_shape": (B, T, 1000),
        "out_sum": -2.0081596536934967,
        "out_abs_sum": 244.220398,
    },
    "tx": {
        "param_count": 12106752,
        "first_param_sum": 19.838683611378656,
        "out_shape": (B, T, 448),
        "out_sum": -4.0249506128020585e-06,
        "out_abs_sum": 7163.81734,
    },
    "hnet": {
        "param_count": 12163840,
        "first_param_sum": 8.620598287103348,
        "out_shape": (B, T, 256),
        "out_sum": -72.63621407691971,
        "out_abs_sum": 4101.591158,
    },
}

BUILDERS = {
    "lstm": lambda: LSTMCore(input_dim=INPUT_DIM),
    "tx": lambda: TransformerCore(input_dim=INPUT_DIM),
    "hnet": lambda: HNetCore(input_dim=INPUT_DIM),
}


def _seed(s: int = 0) -> None:
    torch.manual_seed(s)


def _build(name: str):
    _seed(0)
    return BUILDERS[name]().cpu().eval()


def _fixed_input():
    _seed(123)
    return torch.randn(B, T, INPUT_DIM)


def _forward(core):
    with torch.no_grad():
        out, _ = core(_fixed_input())
    return out


# --------------------------------------------------------------------------- #
# (1) Two fixed-seed constructions are BYTE-IDENTICAL (torch.equal on every
#     parameter tensor). This is the core "byte-identical" claim.
# --------------------------------------------------------------------------- #
def test_lstm_default_construction_byte_identical():
    _assert_construction_byte_identical("lstm")


def test_tx_default_construction_byte_identical():
    _assert_construction_byte_identical("tx")


def test_hnet_default_construction_byte_identical():
    _assert_construction_byte_identical("hnet")


def _assert_construction_byte_identical(name: str):
    a = dict(_build(name).named_parameters())
    b = dict(_build(name).named_parameters())
    assert a.keys() == b.keys(), f"{name}: param key set drifted between builds"
    for k in a:
        assert torch.equal(a[k], b[k]), (
            f"{name}: param {k!r} not byte-identical across two fixed-seed builds"
        )


# --------------------------------------------------------------------------- #
# (2) Structure + deterministic forward match the committed reference
#     fingerprints (regression guard for the relocation + any future edit).
# --------------------------------------------------------------------------- #
def test_lstm_matches_reference_fingerprint():
    _assert_matches_reference("lstm")


def test_tx_matches_reference_fingerprint():
    _assert_matches_reference("tx")


def test_hnet_matches_reference_fingerprint():
    _assert_matches_reference("hnet")


def _assert_matches_reference(name: str):
    ref = REFS[name]
    core = _build(name)
    n = sum(p.numel() for p in core.parameters())
    assert n == ref["param_count"], (
        f"{name}: param_count {n} != reference {ref['param_count']}"
    )
    first_sum = float(next(core.parameters()).detach().double().sum().item())
    assert first_sum == ref["first_param_sum"], (
        f"{name}: first_param_sum {first_sum!r} != reference "
        f"{ref['first_param_sum']!r} (init RNG changed)"
    )
    out = _forward(core)
    assert tuple(out.shape) == ref["out_shape"], (
        f"{name}: out shape {tuple(out.shape)} != {ref['out_shape']}"
    )
    # Primary forward guard: the ABSOLUTE sum. It does not cancel, so it is
    # stable to ~1e-8 relative across thread counts and BLAS reduction orders,
    # while remaining sensitive to any real change in the forward.
    out_abs_sum = float(out.detach().double().abs().sum().item())
    assert abs(out_abs_sum - ref["out_abs_sum"]) <= 1e-7 * ref["out_abs_sum"], (
        f"{name}: forward abs-checksum {out_abs_sum!r} != reference "
        f"{ref['out_abs_sum']!r}"
    )
    # Secondary: the SIGNED sum, tolerance scaled to the magnitude actually
    # summed. A fixed ~1e-6 bound is meaningless for a near-zero cancelling sum
    # -- tx cancels from abs-sum 7163 down to ~1e-5, so its signed sum moves
    # with thread count alone (-1.21e-05 at 1 thread vs -4.43e-05 at 2) and the
    # old absolute bound made this test fail on any machine but the one the
    # reference was recorded on.
    out_sum = float(out.detach().double().sum().item())
    assert abs(out_sum - ref["out_sum"]) <= 1e-8 * ref["out_abs_sum"], (
        f"{name}: forward checksum {out_sum!r} != reference {ref['out_sum']!r}"
    )


# --------------------------------------------------------------------------- #
# (3) output_dim / hidden_dim parity contract across all three cores
#     (BCRNNPolicy reads core.hidden_dim to size the head).
# --------------------------------------------------------------------------- #
def test_all_cores_expose_matching_hidden_and_output_dim():
    for name in BUILDERS:
        core = _build(name)
        assert core.hidden_dim == core.output_dim, (
            f"{name}: hidden_dim {core.hidden_dim} != output_dim {core.output_dim}"
        )
        out = _forward(core)
        assert out.shape[-1] == core.hidden_dim, (
            f"{name}: forward width {out.shape[-1]} != hidden_dim {core.hidden_dim}"
        )
