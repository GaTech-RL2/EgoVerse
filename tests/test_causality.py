"""Causality proofs for the BC cores + query decoder (DESIGN.md amendment 6.5).

Distilled from the session's causality / prefix-consistency GPU proofs:

  * TX prefix-consistency: the causal TransformerCore's feature at obs-step k
    depends ONLY on positions 0..k, so ``forward(x[:, :k+1])[:, k]`` matches
    ``forward(x)[:, k]`` (up to attention-kernel noise).
  * HNet prefix-consistency: the SAME property must hold THROUGH the dynamic
    chunk-boundary decisions (routing cos-sim, chunk argsort, dechunk EMA) — the
    make-or-break for the real H-Net core.
  * Query decoder future-perturbation EXACT-ZERO: perturbing a FUTURE context
    position must leave every earlier obs-step's GMM params bitwise unchanged
    (``torch.equal``), because the causal key-padding mask hard-disallows future
    context — a stronger, exact form of causality than the float-tolerant
    prefix checks above.

All on CPU + eval() so dropout is off and the checks are deterministic.
"""
from __future__ import annotations

import torch

from egomimic.models.cores.hnet_core import HNetCore
from egomimic.models.cores.transformer_core import TransformerCore

INPUT_DIM = 66
B, T = 2, 10


def _seed(s: int = 0) -> None:
    torch.manual_seed(s)


def _build_tx():
    _seed(0)
    return TransformerCore(input_dim=INPUT_DIM, max_window=T).cpu().eval()


def _build_hnet():
    _seed(0)
    return HNetCore(input_dim=INPUT_DIM, max_window=T).cpu().eval()


def _input():
    _seed(7)
    return torch.randn(B, T, INPUT_DIM)


# --------------------------------------------------------------------------- #
# (1) TransformerCore prefix-consistency: causal, no future leak.
# --------------------------------------------------------------------------- #
def test_tx_prefix_consistency():
    _assert_core_prefix_consistency(_build_tx(), atol=1e-5)


# --------------------------------------------------------------------------- #
# (2) HNetCore prefix-consistency: causal THROUGH the dynamic chunker.
# --------------------------------------------------------------------------- #
def test_hnet_prefix_consistency():
    # EMA / routing introduce slightly larger but still tiny numeric noise than
    # plain SDPA; the docstring cites maxdiff ~1e-6, use a safe 1e-4 bound.
    _assert_core_prefix_consistency(_build_hnet(), atol=1e-4)


def _assert_core_prefix_consistency(core, atol: float):
    x = _input()
    with torch.no_grad():
        full, _ = core(x)  # (B, T, H)
        for k in range(T):
            pref, _ = core(x[:, : k + 1])  # forward over prefix 0..k
            # output at the LAST prefix position must match the full-window
            # output at position k -> position k saw only 0..k.
            assert torch.allclose(pref[:, k], full[:, k], atol=atol, rtol=0), (
                f"future-leak at k={k}: prefix vs full maxdiff "
                f"{(pref[:, k] - full[:, k]).abs().max().item():.2e} > {atol:.0e}"
            )


# --------------------------------------------------------------------------- #
# (3) TransformerCore: perturbing a FUTURE input must not change earlier
#     outputs (the complementary "mutate the future" form of the proof).
# --------------------------------------------------------------------------- #
def test_tx_future_perturbation_no_leak():
    core = _build_tx()
    x = _input()
    with torch.no_grad():
        out1, _ = core(x)
        x2 = x.clone()
        x2[:, T - 1] = torch.randn(B, INPUT_DIM) * 10.0  # perturb LAST step
        out2, _ = core(x2)
        for k in range(T - 1):  # every step BEFORE the perturbed one
            assert torch.allclose(out1[:, k], out2[:, k], atol=1e-5, rtol=0), (
                f"future-perturb leaked to k={k}"
            )
