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
from egomimic.models.heads.query_decoder import QueryActionDecoder

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


# --------------------------------------------------------------------------- #
# (4) QueryActionDecoder future-perturbation EXACT-ZERO.
#     Perturbing a future CONTEXT feature must leave earlier obs-steps' GMM
#     params BITWISE unchanged (torch.equal) — the causal key-padding mask.
# --------------------------------------------------------------------------- #
def test_query_decoder_future_perturbation_exact_zero():
    d_model = 64
    chunk_len = 8
    per_step = 5 * (2 * 2 + 1)  # num_modes=5, action_dim=2 => 25
    _seed(0)
    dec = QueryActionDecoder(
        d_model=d_model,
        chunk_len=chunk_len,
        per_step=per_step,
        max_window=T,
    ).cpu().eval()

    _seed(3)
    h = torch.randn(B, T, d_model)  # per-step core features h_0..h_{T-1}
    with torch.no_grad():
        out1 = dec(h)  # (B, T, chunk_len*per_step)
        h2 = h.clone()
        h2[:, T - 1] = torch.randn(B, d_model) * 13.0  # perturb LAST context pos
        out2 = dec(h2)
    # Every obs-step k < T-1 must be EXACTLY unchanged (causal mask disallows the
    # future context position T-1 for those obs-steps).
    for k in range(T - 1):
        assert torch.equal(out1[:, k], out2[:, k]), (
            f"query decoder: future-context perturbation changed obs-step k={k} "
            f"(maxdiff {(out1[:, k] - out2[:, k]).abs().max().item():.2e}); "
            f"causal key-padding mask is not exact-zero"
        )
    # Sanity: the LAST obs-step DID change (it legitimately sees position T-1).
    assert not torch.equal(out1[:, T - 1], out2[:, T - 1]), (
        "query decoder: perturbing context T-1 left obs-step T-1 unchanged — "
        "the context is not actually being attended"
    )
