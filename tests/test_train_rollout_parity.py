"""Train == rollout parity proofs (DESIGN.md amendment 6.5).

The drop-in core contract guarantees that the one-shot teacher-forced
``forward`` over a length-T window produces the SAME per-step features as T
sequential ``init_hidden`` + ``step`` rollout calls (the LSTM gets this from
sharing nn.LSTM weights; the TX / HNet cores from re-running the SAME ``forward``
over the growing buffer). Distilled from the session's train-vs-step parity GPU
proofs:

  * all 3 cores: ``forward(x)[:, k]`` ≈ ``step``-rollout feature at step k.
    growing TransformerCore buffer reproduces the training ``forward`` GMM-param
    row at each obs-step k (the "chunk8 queue" replay).

CPU + eval(); LSTM parity is exact, attention/EMA cores match up to tiny kernel
noise.
"""
from __future__ import annotations

import torch

from egomimic.models.cores.hnet_core import HNetCore
from egomimic.models.cores.lstm_core import LSTMCore
from egomimic.models.cores.transformer_core import TransformerCore

INPUT_DIM = 66
B, T = 2, 10


def _seed(s: int = 0) -> None:
    torch.manual_seed(s)


def _input():
    _seed(11)
    return torch.randn(B, T, INPUT_DIM)


def _rollout_features(core, x):
    """T sequential init_hidden + step calls -> stacked (B, T, H) features.

    Mirrors BCRNNPolicy.step's reinit-then-step order: a FRESH buffer at k=0,
    then one frame appended per step (window length == T == rnn_horizon, so no
    mid-window re-init).
    """
    hidden = core.init_hidden(B, device=x.device)
    feats = []
    for k in range(T):
        h_t, hidden = core.step(x[:, k], hidden)
        feats.append(h_t)
    return torch.stack(feats, dim=1)  # (B, T, H)


def _build(make):
    _seed(0)
    return make().cpu().eval()


# --------------------------------------------------------------------------- #
# (1) LSTM: forward unroll == sequential step (EXACT, shared nn.LSTM weights).
# --------------------------------------------------------------------------- #
def test_lstm_train_rollout_parity_exact():
    core = _build(lambda: LSTMCore(input_dim=INPUT_DIM))
    x = _input()
    with torch.no_grad():
        full, _ = core(x)
        roll = _rollout_features(core, x)
    assert torch.allclose(full, roll, atol=1e-5, rtol=0), (
        f"LSTM train/rollout maxdiff {(full - roll).abs().max().item():.2e}"
    )


# --------------------------------------------------------------------------- #
# (2) TransformerCore: forward window == buffer-replay step rollout.
# --------------------------------------------------------------------------- #
def test_tx_train_rollout_parity():
    core = _build(lambda: TransformerCore(input_dim=INPUT_DIM, max_window=T))
    x = _input()
    with torch.no_grad():
        full, _ = core(x)
        roll = _rollout_features(core, x)
    assert torch.allclose(full, roll, atol=1e-5, rtol=0), (
        f"TX train/rollout maxdiff {(full - roll).abs().max().item():.2e}"
    )


# --------------------------------------------------------------------------- #
# (3) HNetCore: forward window == buffer-replay step rollout (through chunker).
# --------------------------------------------------------------------------- #
def test_hnet_train_rollout_parity():
    core = _build(lambda: HNetCore(input_dim=INPUT_DIM, max_window=T))
    x = _input()
    with torch.no_grad():
        full, _ = core(x)
        roll = _rollout_features(core, x)
    assert torch.allclose(full, roll, atol=1e-4, rtol=0), (
        f"HNet train/rollout maxdiff {(full - roll).abs().max().item():.2e}"
    )
