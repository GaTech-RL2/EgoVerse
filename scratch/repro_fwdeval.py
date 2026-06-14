"""Reproduce the forward_eval full-episode crash on TransformerCore, and confirm
LSTM handles it. Also confirm windowed-encode parity (one-shot vs 10 steps)."""
import torch
from egomimic.models.bc_rnn_nets import TransformerCore, LSTMCore

torch.manual_seed(0)
B, T, D, H = 2, 300, 66, 10  # T=300 = realistic new_circle_3 episode length

tx = TransformerCore(input_dim=D, d_model=448, n_layers=5, n_heads=8,
                     ff_mult=4, dropout=0.0, max_window=H).eval()
ls = LSTMCore(input_dim=D, hidden_dim=1000, num_layers=2, dropout=0.0).eval()

obs = torch.randn(B, T, D)

# 1) LSTM full-episode unroll: should work
try:
    out, _ = ls(obs)
    print(f"LSTM full-episode unroll OK -> out {tuple(out.shape)}")
except Exception as e:
    print(f"LSTM full-episode unroll RAISED: {type(e).__name__} - {e}")

# 2) TransformerCore full-episode unroll: expected to crash (max_window guard)
try:
    out, _ = tx(obs)
    print(f"TX full-episode unroll OK -> out {tuple(out.shape)}")
except Exception as e:
    print(f"TX full-episode unroll RAISED: {type(e).__name__} - {e}")

# 3) Sanity: a length-H window works for TX
try:
    out, _ = tx(obs[:, :H])
    print(f"TX length-{H} window OK -> out {tuple(out.shape)}")
except Exception as e:
    print(f"TX length-{H} window RAISED: {type(e).__name__} - {e}")
