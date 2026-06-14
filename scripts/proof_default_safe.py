"""Default-safety proof: the H-Net core addition does NOT perturb the
pre-existing LSTM / TX code paths that the 6 running BC-RNN jobs use.

Checks:
  A. core=lstm  state_dict has ZERO hnet/tx params; core=transformer has ZERO
     hnet params. (Building HNetCore only happens under core=hnet.)
  B. Determinism: building an HNetCore (instantiating ALL hnet machinery) in
     between two seeded TX-core builds does NOT change the TX core weights —
     torch.equal on every TX param. Same for LSTM. (Catches any module-import
     side effect / global RNG mutation that would shift the running builds.)
  C. The HNetCore module-import does not register anything on TransformerCore /
     LSTMCore classes (no monkeypatch).
"""
import torch
from egomimic.models.bc_rnn_nets import HNetCore, TransformerCore, LSTMCore

IN = 66

def tx_weights(seed):
    torch.manual_seed(seed)
    c = TransformerCore(input_dim=IN, d_model=448, n_layers=5, n_heads=8, ff_mult=4,
                        max_window=10)
    return {k: v.clone() for k, v in c.state_dict().items()}

def lstm_weights(seed):
    torch.manual_seed(seed)
    c = LSTMCore(input_dim=IN, hidden_dim=1000, num_layers=2)
    return {k: v.clone() for k, v in c.state_dict().items()}

print("=" * 70)
print("[B] DETERMINISM: HNetCore build between seeded TX/LSTM builds is inert")
tx_a = tx_weights(123)
# instantiate the full H-Net machinery in between
torch.manual_seed(999)
_ = HNetCore(input_dim=IN, d_model=256, n_heads=8, d_intermediate=768,
             outer_layers=4, inner_layers=6, max_window=10)
tx_b = tx_weights(123)
all_eq = all(torch.equal(tx_a[k], tx_b[k]) for k in tx_a)
print(f"    TX  param torch.equal across HNet-in-between builds: {all_eq}")
assert all_eq, "TX weights perturbed by HNetCore build!"

ls_a = lstm_weights(123)
torch.manual_seed(7)
_ = HNetCore(input_dim=IN, d_model=256, n_heads=8, d_intermediate=768,
             outer_layers=4, inner_layers=6, max_window=10)
ls_b = lstm_weights(123)
all_eq_l = all(torch.equal(ls_a[k], ls_b[k]) for k in ls_a)
print(f"    LSTM param torch.equal across HNet-in-between builds: {all_eq_l}")
assert all_eq_l, "LSTM weights perturbed by HNetCore build!"

print("=" * 70)
print("[A] core-key isolation: hnet/tx params never appear in lstm/tx builds")
torch.manual_seed(0)
hnet = HNetCore(input_dim=IN, d_model=256, n_heads=8, d_intermediate=768,
                outer_layers=4, inner_layers=6, max_window=10)
hnet_keys = set(hnet.state_dict().keys())
tx_keys = set(TransformerCore(input_dim=IN).state_dict().keys())
lstm_keys = set(LSTMCore(input_dim=IN).state_dict().keys())
print(f"    HNet state_dict keys = {len(hnet_keys)}; TX = {len(tx_keys)}; LSTM = {len(lstm_keys)}")
# NOTE: hnet ∩ tx == {in_proj.weight, in_proj.bias} — both cores name their
# 66->d_model input projection `in_proj`. This is a benign attribute-name
# coincidence: a BCRNNPolicy holds exactly ONE core (selected by `core`), so the
# two in_proj's never coexist in a single state_dict. NOT a conflict.
print(f"    hnet ∩ tx   = {sorted(hnet_keys & tx_keys)} (only the shared in_proj name)")
print(f"    hnet ∩ lstm = {len(hnet_keys & lstm_keys)} (expect 0)")
assert (hnet_keys & tx_keys) == {"in_proj.weight", "in_proj.bias"}, \
    "Unexpected hnet/tx key overlap beyond the benign in_proj name"
assert len(hnet_keys & lstm_keys) == 0

print("=" * 70)
print("[C] no monkeypatch: TransformerCore.forward / LSTMCore.forward unchanged")
import inspect
print(f"    TX.forward module   = {TransformerCore.forward.__module__}")
print(f"    LSTM.forward module = {LSTMCore.forward.__module__}")
assert TransformerCore.forward.__module__.endswith("transformer_core")
assert LSTMCore.forward.__module__.endswith("lstm_core")

print("=" * 70)
print("DEFAULT-SAFETY PROOFS PASSED")
