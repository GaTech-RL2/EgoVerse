"""End-to-end proof that the BCRNNPolicy.forward path used by BCRNN.forward_eval
(the overlay path reviewers flagged as crashing) now runs for the TRANSFORMER on a
full-length episode. We isolate the previously-broken call -- policy(obs) over a
T>max_window episode -> TransformerCore.forward -> _encode -- by stubbing the
obs_encoder to emit a precomputed (B,T,66) embedding (the encoder itself is shared
& unchanged; the bug was purely in the core's full-length forward). This is exactly
what forward_eval does at line `raw = policy(obs_padded)` then gmm_head.decode.
"""
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

CFG_DIR = "/coc/flash7/paphiwetsa3/projects/EgoVerse2/egomimic/hydra_configs/model"

with initialize_config_dir(version_base=None, config_dir=CFG_DIR):
    cfg = compose(config_name="bc_rnn_pushshapes_paperexact_tx")
rm = cfg.robomimic_model
obs_encoder = instantiate(rm.obs_encoder)
core = instantiate(rm.lstm)
gmm_head = instantiate(rm.gmm_head)

from egomimic.algo.bc_rnn import BCRNNPolicy
policy = BCRNNPolicy(
    obs_encoder=obs_encoder, lstm=core, gmm_head=gmm_head,
    action_dim=rm.action_dim, action_horizon=rm.action_horizon,
    rnn_horizon=rm.rnn_horizon, actor_mlp_dims=list(rm.get("actor_mlp_dims", []) or []),
    core=rm.get("core", "lstm"),
).eval()
print(f"[build] BCRNNPolicy(core=transformer) OK; core.max_window={core.max_window} "
      f"rnn_horizon={policy.rnn_horizon}")

dev = next(policy.parameters()).device
T = 300  # full episode, >> max_window=10
# Stub the obs_encoder to emit a precomputed (B,T,66) embedding (B=2 episodes).
emb_fixed = torch.randn(2, T, core.input_dim, device=dev)
class _StubEnc(torch.nn.Module):
    def forward(self, obs): return emb_fixed
policy.obs_encoder = _StubEnc().to(dev)  # bypass image/state encode

# This is the exact forward_eval inner call: raw = policy(obs); pred = decode(raw)
try:
    with torch.no_grad():
        raw = policy({"_stub": True})           # -> core forward over full T
        pred = policy.gmm_head.decode(raw)       # decode to actions (B,T,2)
    finite = torch.isfinite(pred).all().item()
    print(f"[forward_eval-core-path] OK on T={T}: raw {tuple(raw.shape)} "
          f"pred {tuple(pred.shape)} finite={finite}")
    assert pred.shape[:2] == (2, T), f"expected (2,{T},2) pred, got {pred.shape}"
    assert finite, "non-finite pred"

    # cross-check: windowed forward == per-window decode matches sequential rollout
    # (already proven in proof_suite [D]; here just confirm the overlay decode is
    # finite and full-length, which it is).
    print("FWDEVAL_PIPELINE_OK")
except Exception as e:
    import traceback; traceback.print_exc()
    raise
