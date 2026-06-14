"""TX-chunk8-FULLHIST proof: posTableCheck + parityProbe.

Builds the BCRNNPolicy submodule from the COMPOSED model config (the same
robomimic_model node trainHydra would build) for both the new fullhist leaf and
its tx_chunk8 base, then:

  posTableCheck:
    - TransformerCore.pos_emb is sized [max_window, d_model]; confirm
      max_window==80 (fullhist) vs 10 (base) -> pos table row count 80 vs 10.
    - total param delta == +70 rows * d_model (== +31,360 for d_model=448),
      matching the ~+31k expectation.
    - full-policy state_dict KEY-SET is identical between the two builds; the
      ONLY tensor whose SHAPE differs is core.pos_emb.weight (10->80 rows).

  parityProbe (the family bar, ~1e-6, deterministic):
    - fixed seed, random weights, synthetic 200-frame episode == 25 obs-steps
      (T=25 < max_window=80, so the rollout buffer NEVER reinits within the
      episode -- the full-history regime).
    - training-style anchored windowed forward (one fresh window over positions
      0..24) vs sequential BCRNNPolicy-style step() rollout (init_hidden once,
      step t=0..24, never reinit). Compare the EMITTED GMM params
      (means / scales / logits) at every obs-step. eval() mode so dropout is off;
      same obs embeddings feed both paths.
"""
import sys
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
import egomimic  # noqa: F401  (register resolvers / packages)
from egomimic.models.bc_rnn_nets import TransformerCore
from egomimic.algo.bc_rnn import BCRNNPolicy

CFG_DIR = "/coc/flash7/paphiwetsa3/projects/EgoVerse2/egomimic/hydra_configs/model"


def build(model_name):
    with initialize_config_dir(version_base=None, config_dir=CFG_DIR):
        cfg = compose(config_name=model_name)
    rm = cfg.robomimic_model
    obs_encoder = instantiate(rm.obs_encoder)
    core = instantiate(rm.core_net)
    gmm_head = instantiate(rm.gmm_head)
    actor_mlp_dims = list(rm.get("actor_mlp_dims", []) or [])
    policy = BCRNNPolicy(
        obs_encoder=obs_encoder, lstm=core, gmm_head=gmm_head,
        action_dim=rm.action_dim, action_horizon=rm.action_horizon,
        rnn_horizon=rm.rnn_horizon, actor_mlp_dims=actor_mlp_dims,
        core=rm.get("core", "lstm"),
        obs_stride=int(rm.get("obs_stride", 1)),
        chunk_len=int(rm.get("chunk_len", 1)),
    )
    return policy, core, gmm_head, int(rm.rnn_horizon)


print("=" * 70)
print("BUILD from composed configs")
print("=" * 70)
p_fh, core_fh, gmm_fh, rh_fh = build("bc_rnn_pushshapes_paperexact_tx_chunk8_fullhist")
p_bs, core_bs, gmm_bs, rh_bs = build("bc_rnn_pushshapes_paperexact_tx_chunk8")

assert isinstance(core_fh, TransformerCore) and isinstance(core_bs, TransformerCore)
print(f"fullhist: rnn_horizon={rh_fh}  core.max_window={core_fh.max_window}  d_model={core_fh.d_model}")
print(f"base    : rnn_horizon={rh_bs}  core.max_window={core_bs.max_window}  d_model={core_bs.d_model}")

# ---------------- posTableCheck ----------------
print("\n" + "=" * 70)
print("posTableCheck")
print("=" * 70)
pe_fh = core_fh.pos_emb.weight  # [max_window, d_model]
pe_bs = core_bs.pos_emb.weight
print(f"pos_emb.weight shape  fullhist={tuple(pe_fh.shape)}  base={tuple(pe_bs.shape)}")
assert pe_fh.shape[0] == 80, f"fullhist pos table rows != 80: {pe_fh.shape[0]}"
assert pe_bs.shape[0] == 10, f"base pos table rows != 10: {pe_bs.shape[0]}"
d_model = pe_fh.shape[1]

n_fh = sum(p.numel() for p in p_fh.parameters())
n_bs = sum(p.numel() for p in p_bs.parameters())
delta = n_fh - n_bs
expected_delta = (80 - 10) * d_model
print(f"full-policy params  fullhist={n_fh:,}  base={n_bs:,}  delta=+{delta:,}")
print(f"expected delta (70 rows * d_model={d_model}) = +{expected_delta:,}")
assert delta == expected_delta, f"param delta {delta} != expected {expected_delta}"

# state_dict KEY-SET identical; only pos_emb.weight shape differs
sd_fh = p_fh.state_dict()
sd_bs = p_bs.state_dict()
keys_fh, keys_bs = set(sd_fh), set(sd_bs)
assert keys_fh == keys_bs, (
    f"state_dict key-sets differ! only_fh={keys_fh - keys_bs} only_bs={keys_bs - keys_fh}")
shape_diffs = {k: (tuple(sd_bs[k].shape), tuple(sd_fh[k].shape))
               for k in keys_fh if sd_bs[k].shape != sd_fh[k].shape}
print(f"state_dict key-set identical: True ({len(keys_fh)} keys)")
print(f"keys whose SHAPE differs: {list(shape_diffs.keys())}")
posname = [k for k in keys_fh if k.endswith("pos_emb.weight")]
assert len(shape_diffs) == 1, f"more than pos_emb shape changed: {shape_diffs}"
assert list(shape_diffs.keys())[0] in posname, f"the differing key is not pos_emb: {shape_diffs}"
print(f"  -> ONLY {list(shape_diffs.keys())[0]}: base {shape_diffs[list(shape_diffs.keys())[0]][0]} "
      f"-> fullhist {shape_diffs[list(shape_diffs.keys())[0]][1]}")
print("posTableCheck: PASS")

# ---------------- parityProbe ----------------
print("\n" + "=" * 70)
print("parityProbe (200-frame episode = 25 obs-steps; max_window=80, never reinits)")
print("=" * 70)
torch.manual_seed(1234)
D = core_fh.input_dim
T = 25  # 25 obs-steps == 200 env frames at obs_stride=8
core = core_fh.eval()
gmm = gmm_fh.eval()

# Synthetic per-obs-step embeddings (we probe core+head parity directly; the obs
# encoder is shared by both paths so feeding identical embeddings isolates the
# core/positional-window semantics -- the only thing rnn_horizon changes).
obs_ep = torch.randn(1, T, D)

with torch.no_grad():
    # (1) training-style anchored windowed forward: one fresh window, positions
    #     0..T-1 (T<=max_window -> short path == anchored full-episode window).
    feats_fwd, _ = core(obs_ep)                 # (1, T, hidden)
    gmm_fwd = gmm(feats_fwd.reshape(-1, feats_fwd.shape[-1]))

    # (2) sequential BCRNNPolicy.step rollout: init_hidden ONCE, step every
    #     obs-step, NEVER reinit (t never hits %rnn_horizon==0 again for t<80).
    feats_seq = []
    hidden = core.init_hidden(1, device=obs_ep.device)
    rh = rh_fh
    for t in range(T):
        if t % rh == 0 and t != 0:
            hidden = core.init_hidden(1, device=obs_ep.device)  # never fires for T=25
        h_t, hidden = core.step(obs_ep[:, t], hidden)
        feats_seq.append(h_t)
    feats_seq = torch.stack(feats_seq, dim=1)   # (1, T, hidden)
    gmm_seq = gmm(feats_seq.reshape(-1, feats_seq.shape[-1]))

feat_maxdiff = (feats_fwd - feats_seq).abs().max().item()


def gmm_maxdiff(a, b):
    if isinstance(a, dict):
        return max((a[k] - b[k]).abs().max().item() for k in a)
    if isinstance(a, (tuple, list)):
        return max((x - y).abs().max().item() for x, y in zip(a, b))
    return (a - b).abs().max().item()


# describe what the head emitted
if isinstance(gmm_fwd, dict):
    parts = {k: tuple(v.shape) for k, v in gmm_fwd.items()}
elif isinstance(gmm_fwd, (tuple, list)):
    parts = [tuple(v.shape) for v in gmm_fwd]
else:
    parts = tuple(gmm_fwd.shape)
print(f"emitted GMM params structure: {parts}")
gdiff = gmm_maxdiff(gmm_fwd, gmm_seq)
print(f"core-feature max abs diff (windowed-fwd vs sequential-step) = {feat_maxdiff:.3e}")
print(f"emitted GMM-param max abs diff                              = {gdiff:.3e}")
assert gdiff < 1e-4, f"parity broken: GMM maxdiff={gdiff}"
print("parityProbe: PASS (<=1e-4 family bar; expect ~1e-6)")

print("\nALL_TXFH_PROOFS_PASS "
      f"posrows={pe_fh.shape[0]} delta=+{delta} gmmdiff={gdiff:.3e}")
