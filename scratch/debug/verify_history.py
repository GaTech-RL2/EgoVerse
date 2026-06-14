"""Verify OBS-HISTORY conditioning for the GMM chunk policy.

Checks (architectural — random init is sufficient, this is about whether
history CONDITIONS the output, not about a trained model):

  1. N=1 byte-identical: chunk_forward_history(single-frame obs) must equal the
     OLD path policy(dummy, obs)[:,0] EXACTLY (same for eval generate's chunk
     branch, N=1, vs the original two-token [cond,BOS] forward).
  2. N>1 conditions: two history windows that share the SAME current frame but
     differ in EARLIER frames must produce DIFFERENT chunks (non-trivially).
  3. eval generate(N) consistency: generate on an (1,N,...) obs window changes
     with the earlier frames too (the eval path conditions on history).
"""
import os, sys
import torch

sys.path.insert(0, os.environ.get("REPO", "."))

from egomimic.models.hnet_nets.cond_encoders import CondEncoderModule
from egomimic.models.hnet_nets.image_encoders import ResNetEncoder
from egomimic.algo.hnet import FlatFusedPolicy, _action_head_cfg
from egomimic.models.hnet_nets.action_heads import action_head_raw, action_head_decode_chunk

torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"

D_COND = 128
D_MODEL = 128
A = 2
K = 32          # chunk size
IMG = 96

cond_enc = CondEncoderModule(
    d_cond=D_COND,
    output_key="fused_cond",
    cond_proj_widths=[128, 128],
    obs_specs={"state_agent_obj": {"input_dim": 2, "embed_dim": 64, "widths": [128], "input_slice": [0, 2]}},
    img_encoders={"front_img_1": ResNetEncoder(in_channels=3, embed_dim=128, resnet_model="resnet18", pretrained=False, spatial=True, image_size=IMG)},
)

policy = FlatFusedPolicy(
    action_dim=A, action_horizon=1024, d_model=D_MODEL, d_cond=D_COND,
    cond_encoder=cond_enc, arch_layout="T8", num_heads=4, d_intermediate=512,
    action_head_cfg=_action_head_cfg({"action_head": "gmm", "gmm_num_modes": 5, "chunk_k": K}),
).to(dev).float().eval()
policy.token_dropout_p = 0.0
policy.reactive = False

def rand_obs(W):
    return {
        "state_agent_obj": torch.randn(W, 5, device=dev),   # input_slice picks [:2]
        "front_img_1": torch.rand(W, 3, IMG, IMG, device=dev),
    }

W = 4
print("=" * 70)
print("CHECK 1: N=1 byte-identical (train forward path)")
print("=" * 70)
obs1 = rand_obs(W)
with torch.no_grad():
    # OLD path: policy(dummy, obs) with T=1 single-frame obs, read [:,0].
    dummy = torch.zeros(W, 1, A, device=dev)
    old_raw, _ = policy(dummy, obs1)         # (W, 1, K, params)
    old = old_raw[:, 0]                      # (W, K, params)
    # NEW path: chunk_forward_history(single-frame obs) -> N inferred as 1.
    new = policy.chunk_forward_history(obs1)  # (W, K, params)
max_abs = (old - new).abs().max().item()
print(f"  old shape={tuple(old.shape)} new shape={tuple(new.shape)}")
print(f"  max|old-new| = {max_abs:.3e}   byte-identical={torch.equal(old, new)}")
assert torch.equal(old, new), "N=1 train forward NOT byte-identical!"
print("  PASS: N=1 train forward is byte-identical.\n")

print("=" * 70)
print("CHECK 1b: N=1 byte-identical (eval generate chunk branch)")
print("=" * 70)
with torch.no_grad():
    # Single-frame obs (B,...) -> generate's chunk branch, N inferred 1.
    gen1 = policy.generate(obs1, batch_size=W, device=dev, T=K)  # (W, K, A) decoded
    # Reference: decode the OLD raw chunk directly.
    ref1 = action_head_decode_chunk(policy, old)                  # (W, K, A)
gmax = (gen1 - ref1).abs().max().item()
print(f"  gen shape={tuple(gen1.shape)}  max|gen-ref|={gmax:.3e}  equal={torch.equal(gen1, ref1)}")
assert torch.equal(gen1, ref1), "N=1 eval generate NOT byte-identical!"
print("  PASS: N=1 eval generate is byte-identical.\n")

print("=" * 70)
print("CHECK 2: N>1 history CONDITIONS the chunk (train forward)")
print("=" * 70)
N = 8
# Two windows: SAME current frame (last), DIFFERENT earlier frames.
cur = rand_obs(W)                      # the shared current (newest) frame
histA = [rand_obs(W) for _ in range(N - 1)]
histB = [rand_obs(W) for _ in range(N - 1)]   # different earlier frames
def stack_win(hist, cur):
    win = hist + [cur]                 # oldest -> newest (cur last == current)
    return {k: torch.stack([f[k] for f in win], dim=1) for k in cur}  # (W, N, ...)
winA = stack_win(histA, cur)
winB = stack_win(histB, cur)
# sanity: current frame identical, earlier frames differ
assert torch.equal(winA["state_agent_obj"][:, -1], winB["state_agent_obj"][:, -1])
assert torch.equal(winA["front_img_1"][:, -1], winB["front_img_1"][:, -1])
assert not torch.equal(winA["state_agent_obj"][:, 0], winB["state_agent_obj"][:, 0])
with torch.no_grad():
    chunkA = action_head_decode_chunk(policy, policy.chunk_forward_history(winA))  # (W,K,A)
    chunkB = action_head_decode_chunk(policy, policy.chunk_forward_history(winB))
    # Reference: a window whose earlier frames repeat the current frame (N==1-equiv)
    win_rep = {k: cur[k].unsqueeze(1).expand(-1, N, *([-1] * (cur[k].dim() - 1))).contiguous() for k in cur}
    chunk_rep = action_head_decode_chunk(policy, policy.chunk_forward_history(win_rep))
diff_AB = (chunkA - chunkB).abs()
print(f"  chunkA vs chunkB (same current, diff history):")
print(f"    mean|dA-dB|={diff_AB.mean().item():.4e}  max={diff_AB.max().item():.4e}")
print(f"  chunkA vs repeated-current window:")
print(f"    mean|dA-rep|={(chunkA-chunk_rep).abs().mean().item():.4e}")
cond_ok = diff_AB.max().item() > 1e-4
print(f"  history demonstrably changes chunk = {cond_ok}")
assert cond_ok, "N>1 history did NOT change the chunk -> NO-OP, control invalid!"
print("  PASS: N>1 history non-trivially conditions the train chunk.\n")

print("=" * 70)
print("CHECK 2b: N>1 history CONDITIONS the chunk (eval generate)")
print("=" * 70)
with torch.no_grad():
    genA = policy.generate(winA, batch_size=W, device=dev, T=K)  # (W,K,A)
    genB = policy.generate(winB, batch_size=W, device=dev, T=K)
gdiff = (genA - genB).abs()
print(f"  genA vs genB (same current, diff history):")
print(f"    mean|gA-gB|={gdiff.mean().item():.4e}  max={gdiff.max().item():.4e}")
egen_ok = gdiff.max().item() > 1e-4
print(f"  history demonstrably changes eval chunk = {egen_ok}")
assert egen_ok, "N>1 eval history did NOT change the chunk -> NO-OP!"
print("  PASS: N>1 history non-trivially conditions the eval chunk.\n")

print("=" * 70)
print("CHECK 3: train forward == eval generate for the SAME N-window")
print("=" * 70)
# Train chunk_forward_history -> decode == eval generate (both no_grad, eval mode)
with torch.no_grad():
    tr = action_head_decode_chunk(policy, policy.chunk_forward_history(winA))  # (W,K,A)
    ev = policy.generate(winA, batch_size=W, device=dev, T=K)
te = (tr - ev).abs().max().item()
print(f"  max|train_decode - eval_generate| = {te:.3e}  equal={torch.equal(tr, ev)}")
# These should match exactly (same token layout, same decode). GMM with sample=False is deterministic argmax-mode.
assert te < 1e-4, "train/eval chunk MISMATCH for the same N-window!"
print("  PASS: train forward and eval generate agree on the same window.\n")

print("ALL HISTORY-CONDITIONING CHECKS PASSED.")
