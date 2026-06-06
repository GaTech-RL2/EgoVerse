"""Full patchify → RoPE → attention → unpatchify consistency check."""
import sys
sys.path.insert(0, ".")
import torch
from einops import rearrange
from egomimic.algo.dfot.dit3d_backbone import DFoTDiT3DBackbone

device = "cuda"
bb = DFoTDiT3DBackbone(
    latent_channels=4, latent_size=6, patch_size=1,
    action_horizon=32, d_model=384, d_cond=384, cond_dim=384,
    depth=12, num_heads=6, mlp_ratio=4.0,
    time_embed_dim=256, cond_emb_dim=256,
).to(device).eval()

B, T, C, H, W = 1, 2, 4, 6, 6

# 1. Check that RoPE gives UNIQUE attention patterns per spatial position
print("=== RoPE spatial discrimination (Q·K dot products) ===")
rope = bb.blocks[0].attn.rope
# Build Q from patchified input
x = torch.randn(B, T, C, H, W, device=device)
x_flat = rearrange(x, "b t c h w -> (b t) c h w")
tokens = bb.patch_embed(x_flat)
tokens_seq = rearrange(tokens, "(b t) p c -> b (t p) c", b=B)

# Run through QKV
attn = bb.blocks[0].attn
qkv = attn.qkv(tokens_seq).reshape(B, T*36, 3, 6, 64).permute(2, 0, 3, 1, 4)
q, k, v = qkv.unbind(0)
q_rot = rope(q)
k_rot = rope(k)

# Check attention scores between frame-0 tokens
scores = torch.matmul(q_rot[0, 0, :36], k_rot[0, 0, :36].transpose(-2, -1)) / (64 ** 0.5)
print(f"Attention scores (frame 0, head 0): shape={scores.shape}")
print(f"  Diagonal (self-attn): mean={scores.diag().mean():.4f}")
print(f"  Off-diag: mean={scores[~torch.eye(36, device=device).bool()].mean():.4f}")
# Check if attention is uniform (all same) or position-dependent
row0 = scores[0]  # attention FROM token 0 TO all others
row1 = scores[1]  # attention FROM token 1 TO all others
print(f"  Row 0 std (variation in tok0's attention): {row0.std():.4f}")
print(f"  Row 1 std (variation in tok1's attention): {row1.std():.4f}")
print(f"  Row 0 vs Row 1 diff: {(row0 - row1).abs().mean():.4f}")

# 2. Full forward and check output spatial coherence
print("\n=== Full forward spatial coherence ===")
with torch.no_grad():
    noise = torch.full((B, T), 0.5, device=device)  # mid-noise level
    cond = torch.randn(B, T, 384, device=device) * 0.1
    v_pred = bb(x, noise, external_cond=cond)
    print(f"Output: shape={v_pred.shape}")

    # Check spatial smoothness: adjacent pixels should be correlated
    for t in range(T):
        frame = v_pred[0, t]  # (4, 6, 6)
        # Horizontal gradient
        h_grad = (frame[:, :, 1:] - frame[:, :, :-1]).abs().mean()
        # Vertical gradient
        v_grad = (frame[:, 1:, :] - frame[:, :-1, :]).abs().mean()
        overall_std = frame.std()
        print(f"  Frame {t}: h_grad={h_grad:.4f} v_grad={v_grad:.4f} overall_std={overall_std:.4f}")
        print(f"    Gradient/std ratio: {((h_grad + v_grad) / 2 / overall_std):.4f} (lower=smoother)")

# 3. Check what happens with the TRAINED model
print("\n=== Trained model spatial coherence ===")
import glob
ckpts = sorted(glob.glob("logs/dit3d_sigmoid_lnorm/*/checkpoints/epoch_epoch=*.ckpt"))
if ckpts:
    ckpt = torch.load(ckpts[-1], map_location="cpu", weights_only=False)
    prefix = "nets.outer_stage.inner_stage."
    bb.load_state_dict({k[len(prefix):]: v for k, v in ckpt["state_dict"].items() if k.startswith(prefix)})
    bb.eval()
    print(f"Loaded: {ckpts[-1]}")

    with torch.no_grad():
        v_pred = bb(x, noise, external_cond=cond)
        for t in range(T):
            frame = v_pred[0, t]
            h_grad = (frame[:, :, 1:] - frame[:, :, :-1]).abs().mean()
            v_grad = (frame[:, 1:, :] - frame[:, :-1, :]).abs().mean()
            overall_std = frame.std()
            print(f"  Frame {t}: h_grad={h_grad:.4f} v_grad={v_grad:.4f} overall_std={overall_std:.4f}")
            print(f"    Gradient/std ratio: {((h_grad + v_grad) / 2 / overall_std):.4f}")
else:
    print("  No checkpoints found")

print("\n=== DONE ===")
