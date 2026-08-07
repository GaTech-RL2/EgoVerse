"""Test trained model's sensitivity to noise level and spatial coherence."""
import sys
sys.path.insert(0, ".")
import torch
import glob

device = "cuda"

from egomimic.algo.dfot.dit3d_backbone import DFoTDiT3DBackbone
from egomimic.algo.dfot.continuous_diffusion import ContinuousDiffusion

bb = DFoTDiT3DBackbone(
    latent_channels=4, latent_size=6, patch_size=2,
    action_horizon=32, d_model=384, d_cond=384, cond_dim=384,
    depth=12, num_heads=6, mlp_ratio=4.0,
    time_embed_dim=256, cond_emb_dim=256,
).to(device).eval()

ckpts = sorted(glob.glob("logs/dit3d_sigmoid_lnorm/dit3d_nopad_sigmoid_addfusion_latentnorm_2026-05-25_22-28-58/checkpoints/*.ckpt"))
if not ckpts:
    print("No checkpoint!"); sys.exit(1)
ckpt = torch.load(ckpts[-1], map_location="cpu", weights_only=False)
prefix = "nets.outer_stage.inner_stage."
bb.load_state_dict({k[len(prefix):]: v for k, v in ckpt["state_dict"].items() if k.startswith(prefix)})
print(f"Loaded: {ckpts[-1]}")

diff = ContinuousDiffusion(action_dim=4).to(device)
x = torch.randn(1, 8, 4, 6, 6, device=device)
cond = torch.randn(1, 8, 384, device=device) * 0.1

print("\n=== Trained model noise sensitivity ===")
with torch.no_grad():
    for tc_val in [-2.0, -1.0, 0.0, 1.0, 2.0]:
        v = bb(x, torch.full((1, 8), tc_val, device=device), external_cond=cond)
        print(f"  time_cond={tc_val:5.1f}: mean={v.mean():.4f} std={v.std():.4f}")

print("\n=== Spatial coherence ===")
with torch.no_grad():
    v = bb(x, torch.full((1, 8), 0.0, device=device), external_cond=cond)
    for t in range(3):
        frame = v[0, t]
        h_grad = (frame[:, :, 1:] - frame[:, :, :-1]).abs().mean()
        v_grad = (frame[:, 1:, :] - frame[:, :-1, :]).abs().mean()
        overall_std = frame.std()
        within_h = h_grad
        cross_h = overall_std
        within_v = v_grad
        cross_v = overall_std
        print(f"  Frame {t}: within_patch_grad={within_h:.4f}/{within_v:.4f}"
              f"  cross_patch_grad={cross_h:.4f}/{cross_v:.4f}"
              f"  ratio={cross_h/max(within_h,1e-8):.2f}/{cross_v/max(within_v,1e-8):.2f}")

print("\n=== AdaLN gate magnitudes (layer 0) ===")
with torch.no_grad():
    from einops import rearrange
    x_flat = rearrange(x, "b t c h w -> (b t) c h w")
    tokens = bb.patch_embed(x_flat)
    tokens_seq = rearrange(tokens, "(b t) p c -> b (t p) c", b=1)
    noise_levels = torch.full((1, 8), 0.0, device=device)
    c = bb._build_cond(noise_levels, cond, is_packed=False, B=1, T=8, force_uncond=False)
    P = bb.num_patches
    c_exp = c.unsqueeze(2).expand(-1, -1, P, -1)
    c_flat = rearrange(c_exp, "b t p c -> b (t p) c")

    # Check gate magnitudes
    for layer_idx in [0, 5, 11]:
        block = bb.blocks[layer_idx]
        _, gate1 = block.norm1(tokens_seq, c_flat)
        print(f"  Layer {layer_idx} attn gate: mean={gate1.mean():.6f} std={gate1.std():.6f} max={gate1.abs().max():.4f}")

print("\nDONE")
