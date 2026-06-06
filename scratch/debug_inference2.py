"""Load actual trained model, encode a GT episode, sample, compare."""
import sys
sys.path.insert(0, ".")
import torch
import numpy as np

device = "cuda"

# Load checkpoint
ckpt_path = "logs/dit3d_spatial/dit3d_spatial_400ep_2026-05-24_00-07-33/checkpoints/epoch_epoch=149.ckpt"
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
state_dict = ckpt["state_dict"]

# Build model components with correct prefix
from egomimic.algo.dfot.dit3d_backbone import DFoTDiT3DBackbone
from egomimic.algo.dfot.discrete_diffusion import DiscreteDiffusion
from egomimic.algo.dfot.sampling import sample as _sample, vanilla_schedule

bb = DFoTDiT3DBackbone(
    latent_channels=4, latent_size=6, patch_size=1,
    action_horizon=32, d_model=384, d_cond=384, cond_dim=384,
    depth=12, num_heads=6, mlp_ratio=4.0,
    time_embed_dim=256, cond_emb_dim=256,
    use_fourier_time=False, cond_dropout_prob=0.1,
).to(device)

diff = DiscreteDiffusion(
    action_dim=4, timesteps=1000, beta_schedule="cosine",
    objective="pred_v", loss_weighting_strategy="min_snr",
).to(device)

# Load backbone weights (correct prefix)
prefix = "nets.outer_stage.inner_stage."
bb_weights = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
print(f"Loading {len(bb_weights)} backbone weights")
bb.load_state_dict(bb_weights)
print("OK")

# Load state_action_proj weights
from torch import nn
state_action_proj = nn.Sequential(
    nn.Linear(7, 384), nn.SiLU(), nn.Linear(384, 384),
).to(device)
proj_prefix = "nets.outer_stage.state_action_proj."
proj_weights = {k[len(proj_prefix):]: v for k, v in state_dict.items() if k.startswith(proj_prefix)}
print(f"Loading {len(proj_weights)} state_action_proj weights")
state_action_proj.load_state_dict(proj_weights)

# Load VAE
from egomimic.algo.vae.algo import load_pretrained_vae
vae = load_pretrained_vae("external_ckpts/vae_v5_last.ckpt").to(device)

# Test 1: forward with trained weights + noise level 500
print("\n--- Test 1: Trained backbone forward ---")
B, T = 1, 32
x = torch.randn(B, T, 4, 6, 6, device=device)
k = torch.full((B, T), 500, device=device, dtype=torch.long)
cond = torch.randn(B, T, 384, device=device)
with torch.no_grad():
    v_pred = bb(x, k, external_cond=cond)
print(f"v_pred: mean={v_pred.mean():.4f}, std={v_pred.std():.4f}")

# Test 2: forward with noise level 0 (clean)
k_clean = torch.zeros(B, T, device=device, dtype=torch.long)
with torch.no_grad():
    v_pred_clean = bb(x, k_clean, external_cond=cond)
print(f"v_pred(k=0): mean={v_pred_clean.mean():.4f}, std={v_pred_clean.std():.4f}")

# Test 3: Encode GT, denoise, decode, compare
print("\n--- Test 3: Encode GT -> noise -> denoise -> decode ---")
# Create fake GT images
gt_img = torch.rand(T, 3, 96, 96, device=device)
with torch.no_grad():
    gt_latent, _ = vae.encode(gt_img)  # (T, 4, 6, 6)
print(f"GT latent: mean={gt_latent.mean():.4f}, std={gt_latent.std():.4f}")

# Build a realistic cond
fake_state = torch.randn(T, 5, device=device) * 0.1
fake_action = torch.randn(T, 2, device=device) * 0.1
fake_concat = torch.cat([fake_state, fake_action], dim=-1)
with torch.no_grad():
    cond_seq = state_action_proj(fake_concat).unsqueeze(0)  # (1, T, 384)
print(f"Cond: mean={cond_seq.mean():.4f}, std={cond_seq.std():.4f}")

# Sample
print("\nSampling...")
schedule = vanilla_schedule(n_steps=50, T=T, discrete_timesteps=1000).to(device)
with torch.no_grad():
    sampled = _sample(diff, bb, schedule_matrix=schedule, x_shape=(4, 6, 6),
                      batch_size=1, external_cond=cond_seq, device=device)
print(f"Sampled latent: mean={sampled.mean():.4f}, std={sampled.std():.4f}")

# Decode
with torch.no_grad():
    decoded = vae.decode(sampled.squeeze(0))  # (T, 3, 96, 96)
    gt_decoded = vae.decode(gt_latent)
print(f"Decoded frames: mean={decoded.mean():.4f}, std={decoded.std():.4f}")
print(f"GT decoded:     mean={gt_decoded.mean():.4f}, std={gt_decoded.std():.4f}")

# MSE between decoded and GT
mse = ((decoded - gt_decoded) ** 2).mean()
print(f"MSE (decoded vs GT decoded): {mse:.4f}")

# Test 4: Can the model round-trip? Encode GT, add small noise, denoise
print("\n--- Test 4: Round-trip (small noise -> denoise) ---")
gt_latent_batched = gt_latent.unsqueeze(0)  # (1, T, 4, 6, 6)
k_low = torch.full((1, T), 10, device=device, dtype=torch.long)  # very low noise
noise = torch.randn_like(gt_latent_batched)
x_noisy = diff.q_sample(gt_latent_batched.squeeze(0), k_low.squeeze(0), noise.squeeze(0))
print(f"Noisy (k=10): mean={x_noisy.mean():.4f}, std={x_noisy.std():.4f}")

# One-step denoise
with torch.no_grad():
    v_pred_rt = bb(x_noisy.unsqueeze(0), k_low, external_cond=cond_seq)
    x0_pred = diff.predict_start_from_v(x_noisy.unsqueeze(0), k_low, v_pred_rt)
print(f"Predicted x0: mean={x0_pred.mean():.4f}, std={x0_pred.std():.4f}")
print(f"GT latent:    mean={gt_latent.mean():.4f}, std={gt_latent.std():.4f}")
rt_mse = ((x0_pred.squeeze(0) - gt_latent) ** 2).mean()
print(f"Round-trip MSE: {rt_mse:.4f}")

print("\n=== DONE ===")
