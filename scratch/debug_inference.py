"""Debug script: load checkpoint, run one forward + sample, inspect outputs."""
import sys
sys.path.insert(0, ".")
import torch
import numpy as np

# Load the model from checkpoint
from egomimic.pl_utils.pl_model import ModelWrapper
import hydra
from omegaconf import OmegaConf

ckpt_path = "logs/dit3d_spatial/dit3d_spatial_400ep_2026-05-24_00-07-33/checkpoints/epoch_epoch=149.ckpt"
print(f"Loading checkpoint: {ckpt_path}")

ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
cfg = ckpt.get("hyper_parameters", {})

# Check what keys are in the checkpoint
print(f"Checkpoint keys: {list(ckpt.keys())}")
if "hyper_parameters" in ckpt:
    hp = ckpt["hyper_parameters"]
    print(f"Hyper params keys: {list(hp.keys()) if isinstance(hp, dict) else type(hp)}")

# Instead of full model load, let's just test the backbone + diffusion + VAE
print("\n--- Testing components directly ---")

from egomimic.algo.dfot.dit3d_backbone import DFoTDiT3DBackbone
from egomimic.algo.dfot.discrete_diffusion import DiscreteDiffusion
from egomimic.algo.dfot.sampling import sample as _sample, vanilla_schedule

device = "cuda"

# Build backbone
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

# Load weights from checkpoint
state_dict = ckpt["state_dict"]
# Filter for backbone weights
bb_prefix = "model.nets.outer_stage.inner_stage."
bb_weights = {k.replace(bb_prefix, ""): v for k, v in state_dict.items() if k.startswith(bb_prefix)}
print(f"\nBackbone weights: {len(bb_weights)} tensors")
if bb_weights:
    print(f"  Sample keys: {list(bb_weights.keys())[:5]}")
    bb.load_state_dict(bb_weights, strict=False)
    print("  Loaded backbone weights OK")

# Test 1: Forward pass with known input
print("\n--- Test 1: Forward pass ---")
B, T = 1, 32
x = torch.randn(B, T, 4, 6, 6, device=device)
k = torch.randint(0, 1000, (B, T), device=device)
cond = torch.randn(B, T, 384, device=device)

with torch.no_grad():
    v_pred = bb(x, k, external_cond=cond)
print(f"v_pred: shape={v_pred.shape}, mean={v_pred.mean():.4f}, std={v_pred.std():.4f}")
print(f"  min={v_pred.min():.4f}, max={v_pred.max():.4f}")

# Test 2: Sample from noise
print("\n--- Test 2: Sample (DDIM, 50 steps) ---")
schedule = vanilla_schedule(n_steps=50, T=T, discrete_timesteps=1000).to(device)
with torch.no_grad():
    out = _sample(diff, bb, schedule_matrix=schedule, x_shape=(4, 6, 6),
                  batch_size=1, external_cond=cond, device=device)
print(f"Sampled: shape={out.shape}, mean={out.mean():.4f}, std={out.std():.4f}")
print(f"  min={out.min():.4f}, max={out.max():.4f}")

# Test 3: Load VAE and decode
print("\n--- Test 3: VAE decode ---")
from egomimic.algo.vae.algo import load_pretrained_vae
vae = load_pretrained_vae("external_ckpts/vae_v5_last.ckpt").to(device)

# First check: encode a real-ish input and decode it
dummy_img = torch.rand(1, 3, 96, 96, device=device)
with torch.no_grad():
    mu, logvar = vae.encode(dummy_img)
    print(f"VAE encode: mu shape={mu.shape}, mean={mu.mean():.4f}, std={mu.std():.4f}")
    recon = vae.decode(mu)
    print(f"VAE decode(mu): shape={recon.shape}, mean={recon.mean():.4f}, std={recon.std():.4f}")
    print(f"  min={recon.min():.4f}, max={recon.max():.4f}")

# Now decode the sampled latent
with torch.no_grad():
    sampled_frames = vae.decode(out.squeeze(0))  # (T, 4, 6, 6) -> (T, 3, 96, 96)?
    print(f"\nVAE decode(sampled): shape={sampled_frames.shape}")
    print(f"  mean={sampled_frames.mean():.4f}, std={sampled_frames.std():.4f}")
    print(f"  min={sampled_frames.min():.4f}, max={sampled_frames.max():.4f}")

# Compare sampled latent stats vs real latent stats
print("\n--- Latent distribution comparison ---")
with torch.no_grad():
    real_mu, _ = vae.encode(torch.rand(16, 3, 96, 96, device=device))
    print(f"Real latents:    mean={real_mu.mean():.4f}, std={real_mu.std():.4f}, min={real_mu.min():.4f}, max={real_mu.max():.4f}")
    print(f"Sampled latents: mean={out.mean():.4f}, std={out.std():.4f}, min={out.min():.4f}, max={out.max():.4f}")

print("\n=== DONE ===")
