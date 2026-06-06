"""End-to-end correctness test: verify the full train → sample → decode pipeline."""
import sys
sys.path.insert(0, ".")
import torch
import numpy as np

device = "cuda"

from egomimic.algo.dfot.dit3d_backbone import DFoTDiT3DBackbone
from egomimic.algo.dfot.continuous_diffusion import ContinuousDiffusion
from egomimic.algo.dfot.sampling import sample_step, vanilla_schedule, sample as _sample
from egomimic.algo.vae.algo import load_pretrained_vae
import json

# Load VAE + latent stats
vae = load_pretrained_vae("external_ckpts/vae_v5_last.ckpt").to(device).eval()
with open("external_ckpts/vae_latent_stats.json") as f:
    ls = json.load(f)
latent_mean = torch.tensor(ls["latent_mean"]).float().reshape(1, -1, 1, 1).to(device)
latent_std = torch.tensor(ls["latent_std"]).float().reshape(1, -1, 1, 1).to(device)

def normalize(z):
    return (z - latent_mean) / latent_std

def denormalize(z):
    return z * latent_std + latent_mean

# Build model
bb = DFoTDiT3DBackbone(
    latent_channels=4, latent_size=6, patch_size=1,
    action_horizon=32, d_model=384, d_cond=384, cond_dim=384,
    depth=12, num_heads=6, mlp_ratio=4.0,
    time_embed_dim=256, cond_emb_dim=256,
    use_fourier_time=False, cond_dropout_prob=0.1,
).to(device)
diff = ContinuousDiffusion(action_dim=4, loss_weighting="uniform").to(device)

print("=== Test 1: Normalize/denormalize round-trip ===")
with torch.no_grad():
    gt_img = torch.rand(4, 3, 96, 96, device=device)
    mu, _ = vae.encode(gt_img)
    print(f"Raw latent: mean={mu.mean():.4f} std={mu.std():.4f}")
    z_norm = normalize(mu)
    print(f"Normalized: mean={z_norm.mean():.4f} std={z_norm.std():.4f}")
    z_denorm = denormalize(z_norm)
    print(f"Denormed:   mean={z_denorm.mean():.4f} std={z_denorm.std():.4f}")
    print(f"Round-trip MSE: {((z_denorm - mu)**2).mean():.10f}")
    recon = vae.decode(z_denorm)
    print(f"Decoded denormed: mean={recon.mean():.4f} std={recon.std():.4f}")

print("\n=== Test 2: Training forward + loss (1 step, untrained model) ===")
T = 16
cond = torch.randn(1, T, 384, device=device) * 0.1
with torch.no_grad():
    gt_img = torch.rand(T, 3, 96, 96, device=device)
    mu, _ = vae.encode(gt_img)
    z_norm = normalize(mu)  # (T, 4, 6, 6) centered
    print(f"Normalized latent: mean={z_norm.mean():.4f} std={z_norm.std():.4f}")

    # q_sample on NORMALIZED latent
    t = torch.rand(T, device=device).clamp(1e-5, 1-1e-5)
    q = diff.q_sample(z_norm, t)
    x_t = q["x_t"]
    print(f"Noised x_t: mean={x_t.mean():.4f} std={x_t.std():.4f}")

    # Backbone forward
    v_pred = bb(x_t.unsqueeze(0), q["time_cond"].unsqueeze(0), external_cond=cond)
    print(f"v_pred (untrained): mean={v_pred.mean():.4f} std={v_pred.std():.4f}")

    # Compute x0 prediction
    alpha_t = q["alpha_t"]  # (T, 1, 1, 1)
    sigma_t = q["sigma_t"]
    x0_pred = alpha_t * x_t - sigma_t * v_pred.squeeze(0)
    print(f"x0_pred: mean={x0_pred.mean():.4f} std={x0_pred.std():.4f}")

    # Denormalize and decode
    x0_denorm = denormalize(x0_pred)
    decoded = vae.decode(x0_denorm)
    print(f"Decoded (untrained): mean={decoded.mean():.4f} std={decoded.std():.4f}")

print("\n=== Test 3: Sampling from untrained model ===")
with torch.no_grad():
    schedule = vanilla_schedule(n_steps=20, T=T, discrete_timesteps=None).to(device)
    sampled = _sample(diff, bb, schedule_matrix=schedule, x_shape=(4, 6, 6),
                      batch_size=1, external_cond=cond, device=device)
    print(f"Sampled (normalized space): mean={sampled.mean():.4f} std={sampled.std():.4f}")

    # Denormalize before decode
    sampled_denorm = denormalize(sampled.squeeze(0))
    decoded_denorm = vae.decode(sampled_denorm)
    print(f"Decoded WITH denorm: mean={decoded_denorm.mean():.4f} std={decoded_denorm.std():.4f}")

    # What if we DON'T denormalize? (the bug case)
    decoded_no_denorm = vae.decode(sampled.squeeze(0))
    print(f"Decoded WITHOUT denorm: mean={decoded_no_denorm.mean():.4f} std={decoded_no_denorm.std():.4f}")

print("\n=== Test 4: Load trained checkpoint and sample ===")
ckpt_path = "logs/dit3d_all_fixes/dit3d_nopad_uniform_addfusion_latentnorm_2026-05-25_17-31-23/checkpoints/epoch_epoch=199.ckpt"
try:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    prefix = "nets.outer_stage.inner_stage."
    bb.load_state_dict({k[len(prefix):]: v for k, v in ckpt["state_dict"].items() if k.startswith(prefix)})
    bb.eval()
    print("Loaded trained checkpoint")

    with torch.no_grad():
        schedule = vanilla_schedule(n_steps=100, T=T, discrete_timesteps=None).to(device)
        sampled = _sample(diff, bb, schedule_matrix=schedule, x_shape=(4, 6, 6),
                          batch_size=1, external_cond=cond, device=device)
        print(f"Sampled (trained, normalized space): mean={sampled.mean():.4f} std={sampled.std():.4f}")

        sampled_denorm = denormalize(sampled.squeeze(0))
        decoded = vae.decode(sampled_denorm)
        print(f"Decoded (trained, WITH denorm): mean={decoded.mean():.4f} std={decoded.std():.4f}")

        # Check spatial variation
        print(f"Spatial variation in sampled latent:")
        for frame in [0, 7, 15]:
            spatial = sampled[0, frame]
            print(f"  Frame {frame}: per-ch std across HW: {spatial.std(dim=(1,2)).cpu().numpy()}")
except FileNotFoundError:
    print(f"  Checkpoint not found: {ckpt_path}")

print("\n=== DONE ===")
