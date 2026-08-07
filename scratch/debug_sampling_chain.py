"""Debug: step through the sampling chain to find where it goes wrong."""
import sys
sys.path.insert(0, ".")
import torch

device = "cuda"

# Load trained model
from egomimic.algo.dfot.dit3d_backbone import DFoTDiT3DBackbone
from egomimic.algo.dfot.discrete_diffusion import DiscreteDiffusion
from egomimic.algo.dfot.sampling import sample_step, vanilla_schedule

ckpt = torch.load(
    "logs/dit3d_spatial/dit3d_spatial_400ep_2026-05-24_00-07-33/checkpoints/epoch_epoch=99.ckpt",
    map_location="cpu", weights_only=False,
)
state_dict = ckpt["state_dict"]

bb = DFoTDiT3DBackbone(
    latent_channels=4, latent_size=6, patch_size=1,
    action_horizon=32, d_model=384, d_cond=384, cond_dim=384,
    depth=12, num_heads=6, mlp_ratio=4.0,
    time_embed_dim=256, cond_emb_dim=256,
    use_fourier_time=False, cond_dropout_prob=0.1,
).to(device).eval()

diff = DiscreteDiffusion(
    action_dim=4, timesteps=1000, beta_schedule="cosine",
    objective="pred_v", loss_weighting_strategy="min_snr",
).to(device)

prefix = "nets.outer_stage.inner_stage."
bb_weights = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
bb.load_state_dict(bb_weights)

# Load VAE for reference
from egomimic.algo.vae.algo import load_pretrained_vae
vae = load_pretrained_vae("external_ckpts/vae_v5_last.ckpt").to(device).eval()

# Get real latent stats from VAE
with torch.no_grad():
    dummy = torch.rand(16, 3, 96, 96, device=device)
    real_mu, _ = vae.encode(dummy)
    print(f"Real latent stats: mean={real_mu.mean():.4f}, std={real_mu.std():.4f}")

# Step through sampling manually
B, T = 1, 32
cond = torch.randn(B, T, 384, device=device) * 0.1  # small cond
schedule = vanilla_schedule(n_steps=50, T=T, discrete_timesteps=1000).to(device)

x = torch.randn(B, T, 4, 6, 6, device=device)
print(f"\nInitial noise: mean={x.mean():.4f}, std={x.std():.4f}")
print(f"Schedule shape: {schedule.shape}")
print(f"Schedule[0] (start, noisiest): {schedule[0, 0].item()}")
print(f"Schedule[-1] (end, cleanest): {schedule[-1, 0].item()}")

# Step through and print stats at key points
with torch.no_grad():
    for s in range(50):
        cur_levels = schedule[s]
        nxt_levels = schedule[s + 1]
        x = sample_step(diff, bb, x=x, current_levels=cur_levels,
                        next_levels=nxt_levels, external_cond=cond, eta=0.0)
        if s in [0, 1, 5, 10, 24, 49]:
            decoded = vae.decode(x.squeeze(0)[:1])  # decode first frame only
            print(f"  step {s:2d}: k={int(cur_levels[0].item()):4d}→{int(nxt_levels[0].item()):4d}  "
                  f"x mean={x.mean():.4f} std={x.std():.4f}  "
                  f"decoded_pixel mean={decoded.mean():.3f} std={decoded.std():.3f}")

# Final decoded output
with torch.no_grad():
    final_decoded = vae.decode(x.squeeze(0))
print(f"\nFinal decoded: mean={final_decoded.mean():.4f}, std={final_decoded.std():.4f}")
print(f"  min={final_decoded.min():.4f}, max={final_decoded.max():.4f}")
