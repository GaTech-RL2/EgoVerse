"""Trace training vs inference paths to find the bug."""
import sys
sys.path.insert(0, ".")
import torch
import numpy as np

device = "cuda"

# Load checkpoint
ckpt = torch.load(
    "logs/dit3d_both_fixes/dit3d_nopad_uniform_addfusion_2026-05-24_20-44-33/checkpoints/epoch_epoch=199.ckpt",
    map_location="cpu", weights_only=False,
)
state_dict = ckpt["state_dict"]

from egomimic.algo.dfot.dit3d_backbone import DFoTDiT3DBackbone
from egomimic.algo.dfot.continuous_diffusion import ContinuousDiffusion
from egomimic.algo.dfot.sampling import sample_step, vanilla_schedule
from egomimic.algo.vae.algo import load_pretrained_vae

# Build and load backbone
bb = DFoTDiT3DBackbone(
    latent_channels=4, latent_size=6, patch_size=1,
    action_horizon=32, d_model=384, d_cond=384, cond_dim=384,
    depth=12, num_heads=6, mlp_ratio=4.0,
    time_embed_dim=256, cond_emb_dim=256,
    use_fourier_time=False, cond_dropout_prob=0.1,
).to(device).eval()
prefix = "nets.outer_stage.inner_stage."
bb.load_state_dict({k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)})

diff = ContinuousDiffusion(action_dim=4, loss_weighting="uniform").to(device)
vae = load_pretrained_vae("external_ckpts/vae_v5_last.ckpt").to(device).eval()

# Load state_action_proj
from torch import nn
proj = nn.Sequential(nn.Linear(7, 384), nn.SiLU(), nn.Linear(384, 384)).to(device)
proj_prefix = "nets.outer_stage.state_action_proj."
proj.load_state_dict({k[len(proj_prefix):]: v for k, v in state_dict.items() if k.startswith(proj_prefix)})

T = 32

print("=== TEST 1: Does backbone output vary spatially? ===")
with torch.no_grad():
    x = torch.randn(1, T, 4, 6, 6, device=device)
    t_cond = torch.full((1, T), -1.0, device=device)  # some noise level
    cond = torch.randn(1, T, 384, device=device) * 0.1
    v = bb(x, t_cond, external_cond=cond)
    # Check spatial variation per frame
    for frame in [0, 15, 31]:
        spatial = v[0, frame]  # (4, 6, 6)
        print(f"  Frame {frame}: spatial std across HW = {spatial.std(dim=(1,2)).mean():.6f}, "
              f"mean = {spatial.mean():.6f}")
    # Check if different spatial positions give different values
    print(f"  v[0,0,:,0,0] = {v[0,0,:,0,0].cpu().numpy()}")
    print(f"  v[0,0,:,0,5] = {v[0,0,:,0,5].cpu().numpy()}")
    print(f"  v[0,0,:,5,0] = {v[0,0,:,5,0].cpu().numpy()}")
    print(f"  v[0,0,:,5,5] = {v[0,0,:,5,5].cpu().numpy()}")

print("\n=== TEST 2: Training path simulation ===")
with torch.no_grad():
    # Simulate a training step
    gt_img = torch.rand(T, 3, 96, 96, device=device)
    gt_latent, _ = vae.encode(gt_img)  # (T, 4, 6, 6)
    print(f"  GT latent: mean={gt_latent.mean():.4f}, std={gt_latent.std():.4f}")

    # Sample noise level
    t = torch.rand(T, device=device).clamp(1e-5, 1-1e-5)
    q = diff.q_sample(gt_latent, t)
    x_t = q["x_t"]  # (T, 4, 6, 6)
    time_cond = q["time_cond"]  # (T,) = precond_scale * logsnr
    print(f"  time_cond range: [{time_cond.min():.3f}, {time_cond.max():.3f}]")

    # Run backbone like training does
    x_ep = x_t.unsqueeze(0)  # (1, T, 4, 6, 6)
    t_ep = time_cond.unsqueeze(0)  # (1, T)
    c_ep = torch.randn(1, T, 384, device=device) * 0.1
    v_pred = bb(x_ep, t_ep, external_cond=c_ep)
    print(f"  v_pred: mean={v_pred.mean():.4f}, std={v_pred.std():.4f}")

    # Compute what x0 would be
    alpha_t = q["alpha_t"]  # (T, 1, 1, 1)
    sigma_t = q["sigma_t"]
    x0_pred = alpha_t.unsqueeze(0) * x_ep - sigma_t.unsqueeze(0) * v_pred
    print(f"  x0_pred: mean={x0_pred.mean():.4f}, std={x0_pred.std():.4f}")
    print(f"  GT latent: mean={gt_latent.mean():.4f}, std={gt_latent.std():.4f}")
    mse = ((x0_pred.squeeze(0) - gt_latent) ** 2).mean()
    print(f"  x0_pred vs GT MSE: {mse:.6f}")

print("\n=== TEST 3: Inference sampling chain ===")
with torch.no_grad():
    cond = torch.randn(1, T, 384, device=device) * 0.1
    schedule = vanilla_schedule(n_steps=100, T=T, discrete_timesteps=None).to(device)
    x = torch.randn(1, T, 4, 6, 6, device=device)

    print(f"  Schedule shape: {schedule.shape}, range: [{schedule.min():.4f}, {schedule.max():.4f}]")
    print(f"  Initial noise: mean={x.mean():.4f}, std={x.std():.4f}")

    for s in range(100):
        cur = schedule[s]
        nxt = schedule[s + 1]
        x = sample_step(diff, bb, x=x, current_levels=cur, next_levels=nxt,
                        external_cond=cond, eta=0.0)
        if s in [0, 1, 9, 49, 99]:
            spatial_std = x[0].std(dim=(2, 3)).mean()
            print(f"  Step {s:3d}: t={float(cur[0]):.4f}→{float(nxt[0]):.4f}  "
                  f"mean={x.mean():.4f} std={x.std():.4f} spatial_std={spatial_std:.4f}")

    # Decode final
    decoded = vae.decode(x.squeeze(0))
    print(f"\n  Final decoded: mean={decoded.mean():.4f}, std={decoded.std():.4f}")
    print(f"  Final latent spatial variation per frame:")
    for frame in [0, 15, 31]:
        spatial = x[0, frame]
        print(f"    Frame {frame}: per-channel std across HW: {spatial.std(dim=(1,2)).cpu().numpy()}")

print("\n=== TEST 4: What does the backbone see as time_cond during sampling? ===")
with torch.no_grad():
    # Check the time_cond values that model_v passes
    t_values = [0.99, 0.5, 0.01]
    for t_val in t_values:
        t_tensor = torch.tensor([[t_val] * T], device=device)
        logsnr = diff.schedule(t_tensor)
        time_cond = diff.precond_scale * logsnr
        print(f"  t={t_val:.2f} -> logsnr={logsnr[0,0]:.3f} -> time_cond={time_cond[0,0]:.3f}")

print("\n=== DONE ===")
