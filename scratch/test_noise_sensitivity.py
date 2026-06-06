"""Test backbone sensitivity to noise level (time conditioning)."""
import sys
sys.path.insert(0, ".")
import torch
from egomimic.algo.dfot.dit3d_backbone import DFoTDiT3DBackbone
from egomimic.algo.dfot.continuous_diffusion import ContinuousDiffusion

device = "cuda"

bb = DFoTDiT3DBackbone(
    latent_channels=4, latent_size=6, patch_size=2,
    action_horizon=32, d_model=384, d_cond=384, cond_dim=384,
    depth=2, num_heads=6, mlp_ratio=4.0,
    time_embed_dim=256, cond_emb_dim=256,
).to(device).eval()

diff = ContinuousDiffusion(action_dim=4, loss_weighting="uniform").to(device)

x = torch.randn(1, 8, 4, 6, 6, device=device)
cond = torch.randn(1, 8, 384, device=device) * 0.1

# Test 1: Raw float noise levels (like training passes time_cond = precond_scale * logsnr)
print("=== Test 1: Raw float noise levels ===")
with torch.no_grad():
    v_01 = bb(x, torch.full((1, 8), 0.1, device=device), external_cond=cond)
    v_05 = bb(x, torch.full((1, 8), 0.5, device=device), external_cond=cond)
    v_09 = bb(x, torch.full((1, 8), 0.9, device=device), external_cond=cond)
    print(f"  noise=0.1: v mean={v_01.mean():.6f} std={v_01.std():.6f}")
    print(f"  noise=0.5: v mean={v_05.mean():.6f} std={v_05.std():.6f}")
    print(f"  noise=0.9: v mean={v_09.mean():.6f} std={v_09.std():.6f}")
    print(f"  All same? {torch.allclose(v_01, v_05) and torch.allclose(v_05, v_09)}")
    print(f"  diff(0.1 vs 0.9): {(v_01 - v_09).abs().mean():.8f}")

# Test 2: What does the time_embed produce for different inputs?
print("\n=== Test 2: Time embedding output ===")
for val in [0.0, 0.1, 0.5, 0.9, 1.0, -2.0, 2.0]:
    t = torch.full((1, 8), val, device=device)
    with torch.no_grad():
        emb = bb.time_embed(t)
    print(f"  time_embed({val:5.1f}): mean={emb.mean():.6f} std={emb.std():.6f} norm={emb.norm():.4f}")

# Test 3: What does training actually pass as noise_levels?
print("\n=== Test 3: Training path noise_levels ===")
gt = torch.randn(8, 4, 6, 6, device=device)
t = torch.rand(8, device=device).clamp(1e-5, 1-1e-5)
q = diff.q_sample(gt, t)
time_cond = q["time_cond"]  # precond_scale * logsnr
print(f"  t range: [{t.min():.4f}, {t.max():.4f}]")
print(f"  time_cond range: [{time_cond.min():.4f}, {time_cond.max():.4f}]")
print(f"  time_cond mean: {time_cond.mean():.4f}")

# Test with training-range values
with torch.no_grad():
    v_neg2 = bb(x, torch.full((1, 8), -2.0, device=device), external_cond=cond)
    v_pos2 = bb(x, torch.full((1, 8), 2.0, device=device), external_cond=cond)
    print(f"\n  noise=-2.0: v mean={v_neg2.mean():.6f} std={v_neg2.std():.6f}")
    print(f"  noise=+2.0: v mean={v_pos2.mean():.6f} std={v_pos2.std():.6f}")
    print(f"  diff(-2 vs +2): {(v_neg2 - v_pos2).abs().mean():.8f}")

# Test 4: What does model_v in sampling pass?
print("\n=== Test 4: Inference path noise_levels ===")
from egomimic.algo.dfot.sampling import vanilla_schedule
schedule = vanilla_schedule(n_steps=10, T=8, discrete_timesteps=None).to(device)
for step in [0, 5, 9]:
    t_val = schedule[step, 0].item()
    logsnr = diff.schedule(schedule[step:step+1].unsqueeze(0)).squeeze()
    time_cond_inf = diff.precond_scale * logsnr
    print(f"  Step {step}: t={t_val:.4f} -> logsnr={logsnr[0]:.4f} -> precond*logsnr={time_cond_inf[0]:.4f}")

print("\n=== DONE ===")
