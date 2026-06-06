"""Comprehensive unit tests for every stage of the video diffusion pipeline.
Tests both training and inference paths."""
import sys
sys.path.insert(0, ".")
import torch
import numpy as np
from einops import rearrange

device = "cuda"
PASS = 0
FAIL = 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} — {detail}")

# ============================================================
# Module 1: PatchEmbed
# ============================================================
print("=" * 60)
print("MODULE 1: PatchEmbed")
print("=" * 60)
from egomimic.algo.dfot.dit3d_backbone import PatchEmbed

pe = PatchEmbed(in_channels=4, hidden_size=384, patch_size=2).to(device)
x = torch.randn(2, 4, 6, 6, device=device)
out = pe(x)
check("PatchEmbed output shape", out.shape == (2, 9, 384),
      f"got {out.shape}, expected (2, 9, 384)")

# Verify spatial ordering: create input where each 2x2 patch has unique values
x_ordered = torch.zeros(1, 1, 6, 6, device=device)  # 1 channel for simplicity
for h in range(3):
    for w in range(3):
        x_ordered[0, 0, h*2:(h+1)*2, w*2:(w+1)*2] = h * 3 + w
# After PatchEmbed, token i should correspond to patch (h=i//3, w=i%3)
# The Conv2d mixes channels so we can't directly read values, but we can
# verify that different patches produce different tokens
pe_1ch = PatchEmbed(in_channels=1, hidden_size=8, patch_size=2).to(device)
tokens = pe_1ch(x_ordered)
check("PatchEmbed distinct tokens per patch",
      tokens[0].unique(dim=0).shape[0] == 9,
      f"got {tokens[0].unique(dim=0).shape[0]} unique tokens, expected 9")

# ============================================================
# Module 2: _rotate_half
# ============================================================
print("\n" + "=" * 60)
print("MODULE 2: _rotate_half")
print("=" * 60)
from egomimic.algo.dfot.dit3d_backbone import _rotate_half

x = torch.tensor([1., 2., 3., 4., 5., 6., 7., 8.])
r = _rotate_half(x)
expected = torch.tensor([-2., 1., -4., 3., -6., 5., -8., 7.])
check("_rotate_half pairs adjacent elements",
      torch.allclose(r, expected),
      f"got {r.numpy()}, expected {expected.numpy()}")

# ============================================================
# Module 3: RotaryEmbedding3D
# ============================================================
print("\n" + "=" * 60)
print("MODULE 3: RotaryEmbedding3D")
print("=" * 60)
from egomimic.algo.dfot.dit3d_backbone import RotaryEmbedding3D

rope = RotaryEmbedding3D(dim=64, sizes=(4, 3, 3)).to(device)
freqs = rope._build_freqs(4, device, torch.float32)
check("RoPE freqs shape", freqs.shape == (4*9, 64),
      f"got {freqs.shape}")

# All 9 spatial positions in frame 0 should have unique frequencies
frame0_freqs = freqs[:9]
check("RoPE unique spatial freqs (frame 0)",
      frame0_freqs.unique(dim=0).shape[0] == 9,
      f"got {frame0_freqs.unique(dim=0).shape[0]} unique, expected 9")

# Same spatial position in different frames should have different freqs
pos00_t0 = freqs[0]   # t=0, h=0, w=0
pos00_t1 = freqs[9]   # t=1, h=0, w=0
check("RoPE temporal discrimination",
      not torch.allclose(pos00_t0, pos00_t1),
      "same position different frames should differ")

# RoPE should produce different Q rotations for different positions
q = torch.ones(1, 1, 36, 64, device=device)
q_rot = rope(q)
check("RoPE rotated Q differs at pos 0 vs 1",
      not torch.allclose(q_rot[0,0,0], q_rot[0,0,1]),
      "adjacent spatial positions should produce different rotations")
check("RoPE rotated Q differs at pos 0 vs 9 (temporal)",
      not torch.allclose(q_rot[0,0,0], q_rot[0,0,9]),
      "different frames should produce different rotations")

# Verify RoPE is a valid rotation (preserves norm)
q_rand = torch.randn(1, 1, 36, 64, device=device)
q_rot_rand = rope(q_rand)
norm_before = q_rand.norm(dim=-1)
norm_after = q_rot_rand.norm(dim=-1)
check("RoPE preserves vector norm",
      torch.allclose(norm_before, norm_after, atol=1e-5),
      f"max norm diff: {(norm_before - norm_after).abs().max():.6f}")

# ============================================================
# Module 4: AdaLN-Zero
# ============================================================
print("\n" + "=" * 60)
print("MODULE 4: AdaLN-Zero")
print("=" * 60)
from egomimic.algo.dfot.dit3d_backbone import AdaLayerNormZero

aln = AdaLayerNormZero(384).to(device)
x = torch.randn(1, 9, 384, device=device)
c = torch.zeros(1, 9, 384, device=device)  # zero conditioning
out, gate = aln(x, c)
# With zero-init modulation and zero cond, shift=0, scale=0, gate=0
check("AdaLN-Zero gate is zero at init with zero cond",
      gate.abs().max() < 1e-6,
      f"max gate: {gate.abs().max():.8f}")

# ============================================================
# Module 5: DiTBlock
# ============================================================
print("\n" + "=" * 60)
print("MODULE 5: DiTBlock")
print("=" * 60)
from egomimic.algo.dfot.dit3d_backbone import DiTBlock

rope_block = RotaryEmbedding3D(dim=64, sizes=(4, 3, 3)).to(device)
block = DiTBlock(384, 6, 4.0, rope=rope_block).to(device)
x = torch.randn(1, 36, 384, device=device)
c = torch.zeros(1, 36, 384, device=device)
out = block(x, c)
check("DiTBlock output shape", out.shape == x.shape)
# With zero cond (zero-init gates), output should equal input (residual only)
check("DiTBlock identity at init with zero cond",
      torch.allclose(out, x, atol=1e-5),
      f"max diff: {(out - x).abs().max():.6f}")

# ============================================================
# Module 6: Full backbone forward
# ============================================================
print("\n" + "=" * 60)
print("MODULE 6: Full DFoTDiT3DBackbone")
print("=" * 60)
from egomimic.algo.dfot.dit3d_backbone import DFoTDiT3DBackbone

bb = DFoTDiT3DBackbone(
    latent_channels=4, latent_size=6, patch_size=2,
    action_horizon=32, d_model=384, d_cond=384, cond_dim=384,
    depth=2, num_heads=6, mlp_ratio=4.0,
    time_embed_dim=256, cond_emb_dim=256,
).to(device).eval()

x = torch.randn(1, 8, 4, 6, 6, device=device)
noise = torch.full((1, 8), 0.5, device=device)
cond = torch.randn(1, 8, 384, device=device) * 0.1

with torch.no_grad():
    v = bb(x, noise, external_cond=cond)
check("Backbone output shape", v.shape == (1, 8, 4, 6, 6),
      f"got {v.shape}")

# Different spatial positions should produce different outputs
check("Backbone spatial variation",
      v[0, 0, :, 0, 0].std() > 0 or v[0, 0].std() > 0,
      "output should not be spatially uniform")

# Different noise levels should produce different outputs
with torch.no_grad():
    v_low = bb(x, torch.full((1, 8), 0.1, device=device), external_cond=cond)
    v_high = bb(x, torch.full((1, 8), 0.9, device=device), external_cond=cond)
check("Backbone sensitive to noise level",
      not torch.allclose(v_low, v_high, atol=1e-3),
      f"diff: {(v_low - v_high).abs().mean():.6f}")

# ============================================================
# Module 7: ContinuousDiffusion q_sample + compute_loss
# ============================================================
print("\n" + "=" * 60)
print("MODULE 7: ContinuousDiffusion")
print("=" * 60)
from egomimic.algo.dfot.continuous_diffusion import ContinuousDiffusion

diff = ContinuousDiffusion(action_dim=4, loss_weighting="uniform").to(device)

x_clean = torch.randn(8, 4, 6, 6, device=device)
t = torch.rand(8, device=device).clamp(1e-5, 1-1e-5)
q = diff.q_sample(x_clean, t)

check("q_sample returns dict with x_t", "x_t" in q)
check("q_sample returns dict with time_cond", "time_cond" in q)
check("q_sample x_t shape matches input", q["x_t"].shape == x_clean.shape)
check("q_sample time_cond shape", q["time_cond"].shape == t.shape)

# At t=0 (clean), x_t should be close to x_clean
t_zero = torch.full((8,), 1e-5, device=device)
q_clean = diff.q_sample(x_clean, t_zero)
check("q_sample at t≈0 preserves input",
      torch.allclose(q_clean["x_t"], x_clean, atol=0.01),
      f"max diff: {(q_clean['x_t'] - x_clean).abs().max():.4f}")

# At t=1 (noise), x_t should be mostly noise
t_one = torch.full((8,), 1-1e-5, device=device)
q_noisy = diff.q_sample(x_clean, t_one)
check("q_sample at t≈1 is mostly noise",
      (q_noisy["x_t"] - x_clean).abs().mean() > 0.5,
      "should be far from clean input")

# compute_loss with spatial v_pred
v_pred = torch.randn(8, 4, 6, 6, device=device)
loss = diff.compute_loss(v_pred, q)
check("compute_loss output shape (uniform)", loss.shape == v_pred.shape,
      f"got {loss.shape}")

# ============================================================
# Module 8: Latent normalization
# ============================================================
print("\n" + "=" * 60)
print("MODULE 8: Latent normalization")
print("=" * 60)
import json
from egomimic.algo.vae.algo import load_pretrained_vae

vae = load_pretrained_vae("external_ckpts/vae_v5_last.ckpt").to(device).eval()
with open("external_ckpts/vae_latent_stats.json") as f:
    ls = json.load(f)
latent_mean = torch.tensor(ls["latent_mean"]).reshape(1, -1, 1, 1).to(device)
latent_std = torch.tensor(ls["latent_std"]).reshape(1, -1, 1, 1).to(device)

imgs = torch.rand(4, 3, 96, 96, device=device)
with torch.no_grad():
    mu, _ = vae.encode(imgs)
z_norm = (mu - latent_mean) / latent_std
z_denorm = z_norm * latent_std + latent_mean

check("Normalize centers latent",
      z_norm.mean().abs() < 0.5,
      f"normalized mean: {z_norm.mean():.4f}")
check("Denormalize round-trip",
      torch.allclose(z_denorm, mu, atol=1e-5),
      f"max diff: {(z_denorm - mu).abs().max():.8f}")
with torch.no_grad():
    recon = vae.decode(z_denorm)
check("Decode(denormalize(normalize(encode(img)))) valid range",
      recon.min() >= -0.1 and recon.max() <= 1.1,
      f"range: [{recon.min():.3f}, {recon.max():.3f}]")

# ============================================================
# Module 9: Sampling (sample_step)
# ============================================================
print("\n" + "=" * 60)
print("MODULE 9: Sampling (sample_step)")
print("=" * 60)
from egomimic.algo.dfot.sampling import sample_step, vanilla_schedule, sample as _sample

schedule = vanilla_schedule(n_steps=10, T=8, discrete_timesteps=None).to(device)
check("Vanilla schedule shape", schedule.shape == (11, 8))
check("Schedule starts at 1.0", schedule[0, 0].item() == 1.0)
check("Schedule ends at 0.0", schedule[-1, 0].item() == 0.0)
check("Schedule is monotonically decreasing",
      all(schedule[i, 0] >= schedule[i+1, 0] for i in range(10)))

# sample_step should produce same shape as input
bb_small = DFoTDiT3DBackbone(
    latent_channels=4, latent_size=6, patch_size=2,
    action_horizon=32, d_model=384, d_cond=384, cond_dim=384,
    depth=2, num_heads=6, mlp_ratio=4.0,
    time_embed_dim=256, cond_emb_dim=256,
).to(device).eval()

x = torch.randn(1, 8, 4, 6, 6, device=device)
cond = torch.randn(1, 8, 384, device=device) * 0.1
with torch.no_grad():
    x_next = sample_step(diff, bb_small, x=x,
                         current_levels=schedule[0], next_levels=schedule[1],
                         external_cond=cond)
check("sample_step output shape", x_next.shape == x.shape,
      f"got {x_next.shape}")
check("sample_step output is finite", torch.isfinite(x_next).all().item())

# Full sampling should produce correct shape
with torch.no_grad():
    sampled = _sample(diff, bb_small, schedule_matrix=schedule,
                      x_shape=(4, 6, 6), batch_size=1,
                      external_cond=cond, device=device)
check("Full sample output shape", sampled.shape == (1, 8, 4, 6, 6),
      f"got {sampled.shape}")
check("Full sample output is finite", torch.isfinite(sampled).all().item())

# Sampled output should be in reasonable range (not exploded)
check("Sampled latent in reasonable range",
      sampled.abs().max() < 50,
      f"max abs: {sampled.abs().max():.2f}")

# ============================================================
# Module 10: Unpatchify correctness
# ============================================================
print("\n" + "=" * 60)
print("MODULE 10: Unpatchify correctness")
print("=" * 60)

# Create known latent, patchify, then verify unpatchify reconstructs it
latent = torch.zeros(1, 4, 6, 6, device=device)
for c in range(4):
    for h in range(6):
        for w in range(6):
            latent[0, c, h, w] = c * 100 + h * 10 + w

# Patchify
pe2 = PatchEmbed(4, 16, patch_size=2).to(device)
# Use identity-like Conv2d to preserve values
with torch.no_grad():
    pe2.proj.weight.zero_()
    pe2.proj.bias.zero_()
    for i in range(4):
        for ph in range(2):
            for pw in range(2):
                pe2.proj.weight[i*4 + ph*2 + pw, i, ph, pw] = 1.0
tokens = pe2(latent)  # (1, 9, 16)

# Unpatchify
result = rearrange(tokens, "bt (h w) (p q c) -> bt c (h p) (w q)",
                   h=3, w=3, p=2, q=2, c=4)

check("Unpatchify round-trip preserves values",
      torch.allclose(result, latent, atol=1e-5),
      f"max diff: {(result - latent).abs().max():.6f}")

# ============================================================
# Module 11: Training path end-to-end
# ============================================================
print("\n" + "=" * 60)
print("MODULE 11: Training path end-to-end")
print("=" * 60)

bb_train = DFoTDiT3DBackbone(
    latent_channels=4, latent_size=6, patch_size=2,
    action_horizon=32, d_model=384, d_cond=384, cond_dim=384,
    depth=2, num_heads=6, mlp_ratio=4.0,
    time_embed_dim=256, cond_emb_dim=256,
).to(device).train()

optimizer = torch.optim.Adam(bb_train.parameters(), lr=1e-3)
diff_train = ContinuousDiffusion(action_dim=4, loss_weighting="uniform").to(device)

# Simulate a training step
gt_latent = torch.randn(8, 4, 6, 6, device=device)  # normalized
t = torch.rand(8, device=device).clamp(1e-5, 1-1e-5)
q = diff_train.q_sample(gt_latent, t)
cond = torch.randn(1, 8, 384, device=device) * 0.1

v_pred = bb_train(q["x_t"].unsqueeze(0), q["time_cond"].unsqueeze(0), external_cond=cond)
loss = diff_train.compute_loss(v_pred.squeeze(0), q)
scalar_loss = loss.mean()

check("Training loss is finite", torch.isfinite(scalar_loss).item(),
      f"loss: {scalar_loss.item()}")
check("Training loss requires grad", scalar_loss.requires_grad)

# Backward pass
scalar_loss.backward()
grad_norm = sum(p.grad.norm().item() for p in bb_train.parameters() if p.grad is not None)
check("Gradients are finite", np.isfinite(grad_norm),
      f"grad_norm: {grad_norm}")
check("Gradients are non-zero", grad_norm > 0,
      f"grad_norm: {grad_norm}")

optimizer.step()
optimizer.zero_grad()

# After one step, loss should change
v_pred2 = bb_train(q["x_t"].unsqueeze(0), q["time_cond"].unsqueeze(0), external_cond=cond)
loss2 = diff_train.compute_loss(v_pred2.squeeze(0), q).mean()
check("Loss changes after optimizer step",
      abs(loss2.item() - scalar_loss.item()) > 1e-6,
      f"before: {scalar_loss.item():.6f}, after: {loss2.item():.6f}")

# ============================================================
# Module 12: Inference path end-to-end
# ============================================================
print("\n" + "=" * 60)
print("MODULE 12: Inference path end-to-end")
print("=" * 60)

bb_eval = DFoTDiT3DBackbone(
    latent_channels=4, latent_size=6, patch_size=2,
    action_horizon=32, d_model=384, d_cond=384, cond_dim=384,
    depth=2, num_heads=6, mlp_ratio=4.0,
    time_embed_dim=256, cond_emb_dim=256,
).to(device).eval()

T_inf = 16
cond_inf = torch.randn(1, T_inf, 384, device=device) * 0.1
schedule = vanilla_schedule(n_steps=20, T=T_inf, discrete_timesteps=None).to(device)

with torch.no_grad():
    sampled = _sample(diff, bb_eval, schedule_matrix=schedule,
                      x_shape=(4, 6, 6), batch_size=1,
                      external_cond=cond_inf, device=device)

check("Inference sample shape", sampled.shape == (1, T_inf, 4, 6, 6))
check("Inference sample finite", torch.isfinite(sampled).all().item())

# Denormalize and decode
sampled_denorm = sampled.squeeze(0) * latent_std + latent_mean
with torch.no_grad():
    decoded = vae.decode(sampled_denorm)
check("Decoded shape", decoded.shape == (T_inf, 3, 96, 96),
      f"got {decoded.shape}")
check("Decoded in valid range",
      decoded.min() >= -0.5 and decoded.max() <= 1.5,
      f"range: [{decoded.min():.3f}, {decoded.max():.3f}]")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print(f"RESULTS: {PASS} passed, {FAIL} failed")
print("=" * 60)
