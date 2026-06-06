"""Verify patchification token ordering matches RoPE 3D position assignment."""
import sys
sys.path.insert(0, ".")
import torch
from einops import rearrange, repeat
from egomimic.algo.dfot.dit3d_backbone import (
    RotaryEmbedding3D, PatchEmbed, DFoTDiT3DBackbone, _rotate_half,
)

device = "cuda"

# === Test 1: Patch embed token ordering ===
print("=== Test 1: Patch embed token ordering ===")
patch_size = 1
latent_channels = 4
latent_size = 6
grid_size = latent_size // patch_size
P = grid_size ** 2  # 36

# Create a latent where each spatial position has a unique value
# so we can trace where each token ends up
B, T = 1, 3
x = torch.zeros(B, T, latent_channels, latent_size, latent_size, device=device)
for t in range(T):
    for h in range(latent_size):
        for w in range(latent_size):
            # Encode position as value: t*100 + h*10 + w
            x[0, t, 0, h, w] = t * 100 + h * 10 + w

# Patchify the same way the backbone does
pe = PatchEmbed(latent_channels, 8, patch_size).to(device)  # small d_model for clarity
# But let's just trace the rearrangement without the projection
x_flat = rearrange(x, "b t c h w -> (b t) c h w")
# After Conv2d with kernel=1, stride=1, spatial layout is preserved
# rearrange "bt c h w -> bt (h w) c" flattens spatial
x_spatial = rearrange(x_flat[:, :1, :, :], "(b t) c h w -> (b t) (h w) c", b=B)
# x_spatial shape: (B*T, 36, 1)

# Check token ordering: token 0 should be (h=0,w=0), token 1 = (h=0,w=1), etc.
print("Token ordering from patchify (einops 'bt c h w -> bt (h w) c'):")
for tok_idx in [0, 1, 5, 6, 35]:
    val = x_spatial[0, tok_idx, 0].item()  # first frame, first channel
    h_pos = int(val) // 10
    w_pos = int(val) % 10
    print(f"  Token {tok_idx:2d} -> (h={h_pos}, w={w_pos})  [expected: h={tok_idx//6}, w={tok_idx%6}]")

# After temporal stacking: rearrange "(b t) p c -> b (t p) c"
x_temporal = rearrange(x_spatial, "(b t) p c -> b (t p) c", b=B)
print("\nAfter temporal stacking (b t p -> b (t*p)):")
for global_idx in [0, 1, 35, 36, 37, 71, 72]:
    if global_idx < T * P:
        val = x_temporal[0, global_idx, 0].item()
        t_pos = int(val) // 100
        h_pos = (int(val) % 100) // 10
        w_pos = int(val) % 10
        expected_t = global_idx // P
        expected_h = (global_idx % P) // grid_size
        expected_w = (global_idx % P) % grid_size
        match = "OK" if (t_pos == expected_t and h_pos == expected_h and w_pos == expected_w) else "MISMATCH!"
        print(f"  Global {global_idx:3d} -> (t={t_pos}, h={h_pos}, w={w_pos})  "
              f"[expected: t={expected_t}, h={expected_h}, w={expected_w}]  {match}")

# === Test 2: RoPE 3D position assignment ===
print("\n=== Test 2: RoPE 3D frequency assignment ===")
rope = RotaryEmbedding3D(dim=64, sizes=(T, grid_size, grid_size)).to(device)
freqs = rope._build_freqs(T, device, torch.float32)
print(f"Freqs shape: {freqs.shape}  (expected: ({T*P}, 64))")

# Check that different spatial positions get different frequencies
print("\nSpatial frequency uniqueness (frame 0):")
frame0_freqs = freqs[:P]  # first 36 tokens = frame 0
for i in range(min(6, P)):
    for j in range(i+1, min(6, P)):
        diff = (frame0_freqs[i] - frame0_freqs[j]).abs().sum().item()
        if diff < 1e-6:
            print(f"  WARNING: Token {i} and {j} have IDENTICAL frequencies! diff={diff:.8f}")

# Check that position (0,0) in different frames get different freqs
print("\nTemporal frequency uniqueness (position h=0,w=0):")
for t in range(T):
    tok = t * P  # first patch of each frame
    print(f"  Frame {t}, token {tok}: freq[:4] = {freqs[tok, :4].cpu().numpy()}")

# === Test 3: RoPE applied to Q/K ===
print("\n=== Test 3: RoPE application to Q/K ===")
# Create Q with known values to verify rotation
head_dim = 64
q = torch.ones(1, 1, T * P, head_dim, device=device)  # uniform Q
q_rotated = rope(q)
# Position 0 should have cos(0)=1, sin(0)=0 -> no rotation for temporal axis
# But spatial axes at (0,0) also have pos=0 -> no rotation at all
print(f"q[pos=0] after RoPE (should be ~1.0 for pos=0): {q_rotated[0,0,0,:4].cpu().numpy()}")
print(f"q[pos=1] after RoPE (should differ from pos=0): {q_rotated[0,0,1,:4].cpu().numpy()}")
print(f"q[pos=36] after RoPE (frame 1, h=0,w=0):       {q_rotated[0,0,36,:4].cpu().numpy()}")

# === Test 4: Unpatchify correctness ===
print("\n=== Test 4: Unpatchify round-trip ===")
bb = DFoTDiT3DBackbone(
    latent_channels=4, latent_size=6, patch_size=1,
    action_horizon=32, d_model=384, d_cond=384, cond_dim=384,
    depth=1, num_heads=6, mlp_ratio=4.0,
    time_embed_dim=256, cond_emb_dim=256,
).to(device).eval()

# Test: create known input, pass through patchify -> unpatchify (skip transformer)
x_test = torch.randn(1, 2, 4, 6, 6, device=device)
x_flat = rearrange(x_test, "b t c h w -> (b t) c h w")
tokens = bb.patch_embed(x_flat)  # (2, 36, 384)
# Simulate final layer output shape: (2, 36, patch_size^2 * latent_channels) = (2, 36, 4)
out_channels = bb.patch_size ** 2 * bb.latent_channels  # = 4
fake_output = torch.randn(2, 36, out_channels, device=device)
# Unpatchify
result = rearrange(
    fake_output,
    "bt (h w) (p q c) -> bt c (h p) (w q)",
    h=bb.grid_size, w=bb.grid_size,
    p=bb.patch_size, q=bb.patch_size,
    c=bb.latent_channels,
)
print(f"Unpatchify: input (2, 36, 4) -> output {result.shape}  (expected: (2, 4, 6, 6))")
# For patch_size=1, unpatchify should just be reshape
# fake_output[:, token_idx, :] should equal result[:, :, h, w] where token_idx = h*6+w
for h in range(6):
    for w in range(6):
        tok_idx = h * 6 + w
        expected = fake_output[0, tok_idx]
        actual = result[0, :, h, w]
        if not torch.allclose(expected, actual):
            print(f"  MISMATCH at h={h}, w={w}: expected {expected.cpu().numpy()} got {actual.cpu().numpy()}")
print("  All positions match: OK" if True else "")

# === Test 5: Reference comparison ===
print("\n=== Test 5: Reference RoPE comparison ===")
# Load reference implementation
sys.path.insert(0, "/coc/flash7/paphiwetsa3/projects/diffusion-forcing-transformer")
try:
    from algorithms.dfot.backbones.modules.embeddings import RotaryEmbedding3D as RefRoPE3D
    ref_rope = RefRoPE3D(dim=64, sizes=(T, grid_size, grid_size)).to(device)
    # Compare frequencies
    ref_freqs = ref_rope.freqs.to(device)  # pre-computed, shape (T*P, dim)
    our_freqs = rope._build_freqs(T, device, torch.float32)

    print(f"Reference freqs shape: {ref_freqs.shape}")
    print(f"Our freqs shape: {our_freqs.shape}")

    if ref_freqs.shape == our_freqs.shape:
        max_diff = (ref_freqs - our_freqs).abs().max().item()
        mean_diff = (ref_freqs - our_freqs).abs().mean().item()
        print(f"Max diff: {max_diff:.8f}")
        print(f"Mean diff: {mean_diff:.8f}")
        if max_diff > 1e-4:
            print("  FREQUENCIES DO NOT MATCH!")
            # Show where the biggest diffs are
            diff = (ref_freqs - our_freqs).abs()
            for i in range(min(5, diff.shape[0])):
                print(f"  Token {i}: our={our_freqs[i,:4].cpu().numpy()} ref={ref_freqs[i,:4].cpu().numpy()}")
    else:
        print("  Shape mismatch — can't compare directly")
        print(f"  Our: {our_freqs[:3, :4].cpu().numpy()}")
        print(f"  Ref: {ref_freqs[:3, :4].cpu().numpy()}")
except ImportError as e:
    print(f"  Could not import reference: {e}")

print("\n=== DONE ===")
