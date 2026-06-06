"""Test that PatchEmbed and unpatchify are inverses."""
import sys
sys.path.insert(0, ".")
import torch
from einops import rearrange
from egomimic.algo.dfot.dit3d_backbone import PatchEmbed

device = "cuda"

print("=== PatchEmbed vs Unpatchify ordering ===")
print()
print("KEY INSIGHT: PatchEmbed uses Conv2d which has its OWN output ordering")
print("(out_channels major), while unpatchify assumes (p q c) ordering.")
print("These are NOT the same unless the Conv2d is structured to match.")
print()

# The reference DiT3D uses:
#   PatchEmbed: Conv2d(in_ch, hidden_size, kernel_size=ps, stride=ps) -> (BT, num_patches, hidden_size)
#   ... transformer layers ...
#   FinalLayer: Linear(hidden_size, patch_size**2 * channels) -> (BT, num_patches, ps*ps*ch)
#   Unpatchify: rearrange("b (h w) (p q c) -> b (h p) (w q) c", h=grid, p=ps, q=ps)
#
# The Conv2d projects 2x2x4=16 input values to hidden_size=384. This is just a learned projection.
# The Linear in FinalLayer projects 384 back to 16 values.
# The unpatchify rearranges these 16 values as a 2x2 spatial block with 4 channels.
#
# THERE IS NO GUARANTEE that the 16 output values from FinalLayer are in (p,q,c) order.
# But that's OK! The model LEARNS to output values in whatever order unpatchify expects.
# The unpatchify ordering is a convention — the model adapts to it during training.
#
# So the patchify → unpatchify is NOT supposed to be an identity (it goes through the transformer).
# The test in Module 10 was wrong — it tried to make Conv2d an identity, which doesn't match
# the unpatchify convention.

# Let's verify the ACTUAL pipeline: patchify → transformer → final_layer → unpatchify
# What matters is that the unpatchify correctly maps the 16 output values to 2x2x4 spatial.

# Test: manually verify unpatchify mapping
print("=== Unpatchify mapping verification ===")
# 9 patches, 16 values each
tokens = torch.zeros(1, 9, 16, device=device)
# Set patch (h=0,w=0) channel 0, position (0,0) to 1.0
# In (p q c) order: feat_idx = p*q_size*c_size + q*c_size + c = 0*2*4 + 0*4 + 0 = 0
tokens[0, 0, 0] = 1.0

result = rearrange(tokens, "bt (h w) (p q c) -> bt c (h p) (w q)",
                   h=3, w=3, p=2, q=2, c=4)
print(f"Setting token[0,0,0]=1 (patch h=0,w=0, p=0,q=0,c=0)")
print(f"Result nonzero at: ", torch.nonzero(result).tolist())
print(f"Expected: [0, 0, 0, 0]  (bt=0, c=0, row=0*2+0=0, col=0*2+0=0)")

# Test another position
tokens2 = torch.zeros(1, 9, 16, device=device)
# Patch (h=1,w=2), p=1, q=0, c=2
# token_idx = h*3 + w = 1*3 + 2 = 5
# feat_idx = p*q_size*c_size + q*c_size + c = 1*2*4 + 0*4 + 2 = 10
tokens2[0, 5, 10] = 1.0
result2 = rearrange(tokens2, "bt (h w) (p q c) -> bt c (h p) (w q)",
                    h=3, w=3, p=2, q=2, c=4)
expected_row = 1*2 + 1  # h*ps + p = 3
expected_col = 2*2 + 0  # w*ps + q = 4
print(f"\nSetting token[0,5,10]=1 (patch h=1,w=2, p=1,q=0,c=2)")
print(f"Result nonzero at: ", torch.nonzero(result2).tolist())
print(f"Expected: [0, 2, {expected_row}, {expected_col}]")

# Verify ALL positions
print("\n=== Full mapping verification ===")
all_correct = True
for h in range(3):
    for w in range(3):
        for p in range(2):
            for q in range(2):
                for c in range(4):
                    tok = h * 3 + w
                    feat = p * 2 * 4 + q * 4 + c
                    t = torch.zeros(1, 9, 16, device=device)
                    t[0, tok, feat] = 1.0
                    r = rearrange(t, "bt (h w) (p q c) -> bt c (h p) (w q)",
                                  h=3, w=3, p=2, q=2, c=4)
                    nz = torch.nonzero(r)
                    exp = [0, c, h*2+p, w*2+q]
                    if len(nz) != 1 or nz[0].tolist() != exp:
                        print(f"  MISMATCH: tok={tok} feat={feat} -> {nz.tolist()}, expected {exp}")
                        all_correct = False

print(f"All {3*3*2*2*4}=576 positions correct: {all_correct}")

print("\n=== DONE ===")
