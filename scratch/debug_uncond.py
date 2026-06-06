"""Check if the model produces coherent output unconditionally."""
import sys
sys.path.insert(0, ".")
import torch
import numpy as np
import cv2

device = "cuda"

ckpt = torch.load(
    "logs/dit3d_both_fixes/dit3d_nopad_uniform_addfusion_2026-05-24_20-44-33/checkpoints/epoch_epoch=199.ckpt",
    map_location="cpu", weights_only=False,
)
state_dict = ckpt["state_dict"]

from egomimic.algo.dfot.dit3d_backbone import DFoTDiT3DBackbone
from egomimic.algo.dfot.continuous_diffusion import ContinuousDiffusion
from egomimic.algo.dfot.sampling import sample as _sample, vanilla_schedule
from egomimic.algo.vae.algo import load_pretrained_vae

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

T = 16

# Sample with ZERO conditioning (unconditional)
print("=== Sampling with ZERO cond ===")
cond_zero = torch.zeros(1, T, 384, device=device)
schedule = vanilla_schedule(n_steps=100, T=T, discrete_timesteps=None).to(device)
with torch.no_grad():
    out_zero = _sample(diff, bb, schedule_matrix=schedule, x_shape=(4, 6, 6),
                       batch_size=1, external_cond=cond_zero, device=device)
    decoded_zero = vae.decode(out_zero.squeeze(0))
print(f"  Latent: mean={out_zero.mean():.4f}, std={out_zero.std():.4f}")
print(f"  Decoded: mean={decoded_zero.mean():.4f}, std={decoded_zero.std():.4f}")

# Sample with force_uncond (all cond dropped via backbone)
print("\n=== Sampling with force_uncond ===")
class UnconBB(torch.nn.Module):
    def __init__(self, bb):
        super().__init__()
        self.bb = bb
    def forward(self, x, noise_levels, external_cond=None, **kw):
        return self.bb(x, noise_levels, external_cond=external_cond, force_uncond=True, **kw)

uncond_bb = UnconBB(bb)
cond_dummy = torch.randn(1, T, 384, device=device) * 0.1
with torch.no_grad():
    out_uncond = _sample(diff, uncond_bb, schedule_matrix=schedule, x_shape=(4, 6, 6),
                         batch_size=1, external_cond=cond_dummy, device=device)
    decoded_uncond = vae.decode(out_uncond.squeeze(0))
print(f"  Latent: mean={out_uncond.mean():.4f}, std={out_uncond.std():.4f}")
print(f"  Decoded: mean={decoded_uncond.mean():.4f}, std={decoded_uncond.std():.4f}")

# Save a few frames as images for visual inspection
print("\n=== Saving frames ===")
for i, (name, decoded) in enumerate([("zero_cond", decoded_zero), ("force_uncond", decoded_uncond)]):
    for f_idx in [0, 7, 15]:
        if f_idx < decoded.shape[0]:
            frame = decoded[f_idx].clamp(0, 1).cpu().numpy()
            frame = (frame * 255).astype(np.uint8).transpose(1, 2, 0)
            frame = cv2.resize(frame, (384, 384), interpolation=cv2.INTER_NEAREST)
            path = f"/tmp/debug_{name}_frame{f_idx}.png"
            cv2.imwrite(path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            print(f"  Saved {path}")

# Also save a GT decoded frame for reference
print("\n=== GT reference ===")
gt_img = torch.rand(1, 3, 96, 96, device=device)
with torch.no_grad():
    mu, _ = vae.encode(gt_img)
    gt_recon = vae.decode(mu)
    print(f"  GT recon: mean={gt_recon.mean():.4f}, std={gt_recon.std():.4f}")

print("\n=== DONE ===")
