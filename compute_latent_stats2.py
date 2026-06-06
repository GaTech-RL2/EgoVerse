"""Compute per-channel VAE latent stats using simplejpeg to decode images."""
import sys
sys.path.insert(0, ".")
import torch
import numpy as np
device = "cuda"

from egomimic.algo.diffusion.vae_algo import load_pretrained_vae
vae = load_pretrained_vae("external_ckpts/vae_v5_last.ckpt").to(device).eval()

# Use simplejpeg to decode the JPEG-stored images from zarr
import zarr
import simplejpeg

zarr_dir = "/coc/flash7/paphiwetsa3/datasets/pushT/circle_750/circle"
import glob, os
zarr_paths = sorted(glob.glob(os.path.join(zarr_dir, "*.zarr")))[:30]

all_mu = []
for zp in zarr_paths:
    z = zarr.open_group(zp, mode="r")
    raw = z["observations.images.front_img_1"][:]  # object array of JPEG bytes
    frames = []
    for j in range(min(20, len(raw))):
        try:
            img = simplejpeg.decode_jpeg(raw[j].tobytes(), colorspace="RGB")
            frames.append(img)
        except:
            continue
    if not frames:
        continue
    imgs = np.stack(frames)  # (N, H, W, 3)
    imgs_t = torch.from_numpy(imgs).float().permute(0, 3, 1, 2).to(device) / 255.0
    with torch.no_grad():
        mu, _ = vae.encode(imgs_t)
    all_mu.append(mu.cpu())

all_mu = torch.cat(all_mu, dim=0)
print(f"Total frames: {all_mu.shape[0]}")
for c in range(4):
    m = all_mu[:, c].mean().item()
    s = all_mu[:, c].std().item()
    print(f"  Ch {c}: mean={m:.6f} std={s:.6f}")
print(f"  Overall: mean={all_mu.mean():.6f} std={all_mu.std():.6f}")

# Save as JSON for use in training
import json
stats = {
    "latent_mean": [all_mu[:, c].mean().item() for c in range(4)],
    "latent_std": [all_mu[:, c].std().item() for c in range(4)],
}
with open("external_ckpts/vae_latent_stats.json", "w") as f:
    json.dump(stats, f, indent=2)
print(f"Saved to external_ckpts/vae_latent_stats.json")
