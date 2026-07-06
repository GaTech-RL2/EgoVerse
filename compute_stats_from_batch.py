"""Compute VAE latent stats by loading a training batch through the dataloader."""
import sys
sys.path.insert(0, ".")
import torch
import json
import hydra
from omegaconf import OmegaConf

device = "cuda"

from egomimic.algo.diffusion.vae_algo import load_pretrained_vae
vae = load_pretrained_vae("external_ckpts/vae_v5_last.ckpt").to(device).eval()

# Load a checkpoint to get norm_stats_state and config
ckpt = torch.load(
    "logs/dit3d_both_fixes/dit3d_nopad_uniform_addfusion_2026-05-24_20-44-33/checkpoints/epoch_epoch=199.ckpt",
    map_location="cpu", weights_only=False,
)
hp = ckpt["hyper_parameters"]
config_tree = hp["config_tree"]
norm_stats_state = hp["norm_stats_state"]

# Instantiate the data module to get a real batch
data_cfg = config_tree["data"]
# Override paths for skynet
data_cfg["train_datasets"]["pushshapes_sim"]["resolver"]["folder_path"] = \
    "/coc/flash7/paphiwetsa3/datasets/pushT/circle_750/circle"
dm = hydra.utils.instantiate(data_cfg)
dm.setup("fit")
train_dl = dm.train_dataloader()

# Get one batch
batch = next(iter(train_dl))
# The batch has emb_id keys; find the image
for key in batch:
    if isinstance(batch[key], dict):
        sub = batch[key]
        img_keys = [k for k in sub if "img" in k.lower()]
        if img_keys:
            imgs = sub[img_keys[0]].to(device).float()
            if imgs.max() > 1.5:
                imgs = imgs / 255.0
            # Handle packed: might be (T_total, 3, H, W) or (T_total, H, W, 3)
            if imgs.dim() == 4 and imgs.shape[-1] == 3:
                imgs = imgs.permute(0, 3, 1, 2)
            print(f"Images: shape={imgs.shape}, range=[{imgs.min():.3f}, {imgs.max():.3f}]")
            with torch.no_grad():
                mu, _ = vae.encode(imgs)
            print(f"Latent shape: {mu.shape}")
            for c in range(mu.shape[1]):
                print(f"  Ch {c}: mean={mu[:,c].mean():.6f} std={mu[:,c].std():.6f}")
            stats = {
                "latent_mean": [mu[:, c].mean().item() for c in range(mu.shape[1])],
                "latent_std": [mu[:, c].std().item() for c in range(mu.shape[1])],
            }
            with open("external_ckpts/vae_latent_stats.json", "w") as f:
                json.dump(stats, f, indent=2)
            print(f"Saved to external_ckpts/vae_latent_stats.json")
            break
