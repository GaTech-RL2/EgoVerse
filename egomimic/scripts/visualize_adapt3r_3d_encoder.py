"""Visualize Adapt3R3DEncoder internals on a real RBY1 egoengine sample.

Renders a single multi-panel figure with:
  1. Input RGB
  2. DINO/ResNet backbone feature map (PCA → RGB)
  3. NeRF positional encoding at FPS points (PCA → RGB, 3D scatter)
  4. Fused per-point feature [NeRF || RGB_feat] (PCA → RGB)
  5. Attention weights over FPS points (one panel per head)
  6. Output token statistics (text)

Usage::

    cd /home/droid_robot/zhenyang/EgoVerse && source emimic/bin/activate
    python egomimic/scripts/visualize_adapt3r_3d_encoder.py \\
        --dataset /home/droid_robot/zhenyang/EgoVerse/datasets/RBY1_egoengine_mustard_cropped_right_arm_eef_hand \\
        --episode 0 --frame 0 --backbone dino --out viz_adapt3r3d.png

A PLY of the lifted point cloud is written next to the PNG when ``--save-ply``
is given.
"""

from __future__ import annotations

import argparse
import io
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

from egomimic.models.adapt3r_3d_encoder import Adapt3R3DEncoder


ARIA_K = [
    [877.96474921, 0.0, 989.30439345],
    [0.0, 877.96474921, 744.17180392],
    [0.0, 0.0, 1.0],
]
ARIA_E = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]


def load_rgb(dataset_dir: str, episode: int, frame: int) -> np.ndarray:
    parquet = os.path.join(
        dataset_dir, "data", "chunk-000", f"episode_{episode:06d}.parquet"
    )
    df = pd.read_parquet(parquet)
    img_bytes = df["obs.images"].iloc[frame]["bytes"]
    return np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))


def pca_rgb(features: np.ndarray) -> np.ndarray:
    """Project a (..., D) feature array to (..., 3) via PCA, scaled to [0,1]."""
    flat = features.reshape(-1, features.shape[-1])
    flat = flat - flat.mean(0, keepdims=True)
    # SVD-based PCA (top-3 components).
    u, s, vh = np.linalg.svd(flat, full_matrices=False)
    proj = flat @ vh[:3].T                              # [N, 3]
    lo = proj.min(0, keepdims=True)
    hi = proj.max(0, keepdims=True)
    proj = (proj - lo) / np.maximum(hi - lo, 1e-8)
    return proj.reshape(*features.shape[:-1], 3)


def save_ply(path: str, points: np.ndarray, colors: np.ndarray):
    """Tiny ASCII PLY writer (no open3d / trimesh dependency)."""
    assert points.shape[-1] == 3 and colors.shape[-1] == 3
    pts = points.reshape(-1, 3)
    cols = (np.clip(colors.reshape(-1, 3), 0, 1) * 255).astype(np.uint8)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {pts.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(pts, cols):
            f.write(f"{p[0]:.5f} {p[1]:.5f} {p[2]:.5f} {c[0]} {c[1]} {c[2]}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--backbone", default="dino", choices=["dino", "resnet18"])
    parser.add_argument("--num-points", type=int, default=256)
    parser.add_argument("--dummy-depth", type=float, default=1.0)
    parser.add_argument("--out", default="viz_adapt3r3d.png")
    parser.add_argument("--save-ply", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    # ----- Load image -----
    rgb_np = load_rgb(args.dataset, args.episode, args.frame)
    H, W, _ = rgb_np.shape
    print(f"Loaded RGB: shape={rgb_np.shape} dtype={rgb_np.dtype}")

    # ----- Build encoder -----
    enc = Adapt3R3DEncoder(
        output_dim=256, hidden_dim=60, num_points=args.num_points,
        intrinsics=ARIA_K, extrinsics=ARIA_E,
        image_size_orig=[2016, 1512], image_size_input=[W, H],
        backbone=args.backbone,
        dino_img_size=224 if args.backbone == "dino" else None,
        dummy_depth=args.dummy_depth,
    ).to(args.device)
    enc.eval()
    print(f"Encoder built (backbone={args.backbone}, feat_dim={enc.feat_dim}).")

    # ----- Forward pass with internals -----
    rgb_t = torch.from_numpy(rgb_np).permute(2, 0, 1).float() / 255.0
    x = rgb_t.unsqueeze(0).unsqueeze(0).unsqueeze(0)            # [1,1,1,3,H,W]
    depth_t = torch.zeros(1, 1, 1, 1, H, W)
    x = torch.cat([x, depth_t], dim=3).to(args.device)          # [1,1,1,4,H,W]
    with torch.no_grad():
        token, internals = enc(x, return_internals=True)
    print(f"Output token: shape={tuple(token.shape)}, "
          f"mean={token.mean().item():+.4f} std={token.std().item():.4f}")

    feat_grid = internals["feat_grid"][0, 0].cpu().numpy()      # [F, h, w]
    pcd_full = internals["pcd_full"][0, 0].cpu().numpy()        # [H, W, 3]
    pcd_fps = internals["pcd_fps"][0, 0].cpu().numpy()          # [N, 3]
    feat_fps = internals["feat_fps"][0, 0].cpu().numpy()        # [N, hidden_dim]
    attn = internals["attn"][0, 0].cpu().numpy()                # [num_heads, N]

    # ----- Compute PCA visualisations -----
    feat_grid_chw = feat_grid.transpose(1, 2, 0)                # [h, w, F]
    feat_grid_pca = pca_rgb(feat_grid_chw)                      # [h, w, 3]

    # NeRF position embedding at FPS points (recompute for visualisation).
    nerf_emb = enc.xyz_proj(torch.from_numpy(pcd_fps).unsqueeze(0)).squeeze(0).cpu().numpy()
    nerf_pca = pca_rgb(nerf_emb)                                # [N, 3]
    fused = np.concatenate([nerf_emb, feat_fps], axis=-1)
    fused_pca = pca_rgb(fused)                                  # [N, 3]

    # ----- Plot -----
    n_heads = attn.shape[0]
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(3, max(4, n_heads), height_ratios=[1, 1, 1])

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(rgb_np); ax.set_title("Input RGB"); ax.axis("off")

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(feat_grid_pca); ax.set_title(f"{args.backbone} backbone (PCA-RGB)\n"
                                           f"grid={feat_grid_pca.shape[:2]}"); ax.axis("off")

    ax = fig.add_subplot(gs[0, 2], projection="3d")
    ax.scatter(pcd_fps[:, 0], pcd_fps[:, 1], pcd_fps[:, 2], c=nerf_pca, s=8)
    ax.set_title(f"NeRF pos-emb @ {pcd_fps.shape[0]} FPS pts (PCA-RGB)")

    ax = fig.add_subplot(gs[0, 3], projection="3d")
    ax.scatter(pcd_fps[:, 0], pcd_fps[:, 1], pcd_fps[:, 2], c=fused_pca, s=8)
    ax.set_title("Fused [pos | rgb] (PCA-RGB)")

    # ----- Attention heads -----
    # Project FPS points to image plane via the rescaled K to overlay attention as 2D scatter.
    K_scaled = enc.K.squeeze().cpu().numpy()
    fx, fy = K_scaled[0, 0], K_scaled[1, 1]
    cx, cy = K_scaled[0, 2], K_scaled[1, 2]
    # Avoid division by zero / negative depth (should all be `dummy_depth`).
    z = np.maximum(pcd_fps[:, 2], 1e-6)
    u = pcd_fps[:, 0] * fx / z + cx
    v = pcd_fps[:, 1] * fy / z + cy

    for h in range(n_heads):
        ax = fig.add_subplot(gs[1, h])
        ax.imshow(rgb_np, alpha=0.6)
        sc = ax.scatter(u, v, c=attn[h], cmap="hot", s=18, vmin=0, vmax=attn[h].max())
        ax.set_title(f"Attention head {h}")
        ax.axis("off")
        plt.colorbar(sc, ax=ax, fraction=0.046)

    # ----- Token stats panel -----
    ax = fig.add_subplot(gs[2, :])
    ax.axis("off")
    token_np = token[0, 0].cpu().numpy()
    text = (
        f"Output token shape: {tuple(token.shape)}\n"
        f"Output token: mean={token.mean().item():+.4f}  std={token.std().item():.4f}  "
        f"min={token.min().item():+.4f}  max={token.max().item():+.4f}\n"
        f"Backbone: {args.backbone} (feat_dim={enc.feat_dim})  | "
        f"hidden_dim={enc.hidden_dim}  num_points={pcd_fps.shape[0]}  "
        f"dummy_depth={args.dummy_depth} m\n"
        f"Aria K rescaled: fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f} "
        f"(orig 2016x1512 → input {W}x{H})"
    )
    ax.text(0.01, 0.5, text, fontsize=10, family="monospace", verticalalignment="center")

    plt.tight_layout()
    plt.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"Saved figure → {args.out}")

    if args.save_ply:
        ply_path = args.out.rsplit(".", 1)[0] + ".ply"
        # Lifted full-res cloud, coloured by RGB.
        rgb01 = rgb_np.astype(np.float32) / 255.0
        save_ply(ply_path, pcd_full.reshape(-1, 3), rgb01.reshape(-1, 3))
        print(f"Saved PLY → {ply_path}")


if __name__ == "__main__":
    main()
