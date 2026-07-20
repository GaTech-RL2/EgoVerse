"""Add the EgoWAM-style world-model target column — DINOv3 edition.

Per frame: DINOv3-B/16 patch features of obs.aria_image (14x14x768, prefix
tokens dropped), adaptive-avg-pooled to a 4x4 spatial grid, per-token LayerNorm
(RAE-style), flattened to 12,288-D -> new column `obs.dino_wm`.

Same downstream contract as add_dino_wm_target.py (delta_timestamps [1.0] aux
target, 768-d world head). DINOv2 is retired: the lab's ResNet paper finds
DINOv3 the embodiment/viewpoint-invariant version to standardize on.

Usage:
  python egomimic/scripts/egoengine_process/add_dino_wm_target_v3.py \
      datasets/aria_fullpp --output-path datasets/aria_fullpp_wam3
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset

GRID = 4          # 14x14 patch grid adaptive-avg-pooled to GRID x GRID
FEAT = 768        # DINOv3-B token dim
OUT_COL = "obs.dino_wm"
FEATURE_SPEC = {OUT_COL: {"dtype": "float32", "shape": [GRID * GRID * FEAT], "names": [OUT_COL]}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_path")
    ap.add_argument("--output-path", required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()
    root, out = Path(args.dataset_path).resolve(), Path(args.output_path).resolve()
    out.mkdir(parents=True, exist_ok=True)

    import timm
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = timm.create_model("vit_base_patch16_dinov3", pretrained=True,
                            num_classes=0, img_size=224).to(device).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    info = json.loads((root / "meta" / "info.json").read_text())
    chunks_size = info.get("chunks_size", 1000)
    ds = load_dataset("parquet", data_dir=str(root / "data"), split="train")

    def to_img(x):
        img = np.asarray(x)
        if img.ndim == 3 and img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        if img.max() <= 1.0:
            img = img * 255
        return img.astype(np.uint8)

    feats_all = np.empty((len(ds), GRID * GRID * FEAT), dtype=np.float32)
    with torch.inference_mode():
        for lo in range(0, len(ds), args.batch_size):
            batch = [to_img(x) for x in ds[lo:lo + args.batch_size]["obs.aria_image"]]
            x = torch.from_numpy(np.stack(batch)).permute(0, 3, 1, 2).float().to(device) / 255.0
            x = (x - mean) / std
            f = net.forward_features(x)[:, net.num_prefix_tokens:]          # (B, 196, 768)
            B = f.shape[0]
            f = f.reshape(B, 14, 14, FEAT).permute(0, 3, 1, 2)               # (B, 768, 14, 14)
            f = torch.nn.functional.adaptive_avg_pool2d(f, GRID)             # (B, 768, 4, 4)
            f = f.permute(0, 2, 3, 1).reshape(B, GRID * GRID, FEAT)          # (B, 16, 768)
            f = torch.nn.functional.layer_norm(f, (FEAT,))                   # per-token LN (RAE)
            feats_all[lo:lo + B] = f.reshape(B, -1).float().cpu().numpy()
            if lo % (args.batch_size * 20) == 0:
                print(f"{lo}/{len(ds)}", flush=True)

    ds = ds.add_column(OUT_COL, [row.tolist() for row in feats_all])

    ep_to_indices = {}
    for i, ev in enumerate(ds["episode_index"]):
        ep = int(ev[0]) if hasattr(ev, "__len__") else int(ev)
        ep_to_indices.setdefault(ep, []).append(i)
    data_out = out / "data"
    for ep in sorted(ep_to_indices):
        chunk_dir = data_out / f"chunk-{ep // chunks_size:03d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        ds.select(ep_to_indices[ep]).to_parquet(str(chunk_dir / f"episode_{ep:06d}.parquet"))

    shutil.copytree(root / "meta", out / "meta", dirs_exist_ok=True)
    info = json.loads((out / "meta" / "info.json").read_text())
    info["features"].update(FEATURE_SPEC)
    (out / "meta" / "info.json").write_text(json.dumps(info, indent=4))
    print(f"Saved WAM-v3 dataset -> {out}  (+{OUT_COL}: {GRID*GRID*FEAT}-D per frame, DINOv3-B)")


if __name__ == "__main__":
    main()
