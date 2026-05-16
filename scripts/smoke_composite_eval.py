"""
Smoke test for the composite eval: HNetEvalVideo (trajectory overlay) +
PCATokenEval + BoundaryStripEval, all stacked side-by-side via
``EvalListSideBySide``. Saves one .mp4 with three panels per frame.

Run on a chunker-based H-Net checkpoint (baseline AdaLN). The boundary
strip only fills in for chunker-based architectures; FlatFusedPolicy
runs would show the same panel as empty.

Usage:
    python scripts/smoke_composite_eval.py \\
        --ckpt logs/.../checkpoints/epoch=149-step=1200.ckpt \\
        --config-path logs/.../.hydra/config.yaml \\
        --out-dir composite_smoke_out
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torchvision.io as tvio
from hydra.utils import instantiate
from omegaconf import OmegaConf

from egomimic.eval.eval_boundary_strip import BoundaryStripEval
from egomimic.eval.eval_composite import EvalListSideBySide
from egomimic.eval.eval_hnet import HNetEvalVideo
from egomimic.eval.eval_pca_tokens import PCATokenEval
from egomimic.rldb.embodiment.pushshapes import viz_gt_preds


def load_algo_from_ckpt(ckpt_path: str, config_path: str):
    print(f"[load] ckpt: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hparams = ckpt.get("hyper_parameters") or ckpt.get("hparams") or {}
    cfg_for_model = (
        OmegaConf.create(hparams["config_tree"])
        if "config_tree" in hparams
        else OmegaConf.load(config_path)
    )
    from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset

    norm_state = hparams.get("norm_stats_state")
    if norm_state is None:
        raise SystemExit("hyper_parameters has no norm_stats_state")
    norm_stats = MultiDataset.from_state(norm_state)
    algo = instantiate(cfg_for_model.model.robomimic_model, norm_stats=norm_stats)

    state_dict = ckpt["state_dict"]
    new_sd = {}
    for k, v in state_dict.items():
        for prefix in ("nets.", "model.nets."):
            if k.startswith(prefix):
                new_sd[k[len(prefix) :]] = v
                break
        else:
            new_sd[k] = v
    algo.nets.load_state_dict(new_sd, strict=False)
    return algo


class _MockTrainer:
    def __init__(self, output_dir, device):
        self.current_epoch = 0
        self.is_global_zero = True
        self.lightning_module = type("M", (), {"device": device})()
        self.default_root_dir = output_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--out-dir", default="composite_smoke_out")
    parser.add_argument("--window", type=int, default=64)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    algo = load_algo_from_ckpt(args.ckpt, args.config_path)
    algo.nets = algo.nets.to(device)
    algo.device = device
    algo.nets.eval()

    full_cfg = OmegaConf.load(args.config_path)
    dm = instantiate(full_cfg.data)
    dm.setup(stage="validate")
    val_loader = dm.val_dataloader()
    first = next(iter(val_loader))
    batch = first[0] if isinstance(first, tuple) else first
    batch = algo.process_batch_for_training(batch)

    composite = EvalListSideBySide(
        evals=[
            HNetEvalVideo(
                limit_val_batches=1,
                viz_func={"pushshapes_sim": viz_gt_preds},
            ),
            PCATokenEval(limit_val_batches=1, panel_h=384, panel_w=384),
            BoundaryStripEval(
                limit_val_batches=1,
                window=args.window,
                strip_width=24,
                pixels_per_step=1,
            ),
        ],
        pad_h="min",
        limit_val_batches=1,
    )
    composite.trainer = _MockTrainer(args.out_dir, device)
    composite.model = algo

    print("[viz] running composite eval...")
    metrics, images_dict = composite.compute_metrics_and_viz(batch)
    print("[metrics]")
    for k, v in metrics.items():
        try:
            print(f"  {k}: {float(v):.4f}")
        except Exception:
            print(f"  {k}: {v}")

    for emb_id, ims in images_dict.items():
        path = out_dir / f"composite_emb{emb_id}.mp4"
        tvio.write_video(str(path), torch.from_numpy(ims), fps=30, video_codec="h264")
        print(f"  wrote {path}  shape={ims.shape}")


if __name__ == "__main__":
    main()
