"""
Smoke test for the teacher-forced eval video.

Loads a trained checkpoint, runs ``HNetEvalVideo.compute_metrics_and_viz``
on the first val batch, and writes the per-frame video showing each
episode playing back with the GT (green) + predicted (red) trajectory
overlay.

Run:
    python scripts/smoke_teacher_eval.py \\
        --ckpt logs/.../checkpoints/epoch=99-step=800.ckpt \\
        --config-path logs/.../.hydra/config.yaml \\
        --out-dir teacher_eval_smoke_out
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torchvision.io as tvio
from hydra.utils import instantiate
from omegaconf import OmegaConf

from egomimic.eval.eval_hnet import HNetEvalVideo


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
    missing, unexpected = algo.nets.load_state_dict(new_sd, strict=False)
    if missing:
        print(f"[load] missing keys ({len(missing)}): {missing[:3]}")
    if unexpected:
        print(f"[load] unexpected keys ({len(unexpected)}): {unexpected[:3]}")
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
    parser.add_argument("--out-dir", default="teacher_eval_smoke_out")
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
    print(
        f"[batch] keys per emb: {[(k, list(v.keys())[:5]) for k, v in batch.items()]}"
    )

    # Build the eval and stub the Lightning trainer.
    from egomimic.rldb.embodiment.pushshapes import viz_gt_preds

    eval = HNetEvalVideo(
        limit_val_batches=1,
        viz_func={"pushshapes_sim": viz_gt_preds},
    )
    eval.trainer = _MockTrainer(args.out_dir, device)
    eval.model = algo

    print("[viz] running teacher-forced eval...")
    metrics, images_dict = eval.compute_metrics_and_viz(batch)
    print("[metrics]")
    for k, v in metrics.items():
        try:
            print(f"  {k}: {float(v):.4f}")
        except Exception:
            print(f"  {k}: {v}")

    for emb_id, ims in images_dict.items():
        path = out_dir / f"teacher_eval_emb{emb_id}.mp4"
        tvio.write_video(str(path), torch.from_numpy(ims), fps=30, video_codec="h264")
        print(f"  wrote {path}  shape={ims.shape}")


if __name__ == "__main__":
    main()
