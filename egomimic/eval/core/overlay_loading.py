"""Standalone teacher-forced ACTION OVERLAY renderer for any packed algo
(H-Net or WindowedBC). Avoids the Hydra `=`-in-ckpt-path override issue of
mode=eval by loading the ckpt directly (reuses ckpt_loading.load_algo_from_ckpt).

Runs HNetEvalVideo (GT-vs-pred action overlay via pushshapes.viz_gt_preds)
on the first val batch and writes the overlay mp4(s).

  python -m egomimic.eval.core.overlay_loading \
     --ckpt <run>/checkpoints/epoch_epoch=1499.ckpt \
     --config-path <run>/.hydra/config.yaml --n-episodes 4 --out-dir tf_overlay
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torchvision.io as tvio
from hydra.utils import get_method
from omegaconf import OmegaConf
from hydra.utils import instantiate

from egomimic.eval.core.ckpt_loading import _MockTrainer, load_algo_from_ckpt
from egomimic.eval.core.eval_hnet import HNetEvalVideo
from egomimic.rldb.embodiment.pushshapes import viz_gt_preds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--config-path", required=True)
    p.add_argument("--n-episodes", type=int, default=4)
    p.add_argument("--out-dir", default="tf_overlay")
    args = p.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    algo, _ = load_algo_from_ckpt(args.ckpt, args.config_path)
    algo.nets = algo.nets.to(device); algo.device = device; algo.nets.eval()

    full_cfg = OmegaConf.load(args.config_path)
    dm = instantiate(full_cfg.data); dm.setup(stage="validate")
    first = next(iter(dm.val_dataloader()))
    batch = first[0] if isinstance(first, tuple) else first
    batch = algo.process_batch_for_training(batch)

    ev = HNetEvalVideo(
        limit_val_batches=args.n_episodes,
        max_videos=args.n_episodes,
        viz_func={"pushshapes_sim": viz_gt_preds, "pushshapes_sim_small_circle": viz_gt_preds},
    )
    ev.trainer = _MockTrainer(str(out), device)
    ev.model = algo
    metrics, images = ev.compute_metrics_and_viz(batch)

    print("\n=== METRICS ===")
    for k, v in metrics.items():
        try: print(f"  {k}: {float(v):.4f}")
        except Exception: pass
    for emb_id, ims in images.items():
        if getattr(ims, "size", 0) == 0:
            print(f"  emb{emb_id}: no frames"); continue
        f = out / f"tf_overlay_emb{emb_id}.mp4"
        tvio.write_video(str(f), torch.from_numpy(np.ascontiguousarray(ims)),
                         fps=30, video_codec="h264")
        print(f"  wrote {f}  shape={ims.shape}")
    print("=== DONE ===")


if __name__ == "__main__":
    main()
