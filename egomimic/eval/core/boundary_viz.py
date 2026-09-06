"""Render the H-Net chunk-length / boundary-strip viz (BoundaryStripEval, inside
the run's EvalVideoList composite) from a checkpoint, offline. Reuses the run's
OWN evaluator config so the boundary strip is wired exactly as in training.

  python -m egomimic.eval.core.boundary_viz --ckpt <ckpt> \
     --config-path <run>/.hydra/config.yaml --n-episodes 2 --out-dir chunkviz
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
import torchvision.io as tvio
from omegaconf import OmegaConf
from hydra.utils import instantiate

from egomimic.eval.core.ckpt_loading import _MockTrainer, load_algo_from_ckpt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--config-path", required=True)
    p.add_argument("--n-episodes", type=int, default=2)
    p.add_argument("--out-dir", default="chunkviz")
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

    evaluator = instantiate(full_cfg.evaluator)
    cands = list(getattr(evaluator, "evals", [evaluator]))
    video_eval = next((e for e in cands if type(e).__name__ == "EvalVideoList"), cands[0])
    for e in [video_eval] + list(getattr(video_eval, "evals", [])):
        e.trainer = _MockTrainer(str(out), device); e.model = algo

    metrics, images = video_eval.compute_metrics_and_viz(batch)
    for emb_id, ims in images.items():
        if getattr(ims, "size", 0) == 0:
            print(f"emb{emb_id}: no frames"); continue
        f = out / f"chunkviz_emb{emb_id}.mp4"
        tvio.write_video(str(f), torch.from_numpy(np.ascontiguousarray(ims)),
                         fps=30, video_codec="h264")
        print(f"wrote {f}  shape={ims.shape}")
    print("DONE")


if __name__ == "__main__":
    main()
