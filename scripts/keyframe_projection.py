"""
Offline runner for KeyframeProjectionEval (the Eval class) on a checkpoint.

Loads the algo from a lightning ckpt, builds one val batch, runs the eval
class, and saves each episode's per-stage keyframe filmstrip as a PNG.

Run (GPU node):
  python scripts/keyframe_projection.py \
      --ckpt <run>/checkpoints/epoch_epoch=1499.ckpt \
      --config-path <run>/.hydra/config.yaml \
      --n-episodes 6 --out-dir keyframes_out/<tag>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from egomimic.eval.core.ckpt_loading import _MockTrainer, load_algo_from_ckpt
from egomimic.eval.probes.eval_keyframes import KeyframeProjectionEval


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--config-path", required=True)
    p.add_argument("--n-episodes", type=int, default=6)
    p.add_argument("--out-dir", default="keyframes_out")
    p.add_argument("--img-key", default="front_img_1")
    args = p.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    algo, _ = load_algo_from_ckpt(args.ckpt, args.config_path)
    algo.nets = algo.nets.to(device)
    algo.device = device
    algo.nets.eval()

    full_cfg = OmegaConf.load(args.config_path)
    dm = instantiate(full_cfg.data)
    dm.setup(stage="validate")
    first = next(iter(dm.val_dataloader()))
    batch = first[0] if isinstance(first, tuple) else first
    batch = algo.process_batch_for_training(batch)

    ev = KeyframeProjectionEval(max_videos=args.n_episodes, img_key=args.img_key)
    ev.trainer = _MockTrainer(str(out_dir), device)
    ev.model = algo
    metrics, _videos = ev.compute_metrics_and_viz(batch)

    from PIL import Image

    for emb_id, canvases in ev.last_canvases.items():
        for b, canvas in enumerate(canvases):
            f = out_dir / f"keyframes_emb{emb_id}_ep{b}.png"
            Image.fromarray(canvas).save(f)
            print(f"[keyframes] wrote {f}")
    for k, v in metrics.items():
        print(f"  {k}: {float(v):.4f}")


if __name__ == "__main__":
    main()
