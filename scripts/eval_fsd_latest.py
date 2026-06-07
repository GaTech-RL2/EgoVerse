"""Reload latest DFoT ckpt, flip inference_mode -> 'chunk' (FSD), run sim eval."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torchvision.io as tvio
from hydra.utils import instantiate
from omegaconf import OmegaConf

# Reuse the helper in smoke_sim_eval (same dir).
sys.path.insert(0, str(Path(__file__).parent))
from egomimic.eval.core.ckpt_loading import _MockTrainer, load_algo_from_ckpt

from egomimic.eval.eval_sim import PackedSimEval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config-path", default=None)
    ap.add_argument("--n-episodes", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--out-dir", default="fsd_eval_out")
    ap.add_argument("--inference-mode", default="chunk")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.config_path is None:
        guessed = Path(args.ckpt).parents[1] / ".hydra" / "config.yaml"
        if not guessed.exists():
            guessed = Path(args.ckpt).parents[4] / ".hydra" / "config.yaml"
        if guessed.exists():
            args.config_path = str(guessed)
            print(f"[config] {args.config_path}")

    algo, _ = load_algo_from_ckpt(args.ckpt, args.config_path)
    algo.nets = algo.nets.to(device)
    algo.device = device
    algo.nets.eval()

    # FLIP TO FSD.
    algo.inference_mode = args.inference_mode
    print(f"[algo] inference_mode -> {algo.inference_mode}")
    print(f"[algo] action_horizon = {algo.action_horizon}")

    full_cfg = OmegaConf.load(args.config_path)
    dm = instantiate(full_cfg.data)
    dm.setup(stage="validate")
    val_loader = dm.val_dataloader()
    first = next(iter(val_loader))
    batch = first[0] if isinstance(first, tuple) else first
    batch = algo.process_batch_for_training(batch)

    # Trim to n_episodes.
    for emb_id, _b in batch.items():
        if "cu_seqlens" not in _b:
            continue
        cu = _b["cu_seqlens"]
        B = len(cu) - 1
        if B > args.n_episodes:
            new_B = args.n_episodes
            new_end = int(cu[new_B].item())
            _b["cu_seqlens"] = cu[: new_B + 1].contiguous()
            _b["seq_lens"] = _b["seq_lens"][:new_B].contiguous()
            for k, v in list(_b.items()):
                if (
                    torch.is_tensor(v)
                    and v.dim() >= 1
                    and v.shape[0] == int(cu[-1].item())
                ):
                    _b[k] = v[:new_end].contiguous()

    sim_eval = PackedSimEval(
        env_kwargs={
            "object_shape": "T",
            "pusher_shape": "circle",
            "obstacle_level": 0,
            "image_size": 96,
        },
        embodiment_name="pushshapes_sim",
        init_mode="replay",
        init_seeds=[0, 1, 2, 3][: args.n_episodes],
        max_steps=args.max_steps,
        coverage_threshold=0.7,
        limit_val_batches=1,
        max_videos=args.n_episodes,
    )
    sim_eval.trainer = _MockTrainer(args.out_dir, device)
    sim_eval.model = algo

    print("[rollout] starting FSD sim eval ...")
    metrics, images_dict = sim_eval.compute_metrics_and_viz(batch)

    print("\n=== METRICS ===")
    for k, v in metrics.items():
        print(f"  {k}: {float(v):.4f}")

    for emb_id, ims in images_dict.items():
        if ims.size == 0:
            continue
        path = out_dir / f"fsd_emb{emb_id}.mp4"
        tvio.write_video(str(path), torch.from_numpy(ims), fps=30, video_codec="h264")
        print(f"[video] {path}  shape={ims.shape}")


if __name__ == "__main__":
    main()
