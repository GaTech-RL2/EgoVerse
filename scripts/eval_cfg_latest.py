"""DFoT eval with cfg + AR-staircase + per-tick-step-size knobs.
Supports teacher-forced val (DFoTValEval) and closed-loop sim (PackedSimEval)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torchvision.io as tvio
from hydra.utils import instantiate
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent))
from smoke_sim_eval import _MockTrainer, load_algo_from_ckpt  # type: ignore

from egomimic.eval.eval_dfot_val import DFoTValEval
from egomimic.eval.eval_sim import PackedSimEval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config-path", default=None)
    ap.add_argument("--n-episodes", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--out-dir", default="cfg_eval_out")
    ap.add_argument("--inference-mode", default="ar")
    ap.add_argument("--cfg-scale", type=float, required=True)
    ap.add_argument(
        "--ar-chunk-size",
        type=int,
        default=1,
        help="DFoTValEval offline staircase chunk_size (NOT the closed-loop knob).",
    )
    ap.add_argument(
        "--ar-step-size",
        type=int,
        default=1,
        help="DFoTValEval offline staircase step_size (NOT the closed-loop knob).",
    )
    ap.add_argument(
        "--ar-inference-chunk-size",
        type=int,
        default=None,
        help="Closed-loop algo.ar_inference_chunk_size override.",
    )
    ap.add_argument(
        "--ar-inference-step-size",
        type=int,
        default=1,
        help="Closed-loop algo.ar_inference_step_size: sub-steps per env tick.",
    )
    ap.add_argument("--skip-sim", action="store_true")
    ap.add_argument("--skip-val", action="store_true")
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

    algo.inference_mode = args.inference_mode
    algo.cfg_scale = args.cfg_scale
    algo.ar_inference_step_size = int(args.ar_inference_step_size)
    if args.ar_inference_chunk_size is not None:
        algo.ar_inference_chunk_size = int(args.ar_inference_chunk_size)
    print(
        f"[algo] inference_mode={algo.inference_mode}  cfg_scale={algo.cfg_scale}  "
        f"action_horizon={algo.action_horizon}  ar_chunk={algo.ar_inference_chunk_size}  "
        f"ar_step={algo.ar_inference_step_size}"
    )

    full_cfg = OmegaConf.load(args.config_path)
    dm = instantiate(full_cfg.data)
    dm.setup(stage="validate")
    val_loader = dm.val_dataloader()
    first = next(iter(val_loader))
    batch = first[0] if isinstance(first, tuple) else first
    batch = algo.process_batch_for_training(batch)

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

    if not args.skip_val:
        val_eval = DFoTValEval(
            do_full_chunk=True,
            do_ar=True,
            n_chunk_steps=50,
            ar_chunk_size=args.ar_chunk_size,
            ar_step_size=args.ar_step_size,
            cfg_scale=args.cfg_scale,
            world_size=512.0,
            limit_val_batches=1,
            max_videos=args.n_episodes,
            embodiment_name="pushshapes_sim",
        )
        val_eval.trainer = _MockTrainer(args.out_dir, device)
        val_eval.model = algo
        print("[val] teacher-forced eval ...")
        val_metrics, val_imgs = val_eval.compute_metrics_and_viz(batch)
        print("\n=== VAL METRICS ===")
        for k, v in val_metrics.items():
            print(f"  {k}: {float(v):.4f}")
        for emb_id, ims in val_imgs.items():
            if ims.size == 0:
                continue
            p = out_dir / f"val_cfg{args.cfg_scale}_emb{emb_id}.mp4"
            tvio.write_video(str(p), torch.from_numpy(ims), fps=30, video_codec="h264")
            print(f"[video] {p}  shape={ims.shape}")
    else:
        print("[val] skipped (--skip-val)")

    if args.skip_sim:
        print("[sim] skipped (--skip-sim)")
        return

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
    print("[sim] closed-loop eval ...")
    sim_metrics, sim_imgs = sim_eval.compute_metrics_and_viz(batch)
    print("\n=== SIM METRICS ===")
    for k, v in sim_metrics.items():
        print(f"  {k}: {float(v):.4f}")
    for emb_id, ims in sim_imgs.items():
        if ims.size == 0:
            continue
        p = out_dir / f"sim_cfg{args.cfg_scale}_emb{emb_id}.mp4"
        tvio.write_video(str(p), torch.from_numpy(ims), fps=30, video_codec="h264")
        print(f"[video] {p}  shape={ims.shape}")


if __name__ == "__main__":
    main()
