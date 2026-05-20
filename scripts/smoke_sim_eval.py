"""
Smoke test for the H-Net closed-loop sim evaluator.

Loads the existing baseline checkpoint and runs a few sim rollouts against
the PushShapes env, then prints per-episode coverage and saves a short
video. Useful to debug ``eval_hnet_sim.HNetSimEval`` without running a
full Lightning validation.

Run on an A40 node:
    python scripts/smoke_sim_eval.py \
        --ckpt logs/hnet_pushshapes/full_run_150ep_2026-05-15_03-18-17/csv_logs/lightning_logs/version_0/checkpoints/epoch=149-step=1200.ckpt \
        --n-episodes 2 --max-steps 200
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torchvision.io as tvio
from hydra.utils import instantiate
from omegaconf import OmegaConf

from egomimic.eval.eval_sim import SimRolloutEval


def load_algo_from_ckpt(ckpt_path: str, config_path: str | None = None):
    """Reconstruct the algo + load its weights from the lightning ckpt."""
    print(f"[load] ckpt: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # Lightning saves under "state_dict"; param keys are "nets.policy.<name>".
    print(f"[load] keys: {list(ckpt.keys())[:5]} ...")

    hparams = ckpt.get("hyper_parameters") or ckpt.get("hparams") or {}
    if "config_tree" not in hparams:
        if config_path is None:
            raise SystemExit(
                "Checkpoint has no config_tree in hyper_parameters; pass --config-path "
                "to point at the hydra .hydra/config.yaml that built it."
            )
        cfg = OmegaConf.load(config_path)
    else:
        cfg = OmegaConf.create(hparams["config_tree"])

    # Build the algo via the same Hydra path the trainer uses.
    from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset

    norm_state = hparams.get("norm_stats_state")
    if norm_state is None:
        raise SystemExit("hyper_parameters has no norm_stats_state")
    norm_stats = MultiDataset.from_state(norm_state)
    algo = instantiate(cfg.model.robomimic_model, norm_stats=norm_stats)

    # Strip "nets." prefix or "model.nets." prefix when loading.
    state_dict = ckpt["state_dict"]
    new_sd = {}
    for k, v in state_dict.items():
        for prefix in ("nets.", "model.nets."):
            if k.startswith(prefix):
                new_sd[k[len(prefix) :]] = v
                break
        else:
            # keep as-is
            new_sd[k] = v
    missing, unexpected = algo.nets.load_state_dict(new_sd, strict=False)
    if missing:
        print(f"[load] missing keys ({len(missing)}): {missing[:5]}")
    if unexpected:
        print(f"[load] unexpected keys ({len(unexpected)}): {unexpected[:5]}")
    return algo, cfg


class _MockTrainer:
    """Stub for the trainer attributes that EvalVideo / HNetSimEval read."""

    def __init__(self, output_dir: str, device: torch.device):
        self.current_epoch = 0
        self.is_global_zero = True
        self.lightning_module = type("M", (), {"device": device})()
        # ``Eval.root_dir()`` uses os.getcwd() or trainer.default_root_dir.
        self.default_root_dir = output_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument(
        "--config-path",
        default=None,
        help="Optional path to .hydra/config.yaml if ckpt has no config_tree.",
    )
    parser.add_argument("--n-episodes", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--out-dir", default="sim_smoke_out")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.config_path is None:
        # If ckpt lives in a hydra run dir, infer the .hydra path. The ckpt
        # path is run_dir/csv_logs/lightning_logs/version_0/checkpoints/*.ckpt
        # → parents[4] is run_dir.
        guessed = Path(args.ckpt).parents[4] / ".hydra" / "config.yaml"
        if guessed.exists():
            args.config_path = str(guessed)
            print(f"[config] using {args.config_path}")

    algo, _ = load_algo_from_ckpt(args.ckpt, args.config_path)
    # HNet algo doesn't inherit nn.Module — move the inner ModuleDict.
    algo.nets = algo.nets.to(device)
    algo.device = device
    algo.nets.eval()

    # Build the dataset from the full hydra .hydra/config.yaml (the ckpt's
    # config_tree only contains the model subtree — no data/evaluator).
    if args.config_path is None:
        raise SystemExit(
            "Pass --config-path to point at the original run's .hydra/config.yaml "
            "for the data config."
        )
    full_cfg = OmegaConf.load(args.config_path)
    data_cfg = full_cfg.data
    print(f"[data] target: {data_cfg._target_}")
    dm = instantiate(data_cfg)
    dm.setup(stage="validate")
    val_loader = dm.val_dataloader()
    # MultiDataModuleWrapper returns a CombinedLoader that yields
    # ``(batch_dict, batch_idx, dataloader_idx)`` tuples.
    first = next(iter(val_loader))
    if isinstance(first, tuple) and len(first) == 3:
        batch = first[0]
    elif isinstance(first, dict):
        batch = first
    else:
        raise SystemExit(f"unexpected batch type: {type(first)}")
    batch = algo.process_batch_for_training(batch)
    print(f"[batch] embodiments: {list(batch.keys())}")
    for emb_id, _b in batch.items():
        if "cu_seqlens" in _b:
            cu = _b["cu_seqlens"]
            print(f"[batch] emb={emb_id} cu_seqlens={cu.tolist()}  B={len(cu)-1}")

    # Build the sim evaluator and wire trainer/model stubs.
    sim_eval = SimRolloutEval(
        env_kwargs={"object_shape": "T", "pusher_shape": "circle"},
        embodiment_name="pushshapes_sim",
        init_mode="replay",
        max_steps=args.max_steps,
        coverage_threshold=0.7,
        limit_val_batches=1,
    )
    sim_eval.trainer = _MockTrainer(args.out_dir, device)
    sim_eval.model = algo

    # Cap how many episodes per batch we roll out (smoke).
    # We do this by slicing the cu_seqlens to keep only n_episodes.
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
                if not torch.is_tensor(v):
                    continue
                if v.dim() >= 1 and v.shape[0] == int(cu[-1].item()):
                    _b[k] = v[:new_end].contiguous()
            print(f"[batch] trimmed to {new_B} episodes  T_total={new_end}")

    print("[rollout] starting sim eval ...")
    metrics, images_dict = sim_eval.compute_metrics_and_viz(batch)

    print("\n=== METRICS ===")
    for k, v in metrics.items():
        print(f"  {k}: {float(v):.4f}")

    print("\n=== SAVING VIDEOS ===")
    for emb_id, ims in images_dict.items():
        if ims.size == 0:
            print(f"  emb={emb_id}: no frames")
            continue
        path = out_dir / f"sim_smoke_emb{emb_id}.mp4"
        # ims is (N, H, W, 3) uint8 — tvio expects (T, H, W, C).
        tvio.write_video(str(path), torch.from_numpy(ims), fps=30, video_codec="h264")
        print(f"  wrote {path}  shape={ims.shape}")

    print("\n=== SMOKE PASSED ===")


if __name__ == "__main__":
    main()
