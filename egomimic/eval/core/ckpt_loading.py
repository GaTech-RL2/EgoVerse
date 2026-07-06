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

from egomimic.eval.core.eval_sim import HPTSimEval, PackedSimEval


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
    parser.add_argument(
        "--max-coverage",
        action="store_true",
        help="report PEAK IoU over the rollout (did it ever align?) "
        "instead of the default final-step IoU",
    )
    parser.add_argument("--out-dir", default="sim_smoke_out")
    parser.add_argument(
        "--per-episode-videos",
        action="store_true",
        help="write one mp4 per episode, named ep{i}_cov{c}.mp4",
    )
    parser.add_argument(
        "--embodiment-name",
        default="pushshapes_sim",
        help="Which embodiment to roll out (e.g. pushshapes_sim_small_circle for the small run).",
    )
    parser.add_argument("--pusher", default="circle", help="env pusher_shape")
    parser.add_argument(
        "--obstacle-level",
        type=int,
        default=0,
        help="env obstacle_level (0-29 with the ported 30-level obstacles.py)",
    )
    parser.add_argument(
        "--coverage-threshold",
        type=float,
        default=0.7,
        help="episode early-stop + success cutoff; 0.95 for true peak + SR@0.95",
    )
    parser.add_argument(
        "--full-horizon",
        action="store_true",
        help="run the full max_steps; ignore the env success-termination "
        "(0.95) so every episode yields the uncapped true peak and "
        "uniform-length rollouts",
    )
    parser.add_argument(
        "--obs-stride",
        type=int,
        default=None,
        help="Override the model's obs_stride (the ARCH re-plan cadence) for "
        "this eval: the policy re-observes every obs_stride frames. 1 = dense "
        "(open-loop-1, matches per-frame training); = chunk_len = sparse "
        "(open-loop-C). Default: keep the model's own obs_stride.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--init-mode",
        choices=["replay", "seeds"],
        default="replay",
        help="Episode init source. 'replay' (default) replays frame-0 states "
        "of the run's OWN val episodes — fine within a run, but init pools "
        "differ across runs trained on different data. 'seeds' resets the env "
        "from a deterministic seed list (base + episode index) — identical "
        "inits across runs; use for any cross-run comparison.",
    )
    parser.add_argument(
        "--init-seed-base",
        type=int,
        default=1000,
        help="With --init-mode seeds: episode i resets with seed base+i.",
    )
    parser.add_argument(
        "--rng-pairing",
        action="store_true",
        help="Reseed the sampler RNG per episode (inside fork_rng) so GMM "
        "sampling noise is paired across runs for the same episode index.",
    )
    parser.add_argument(
        "--only-emb",
        type=int,
        default=None,
        help="Evaluate only this embodiment id from the batch (e.g. 17). "
        "The evaluator builds ONE env per invocation, so multi-embodiment "
        "batches need one invocation per embodiment with the matching --pusher.",
    )
    parser.add_argument(
        "--eval-class",
        choices=["packed", "hpt"],
        default="packed",
        help="Which sim evaluator matches the algo's batch layout: 'packed' "
        "(PackedSimEval — cu_seqlens episode batches; H-Net/WindowedBC) or "
        "'hpt' (HPTSimEval — per-frame row batches).",
    )
    parser.add_argument(
        "--replan-every",
        type=int,
        default=None,
        help="Re-plan after this many actions of each predicted chunk "
        "(receding horizon). Default: consume the full chunk open-loop. "
        "1 = re-plan every env step.",
    )
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
    # Re-plan cadence is the model's obs_stride (arch property). Optionally
    # override it for this eval to probe a different cadence on the same ckpt.
    if args.obs_stride is not None:
        algo.obs_stride = int(args.obs_stride)
        print(
            f"[sim] obs_stride override = {algo.obs_stride} "
            f"(re-observe every {algo.obs_stride} frame(s))"
        )
    if args.replan_every is not None:
        algo.replan_every = int(args.replan_every)
        print(f"[sim] replan_every override = {algo.replan_every}")
    import torch as _torch

    _torch.manual_seed(int(args.seed))

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
    if args.only_emb is not None:
        if args.only_emb not in batch:
            raise SystemExit(
                f"--only-emb {args.only_emb} not in batch embodiments "
                f"{list(batch.keys())}"
            )
        batch = {args.only_emb: batch[args.only_emb]}
        print(f"[batch] filtered to emb={args.only_emb}")
    for emb_id, _b in batch.items():
        if "cu_seqlens" in _b:
            cu = _b["cu_seqlens"]
            print(f"[batch] emb={emb_id} cu_seqlens={cu.tolist()}  B={len(cu)-1}")

    # Build the sim evaluator and wire trainer/model stubs.
    init_seeds = (
        [args.init_seed_base + i for i in range(args.n_episodes)]
        if args.init_mode == "seeds"
        else None
    )
    if init_seeds is not None:
        print(f"[sim] init_mode=seeds  seeds={init_seeds[0]}..{init_seeds[-1]}")
    eval_cls = HPTSimEval if args.eval_class == "hpt" else PackedSimEval
    sim_eval = eval_cls(
        env_kwargs={
            "object_shape": "T",
            "pusher_shape": args.pusher,
            "obstacle_level": args.obstacle_level,
        },
        embodiment_name=args.embodiment_name,
        init_mode=args.init_mode,
        init_seeds=init_seeds,
        max_steps=args.max_steps,
        report_max_coverage=args.max_coverage,
        coverage_threshold=args.coverage_threshold,
        limit_val_batches=args.n_episodes,
        rng_pairing=args.rng_pairing,
        run_full_horizon=args.full_horizon,
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

    print(
        "[rollout] starting sim eval (fp32 — inference_step now matches model dtype) ..."
    )
    # NO autocast: the model is fp32 and inference_step now allocates the AR
    # state in the model's dtype (fp32), so the whole rollout is pure fp32 —
    # consistent with training, the teacher-forced overlay, and txar's sim. (The
    # previous bf16-autocast wrap was a band-aid for the old bf16-state hardcode,
    # which was the bug that uniquely degraded the H-Net closed-loop coverage.)
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

    if args.per_episode_videos:
        print("\n=== SAVING PER-EPISODE VIDEOS ===")
        pe_frames = getattr(sim_eval, "_last_per_ep_frames", {})
        pe_cov = getattr(sim_eval, "_last_per_ep_coverages", {})
        for emb_id, ep_list in pe_frames.items():
            covs = pe_cov.get(emb_id, [])
            for i, ims in enumerate(ep_list):
                if ims.size == 0:
                    continue
                c = covs[i] if i < len(covs) else 0.0
                fp = out_dir / f"ep{i:02d}_emb{emb_id}_cov{c:.3f}.mp4"
                tvio.write_video(
                    str(fp), torch.from_numpy(ims), fps=30, video_codec="h264"
                )
                print(f"  wrote {fp}  cov={c:.4f}  shape={ims.shape}")

    print("\n=== SMOKE PASSED ===")


if __name__ == "__main__":
    main()
