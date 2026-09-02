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
import json
from pathlib import Path

import torch
import torchvision.io as tvio
from hydra.utils import instantiate
from omegaconf import OmegaConf

from egomimic.eval.core.eval_sim import HPTSimEval, PackedSimEval


def _parse_init_seeds(value: str | None) -> list[int] | None:
    """Parse the canonical launcher's explicit comma-separated seed list."""
    if value is None:
        return None
    pieces = [piece.strip() for piece in value.split(",") if piece.strip()]
    if not pieces:
        raise ValueError("--init-seeds must contain at least one integer")
    try:
        seeds = [int(piece) for piece in pieces]
    except ValueError as exc:
        raise ValueError(f"invalid --init-seeds value {value!r}") from exc
    if len(set(seeds)) != len(seeds):
        raise ValueError("--init-seeds must not contain duplicates")
    return seeds


def _assert_completed_seed_rollouts(
    sim_eval, embodiment_ids, expected: int
) -> dict[int, int]:
    """Fail if seed-mode evaluation did not consume every requested seed."""
    coverages = getattr(sim_eval, "_last_per_ep_coverages", {})
    counts = {int(emb_id): len(coverages.get(emb_id, [])) for emb_id in embodiment_ids}
    mismatches = [
        f"emb{emb_id} completed={completed} expected={expected}"
        for emb_id, completed in counts.items()
        if completed != expected
    ]
    if mismatches:
        raise RuntimeError("seed rollout count mismatch: " + "; ".join(mismatches))
    return counts


def _checkpoint_model_config(ckpt: dict, config_path: str | None):
    """Return the checkpoint-pinned model config and verify the resolved config."""
    hparams = ckpt.get("hyper_parameters") or ckpt.get("hparams") or {}
    embedded_tree = hparams.get("config_tree")
    if embedded_tree is None:
        raise RuntimeError("checkpoint has no embedded hyper_parameters.config_tree")
    embedded_cfg = OmegaConf.create(embedded_tree)
    if OmegaConf.select(embedded_cfg, "model.robomimic_model") is None:
        raise RuntimeError("checkpoint config_tree has no model.robomimic_model")

    if config_path is not None:
        full_cfg = OmegaConf.load(config_path)
        embedded_model = OmegaConf.to_container(
            embedded_cfg.model.robomimic_model, resolve=True
        )
        resolved_model = OmegaConf.to_container(
            full_cfg.model.robomimic_model, resolve=True
        )
        if embedded_model != resolved_model:
            raise RuntimeError(
                "resolved config model.robomimic_model differs from the "
                "checkpoint-embedded model config"
            )
    return embedded_cfg, hparams


def _strict_state_dict(ckpt: dict, use_ema: bool) -> dict:
    """Extract the exact Algo ModuleDict state, including registered EMA trees."""
    if "state_dict" not in ckpt:
        raise RuntimeError("checkpoint has no state_dict")
    state_dict = ckpt["state_dict"]
    if not isinstance(state_dict, dict):
        raise RuntimeError("checkpoint state_dict is not a mapping")

    online = {}
    registered_ema = {}
    wrapper_buffers = {}
    rejected = []
    allowed_wrapper_buffers = {"ema_optimization_step", "ema_decay"}
    for key, value in state_dict.items():
        if key.startswith("ema_nets."):
            stripped_key = key[len("ema_nets.") :]
            if stripped_key in registered_ema:
                raise RuntimeError(f"duplicate registered EMA state key {stripped_key!r}")
            registered_ema[stripped_key] = value
            continue
        if key in allowed_wrapper_buffers:
            wrapper_buffers[key] = value
            continue
        for prefix in ("nets.", "model.nets."):
            if key.startswith(prefix):
                stripped_key = key[len(prefix) :]
                if stripped_key in online:
                    raise RuntimeError(f"duplicate stripped state key {stripped_key!r}")
                online[stripped_key] = value
                break
        else:
            rejected.append(key)
    if rejected:
        raise RuntimeError(
            f"checkpoint contains {len(rejected)} non-model state keys: {rejected[:5]}"
        )
    if not online:
        raise RuntimeError("checkpoint state_dict contains no nets.* model tensors")

    legacy_ema = ckpt.get("ema_state_dict")
    if registered_ema:
        if legacy_ema:
            raise RuntimeError(
                "checkpoint contains both registered ema_nets.* and ema_state_dict"
            )
        missing_buffers = allowed_wrapper_buffers - set(wrapper_buffers)
        if missing_buffers:
            raise RuntimeError(
                "registered EMA checkpoint is missing wrapper buffers: "
                f"{sorted(missing_buffers)}"
            )
        if set(registered_ema) != set(online):
            missing = sorted(set(online) - set(registered_ema))[:5]
            extra = sorted(set(registered_ema) - set(online))[:5]
            raise RuntimeError(
                "registered EMA tensor keys do not exactly match online keys: "
                f"missing={missing} extra={extra}"
            )
    elif wrapper_buffers:
        raise RuntimeError(
            "EMA wrapper buffers are present without a registered ema_nets.* tree"
        )

    normalized_legacy_ema = None
    if legacy_ema:
        if not isinstance(legacy_ema, dict):
            raise RuntimeError("ema_state_dict is not a mapping")
        normalized_legacy_ema = {}
        for key, value in legacy_ema.items():
            stripped_key = key
            for prefix in ("nets.", "model.nets.", "ema_nets."):
                if key.startswith(prefix):
                    stripped_key = key[len(prefix) :]
                    break
            if stripped_key in normalized_legacy_ema:
                raise RuntimeError(
                    f"duplicate legacy EMA state key {stripped_key!r}"
                )
            normalized_legacy_ema[stripped_key] = value
        if set(normalized_legacy_ema) != set(online):
            missing = sorted(set(online) - set(normalized_legacy_ema))[:5]
            extra = sorted(set(normalized_legacy_ema) - set(online))[:5]
            raise RuntimeError(
                "legacy EMA tensor keys do not exactly match online keys: "
                f"missing={missing} extra={extra}"
            )

    if use_ema:
        selected = registered_ema or normalized_legacy_ema
        if not selected:
            raise RuntimeError("requested EMA weights but checkpoint has no EMA tree")
        print(f"[load] using EMA weights ({len(selected)} tensors)")
        return selected
    print(f"[load] using online weights ({len(online)} tensors)")
    return online


def load_algo_from_ckpt(
    ckpt_path: str, config_path: str | None = None, use_ema: bool = False
):
    """Reconstruct the algo and load every checkpoint tensor strictly."""
    print(f"[load] ckpt: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # Lightning saves under "state_dict"; param keys are "nets.policy.<name>".
    print(f"[load] keys: {list(ckpt.keys())[:5]} ...")

    cfg, hparams = _checkpoint_model_config(ckpt, config_path)

    # Build the algo via the same Hydra path the trainer uses.
    from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset

    norm_state = hparams.get("norm_stats_state")
    if norm_state is None:
        raise RuntimeError("hyper_parameters has no norm_stats_state")
    norm_stats = MultiDataset.from_state(norm_state)
    algo = instantiate(cfg.model.robomimic_model, norm_stats=norm_stats)

    new_sd = _strict_state_dict(ckpt, use_ema=use_ema)
    algo.nets.load_state_dict(new_sd, strict=True)
    print(f"[load] strict state load passed ({len(new_sd)} tensors)")
    return algo, cfg


def strict_no_rollout_preflight(
    ckpt_path: str,
    config_path: str,
    selected_embodiment_name: str,
    selected_embodiment_id: int,
    expected_native_action_dim: int,
    use_ema: bool = False,
    action_chunk_start_index: int = 0,
    replan_every: int | None = None,
) -> dict:
    """Strictly validate checkpoint/config/model binding without an env or inference."""
    from egomimic.rldb.embodiment.embodiment import get_embodiment_id
    from egomimic.pipeline.algo import _resolve_action_chunk_execution_slice

    algo, _ = load_algo_from_ckpt(ckpt_path, config_path, use_ema=use_ema)
    emb_name = str(selected_embodiment_name)
    emb_id = int(selected_embodiment_id)
    native_dim = int(expected_native_action_dim)
    if native_dim <= 0:
        raise RuntimeError(f"expected native action dim must be positive, got {native_dim}")
    resolved_id = int(get_embodiment_id(emb_name))
    if resolved_id != emb_id:
        raise RuntimeError(
            f"embodiment name/id mismatch: {emb_name!r} maps to {resolved_id}, "
            f"expected {emb_id}"
        )
    if algo.domain_by_id.get(emb_id) != emb_name:
        raise RuntimeError(
            "selected embodiment is absent from loaded model domains: "
            f"selected={emb_id}:{emb_name} loaded={algo.domain_by_id}"
        )

    adapter = algo.rollout_adapter_for(emb_id)
    if adapter is None or not callable(getattr(adapter, "decode", None)):
        raise RuntimeError(f"selected embodiment {emb_name!r} has no rollout adapter")
    if not bool(getattr(adapter, "preserves_decoded_timing", False)):
        raise RuntimeError("rollout adapter does not declare preserves_decoded_timing=True")

    action_dims = []
    for stage in algo.policy.stages:
        dims = getattr(stage, "action_dims", None)
        if dims is not None and emb_name in dims:
            action_dims.append(int(dims[emb_name]))
    if len(set(action_dims)) != 1:
        raise RuntimeError(
            f"could not resolve one model token width for {emb_name!r}: {action_dims}"
        )
    token_dim = action_dims[0]
    token_horizon = int(algo.action_horizon)
    probe = torch.zeros((1, token_horizon, token_dim), dtype=torch.float32)
    decoded = adapter.decode(probe, context={})
    if not torch.is_tensor(decoded):
        decoded = torch.as_tensor(decoded)
    if decoded.ndim != 3 or decoded.shape[0] != 1:
        raise RuntimeError(
            f"adapter probe must decode to (1,H,D), got {tuple(decoded.shape)}"
        )
    if int(decoded.shape[-1]) != native_dim:
        raise RuntimeError(
            f"adapter native action width {decoded.shape[-1]} != expected {native_dim}"
        )
    decoded_horizon = int(decoded.shape[1])
    if decoded_horizon <= 0:
        raise RuntimeError("adapter decoded an empty action horizon")
    execution_start, execution_stop = _resolve_action_chunk_execution_slice(
        decoded_horizon=decoded_horizon,
        start_index=action_chunk_start_index,
        replan_every=replan_every,
    )

    report = {
        "status": "audited_strict_no_rollout_ok",
        "embodiment_id": emb_id,
        "embodiment_name": emb_name,
        "model_token_horizon": token_horizon,
        "model_token_dim": token_dim,
        "decoded_action_horizon": decoded_horizon,
        "action_chunk_start_index": execution_start,
        "execution_horizon": execution_stop - execution_start,
        "execution_stop_index_exclusive": execution_stop,
        "native_action_dim": native_dim,
        "state_tensor_count": len(algo.nets.state_dict()),
        "weights": "ema" if use_ema else "raw",
    }
    print("[strict-preflight] " + json.dumps(report, sort_keys=True))
    return report


def _override_sampler_inference_steps(
    algo,
    inference_steps: int,
    embodiment_name: str | None = None,
) -> tuple[str, int]:
    """Apply an evaluation-only override after the exact state loads."""
    requested = int(inference_steps)
    if requested <= 0:
        raise ValueError("sampler inference steps must be positive")
    candidates = []
    seen_ids = set()

    def add_candidate(candidate) -> None:
        if not hasattr(candidate, "num_inference_steps"):
            return
        candidate_id = id(candidate)
        if candidate_id not in seen_ids:
            seen_ids.add(candidate_id)
            candidates.append(candidate)

    for stage in algo.policy.stages:
        add_candidate(stage)
        policies = getattr(stage, "policies", None)
        if policies is None:
            continue
        if embodiment_name is not None and embodiment_name in policies:
            add_candidate(policies[embodiment_name])
        elif embodiment_name is None and len(policies) == 1:
            add_candidate(next(iter(policies.values())))

    if len(candidates) != 1:
        names = [type(stage).__name__ for stage in candidates]
        raise RuntimeError(
            "sampler inference-step override requires exactly one compatible "
            f"stage, found {names}"
        )
    stage = candidates[0]
    original = int(stage.num_inference_steps)
    stage.num_inference_steps = requested
    return type(stage).__name__, original


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
        "--use-ema",
        action="store_true",
        help="load ema_state_dict weights (EMACallback runs)",
    )
    parser.add_argument(
        "--config-path",
        default=None,
        help="Optional path to .hydra/config.yaml if ckpt has no config_tree.",
    )
    parser.add_argument("--n-episodes", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument(
        "--rollout-timeout",
        type=int,
        default=120,
        help="Per-episode watchdog in seconds; 0 disables it.",
    )
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
        "--init-seeds",
        default=None,
        help="Explicit comma-separated reset seeds. Requires --init-mode seeds "
        "and must contain exactly --n-episodes unique values.",
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
    parser.add_argument(
        "--action-chunk-start-index",
        type=int,
        default=0,
        help="Zero-based index of the first decoded native action to execute. "
        "Default: 0. This is an explicit evaluation-time temporal-contract "
        "setting and is never inferred from the model family.",
    )
    parser.add_argument(
        "--sampler-inference-steps",
        type=int,
        default=None,
        help="Evaluation-only override for a single compatible sampler stage. "
        "The checkpoint/config remain strictly pinned.",
    )
    args = parser.parse_args()

    try:
        explicit_init_seeds = _parse_init_seeds(args.init_seeds)
    except ValueError as exc:
        parser.error(str(exc))
    if explicit_init_seeds is not None:
        if args.init_mode != "seeds":
            parser.error("--init-seeds requires --init-mode seeds")
        if len(explicit_init_seeds) != args.n_episodes:
            parser.error(
                "--init-seeds count must equal --n-episodes "
                f"({len(explicit_init_seeds)} != {args.n_episodes})"
            )
    if args.rollout_timeout < 0:
        parser.error("--rollout-timeout must be >= 0")
    if args.replan_every is not None and args.replan_every <= 0:
        parser.error("--replan-every must be positive")
    if args.action_chunk_start_index < 0:
        parser.error("--action-chunk-start-index must be >= 0")
    if args.sampler_inference_steps is not None and args.sampler_inference_steps <= 0:
        parser.error("--sampler-inference-steps must be positive")

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

    algo, _ = load_algo_from_ckpt(args.ckpt, args.config_path, use_ema=args.use_ema)
    if args.sampler_inference_steps is not None:
        stage_name, original_steps = _override_sampler_inference_steps(
            algo,
            args.sampler_inference_steps,
            embodiment_name=args.embodiment_name,
        )
        print(
            "[sim] sampler inference steps override: "
            f"stage={stage_name} checkpoint_default={original_steps} "
            f"effective={args.sampler_inference_steps}"
        )
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
    if not hasattr(algo, "action_chunk_start_index"):
        if args.action_chunk_start_index != 0:
            raise RuntimeError(
                f"{type(algo).__name__} does not support --action-chunk-start-index"
            )
    else:
        algo.action_chunk_start_index = int(args.action_chunk_start_index)
        print(f"[sim] action_chunk_start_index = {algo.action_chunk_start_index}")
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
            print(f"[batch] emb={emb_id} cu_seqlens={cu.tolist()}  B={len(cu) - 1}")

    # Build the sim evaluator and wire trainer/model stubs.
    init_seeds = explicit_init_seeds
    if init_seeds is None and args.init_mode == "seeds":
        init_seeds = [args.init_seed_base + i for i in range(args.n_episodes)]
    if init_seeds is not None:
        print(
            "[sim] init_mode=seeds  seeds=" + ",".join(str(seed) for seed in init_seeds)
        )
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
        rollout_timeout_s=args.rollout_timeout,
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

    if args.init_mode == "seeds":
        completed_counts = _assert_completed_seed_rollouts(
            sim_eval, batch.keys(), args.n_episodes
        )
        for emb_id, completed in completed_counts.items():
            print(
                f"[sim] emb{emb_id} rollout_count: "
                f"completed={completed} requested={args.n_episodes}"
            )

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
