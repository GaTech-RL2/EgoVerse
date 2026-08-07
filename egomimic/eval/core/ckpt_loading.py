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


def load_algo_from_ckpt(ckpt_path: str, config_path: str | None = None,
                        use_ema: bool = False):
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
        if config_path is not None:
            print("[config] WARNING: --config-path %s is IGNORED for model "
                  "construction -- the checkpoint embeds hyper_parameters"
                  "['config_tree'], which wins. Editing a frozen-config copy "
                  "does NOT change the model (this voided a center-crop A/B on "
                  "2026-07-30). Force model behaviour AFTER load instead."
                  % config_path)

    # Build the algo via the same Hydra path the trainer uses.
    from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset

    norm_state = hparams.get("norm_stats_state")
    if norm_state is None:
        raise SystemExit("hyper_parameters has no norm_stats_state")
    norm_stats = MultiDataset.from_state(norm_state)
    algo = instantiate(cfg.model.robomimic_model, norm_stats=norm_stats)

    # Strip "nets." prefix or "model.nets." prefix when loading.
    state_dict = ckpt["state_dict"]
    if use_ema:
        ema = ckpt.get("ema_state_dict")
        if ema is None:
            raise SystemExit("--use-ema: checkpoint has no ema_state_dict "
                             "(was the run trained with EMACallback?)")
        state_dict = dict(state_dict)
        state_dict.update(ema)  # float params/buffers -> EMA; int buffers stay live
        print(f"[load] using EMA weights ({len(ema)} tensors)")
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
    parser.add_argument("--use-ema", action="store_true",
                        help="load ema_state_dict weights (EMACallback runs)")
    parser.add_argument(
        "--config-path",
        default=None,
        help="Optional path to .hydra/config.yaml if ckpt has no config_tree.",
    )
    parser.add_argument("--n-episodes", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--max-coverage", action="store_true",
                        help="report PEAK IoU over the rollout (did it ever align?) "
                             "instead of the default final-step IoU")
    parser.add_argument("--out-dir", default="sim_smoke_out")
    parser.add_argument("--solid-pusher", action="store_true",
                        help="sim_v2 protocol: solid-pusher physics (matches v2 collection)")
    parser.add_argument("--init-seeds", default=None,
                        help="comma-separated explicit seed list (sim_v2 curated seeds); "
                             "overrides --init-seed-base/--n-episodes count")
    parser.add_argument("--save-rollout-zarr", default=None,
                        help="ADDITIVE: write each rollout as a dataset-format "
                             "zarr episode into this dir (for chunkviz on rollouts)")
    parser.add_argument("--per-episode-videos", action="store_true",
                        help="write one mp4 per episode, named ep{i}_cov{c}.mp4")
    parser.add_argument("--embodiment-name", default="pushshapes_sim",
                        help="Which embodiment to roll out (e.g. pushshapes_sim_small_circle for the small run).")
    parser.add_argument("--pusher", default="circle", help="env pusher_shape")
    parser.add_argument("--obstacle-level", type=int, default=0,
                        help="env obstacle_level (0-29 with the ported 30-level obstacles.py)")
    parser.add_argument(
        "--fixed-goal",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "THETA"),
        help="Pin the rollout goal pose while preserving the seeded object and pusher.",
    )
    parser.add_argument("--coverage-threshold", type=float, default=0.7,
                        help="episode early-stop + success cutoff; 0.95 for true peak + SR@0.95")
    parser.add_argument("--full-horizon", action="store_true",
                        help="run the full max_steps; ignore the env success-termination "
                             "(0.95) so every episode yields the uncapped true peak and "
                             "uniform-length rollouts")
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
    parser.add_argument(
        "--rollout-timeout", type=int, default=120,
        help="Per-rollout watchdog in seconds. A timeout is scored coverage=0.0, "
             "so raise it (or pass 0 to disable) whenever the GPU is contended "
             "or a comparison must be trustworthy.",
    )
    parser.add_argument(
        "--apex-attention", choices=["full", "self", "windowed"], default=None,
        help="Force the ApexLevel attention regime AFTER loading. Apex "
             "windowing exists only as a training-time callback, so a fresh "
             "eval load is always 'full' -- use this to roll out the "
             "windowed regime the mix arms were trained on.",
    )
    parser.add_argument(
        "--apex-window", type=int, default=None,
        help="Window size for --apex-attention windowed.",
    )
    parser.add_argument(
        "--crop-eval-mode", choices=["center", "random"], default=None,
        help="Force VisualCore.crop_eval_mode AFTER loading. A frozen-config "
             "copy does NOT work: the ckpt embeds config_tree, which wins over "
             "--config-path for model construction.",
    )
    args = parser.parse_args()
    if not getattr(args, "max_coverage", False):
        print("[metric] WARNING: --max-coverage NOT set -> coverage is "
              "FINAL-STEP IoU, not the standard PEAK IoU. Numbers will NOT be "
              "comparable to the ledger. (Mixing the two manufactured a fake "
              "-0.34 'center crop' effect on 2026-07-30.)")

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

    algo, _ = load_algo_from_ckpt(args.ckpt, args.config_path,
                                  use_ema=args.use_ema)
    # HNet algo doesn't inherit nn.Module — move the inner ModuleDict.
    algo.nets = algo.nets.to(device)
    if args.crop_eval_mode is not None:
        from egomimic.models.stems.visual_core import VisualCore
        _vcs = [m for m in algo.nets.modules() if isinstance(m, VisualCore)]
        if not _vcs:
            raise SystemExit("--crop-eval-mode: model has no VisualCore")
        for _v in _vcs:
            _v.crop_eval_mode = args.crop_eval_mode
        print("[crop] forced crop_eval_mode=%s on %d VisualCore(s) "
              "(crop_aug=%s %sx%s scope=%s)"
              % (args.crop_eval_mode, len(_vcs), _vcs[0].crop_aug,
                 _vcs[0].crop_height, _vcs[0].crop_width,
                 getattr(_vcs[0], "crop_scope", "?")))
    else:
        from egomimic.models.stems.visual_core import VisualCore as _VC
        _vcs = [m for m in algo.nets.modules() if isinstance(m, _VC)]
        if _vcs:
            print("[crop] crop_eval_mode=%s (from ckpt config_tree; NOT settable "
                  "via --config-path)" % _vcs[0].crop_eval_mode)
    if args.apex_attention is not None:
        from egomimic.pipeline.stages_hnet import ApexLevel
        _apexes = [m for m in algo.nets.modules() if isinstance(m, ApexLevel)]
        if not _apexes:
            raise SystemExit("--apex-attention: model has no ApexLevel")
        for _a in _apexes:
            _a.set_attention_mode(args.apex_attention, window=args.apex_window)
        # init_step_state -> activate_rollout_apex_attention re-applies
        # rollout_apex_mode at EVERY episode, so setting the modules alone is
        # NOT enough: the flag would be silently reverted at t=0. Pin the
        # rollout knob too so the requested regime actually survives.
        algo.rollout_apex_mode = args.apex_attention
        algo.rollout_apex_window = args.apex_window
        print("[apex] forced attention_mode=%s window=%s on %d ApexLevel(s) "
              "(and pinned rollout_apex_mode so it survives episode reset)"
              % (args.apex_attention, args.apex_window, len(_apexes)))
    else:
        from egomimic.pipeline.stages_hnet import ApexLevel as _AL
        _rm = getattr(algo, "rollout_apex_mode", "configured")
        _rw = getattr(algo, "rollout_apex_window", None)
        for _a in [m for m in algo.nets.modules() if isinstance(m, _AL)]:
            print("[apex] load-time attention_mode=%s ; ROLLOUT will use "
                  "rollout_apex_mode=%r window=%r (applied at every episode "
                  "start by init_step_state -- the load-time value is NOT what "
                  "runs unless rollout_apex_mode is 'configured')"
                  % (_a.attention_mode, _rm, _rw))
    algo.device = device
    # --- self-only ablation (orbit probe, 2026-08-02): force L0/L1 trunk
    # attention to self-only window=1 AFTER load, per the harness rule that
    # config edits don't reliably change the model. Gated on ABLATE_SELFONLY.
    import os as _os
    if _os.environ.get("ABLATE_SELFONLY"):
        _n = 0
        for _m in algo.nets.modules():
            if type(_m).__name__ == "DualTrunkLevel" and hasattr(_m, "attn_window"):
                _m.attn_window = 1
                _n += 1
        print("[selfonly] forced attn_window=1 on %d DualTrunkLevel(s)" % _n, flush=True)
    if _os.environ.get("ABLATE_FULLATTN"):
        _n = 0
        for _m in algo.nets.modules():
            if type(_m).__name__ == "DualTrunkLevel" and hasattr(_m, "attn_window"):
                _m.attn_window = None
                _n += 1
        print("[fullattn] forced attn_window=None (full) on %d DualTrunkLevel(s)" % _n, flush=True)
    if _os.environ.get("PUSHSHAPES_UNIFORM_ATTN"):
        # K/Q-gating discriminator: replace selective self-attention with a
        # mask-normalized UNIFORM average of V (per head, out proj kept) in
        # the trunk's IntraStreamModule blocks. If withattn's stabilization
        # survives this, the mechanism is averaging, not K/Q gating.
        _filt = _os.environ["PUSHSHAPES_UNIFORM_ATTN"]  # name substring or 'all'
        import torch as _t
        import types as _types

        def _uniform_forward(self, x, mask, rope_positions):
            T, H, Dh = x.shape[0], self.num_heads, self.head_dim
            h = self.norm_attn(x)
            v = self.qkv(h).reshape(T, 3, H, Dh).unbind(1)[2]
            w = mask.to(v.dtype)
            w = w / w.sum(dim=-1, keepdim=True).clamp(min=1)
            o = _t.einsum("ij,jhd->ihd", w, v).reshape(T, H * Dh)
            x = x + self.out(o)
            x = x + self.mlp(self.norm_mlp(x))
            return x

        _n = 0
        for _nm, _m in algo.nets.named_modules():
            if type(_m).__name__ == "IntraStreamModule" and (
                _filt == "all" or _filt in _nm
            ):
                _m.forward = _types.MethodType(_uniform_forward, _m)
                _n += 1
                print("[uniform_attn] patched", _nm, flush=True)
        assert _n > 0, f"PUSHSHAPES_UNIFORM_ATTN={_filt!r} matched no modules"
        print(f"[uniform_attn] {_n} IntraStreamModule(s) uniformized", flush=True)
    algo.nets.eval()
    # Re-plan cadence is the model's obs_stride (arch property). Optionally
    # override it for this eval to probe a different cadence on the same ckpt.
    if args.obs_stride is not None:
        algo.obs_stride = int(args.obs_stride)
        print(f"[sim] obs_stride override = {algo.obs_stride} "
              f"(re-observe every {algo.obs_stride} frame(s))")
    if args.replan_every is not None:
        algo.replan_every = int(args.replan_every)
        print(f"[sim] replan_every override = {algo.replan_every}")
    import torch as _torch
    _torch.manual_seed(int(args.seed))

    # Build the dataset from the full hydra .hydra/config.yaml (the ckpt's
    # config_tree only contains the model subtree — no data/robot/evaluator).
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
    if args.init_seeds:
        # sim_v2 curated seeds: explicit list wins over base+range
        _explicit = [int(x) for x in str(args.init_seeds).split(",") if x.strip()]
        args.n_episodes = len(_explicit)
    else:
        _explicit = None
    init_seeds = (
        (_explicit if _explicit is not None
         else [args.init_seed_base + i for i in range(args.n_episodes)])
        if args.init_mode == "seeds"
        else None
    )
    if init_seeds is not None:
        print(f"[sim] init_mode=seeds  seeds={init_seeds[0]}..{init_seeds[-1]}")
    eval_cls = HPTSimEval if args.eval_class == "hpt" else PackedSimEval
    # --- action revert list, derived from the run's own config -------------
    # A load-time transform can change the ACTION LAYOUT (u_socket
    # ThetaToRotVec: (x,y,th) -> (x,y,cos,sin)). The model then predicts in that
    # space while env.step wants the raw one, so the rollout must apply the
    # REVERT list. Detect it from the resolved config rather than a flag, so a
    # rotvec run can never be silently evaluated as if it were theta.
    _revert_lists = {}
    # Read the RUN'S OWN resolved config FILE. `cfg` is local to
    # load_algo_from_ckpt and not in scope here (a NameError would silently
    # disable detection); reading the path is also more robust than depending
    # on an interpolation-clean to_yaml().
    _raw_cfg = ""
    try:
        if getattr(args, "config_path", None):
            with open(args.config_path) as _f:
                _raw_cfg = _f.read()
    except Exception as _e:
        print("[revert] WARNING: could not read config for transform "
              "detection: %r" % (_e,))
    _emb = args.embodiment_name
    if "ThetaToRotVec" in _raw_cfg or "rotvec" in _raw_cfg:
        from egomimic.rldb.embodiment.pushshapes import (
            get_rotvec_revert_transform_list,
        )
        _revert_lists[_emb] = get_rotvec_revert_transform_list()
        print("[revert] ThetaToRotVec detected in config -> applying "
              "get_rotvec_revert_transform_list() to predictions for emb=%s "
              "(atan2 decode BEFORE the env action-width slice)" % _emb)
    else:
        print("[revert] no action-layout transform in config; predictions used raw")
    sim_eval = eval_cls(
        transform_lists=_revert_lists,
        env_kwargs={"object_shape": "T", "pusher_shape": args.pusher,
                    "obstacle_level": args.obstacle_level,
                    **({"solid_pusher": True} if args.solid_pusher else {})},
        embodiment_name=args.embodiment_name,
        init_mode=args.init_mode,
        init_seeds=init_seeds,
        max_steps=args.max_steps,
        report_max_coverage=args.max_coverage,
        coverage_threshold=args.coverage_threshold,
        limit_val_batches=args.n_episodes,
        rng_pairing=args.rng_pairing,
        run_full_horizon=args.full_horizon,
        fixed_goal=args.fixed_goal,
        rollout_timeout_s=args.rollout_timeout,
    )
    sim_eval.trainer = _MockTrainer(args.out_dir, device)
    sim_eval.model = algo
    # ADDITIVE: enable per-step rollout capture BEFORE the rollout runs
    sim_eval.capture_rollout = bool(args.save_rollout_zarr)

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

    print("[rollout] starting sim eval (fp32 — inference_step now matches model dtype) ...")
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

    if args.save_rollout_zarr:
        _write_rollout_zarrs(sim_eval, args)
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
                tvio.write_video(str(fp), torch.from_numpy(ims), fps=30, video_codec="h264")
                print(f"  wrote {fp}  cov={c:.4f}  shape={ims.shape}")

    print("\n=== SMOKE PASSED ===")



def _write_rollout_zarrs(sim_eval, args):
    """ADDITIVE: dump captured rollouts as dataset-format zarr episodes."""
    import numpy as np
    import importlib as _il
    # one resolver for every spelling (PUSHSHAPES_SIM=Tsimulation_legacy -> v1)
    from Tsimulation.pushshapes import package_name, version_from_env
    ZarrDemoWriter = _il.import_module(
        package_name(version_from_env()) + ".collect.zarr_writer").ZarrDemoWriter

    caps = getattr(sim_eval, "_last_per_ep_rollout", {})
    if not caps:
        print("[rollout-zarr] nothing captured (capture_rollout not set?)")
        return
    env_args = {
        "object_shape": "T",
        "pusher_shape": args.pusher,
        "obstacle_level": args.obstacle_level,
        "image_size": 96,
        "fps": 30,
        "collector": "rollout_capture",
    }
    total = 0
    for emb_id, eps in caps.items():
        w = ZarrDemoWriter(
            path=args.save_rollout_zarr, env_args=env_args, image_size=96, fps=30,
            embodiment=("pushshapes_sim_small_circle" if int(emb_id) == 17
                        else "pushshapes_sim"),
        )
        for rec in eps:
            steps = rec['steps'] if isinstance(rec, dict) else rec
            ep_init = rec.get('episode_init') if isinstance(rec, dict) else None
            if not steps:
                continue
            w.start_episode()
            for st in steps:
                w.add_step(
                    image=st["image"],
                    pusher_obs_pose=st["agent_pos"],
                    object_obs_pose=st["object_pose"],
                    pusher_cmd_pose=st["agent_pos"],
                    action=st["action"],
                    reward=0.0,
                    goal_pose=st["goal_pose"],
                )
            idx = w.commit_episode()
            # parity with base datasets: episode_init in group attrs
            if ep_init is not None:
                try:
                    import zarr as _z, glob as _g, os as _o
                    _cands = sorted(_g.glob(_o.path.join(args.save_rollout_zarr,
                                                        f'episode_*_{idx:06d}.zarr')))
                    if _cands:
                        _g0 = _z.open_group(_cands[-1], mode='a')
                        _g0.attrs['episode_init'] = json.dumps(ep_init)
                        print(f'[rollout-zarr]   + episode_init -> {_o.path.basename(_cands[-1])}')
                except Exception as _e:
                    print(f'[rollout-zarr]   ! episode_init failed: {_e}')
            total += 1
            print(f"[rollout-zarr] emb{emb_id} -> episode idx {idx} "
                  f"({len(steps)} steps)")
        w.close()
    print(f"[rollout-zarr] wrote {total} episodes to {args.save_rollout_zarr}")

if __name__ == "__main__":
    main()
