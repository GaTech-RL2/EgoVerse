"""Train + verify EgoVerse's RICL pipeline on the original-RICL DROID data.

This is the train-and-check-it-improves entrypoint for the bring-up plan: it runs
the *real* RICL machinery (``build_ricl_collate`` -> ``PIRicl`` prefix
conditioning -> pi0.5 flow matching -> loss -> backward -> AdamW) on the original
RICL paper's flat ``processed_demo.npz`` demos via the in-memory shim in
``egomimic.ricl.droid_data`` (no Zarr/SQL port). It reuses the production
``ModelWrapper`` + Lightning ``Trainer`` so optimizer / grad-norm / logging are
the real ones; only the Zarr data layer and the Hydra entrypoint are bypassed.

Success = (1) train action-loss decreases, and (2) retrieval-conditioning beats
the zero-context floor on held-out frames (``RICL/retrieval_helps`` / negative
``RICL/delta_*``). The eval here compares the flow-matching val loss with the
full retrieved context vs a *truly clean* floor (ricl keys stripped from the RAW
batch, so the floor prompt has no spliced demos and no retrieved images) under
identical sampled noise, so the delta reflects only the conditioning.

Stages:
  --stage cpu  : build/load cache, collate one batch, assert shapes/masks/prompt
                 splice/token budget. No model, runs on the login node.
  --stage full : construct PIRicl + ModelWrapper + DroidRiclEval + Trainer and
                 fit. Needs a GPU + the EgoVerse openpi env.

Run:
  PYTHONPATH=/storage/project/r-dxu345-0/rco3/EgoVerse2 \
    /storage/project/r-dxu345-0/rco3/EgoVerse/emimic/bin/python \
    egomimic/ricl/scripts/train_droid_ricl.py --stage full \
      --cache-dir egomimic/ricl/outputs/droid_cache --max-steps 4000 --batch-size 4
"""

from __future__ import annotations

import argparse
import functools
import os

import torch

from egomimic.ricl import droid_data as D

# pg_tokenizer/ and outputs/ live in the parent ricl/ dir, one level up from scripts/.
RICL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Local (gitignored) PaliGemma tokenizer — avoids the gated HF download.
TOKENIZER_DIR = os.path.join(RICL_DIR, "pg_tokenizer")
CKPT_DIR = (
    "/storage/project/r-dxu345-0/rco3/EgoVerse/egomimic/algo/"
    "pi_checkpoints/pi05_base_pytorch"
)
DEFAULT_OUT = os.path.join(RICL_DIR, "outputs", "droid_train")


# --------------------------------------------------------------------------- #
# Data: split episodes, build datasets + ricl collate over the DROID corpus
# --------------------------------------------------------------------------- #
def _retrieved_steps(args) -> int:
    """How many retrieved action steps to encode in the prompt (#1). 0 (the
    default) means the full chunk == ``action_horizon``."""
    return (
        args.retrieved_action_steps
        if args.retrieved_action_steps > 0
        else args.action_horizon
    )


def _make_collate(args, cache, provider):
    from egomimic.ricl.data import build_ricl_collate

    # Retrieved state/action are TEXT-ONLY (spliced into the prompt as discretized
    # bins); encode just DROID's real 8 dims, not the 32-D slot-fill, to keep the
    # prompt short. Carry the FULL retrieved action chunk (action_horizon steps, #1)
    # so the demo's trajectory — not one near-zero step — reaches the model.
    # Retrieved images + the model's continuous inputs are unaffected.
    return build_ricl_collate(
        cache,
        provider,
        k=args.k,
        action_horizon=args.action_horizon,
        image_hw=(224, 224),
        state_dim=D.RAW_DIM,
        action_dim=D.RAW_DIM,
    )


# Fields the ricl collate attaches per query (sans the ``ricl_retrieved_`` prefix).
_RICL_COLLATE_FIELDS = ("images", "state", "action", "mask", "dist")


def _combined_collate(primary, variants):
    """Wrap the kNN collate so each validation batch ALSO carries the random-control
    neighbor blocks under tagged keys.

    ``variants`` maps ``tag -> collate`` (each built from a *random*-neighbor cache).
    For every variant we re-run its collate over the same query samples and copy its
    ``ricl_retrieved_<field>`` into ``ricl_<tag>_retrieved_<field>``, so
    :class:`DroidRiclEval` can score retrieval / random-within / random-bank / floor
    on the *same* query frames in a single validation pass.
    """

    def collate(samples):
        batch = primary(samples)
        for tag, vcollate in variants.items():
            vbatch = vcollate(samples)
            for f in _RICL_COLLATE_FIELDS:
                key = f"ricl_retrieved_{f}"
                if key in vbatch:
                    batch[f"ricl_{tag}_retrieved_{f}"] = vbatch[key]
        return batch

    return collate


def build_data(args):
    from egomimic.ricl.retrieval import RetrievalCache

    groups = None
    if args.groups > 0:
        all_g = sorted(
            d
            for d in os.listdir(D.DROID_ROOT)
            if os.path.isdir(os.path.join(D.DROID_ROOT, d)) and not d.startswith(".")
        )
        groups = all_g[: args.groups]
    corpus = D.DroidCorpus(groups=groups)

    if args.cache_dir and os.path.isdir(args.cache_dir):
        cache = RetrievalCache.load(args.cache_dir)
        print(
            f"[data] loaded train retrieval cache ({len(cache.query_hashes)} queries) "
            f"from {args.cache_dir}",
            flush=True,
        )
    else:
        cache = D.build_droid_retrieval_cache(corpus, k=args.k)
        print(
            f"[data] built train retrieval cache ({len(cache.query_hashes)} queries)",
            flush=True,
        )
    provider = D.make_droid_bank_provider(corpus, action_horizon=args.action_horizon)
    train_collate = _make_collate(args, cache, provider)
    if args.no_incontext:
        # PLAIN-FINETUNE BASELINE: train with NO in-context demos. The base
        # annotation_collate attaches no ``ricl_*`` keys, so PIRicl._build_prompts /
        # _robomimic_to_pi_data fall through to base PI — i.e. a vanilla pi0.5
        # finetune on the train tasks only. This is the honest comparison the
        # k=0 "floor" condition does NOT provide: the floor reuses the RICL model's
        # weights (trained WITH the in-context mechanism) at k=0, whereas this
        # baseline's weights NEVER saw a retrieved demo. The val collate is left
        # UNCHANGED (still carries retrieval/random/floor) so the baseline is scored
        # on the same new tasks/frames and compared head-to-head with RICL.
        from egomimic.pl_utils.pl_data_utils import annotation_collate

        train_collate = annotation_collate
        print(
            "[data] NO-INCONTEXT baseline: train collate strips retrieved demos "
            "(plain pi0.5 finetune); val collate unchanged",
            flush=True,
        )

    # --- Original-RICL split: train on collected_demos_training (ALL groups),
    #     eval on the SEPARATE new-task set (collected_demos). Each eval query
    #     retrieves from its own new-task group (within-group leave-one-out),
    #     exactly the original repo's `baseline` bank. ----------------------- #
    if args.eval_root:
        import re

        all_eval = sorted(
            d
            for d in os.listdir(args.eval_root)
            if os.path.isdir(os.path.join(args.eval_root, d)) and not d.startswith(".")
        )
        # Drop the "-5demos"/"-10demos" sample-efficiency subset variants so each
        # new task is counted once (keep the full group, e.g. the full idli set).
        eval_groups = [d for d in all_eval if not re.search(r"-\d+demos", d)]
        eval_corpus = D.DroidCorpus(root=args.eval_root, groups=eval_groups)
        train_hashes = [h for _, hs in corpus.group_to_hashes.items() for h in hs]
        val_hashes = [h for _, hs in eval_corpus.group_to_hashes.items() for h in hs]
        eval_cache = D.build_droid_retrieval_cache(eval_corpus, k=args.k)
        eval_provider = D.make_droid_bank_provider(
            eval_corpus, action_horizon=args.action_horizon
        )
        val_collate = _make_collate(args, eval_cache, eval_provider)
        # Random-retrieval controls (independent of loader shuffling / batch
        # composition): random demos from the query's OWN new task (within) and
        # from ALL new tasks pooled (bank). _combined_collate carries both blocks
        # alongside the kNN block so all four conditions score the same frames.
        if not args.no_random:
            within_cache = D.build_random_neighbor_cache(
                eval_corpus, k=args.k, scope="within", seed=args.seed
            )
            bank_cache = D.build_random_neighbor_cache(
                eval_corpus, k=args.k, scope="bank", seed=args.seed
            )
            val_collate = _combined_collate(
                val_collate,
                {
                    "randwithin": _make_collate(args, within_cache, eval_provider),
                    "randbank": _make_collate(args, bank_cache, eval_provider),
                },
            )
            print("[data] random-control caches built (within + bank)", flush=True)
        train_ds = D.DroidQueryDataset(corpus, train_hashes, args.action_horizon)
        val_ds = D.DroidQueryDataset(eval_corpus, val_hashes, args.action_horizon)
        print(
            f"[data] ORIGINAL-RICL SPLIT: train on {len(corpus.group_to_hashes)} "
            f"training tasks ({len(train_hashes)} eps) | eval on "
            f"{len(eval_corpus.group_to_hashes)} NEW tasks ({len(val_hashes)} eps)",
            flush=True,
        )
        print(
            "       new (eval) tasks: " + ", ".join(eval_corpus.group_to_hashes),
            flush=True,
        )
        print(
            f"[data] {len(train_ds)} train frames | {len(val_ds)} val frames",
            flush=True,
        )
        return corpus, cache, train_collate, val_collate, train_ds, val_ds

    # NOTE: the canonical RICL design is `--eval-root` above (train on one task set,
    # eval on a SEPARATE new-task set). The same-corpus splits below are diagnostic
    # only and were found flawed for measuring "does retrieval help on a new task"
    # (the model has seen the task during training) — kept for quick smoke checks.
    train_hashes, val_hashes = [], []
    gitems = list(corpus.group_to_hashes.items())
    if args.holdout_groups > 0:
        # Diagnostic TASK-level holdout: the last N training groups are val-only.
        # Flawed as a RICL test because these tasks are near-siblings of the trained
        # ones, so the floor stays strong; use --eval-root for the real new-task eval.
        train_groups, val_groups = (
            gitems[: -args.holdout_groups],
            gitems[-args.holdout_groups :],
        )
        for _, hs in train_groups:
            train_hashes += hs
        for _, hs in val_groups:
            val_hashes += hs
        print(
            f"[data] TASK-HOLDOUT: {len(train_groups)} train tasks | "
            f"{len(val_groups)} held-out (unseen) tasks",
            flush=True,
        )
        print(
            "       held-out tasks: " + ", ".join(g for g, _ in val_groups), flush=True
        )
    else:
        # per-group episode split: hold out the last `val_per_group` episodes of
        # EVERY group (the model trains on other episodes of the same task ->
        # within-task generalization, NOT an unseen task).
        for _, hs in gitems:
            n_val = min(args.val_per_group, max(1, len(hs) // 5))
            train_hashes += hs[:-n_val]
            val_hashes += hs[-n_val:]
        print(
            f"[data] EPISODE-SPLIT: {len(gitems)} groups (all in train+val) | "
            f"{len(train_hashes)} train eps | {len(val_hashes)} val eps",
            flush=True,
        )

    # Same-corpus split: train and val retrieve from the SAME cache/provider.
    train_ds = D.DroidQueryDataset(corpus, train_hashes, args.action_horizon)
    val_ds = D.DroidQueryDataset(corpus, val_hashes, args.action_horizon)
    print(f"[data] {len(train_ds)} train frames | {len(val_ds)} val frames", flush=True)
    return corpus, cache, train_collate, train_collate, train_ds, val_ds


# --------------------------------------------------------------------------- #
# Stage: cpu — verify the data path without the model
# --------------------------------------------------------------------------- #
def stage_cpu(args):
    from transformers import AutoTokenizer

    from egomimic.ricl import conditioning as C

    corpus, cache, collate, val_collate, train_ds, val_ds = build_data(args)
    batch = collate([train_ds[i] for i in (0, len(train_ds) // 2, len(train_ds) - 1)])
    B, k = 3, args.k
    assert batch["ricl_retrieved_images"].shape == (B, k, 3, 224, 224)
    assert batch["ricl_retrieved_state"].shape == (B, k, D.RAW_DIM)
    # (#1) full retrieved action chunk: (B, k, action_horizon, 8), not (B,k,1,8)
    assert batch["ricl_retrieved_action"].shape == (
        B,
        k,
        args.action_horizon,
        D.RAW_DIM,
    )
    assert batch["ricl_retrieved_mask"].shape == (B, k)
    assert (
        batch["ricl_retrieved_images"].min() >= 0
        and batch["ricl_retrieved_images"].max() <= 1.0
    )
    assert batch[D.AC_KEY].shape == (B, args.action_horizon, 32)
    assert batch["ricl_retrieved_mask"].any(), "no neighbors found for train frames"
    print(
        f"[cpu] collate shapes OK; "
        f"valid neighbors/sample={batch['ricl_retrieved_mask'].sum(1).tolist()}",
        flush=True,
    )

    # prompt splice + token budget against the real tokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    states = batch["ricl_retrieved_state"]
    actions = batch["ricl_retrieved_action"]
    valid = batch["ricl_retrieved_mask"]
    steps = _retrieved_steps(args)
    max_tokens = 0
    for i in range(B):
        task = corpus.prompt(batch["episode_hash"][i])
        block = C.build_retrieved_prompt_block(
            states[i],
            actions[i],
            valid[i],
            prompt=task,
            num_bins=256,
            action_steps=steps,
        )
        base = (
            f"Task: {task}, Embodiment: eva bimanual, State: "
            + " ".join(["128"] * D.RAW_DIM)
            + C.PI05_ANCHOR
        )
        spliced = C.splice_retrieved_into_prompt(base, block)
        assert spliced.endswith(
            C.PI05_ANCHOR
        ), "anchor must stay terminal (query block)"
        if valid[i].any():
            # demos are prepended as in-context exemplars before the query block
            assert block and spliced.startswith(block)
            assert spliced.count("|") == int(valid[i].sum())  # one '|' per valid demo
        n = len(tok(spliced)["input_ids"])
        max_tokens = max(max_tokens, n)
    print(
        f"[cpu] prompt splice OK; worst-case prompt = {max_tokens} tokens "
        f"(max_token_len={args.max_token_len})",
        flush=True,
    )
    assert (
        max_tokens < args.max_token_len
    ), f"prompt {max_tokens} >= max_token_len {args.max_token_len}: bump it"
    print("[cpu] ALL CHECKS PASSED", flush=True)


# --------------------------------------------------------------------------- #
# Model + Lightning wiring
# --------------------------------------------------------------------------- #
def build_model(args):
    from types import SimpleNamespace

    from egomimic.algo.pi_ricl import PIRicl

    config = SimpleNamespace(
        pytorch_training_precision="bfloat16",
        pytorch_weight_path=CKPT_DIR,
        model=SimpleNamespace(
            pi05=True,
            action_dim=32,
            action_horizon=args.action_horizon,
            max_token_len=args.max_token_len,
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m",
        ),
    )
    action_converters = SimpleNamespace(
        rules={D.EMB_NAME: D.Passthrough32()},
        fallback=D.Passthrough32(),
        ac_key=D.AC_KEY,
    )
    algo = PIRicl(
        norm_stats=D.DroidNormStats(),
        camera_transforms=None,
        domains=[D.EMB_NAME],
        train_image_augs=None,
        eval_image_augs=None,
        config=config,
        ac_keys={D.EMB_NAME: D.AC_KEY},
        action_converters=action_converters,
        tokenizer_model_name=TOKENIZER_DIR,
        tokenizer_max_length=args.max_token_len,
        sampling_mode="random",
        annotation_key="annotations",
        default_prompt="",
        proprio_in_prompt=True,
        embodiment_label=True,
        state_num_bins=256,
        # text State block uses the compact 8-D state; the model's continuous
        # proprio still comes from the 32-D STATE_KEY (norm_stats proprio_keys).
        proprio_keys_for_prompt=[D.PROMPT_STATE_KEY],
        num_retrieved_observations=args.k,
        # (#1) encode the full retrieved chunk by default (0 -> action_horizon).
        retrieved_action_steps=_retrieved_steps(args),
        ricl_base_key="base_0_rgb",
        image_resolution=(224, 224),
    )
    return algo


def stage_full(args):
    import lightning as L
    from lightning.pytorch.loggers import CSVLogger
    from torch.utils.data import DataLoader

    from egomimic.pl_utils.pl_data_utils import annotation_collate  # noqa: F401
    from egomimic.ricl.droid_eval import DroidRiclEval, DroidRiclModelWrapper

    torch.manual_seed(args.seed)
    L.seed_everything(args.seed, workers=True)

    corpus, cache, train_collate, val_collate, train_ds, val_ds = build_data(args)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_collate,
        num_workers=args.num_workers,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=val_collate,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )
    # CombinedLoader keyed by embodiment name -> batch is {emb_name: {...}}
    from lightning.pytorch.utilities import CombinedLoader

    train_cl = CombinedLoader({D.EMB_NAME: train_loader}, "max_size_cycle")
    val_cl = CombinedLoader({D.EMB_NAME: val_loader}, "max_size_cycle")

    algo = build_model(args)

    optimizer = functools.partial(
        torch.optim.AdamW,
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=1e-10,
    )
    from transformers import get_cosine_schedule_with_warmup

    scheduler = functools.partial(
        get_cosine_schedule_with_warmup,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
        num_cycles=0.5,
    )

    model = DroidRiclModelWrapper(
        robomimic_model=algo,
        optimizer=optimizer,
        scheduler=scheduler,
        scheduler_interval="step",
        enable_grad_norm=True,
    )
    evaluator = DroidRiclEval(
        compute_floor=True,
        compute_random=not args.no_random,
        n_flow_samples=args.flow_samples,  # (#1) low-variance flow loss
        compute_sampled=args.compute_sampled,  # (#2) sampled-action MSE/L1
        # (#2) distance-weighted interpolation: sweep these lambdas (each emits its
        # own sampled_mse_l{lam}); the headline sampled_mse stays the no-interp model.
        interp_lamdas=[float(x) for x in args.interp_lamdas.split(",") if x.strip()]
        if args.interp_lamdas
        else ([args.interp_lamda] if args.interp_lamda > 0 else []),
        gripper_indices=(D.GRIPPER_SLOT,),
        limit_val_batches=args.limit_val_batches,
    )

    os.makedirs(args.out, exist_ok=True)
    csv_logger = CSVLogger(args.out, name=args.name, flush_logs_every_n_steps=10)
    # The CSV logger owns log_dir (used below for the checkpoint dir + the metrics
    # path printout); W&B is added alongside it when --wandb is set.
    logger = csv_logger
    loggers = [csv_logger]
    if args.wandb:
        from lightning.pytorch.loggers import WandbLogger

        wandb_logger = WandbLogger(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            name=args.wandb_name or args.name,
            save_dir=args.out,
            config=vars(args),
        )
        loggers.append(wandb_logger)
    # Save best (by held-out loss) + last so a run is resumable and new eval
    # conditions can be added without retraining from scratch. Both are full
    # Lightning checkpoints (model + optimizer); top_k=1 keeps disk to ~2 files.
    from lightning.pytorch.callbacks import ModelCheckpoint

    ckpt_cb = ModelCheckpoint(
        dirpath=os.path.join(logger.log_dir, "checkpoints"),
        filename="step{step}-{Valid/action_loss:.4f}",
        monitor="Valid/action_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        every_n_train_steps=args.ckpt_every,
        auto_insert_metric_name=False,
    )
    trainer = L.Trainer(
        accelerator="gpu",
        devices=args.devices,
        strategy="ddp_find_unused_parameters_true" if args.devices > 1 else "auto",
        precision="bf16-mixed",  # matches trainer/default.yaml (`precision: bf16`)
        max_steps=args.max_steps,
        val_check_interval=args.val_every,
        check_val_every_n_epoch=None,
        limit_val_batches=args.limit_val_batches,
        num_sanity_val_steps=0,
        log_every_n_steps=args.log_every,
        logger=loggers,
        default_root_dir=args.out,
        callbacks=[ckpt_cb],
    )
    evaluator.trainer = trainer
    evaluator.model = model.model
    model.evaluator = evaluator

    print(f"[full] metrics CSV -> {logger.log_dir}/metrics.csv", flush=True)
    if args.validate_only:
        # One-shot validation of a finished checkpoint — applies new eval
        # conditions (#1 low-variance flow loss / #2 sampled-action MSE) WITHOUT
        # retraining. Resolves to `--resume-from` (the saved .ckpt) so the model
        # weights come from disk, not the freshly-loaded base.
        if not args.resume_from:
            raise SystemExit("--validate-only requires --resume-from <ckpt>")
        print(
            f"[full] VALIDATE-ONLY on {args.resume_from} "
            f"(flow_samples={args.flow_samples}, sampled={args.compute_sampled})",
            flush=True,
        )
        # weights_only=False: our own checkpoint stores a functools.partial
        # (optimizer/scheduler factory) in its hparams, which PyTorch 2.6's default
        # weights-only unpickler rejects. The ckpt is trusted (we wrote it).
        results = trainer.validate(
            model, dataloaders=val_cl, ckpt_path=args.resume_from, weights_only=False
        )
        # Echo the aggregated metrics to stdout so the numbers are captured even if
        # CSVLogger flushing is finicky for a validate-only (no-fit) run.
        if results:
            print("[validate-only] aggregated metrics:", flush=True)
            for k, v in sorted(results[0].items()):
                print(f"    {k} = {v:.6f}", flush=True)
        return
    print(
        f"[full] starting fit: max_steps={args.max_steps} bs={args.batch_size} "
        f"devices={args.devices} lr={args.lr}",
        flush=True,
    )
    trainer.fit(
        model,
        train_dataloaders=train_cl,
        val_dataloaders=val_cl,
        ckpt_path=args.resume_from,
    )


# --------------------------------------------------------------------------- #
def build_argparser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=["cpu", "full"], required=True)
    p.add_argument("--cache-dir", default=D.DEFAULT_CACHE_DIR)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--name", default="droid_ricl")
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--action-horizon", type=int, default=15)
    p.add_argument(
        "--retrieved-action-steps",
        type=int,
        default=0,
        help="(#1) retrieved action steps encoded in the prompt; 0 = full chunk "
        "(== action_horizon). Full chunk is the faithful RICL signal but costs "
        "tokens: k=4 full 15-step worst case ~2150 tokens -> needs max_token_len ~2304.",
    )
    # Full 15-step retrieved chunks (#1) push the worst-case prompt to ~2150 tokens
    # (k=4, 8-D, 256 bins); 2304 leaves margin so the trailing anchor is never
    # truncated. Lower --retrieved-action-steps to shrink this.
    p.add_argument("--max-token-len", type=int, default=2304)
    p.add_argument(
        "--interp-lamda",
        type=float,
        default=0.0,
        help="(#2) single distance-weighted interpolation strength (w=exp(-lamda*"
        "dist/dist_max) toward the nearest demo's chunk). 0 disables. Prefer "
        "--interp-lamdas to sweep. Only affects --compute-sampled metrics.",
    )
    p.add_argument(
        "--interp-lamdas",
        type=str,
        default="",
        help="(#2) comma-separated lambda SWEEP, e.g. '0.5,1,2,5,10'. Each emits a "
        "sampled_mse_l{lam} per condition (one sampling pass); the headline "
        "sampled_mse stays the no-interp model so beats_random reflects the model. "
        "dist is normalized by the batch-max kNN distance, so lambda~1-2 is moderate.",
    )
    p.add_argument("--groups", type=int, default=0, help="limit task groups (0=all)")
    p.add_argument(
        "--eval-root",
        default="",
        help="ORIGINAL-RICL SPLIT: eval on a SEPARATE new-task set at this root "
        "(e.g. .../collected_demos). Each eval query retrieves within its own new "
        "task (leave-one-out); the random-within/bank controls are built in-process. "
        "When unset, falls back to the diagnostic same-corpus splits below.",
    )
    p.add_argument(
        "--val-per-group",
        type=int,
        default=2,
        help="episode-split mode: held-out episodes per group (model sees the task)",
    )
    p.add_argument(
        "--holdout-groups",
        type=int,
        default=0,
        help="task-holdout mode: hold out the last N task GROUPS entirely "
        "for val (model never trains on them) -> the real RICL test",
    )
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=4000)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--val-every", type=int, default=250)
    p.add_argument("--limit-val-batches", type=int, default=10)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--no-random",
        action="store_true",
        help="skip the random-retrieval control condition during eval",
    )
    p.add_argument(
        "--no-incontext",
        action="store_true",
        help="PLAIN-FINETUNE BASELINE: train with NO retrieved demos (base "
        "annotation_collate, no ricl_* keys) -> a vanilla pi0.5 finetune on the "
        "train tasks. The eval/val collate is unchanged (retrieval/random/floor "
        "still scored), so this baseline checkpoint can be compared head-to-head "
        "with the RICL model on the new tasks. The honest comparison the k=0 "
        "floor (RICL weights at inference k=0) does not provide.",
    )
    p.add_argument(
        "--flow-samples",
        type=int,
        default=8,
        help="(#1) average the flow val loss over N noise/time draws "
        "(shared seeds across conditions) to cut MC variance",
    )
    p.add_argument(
        "--compute-sampled",
        action="store_true",
        help="(#2) also score sampled-action MSE/L1 (slow torch.compile "
        "sampling path) — the real action-prediction error",
    )
    p.add_argument(
        "--validate-only",
        action="store_true",
        help="run ONE validation pass on --resume-from and exit (apply new "
        "eval conditions to a finished checkpoint without retraining)",
    )
    p.add_argument(
        "--ckpt-every",
        type=int,
        default=1000,
        help="save a checkpoint every N train steps (best+last always kept)",
    )
    p.add_argument(
        "--resume-from",
        default=None,
        help="path to a .ckpt to resume training/eval from (e.g. .../last.ckpt)",
    )
    p.add_argument(
        "--wandb",
        action="store_true",
        help="also log metrics to Weights & Biases (in addition to the CSV logger)",
    )
    p.add_argument(
        "--wandb-project",
        default="ricl",
        help="W&B project to log into (default: ricl)",
    )
    p.add_argument(
        "--wandb-entity",
        default="rl2-group",
        help="W&B team/entity (default: rl2-group; pass '' to use your default)",
    )
    p.add_argument(
        "--wandb-name",
        default=None,
        help="W&B run name (default: --name)",
    )
    return p


def main():
    args = build_argparser().parse_args()
    if args.stage == "cpu":
        stage_cpu(args)
    else:
        stage_full(args)


if __name__ == "__main__":
    main()
