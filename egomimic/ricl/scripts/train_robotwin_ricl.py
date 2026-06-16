"""Train + verify EgoVerse's RICL pipeline on RoboTwin 2.0 (joint-space shim).

The RoboTwin analog of ``train_droid_ricl.py``: runs the *real* RICL machinery
(``build_ricl_collate`` -> ``PIRicl`` prefix conditioning -> pi0.5 flow matching ->
loss -> backward -> AdamW) on RoboTwin bimanual demos via the in-memory shim in
``egomimic.ricl.robotwin_data`` (no Zarr/SQL port). It reuses the production
``ModelWrapper`` + Lightning ``Trainer``; only the data layer + Hydra entrypoint
are bypassed. Eval mirrors the DROID conditions (retrieval / random-within /
random-bank / clean floor) via the same evaluator wrapper.

Stages:
  --stage cpu  : build/load the retrieval cache, collate one batch, assert
                 shapes/masks/prompt-splice + token budget. A fast data-path smoke
                 with no model load; ``--embed fake`` skips the DINOv2 encoder.
  --stage full : construct PIRicl + ModelWrapper + RoboTwin eval + Trainer and fit
                 on GPU (submit via Hydra submitit).

Data-path smoke:
  python egomimic/ricl/scripts/train_robotwin_ricl.py --stage cpu \\
    --root <robotwin_task_root> --embed fake --k 4 --action-horizon 15
"""

from __future__ import annotations

import argparse
import functools
import os

import torch

from egomimic.ricl import robotwin_data as R

RICL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Local (gitignored) PaliGemma tokenizer — avoids the gated HF download (cluster).
TOKENIZER_DIR = os.path.join(RICL_DIR, "pg_tokenizer")
CKPT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(RICL_DIR)),
    "egomimic/algo/pi_checkpoints/pi05_base_pytorch",
)
DEFAULT_OUT = os.path.join(RICL_DIR, "outputs", "robotwin_train")


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def _retrieved_steps(args) -> int:
    """Retrieved action steps to encode in the prompt (#1). 0 -> full chunk."""
    return (
        args.retrieved_action_steps
        if args.retrieved_action_steps > 0
        else args.action_horizon
    )


def _make_embed_provider(args, corpus):
    if args.embed == "fake":
        return R.make_fake_embedding_provider(corpus, dim=args.fake_embed_dim)
    return R.make_dinov2_embedding_provider(corpus)


def _make_collate(args, cache, provider, state_dim):
    from egomimic.ricl.data import build_ricl_collate

    # Retrieved state/action are TEXT-ONLY (discretized bins spliced into the
    # prompt); encode RoboTwin's real qpos dims (``state_dim``, e.g. 14 for
    # aloha-agilex), not the 32-D slot-fill, so retrieved bins match the query's
    # State block. Carry the FULL retrieved action chunk (#1).
    return build_ricl_collate(
        cache,
        provider,
        k=args.k,
        action_horizon=args.action_horizon,
        image_hw=(224, 224),
        state_dim=state_dim,
        action_dim=state_dim,
    )


_RICL_COLLATE_FIELDS = ("images", "state", "action", "mask", "dist")


def _combined_collate(primary, variants):
    """Wrap the kNN collate so each val batch ALSO carries random-control blocks
    under tagged keys (so all conditions score the same frames in one pass)."""

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

    corpus = R.RoboTwinCorpus(args.root)
    embed = _make_embed_provider(args, corpus)
    if args.cache_dir and os.path.isdir(args.cache_dir):
        cache = RetrievalCache.load(args.cache_dir)
        print(
            f"[data] loaded train retrieval cache ({len(cache.query_hashes)} queries) "
            f"from {args.cache_dir}",
            flush=True,
        )
    else:
        cache = R.build_robotwin_retrieval_cache(
            corpus, k=args.k, embeddings_provider=embed
        )
        print(
            f"[data] built train retrieval cache ({len(cache.query_hashes)} queries)",
            flush=True,
        )
    provider = R.make_robotwin_bank_provider(corpus, action_horizon=args.action_horizon)
    train_collate = _make_collate(args, cache, provider, corpus.state_dim)
    if args.no_incontext:
        # PLAIN-FINETUNE BASELINE: no in-context demos in training (the honest
        # comparison the k=0 floor can't give). Val collate left unchanged.
        from egomimic.pl_utils.pl_data_utils import annotation_collate

        train_collate = annotation_collate
        print("[data] NO-INCONTEXT baseline: plain pi0.5 finetune collate", flush=True)

    if args.eval_root:
        # Separate new-task eval set; share the TRAIN corpus's quantiles so both
        # sides normalize identically (the model was trained in train-norm units).
        eval_corpus = R.RoboTwinCorpus(args.eval_root, quantiles=corpus.quantiles)
        eval_embed = _make_embed_provider(args, eval_corpus)
        train_hashes = [h for hs in corpus.group_to_hashes.values() for h in hs]
        val_hashes = [h for hs in eval_corpus.group_to_hashes.values() for h in hs]
        eval_cache = R.build_robotwin_retrieval_cache(
            eval_corpus, k=args.k, embeddings_provider=eval_embed
        )
        eval_provider = R.make_robotwin_bank_provider(
            eval_corpus, action_horizon=args.action_horizon
        )
        val_collate = _make_collate(
            args, eval_cache, eval_provider, eval_corpus.state_dim
        )
        if not args.no_random:
            within_cache = R.build_random_neighbor_cache(
                eval_corpus,
                k=args.k,
                scope="within",
                seed=args.seed,
                embeddings_provider=eval_embed,
            )
            bank_cache = R.build_random_neighbor_cache(
                eval_corpus,
                k=args.k,
                scope="bank",
                seed=args.seed,
                embeddings_provider=eval_embed,
            )
            val_collate = _combined_collate(
                val_collate,
                {
                    "randwithin": _make_collate(
                        args, within_cache, eval_provider, eval_corpus.state_dim
                    ),
                    "randbank": _make_collate(
                        args, bank_cache, eval_provider, eval_corpus.state_dim
                    ),
                },
            )
            print("[data] random-control caches built (within + bank)", flush=True)
        train_ds = R.RoboTwinQueryDataset(corpus, train_hashes, args.action_horizon)
        val_ds = R.RoboTwinQueryDataset(eval_corpus, val_hashes, args.action_horizon)
        print(
            f"[data] NEW-TASK SPLIT: train {len(corpus.group_to_hashes)} tasks "
            f"({len(train_hashes)} eps) | eval {len(eval_corpus.group_to_hashes)} "
            f"NEW tasks ({len(val_hashes)} eps)",
            flush=True,
        )
        return corpus, cache, train_collate, val_collate, train_ds, val_ds

    # Diagnostic same-corpus episode split (smoke only): hold out the last episode
    # of each task. The canonical RICL test is --eval-root (a separate new-task set).
    train_hashes, val_hashes = [], []
    for _, hs in corpus.group_to_hashes.items():
        n_val = max(1, len(hs) // 5)
        train_hashes += hs[:-n_val]
        val_hashes += hs[-n_val:]
    train_ds = R.RoboTwinQueryDataset(corpus, train_hashes, args.action_horizon)
    val_ds = R.RoboTwinQueryDataset(corpus, val_hashes, args.action_horizon)
    print(
        f"[data] EPISODE-SPLIT (diagnostic): {len(train_hashes)} train eps | "
        f"{len(val_hashes)} val eps",
        flush=True,
    )
    return corpus, cache, train_collate, train_collate, train_ds, val_ds


# --------------------------------------------------------------------------- #
# Stage: cpu — verify the data path without the model
# --------------------------------------------------------------------------- #
def _load_tokenizer(args):
    """Tokenizer for the budget check: the vendored pg_tokenizer or an explicit
    --tokenizer. Returns None only if none is available (then the structural prompt
    checks still run but the exact token-budget assertion is skipped)."""
    from transformers import AutoTokenizer

    for cand in (args.tokenizer, TOKENIZER_DIR):
        if cand and (os.path.isdir(cand) or "/" not in cand):
            try:
                return AutoTokenizer.from_pretrained(cand)
            except Exception as e:  # gated / missing
                print(
                    f"[cpu] tokenizer {cand!r} unavailable ({e}); trying next",
                    flush=True,
                )
    return None


def stage_cpu(args):
    from egomimic.ricl import conditioning as C

    corpus, cache, collate, val_collate, train_ds, val_ds = build_data(args)
    D = corpus.state_dim
    print(
        f"[cpu] detected state_dim={D}, gripper_slots={corpus.gripper_slots}",
        flush=True,
    )
    idxs = (0, len(train_ds) // 2, len(train_ds) - 1)
    batch = collate([train_ds[i] for i in idxs])
    B, k = len(idxs), args.k
    assert batch["ricl_retrieved_images"].shape == (B, k, 3, 224, 224)
    assert batch["ricl_retrieved_state"].shape == (B, k, D)
    # (#1) full retrieved action chunk: (B, k, action_horizon, D)
    assert batch["ricl_retrieved_action"].shape == (
        B,
        k,
        args.action_horizon,
        D,
    )
    assert batch["ricl_retrieved_mask"].shape == (B, k)
    assert (
        batch["ricl_retrieved_images"].min() >= 0
        and batch["ricl_retrieved_images"].max() <= 1.0
    )
    assert batch[R.AC_KEY].shape == (B, args.action_horizon, 32)
    assert batch[R.STATE_KEY].shape == (B, 32)
    assert batch["ricl_retrieved_mask"].any(), "no neighbors found for train frames"
    print(
        f"[cpu] collate shapes OK; "
        f"valid neighbors/sample={batch['ricl_retrieved_mask'].sum(1).tolist()}",
        flush=True,
    )

    # prompt splice (structural — no tokenizer needed)
    states = batch["ricl_retrieved_state"]
    actions = batch["ricl_retrieved_action"]
    valid = batch["ricl_retrieved_mask"]
    steps = _retrieved_steps(args)
    spliced_prompts = []
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
            + " ".join(["128"] * D)
            + C.PI05_ANCHOR
        )
        spliced = C.splice_retrieved_into_prompt(base, block)
        assert spliced.endswith(
            C.PI05_ANCHOR
        ), "anchor must stay terminal (query block)"
        if valid[i].any():
            assert block and spliced.startswith(block)
            assert spliced.count("|") == int(valid[i].sum())  # one '|' per valid demo
        spliced_prompts.append(spliced)
    print(
        "[cpu] prompt splice OK (anchor terminal, one '|' per valid demo)", flush=True
    )

    # token budget (requires a tokenizer; skipped only if none is available)
    tok = _load_tokenizer(args)
    if tok is None:
        wc = max(len(p.split()) for p in spliced_prompts)
        print(
            f"[cpu] no tokenizer found -> skipping exact token-budget check "
            f"(worst-case ~{wc} whitespace tokens; provide --tokenizer or pg_tokenizer "
            f"for the strict --max-token-len {args.max_token_len} check)",
            flush=True,
        )
    else:
        max_tokens = max(len(tok(p)["input_ids"]) for p in spliced_prompts)
        print(
            f"[cpu] worst-case prompt = {max_tokens} tokens "
            f"(max_token_len={args.max_token_len})",
            flush=True,
        )
        assert (
            max_tokens < args.max_token_len
        ), f"prompt {max_tokens} >= max_token_len {args.max_token_len}: bump it"
    print("[cpu] ALL CHECKS PASSED", flush=True)


# --------------------------------------------------------------------------- #
# Model + Lightning wiring (cluster)
# --------------------------------------------------------------------------- #
def build_model(args):
    from types import SimpleNamespace

    from egomimic.algo.pi_ricl import PIRicl

    config = SimpleNamespace(
        pytorch_training_precision="bfloat16",
        pytorch_weight_path=args.ckpt,
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
        rules={R.EMB_NAME: R.Passthrough32()},
        fallback=R.Passthrough32(),
        ac_key=R.AC_KEY,
    )
    algo = PIRicl(
        norm_stats=R.RoboTwinNormStats(),
        camera_transforms=None,
        domains=[R.EMB_NAME],
        train_image_augs=None,
        eval_image_augs=None,
        config=config,
        ac_keys={R.EMB_NAME: R.AC_KEY},
        action_converters=action_converters,
        tokenizer_model_name=args.tokenizer or TOKENIZER_DIR,
        tokenizer_max_length=args.max_token_len,
        sampling_mode="random",
        annotation_key="annotations",
        default_prompt="",
        proprio_in_prompt=True,
        embodiment_label=True,
        state_num_bins=256,
        proprio_keys_for_prompt=[R.PROMPT_STATE_KEY],
        num_retrieved_observations=args.k,
        retrieved_action_steps=_retrieved_steps(args),
        ricl_base_key="base_0_rgb",
        image_resolution=(224, 224),
    )
    return algo


def stage_full(args):
    import lightning as L
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
    from lightning.pytorch.utilities import CombinedLoader
    from torch.utils.data import DataLoader
    from transformers import get_cosine_schedule_with_warmup

    from egomimic.ricl.robotwin_eval import (
        RoboTwinRiclEval,
        RoboTwinRiclModelWrapper,
    )

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
    train_cl = CombinedLoader({R.EMB_NAME: train_loader}, "max_size_cycle")
    val_cl = CombinedLoader({R.EMB_NAME: val_loader}, "max_size_cycle")

    algo = build_model(args)
    if args.adam8bit:
        # 8-bit AdamW (bitsandbytes): optimizer state ~7GB vs ~29GB for fp32 moments,
        # so the 3.6B pi0.5 full fine-tune fits on a single <=48GB GPU (a40/l40s).
        # Required on this cluster (no >48GB GPU); H200 runs can keep plain AdamW.
        import bitsandbytes as bnb

        optimizer = functools.partial(
            bnb.optim.AdamW8bit,
            lr=args.lr,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=1e-10,
        )
        print("[full] optimizer: 8-bit AdamW (bitsandbytes)", flush=True)
    else:
        optimizer = functools.partial(
            torch.optim.AdamW,
            lr=args.lr,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=1e-10,
        )
    scheduler = functools.partial(
        get_cosine_schedule_with_warmup,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
        num_cycles=0.5,
    )
    model = RoboTwinRiclModelWrapper(
        robomimic_model=algo,
        optimizer=optimizer,
        scheduler=scheduler,
        scheduler_interval="step",
        enable_grad_norm=True,
    )
    evaluator = RoboTwinRiclEval(
        compute_floor=True,
        compute_random=not args.no_random,
        n_flow_samples=args.flow_samples,
        compute_sampled=args.compute_sampled,
        interp_lamdas=[float(x) for x in args.interp_lamdas.split(",") if x.strip()]
        if args.interp_lamdas
        else [],
        gripper_indices=corpus.gripper_slots,  # detected (e.g. (6, 13) for aloha-agilex)
        raw_dim=corpus.state_dim,  # detected (e.g. 14 for aloha-agilex)
        limit_val_batches=args.limit_val_batches,
    )

    os.makedirs(args.out, exist_ok=True)
    logger = CSVLogger(args.out, name=args.name, flush_logs_every_n_steps=10)
    # Persist the TRAIN corpus's norm quantiles next to the checkpoints so the
    # closed-loop eval (robotwin_policy.PIRiclPolicy) normalizes identically to
    # training. Pair (quantiles.json, checkpoint) when wiring deploy_policy.yml.
    corpus.save_quantiles(os.path.join(logger.log_dir, "quantiles.json"))
    print(f"[full] saved quantiles -> {logger.log_dir}/quantiles.json", flush=True)
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
        precision="bf16-mixed",
        max_steps=args.max_steps,
        val_check_interval=args.val_every,
        check_val_every_n_epoch=None,
        limit_val_batches=args.limit_val_batches,
        num_sanity_val_steps=0,
        log_every_n_steps=args.log_every,
        logger=logger,
        default_root_dir=args.out,
        callbacks=[ckpt_cb],
    )
    evaluator.trainer = trainer
    evaluator.model = model.model
    model.evaluator = evaluator

    print(f"[full] metrics CSV -> {logger.log_dir}/metrics.csv", flush=True)
    if args.validate_only:
        if not args.resume_from:
            raise SystemExit("--validate-only requires --resume-from <ckpt>")
        trainer.validate(
            model, dataloaders=val_cl, ckpt_path=args.resume_from, weights_only=False
        )
        return
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
    p.add_argument(
        "--root", required=True, help="RoboTwin task root (parent of <task>/data)"
    )
    p.add_argument("--eval-root", default="", help="separate new-task eval root")
    p.add_argument(
        "--cache-dir", default="", help="load a prebuilt retrieval cache from here"
    )
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--name", default="robotwin_ricl")
    p.add_argument(
        "--ckpt", default=CKPT_DIR, help="pi0.5 base checkpoint (full stage)"
    )
    p.add_argument(
        "--tokenizer", default="", help="tokenizer dir/name (default: pg_tokenizer)"
    )
    p.add_argument(
        "--embed",
        choices=["fake", "dinov2"],
        default="fake",
        help="per-frame embedding source for the retrieval cache (fake=stub for a fast smoke)",
    )
    p.add_argument("--fake-embed-dim", type=int, default=256)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--action-horizon", type=int, default=15)
    p.add_argument(
        "--retrieved-action-steps",
        type=int,
        default=0,
        help="(#1) retrieved action steps encoded in the prompt; 0 = full chunk. "
        "16-D RoboTwin chunks are token-heavy: k=4 full 15-step worst case is large, "
        "so --max-token-len defaults high. Lower this to shrink the prompt.",
    )
    p.add_argument("--max-token-len", type=int, default=4608)
    p.add_argument("--interp-lamdas", type=str, default="")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=4000)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument(
        "--adam8bit",
        action="store_true",
        help="use 8-bit AdamW (bitsandbytes) to fit the optimizer state on <=48GB GPUs",
    )
    p.add_argument("--val-every", type=int, default=250)
    p.add_argument("--limit-val-batches", type=int, default=10)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-random", action="store_true")
    p.add_argument(
        "--no-incontext", action="store_true", help="plain-finetune baseline"
    )
    p.add_argument("--flow-samples", type=int, default=8)
    p.add_argument("--compute-sampled", action="store_true")
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--ckpt-every", type=int, default=1000)
    p.add_argument("--resume-from", default=None)
    return p


def main():
    args = build_argparser().parse_args()
    if args.stage == "cpu":
        stage_cpu(args)
    else:
        stage_full(args)


if __name__ == "__main__":
    main()
