"""Portable per-cell training and durable checkpoint evaluation workers.

No Slurm submission, remote shell, downloads, or credentials are hidden here.
Stage the repository, environment, data, normalization and checkpoints locally.
Use prepare, inspect, train, worker, or import-ready; GPU execution is explicit.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
BUDGETS = (24, 32, 48, 64, 72, 96)
OPERATORS = (4, 8, 16, 32, 64, 128)
CELLS = tuple(f"b{b}_op{n}" for b in BUDGETS for n in OPERATORS)
BASE_COMMIT = "79a9815c9e3990d26a47bed784a243075808489b"
PRESERVED_SPLIT_KEYS = (
    "operators",
    "train",
    "seen_matched",
    "seen_perlevel",
    "video_seen_matched",
    "video_seen_perlevel",
)


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("x") as stream:
        os.chmod(temporary, 0o600)
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextlib.contextmanager
def lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def entry(manifest_path, cell):
    manifest = manifest_path if isinstance(manifest_path, dict) else read(manifest_path)
    need(manifest.get("schema_version") == 1, "Unknown manifest schema")
    need(
        cell in CELLS and set(manifest["cells"]) == set(CELLS),
        "Need the exact 36-cell grid",
    )
    run = manifest["cells"][cell]
    private = Path(manifest["private_dir"]).resolve()
    split_path = private / "splits" / f"fold_moving_{cell}.json"
    need(digest(split_path) == run["split_sha256"], "Evaluation split changed")
    need(
        digest(Path(run["norm_stats_dir"]) / "norm_stats.json")
        == run["norm_stats_sha256"],
        "Reference normalization changed",
    )
    need(
        manifest["source_repository_commit"] == BASE_COMMIT, "Unexpected source recipe"
    )
    need(
        len(
            {
                run["training_wandb_id"],
                run["evaluation_wandb_id"],
                run["source_wandb_id"],
            }
        )
        == 3,
        "Training, evaluation and source W&B IDs must be separate",
    )
    return manifest, run


def prepare(args):
    """Materialize a private, portable launch manifest; never launch a process."""
    inputs = read(args.inputs)
    need(set(inputs["cells"]) == set(CELLS), "Inputs must contain exactly all 36 cells")
    private = Path(inputs["private_dir"]).resolve()
    source_private = Path(inputs["source_private_dir"]).resolve()
    root = Path(inputs["run_root"]).resolve()
    need(
        not root.exists() or not any(root.iterdir()),
        "Choose an empty new Lambda run root",
    )
    episodes = read(private / "moving_opscale_design.json")["episodes"]
    splits = {
        cell: read(private / "splits" / f"fold_moving_{cell}.json") for cell in CELLS
    }
    all_train_ops = set().union(*(set(x["operators"]) for x in splits.values()))
    all_train_eps = set().union(*(set(x["train"]) for x in splits.values()))
    shared_unseen = None
    manifest = {
        "schema_version": 1,
        "created_at": now(),
        "run_root": str(root),
        "private_dir": str(private),
        "source_repository_commit": BASE_COMMIT,
        "source_private_dir": str(source_private),
        "cells": {},
        "checkpoint_every_epochs": 50,
        "checkpoint_every_updates": 5000,
        "target_global_step": 400000,
        "validation_batch_limit": 800,
        "r2_sources": inputs.get("r2_sources", []),
    }
    for cell in CELLS:
        source = inputs["cells"][cell]
        split = splits[cell]
        original = read(source_private / "splits" / f"fold_moving_{cell}.json")
        for key in PRESERVED_SPLIT_KEYS:
            need(split[key] == original[key], f"{cell}: original {key} changed")
        unseen = split["unseen"]
        need(
            len(unseen) == len(set(unseen)) == 400, "Need 400 distinct unseen episodes"
        )
        unseen_ops = {episodes[x]["user_id"] for x in unseen}
        need(
            len(unseen_ops) == 20
            and all(
                sum(episodes[x]["user_id"] == op for x in unseen) == 20
                for op in unseen_ops
            ),
            "Need 20 unseen operators x20 episodes",
        )
        need(
            not unseen_ops & all_train_ops and not set(unseen) & all_train_eps,
            "Unseen cohort intersects the training union",
        )
        need(
            shared_unseen is None or unseen == shared_unseen,
            "Unseen membership must match all cells",
        )
        shared_unseen = unseen
        norm = Path(source["norm_stats_dir"]).resolve()
        need((norm / "norm_stats.json").is_file(), f"{cell}: normalization missing")
        resume = Path(source["resume_checkpoint"]["local_path"]).resolve()
        need(resume.is_file(), f"{cell}: resume checkpoint missing")
        need(
            re.fullmatch(r"[0-9a-f]{64}", source["resume_checkpoint"]["sha256"]),
            "Invalid resume SHA",
        )
        identity = hashlib.sha256((str(root) + cell).encode()).hexdigest()[:10]
        run = {
            "budget_hours": int(cell.split("_")[0][1:]),
            "operator_count": int(cell.split("op")[1]),
            "source_wandb_id": source["source_wandb_id"],
            "training_wandb_id": f"lat{identity}",
            "evaluation_wandb_id": f"lae{identity}",
            "split_sha256": digest(private / "splits" / f"fold_moving_{cell}.json"),
            "original_split_sha256": digest(
                source_private / "splits" / f"fold_moving_{cell}.json"
            ),
            "norm_stats_dir": str(norm),
            "norm_stats_sha256": digest(norm / "norm_stats.json"),
            "resume_checkpoint": {
                "local_path": str(resume),
                "sha256": source["resume_checkpoint"]["sha256"],
            },
            "source_stop_receipt": source.get("source_stop_receipt"),
            "directory": str(root / cell),
            "queue_dir": str(root / cell / "queue"),
        }
        manifest["cells"][cell] = run
    # Compose every configuration before publishing any launchable state.
    import shlex
    import shutil

    from omegaconf import OmegaConf

    prepared = {}
    for cell in CELLS:
        train_cfg = compose(manifest, cell, "train")
        eval_cfg = compose(
            manifest,
            cell,
            "eval",
            checkpoint=manifest["cells"][cell]["resume_checkpoint"]["local_path"],
            metadata={"epoch_zero_based": 0, "global_step": 100},
        )
        eval_cfg.trainer.checkpoint_epoch = "???"
        eval_cfg.trainer.checkpoint_step = "???"
        prepared[cell] = (
            OmegaConf.to_yaml(train_cfg, resolve=False),
            OmegaConf.to_yaml(eval_cfg, resolve=False),
        )
    root.parent.mkdir(parents=True, exist_ok=True)
    stage = root.with_name(root.name + f".prepare-{os.getpid()}")
    stage.mkdir(mode=0o700)
    target = root / "manifest.json"
    try:
        atomic(stage / "manifest.json", manifest)
        for cell, (train_yaml, eval_yaml) in prepared.items():
            (stage / cell).mkdir()
            (stage / cell / "train.yaml").write_text(train_yaml)
            (stage / cell / "eval_reference.yaml").write_text(eval_yaml)
            for role in ("train", "worker"):
                command = [
                    sys.executable,
                    "-u",
                    "-m",
                    "experiments.fold_clothes_moving_opscale.lambda_async.runtime",
                    role,
                    "--manifest",
                    str(target),
                    "--cell",
                    cell,
                ]
                path = stage / cell / f"{role}.sh"
                path.write_text(
                    "#!/usr/bin/env bash\nset -euo pipefail\ncd "
                    + shlex.quote(str(REPO))
                    + "\nexec "
                    + shlex.join(command)
                    + "\n"
                )
                path.chmod(0o700)
        if root.exists():
            root.rmdir()  # Only the empty directory accepted above.
        stage.rename(root)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    print(
        json.dumps(
            {
                "manifest": str(target),
                "training_configs": 36,
                "evaluation_configs": 36,
                "launched": False,
            }
        )
    )


def compose(
    manifest_path,
    cell,
    role,
    *,
    checkpoint=None,
    metadata=None,
    output=None,
    checkpoint_sha256=None,
):
    import hydra
    from omegaconf import OmegaConf, open_dict

    import egomimic.utils.hydra_resolvers  # noqa: F401

    manifest, run = entry(manifest_path, cell)
    os.environ["EGOVERSE_MECKA_PRIVATE_DIR"] = manifest["private_dir"]
    os.environ["EGOVERSE_ROOT"] = str(REPO)
    with hydra.initialize_config_dir(
        config_dir=str(REPO / "egomimic/hydra_configs"), version_base="1.3"
    ):
        cfg = hydra.compose(
            config_name="fold_moving_common", overrides=["hydra/launcher=basic"]
        )
    base = OmegaConf.load(
        REPO
        / "experiments/fold_clothes_moving_opscale"
        / f"budget_{run['budget_hours']}"
        / f"op_{run['operator_count']}.yaml"
    )
    del base["defaults"]
    with open_dict(cfg):
        cfg = OmegaConf.merge(cfg, base)
    # trainHydra normally registers this trusted-recipe resolver at import.
    # Configuration inspection happens before importing the training runtime.
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    with open_dict(cfg):
        cfg.paths.root_dir = str(REPO)
        cfg.paths.work_dir = str(REPO)
        cfg.paths.output_dir = str(output or Path(run["directory"]) / role)
        cfg.paths.log_dir = str(Path(run["directory"]) / "logs")
        cfg.paths.cache_dir = os.environ.get(
            "EGOVERSE_CACHE_DIR", str(Path(manifest["run_root"]) / "cache")
        )
        cfg.trainer.default_root_dir = cfg.paths.output_dir
        cfg.norm_stats.precomputed_norm_path = run["norm_stats_dir"]
        cfg.norm_stats.save_cache_dir = cfg.paths.output_dir
        cfg.train_viz = False
        cfg.val_at_start = False
        cfg.model.ema.enabled = False
        cfg.logger.wandb.id = (
            run["training_wandb_id"] if role == "train" else run["evaluation_wandb_id"]
        )
        cfg.logger.wandb.name = f"lambda_{role}_{cell}"
        cfg.logger.wandb.resume = "allow"
        cfg.logger.wandb.group = "fold_clothes_moving_opscale_lambda_async_v1"
        cfg.logger.wandb.job_type = (
            "training" if role == "train" else "checkpoint_evaluation"
        )
        cfg.logger.wandb.tags = list(cfg.logger.wandb.tags) + [
            "lambda_async_v1",
            "fold-moving-opscale-async-20261107",
            "fold-moving-opscale-async-train-20261107"
            if role == "train"
            else "fold-moving-opscale-async-eval-20261107",
            f"source_run={run['source_wandb_id']}",
            f"cell={cell}",
            f"role={role}",
        ]
        cfg.async_lineage = {
            "source_wandb_id": run["source_wandb_id"],
            "training_wandb_id": run["training_wandb_id"],
            "evaluation_wandb_id": run["evaluation_wandb_id"],
            "split_sha256": run["split_sha256"],
            "cell": cell,
        }
        cfg.logger.wandb.config = dict(cfg.async_lineage)
        if role == "train":
            cfg.mode = "train"
            cfg.ckpt_path = run["resume_checkpoint"]["local_path"]
            cfg.trainer.limit_val_batches = 0
            cfg.trainer.num_sanity_val_steps = 0
            cfg.data.valid_datasets = {}
            cfg.data.extra_valid_datasets = {}
            cfg.data.unseen_op_valid_datasets = {}
            cfg.data.video_episodes = {}
            cfg.evaluator = None
            cfg.callbacks.pop("moving_opscale_evaluation", None)
            cfg.callbacks.model_checkpoint = {
                "_target_": "egomimic.eval.async_opscale.PublishedCheckpoint",
                "dirpath": str(Path(run["directory"]) / "train/checkpoints"),
                "filename": "epoch_{epoch}",
                "save_last": "link",
                "save_top_k": -1,
                "every_n_epochs": 50,
                "save_weights_only": False,
                "save_on_train_epoch_end": True,
                "monitor": None,
                "cell": cell,
                "source_wandb_id": run["source_wandb_id"],
                "training_wandb_id": run["training_wandb_id"],
                "split_sha256": run["split_sha256"],
                "queue_dir": run["queue_dir"],
            }
        else:
            need(
                checkpoint and metadata is not None,
                "Evaluation checkpoint identity required",
            )
            cfg.mode = "validate"
            cfg.ckpt_path = str(checkpoint)
            cfg.external_checkpoint_evaluation = True
            cfg.trainer._target_ = (
                "egomimic.eval.async_opscale.CheckpointEvaluationTrainer"
            )
            cfg.trainer.checkpoint_epoch = metadata["epoch_zero_based"]
            cfg.trainer.checkpoint_step = metadata["global_step"]
            cfg.trainer.limit_val_batches = 800
            cfg.trainer.enable_checkpointing = False
            cfg.trainer.strategy = "auto"
            cfg.callbacks.pop("model_checkpoint", None)
            cfg.callbacks.checkpoint_identity = {
                "_target_": "egomimic.eval.async_opscale.CheckpointIdentity",
                "checkpoint_sha256": checkpoint_sha256 or "???",
                "split_sha256": run["split_sha256"],
                "source_wandb_id": run["source_wandb_id"],
            }
            # No training optimizer is created by validate(); its checkpoint
            # weights are loaded strictly by the existing trainHydra path.
        need(
            cfg.data.train_dataloader_params.human_bimanual.batch_size == 64,
            "Training batch changed",
        )
        need(
            cfg.trainer.limit_train_batches == 100
            and cfg.trainer.accumulate_grad_batches == 1,
            "The 100-updates-per-epoch clock changed",
        )
        need(
            cfg.trainer.max_epochs == 4000
            and cfg.model.scheduler.warmup == 7000
            and cfg.model.scheduler.stable == 323000
            and cfg.model.scheduler.decay == 70000,
            "The original optimizer schedule changed",
        )
    return cfg


def verified_record(manifest_path, cell, path, expected_sha=None):
    from egomimic.eval.async_opscale import checkpoint_metadata

    _, run = entry(manifest_path, cell)
    path = Path(path).resolve()
    before = path.stat()
    actual = digest(path)
    need(expected_sha is None or actual == expected_sha, "Checkpoint SHA mismatch")
    metadata = checkpoint_metadata(path)
    after = path.stat()
    need(
        (before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_ino, after.st_size, after.st_mtime_ns),
        "Checkpoint changed during verification",
    )
    return {
        "schema_version": 1,
        "cell": cell,
        "source_wandb_id": run["source_wandb_id"],
        "training_wandb_id": run["training_wandb_id"],
        "split_sha256": run["split_sha256"],
        "local_path": str(path),
        "sha256": actual,
        "size": after.st_size,
        "checkpoint": metadata,
        "ready_after_verification": True,
    }


def publish_record(run, record):
    path = (
        Path(run["queue_dir"])
        / "ready"
        / f"{record['checkpoint']['global_step']:09d}-{record['sha256']}.json"
    )
    if path.exists():
        previous = read(path)
        need(
            all(
                previous.get(k) == record.get(k)
                for k in (
                    "schema_version",
                    "cell",
                    "source_wandb_id",
                    "training_wandb_id",
                    "split_sha256",
                    "sha256",
                    "size",
                    "checkpoint",
                    "ready_after_verification",
                )
            ),
            "Conflicting existing READY record",
        )
    else:
        atomic(path, record)


def import_ready(args):
    _, run = entry(args.manifest, args.cell)
    remote = read(args.record)
    need(
        remote["cell"] == args.cell and remote["ready_after_verification"] is True,
        "Unverified or wrong-cell source READY record",
    )
    source_id = remote.get("wandb_id", remote.get("source_wandb_id"))
    need(source_id == run["source_wandb_id"], "Wrong source W&B identity")
    record = verified_record(
        args.manifest, args.cell, args.checkpoint, remote["sha256"]
    )
    need(
        record["size"] == remote["size"]
        and record["checkpoint"]["global_step"] == remote["checkpoint"]["global_step"],
        "Source checkpoint metadata changed",
    )
    publish_record(run, record)
    print(
        json.dumps(
            {
                "imported": args.cell,
                "step": record["checkpoint"]["global_step"],
                "sha256": record["sha256"],
            }
        )
    )


def run_train(args):
    _, run = entry(args.manifest, args.cell)
    with lock(Path(run["directory"]) / "training.lock"):
        resume = run["resume_checkpoint"]
        record = verified_record(
            args.manifest, args.cell, resume["local_path"], resume["sha256"]
        )
        stop_proof = run.get("source_stop_receipt")
        need(
            isinstance(stop_proof, dict)
            and stop_proof.get("local_path")
            and stop_proof.get("sha256"),
            "Training blocked: provide the source writer stop receipt and pinned resume cutoff",
        )
        need(
            digest(stop_proof["local_path"]) == stop_proof["sha256"],
            "Source stop receipt changed",
        )
        stopped = read(stop_proof["local_path"])
        need(
            stopped.get("schema_version") == 1
            and stopped.get("cell") == args.cell
            and stopped.get("source_wandb_id") == run["source_wandb_id"]
            and stopped.get("stopped") is True,
            "Source writer is not verified stopped",
        )
        need(
            stopped.get("source_checkpoint_sha256") == record["sha256"]
            and stopped.get("source_resume_cutoff")
            == record["checkpoint"]["global_step"],
            "Chosen source checkpoint does not match the verified migration cutoff",
        )
        ready = Path(run["queue_dir"]) / "ready"
        existing = [read(p) for p in ready.glob("*.json")] if ready.exists() else []
        # A restarted Lambda trainer resumes the newest *published* full state,
        # never the original Pace checkpoint if newer work already exists.
        checkpoint_dir = Path(run["directory"]) / "train/checkpoints"
        unpublished = []
        for p in checkpoint_dir.glob("*.ckpt") if checkpoint_dir.exists() else []:
            if p.is_symlink() or p.name == "last.ckpt":
                continue
            if not any(
                Path(x["local_path"]).resolve() == p.resolve() for x in existing
            ):
                recovered = verified_record(args.manifest, args.cell, p)
                need(
                    recovered["checkpoint"]["completed_epochs"] % 50 == 0,
                    "Unexpected recovered cadence",
                )
                publish_record(run, recovered)
                unpublished.append(recovered)
        existing.extend(unpublished)
        for item in existing:
            need(
                item["cell"] == args.cell
                and item["split_sha256"] == run["split_sha256"],
                "Queue contains another training identity",
            )
        # Remote history is for scoring. Only checkpoints produced in this
        # trainer's directory may supersede its explicitly selected resume.
        local_training = [
            x
            for x in existing
            if Path(x["local_path"]).resolve().is_relative_to(checkpoint_dir.resolve())
        ]
        if local_training:
            newest = max(
                local_training,
                key=lambda x: (x["checkpoint"]["global_step"], x["sha256"]),
            )
            if (
                newest["checkpoint"]["global_step"]
                >= record["checkpoint"]["global_step"]
            ):
                record = verified_record(
                    args.manifest, args.cell, newest["local_path"], newest["sha256"]
                )
        publish_record(run, record)
        if record["checkpoint"]["global_step"] >= 400000:
            need(record["checkpoint"]["global_step"] == 400000, "Resume exceeds target")
            atomic(
                Path(run["queue_dir"]) / "training_complete.json",
                {
                    "schema_version": 1,
                    "cell": args.cell,
                    "completed": True,
                    "global_step": 400000,
                    "checkpoint_sha256": record["sha256"],
                    "training_wandb_id": run["training_wandb_id"],
                },
            )
            return
        cfg = compose(args.manifest, args.cell, "train")
        cfg.ckpt_path = record["local_path"]
        execute(cfg)


def execute(cfg):
    from omegaconf import OmegaConf

    output = Path(cfg.paths.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output / "resolved_config.yaml", resolve=True)
    from egomimic.trainHydra import train

    # The per-cell daemon owns the queue lock across this subprocess. GPU locks
    # are optional and configured only for evaluation pools sharing devices.
    train(cfg)


def evaluate(args):
    """One checkpoint subprocess; completion is committed after W&B finishes."""
    _, run = entry(args.manifest, args.cell)
    original = read(args.record)
    need(
        original["cell"] == args.cell
        and original["split_sha256"] == run["split_sha256"],
        "Wrong checkpoint queue identity",
    )
    need(
        original.get("ready_after_verification") is True,
        "Checkpoint was not published READY",
    )
    record = verified_record(
        args.manifest, args.cell, original["local_path"], original["sha256"]
    )
    need(record["checkpoint"] == original["checkpoint"], "Checkpoint metadata changed")
    step = record["checkpoint"]["global_step"]
    key = f"{step:09d}-{record['sha256']}"
    output = Path(run["directory"]) / "eval/checkpoints" / key
    receipt_path = output / "evaluation_receipt.json"
    if receipt_path.exists():
        receipt = read(receipt_path)
        need(
            receipt["checkpoint_sha256"] == record["sha256"]
            and receipt["split_sha256"] == run["split_sha256"],
            "Conflicting evaluation receipt",
        )
        return
    cfg = compose(
        args.manifest,
        args.cell,
        "eval",
        checkpoint=record["local_path"],
        metadata=record["checkpoint"],
        output=output,
        checkpoint_sha256=record["sha256"],
    )
    execute(cfg)
    evidence = read(output / "evaluation_status.json")
    need(
        evidence["global_step"] == step
        and evidence["epoch"] == record["checkpoint"]["epoch_zero_based"],
        "Evaluation clock is not the checkpoint clock",
    )
    split = read(
        Path(read(args.manifest)["private_dir"])
        / "splits"
        / f"fold_moving_{args.cell}.json"
    )
    for cohort in ("seen_matched", "seen_perlevel", "unseen"):
        cover = evidence["coverage"][cohort]
        need(
            cover["observed"] == cover["expected"] == len(split[cohort]),
            "Incomplete cohort coverage",
        )
        tracking = evidence["tracking_coverage"][cohort]["episodes"]
        need(
            set(tracking) == set(split[cohort])
            and all(x["scored"] > 0 for x in tracking.values()),
            "Tracking-valid episode coverage incomplete",
        )
    metrics = evidence["metrics"]
    need(
        metrics and all(math.isfinite(float(x)) for x in metrics.values()),
        "Non-finite metrics",
    )
    render_due = record["checkpoint"]["completed_epochs"] % 200 == 0 or step == 400000
    expected_videos = (
        sum(
            len(split[f"video_{c}"])
            for c in ("seen_matched", "seen_perlevel", "unseen")
        )
        if render_due
        else 0
    )
    need(len(evidence["videos"]) == expected_videos, "Pinned video coverage incomplete")
    for video in evidence["videos"]:
        need(
            Path(video["path"]).is_file() and Path(video["path"]).stat().st_size > 0,
            "Missing video",
        )
    validation = read(output / "validation_metrics.json")
    need(validation["ema"] is False, "EMA weights unexpectedly scored")
    # train() has returned through task_wrapper's wandb.finish() before this
    # durable receipt is written. A crash beforehand leaves an explicit retry.
    receipt = {
        "schema_version": 1,
        "completed": True,
        "completed_at": now(),
        "cell": args.cell,
        "checkpoint_sha256": record["sha256"],
        "split_sha256": run["split_sha256"],
        "checkpoint": record["checkpoint"],
        "source_wandb_id": run["source_wandb_id"],
        "training_wandb_id": run["training_wandb_id"],
        "evaluation_wandb_id": run["evaluation_wandb_id"],
        "validation_batch_limit": 800,
        "ema": False,
        "metrics": metrics,
        "coverage": evidence["coverage"],
        "videos": evidence["videos"],
        "evaluation_status_sha256": digest(output / "evaluation_status.json"),
        "resolved_config_sha256": digest(output / "resolved_config.yaml"),
    }
    atomic(receipt_path, receipt)


def sync_r2_sources(manifest_path, cell):
    """Consume immutable R2 READY records; full local SHA gates every download.

    Credentials use boto3's normal environment/profile provider chain. Only
    explicitly configured private prefixes are listed; no bucket discovery.
    """
    import boto3
    from botocore.config import Config

    manifest, run = entry(manifest_path, cell)
    all_complete = True
    covered_by_source = False
    for source in manifest.get("r2_sources", []):
        endpoint = os.environ.get(source.get("endpoint_env", "R2_ENDPOINT_URL"))
        need(
            endpoint and endpoint.startswith("https://"),
            "R2 endpoint environment variable is missing",
        )
        client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            config=Config(
                retries={"max_attempts": 5, "mode": "standard"},
                connect_timeout=15,
                read_timeout=120,
                response_checksum_validation="when_required",
            ),
        )
        bucket = source["bucket"]
        prefix = source["ready_prefix"].strip("/") + "/" + cell + "/"
        cache = Path(manifest["run_root"]) / "downloaded_checkpoints" / cell
        imported_dir = Path(run["queue_dir"]) / "remote_imports"
        for page in client.get_paginator("list_objects_v2").paginate(
            Bucket=bucket, Prefix=prefix
        ):
            for item in page.get("Contents", []):
                key = item["Key"]
                if not key.endswith(".json"):
                    continue
                need(item["Size"] <= 1024 * 1024, "Oversized READY metadata")
                source_key = hashlib.sha256((bucket + "/" + key).encode()).hexdigest()
                import_receipt = imported_dir / f"{source_key}.json"
                if import_receipt.exists():
                    saved = read(import_receipt)
                    need(
                        saved["etag"] == item["ETag"],
                        "Immutable remote READY record changed",
                    )
                    continue
                response = client.get_object(Bucket=bucket, Key=key)
                try:
                    raw = response["Body"].read(1024 * 1024 + 1)
                finally:
                    response["Body"].close()
                need(len(raw) <= 1024 * 1024, "Oversized READY metadata body")
                remote = json.loads(raw)
                need(
                    remote.get("schema_version") == 1
                    and remote.get("cell") == cell
                    and remote.get("ready_after_verification") is True,
                    "Invalid remote READY record",
                )
                need(
                    remote.get("wandb_id", remote.get("source_wandb_id"))
                    == run["source_wandb_id"],
                    "Remote READY belongs to a different source run",
                )
                checksum = remote["sha256"]
                need(re.fullmatch(r"[0-9a-f]{64}", checksum), "Invalid checkpoint SHA")
                object_prefix = source["object_prefix"].strip("/") + "/"
                need(
                    remote["object_key"].startswith(object_prefix),
                    "Checkpoint object is outside approved prefix",
                )
                local = cache / f"{checksum}.ckpt"
                cache.mkdir(parents=True, exist_ok=True)
                if not local.exists():
                    temporary = local.with_name(local.name + f".{os.getpid()}.partial")
                    try:
                        client.download_file(
                            bucket, remote["object_key"], str(temporary)
                        )
                        need(
                            temporary.stat().st_size == remote["size"]
                            and digest(temporary) == checksum,
                            "Downloaded checkpoint size or SHA mismatch",
                        )
                        with temporary.open("rb") as stream:
                            os.fsync(stream.fileno())
                        temporary.replace(local)
                    finally:
                        if temporary.exists():
                            temporary.unlink()
                record = verified_record(manifest_path, cell, local, checksum)
                need(
                    record["size"] == remote["size"]
                    and record["checkpoint"]["global_step"]
                    == remote["checkpoint"]["global_step"],
                    "Remote checkpoint metadata mismatch",
                )
                publish_record(run, record)
                atomic(
                    import_receipt,
                    {
                        "bucket": bucket,
                        "key": key,
                        "etag": item["ETag"],
                        "checkpoint_sha256": checksum,
                        "imported_at": now(),
                    },
                )
        complete_key = source.get("complete_key")
        if not complete_key:
            all_complete = False
            continue
        try:
            response = client.get_object(Bucket=bucket, Key=complete_key)
        except client.exceptions.ClientError as error:
            if str(error.response.get("Error", {}).get("Code")) in ("404", "NoSuchKey"):
                all_complete = False
                continue
            raise
        try:
            raw = response["Body"].read(4 * 1024 * 1024 + 1)
        finally:
            response["Body"].close()
        need(len(raw) <= 4 * 1024 * 1024, "Oversized export completion marker")
        complete = json.loads(raw)
        need(
            complete.get("schema_version") == 1 and complete.get("complete") is True,
            "Invalid export completion marker",
        )
        source_cell = complete.get("cells", {}).get(cell)
        if source_cell is None:
            need(
                cell in complete.get("excluded_non_pace_cells", []),
                "Cell absent from completion scope",
            )
            continue
        covered_by_source = True
        need(
            source_cell["wandb_id"] == run["source_wandb_id"],
            "Completion source run differs",
        )
        imported_shas = []
        for expected_key in source_cell["ready_keys"]:
            need(
                expected_key.startswith(prefix),
                "Completion READY key outside cell prefix",
            )
            expected_id = hashlib.sha256(
                (bucket + "/" + expected_key).encode()
            ).hexdigest()
            receipt = imported_dir / f"{expected_id}.json"
            if not receipt.exists():
                all_complete = False
            else:
                imported_shas.append(read(receipt)["checkpoint_sha256"])
        if set(imported_shas) != set(source_cell["checkpoint_sha256s"]):
            all_complete = False
    return all_complete and covered_by_source


def worker(args):
    manifest, run = entry(args.manifest, args.cell)
    root = Path(run["queue_dir"])
    shutdown = False
    child = None
    stop_started = None

    def stop(signum, frame):
        nonlocal shutdown, stop_started
        shutdown = True
        stop_started = stop_started or time.monotonic()
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    with lock(root / "evaluation_worker.lock"):
        while not shutdown:
            remote_complete = (
                sync_r2_sources(args.manifest, args.cell)
                if manifest.get("r2_sources")
                else False
            )
            local_seal = root / "source_imports_complete.json"
            local_complete = False
            if local_seal.exists():
                sealed = read(local_seal)
                need(
                    sealed.get("schema_version") == 1
                    and sealed.get("cell") == args.cell,
                    "Invalid local source-import completion receipt",
                )
                local_complete = True
            imports_complete = remote_complete or local_complete
            pending = [(read(p), p) for p in (root / "ready").glob("*.json")]
            pending.sort(
                key=lambda pair: (
                    pair[0]["checkpoint"]["global_step"],
                    pair[0]["sha256"],
                )
            )
            seen_steps = {}
            work = []
            for record, path in pending:
                need(
                    record["cell"] == args.cell
                    and record["split_sha256"] == run["split_sha256"],
                    "Queue identity changed",
                )
                step = record["checkpoint"]["global_step"]
                # Named and last snapshots can serialize differently at the
                # same update. Score each distinct SHA; do not infer equivalence.
                seen_steps.setdefault(step, set()).add(record["sha256"])
                output = (
                    Path(run["directory"])
                    / "eval/checkpoints"
                    / f"{step:09d}-{record['sha256']}"
                )
                receipt_path = output / "evaluation_receipt.json"
                if receipt_path.exists():
                    receipt = read(receipt_path)
                    need(
                        receipt["completed"] is True
                        and receipt["checkpoint_sha256"] == record["sha256"]
                        and receipt["split_sha256"] == run["split_sha256"],
                        "Invalid existing evaluation receipt",
                    )
                else:
                    work.append((record, path, output))
            atomic(
                root / "worker_status.json",
                {
                    "updated_at": now(),
                    "state": "waiting" if not work else "evaluating",
                    "queued": len(pending),
                    "pending": len(work),
                    "pid": os.getpid(),
                    "source_imports_complete": imports_complete,
                },
            )
            if work:
                record, path, output = work[0]
                output.mkdir(parents=True, exist_ok=True)
                attempts_path = output / "attempts.json"
                attempts = read(attempts_path) if attempts_path.exists() else []
                if len(attempts) >= args.max_attempts:
                    atomic(
                        root / "worker_status.json",
                        {
                            "updated_at": now(),
                            "state": "blocked",
                            "checkpoint_sha256": record["sha256"],
                            "reason": "evaluation retry limit",
                            "attempts": len(attempts),
                        },
                    )
                    raise RuntimeError(
                        "Evaluation retry limit reached; inspect logs and restart after repair"
                    )
                attempt = {"started_at": now(), "checkpoint_sha256": record["sha256"]}
                attempts.append(attempt)
                atomic(attempts_path, attempts)
                command = [
                    sys.executable,
                    "-u",
                    "-m",
                    "experiments.fold_clothes_moving_opscale.lambda_async.runtime",
                    "_evaluate",
                    "--manifest",
                    str(Path(args.manifest).resolve()),
                    "--cell",
                    args.cell,
                    "--record",
                    str(path.resolve()),
                ]
                gpu_lock = None
                try:
                    if args.gpu_lock:
                        lock_path = Path(args.gpu_lock)
                        lock_path.parent.mkdir(parents=True, exist_ok=True)
                        gpu_lock = lock_path.open("a")
                        while not shutdown:
                            try:
                                fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                                break
                            except BlockingIOError:
                                time.sleep(1)
                        if shutdown:
                            break
                    with (output / f"evaluation-attempt-{len(attempts):02d}.log").open(
                        "a"
                    ) as log:
                        child = subprocess.Popen(
                            command,
                            cwd=REPO,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                        )
                        heartbeat = 0.0
                        while child.poll() is None:
                            if (
                                shutdown
                                and stop_started is not None
                                and time.monotonic() - stop_started > 180
                            ):
                                try:
                                    os.killpg(child.pid, signal.SIGKILL)
                                except ProcessLookupError:
                                    pass
                            if time.monotonic() - heartbeat > 30:
                                atomic(
                                    root / "worker_status.json",
                                    {
                                        "updated_at": now(),
                                        "state": "evaluating",
                                        "checkpoint_sha256": record["sha256"],
                                        "global_step": record["checkpoint"][
                                            "global_step"
                                        ],
                                        "pending": len(work),
                                        "pid": os.getpid(),
                                        "child_pid": child.pid,
                                    },
                                )
                                heartbeat = time.monotonic()
                            time.sleep(1)
                        code = child.returncode
                        child = None
                finally:
                    if gpu_lock is not None:
                        gpu_lock.close()
                attempt.update({"finished_at": now(), "returncode": code})
                atomic(attempts_path, attempts)
                if shutdown:
                    break
                if code:
                    time.sleep(min(args.poll_seconds, 30))
                continue
            final = root / "training_complete.json"
            if final.exists() and imports_complete:
                complete = read(final)
                need(
                    complete["completed"] is True and complete["global_step"] == 400000,
                    "Invalid training completion receipt",
                )
                need(
                    complete["checkpoint_sha256"] in seen_steps.get(400000, set()),
                    "Final checkpoint missing from queue",
                )
                atomic(
                    root / "evaluation_complete.json",
                    {
                        "schema_version": 1,
                        "completed": True,
                        "completed_at": now(),
                        "cell": args.cell,
                        "evaluated_checkpoints": len(pending),
                        "final_checkpoint_sha256": complete["checkpoint_sha256"],
                        "split_sha256": run["split_sha256"],
                        "evaluation_wandb_id": run["evaluation_wandb_id"],
                    },
                )
                return
            for _ in range(max(1, args.poll_seconds)):
                if shutdown:
                    break
                time.sleep(1)
        atomic(
            root / "worker_status.json",
            {"updated_at": now(), "state": "stopped", "pid": os.getpid()},
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    q = commands.add_parser("prepare")
    q.add_argument("--inputs", required=True)
    for name in (
        "inspect",
        "train",
        "worker",
        "import-ready",
        "seal-imports",
        "_evaluate",
    ):
        q = commands.add_parser(name)
        q.add_argument("--manifest", required=True)
        q.add_argument("--cell", choices=CELLS, required=True)
        if name in ("import-ready", "_evaluate"):
            q.add_argument("--record", required=True)
        if name == "import-ready":
            q.add_argument("--checkpoint", required=True)
        if name == "worker":
            q.add_argument("--poll-seconds", type=int, default=30)
            q.add_argument("--max-attempts", type=int, default=3)
            q.add_argument(
                "--gpu-lock",
                default=os.environ.get("EGOVERSE_EVAL_GPU_LOCK"),
                help="Shared local lock path for workers assigned the same evaluation GPU",
            )
    args = p.parse_args()
    os.umask(0o077)
    if args.command == "prepare":
        prepare(args)
    elif args.command == "train":
        run_train(args)
    elif args.command == "worker":
        worker(args)
    elif args.command == "import-ready":
        import_ready(args)
    elif args.command == "_evaluate":
        evaluate(args)
    elif args.command == "seal-imports":
        _, run = entry(args.manifest, args.cell)
        queue = Path(run["queue_dir"])
        records = [read(p) for p in (queue / "ready").glob("*.json")]
        need(records, "Import at least the chosen source checkpoint first")
        atomic(
            queue / "source_imports_complete.json",
            {
                "schema_version": 1,
                "cell": args.cell,
                "completed_at": now(),
                "ready_sha256s": sorted({x["sha256"] for x in records}),
                "note": "Operator confirms all intended source history has been imported",
            },
        )
    else:
        from omegaconf import OmegaConf

        cfg = compose(args.manifest, args.cell, "train")
        print(OmegaConf.to_yaml(cfg, resolve=False))


if __name__ == "__main__":
    main()
