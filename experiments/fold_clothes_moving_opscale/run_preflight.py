#!/usr/bin/env python3
"""Run CPU normalization or the required real GPU evaluation/resume smoke.

No allocation is submitted here. The conductor runs norm inside CPU jobs and
smoke inside a single H100/H200 allocation after staging is verified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import runpy
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from experiments.fold_clothes_moving_opscale.normalization_provenance import (
    normalization_reuse_identity,
    normalization_source_matches,
)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
EVALUATION_POLICY = "tracking_valid_windows_v1"
TRAINING_POLICY = "raw_tracking_valid_windows_v1"
SMOKE_TRAIN_STEPS = 64
DIAGNOSTIC_TRAIN_STEPS = 512
TRAINING_BATCH_SIZE = 64
TRAINING_LOSS_CEILING = 1000.0


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n")
    temp.replace(path)


def context(args):
    root = args.private_dir.resolve()
    runs = args.run_root.resolve()
    repo = training_repository(args)
    training_repo = getattr(args, "training_repo", None)
    controller_commit = getattr(args, "controller_commit", None)
    need(
        bool(training_repo) == bool(controller_commit),
        "Both training repository and controller commit are required",
    )
    if training_repo:
        from launch_pace import configure_training_source

        args.operational_controller = configure_training_source(
            repo, controller_commit, args.commit
        )
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    need(len(args.commit) == 40 and commit == args.commit, "Preflight commit mismatch")
    need(
        not subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo, text=True
        ).strip(),
        "Preflight requires the clean recorded source commit",
    )
    design_hash = sha(root / "moving_opscale_design.json")
    ready = json.loads((runs / "readiness.json").read_text())
    need(
        ready.get("status") == "passed" and ready.get("design_sha256") == design_hash,
        "Data readiness proof is missing, blocked, or for another design",
    )
    from launch_pace import validate_data_proof

    validate_data_proof(root, ready)
    os.environ["EGOVERSE_MECKA_PRIVATE_DIR"] = str(root)
    os.environ["EGOVERSE_ROOT"] = str(repo)
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [str(repo), str(runs / "python-deps"), os.environ.get("PYTHONPATH", "")]
    )
    sys.path[:0] = [str(repo), str(runs / "python-deps")]
    return root, runs, commit, design_hash


def training_repository(args):
    return Path(getattr(args, "training_repo", None) or REPO).resolve()


def training_norm_summary(payload, episodes):
    """Authenticate full training-window eligibility in newly computed norms."""
    metadata = payload.get("norm_run_metadata", {})
    need(
        metadata.get("training_tracking_policy") == TRAINING_POLICY,
        "Normalization lacks the strict raw training tracking policy",
    )
    need(metadata.get("sampling_seed") == 42026, "Normalization sampling seed differs")
    coverage = metadata.get("tracking_coverage", {})
    need(
        isinstance(coverage, dict) and set(coverage) == set(episodes) and coverage,
        "Normalization tracking coverage does not match every selected training episode",
    )
    totals = {key: 0 for key in ("eligible_starts", "valid_starts", "rejected_starts")}
    for item in coverage.values():
        need(
            isinstance(item, dict)
            and all(type(item.get(key)) is int and item[key] >= 0 for key in totals),
            "Normalization tracking counts are invalid",
        )
        need(
            item["valid_starts"] > 0
            and item["eligible_starts"]
            == item["valid_starts"] + item["rejected_starts"],
            "Normalization has an unusable episode or inconsistent tracking counts",
        )
        for key in totals:
            totals[key] += item[key]
    return {
        "policy": TRAINING_POLICY,
        "sampling_seed": 42026,
        "episode_count": len(coverage),
        **totals,
        "coverage_sha256": hashlib.sha256(
            json.dumps(coverage, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def validate_training_norm(report, payload, episodes, commit):
    proof = training_norm_summary(payload, episodes)
    need(
        report.get("status") == "passed"
        and report.get("commit") == commit
        and report.get("training_tracking_policy") == TRAINING_POLICY,
        "Training norms must be freshly computed under this source and tracking policy",
    )
    need(
        report.get("training_validity") == proof,
        "Normalization tracking proof changed or is missing",
    )
    return proof


def verify_training_evidence(
    evidence, *, commit, job_id, cell, start_step, steps, norm_sha256
):
    need(
        evidence.get("status") == "passed"
        and evidence.get("training_tracking_policy") == TRAINING_POLICY,
        "Strict training evidence is missing or failed",
    )
    need(
        evidence.get("commit") == commit
        and evidence.get("job_id") == str(job_id)
        and evidence.get("cell") == cell
        and evidence.get("seed") == 42026
        and evidence.get("norm_stats_sha256") == norm_sha256,
        "Training evidence identity differs",
    )
    need(
        evidence.get("start_step") == start_step
        and evidence.get("expected_steps") == steps
        and evidence.get("batch_size") == TRAINING_BATCH_SIZE
        and evidence.get("verified_samples") == steps * TRAINING_BATCH_SIZE
        and evidence.get("loss_ceiling") == TRAINING_LOSS_CEILING,
        "Training evidence does not prove the required update count and full batch64",
    )
    updates = evidence.get("updates", [])
    need(
        isinstance(updates, list) and len(updates) == steps,
        "Training update evidence is incomplete",
    )
    for index, update in enumerate(updates, start_step + 1):
        need(
            update.get("step") == index
            and update.get("batch_size") == TRAINING_BATCH_SIZE,
            "Training update sequence or batch size differs",
        )
        loss, grad = update.get("loss"), update.get("gradient_norm")
        need(
            type(loss) in (int, float)
            and math.isfinite(loss)
            and 0 <= loss <= TRAINING_LOSS_CEILING,
            "Training loss is nonfinite or exceeds the diagnostic ceiling",
        )
        need(
            type(grad) in (int, float) and math.isfinite(grad) and grad >= 0,
            "Training gradients are nonfinite or missing",
        )
    if cell == "b24_op16":
        probe = evidence.get("forced_resampling", {})
        need(
            probe.get("requested_raw_valid") is False
            and probe.get("actual_raw_valid") is True
            and probe.get("same_episode") is True
            and bool(probe.get("episode"))
            and type(probe.get("requested_frame")) is int
            and type(probe.get("actual_frame")) is int
            and probe["requested_frame"] != probe["actual_frame"],
            "Representative training did not exercise a real invalid start within the same episode",
        )
    return {
        "updates": steps,
        "samples": steps * TRAINING_BATCH_SIZE,
        "loss_min": min(item["loss"] for item in updates),
        "loss_max": max(item["loss"] for item in updates),
        "gradient_norm_max": max(item["gradient_norm"] for item in updates),
    }


def validate_smoke_training(report, runs, commit):
    """Reopen all four per-update receipts; booleans alone never grant launch."""
    runs = Path(runs).resolve()
    need(
        report.get("status") == "passed"
        and report.get("commit") == commit
        and report.get("training_tracking_policy") == TRAINING_POLICY
        and report.get("run_root") == str(runs),
        "Strict training smoke identity missing",
    )
    job = str(report.get("job_id", ""))
    need(job.isdigit(), "Training smoke job identity missing")
    phases = report.get("runs", [])
    need(
        len(phases) == 3
        and [(item.get("operator_count"), item.get("epochs")) for item in phases]
        == [(4, 1), (4, 2), (128, 1)],
        "Full evaluation/resume smoke phases are incomplete",
    )
    diagnostic = report.get("training_diagnostic", {})
    need(
        diagnostic.get("operator_count") == 16,
        "Representative op16 training diagnostic missing",
    )
    results = []
    for item, cell, start, steps in [
        (diagnostic, "b24_op16", 0, DIAGNOSTIC_TRAIN_STEPS),
        (phases[0], "b24_op4", 0, SMOKE_TRAIN_STEPS),
        (phases[1], "b24_op4", SMOKE_TRAIN_STEPS, SMOKE_TRAIN_STEPS),
        (phases[2], "b24_op128", 0, SMOKE_TRAIN_STEPS),
    ]:
        proof = item.get("training_evidence", {})
        path = (
            runs
            / "smoke"
            / commit[:12]
            / f"attempt_{job}"
            / cell
            / f"training_evidence_{start}_{start + steps}.json"
        )
        need(
            proof.get("path") == str(path)
            and path.is_file()
            and proof.get("sha256") == sha(path),
            "Training evidence artifact is missing, misplaced or changed",
        )
        normalized = json.loads(
            (runs / "norm" / cell / "norm_complete.json").read_text()
        )
        stats = runs / "norm" / cell / "norm_stats/norm_stats.json"
        split = json.loads(
            (runs / "private/splits" / f"fold_moving_{cell}.json").read_text()
        )
        need(
            normalized.get("stats_path") == str(stats)
            and normalized.get("stats_sha256") == sha(stats),
            "Training diagnostic normalization changed",
        )
        norm_proof = validate_training_norm(
            normalized, json.loads(stats.read_text()), split["train"], commit
        )
        need(
            item.get("normalization") == norm_proof,
            "Smoke normalization policy proof differs",
        )
        result = verify_training_evidence(
            json.loads(path.read_text()),
            commit=commit,
            job_id=job,
            cell=cell,
            start_step=start,
            steps=steps,
            norm_sha256=sha(stats),
        )
        need(
            all(proof.get(key) == value for key, value in result.items()),
            "Training evidence summary differs",
        )
        need(
            item.get("checkpoint", {}).get("step") == start + steps,
            "Smoke training/checkpoint steps differ",
        )
        results.append(result)
    return results


def training_audit_callback(
    output,
    cell,
    commit,
    job_id,
    norm_stats_path,
    norm_stats_sha256,
    start_step,
    expected_steps,
):
    """Smoke-only observer; no model, optimizer, loss or gradient modifications."""
    import random

    import numpy as np
    import torch
    from lightning.pytorch import Callback

    class TrainingAudit(Callback):
        def __init__(self):
            self.path = Path(output)
            self.report = {
                "schema_version": 1,
                "status": "running",
                "cell": cell,
                "commit": commit,
                "job_id": str(job_id),
                "seed": 42026,
                "training_tracking_policy": TRAINING_POLICY,
                "norm_stats_sha256": norm_stats_sha256,
                "start_step": start_step,
                "expected_steps": expected_steps,
                "batch_size": TRAINING_BATCH_SIZE,
                "loss_ceiling": TRAINING_LOSS_CEILING,
                "updates": [],
                "verified_samples": 0,
            }

        def on_train_start(self, trainer, pl_module):
            need(
                trainer.global_step == start_step
                and trainer.accumulate_grad_batches == 1,
                "Training audit requires the exact starting step and no accumulation",
            )
            need(os.environ.get("PL_GLOBAL_SEED") == "42026", "Training seed differs")
            self.dataset = trainer.datamodule.train_datasets["human_bimanual"]
            need(
                self.dataset.training_tracking_policy == TRAINING_POLICY,
                "Instantiated training dataset lacks strict tracking validity",
            )
            need(
                sha(norm_stats_path) == norm_stats_sha256, "Training norm file changed"
            )
            stats = json.loads(Path(norm_stats_path).read_text())
            summary = training_norm_summary(stats, self.dataset.datasets)
            need(
                stats["norm_run_metadata"]["tracking_coverage"]
                == self.dataset.training_tracking_coverage,
                "Live training eligibility differs from normalization eligibility",
            )
            self.report["normalization"] = summary
            need(
                pl_module.enable_grad_norm,
                "Existing gradient norm instrumentation is required",
            )
            # This probe consumes no training RNG state and never supplies its
            # decoded sample to the optimizer. It exercises real selected data.
            if cell == "b24_op16":
                rng = (
                    random.getstate(),
                    np.random.get_state(),
                    torch.random.get_rng_state(),
                )
                try:
                    for episode, leaf in sorted(self.dataset.datasets.items()):
                        if not self.dataset.training_tracking_coverage[episode][
                            "rejected_starts"
                        ]:
                            continue
                        eligible = getattr(leaf, "sample_indices", None)
                        eligible = range(len(leaf)) if eligible is None else eligible
                        requested = next(
                            (
                                int(i)
                                for i in eligible
                                if not leaf._training_tracking_valid_mask[int(i)]
                            ),
                            None,
                        )
                        if requested is None:
                            continue
                        need(
                            not leaf.evaluation_tracking_validity(requested),
                            "Forced invalid start unexpectedly tracked",
                        )
                        sample = leaf[requested]
                        actual = int(sample["frame_idx"])
                        actual_episode = sample["episode_hash"]
                        valid = bool(leaf.evaluation_tracking_validity(actual))
                        need(
                            actual_episode == episode
                            and actual != requested
                            and valid
                            and actual in leaf._training_tracking_valid_indices,
                            "Strict guard did not resample to a tracked start within the same episode",
                        )
                        self.report["forced_resampling"] = {
                            "episode": episode,
                            "requested_frame": requested,
                            "actual_frame": actual,
                            "requested_raw_valid": False,
                            "actual_raw_valid": valid,
                            "same_episode": True,
                        }
                        break
                    need(
                        "forced_resampling" in self.report,
                        "Representative split has no exercised invalid raw window",
                    )
                finally:
                    random.setstate(rng[0])
                    np.random.set_state(rng[1])
                    torch.random.set_rng_state(rng[2])
            write(self.path, self.report)

        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            sample = batch["human_bimanual"]
            episodes = sample["episode_hash"]
            frames = sample["frame_idx"].detach().cpu().tolist()
            need(
                len(episodes) == len(frames) == TRAINING_BATCH_SIZE
                and sample["actions_keypoints"].shape[0] == TRAINING_BATCH_SIZE,
                "Actual training batch size differs from64",
            )
            for episode, frame in zip(episodes, frames, strict=True):
                leaf = self.dataset.datasets.get(episode)
                need(
                    leaf is not None
                    and type(frame) is int
                    and 0 <= frame < len(leaf._training_tracking_valid_mask)
                    and bool(leaf._training_tracking_valid_mask[frame])
                    and frame in leaf._training_tracking_valid_indices,
                    "An untracked raw window reached an actual training batch",
                )
            self.report["verified_samples"] += TRAINING_BATCH_SIZE

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            value = outputs["loss"] if isinstance(outputs, dict) else outputs
            loss = float(value.detach().cpu())
            need(
                bool(pl_module.grad_norm_history),
                "Per-update gradient norm evidence missing",
            )
            grad = float(pl_module.grad_norm_history[-1])
            self.report["updates"].append(
                {
                    "step": int(trainer.global_step),
                    "batch_size": TRAINING_BATCH_SIZE,
                    "loss": loss,
                    "gradient_norm": grad,
                }
            )
            write(self.path, self.report)
            need(
                math.isfinite(loss) and 0 <= loss <= TRAINING_LOSS_CEILING,
                "Training loss is nonfinite or exceeds the diagnostic ceiling",
            )
            need(math.isfinite(grad), "Training gradients are nonfinite")

        def on_train_end(self, trainer, pl_module):
            self.report["status"] = "passed"
            try:
                self.report["summary"] = verify_training_evidence(
                    self.report,
                    commit=commit,
                    job_id=job_id,
                    cell=cell,
                    start_step=start_step,
                    steps=expected_steps,
                    norm_sha256=norm_stats_sha256,
                )
            except Exception:
                self.report["status"] = "failed"
                write(self.path, self.report)
                raise
            write(self.path, self.report)

        def on_exception(self, trainer, pl_module, exception):
            self.report.update(status="failed", error_type=type(exception).__name__)
            write(self.path, self.report)

    return TrainingAudit()


def training_audit_overrides(
    folder, cell, commit, job_id, normalized, start_step, steps
):
    output = folder / f"training_evidence_{start_step}_{start_step + steps}.json"
    values = {
        "_target_": "experiments.fold_clothes_moving_opscale.run_preflight.training_audit_callback",
        "output": str(output),
        "cell": cell,
        "commit": commit,
        "job_id": str(job_id),
        "norm_stats_path": normalized["stats_path"],
        "norm_stats_sha256": normalized["stats_sha256"],
        "start_step": start_step,
        "expected_steps": steps,
    }
    # Quote strings so numeric-looking job IDs and source digests stay strings.
    return output, [
        f"++callbacks.training_validity_audit.{key}={json.dumps(value)}"
        for key, value in values.items()
    ]


def norm(args):
    root, runs, commit, design_hash = context(args)
    repo = training_repository(args)
    here = repo / "experiments/fold_clothes_moving_opscale"
    need(
        args.budget in [24, 32, 48, 64, 72, 96]
        and args.operators in [4, 8, 16, 32, 64, 128],
        "Unknown moving-budget cell",
    )
    output = runs / "norm" / f"b{args.budget}_op{args.operators}"
    report = output / "norm_complete.json"
    stats = output / "norm_stats" / "norm_stats.json"
    selected_train = json.loads(
        (
            root / "splits" / f"fold_moving_b{args.budget}_op{args.operators}.json"
        ).read_text()
    )["train"]
    identity = {
        "commit": commit,
        "training_tracking_policy": TRAINING_POLICY,
        "design_sha256": design_hash,
        "budget_hours": args.budget,
        "operator_count": args.operators,
        "canonical_index_sha256": sha(root / "canonical_zarr/_episode_index.json"),
        "config_sha256": sha(
            here / f"budget_{args.budget}" / f"op_{args.operators}.yaml"
        ),
        "split_sha256": sha(
            root / "splits" / f"fold_moving_b{args.budget}_op{args.operators}.json"
        ),
    }
    if report.exists():
        previous = json.loads(report.read_text())
        if (
            previous.get("status") == "passed"
            and normalization_source_matches(runs, previous, commit, repo)
            and all(previous.get(k) == v for k, v in identity.items() if k != "commit")
            and stats.is_file()
            and previous.get("stats_sha256") == sha(stats)
        ):
            validate_training_norm(
                previous, json.loads(stats.read_text()), selected_train, commit
            )
            print(json.dumps({"norm": str(report), "status": "already_passed"}))
            return
    output.mkdir(parents=True, exist_ok=True)
    write(report, {**identity, "status": "running", "started_at": now()})
    # The original helper defaults to seed=42; the trailing seed override is
    # mandatory so global RNG setup matches the reference's 42026.
    from omegaconf import OmegaConf

    import egomimic.utils.hydra_resolvers  # noqa: F401

    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    sys.argv = [
        str(repo / "egomimic/scripts/precompute_norm_stats.py"),
        "--config-name",
        f"op_{args.operators}",
        "--config-dir",
        str(here / f"budget_{args.budget}"),
        "--data",
        "mecka_fold_moving_budget",
        "--model",
        "rdt_bc_mecka_kp_dinov3_vitb",
        "--sample-frac",
        "1.0",
        "--max-samples",
        "200000",
        "--num-workers",
        "4",
        "--out",
        str(output),
        f"hydra.searchpath=[file://{repo / 'egomimic/hydra_configs'}]",
        "seed=42026",
    ]
    try:
        runpy.run_path(
            str(repo / "egomimic/scripts/precompute_norm_stats.py"), run_name="__main__"
        )
        need(
            stats.is_file() and stats.stat().st_size > 0,
            "Normalization did not write stats",
        )
        training_validity = training_norm_summary(
            json.loads(stats.read_text()), selected_train
        )
        write(
            report,
            {
                **identity,
                "status": "passed",
                "created_at": now(),
                "stats_path": str(stats),
                "stats_sha256": sha(stats),
                "training_validity": training_validity,
                "seed": 42026,
                "sample_frac": 1.0,
                "max_samples": 200000,
                "pool_horizon": False,
            },
        )
    except Exception as error:
        write(
            report,
            {**identity, "status": "failed", "error": str(error), "finished_at": now()},
        )
        raise


def inspect_checkpoint(path):
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    need(bool(checkpoint.get("optimizer_states")), "Checkpoint has no optimizer state")
    need(bool(checkpoint.get("lr_schedulers")), "Checkpoint has no LR scheduler state")
    need(bool(checkpoint.get("state_dict")), "Checkpoint has no weights")
    result = {
        "path": str(path),
        "step": int(checkpoint["global_step"]),
        "epoch": int(checkpoint["epoch"]),
        "optimizer_state": True,
        "scheduler_state": True,
        "bytes": Path(path).stat().st_size,
    }
    del checkpoint
    return result


def wandb_video_path(record, key):
    """Accept one legacy video or Lightning's one-element W&B video group."""
    need(isinstance(record, dict), f"W&B missing video: {key}")
    if "videos" in record or record.get("_type") == "videos":
        videos = record.get("videos")
        need(
            record.get("_type") == "videos"
            and type(record.get("count")) is int
            and record["count"] == 1
            and isinstance(videos, list)
            and len(videos) == 1
            and "path" not in record,
            f"W&B expected exactly one grouped video: {key}",
        )
        record = videos[0]
    need(
        isinstance(record, dict)
        and record.get("_type") in (None, "video-file")
        and "videos" not in record,
        f"W&B invalid video record: {key}",
    )
    path = record.get("path")
    need(
        isinstance(path, str)
        and path.strip() == path
        and bool(path)
        and not path.startswith("/")
        and ".." not in path.split("/")
        and "\\" not in path,
        f"W&B missing or invalid video path: {key}",
    )
    return path


def expected_wandb_video_keys(status):
    videos = status.get("videos") if isinstance(status, dict) else None
    need(isinstance(videos, list) and videos, "Expected W&B videos must be nonempty")
    keys = []
    for video in videos:
        key = video.get("wandb_key") if isinstance(video, dict) else None
        need(
            isinstance(key, str) and bool(key) and key.strip() == key,
            "Expected W&B video key is missing or invalid",
        )
        keys.append(key)
    need(len(set(keys)) == len(keys), "Expected W&B video keys must be unique")
    return keys


def verify_wandb(run_id, status, expected_step):
    import wandb

    video_keys = expected_wandb_video_keys(status)
    api = wandb.Api(timeout=45)
    deadline = time.monotonic() + 1200
    last_error = ""
    while time.monotonic() < deadline:
        try:
            # Api.run caches Run/summary objects; each retry needs fresh backend data.
            api.flush()
            run = api.run(f"rl2-group/egoverse/{run_id}")
            summary = dict(run.summary._json_dict)
            need(run.state == "finished", f"W&B run state is {run.state}")
            losses = [
                float(v)
                for k, v in summary.items()
                if k.startswith("Train/")
                and "loss" in k.lower()
                and isinstance(v, (float, int))
            ]
            need(
                losses and all(math.isfinite(v) for v in losses),
                "W&B finite training loss missing",
            )
            need(
                int(summary.get("trainer/global_step", -1)) >= expected_step - 1,
                "W&B latest global step is behind the checkpoint",
            )
            for cohort in ["seen_matched", "seen_perlevel", "unseen"]:
                prefix = f"val/{cohort}/"
                values = [
                    float(v)
                    for k, v in summary.items()
                    if k.startswith(prefix) and isinstance(v, (float, int))
                ]
                need(
                    values and all(math.isfinite(v) for v in values),
                    f"W&B missing finite metrics: {cohort}",
                )
            paths = [wandb_video_path(summary.get(key), key) for key in video_keys]
            need(
                len(set(paths)) == len(video_keys),
                "W&B expected videos must reference distinct uploaded files",
            )
            for key, path in zip(video_keys, paths):
                size = run.file(path).size
                need(
                    type(size) is int and size > 0,
                    f"W&B video not uploaded: {key}",
                )
            return {
                "run_id": run_id,
                "url": run.url,
                "state": run.state,
                "finite_losses": losses,
                "backend_step": summary["trainer/global_step"],
                "verified_videos": len(video_keys),
            }
        except Exception as error:
            last_error = f"{type(error).__name__}: {error}"
            time.sleep(20)
    raise RuntimeError(f"W&B smoke evidence did not become available: {last_error}")


def verify_tracking_status(status, counts, *, split=None, canonical_index=None):
    """Require genuine scored coverage, not merely visited invalid data."""
    need(
        status.get("evaluation_policy") == EVALUATION_POLICY,
        "Missing current evaluation tracking policy",
    )
    coverage = status.get("tracking_coverage", {})
    need(set(coverage) == set(counts), "Missing tracking coverage cohorts")

    def check_counts(item):
        need(isinstance(item, dict), "Malformed tracking counts")
        values = [item.get(key) for key in ("visited", "scored", "skipped")]
        need(
            all(type(value) is int and value >= 0 for value in values),
            "Invalid tracking counts",
        )
        visited, scored, skipped = values
        need(
            scored > 0 and visited == scored + skipped,
            "Tracking coverage has no scored data or inconsistent counts",
        )

    for cohort, expected_count in counts.items():
        item = coverage[cohort]
        check_counts(item)
        episodes = item.get("episodes", {})
        need(len(episodes) == expected_count, "Tracking scored episode count differs")
        if split is not None:
            need(
                set(episodes) == set(split[cohort]),
                "Tracking episode identities differ from private split",
            )
        for episode in episodes.values():
            check_counts(episode)
        for key in ("visited", "scored", "skipped"):
            need(
                sum(episode[key] for episode in episodes.values()) == item[key],
                "Tracking cohort totals disagree with episodes",
            )
    for video in status.get("videos", []):
        need(
            video.get("evaluation_policy") == EVALUATION_POLICY,
            "Video tracking policy missing",
        )
        check_counts(video.get("tracking"))
        frames, unscored = video.get("source_frames"), video.get("unscored_frames")
        need(
            type(frames) is int
            and frames > 0
            and type(unscored) is int
            and 0 <= unscored <= frames,
            "Video frame accounting is invalid",
        )
        need(
            video.get("timing") == "original_source_frames_with_labeled_unscored_spans",
            "Video source timing is unverified",
        )
        if canonical_index is not None:
            need(
                frames == canonical_index[video["episode"]]["total_frames"],
                "Video does not retain all original source frames",
            )
        spans = video.get("unscored_spans")
        need(isinstance(spans, list), "Video missing unscored span accounting")
        end, total = 0, 0
        for span in spans:
            start, length = span.get("start_frame"), span.get("frame_count")
            need(
                type(start) is int
                and type(length) is int
                and start >= end
                and length > 0
                and start + length <= frames,
                "Invalid or overlapping unscored video span",
            )
            end, total = start + length, total + length
        need(total == unscored, "Unscored video frame count mismatch")
    return coverage


def reusable_first_phase(args, runs, commit, design_hash):
    """Authenticate a stopped first phase; this never grants a smoke gate pass.

    The caller must still inspect the checkpoint, verify exact local coverage,
    and query the W&B backend before running the genuine resume phase.
    """
    path = getattr(args, "reuse_first_phase", None)
    checksum = getattr(args, "reuse_first_phase_sha256", None)
    need(bool(path) == bool(checksum), "Reuse receipt and checksum must be paired")
    if not path:
        return None
    path = Path(path).resolve()
    need(sha(path) == checksum, "First-phase reuse receipt changed")
    receipt = json.loads(path.read_text())
    need(
        receipt.get("schema_version") == 1
        and receipt.get("status") == "ready_for_reverification"
        and receipt.get("commit") == commit
        and receipt.get("design_sha256") == design_hash,
        "First-phase reuse identity mismatch",
    )
    source_job = receipt.get("job_id", "")
    need(
        isinstance(source_job, str) and source_job.isdigit(), "Invalid prior smoke job"
    )
    expected_id = f"fold_moving_smoke_b24_op4_{commit[:10]}_{source_job}"
    need(receipt.get("run_id") == expected_id, "First-phase W&B identity mismatch")
    folder = Path(receipt["folder"]).resolve()
    need(
        folder.is_relative_to(runs / "smoke" / commit[:12]),
        "First phase is outside this smoke root",
    )
    need(
        not (folder / "epoch_2.log").exists(),
        "Prior resume was already attempted; review it first",
    )
    files = receipt.get("files", {})
    checkpoint_name = "0/checkpoints/epoch_epoch=0.ckpt"
    required = {
        "epoch_1.log",
        "0/.hydra/config.yaml",
        "0/.hydra/overrides.yaml",
        "0/.hydra/hydra.yaml",
        "0/git_sha.txt",
        "0/git.diff",
        "0/git_untracked.txt",
        "0/evaluation_status.json",
        checkpoint_name,
    }
    need(
        isinstance(files, dict) and required <= files.keys(),
        "First-phase artifact manifest is incomplete",
    )
    for name, expected in files.items():
        artifact = (folder / name).resolve()
        need(
            artifact.is_relative_to(folder) and artifact.is_file(),
            "Invalid first-phase artifact path",
        )
        need(sha(artifact) == expected, f"First-phase artifact changed: {name}")
    need(
        (folder / "0/git_sha.txt").read_text().strip() == commit,
        "First-phase training source differs",
    )
    need(
        not (folder / "0/git.diff").read_text().strip(), "First-phase source was dirty"
    )
    need(
        not (folder / "0/git_untracked.txt").read_text().strip(),
        "First-phase source had untracked files",
    )
    failure_path = Path(receipt["failed_report"])
    need(
        sha(failure_path) == receipt["failed_report_sha256"],
        "Prior failure evidence changed",
    )
    failure = json.loads(failure_path.read_text())
    need(
        failure.get("status") == "failed"
        and failure.get("commit") == commit
        and failure.get("design_sha256") == design_hash
        and failure.get("error", "").startswith(
            "W&B smoke evidence did not become available: RuntimeError: W&B missing video:"
        ),
        "Prior failure is not the reviewed video verifier failure",
    )
    from omegaconf import OmegaConf

    overrides = list(OmegaConf.load(folder / "0/.hydra/overrides.yaml"))
    expected_overrides = [
        "trainer.max_epochs=1",
        "trainer.min_epochs=1",
        "trainer.limit_train_batches=2",
        "trainer.check_val_every_n_epoch=1",
        "callbacks.model_checkpoint.every_n_epochs=1",
        "callbacks.moving_opscale_evaluation.viz_every_n_epochs=1",
        f"norm_stats.precomputed_norm_path={runs / 'norm/b24_op4/norm_stats'}",
        f"logger.wandb.id={expected_id}",
        "logger.wandb.group=fold_clothes_moving_opscale_smoke",
        "++logger.wandb.resume=never",
    ]
    need(
        overrides == expected_overrides,
        "First-phase overrides differ from the required smoke recipe",
    )
    status = json.loads((folder / "0/evaluation_status.json").read_text())
    need(
        status.get("global_step") == 2 and status.get("epoch") == 0,
        "First phase is not epoch one / step two",
    )
    need(len(status.get("videos", [])) == 13, "First-phase video evidence incomplete")
    for video in status["videos"]:
        artifact = Path(video["path"]).resolve()
        need(artifact.is_relative_to(folder), "First-phase video is outside its folder")
        need(
            str(artifact.relative_to(folder)) in files and artifact.stat().st_size > 0,
            "First-phase video was not authenticated",
        )
    return {
        "folder": folder,
        "run_id": expected_id,
        "job_id": source_job,
        "checkpoint": folder / checkpoint_name,
        "receipt": str(path),
        "receipt_sha256": checksum,
        "checkpoint_sha256": files[checkpoint_name],
    }


def smoke(args):
    root, runs, commit, design_hash = context(args)
    repo = training_repository(args)
    here = repo / "experiments/fold_clothes_moving_opscale"
    need(os.environ.get("SLURM_JOB_ID"), "GPU smoke must run inside a Slurm allocation")
    import torch

    need(
        torch.cuda.is_available() and torch.cuda.device_count() == 1,
        "Smoke requires exactly one visible GPU",
    )
    gpu = torch.cuda.get_device_name(0)
    need("H100" in gpu or "H200" in gpu, f"Unexpected GPU: {gpu}")
    need(
        not getattr(args, "reuse_first_phase", None)
        and not getattr(args, "reuse_first_phase_sha256", None),
        "Fresh strict-policy training smoke forbids legacy two-step first-phase reuse",
    )
    reused = None
    output = args.output or runs / "evaluation_smoke.json"
    smoke_root = runs / "smoke" / commit[:12] / f"attempt_{os.environ['SLURM_JOB_ID']}"
    report = {
        "schema_version": 1,
        "status": "running",
        "commit": commit,
        "design_sha256": design_hash,
        "gpu": gpu,
        "started_at": now(),
        "checks": {},
        "runs": [],
        "job_id": os.environ["SLURM_JOB_ID"],
        "normalization_reuse": normalization_reuse_identity(runs, commit),
        "training_tracking_policy": TRAINING_POLICY,
        "run_root": str(runs),
    }
    if getattr(args, "operational_controller", None):
        report["operational_controller"] = args.operational_controller
    write(output, report)
    try:
        for operators in [16, 4, 128]:
            cell = f"b24_op{operators}"
            norm_file = runs / "norm" / cell / "norm_complete.json"
            normalized = json.loads(norm_file.read_text())
            need(
                normalized.get("status") == "passed"
                and normalization_source_matches(runs, normalized, commit, repo)
                and normalized["design_sha256"] == design_hash,
                f"Missing matching normalization for {cell}",
            )
            need(
                normalized["stats_sha256"] == sha(normalized["stats_path"]),
                "Normalization checksum changed",
            )
            need(
                normalized.get("canonical_index_sha256")
                == sha(root / "canonical_zarr/_episode_index.json"),
                "Normalization canonical data changed",
            )
            need(
                normalized["split_sha256"]
                == sha(root / "splits" / f"fold_moving_{cell}.json"),
                "Normalization split changed",
            )
            split = json.loads(
                (root / "splits" / f"fold_moving_{cell}.json").read_text()
            )
            norm_proof = validate_training_norm(
                normalized,
                json.loads(Path(normalized["stats_path"]).read_text()),
                split["train"],
                commit,
            )
            folder = smoke_root / cell
            run_id = (
                f"fold_moving_smoke_{cell}_{commit[:10]}_{os.environ['SLURM_JOB_ID']}"
            )
            if reused and operators == 4:
                folder, run_id = reused["folder"], reused["run_id"]
            checkpoint = None
            first_step = None
            for epochs in [1, 2] if operators == 4 else [1]:
                phase_steps = (
                    DIAGNOSTIC_TRAIN_STEPS if operators == 16 else SMOKE_TRAIN_STEPS
                )
                start_step = (epochs - 1) * phase_steps
                training_path, audit_overrides = training_audit_overrides(
                    folder,
                    cell,
                    commit,
                    os.environ["SLURM_JOB_ID"],
                    normalized,
                    start_step,
                    phase_steps,
                )
                command = [
                    sys.executable,
                    "-u",
                    str(repo / "egomimic/trainHydra.py"),
                    "-m",
                    f"--config-dir={here / 'budget_24'}",
                    f"--config-name=op_{operators}",
                    "hydra/launcher=basic",
                    f"hydra.sweep.dir={folder}",
                    "hydra.sweep.subdir=0",
                    f"trainer.max_epochs={epochs}",
                    f"trainer.min_epochs={epochs}",
                    f"trainer.limit_train_batches={phase_steps}",
                    "trainer.check_val_every_n_epoch=1",
                    "callbacks.model_checkpoint.every_n_epochs=1",
                    "callbacks.moving_opscale_evaluation.viz_every_n_epochs=1",
                    f"norm_stats.precomputed_norm_path={Path(normalized['stats_path']).parent}",
                    f"logger.wandb.id={run_id}",
                    "logger.wandb.group=fold_clothes_moving_opscale_smoke",
                    "++logger.wandb.resume=allow"
                    if checkpoint
                    else "++logger.wandb.resume=never",
                ]
                command.extend(audit_overrides)
                if operators == 16:
                    command.extend(
                        [
                            "trainer.limit_val_batches=0",
                            "callbacks.moving_opscale_evaluation=null",
                        ]
                    )
                if checkpoint:
                    command.append(f"ckpt_path={checkpoint}")
                folder.mkdir(parents=True, exist_ok=True)
                reuse_this_phase = bool(reused and operators == 4 and epochs == 1)
                if not reuse_this_phase:
                    log_path = folder / f"epoch_{epochs}.log"
                    need(
                        not log_path.exists(),
                        "Smoke phase was already attempted; preserve its evidence",
                    )
                    with log_path.open("x") as stream:
                        result = subprocess.run(
                            command, cwd=repo, stdout=stream, stderr=subprocess.STDOUT
                        )
                    need(
                        result.returncode == 0,
                        f"{cell} smoke epoch {epochs} failed; see {folder}",
                    )
                candidates = (
                    [reused["checkpoint"]]
                    if reuse_this_phase
                    else list(folder.glob("**/last.ckpt"))
                )
                need(len(candidates) == 1, f"Expected one last checkpoint for {cell}")
                checkpoint = candidates[0]
                saved = inspect_checkpoint(checkpoint)
                need(
                    saved["step"] == epochs * phase_steps,
                    "Checkpoint step does not match requested optimizer updates",
                )
                if first_step is None:
                    first_step = saved["step"]
                else:
                    need(
                        saved["step"] > first_step,
                        "Checkpoint resume did not advance the step",
                    )
                    resume_log = (folder / f"epoch_{epochs}.log").read_text(
                        errors="replace"
                    )
                    need(
                        "Restored all states" in resume_log,
                        "Lightning did not confirm optimizer/scheduler restoration",
                    )
                    saved["resumed_from_step"] = first_step
                    saved["restored_all_states"] = True
                training_evidence = json.loads(training_path.read_text())
                training_summary = verify_training_evidence(
                    training_evidence,
                    commit=commit,
                    job_id=os.environ["SLURM_JOB_ID"],
                    cell=cell,
                    start_step=start_step,
                    steps=phase_steps,
                    norm_sha256=normalized["stats_sha256"],
                )
                training_proof = {
                    "path": str(training_path),
                    "sha256": sha(training_path),
                    "norm_stats_sha256": normalized["stats_sha256"],
                    **training_summary,
                }
                if operators == 16:
                    report["training_diagnostic"] = {
                        "operator_count": 16,
                        "checkpoint": saved,
                        "training_evidence": training_proof,
                        "normalization": norm_proof,
                        "config_sha256": sha(here / "budget_24/op_16.yaml"),
                    }
                    write(output, report)
                    continue
                status = json.loads((folder / "0/evaluation_status.json").read_text())
                metrics = list(status.get("metrics", {}).values())
                need(
                    metrics
                    and all(
                        type(value) in (int, float) and math.isfinite(value)
                        for value in metrics
                    ),
                    "Smoke local metrics are missing or nonfinite",
                )
                split = json.loads(
                    (root / "splits" / f"fold_moving_{cell}.json").read_text()
                )
                for cohort in ["seen_matched", "seen_perlevel", "unseen"]:
                    coverage = status["coverage"][cohort]
                    need(
                        coverage
                        == {
                            "expected": len(split[cohort]),
                            "observed": len(split[cohort]),
                        },
                        f"Smoke skipped episodes in {cohort}",
                    )
                    expected = set(split[f"video_{cohort}"])
                    actual = {
                        v["episode"] for v in status["videos"] if v["cohort"] == cohort
                    }
                    need(
                        expected == actual,
                        f"Smoke visualization selections disagree for {cohort}",
                    )
                need(
                    len(status["videos"]) == (13 if operators == 4 else 14),
                    "Smoke video count mismatch",
                )
                need(
                    status["global_step"] == saved["step"],
                    "Evaluation and checkpoint steps disagree",
                )
                tracking = verify_tracking_status(
                    status,
                    {
                        cohort: len(split[cohort])
                        for cohort in ("seen_matched", "seen_perlevel", "unseen")
                    },
                    split=split,
                    canonical_index=json.loads(
                        (root / "canonical_zarr/_episode_index.json").read_text()
                    ),
                )
                backend = verify_wandb(run_id, status, saved["step"])
                if reuse_this_phase:
                    saved["sha256"] = reused["checkpoint_sha256"]
                report["runs"].append(
                    {
                        "operator_count": operators,
                        "training_evidence": training_proof,
                        "normalization": norm_proof,
                        "norm_stats_producer_commit": normalized["commit"],
                        "epochs": epochs,
                        "checkpoint": saved,
                        "backend": backend,
                        "coverage": status["coverage"],
                        "evaluation_policy": EVALUATION_POLICY,
                        "tracking_coverage": tracking,
                        "config_sha256": sha(
                            here / "budget_24" / f"op_{operators}.yaml"
                        ),
                        "execution_job_id": reused["job_id"]
                        if reuse_this_phase
                        else os.environ["SLURM_JOB_ID"],
                        "verification_job_id": os.environ["SLURM_JOB_ID"],
                        "reused_phase_receipt": reused["receipt"]
                        if reuse_this_phase
                        else None,
                        "reused_phase_receipt_sha256": reused["receipt_sha256"]
                        if reuse_this_phase
                        else None,
                        "evaluation_status_sha256": sha(
                            folder / "0/evaluation_status.json"
                        ),
                    }
                )
                write(output, report)
                if reuse_this_phase:
                    # Preserve the independently reverified first-phase proof before
                    # the real epoch-two resume updates the live output directory.
                    write(
                        Path(reused["receipt"]).with_name(
                            "first_phase_reverified.json"
                        ),
                        report["runs"][-1],
                    )
        report.update(
            {
                "status": "passed",
                "finished_at": now(),
                "evaluation_policy": EVALUATION_POLICY,
                "checks": {
                    "finite_loss": True,
                    "checkpoint_resume": True,
                    "all_validation_episodes": True,
                    "wandb_metrics": True,
                    "video_counts": True,
                    "tracking_validity": True,
                    "training_validity": True,
                    "fresh_training_norms": True,
                    "representative_training": True,
                    "finite_gradients": True,
                },
            }
        )
        validate_smoke_training(report, runs, commit)
        write(output, report)
    except Exception as error:
        report.update({"status": "failed", "finished_at": now(), "error": str(error)})
        write(output, report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["norm", "smoke"])
    parser.add_argument("--private-dir", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--budget", type=int)
    parser.add_argument("--operators", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--training-repo", type=Path)
    parser.add_argument("--controller-commit")
    parser.add_argument("--reuse-first-phase", type=Path)
    parser.add_argument("--reuse-first-phase-sha256")
    args = parser.parse_args()
    (norm if args.mode == "norm" else smoke)(args)


if __name__ == "__main__":
    main()
