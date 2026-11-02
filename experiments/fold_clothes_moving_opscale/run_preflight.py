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
    identity = {
        "commit": commit,
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
        json.loads(stats.read_text())
        write(
            report,
            {
                **identity,
                "status": "passed",
                "created_at": now(),
                "stats_path": str(stats),
                "stats_sha256": sha(stats),
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
    reused = reusable_first_phase(args, runs, commit, design_hash)
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
    }
    if getattr(args, "operational_controller", None):
        report["operational_controller"] = args.operational_controller
    write(output, report)
    try:
        for operators in [4, 128]:
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
            folder = smoke_root / cell
            run_id = (
                f"fold_moving_smoke_{cell}_{commit[:10]}_{os.environ['SLURM_JOB_ID']}"
            )
            if reused and operators == 4:
                folder, run_id = reused["folder"], reused["run_id"]
            checkpoint = None
            first_step = None
            for epochs in [1, 2] if operators == 4 else [1]:
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
                    "trainer.limit_train_batches=2",
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
                    saved["step"] == epochs * 2,
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
                },
            }
        )
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
