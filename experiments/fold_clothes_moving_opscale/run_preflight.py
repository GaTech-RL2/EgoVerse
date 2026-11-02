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

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


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
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()
    need(len(args.commit) == 40 and commit == args.commit, "Preflight commit mismatch")
    need(
        not subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO, text=True
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
    os.environ["EGOVERSE_ROOT"] = str(REPO)
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [str(REPO), str(runs / "python-deps"), os.environ.get("PYTHONPATH", "")]
    )
    sys.path[:0] = [str(REPO), str(runs / "python-deps")]
    return root, runs, commit, design_hash


def norm(args):
    root, runs, commit, design_hash = context(args)
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
            HERE / f"budget_{args.budget}" / f"op_{args.operators}.yaml"
        ),
        "split_sha256": sha(
            root / "splits" / f"fold_moving_b{args.budget}_op{args.operators}.json"
        ),
    }
    if report.exists():
        previous = json.loads(report.read_text())
        if (
            previous.get("status") == "passed"
            and all(previous.get(k) == v for k, v in identity.items())
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
        str(REPO / "egomimic/scripts/precompute_norm_stats.py"),
        "--config-name",
        f"op_{args.operators}",
        "--config-dir",
        str(HERE / f"budget_{args.budget}"),
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
        f'hydra.searchpath=[file://{REPO / "egomimic/hydra_configs"}]',
        "seed=42026",
    ]
    try:
        runpy.run_path(
            str(REPO / "egomimic/scripts/precompute_norm_stats.py"), run_name="__main__"
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


def verify_wandb(run_id, status, expected_step):
    import wandb

    api = wandb.Api(timeout=45)
    deadline = time.monotonic() + 1200
    last_error = ""
    while time.monotonic() < deadline:
        try:
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
            for video in status["videos"]:
                record = summary.get(video["wandb_key"])
                need(
                    isinstance(record, dict) and record.get("path"),
                    f'W&B missing video: {video["wandb_key"]}',
                )
                need(
                    run.file(record["path"]).size > 0,
                    f'W&B video not uploaded: {video["wandb_key"]}',
                )
            return {
                "run_id": run_id,
                "url": run.url,
                "state": run.state,
                "finite_losses": losses,
                "backend_step": summary["trainer/global_step"],
                "verified_videos": len(status["videos"]),
            }
        except Exception as error:
            last_error = f"{type(error).__name__}: {error}"
            time.sleep(20)
    raise RuntimeError(f"W&B smoke evidence did not become available: {last_error}")


def smoke(args):
    root, runs, commit, design_hash = context(args)
    need(os.environ.get("SLURM_JOB_ID"), "GPU smoke must run inside a Slurm allocation")
    import torch

    need(
        torch.cuda.is_available() and torch.cuda.device_count() == 1,
        "Smoke requires exactly one visible GPU",
    )
    gpu = torch.cuda.get_device_name(0)
    need("H100" in gpu or "H200" in gpu, f"Unexpected GPU: {gpu}")
    output = args.output or runs / "evaluation_smoke.json"
    smoke_root = runs / "smoke" / commit[:12]
    report = {
        "schema_version": 1,
        "status": "running",
        "commit": commit,
        "design_sha256": design_hash,
        "gpu": gpu,
        "started_at": now(),
        "checks": {},
        "runs": [],
    }
    write(output, report)
    try:
        for operators in [4, 128]:
            cell = f"b24_op{operators}"
            norm_file = runs / "norm" / cell / "norm_complete.json"
            normalized = json.loads(norm_file.read_text())
            need(
                normalized.get("status") == "passed"
                and normalized["commit"] == commit
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
                f'fold_moving_smoke_{cell}_{commit[:10]}_{os.environ["SLURM_JOB_ID"]}'
            )
            checkpoint = None
            first_step = None
            for epochs in [1, 2] if operators == 4 else [1]:
                command = [
                    sys.executable,
                    "-u",
                    str(REPO / "egomimic/trainHydra.py"),
                    "-m",
                    f'--config-dir={HERE / "budget_24"}',
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
                    f'norm_stats.precomputed_norm_path={Path(normalized["stats_path"]).parent}',
                    f"logger.wandb.id={run_id}",
                    "logger.wandb.group=fold_clothes_moving_opscale_smoke",
                    "++logger.wandb.resume=allow"
                    if checkpoint
                    else "++logger.wandb.resume=never",
                ]
                if checkpoint:
                    command.append(f"ckpt_path={checkpoint}")
                folder.mkdir(parents=True, exist_ok=True)
                with (folder / f"epoch_{epochs}.log").open("w") as stream:
                    result = subprocess.run(
                        command, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT
                    )
                need(
                    result.returncode == 0,
                    f"{cell} smoke epoch {epochs} failed; see {folder}",
                )
                candidates = list(folder.glob("**/last.ckpt"))
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
                backend = verify_wandb(run_id, status, saved["step"])
                report["runs"].append(
                    {
                        "operator_count": operators,
                        "epochs": epochs,
                        "checkpoint": saved,
                        "backend": backend,
                        "coverage": status["coverage"],
                        "config_sha256": sha(
                            HERE / "budget_24" / f"op_{operators}.yaml"
                        ),
                    }
                )
                write(output, report)
        report.update(
            {
                "status": "passed",
                "finished_at": now(),
                "checks": {
                    "finite_loss": True,
                    "checkpoint_resume": True,
                    "all_validation_episodes": True,
                    "wandb_metrics": True,
                    "video_counts": True,
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
    args = parser.parse_args()
    (norm if args.mode == "norm" else smoke)(args)


if __name__ == "__main__":
    main()
