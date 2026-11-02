#!/usr/bin/env python3
"""Fail-closed Pace submission for the 36-cell folding sweep.

Default is a validated dry run. --submit is required to allocate GPUs. Each
native sbatch job runs trainHydra -m with Hydra's basic launcher, avoiding a
second Slurm allocation. Gate/report schema is defined by validate_gate below.
Submission attempts are recorded BEFORE sbatch; an uncertain attempt is never
silently retried. Use a new run root only for an intentionally new experiment.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import fcntl
import hashlib
import json
import netrc
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

BUDGETS = (24, 32, 48, 64, 72, 96)
OPERATORS = (4, 8, 16, 32, 64, 128)
MATRIX = {(b, n) for b in BUDGETS for n in OPERATORS}
GATE_CHECKS = (
    "dataset_integrity",
    "operator_consistency",
    "budget_consistency",
    "configuration_integrity",
    "reference_recipe",
    "evaluation_smoke",
)
DATA_CHECKS = (
    "transfer_complete",
    "selected_episode_objects",
    "required_arrays",
    "episode_index",
    "canonical_view",
)
REPO = Path(__file__).resolve().parents[2]


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        os.chmod(temporary, 0o600)
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def command(args, **kwargs):
    return subprocess.run(
        args, text=True, capture_output=True, check=True, **kwargs
    ).stdout.strip()


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def production_enabled(root):
    """Only an explicit run-root opt-in may release production training."""
    try:
        control = read_json(Path(root) / "production_control.json")
    except (OSError, ValueError):
        return False
    return (
        isinstance(control, dict)
        and control.get("allow_training") is True
        and control.get("scope") != "preflight_only"
    )


def require_production_enabled(root):
    need(
        production_enabled(root),
        "Production training is disabled; production_control.json must explicitly allow training",
    )


def submit_training_job(root, arguments, **kwargs):
    # Recheck immediately before every initial or continuation allocation.
    require_production_enabled(root)
    return command(arguments, **kwargs)


def validate_data_proof(private, ready, selected=None):
    """Bind canonical resolver metadata to the verified source frame receipts."""
    # Local import avoids the canonicalizer's dependency on this module's
    # atomic JSON/hash helpers while binding every downstream consumer to the
    # current correction algorithm, including its missing-hand semantics.
    from canonicalize import VERSION as CORRECTION_VERSION

    private = Path(private).resolve()
    need(
        ready.get("schema_version") == 1 and ready.get("status") == "passed",
        "Data readiness has not passed",
    )
    need(
        all(ready.get("checks", {}).get(key) is True for key in DATA_CHECKS),
        "Data readiness checks incomplete",
    )
    need(
        ready.get("design_sha256") == digest(private / "moving_opscale_design.json"),
        "Data proof design changed",
    )
    canonical = private / "canonical_zarr"
    need(
        Path(ready.get("canonical_root", "/nonexistent")).resolve()
        == canonical.resolve(),
        "Canonical resolver root differs",
    )
    index_path = canonical / "_episode_index.json"
    need(
        digest(index_path) == ready.get("canonical_index_sha256"),
        "Canonical episode index changed",
    )
    proof = ready.get("converted_frame_index", {})
    main = private / "inputs/main_episodes.csv"
    recovery = private / "recovery2/report.json"
    need(
        digest(main) == proof.get("main_episodes_sha256"),
        "Converted source frame index changed",
    )
    need(
        digest(recovery) == proof.get("recovery_report_sha256"),
        "Recovery frame receipt changed",
    )
    if selected is None:
        selected = set()
        for b, n in MATRIX:
            split = read_json(private / "splits" / f"fold_moving_b{b}_op{n}.json")
            for key in ("train", "seen_matched", "seen_perlevel", "unseen"):
                selected.update(split[key])
    index = read_json(index_path)
    need(set(index) == selected, "Canonical index differs from selected episode union")
    counts = {}
    with main.open(newline="") as stream:
        for row in csv.DictReader(stream):
            eid = row["episode_id"]
            if eid in selected:
                need(eid not in counts, "Duplicate selected source frame record")
                counts[eid] = int(row["frames"])
    for row in read_json(recovery)["episodes"]:
        eid = row["episode_id"]
        if eid in selected:
            need(
                eid not in counts and row.get("validated") is True,
                "Invalid/duplicate recovered frame record",
            )
            counts[eid] = int(row["frames"])
    need(
        set(counts) == selected and all(n >= 30 for n in counts.values()),
        "Missing/invalid source frame counts",
    )
    need(
        sum(counts.values()) == proof.get("selected_frames"),
        "Retained frame total differs",
    )
    for eid, meta in index.items():
        convention = meta.get("moving_opscale_canonicalization", {})
        need(
            convention.get("version") == CORRECTION_VERSION,
            "Canonical correction version is stale or missing",
        )
        need(
            meta.get("episode_id") == eid
            and int(meta.get("total_frames", -1)) == counts[eid],
            "Canonical episode frame count differs from conversion receipt",
        )
        need(
            convention.get("target_convention") == "post_fix"
            and convention.get("frames") == counts[eid]
            and convention.get("source_convention") in ("pre_fix", "post_fix"),
            "Canonical wrist convention is unverified",
        )
    return ready["canonical_index_sha256"]


def validate_gate(gate_path, ready_path, repo=REPO, *, freshness=True):
    gate, ready = read_json(gate_path), read_json(ready_path)
    for label, report, checks in [
        ("validation", gate, GATE_CHECKS),
        ("data readiness", ready, DATA_CHECKS),
    ]:
        need(
            report.get("schema_version") == 1 and report.get("status") == "passed",
            f"{label} report has not passed",
        )
        need(
            all(report.get("checks", {}).get(key) is True for key in checks),
            f"{label} has missing/failed checks: {checks}",
        )
    commit = command(["git", "-C", str(repo), "rev-parse", "HEAD"])
    need(
        re.fullmatch(r"[0-9a-f]{40}", commit) is not None
        and gate.get("commit") == commit,
        "Validation report does not match the full current commit",
    )
    need(
        not command(["git", "-C", str(repo), "status", "--porcelain"]),
        "Refusing launch from a dirty worktree",
    )
    private = Path(gate["private_root"]).resolve()
    design = private / "moving_opscale_design.json"
    need(design.is_file(), "Missing private experiment design")
    design_sha = digest(design)
    need(
        gate.get("design_sha256") == ready.get("design_sha256") == design_sha,
        "Design and readiness digests disagree",
    )
    validate_data_proof(private, ready)
    if freshness:
        checked = dt.datetime.fromisoformat(ready["checked_at"].replace("Z", "+00:00"))
        need(checked.tzinfo is not None, "Readiness timestamp needs a timezone")
        age = (dt.datetime.now(dt.timezone.utc) - checked).total_seconds()
        need(
            0 <= age <= 86400,
            "Data readiness verification must be within the past 24 hours",
        )
    entries = gate.get("configurations", [])
    need(len(entries) == 36, "Gate must validate all 36 configurations")
    indexed = {(int(e["budget_hours"]), int(e["operator_count"])): e for e in entries}
    need(
        set(indexed) == MATRIX,
        "Gate must contain each requested budget/operator pair once",
    )
    for (budget, count), entry in indexed.items():
        config = (
            repo
            / "experiments/fold_clothes_moving_opscale"
            / f"budget_{budget}/op_{count}.yaml"
        )
        split = private / "splits" / f"fold_moving_b{budget}_op{count}.json"
        need(
            config.is_file() and digest(config) == entry["config_sha256"],
            f"Changed/missing config: {config}",
        )
        need(
            split.is_file() and digest(split) == entry["split_sha256"],
            f"Changed/missing private split: {split.name}",
        )
    return gate, ready, indexed


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gate", required=True, type=Path)
    p.add_argument("--readiness", required=True, type=Path)
    p.add_argument("--run-root", required=True, type=Path)
    p.add_argument(
        "--python",
        required=True,
        type=Path,
        help="Existing EgoVerse environment Python",
    )
    p.add_argument("--gpu", choices=("h100", "h200"), default="h200")
    p.add_argument("--submit", action="store_true")
    p.add_argument(
        "--cells",
        nargs="*",
        help="Optional b24_op4-style subset; all 36 still must pass the gate",
    )
    return p


def verified_norm(root, budget, count, commit, design_sha, entry):
    report = read_json(
        Path(root) / "norm" / f"b{budget}_op{count}" / "norm_complete.json"
    )
    need(
        report.get("status") == "passed"
        and report.get("commit") == commit
        and report.get("design_sha256") == design_sha
        and report.get("budget_hours") == budget
        and report.get("operator_count") == count
        and report.get("config_sha256") == entry["config_sha256"]
        and report.get("split_sha256") == entry["split_sha256"]
        and report.get("seed") == 42026
        and report.get("sample_frac") == 1.0
        and report.get("max_samples") == 200000
        and report.get("pool_horizon") is False
        and report.get("canonical_index_sha256")
        == digest(Path(root) / "private/canonical_zarr/_episode_index.json"),
        f"Normalization provenance mismatch for b{budget}_op{count}",
    )
    path = Path(report["stats_path"]).resolve()
    need(
        path.is_relative_to(Path(root).resolve() / "norm"),
        "Norm statistics escape the sweep cache",
    )
    need(
        digest(path) == report["stats_sha256"],
        "Norm statistics changed after precompute",
    )
    return path, report["stats_sha256"]


def wandb_auth_available(environ=None, netrc_path=None):
    """Recognize supported environment/netrc auth without exposing credentials."""
    environ = os.environ if environ is None else environ
    if environ.get("WANDB_API_KEY"):
        return True
    hostname = urlparse(environ.get("WANDB_BASE_URL", "https://api.wandb.ai")).hostname
    if not hostname:
        return False
    try:
        credentials = netrc.netrc(
            str(netrc_path) if netrc_path else None
        ).authenticators(hostname)
        return bool(credentials and credentials[2])
    except (OSError, netrc.NetrcParseError):
        return False


def verify_backend_completion(run, output, step, commit, verifier=None):
    """Require the final metrics/video artifacts to exist on the W&B backend."""
    evidence_path = Path(output) / "0/evaluation_complete.json"
    evidence = read_json(evidence_path)
    need(
        evidence.get("status") == "passed"
        and evidence.get("wandb_id") == run["wandb_id"]
        and int(evidence.get("step", -1)) == step,
        "Final evaluation identity/step mismatch",
    )
    expected_videos = 13 if run["operator_count"] == 4 else 14
    need(
        len(evidence.get("videos", [])) == expected_videos, "Final video count mismatch"
    )
    if verifier is None:
        from run_preflight import verify_wandb

        verifier = verify_wandb
    backend = verifier(run["wandb_id"], evidence, step)
    need(
        backend.get("run_id") == run["wandb_id"]
        and backend.get("state") == "finished"
        and int(backend.get("backend_step", -1)) >= step - 1
        and backend.get("verified_videos") == expected_videos,
        "W&B completion proof mismatch",
    )
    return {
        "status": "passed",
        "verified_at": now(),
        "commit": commit,
        "wandb_id": run["wandb_id"],
        "global_step": step,
        "evaluation_sha256": digest(evidence_path),
        "backend": backend,
    }


def submit(args):
    if args.submit:
        require_production_enabled(args.run_root)
    gate, ready, indexed = validate_gate(args.gate, args.readiness)
    selected = sorted(MATRIX)
    if args.cells:
        selected = []
        for cell in args.cells:
            match = re.fullmatch(r"b(\d+)_op(\d+)", cell)
            need(match is not None, f"Invalid cell {cell}")
            key = tuple(map(int, match.groups()))
            need(key in MATRIX, f"Unknown cell {cell}")
            selected.append(key)
        need(len(set(selected)) == len(selected), "Duplicate cells requested")
    python = args.python.absolute()
    need(python.is_file(), "Python executable does not exist")
    root = args.run_root.resolve()
    norms = {
        (b, n): verified_norm(
            root, b, n, gate["commit"], gate["design_sha256"], indexed[(b, n)]
        )
        for b, n in selected
    }
    identity = hashlib.sha256(
        (gate["commit"] + gate["design_sha256"] + str(root)).encode()
    ).hexdigest()[:12]
    plan = {
        "commit": gate["commit"],
        "run_root": str(root),
        "cells": len(selected),
        "partition": f"gpu-{args.gpu}",
        "gres": f"gpu:{args.gpu}:1",
        "account": "gts-dxu345-rl2",
        "qos": "inferno",
        "cpus": 8,
        "memory": "128G",
        "time": "3-00:00:00",
        "effective_batch_size": 64,
        "max_steps": 400000,
    }
    print(json.dumps(plan, indent=2), flush=True)
    if not args.submit:
        print("Validated dry run; no jobs submitted.")
        return
    for key in ("R2_ENDPOINT_URL", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY"):
        need(
            bool(os.environ.get(key)),
            f"{key} must be configured in the submission environment",
        )
    need(
        wandb_auth_available(),
        "W&B authentication must be configured through its environment or netrc entry",
    )
    need(
        command(["git", "-C", str(REPO), "rev-parse", "HEAD"]) == gate["commit"],
        "Commit changed",
    )
    root.mkdir(parents=True, exist_ok=True)
    os.chmod(root, 0o700)
    lock = (root / "submission.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    registry_path = root / "registry.json"
    registry = (
        read_json(registry_path)
        if registry_path.exists()
        else {
            "schema_version": 1,
            "created_at": now(),
            "commit": gate["commit"],
            "target": "pace",
            "seed": 42026,
            "overlay": "native-single-gpu-hydra-basic",
            "approval": "User explicitly requested execution on H100/H200 and run monitors",
            "design_sha256": gate["design_sha256"],
            "repository": str(REPO),
            "gate": str(args.gate.resolve()),
            "readiness": str(args.readiness.resolve()),
            "run_root": str(root),
            "automatic_continuation": {
                "states": ["TIMEOUT", "PREEMPTED"],
                "max_attempts": 3,
            },
            "runs": {},
        }
    )
    need(
        registry["commit"] == gate["commit"]
        and registry["design_sha256"] == gate["design_sha256"],
        "Run root belongs to a different commit/design",
    )
    # A monitor takes its own lock: repeated launches cannot create duplicate monitors.
    with (root / "monitor.log").open("a") as stream:
        subprocess.Popen(
            [
                str(python),
                str(Path(__file__).with_name("monitor.py")),
                "--run-root",
                str(root),
            ],
            stdout=stream,
            stderr=stream,
            start_new_session=True,
            close_fds=True,
        )
    for budget, count in selected:
        cell = f"b{budget}_op{count}"
        if cell in registry["runs"]:
            print(f"{cell}: existing submission record; skipped", flush=True)
            continue
        run_dir = root / cell
        run_dir.mkdir(exist_ok=True)
        output_dir = (
            REPO
            / "logs/fold_clothes_moving_opscale/finetuned"
            / f"{cell}_v1_{identity}"
        )
        wandb_id = f"foldmb-{identity}-{cell}"
        record = dict(
            indexed[(budget, count)],
            cell=cell,
            wandb_id=wandb_id,
            directory=str(run_dir),
            output_dir=str(output_dir),
            state="SUBMITTING",
            submitted_at=now(),
            gpu=args.gpu,
            expected_steps=400000,
            norm_stats_path=str(norms[(budget, count)][0]),
            norm_stats_sha256=norms[(budget, count)][1],
        )
        registry["runs"][cell] = record
        write_json(registry_path, registry)
        script = run_dir / "job.sbatch"
        script.write_text(
            "#!/bin/bash\nset -euo pipefail\numask 077\n"
            + "export PYTHONDONTWRITEBYTECODE=1\n"
            + "export PYTHONPATH="
            + shlex.quote(str(REPO))
            + ':"${PYTHONPATH:-}"\n'
            + "export EGOVERSE_ROOT="
            + shlex.quote(str(REPO))
            + "\n"
            + "export EGOVERSE_MECKA_PRIVATE_DIR="
            + shlex.quote(gate["private_root"])
            + "\n"
            + "export PATH="
            + shlex.quote(str(Path.home() / ".local/bin"))
            + ':"$PATH"\n'
            + "cd "
            + shlex.quote(str(REPO))
            + "\n"
            + "exec "
            + shlex.join(
                [
                    str(python),
                    str(Path(__file__).resolve()),
                    "_execute",
                    str(registry_path),
                    cell,
                ]
            )
            + "\n"
        )
        os.chmod(script, 0o700)
        sbatch = [
            "sbatch",
            "--parsable",
            f"--job-name=fold-{cell}-{identity[:6]}",
            f"--partition=gpu-{args.gpu}",
            "--account=gts-dxu345-rl2",
            "--qos=inferno",
            f"--gres=gpu:{args.gpu}:1",
            "--nodes=1",
            "--ntasks-per-node=1",
            "--cpus-per-task=8",
            "--mem=128G",
            "--time=3-00:00:00",
            "--no-requeue",
            f"--output={run_dir}/slurm-%j.out",
            f"--error={run_dir}/slurm-%j.err",
            str(script),
        ]
        record["sbatch_command"] = sbatch
        write_json(registry_path, registry)
        try:
            raw = submit_training_job(root, sbatch, timeout=90)
            job = raw.split(";")[0]
            need(
                job.isdigit(),
                "Unrecognized sbatch response; reconcile manually before another attempt",
            )
            record.update(
                job_id=job,
                state="SUBMITTED",
                attempts=[{"job_id": job, "submitted_at": now(), "resume_step": 0}],
            )
            print(f"{cell}: submitted {job}", flush=True)
        except Exception as error:
            record.update(state="SUBMISSION_UNCERTAIN", error_type=type(error).__name__)
            write_json(registry_path, registry)
            raise RuntimeError(
                f"{cell}: submission uncertain; inspect Slurm before taking further action"
            ) from None
        write_json(registry_path, registry)


def execute(registry_path, cell):
    require_production_enabled(Path(registry_path).parent)
    registry = read_json(registry_path)
    # The transfer proof is pinned by content, but may be older after time in queue.
    gate, _, _ = validate_gate(registry["gate"], registry["readiness"], freshness=False)
    need(
        gate["commit"] == registry["commit"]
        and gate["design_sha256"] == registry["design_sha256"],
        "Gate changed after submission",
    )
    run = registry["runs"][cell]
    stats_path, stats_sha = verified_norm(
        registry["run_root"],
        run["budget_hours"],
        run["operator_count"],
        registry["commit"],
        registry["design_sha256"],
        run,
    )
    need(
        stats_sha == run["norm_stats_sha256"],
        "Normalization cache changed since submission",
    )
    run_dir = Path(run["directory"])
    config_dir = (
        REPO
        / "experiments/fold_clothes_moving_opscale"
        / f"budget_{run['budget_hours']}"
    )
    output = Path(run["output_dir"])
    write_json(
        run_dir / "worker.json",
        {
            "state": "RUNNING",
            "started_at": now(),
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "wandb_id": run["wandb_id"],
        },
    )
    overlay = Path(registry["run_root"]) / "python-deps"
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join(
            [str(REPO), str(overlay), os.environ.get("PYTHONPATH", "")]
        ),
        EGOVERSE_MECKA_PRIVATE_DIR=gate["private_root"],
        EGOVERSE_ROOT=str(REPO),
    )
    args = [
        sys.executable,
        "-u",
        str(REPO / "egomimic/trainHydra.py"),
        "-m",
        f"--config-dir={config_dir}",
        f"--config-name=op_{run['operator_count']}",
        "hydra/launcher=basic",
        "launch_params.nodes=1",
        "launch_params.gpus_per_node=1",
        f"norm_stats.precomputed_norm_path={stats_path.parent}",
        f"logger.wandb.id={run['wandb_id']}",
        f"hydra.sweep.dir={output}",
        "hydra.sweep.subdir=0",
    ]
    if run.get("resume_checkpoint"):
        checkpoint = Path(run["resume_checkpoint"])
        need(checkpoint.is_file(), "Resume checkpoint disappeared")
        need(
            checkpoint.stat().st_size == run["resume_checkpoint_size"],
            "Resume checkpoint changed",
        )
        need(
            checkpoint.stat().st_mtime_ns == run["resume_checkpoint_mtime_ns"],
            "Resume checkpoint changed",
        )
        args.extend([f"ckpt_path={checkpoint}", "++logger.wandb.resume=allow"])
    with (run_dir / "training.log").open("a") as stream:
        result = subprocess.run(
            args, cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT
        )
    write_json(
        run_dir / "worker.json",
        {
            "state": "EXITED",
            "finished_at": now(),
            "returncode": result.returncode,
            "wandb_id": run["wandb_id"],
        },
    )
    if result.returncode:
        return result.returncode
    checkpoints = list(output.glob("**/last.ckpt"))
    need(
        checkpoints,
        "Hydra exited successfully but no last.ckpt exists; completion unverified",
    )
    import torch

    checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
    # Only read the checkpoint generated by this trusted local training invocation.
    value = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    step = int(value.get("global_step", -1))
    del value
    need(step >= 400000, f"Training exited at step {step}, before required 400000")
    proof = verify_backend_completion(run, output, step, registry["commit"])
    proof_path = run_dir / "wandb_complete.json"
    write_json(proof_path, proof)
    write_json(
        run_dir / "training_complete.json",
        {
            "status": "passed",
            "finished_at": now(),
            "wandb_id": run["wandb_id"],
            "commit": registry["commit"],
            "global_step": step,
            "checkpoint": str(checkpoint),
            "backend_verified": True,
            "wandb_proof_sha256": digest(proof_path),
        },
    )
    return 0


def continue_timed_out_run(registry_path, cell, observed_job, observed_state):
    """Resume only an exhausted/preempted allocation with full checkpoint state.

    Called by the monitor; never retries failed training, data errors, or NaNs.
    At most three allocations total, retaining W&B identity and Hydra output.
    """
    if observed_state not in ("TIMEOUT", "PREEMPTED"):
        return False
    path = Path(registry_path)
    if not production_enabled(path.parent):
        return False
    with (path.parent / "submission.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        registry = read_json(path)
        policy = registry.get("automatic_continuation", {})
        run = registry["runs"][cell]
        if (
            observed_state not in policy.get("states", [])
            or run.get("job_id") != observed_job
        ):
            return False
        if run.get("state") in (
            "SUBMITTING",
            "SUBMISSION_UNCERTAIN",
            "CONTINUATION_BLOCKED",
        ):
            return False
        attempts = run.get("attempts", [{"job_id": observed_job}])
        if len(attempts) >= min(3, int(policy.get("max_attempts", 0))):
            return False
        try:
            gate, _, _ = validate_gate(
                registry["gate"], registry["readiness"], freshness=False
            )
            need(
                gate["commit"] == registry["commit"]
                and gate["design_sha256"] == registry["design_sha256"],
                "Continuation provenance differs",
            )
            # Read bounded tail and reject recognizable scientific/data failures.
            log = Path(run["directory"]) / "training.log"
            if log.exists():
                with log.open("rb") as stream:
                    stream.seek(max(0, log.stat().st_size - 262144))
                    tail = stream.read().decode(errors="replace")
                need(
                    not re.search(
                        r"\bNaN\b|nonfinite|non-finite|CUDA out of memory|FileNotFoundError|DataLoader worker.*(?:killed|exited)",
                        tail,
                        re.I,
                    ),
                    "Data/numerical failure present; automatic continuation forbidden",
                )
            output = Path(run["output_dir"]).resolve()
            candidates = [
                p.resolve() for p in output.glob("**/last.ckpt") if p.is_file()
            ]
            need(candidates, "No last.ckpt available for full-state continuation")
            checkpoint = max(candidates, key=lambda p: p.stat().st_mtime)
            need(
                checkpoint.is_relative_to(output), "Checkpoint escapes this run output"
            )
            import torch

            value = torch.load(
                checkpoint, map_location="cpu", weights_only=False, mmap=True
            )
            step = int(value.get("global_step", -1))
            need(
                0 < step < 400000
                and bool(value.get("state_dict"))
                and bool(value.get("optimizer_states"))
                and bool(value.get("lr_schedulers"))
                and bool(value.get("loops")),
                "Checkpoint does not contain resumable model/optimizer/scheduler/loop state",
            )
            del value
            need(
                step > int(attempts[-1].get("resume_step", -1)),
                "Prior allocation made no checkpoint progress",
            )
            stat = checkpoint.stat()
            run.update(
                state="SUBMITTING",
                resume_checkpoint=str(checkpoint),
                resume_step=step,
                resume_checkpoint_size=stat.st_size,
                resume_checkpoint_mtime_ns=stat.st_mtime_ns,
            )
            attempts[-1]["terminal_state"] = observed_state
            run["attempts"] = attempts
            write_json(path, registry)
        except Exception as error:
            run.update(
                state="CONTINUATION_BLOCKED",
                continuation_error_type=type(error).__name__,
            )
            write_json(path, registry)
            return False
        try:
            job = submit_training_job(
                path.parent, run["sbatch_command"], timeout=90
            ).split(";")[0]
            need(job.isdigit(), "Unrecognized continuation submission response")
        except Exception:
            run["state"] = "SUBMISSION_UNCERTAIN"
            write_json(path, registry)
            return False
        attempts.append({"job_id": job, "submitted_at": now(), "resume_step": step})
        run.update(job_id=job, state="SUBMITTED", attempts=attempts)
        write_json(path, registry)
        return True


if __name__ == "__main__":
    try:
        if len(sys.argv) > 1 and sys.argv[1] == "_execute":
            sys.exit(execute(sys.argv[2], sys.argv[3]))
        submit(parser().parse_args())
    except Exception as error:
        print(f"BLOCKED: {error}", file=sys.stderr)
        sys.exit(2)
