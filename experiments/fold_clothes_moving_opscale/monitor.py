#!/usr/bin/env python3
"""Durable read-only sweep monitor; writes heartbeat and change-only events.

May start before any jobs exist. Only explicitly configured TIMEOUT/PREEMPTED continuation is permitted, using verified full-state checkpoints; no failed-run retry.
Slurm COMPLETED alone never establishes experiment completion. A verified final
evaluation report must accompany the launcher's training_complete.json.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import subprocess
import time
from pathlib import Path

from launch_pace import (
    MATRIX,
    continue_timed_out_run,
    digest,
    now,
    read_json,
    write_json,
)

TERMINAL = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
    "PREEMPTED",
    "BOOT_FAIL",
    "DEADLINE",
    "REVOKED",
}
ERROR = re.compile(
    r"Traceback \(most recent call last\)|CUDA out of memory|"
    r"(?:RuntimeError|FileNotFoundError|ModuleNotFoundError|OutOfMemoryError):|"
    r"NCCL.*(?:error|timed out)",
    re.I,
)


def query(command):
    result = subprocess.run(command, capture_output=True, text=True, timeout=45)
    if result.returncode:
        raise RuntimeError(f"{command[0]} returned {result.returncode}")
    return result.stdout


def state_tables(ids):
    if not ids:
        return {}, None
    try:
        joined = ",".join(ids)
        table = {}
        for line in query(
            [
                "sacct",
                "-n",
                "-X",
                "-P",
                "-j",
                joined,
                "--format=JobIDRaw,State,ExitCode,Elapsed,NodeList",
            ]
        ).splitlines():
            parts = line.split("|")
            if len(parts) >= 5 and parts[0] in ids:
                table[parts[0]] = dict(
                    zip(("job_id", "state", "exit_code", "elapsed", "nodes"), parts[:5])
                )
        for line in query(
            ["squeue", "-h", "-j", joined, "-o", "%i|%T|%R|%N|%M"]
        ).splitlines():
            parts = line.split("|")
            if len(parts) == 5 and parts[0] in ids:
                table[parts[0]] = dict(
                    zip(("job_id", "state", "reason", "nodes", "elapsed"), parts)
                )
        return table, None
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        return {}, type(error).__name__


def safe_report(path):
    try:
        return read_json(path)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def verify_evaluation(report, run, training):
    if not report or not training:
        return False
    if report.get("status") != "passed" or report.get("wandb_id") != run["wandb_id"]:
        return False
    if int(report.get("step", -1)) < int(training.get("global_step", 400000)):
        return False
    # The exact unseen count comes from the validated split's recorded count.
    unseen_count = run.get("unseen_episode_count")
    if not isinstance(unseen_count, int) or unseen_count < 1:
        return False
    targets = {
        "seen_matched": (40, 4),
        "seen_perlevel": (
            3 * run["operator_count"],
            4 if run["operator_count"] == 4 else 5,
        ),
        "unseen": (unseen_count, 5),
    }
    for key, (episodes, videos) in targets.items():
        item = report.get("sets", {}).get(key, {})
        if (
            item.get("episodes") != episodes
            or item.get("videos") != videos
            or item.get("metrics_logged") is not True
        ):
            return False
    return True


def verify_backend_report(root, run, registry, training):
    proof_path = root / "wandb_complete.json"
    proof = safe_report(proof_path)
    if not proof or not training or training.get("backend_verified") is not True:
        return False
    expected = 13 if run["operator_count"] == 4 else 14
    evidence_path = Path(run["output_dir"]) / "0/evaluation_complete.json"
    return bool(
        evidence_path.is_file()
        and proof.get("status") == "passed"
        and proof.get("commit") == registry["commit"]
        and proof.get("wandb_id") == run["wandb_id"]
        and proof.get("global_step") == training.get("global_step")
        and proof.get("evaluation_sha256") == digest(evidence_path)
        and training.get("wandb_proof_sha256") == digest(proof_path)
        and proof.get("backend", {}).get("run_id") == run["wandb_id"]
        and proof.get("backend", {}).get("state") == "finished"
        and int(proof.get("backend", {}).get("backend_step", -1))
        >= training["global_step"] - 1
        and proof.get("backend", {}).get("verified_videos") == expected
    )


def inspect_run(run, registry, scheduler):
    root = Path(run["directory"])
    result = {
        k: run.get(k)
        for k in ("cell", "job_id", "wandb_id", "budget_hours", "operator_count")
    }
    result["scheduler"] = scheduler or {"state": "UNKNOWN"}
    current = result["scheduler"]["state"].split()[0].rstrip("+")
    result["state"] = current
    log = root / "training.log"
    result["error_markers"] = []
    if log.exists():
        stat = log.stat()
        result["log_age_seconds"] = round(time.time() - stat.st_mtime)
        result["log_bytes"] = stat.st_size
        with log.open("rb") as stream:
            stream.seek(max(0, stat.st_size - 131072))
            text = stream.read().decode(errors="replace")
        # Store marker labels only: log lines may contain signed URLs or credentials.
        result["error_markers"] = sorted(
            {
                "NCCL_error" if m.upper().startswith("NCCL") else m
                for m in ERROR.findall(text)
            }
        )
        steps = re.findall(r"(?:global_step|step)[=: ]+(\d+)", text)
        if steps:
            result["observed_step"] = max(map(int, steps))
        result["log_stale"] = current == "RUNNING" and result["log_age_seconds"] > 1800
    progress_path = Path(run["output_dir"]) / "0/evaluation_progress.json"
    progress = safe_report(progress_path)
    if progress:
        result["evaluation_progress"] = {
            k: progress.get(k)
            for k in (
                "state",
                "global_step",
                "completed_videos",
                "expected_videos",
                "cohort",
                "updated_at",
            )
        }
        result["evaluation_progress_age_seconds"] = round(
            time.time() - progress_path.stat().st_mtime
        )
        if result["evaluation_progress_age_seconds"] < 1800:
            result["log_stale"] = False
    training = safe_report(root / "training_complete.json")
    valid_training = bool(
        training
        and training.get("status") == "passed"
        and training.get("wandb_id") == run["wandb_id"]
        and training.get("commit") == registry["commit"]
        and int(training.get("global_step", -1)) >= 400000
        and Path(training.get("checkpoint", "/nonexistent")).is_file()
    )
    if current == "COMPLETED":
        result["state"] = "SLURM_COMPLETED_UNVERIFIED"
        if valid_training:
            result["global_step"] = training["global_step"]
            result["state"] = "TRAINING_COMPLETE_EVALUATION_UNVERIFIED"
            if verify_evaluation(
                safe_report(root / "evaluation_complete.json")
                or safe_report(Path(run["output_dir"]) / "0/evaluation_complete.json"),
                run,
                training,
            ) and verify_backend_report(root, run, registry, training):
                result["state"] = "COMPLETE"
    if not run.get("job_id"):
        result["state"] = run.get("state", "NOT_SUBMITTED")
    result["terminal"] = current in TERMINAL or result["state"] == "COMPLETE"
    return result


def verify_preflight(root, pipeline):
    """Preflight completion is separate from verified production-run completion."""
    report = safe_report(root / "preflight_complete.json") or {}
    expected = {f"b{b}_op{n}" for b, n in MATRIX}
    if not (
        report.get("status") == "passed"
        and report.get("scope") == "preflight_only"
        and report.get("commit") == pipeline.get("commit")
        and set(report.get("normalization_sha256", {})) == expected
    ):
        return False
    try:
        return (
            digest(root / "readiness.json") == report.get("readiness_sha256")
            and digest(root / "gate.json") == report.get("gate_sha256")
            and digest(root / "evaluation_smoke.json") == report.get("smoke_sha256")
            and all(
                digest(root / "norm" / cell / "norm_complete.json") == checksum
                for cell, checksum in report["normalization_sha256"].items()
            )
        )
    except OSError:
        return False


def snapshot(root):
    registry = safe_report(root / "registry.json")
    result = {
        "schema_version": 1,
        "updated_at": now(),
        "heartbeat_epoch": time.time(),
        "monitor_pid": os.getpid(),
        "state": "WAITING_FOR_LAUNCH",
        "gate": "pending",
        "readiness": "pending",
        "runs": {},
    }
    if not registry:
        result["reason"] = "No training submission registry; preflight is pending"
        gate = safe_report(root / "gate.json")
        result["gate"] = gate.get("status", "pending") if gate else "pending"
        pipeline = safe_report(root / "pipeline_status.json")
        if pipeline:
            result["pipeline"] = {
                k: pipeline.get(k) for k in ("state", "phase", "updated_at", "error")
            }
        staging = safe_report(root / "staging.json")
        if staging and staging.get("job_id"):
            ids = [staging["job_id"]] + staging.get("recovery_job_ids", [])
            table, error = state_tables(ids)
            result["staging"] = table.get(staging["job_id"], {"state": "UNKNOWN"})
            result["recovery_jobs"] = {
                job: table.get(job, {"state": "UNKNOWN"})
                for job in staging.get("recovery_job_ids", [])
            }
            inventory = root / "destination_inventory.json"
            result["inventory_available"] = inventory.is_file()
            transfer = root / "staging-transfer.log"
            if transfer.exists():
                result["transfer_log_bytes"] = transfer.stat().st_size
                result["transfer_log_age_seconds"] = round(
                    time.time() - transfer.stat().st_mtime
                )
            result["scheduler_error"] = error
            result["state"] = "WAITING_FOR_DATA"
        readiness = safe_report(root / "readiness.json")
        if readiness:
            result["readiness"] = readiness.get("status", "pending")
            result["verified_episodes"] = readiness.get("verified_episodes", 0)
        if pipeline:
            result["phase"] = pipeline.get("phase")
            result["state"] = (
                "PREFLIGHT_BLOCKED"
                if pipeline.get("state") == "BLOCKED"
                else "PREFLIGHT_ACTIVE"
            )
            if pipeline.get("phase") == "PREFLIGHT_COMPLETE":
                result["state"] = (
                    "PREFLIGHT_COMPLETE"
                    if verify_preflight(root, pipeline)
                    else "PREFLIGHT_COMPLETE_UNVERIFIED"
                )
                result["reason"] = (
                    "Preflight finished; production training remains disabled"
                    if result["state"] == "PREFLIGHT_COMPLETE"
                    else "Preflight completion evidence is missing or changed"
                )
        elif result.get("staging", {}).get("state", "UNKNOWN").split()[
            0
        ] in TERMINAL - {"COMPLETED"}:
            result["state"] = "STAGING_FAILED"
        elif any(
            v.get("state", "UNKNOWN").split()[0] in TERMINAL - {"COMPLETED"}
            for v in result.get("recovery_jobs", {}).values()
        ):
            result["state"] = "RECOVERY_FAILED"
        return result
    for label in ("gate", "readiness"):
        report = safe_report(registry[label])
        result[label] = report.get("status", "missing") if report else "missing"
    runs = registry["runs"]
    ids = [run["job_id"] for run in runs.values() if run.get("job_id")]
    table, error = state_tables(ids)
    result["scheduler_error"] = error
    for cell, run in runs.items():
        result["runs"][cell] = inspect_run(run, registry, table.get(run.get("job_id")))
        observed = result["runs"][cell]["state"]
        if observed in ("TIMEOUT", "PREEMPTED") and continue_timed_out_run(
            root / "registry.json", cell, run.get("job_id"), observed
        ):
            result["runs"][cell]["state"] = "CONTINUATION_SUBMITTED"
            result["runs"][cell]["terminal"] = False
    states = [r["state"] for r in result["runs"].values()]
    result["state"] = "MONITORING" if states else "WAITING_FOR_LAUNCH"
    if error:
        result["state"] = "SCHEDULER_UNAVAILABLE"
    elif states and all(s == "COMPLETE" for s in states):
        result["state"] = "COMPLETE" if len(states) == 36 else "SUBSET_COMPLETE"
    elif states and all(r["terminal"] for r in result["runs"].values()):
        result["state"] = "TERMINAL_REVIEW_REQUIRED"
    return result


def event_view(value):
    return {
        "state": value["state"],
        "scheduler_error": value.get("scheduler_error"),
        "staging": value.get("staging"),
        "recovery_jobs": value.get("recovery_jobs"),
        "readiness": value.get("readiness"),
        "pipeline": {
            k: (value.get("pipeline") or {}).get(k) for k in ("state", "phase", "error")
        },
        "runs": {
            cell: {k: run.get(k) for k in ("state", "error_markers", "log_stale")}
            for cell, run in value["runs"].items()
        },
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--interval", type=int, default=300)
    p.add_argument("--once", action="store_true")
    args = p.parse_args()
    if args.interval < 60:
        p.error("--interval must be at least 60 seconds")
    root = args.run_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    os.chmod(root, 0o700)
    lock = (root / "monitor.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("A monitor already owns this run root.", flush=True)
        return
    prior = safe_report(root / "status.json")
    while True:
        try:
            value = snapshot(root)
        except Exception as error:
            value = {
                "schema_version": 1,
                "updated_at": now(),
                "heartbeat_epoch": time.time(),
                "monitor_pid": os.getpid(),
                "state": "MONITOR_ERROR",
                "error_type": type(error).__name__,
                "runs": {},
            }
        write_json(root / "status.json", value)
        if prior is None or event_view(prior) != event_view(value):
            with (root / "events.jsonl").open("a") as stream:
                os.chmod(root / "events.jsonl", 0o600)
                stream.write(
                    json.dumps({"timestamp": now(), **event_view(value)}) + "\n"
                )
            print(json.dumps({"timestamp": now(), "state": value["state"]}), flush=True)
        prior = value
        if args.once or value["state"] in (
            "COMPLETE",
            "SUBSET_COMPLETE",
            "TERMINAL_REVIEW_REQUIRED",
            "PREFLIGHT_COMPLETE",
        ):
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
