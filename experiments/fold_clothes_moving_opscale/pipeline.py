#!/usr/bin/env python3
"""Durable, fail-closed conductor: staging -> normalization -> GPU smoke -> sweep.

Start only after committing and testing the experiment. All actions are logged
before submission. Ambiguous submissions stop for reconciliation; no command is
silently retried. This conductor never cancels jobs or changes scientific config.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
import re
import shlex
import subprocess
import time
from pathlib import Path

from launch_pace import (
    BUDGETS,
    OPERATORS,
    REPO,
    command,
    digest,
    need,
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
}


def scheduler(job_id, array_count=0):
    """Return complete only when every expected array element is accounted for."""
    raw = command(
        ["sacct", "-n", "-X", "-P", "-j", job_id, "--format=JobIDRaw,State,ExitCode"],
        timeout=45,
    )
    rows = {}
    for line in raw.splitlines():
        parts = line.split("|")
        if len(parts) >= 3:
            key, state, code = parts[:3]
            if key == job_id or key.startswith(job_id + "_"):
                rows[key] = {"state": state.split()[0].rstrip("+"), "exit_code": code}
    # JobIDRaw for arrays can contain the scheduler's underlying numeric IDs;
    # ask for logical JobID too so expected task indices remain unambiguous.
    if array_count:
        raw = command(
            ["sacct", "-n", "-X", "-P", "-j", job_id, "--format=JobID,State,ExitCode"],
            timeout=45,
        )
        rows = {
            p[0]: {"state": p[1].split()[0].rstrip("+"), "exit_code": p[2]}
            for line in raw.splitlines()
            if len(p := line.split("|")) >= 3 and p[0].startswith(job_id + "_")
        }
        expected = {f"{job_id}_{i}" for i in range(array_count)}
    else:
        expected = {job_id}
    if any(
        rows[key]["state"] in TERMINAL
        and (rows[key]["state"] != "COMPLETED" or rows[key]["exit_code"] != "0:0")
        for key in expected & rows.keys()
    ):
        return "FAILED", rows
    if not expected <= rows.keys():
        return "WAITING", rows
    states = [rows[key]["state"] for key in expected]
    if all(
        s == "COMPLETED" and rows[key]["exit_code"] == "0:0"
        for s, key in zip(states, expected)
    ):
        return "COMPLETED", rows
    if any(s in TERMINAL - {"COMPLETED"} for s in states):
        return "FAILED", rows
    return "WAITING", rows


def guard(commit):
    need(
        command(["git", "-C", str(REPO), "rev-parse", "HEAD"]) == commit,
        "Worktree commit changed",
    )
    need(
        not command(["git", "-C", str(REPO), "status", "--porcelain"]),
        "Experiment worktree is dirty",
    )


def script_header(root):
    return "\n".join(
        [
            "#!/bin/bash",
            "set -euo pipefail",
            "umask 077",
            "set -a",
            "source " + shlex.quote(str(Path.home() / ".egoverse_env")),
            "set +a",
            "export PYTHONDONTWRITEBYTECODE=1",
            "export PYTHONPATH="
            + shlex.quote(str(REPO) + ":" + str(root / "python-deps"))
            + ':"${PYTHONPATH:-}"',
            "export PATH=" + shlex.quote(str(Path.home() / ".local/bin")) + ':"$PATH"',
            "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1",
            "export EGOVERSE_MECKA_PRIVATE_DIR=" + shlex.quote(str(root / "private")),
            "export EGOVERSE_ROOT=" + shlex.quote(str(REPO)),
            "cd " + shlex.quote(str(REPO)),
            "",
        ]
    )


def ensure_job(state, root, key, body, *, gpu=None, array=False):
    actions = state["actions"]
    if key in actions:
        need(
            actions[key]["state"] != "SUBMITTING",
            f"{key}: prior submission ambiguous; reconcile before continuing",
        )
        return actions[key]
    path = root / "pipeline_jobs" / f"{key}.sbatch"
    path.parent.mkdir(exist_ok=True)
    path.write_text(script_header(root) + body + "\n")
    os.chmod(path, 0o700)
    cmd = [
        "sbatch",
        "--parsable",
        "--job-name=fold-" + key,
        "--partition=" + ("gpu-" + gpu if gpu else "cpu-amd"),
        "--account=gts-dxu345-rl2",
        "--qos=inferno",
        "--nodes=1",
        "--ntasks-per-node=1",
        "--cpus-per-task=" + ("8" if gpu else "16"),
        "--mem=" + ("128G" if gpu or array else "64G"),
        "--time=" + ("12:00:00" if gpu else "1-00:00:00"),
        "--no-requeue",
        "--output=" + str(path.parent / (key + "-%A_%a.out")),
        "--error=" + str(path.parent / (key + "-%A_%a.err")),
    ]
    if gpu:
        cmd.append("--gres=gpu:" + gpu + ":1")
    if array:
        cmd.append("--array=0-35%8")
    cmd.append(str(path))
    record = {
        "state": "SUBMITTING",
        "created_at": now(),
        "command": cmd,
        "script_sha256": digest(path),
        "array_count": 36 if array else 0,
    }
    actions[key] = record
    write_json(root / "pipeline.json", state)
    raw = command(cmd, timeout=90).split(";")[0]
    need(raw.isdigit(), "Unrecognized sbatch response")
    record.update(state="SUBMITTED", job_id=raw)
    write_json(root / "pipeline.json", state)
    return record


def await_action(action):
    status, rows = scheduler(action["job_id"], action["array_count"])
    need(
        status != "FAILED", f"Preflight job {action['job_id']} failed; inspect its logs"
    )
    return status == "COMPLETED"


def norm_evidence(root, commit, design_sha, cells=None):
    cells = cells if cells is not None else ((b, n) for b in BUDGETS for n in OPERATORS)
    canonical_sha = digest(root / "private/canonical_zarr/_episode_index.json")
    for b, n in cells:
        report = read_json(root / "norm" / f"b{b}_op{n}" / "norm_complete.json")
        config = (
            REPO / "experiments/fold_clothes_moving_opscale" / f"budget_{b}/op_{n}.yaml"
        )
        split = root / "private/splits" / f"fold_moving_b{b}_op{n}.json"
        need(
            report.get("status") == "passed"
            and report.get("commit") == commit
            and report.get("design_sha256") == design_sha
            and report.get("canonical_index_sha256") == canonical_sha
            and report.get("budget_hours") == b
            and report.get("operator_count") == n
            and report.get("config_sha256") == digest(config)
            and report.get("split_sha256") == digest(split)
            and report.get("stats_sha256") == digest(report["stats_path"]),
            f"Normalization evidence mismatch for b{b}_op{n}",
        )


def step(state, args):
    root = args.run_root.resolve()
    private = root / "private"
    guard(args.commit)
    need(
        digest(private / "moving_opscale_design.json") == state["design_sha256"],
        "Design changed",
    )
    stage = read_json(root / "staging.json")
    current, rows = scheduler(stage["job_id"])
    if current == "WAITING":
        return "WAITING_FOR_STAGING"
    if (
        current == "FAILED"
        and state.get("accepted_staging", {}).get("job_id") != stage["job_id"]
    ):
        # The old data-only checker can fail on the two absent episodes or old
        # schema assumptions after sync succeeds. Preserve proof that it reached
        # its completed verification phase BEFORE the corrected checker replaces
        # readiness.json with an in-progress report. This never grants readiness.
        report = read_json(root / "readiness.json")
        need(
            "verified_episodes" in report
            and "failures" in report
            and report.get("design_sha256") == state["design_sha256"]
            and report["verified_episodes"] + len(report["failures"])
            == report.get("selected_episodes"),
            "Staging failed before completing its transfer/check phase",
        )
        state["accepted_staging"] = {
            "job_id": stage["job_id"],
            "accepted_at": now(),
            "transfer_reached_verification": True,
            "report_sha256": digest(root / "readiness.json"),
            "selected_episodes": report["selected_episodes"],
        }
        write_json(root / "pipeline.json", state)
    for job in stage.get("recovery_job_ids", []):
        status, _ = scheduler(job)
        need(status != "FAILED", f"Recovery job {job} failed")
        if status == "WAITING":
            return "WAITING_FOR_RECOVERY"
    python = str(args.python.absolute())
    tools = REPO / "experiments/fold_clothes_moving_opscale"
    checker = [
        python,
        str(tools / "check_readiness.py"),
        "--private-root",
        str(private),
        "--output",
        str(root / "readiness.json"),
        "--inventory",
        str(root / "destination_inventory.json"),
    ]
    data_body = (
        shlex.join(
            checker
            + [
                "--refresh-missing",
                "--inventory-only",
                "--stage-missing-only",
                "--prepare-stage",
                str(root / "recovered_stage_commands.txt"),
            ]
        )
        + "\n"
    )
    data_body += 'export AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY" AWS_DEFAULT_REGION=auto\nunset AWS_SESSION_TOKEN AWS_PROFILE\n'
    batch = shlex.quote(str(root / "recovered_stage_commands.txt"))
    data_body += f'if [ -s {batch} ]; then s5cmd --numworkers 16 --endpoint-url "$R2_ENDPOINT_URL" run {batch}; fi\n'
    data_body += shlex.join(checker + ["--workers", "8"])
    action = ensure_job(state, root, "final-data-check", data_body)
    if not await_action(action):
        return "VERIFYING_FINAL_DATA"
    if "refresh-readiness" in state["actions"] and not await_action(
        state["actions"]["refresh-readiness"]
    ):
        return "REFRESHING_READINESS"
    report = read_json(root / "readiness.json")
    need(
        report.get("status") == "passed" and all(report["checks"].values()),
        "Final data readiness failed",
    )
    base = [python, str(tools / "run_preflight.py")]
    norm_body = "budgets=(24 32 48 64 72 96)\noperators=(4 8 16 32 64 128)\n"
    norm_body += shlex.join(
        base
        + [
            "norm",
            "--private-dir",
            str(private),
            "--run-root",
            str(root),
            "--commit",
            args.commit,
        ]
    )
    norm_body += ' --budget "${budgets[$((SLURM_ARRAY_TASK_ID / 6))]}" --operators "${operators[$((SLURM_ARRAY_TASK_ID % 6))]}"'
    action = ensure_job(state, root, "normalization", norm_body, array=True)
    norms_done = await_action(action)  # Fail immediately if any array task fails.
    smoke_cells = ((24, 4), (24, 128))
    smoke_norms_pending = False
    for b, n in smoke_cells:
        path = root / "norm" / f"b{b}_op{n}" / "norm_complete.json"
        if not path.is_file():
            smoke_norms_pending = True
            continue
        progress = read_json(path)
        need(
            progress.get("status") in ("running", "passed"),
            f"Smoke normalization b{b}_op{n} failed or is malformed",
        )
        need(
            progress.get("commit") == args.commit
            and progress.get("design_sha256") == state["design_sha256"]
            and progress.get("budget_hours") == b
            and progress.get("operator_count") == n,
            f"Smoke normalization b{b}_op{n} belongs to another experiment",
        )
        smoke_norms_pending |= progress["status"] == "running"
    if smoke_norms_pending:
        need(
            not norms_done,
            "Normalization array completed without both passed smoke normalization reports",
        )
        return "COMPUTING_NORMALIZATION"
    norm_evidence(root, args.commit, state["design_sha256"], cells=smoke_cells)
    if norms_done:
        norm_evidence(root, args.commit, state["design_sha256"])
    smoke_body = shlex.join(
        base
        + [
            "smoke",
            "--private-dir",
            str(private),
            "--run-root",
            str(root),
            "--commit",
            args.commit,
            "--output",
            str(root / "evaluation_smoke.json"),
        ]
    )
    action = ensure_job(state, root, "evaluation-smoke", smoke_body, gpu=args.gpu)
    if not await_action(action):
        return "RUNNING_GPU_SMOKE" if norms_done else "NORMALIZATION_AND_GPU_SMOKE"
    evidence = read_json(root / "evaluation_smoke.json")
    need(
        evidence.get("status") == "passed"
        and evidence.get("commit") == args.commit
        and evidence.get("design_sha256") == state["design_sha256"],
        "GPU smoke evidence mismatches experiment",
    )
    if not norms_done:
        return "WAITING_FOR_REMAINING_NORMALIZATION"
    # Recheck immutable local data if long norm/smoke jobs made the launch proof stale.
    checked = dt.datetime.fromisoformat(report["checked_at"].replace("Z", "+00:00"))
    if (dt.datetime.now(dt.timezone.utc) - checked).total_seconds() > 23 * 3600:
        action = ensure_job(
            state, root, "refresh-readiness", shlex.join(checker + ["--workers", "8"])
        )
        if not await_action(action):
            return "REFRESHING_READINESS"
        report = read_json(root / "readiness.json")
        need(report.get("status") == "passed", "Refreshed readiness failed")
    validator = [
        python,
        str(tools / "validate_setup.py"),
        "--private-dir",
        str(private),
        "--output",
        str(root / "gate.json"),
        "--evaluation-smoke",
        str(root / "evaluation_smoke.json"),
    ]
    command(validator, cwd=REPO, timeout=600)
    need(
        read_json(root / "gate.json")["status"] == "passed",
        "Launch gate remains blocked",
    )
    if "production" not in state["actions"]:
        state["actions"]["production"] = {"state": "SUBMITTING", "created_at": now()}
        write_json(root / "pipeline.json", state)
        launch = [
            python,
            str(tools / "launch_pace.py"),
            "--gate",
            str(root / "gate.json"),
            "--readiness",
            str(root / "readiness.json"),
            "--run-root",
            str(root),
            "--python",
            python,
            "--gpu",
            args.gpu,
            "--submit",
        ]
        with (root / "production-submission.log").open("a") as stream:
            result = subprocess.run(
                launch, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT
            )
        need(
            result.returncode == 0,
            "Production submission interrupted; reconcile registry before retrying",
        )
        state["actions"]["production"]["state"] = "SUBMITTED"
        write_json(root / "pipeline.json", state)
    need(
        state["actions"]["production"]["state"] == "SUBMITTED",
        "Prior production submission is ambiguous",
    )
    registry = read_json(root / "registry.json")
    need(
        len(registry["runs"]) == 36
        and all(r.get("job_id") for r in registry["runs"].values()),
        "Not all 36 training jobs have confirmed submission IDs",
    )
    return "SWEEP_SUBMITTED"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--commit", required=True)
    p.add_argument("--python", type=Path, required=True)
    p.add_argument("--gpu", choices=("h100", "h200"), default="h200")
    p.add_argument("--interval", type=int, default=120)
    p.add_argument("--once", action="store_true")
    a = p.parse_args()
    need(re.fullmatch("[0-9a-f]{40}", a.commit) is not None, "Full commit required")
    need(a.interval >= 60, "Poll interval must be at least 60 seconds")
    root = a.run_root.resolve()
    lock = (root / "pipeline.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    guard(a.commit)
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [str(REPO), str(root / "python-deps"), os.environ.get("PYTHONPATH", "")]
    )
    state_path = root / "pipeline.json"
    state = (
        read_json(state_path)
        if state_path.exists()
        else {
            "schema_version": 1,
            "created_at": now(),
            "commit": a.commit,
            "design_sha256": digest(root / "private/moving_opscale_design.json"),
            "actions": {},
        }
    )
    need(state["commit"] == a.commit, "Existing pipeline belongs to another commit")
    write_json(state_path, state)
    while True:
        try:
            phase = step(state, a)
            status = {
                "state": "COMPLETE" if phase == "SWEEP_SUBMITTED" else "ACTIVE",
                "phase": phase,
                "updated_at": now(),
                "pid": os.getpid(),
                "commit": a.commit,
            }
        except Exception as error:
            status = {
                "state": "BLOCKED",
                "phase": "REVIEW_REQUIRED",
                "updated_at": now(),
                "pid": os.getpid(),
                "commit": a.commit,
                "error": str(error)[:500],
            }
            write_json(root / "pipeline_status.json", status)
            print(json.dumps(status), flush=True)
            return 2
        write_json(root / "pipeline_status.json", status)
        if state.get("last_phase") != phase:
            print(json.dumps(status), flush=True)
            state["last_phase"] = phase
            write_json(state_path, state)
        if a.once or phase == "SWEEP_SUBMITTED":
            return 0
        time.sleep(a.interval)


if __name__ == "__main__":
    raise SystemExit(main())
