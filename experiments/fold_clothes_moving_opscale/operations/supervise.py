#!/usr/bin/env python3
"""Read-only supervisor of the pinned sweep conductor, monitor and Slurm jobs.

Writes only its own operations/ artifacts. Never submits, cancels, restarts or
signals a process. Alert entries contain labels/numbers, never raw log lines,
command lines, credentials, signed URLs, or exception text from child tools.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
import re
import socket
import subprocess
import time
from collections import Counter
from pathlib import Path

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
ERROR_PATTERNS = {
    "traceback": r"Traceback \(most recent call last\)",
    "cuda_oom": r"CUDA out of memory",
    "missing_file": r"FileNotFoundError",
    "missing_module": r"ModuleNotFoundError",
    "nonfinite": r"\bNaN\b|non[- ]?finite",
    "nccl_error": r"NCCL[^\n]{0,200}(?:error|timed out)",
    "runtime_error": r"RuntimeError:",
}
PROGRESS_FIELDS = (
    "checked",
    "total",
    "failed",
    "inventory_pages",
    "selected_episode_prefixes_seen",
    "global_step",
    "step",
    "epoch",
    "completed_videos",
    "expected_videos",
)


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def load(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        os.chmod(temporary, 0o600)
        json.dump(value, stream, indent=2)
        stream.write("\n")
    temporary.replace(path)


def label(value, fallback="UNKNOWN"):
    text = str(value or "")
    return text if re.fullmatch(r"[a-zA-Z0-9_.:/-]{1,100}", text) else fallback


def numeric(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool)


def heartbeat_age(report, current):
    try:
        when = report.get("heartbeat_epoch")
        if not numeric(when):
            stamp = report["updated_at"]
            when = dt.datetime.fromisoformat(stamp.replace("Z", "+00:00")).timestamp()
        return max(0, current - when)
    except (KeyError, ValueError, TypeError, AttributeError):
        return None


def process_state(
    pid, script, root, *, commit=None, expected_ticks=None, proc_root=Path("/proc")
):
    result = {"pid": pid, "alive": False, "identity_match": False, "state": "MISSING"}
    base = Path(proc_root) / str(pid)
    try:
        tokens = (base / "cmdline").read_bytes().split(b"\0")
        argv = [x.decode(errors="replace") for x in tokens if x]
        stat = (base / "stat").read_text().rsplit(")", 1)[1].split()
        result["start_ticks"] = int(stat[19])
        result["alive"] = stat[0] not in ("Z", "X")

        def option(name, expected):
            return (
                name in argv
                and argv.index(name) + 1 < len(argv)
                and argv[argv.index(name) + 1] == expected
            )

        identity = str(script) in argv and option("--run-root", str(root))
        if commit is not None:
            identity &= option("--commit", commit)
        if expected_ticks is not None:
            identity &= result["start_ticks"] == expected_ticks
        result["identity_match"] = bool(identity)
        result["state"] = (
            "RUNNING"
            if result["alive"] and identity
            else "IDENTITY_MISMATCH"
            if result["alive"]
            else "EXITED"
        )
    except FileNotFoundError:
        pass
    except (PermissionError, OSError):
        result["state"] = "UNREADABLE"
    except (IndexError, ValueError):
        result["state"] = "UNPARSEABLE"
    return result


def run_command(args):
    process = subprocess.run(args, text=True, capture_output=True, timeout=25)
    if process.returncode and not (
        args[0] == "squeue" and "Invalid job id specified" in process.stderr
    ):
        raise RuntimeError(label(args[0]) + "_FAILED")
    return process.stdout


def parse_slurm(accounting, queue):
    rows = {}
    for line in accounting.splitlines():
        parts = line.strip().split("|")
        if len(parts) >= 3 and re.fullmatch(r"\d+(?:_\d+)?", parts[0]):
            state = parts[1].split()[0].rstrip("+") if parts[1].split() else "UNKNOWN"
            rows[parts[0]] = {"state": label(state), "exit_code": label(parts[2])}
    for line in queue.splitlines():
        parts = line.strip().split("|")
        if len(parts) >= 2 and re.fullmatch(r"\d+(?:_\d+)?", parts[0]):
            state = label(parts[1])
            old = rows.get(parts[0], {})
            rows[parts[0]] = {
                "state": state,
                "exit_code": old.get("exit_code") if state in TERMINAL else None,
            }
    return rows


def job_summary(job_id, array_count, rows):
    expected = (
        [f"{job_id}_{i}" for i in range(array_count)] if array_count else [job_id]
    )
    tasks = {
        job: rows.get(job, {"state": "UNKNOWN", "exit_code": None}) for job in expected
    }
    failures = {
        job: row
        for job, row in tasks.items()
        if row["state"] in TERMINAL
        and (row["state"] != "COMPLETED" or row["exit_code"] != "0:0")
    }
    counts = dict(Counter(row["state"] for row in tasks.values()))
    complete = all(
        row["state"] == "COMPLETED" and row["exit_code"] == "0:0"
        for row in tasks.values()
    )
    state = (
        "FAILED"
        if failures
        else "COMPLETED"
        if complete
        else "RUNNING"
        if counts.get("RUNNING")
        else "PENDING"
        if counts.get("PENDING")
        else "UNKNOWN"
    )
    return {
        "job_id": job_id,
        "state": state,
        "expected_tasks": len(expected),
        "counts": counts,
        "tasks": tasks,
        "failures": failures,
    }


def tail_progress(path, current, limit=32768):
    try:
        stat = path.stat()
        with path.open("rb") as stream:
            stream.seek(max(0, stat.st_size - limit))
            text = stream.read(limit).decode(errors="replace")
    except OSError:
        return None
    result = {
        "bytes": stat.st_size,
        "age_seconds": round(max(0, current - stat.st_mtime)),
        "errors": sorted(
            key
            for key, pattern in ERROR_PATTERNS.items()
            if re.search(pattern, text, re.I)
        ),
        "progress": {},
    }
    for line in text.splitlines():
        try:
            parsed = json.loads(line)
        except (ValueError, TypeError):
            continue
        if isinstance(parsed, dict):
            result["progress"].update(
                {
                    key: parsed[key]
                    for key in PROGRESS_FIELDS
                    if numeric(parsed.get(key))
                }
            )
    for key in ("global_step", "epoch"):
        found = re.findall(r"\b" + key + r"[=:\s]+(\d+)", text)
        if found:
            result["progress"][key] = int(found[-1])
    fractions = re.findall(r"(\d{1,3})%[^\n]{0,120}?(\d+)\s*/\s*(\d+)", text)
    if fractions:
        pct, finished, total = map(int, fractions[-1])
        if 0 <= pct <= 100 and finished <= total:
            result["progress"].update(
                percent=pct, items_finished=finished, items_total=total
            )
    return result


def transition_view(snapshot):
    return {
        "state": snapshot["state"],
        "phase": snapshot.get("phase"),
        "processes": {
            key: {
                field: value.get(field)
                for field in ("pid", "state", "heartbeat_stale", "expected_exit")
            }
            for key, value in snapshot.get("processes", {}).items()
        },
        "actions": {
            key: {field: value.get(field) for field in ("job_id", "state", "counts")}
            for key, value in snapshot.get("actions", {}).items()
        },
        "training": {
            key: {field: value.get(field) for field in ("state", "verified_state")}
            for key, value in snapshot.get("training", {}).items()
        },
        "readiness": snapshot.get("readiness"),
        "normalization": snapshot.get("normalization"),
        "smoke": snapshot.get("smoke"),
        "scheduler_error": snapshot.get("scheduler_error"),
    }


def alert_changes(previous, current):
    events = []
    for key in sorted(current):
        if previous.get(key) != current[key]:
            events.append(
                {
                    "event": "raised" if key not in previous else "changed",
                    "alert_id": key,
                    **current[key],
                }
            )
    for key in sorted(previous.keys() - current.keys()):
        events.append(
            {
                "event": "resolved",
                "alert_id": key,
                "kind": previous[key]["kind"],
                "severity": "info",
            }
        )
    return events


def dynamic_targets(root, commit, defaults):
    """Follow only controller restarts with matching persisted audit receipts."""
    document = load(root / "operations/process_targets.json")
    if document is None:
        return defaults, []
    invalid = []
    if document.get("commit") != commit or document.get("run_root") != str(root):
        return defaults, ["document"]
    audit = load(root / "operations/recovery_state.json", {}).get(
        "restart_attempts", {}
    )
    result = dict(defaults)
    for name in ("pipeline", "monitor"):
        target = document.get("targets", {}).get(name, {})
        if target == defaults[name]:
            continue
        valid = (
            isinstance(target.get("pid"), int)
            and isinstance(target.get("start_ticks"), int)
            and any(
                receipt.get("state") == "VERIFIED"
                and receipt.get("id") == target.get("restart_id")
                and receipt.get("pid") == target["pid"]
                and receipt.get("start_ticks") == target["start_ticks"]
                for receipt in audit.get(name, [])
            )
        )
        if valid:
            result[name] = target
        elif (
            target.get("pid") == defaults[name]["pid"]
            and target.get("start_ticks") == defaults[name]["start_ticks"]
        ):
            continue
        else:
            invalid.append(name)
    return result, invalid


def operation_process(
    saved, *, script=None, root=None, mode=None, proc_root=Path("/proc")
):
    """Verify persisted host/PID/start/group identity; never infer death remotely."""
    result = {"state": "IDENTITY_MISSING", "alive": False}
    if not isinstance(saved, dict) or any(
        not isinstance(saved.get(k), int) for k in ("pid", "start_ticks", "pgid")
    ):
        return result
    result["pid"] = saved["pid"]
    if saved.get("host") != socket.gethostname():
        result["state"] = "REMOTE_IDENTITY_UNVERIFIABLE"
        return result
    try:
        base = Path(proc_root) / str(saved["pid"])
        stat = (base / "stat").read_text().rsplit(")", 1)[1].split()
        result["alive"] = stat[0] not in ("Z", "X")
        matches = (
            int(stat[19]) == saved["start_ticks"] and int(stat[2]) == saved["pgid"]
        )
        if script is not None:
            argv = [
                x.decode(errors="replace")
                for x in (base / "cmdline").read_bytes().split(b"\0")
                if x
            ]
            matches &= str(script) in argv
            matches &= "--run-root" in argv and argv[
                argv.index("--run-root") + 1
            ] == str(root)
            matches &= str(script) in argv and argv[argv.index(str(script)) + 1] == mode
        result["state"] = (
            ("RUNNING" if matches else "IDENTITY_MISMATCH")
            if result["alive"]
            else "EXITED"
        )
    except FileNotFoundError:
        result["state"] = "MISSING"
    except (PermissionError, OSError):
        result["state"] = "UNREADABLE"
    except (ValueError, IndexError):
        result["state"] = "UNPARSEABLE"
    return result


def recovery_health(args, current, *, expected_complete=False, proc_root=Path("/proc")):
    """Observer-only health of the recovery system, enabled only after setup.

    These alerts explicitly cannot dispatch a fixer for its own control policy.
    A bounded, live-owner promotion receipt temporarily suppresses expected
    source/process transitions; expired or invalid receipts never hide health.
    """
    ops = args.run_root / "operations"
    result = {"maintenance": False, "services": {}, "alerts": {}}

    def issue(key, kind, **fields):
        result["alerts"]["operations:" + key] = {
            "kind": kind,
            "severity": "error",
            "automatic_repair": False,
            **fields,
        }

    intent = load(ops / "promotion_maintenance.json", {})
    if intent.get("active") is True:
        valid = False
        try:
            plan_path = Path(intent["plan_path"]).resolve()
            plan = load(plan_path, {})
            started = dt.datetime.fromisoformat(
                intent["started_at"].replace("Z", "+00:00")
            ).timestamp()
            deadline = float(intent["deadline_epoch"])
            source_pairs = {
                (intent["old_repo"], intent["old_commit"]),
                (intent["new_repo"], intent["new_commit"]),
            }
            valid = (
                plan_path.is_relative_to(ops.resolve())
                and plan_path.is_file()
                and plan.get("run_root") == str(args.run_root)
                and bool(intent.get("transaction_id"))
                and all(
                    plan.get(k) == intent.get(k)
                    for k in (
                        "transaction_id",
                        "old_repo",
                        "old_commit",
                        "new_repo",
                        "new_commit",
                    )
                )
                and (str(args.repo), args.commit) in source_pairs
                and started <= current + 5
                and started < deadline <= started + 3601
            )
            owner = operation_process(intent.get("owner_identity"), proc_root=proc_root)
            if not valid:
                issue("promotion", "promotion_maintenance_invalid")
            elif current > deadline:
                issue("promotion", "promotion_maintenance_expired")
            elif owner["state"] != "RUNNING":
                issue(
                    "promotion",
                    "promotion_maintenance_owner_unverified",
                    owner_state=owner["state"],
                )
            else:
                result.update(
                    maintenance=True, transaction_id=label(intent["transaction_id"])
                )
        except (KeyError, TypeError, ValueError, OverflowError):
            issue("promotion", "promotion_maintenance_invalid")
    if result["maintenance"]:
        return result

    config_path = ops / "recovery_config.json"
    config = load(config_path, {})
    initialized = config_path.exists()
    control = load(ops / "dispatcher_control.json", {})
    activated = control.get("enabled") is True
    if not initialized and not activated:
        return result
    if (
        not config
        or config.get("run_root") != str(args.run_root)
        or config.get("repo") != str(args.repo)
        or config.get("commit") != args.commit
    ):
        issue("configuration", "recovery_configuration_identity_mismatch")
    services = []
    if initialized:
        services.append(
            (
                "recovery",
                "recovery_watch_status.json",
                "recovery_controller.py",
                "watch",
            )
        )
    if activated:
        services.append(
            ("dispatcher", "dispatcher_status.json", "agent_dispatcher.py", "serve")
        )
    for name, receipt, script, mode in services:
        report = load(ops / receipt, {})
        state = label(report.get("state"))
        process = operation_process(
            report.get("identity"),
            script=ops / script,
            root=args.run_root,
            mode=mode,
            proc_root=proc_root,
        )
        age = heartbeat_age(report, current)
        done = name == "recovery" and state == "COMPLETE" and expected_complete
        stale = not done and (age is None or age > 600)
        result["services"][name] = {
            "state": process["state"],
            "reported_state": state,
            "pid": process.get("pid"),
            "heartbeat_stale": stale,
            "heartbeat_age_seconds": round(age) if age is not None else None,
        }
        if not done and process["state"] != "RUNNING":
            issue(
                name + ":process",
                "recovery_service_identity_unverified",
                service=name,
                process_state=process["state"],
            )
        if stale:
            issue(name + ":heartbeat", "recovery_service_heartbeat_stale", service=name)
        if state == "BLOCKED":
            issue(name + ":blocked", "recovery_service_blocked", service=name)
        if name == "recovery" and state == "COMPLETE" and not expected_complete:
            issue(name + ":completion", "recovery_controller_stopped_before_completion")
    if initialized:
        for key, incident in (
            load(ops / "recovery_state.json", {}).get("incidents", {}).items()
        ):
            if incident.get("active") is not True:
                continue
            state = incident.get("state")
            if state in ("NEEDS_REVIEW", "LEASE_EXPIRED"):
                issue(
                    "incident:" + label(key),
                    "recovery_incident_requires_review",
                    incident=label(key),
                    state=state,
                )
    return result


def collect(args, identities, current=None):
    current = time.time() if current is None else current
    root, repo = args.run_root, args.repo
    pipeline = load(root / "pipeline.json", {})
    progress = load(root / "pipeline_status.json", {})
    monitored = load(root / "status.json", {})
    registry = load(root / "registry.json", {})
    alerts = {}

    def issue(key, kind, severity="error", **fields):
        alerts[key] = {"kind": kind, "severity": severity, **fields}

    snapshot = {
        "schema_version": 1,
        "updated_at": utc(),
        "heartbeat_epoch": current,
        "supervisor_pid": os.getpid(),
        "commit": args.commit,
        "state": "SUPERVISING",
        "phase": label(progress.get("phase")),
        "processes": {},
        "actions": {},
        "training": {},
    }
    recovery = recovery_health(
        args,
        current,
        expected_complete=(
            monitored.get("state") == "COMPLETE"
            and len(monitored.get("runs", {})) == 36
            and all(r.get("state") == "COMPLETE" for r in monitored["runs"].values())
        ),
    )
    snapshot["recovery_system"] = recovery
    alerts.update(recovery["alerts"])
    maintenance = recovery["maintenance"]
    defaults = {
        name: {"pid": pid, "start_ticks": identities.get(name)}
        for name, pid in (
            ("pipeline", args.pipeline_pid),
            ("monitor", args.monitor_pid),
        )
    }
    targets, invalid_targets = dynamic_targets(root, args.commit, defaults)
    for name in [] if maintenance else invalid_targets:
        issue("identity_targets:" + name, "invalid_recovery_identity", process=name)
    submitted = (
        progress.get("phase") == "SWEEP_SUBMITTED"
        and len(registry.get("runs", {})) == 36
    )
    preflight_complete = (
        progress.get("phase") == "PREFLIGHT_COMPLETE"
        and monitored.get("state") == "PREFLIGHT_COMPLETE"
    )
    monitor_terminal = monitored.get("state") in (
        "COMPLETE",
        "TERMINAL_REVIEW_REQUIRED",
        "SUBSET_COMPLETE",
        "PREFLIGHT_COMPLETE",
    )
    for name, pid, report, expected_exit, stale_seconds in (
        (
            "pipeline",
            targets["pipeline"]["pid"],
            progress,
            submitted
            or progress.get("state") in ("BLOCKED", "PREFLIGHT_COMPLETE")
            or progress.get("phase") == "PREFLIGHT_COMPLETE",
            600,
        ),
        ("monitor", targets["monitor"]["pid"], monitored, monitor_terminal, 900),
    ):
        script = (
            repo
            / "experiments/fold_clothes_moving_opscale"
            / ("pipeline.py" if name == "pipeline" else "monitor.py")
        )
        process = process_state(
            pid,
            script,
            root,
            commit=args.commit if name == "pipeline" else None,
            expected_ticks=targets[name]["start_ticks"],
        )
        process["expected_exit"] = expected_exit
        age = heartbeat_age(report, current)
        process["heartbeat_age_seconds"] = round(age) if age is not None else None
        process["heartbeat_stale"] = not expected_exit and (
            age is None or age > stale_seconds
        )
        snapshot["processes"][name] = process
        heartbeat_pid = report.get("pid" if name == "pipeline" else "monitor_pid")
        if not maintenance and heartbeat_pid is not None and heartbeat_pid != pid:
            issue(
                "process:" + name + ":heartbeat_identity",
                "heartbeat_pid_mismatch",
                process=name,
                pid=pid,
            )
        if not maintenance and not expected_exit and process["state"] != "RUNNING":
            issue(
                "process:" + name,
                "process_" + process["state"].lower(),
                process=name,
                pid=pid,
            )
        if not maintenance and process["heartbeat_stale"]:
            issue("heartbeat:" + name, "heartbeat_stale", process=name, pid=pid)
    if not maintenance and progress.get("state") == "BLOCKED":
        issue(
            "pipeline:blocked", "pipeline_blocked", phase=label(progress.get("phase"))
        )
    if not maintenance and monitored.get("state") == "MONITOR_ERROR":
        issue("monitor:error", "primary_monitor_error")
    if not maintenance and monitored.get("state") in (
        "TERMINAL_REVIEW_REQUIRED",
        "SUBSET_COMPLETE",
    ):
        issue(
            "monitor:review",
            "completion_review_required",
            state=label(monitored.get("state")),
        )
    known = {}
    for name, action in pipeline.get("actions", {}).items():
        name = label(name)
        if action.get("job_id") and str(action["job_id"]).isdigit():
            known["prepare:" + name] = (
                str(action["job_id"]),
                int(action.get("array_count", 0)),
                name,
            )
        elif action.get("state") in ("SUBMITTING", "SUBMISSION_UNCERTAIN"):
            issue(
                "submission:" + name,
                "submission_unconfirmed",
                severity="warning",
                action=name,
            )
    for cell, run in registry.get("runs", {}).items():
        cell = label(cell)
        if run.get("job_id") and str(run["job_id"]).isdigit():
            known["train:" + cell] = (str(run["job_id"]), 0, cell)
        if run.get("state") in ("CONTINUATION_BLOCKED", "SUBMISSION_UNCERTAIN"):
            issue(
                "training_registry:" + cell,
                "training_registry_requires_review",
                cell=cell,
                state=label(run["state"]),
            )
    rows = {}
    if known:
        joined = ",".join(sorted({value[0] for value in known.values()}))
        try:
            accounting = run_command(
                [
                    "sacct",
                    "-n",
                    "-X",
                    "-P",
                    "-j",
                    joined,
                    "--format=JobID%60,State%40,ExitCode",
                ]
            )
            queue = run_command(
                ["squeue", "--array", "-h", "-j", joined, "-o", "%i|%T"]
            )
            rows = parse_slurm(accounting, queue)
            snapshot["scheduler_error"] = None
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
            snapshot["scheduler_error"] = type(error).__name__
            issue("scheduler", "scheduler_unavailable", severity="warning")
    else:
        snapshot["scheduler_error"] = None
    for key, (job, array_count, name) in known.items():
        summary = job_summary(job, array_count, rows)
        category = "actions" if key.startswith("prepare:") else "training"
        snapshot[category][name] = summary
        for task, failure in summary["failures"].items():
            severity = (
                "warning"
                if category == "training"
                and failure["state"] in ("TIMEOUT", "PREEMPTED")
                else "error"
            )
            issue(
                "job:" + key + ":" + task,
                "preflight_job_failed"
                if category == "actions"
                else "training_allocation_failed",
                severity=severity,
                action=name,
                job_id=task,
                state=failure["state"],
                exit_code=failure["exit_code"],
            )
        if category == "actions":
            log_paths = sorted((root / "pipeline_jobs").glob(f"{name}-{job}_*.*"))[:96]
            summaries = [
                tail_progress(path, current)
                for path in log_paths
                if path.suffix in (".out", ".err")
            ]
            summaries = [value for value in summaries if value]
            summary["logs"] = {
                "files": len(summaries),
                "bytes": sum(value["bytes"] for value in summaries),
                "newest_age_seconds": min(
                    (value["age_seconds"] for value in summaries), default=None
                ),
                "error_labels": sorted(
                    {error for value in summaries for error in value["errors"]}
                ),
                "progress": [
                    value["progress"] for value in summaries if value["progress"]
                ],
            }
            if summary["logs"]["error_labels"]:
                issue(
                    "log:" + key,
                    "preflight_log_errors",
                    action=name,
                    labels=summary["logs"]["error_labels"],
                )
            if name == "final-data-check" and any(
                value["progress"].get("failed", 0) > 0 for value in summaries
            ):
                issue("data:verification", "data_verification_failures", action=name)
            age = summary["logs"]["newest_age_seconds"]
            if (
                summary["state"] == "RUNNING"
                and age is not None
                and age > args.progress_stale
            ):
                issue(
                    "progress:" + key,
                    "preflight_log_progress_stale",
                    severity="warning",
                    action=name,
                )
        else:
            observed = monitored.get("runs", {}).get(name, {})
            summary["verified_state"] = label(observed.get("state"))
            summary["observed_step"] = observed.get(
                "global_step", observed.get("observed_step")
            )
            if observed.get("log_stale"):
                issue(
                    "progress:" + key,
                    "training_progress_stale",
                    severity="warning",
                    cell=name,
                )
            if observed.get("error_markers"):
                issue("log:" + key, "training_log_errors", cell=name)
            if observed.get("state") in (
                "SLURM_COMPLETED_UNVERIFIED",
                "TRAINING_COMPLETE_EVALUATION_UNVERIFIED",
            ):
                issue("completion:" + key, "training_completion_unverified", cell=name)
    readiness = load(root / "readiness.json", {})
    snapshot["readiness"] = label(readiness.get("status"))
    snapshot["data_progress"] = {
        key: readiness[key]
        for key in ("selected_episodes", "verified_episodes")
        if numeric(readiness.get(key))
    }
    norm_counts = Counter()
    for path in sorted((root / "norm").glob("b*_op*/norm_complete.json"))[:36]:
        report = load(path, {})
        state = label(report.get("status"))
        norm_counts[state] += 1
        if state == "failed":
            issue(
                "norm:" + path.parent.name,
                "normalization_failed",
                cell=label(path.parent.name),
            )
    snapshot["normalization"] = dict(norm_counts)
    smoke = load(root / "evaluation_smoke.json", {})
    snapshot["smoke"] = label(smoke.get("status"))
    if smoke.get("status") == "failed":
        issue("smoke", "evaluation_smoke_failed")
    snapshot["smoke_progress"] = []
    for path in sorted((root / "smoke").glob("*/*/0/evaluation_progress.json"))[-4:]:
        report = load(path, {})
        snapshot["smoke_progress"].append(
            {key: report[key] for key in PROGRESS_FIELDS if numeric(report.get(key))}
        )
    snapshot["smoke_logs"] = []
    for path in sorted((root / "smoke").glob("*/*/epoch_*.log"))[-4:]:
        summary = tail_progress(path, current)
        if summary:
            snapshot["smoke_logs"].append(
                {"cell": label(path.parent.name), "epoch_log": path.name, **summary}
            )
            if summary["errors"]:
                issue(
                    "smoke_log:" + path.parent.name + ":" + path.stem,
                    "smoke_log_errors",
                    cell=label(path.parent.name),
                    labels=summary["errors"],
                )
    accepted = pipeline.get("accepted_staging", {})
    if accepted.get("transfer_reached_verification") is True:
        snapshot["superseded_staging"] = {
            "job_id": label(accepted.get("job_id")),
            "state": "SUPERSEDED_BY_FINAL_DATA_CHECK",
        }
    # Success is consumed from the verifying monitor, never inferred from Slurm alone.
    all_verified = (
        len(registry.get("runs", {})) == 36
        and len(monitored.get("runs", {})) == 36
        and all(run.get("state") == "COMPLETE" for run in monitored["runs"].values())
    )
    if (
        all_verified
        and monitored.get("state") == "COMPLETE"
        and not snapshot["scheduler_error"]
        and all(
            value["state"] == "COMPLETED" for value in snapshot["training"].values()
        )
        and len(snapshot["training"]) == 36
    ):
        snapshot["state"] = "COMPLETE"
    elif alerts:
        snapshot["state"] = "ATTENTION_REQUIRED"
    elif preflight_complete:
        snapshot["state"] = "PREFLIGHT_COMPLETE"
    if maintenance:
        snapshot["state"] = "PROMOTION_MAINTENANCE"
        for alert in alerts.values():
            alert["automatic_repair"] = False
    snapshot["alerts"] = alerts
    return snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--pipeline-pid", required=True, type=int)
    parser.add_argument("--monitor-pid", required=True, type=int)
    parser.add_argument("--interval", type=int, default=120)
    parser.add_argument("--progress-stale", type=int, default=7200)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    args.run_root, args.repo = args.run_root.resolve(), args.repo.resolve()
    if args.interval < 60 or not re.fullmatch("[0-9a-f]{40}", args.commit):
        parser.error("Use a full commit and interval >=60 seconds")
    operations = args.run_root / "operations"
    operations.mkdir(exist_ok=True)
    os.chmod(operations, 0o700)
    lock = (operations / "supervision.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("An operational supervisor already owns this run root.", flush=True)
        lock.close()
        return
    identities = {}
    for name, pid in (("pipeline", args.pipeline_pid), ("monitor", args.monitor_pid)):
        script = args.repo / "experiments/fold_clothes_moving_opscale" / (name + ".py")
        process = process_state(
            pid,
            script,
            args.run_root,
            commit=args.commit if name == "pipeline" else None,
        )
        identities[name] = process.get("start_ticks")
    save(
        operations / "supervisor_identity.json",
        {
            "commit": args.commit,
            "pipeline_pid": args.pipeline_pid,
            "monitor_pid": args.monitor_pid,
            "start_ticks": identities,
            "supervisor_pid": os.getpid(),
            "started_at": utc(),
        },
    )
    for filename in ("alerts.jsonl", "events.jsonl"):
        with (operations / filename).open("a"):
            os.chmod(operations / filename, 0o600)
    if not (operations / "active_alerts.json").exists():
        save(operations / "active_alerts.json", {})
    prior = load(operations / "supervision.json")
    active = load(operations / "active_alerts.json", {})
    while True:
        try:
            value = collect(args, identities)
        except Exception as error:
            value = {
                "state": "SUPERVISOR_ERROR",
                "updated_at": utc(),
                "heartbeat_epoch": time.time(),
                "supervisor_pid": os.getpid(),
                "error_type": type(error).__name__,
                "alerts": {
                    "supervisor": {
                        "kind": "supervisor_error",
                        "severity": "error",
                        "automatic_repair": False,
                    }
                },
            }
        save(operations / "supervision.json", value)
        changes = alert_changes(active, value["alerts"])
        if changes:
            with (operations / "alerts.jsonl").open("a") as stream:
                os.chmod(operations / "alerts.jsonl", 0o600)
                for event in changes:
                    stream.write(json.dumps({"timestamp": utc(), **event}) + "\n")
            save(operations / "active_alerts.json", value["alerts"])
        if prior is None or transition_view(prior) != transition_view(value):
            with (operations / "events.jsonl").open("a") as stream:
                os.chmod(operations / "events.jsonl", 0o600)
                stream.write(
                    json.dumps(
                        {
                            "timestamp": utc(),
                            "event": "transition",
                            **transition_view(value),
                        }
                    )
                    + "\n"
                )
            print(
                json.dumps(
                    {
                        "timestamp": utc(),
                        "state": value["state"],
                        "phase": value.get("phase"),
                        "active_alerts": len(value["alerts"]),
                    }
                ),
                flush=True,
            )
        active, prior = value["alerts"], value
        if args.once or value["state"] == "COMPLETE":
            lock.close()
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
