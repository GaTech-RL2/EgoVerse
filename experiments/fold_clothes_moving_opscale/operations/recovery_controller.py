#!/usr/bin/env python3
"""Locked incident coordination and opt-in recovery of known dead local workers.

Never submits/cancels Slurm jobs or edits experiment code. Local restarts resume
existing idempotent workers, which retain their own launch and checkpoint gates.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import socket
import subprocess
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

if __package__:
    from .authorization_scope import read_scope
    from .supervise import heartbeat_age, load, process_state, save, utc
else:
    from authorization_scope import read_scope
    from supervise import heartbeat_age, load, process_state, save, utc

COOLDOWN = 1800
MAX_ATTEMPTS = 3
POLICY = {
    "schema_version": 1,
    "scope_source": "run_root/production_control.json",
    "default_scope": "preflight_only",
    "initial_production_submission": "existing_pipeline_with_all_gates",
    "production_source_migration": "blocked",
    "allowed_automatic": [
        "inspect_logs_and_artifacts",
        "repair_operations_wrappers_with_tests",
        "restart_verified_dead_known_local_worker",
        "relaunch_same_configuration_after_verified_operational_fix",
        "resume_verified_full_state_checkpoint_with_same_wandb_id",
    ],
    "requires_review": [
        "scientific_design_change",
        "new_source_commit_or_dirty_training_source",
        "cancel_or_kill_healthy_or_live_stale_process",
        "ambiguous_submission",
        "unverified_resume",
        "delete_or_overwrite_remote_data",
        "credential_changes",
    ],
    "protected_invariants": [
        "36_budget_operator_cells",
        "split_membership_and_holdouts",
        "RDT300M_architecture",
        "train_batch64_effective64",
        "validation_batch32_limit800",
        "400000_steps_seed42026",
        "EMA_off_bf16_compile_on",
        "H100_or_H200_only",
        "canonical_representation_and_frame_counts",
        "pinned_clean_full_commit",
        "readiness_norm_evaluation_backend_gates",
    ],
    "max_agent_attempts_per_incident": MAX_ATTEMPTS,
    "cooldown_seconds": COOLDOWN,
    "max_parallel_repair_agents": 1,
    "automatic_local_restart_default": False,
}


def recovery_policy(root):
    authorization = read_scope(root)
    return {
        **POLICY,
        "scope": authorization["scope"],
        "allow_production_training": authorization["allow_training"],
        "authorization_scope": authorization,
    }


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


@contextmanager
def locked(root):
    operations = root / "operations"
    operations.mkdir(exist_ok=True)
    with (operations / "recovery.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield operations


def empty_state():
    return {
        "schema_version": 1,
        "incidents": {},
        "occurrences": {},
        "restart_attempts": {},
        "observations": {},
    }


def event(operations, kind, **fields):
    with (operations / "recovery_events.jsonl").open("a") as stream:
        os.chmod(operations / "recovery_events.jsonl", 0o600)
        stream.write(json.dumps({"timestamp": utc(), "event": kind, **fields}) + "\n")


def group_action_alerts(alerts, pipeline):
    """Group related symptoms by action + submitted Slurm array parent.

    Raw supervisor alerts remain unchanged. Member IDs/details are retained so
    operators and agents can inspect every failed task/cell in one incident.
    """
    actions = pipeline.get("actions", {})
    grouped = {}
    pending_pipeline = []

    def job_parent(value):
        match = re.fullmatch(r"(\d+)(?:_\d+)?", str(value or ""))
        return match.group(1) if match else None

    for alert_id, alert in alerts.items():
        if alert.get("automatic_repair") is False:
            grouped[alert_id] = alert
            continue
        if alert_id == "pipeline:blocked":
            pending_pipeline.append((alert_id, alert))
            continue
        action = alert.get("action")
        if not action and alert_id.startswith("norm:"):
            action = "normalization"
        if not action and (alert_id == "smoke" or alert_id.startswith("smoke_log:")):
            action = "evaluation-smoke"
        task_job = job_parent(alert.get("job_id"))
        action_job = (
            job_parent(actions.get(action, {}).get("job_id")) if action else None
        )
        # Do not silently join an explicit old allocation to a newer action.
        parent = task_job or action_job
        if not action or not parent:
            grouped[alert_id] = alert
            continue
        key = f"action:{action}:job:{parent}"
        group = grouped.setdefault(
            key,
            {
                "kind": "action_failure",
                "severity": "warning",
                "action": action,
                "job_id": parent,
                "member_alerts": {},
            },
        )
        group["member_alerts"][alert_id] = alert
        if alert.get("severity") == "error":
            group["severity"] = "error"
    action_groups = [
        key for key, value in grouped.items() if value.get("kind") == "action_failure"
    ]
    for alert_id, alert in pending_pipeline:
        if len(action_groups) == 1:
            grouped[action_groups[0]]["member_alerts"][alert_id] = alert
            if alert.get("severity") == "error":
                grouped[action_groups[0]]["severity"] = "error"
        elif action_groups:
            # A generic consequence must not acquire a separate repair budget.
            grouped[alert_id] = {
                **alert,
                "automatic_repair": False,
                "related_incidents": sorted(action_groups),
            }
        else:
            grouped[alert_id] = alert
    return grouped


def merge_existing_groups(state, alerts, now):
    """Preserve old repair budgets/receipts when migrating per-symptom incidents."""
    events = []
    for group_id, alert in alerts.items():
        members = set(alert.get("member_alerts", {}))
        if not members:
            continue
        previous = [
            i
            for i in state["incidents"].values()
            if i["state"] not in ("RESOLVED", "COALESCED")
            and (i["alert_id"] == group_id or i["alert_id"] in members)
        ]
        if not previous:
            continue
        # Preserve an existing owner's incident ID so renewal/finish stay valid.
        anchor = min(
            previous,
            key=lambda i: (
                i["state"] not in ("CLAIMED", "LEASE_EXPIRED"),
                i["alert_id"] != group_id,
                i["created_at"],
                i["id"],
            ),
        )
        if len(previous) == 1 and anchor["alert_id"] == group_id:
            continue
        anchor["attempts"] = sum(i.get("attempts", 0) for i in previous)
        anchor["available_after"] = max(i.get("available_after", 0) for i in previous)
        prior_states = {i["state"] for i in previous}
        anchor["source_alert_ids"] = sorted(
            set(anchor.get("source_alert_ids", []))
            | members
            | {i["alert_id"] for i in previous}
        )
        for other in previous:
            if other is anchor:
                continue
            # The existing global claim guard makes two owners impossible.
            require(
                other["state"] not in ("CLAIMED", "LEASE_EXPIRED"),
                "Multiple active owners require manual coalescing",
            )
            anchor["outcomes"].extend(
                {**outcome, "source_incident_id": other["id"]}
                for outcome in other.get("outcomes", [])
            )
            other.update(
                state="COALESCED",
                active=False,
                coalesced_into=anchor["id"],
                updated_at=now,
            )
            events.append(("incident_coalesced", other["id"]))
        anchor.update(alert_id=group_id, alert=alert, updated_at=now)
        if anchor["state"] not in ("CLAIMED", "LEASE_EXPIRED"):
            if anchor["attempts"] >= MAX_ATTEMPTS or "NEEDS_REVIEW" in prior_states:
                anchor["state"] = "NEEDS_REVIEW"
            elif "AWAITING_VERIFICATION" in prior_states:
                anchor["state"] = "AWAITING_VERIFICATION"
        state["occurrences"][group_id] = max(
            state["occurrences"].get(group_id, 0), anchor.get("occurrence", 1)
        )
        events.append(("incident_group_migrated", anchor["id"]))
    return events


def reconcile_incidents(state, alerts, now):
    events = merge_existing_groups(state, alerts, now)
    unresolved = {
        v["alert_id"]: v
        for v in state["incidents"].values()
        if v["state"] not in ("RESOLVED", "COALESCED")
    }
    for alert_id, alert in alerts.items():
        incident = unresolved.get(alert_id)
        if incident is None:
            occurrence = state["occurrences"].get(alert_id, 0) + 1
            state["occurrences"][alert_id] = occurrence
            identity = hashlib.sha256(f"{alert_id}:{occurrence}".encode()).hexdigest()[
                :16
            ]
            incident = {
                "id": identity,
                "alert_id": alert_id,
                "alert": alert,
                "state": "QUEUED",
                "active": True,
                "occurrence": occurrence,
                "created_at": now,
                "updated_at": now,
                "available_after": now,
                "attempts": 0,
                "claim": None,
                "outcomes": [],
            }
            state["incidents"][identity] = incident
            events.append(("incident_opened", identity))
        elif incident["alert"] != alert or not incident["active"]:
            incident.update(alert=alert, active=True, updated_at=now)
            events.append(("incident_evidence_changed", incident["id"]))
    for incident in state["incidents"].values():
        if incident["state"] == "COALESCED":
            continue
        active = incident["alert_id"] in alerts
        incident["active"] = active
        if not active and incident["state"] not in (
            "RESOLVED",
            "CLAIMED",
            "LEASE_EXPIRED",
        ):
            incident.update(state="RESOLVED", updated_at=now)
            events.append(("incident_resolved", incident["id"]))
        if incident["state"] == "CLAIMED" and incident["claim"]["expires_at"] <= now:
            incident.update(state="LEASE_EXPIRED", updated_at=now)
            events.append(("claim_expired_requires_reconciliation", incident["id"]))
        if (
            incident["state"] == "AWAITING_VERIFICATION"
            and active
            and now >= incident["available_after"]
        ):
            incident.update(
                state="NEEDS_REVIEW"
                if incident["attempts"] >= MAX_ATTEMPTS
                else "QUEUED",
                updated_at=now,
            )
            events.append(("fix_not_verified", incident["id"]))
    return events


def claim_incident(state, incident_id, owner, lease_seconds, now):
    require(re.fullmatch(r"[a-zA-Z0-9_.:-]{1,100}", owner), "Invalid owner label")
    require(60 <= lease_seconds <= 7200, "Lease must be between 60 and 7200 seconds")
    require(
        not any(
            v["state"] in ("CLAIMED", "LEASE_EXPIRED")
            for v in state["incidents"].values()
        ),
        "Another claim is active or requires reconciliation",
    )
    incident = state["incidents"][incident_id]
    require(
        incident["state"] == "QUEUED" and incident["active"],
        "Incident is not claimable",
    )
    require(now >= incident["available_after"], "Incident is cooling down")
    require(incident["attempts"] < MAX_ATTEMPTS, "Incident attempt limit reached")
    incident.update(
        state="CLAIMED",
        updated_at=now,
        attempts=incident["attempts"] + 1,
        claim={"owner": owner, "expires_at": now + lease_seconds, "claimed_at": now},
    )
    return incident


def owned_incident(state, incident_id, owner):
    incident = state["incidents"][incident_id]
    require(
        incident["state"] in ("CLAIMED", "LEASE_EXPIRED")
        and (incident.get("claim") or {}).get("owner") == owner,
        "Incident is not owned by this agent",
    )
    return incident


def finish_incident(state, incident_id, owner, outcome, receipt, now):
    incident = owned_incident(state, incident_id, owner)
    require(outcome in ("fixed", "blocked", "failed", "review"), "Invalid outcome")
    incident["outcomes"].append(
        {
            "outcome": outcome,
            "receipt": str(receipt),
            "owner": owner,
            "finished_at": now,
        }
    )
    incident.update(claim=None, updated_at=now, available_after=now + COOLDOWN)
    if not incident["active"]:
        incident["state"] = "RESOLVED"
    elif outcome == "fixed":
        incident["state"] = "AWAITING_VERIFICATION"
    elif outcome == "failed" and incident["attempts"] < MAX_ATTEMPTS:
        incident["state"] = "QUEUED"
    else:
        incident["state"] = "NEEDS_REVIEW"
    return incident


def restart_decision(process, report, attempts, observations, now, ambiguous=False):
    if process.get("alive") or process.get("state") not in ("MISSING", "EXITED"):
        return "DENY_LIVE_OR_UNVERIFIED_IDENTITY"
    if report.get("expected_exit"):
        return "DENY_EXPECTED_EXIT"
    if ambiguous:
        return "DENY_AMBIGUOUS_SUBMISSION"
    if observations.get("consecutive_dead", 0) < 2:
        return "WAIT_SECOND_DEAD_OBSERVATION"
    if any(x["state"] in ("STARTING", "UNCONFIRMED") for x in attempts):
        return "DENY_UNCONFIRMED_RESTART"
    recent = [x for x in attempts if now - x["created_at"] < 86400]
    if len(recent) >= MAX_ATTEMPTS:
        return "DENY_RESTART_LIMIT"
    if attempts and now - attempts[-1]["created_at"] < COOLDOWN:
        return "WAIT_COOLDOWN"
    return "READY"


def command_for(config, name):
    tool = (
        Path(config["repo"])
        / "experiments/fold_clothes_moving_opscale"
        / (name + ".py")
    )
    command = [config["python"], "-u", str(tool), "--run-root", config["run_root"]]
    if name == "pipeline":
        command += [
            "--commit",
            config["commit"],
            "--python",
            config["python"],
            "--gpu",
            config["gpu"],
            "--interval",
            "120",
        ]
    else:
        command += ["--interval", "120"]
    return command


def source_guard(config):
    def git(*args):
        result = subprocess.run(
            ["git", "-C", config["repo"], *args],
            text=True,
            capture_output=True,
            timeout=30,
        )
        require(result.returncode == 0, "Source identity query failed")
        return result.stdout.strip()

    require(git("rev-parse", "HEAD") == config["commit"], "Pinned source changed")
    require(not git("status", "--porcelain"), "Pinned source is dirty")


def current_targets(config, operations):
    targets = load(operations / "process_targets.json", {})
    if (
        targets.get("commit") == config["commit"]
        and targets.get("run_root") == config["run_root"]
    ):
        return targets.get("targets", config["initial_targets"])
    return config["initial_targets"]


def inspect_local(config, operations, state, now, apply):
    root = Path(config["run_root"])
    targets = current_targets(config, operations)
    pipeline = load(root / "pipeline.json", {})
    registry = load(root / "registry.json", {})
    ambiguous = any(
        a.get("state") in ("SUBMITTING", "SUBMISSION_UNCERTAIN")
        for a in pipeline.get("actions", {}).values()
    )
    ambiguous |= any(
        a.get("state")
        in ("SUBMITTING", "SUBMISSION_UNCERTAIN", "CONTINUATION_SUBMITTING")
        for a in registry.get("runs", {}).values()
    )
    result = {}
    for name in ("pipeline", "monitor"):
        target = targets[name]
        script = (
            Path(config["repo"])
            / "experiments/fold_clothes_moving_opscale"
            / (name + ".py")
        )
        process = process_state(
            target["pid"],
            script,
            root,
            commit=config["commit"] if name == "pipeline" else None,
            expected_ticks=target["start_ticks"],
        )
        report = load(
            root / ("pipeline_status.json" if name == "pipeline" else "status.json"), {}
        )
        if name == "pipeline":
            expected = report.get("state") in (
                "BLOCKED",
                "PREFLIGHT_COMPLETE",
            ) or report.get("phase") in ("SWEEP_SUBMITTED", "PREFLIGHT_COMPLETE")
            expected |= (
                report.get("commit") != config["commit"]
                or pipeline.get("commit") != config["commit"]
            )
        else:
            expected = report.get("state") in (
                "COMPLETE",
                "TERMINAL_REVIEW_REQUIRED",
                "SUBSET_COMPLETE",
                "PREFLIGHT_COMPLETE",
            )
        observation = state["observations"].setdefault(name, {"consecutive_dead": 0})
        observation["consecutive_dead"] = (
            observation["consecutive_dead"] + 1
            if process["state"] in ("MISSING", "EXITED")
            else 0
        )
        attempts = state["restart_attempts"].setdefault(name, [])
        decision = restart_decision(
            process, {"expected_exit": expected}, attempts, observation, now, ambiguous
        )
        result[name] = {
            "pid": target["pid"],
            "state": process["state"],
            "restart_decision": decision,
        }
        if any(
            v["state"] in ("CLAIMED", "LEASE_EXPIRED")
            for v in state["incidents"].values()
        ):
            decision = "DENY_ACTIVE_AGENT_CLAIM"
            result[name]["restart_decision"] = decision
        if not apply or decision != "READY":
            continue
        try:
            source_guard(config)
        except RuntimeError:
            result[name]["restart_decision"] = "DENY_SOURCE_IDENTITY"
            continue
        # Take the existing worker lock before checking again, then release it for
        # the new child. Both workers also lock themselves, preventing duplicates.
        with (root / (name + ".lock")).open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                result[name]["restart_decision"] = "DENY_WORKER_LOCK_HELD"
                continue
            again = process_state(
                target["pid"], script, root, expected_ticks=target["start_ticks"]
            )
            require(
                not again["alive"] and again["state"] in ("MISSING", "EXITED"),
                "Worker identity changed during restart check",
            )
        attempt = {
            "id": uuid.uuid4().hex,
            "created_at": now,
            "state": "STARTING",
            "old_pid": target["pid"],
        }
        attempts.append(attempt)
        save(operations / "recovery_state.json", state)
        directory = operations / "restarts"
        directory.mkdir(exist_ok=True)
        env = dict(os.environ)
        env.update(
            PYTHONDONTWRITEBYTECODE="1",
            EGOVERSE_MECKA_PRIVATE_DIR=str(root / "private"),
            EGOVERSE_ROOT=config["repo"],
        )
        env["PYTHONPATH"] = os.pathsep.join(
            [config["repo"], str(root / "python-deps"), env.get("PYTHONPATH", "")]
        )
        env["PATH"] = str(Path.home() / ".local/bin") + os.pathsep + env.get("PATH", "")
        with (directory / (name + "-" + attempt["id"] + ".log")).open("ab") as log:
            child = subprocess.Popen(
                command_for(config, name),
                cwd=config["repo"],
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        attempt["pid"] = child.pid
        save(operations / "recovery_state.json", state)
        verified = None
        for unused in range(20):
            time.sleep(0.1)
            candidate = process_state(
                child.pid,
                script,
                root,
                commit=config["commit"] if name == "pipeline" else None,
            )
            if candidate["state"] == "RUNNING":
                verified = candidate
                break
            if child.poll() is not None:
                break
        if verified is None:
            attempt["state"] = "UNCONFIRMED"
            result[name]["restart_decision"] = "RESTART_UNCONFIRMED_REVIEW_REQUIRED"
        else:
            attempt.update(state="VERIFIED", start_ticks=verified["start_ticks"])
            targets[name] = {
                "pid": child.pid,
                "start_ticks": verified["start_ticks"],
                "restart_id": attempt["id"],
            }
            save(operations / "recovery_state.json", state)
            save(
                operations / "process_targets.json",
                {
                    "schema_version": 1,
                    "run_root": config["run_root"],
                    "commit": config["commit"],
                    "targets": targets,
                    "updated_at": utc(),
                },
            )
            save(root / (name + ".pid"), child.pid)
            result[name].update(
                pid=child.pid, state="RUNNING", restart_decision="RESTARTED"
            )
            event(
                operations,
                "local_worker_restarted",
                process=name,
                pid=child.pid,
                attempt_id=attempt["id"],
            )
        save(operations / "recovery_state.json", state)
    return result


def watch_identity():
    fields = (
        (Path("/proc") / str(os.getpid()) / "stat")
        .read_text()
        .rsplit(")", 1)[1]
        .split()
    )
    return {
        "pid": os.getpid(),
        "start_ticks": int(fields[19]),
        "pgid": int(fields[2]),
        "host": socket.gethostname(),
    }


def reconcile(root, apply=False):
    with locked(root) as operations:
        maintenance = load(operations / "promotion_maintenance.json", {})
        if maintenance.get("active"):
            valid_deadline = (
                isinstance(maintenance.get("deadline_epoch"), (int, float))
                and time.time() <= maintenance["deadline_epoch"]
            )
            return {
                "state": "MAINTENANCE" if valid_deadline else "BLOCKED",
                "error_type": None
                if valid_deadline
                else "PromotionMaintenanceExpiredOrInvalid",
                "transaction_id": maintenance.get("transaction_id"),
                "incidents": {},
                "local_processes": {},
            }

        config = load(operations / "recovery_config.json")
        require(config, "Run recovery init first")
        state = load(operations / "recovery_state.json", empty_state())
        snapshot = load(operations / "supervision.json", {})
        alerts = dict(snapshot.get("alerts", {}))
        now = time.time()
        age = heartbeat_age(snapshot, now)
        if snapshot.get("state") != "COMPLETE" and (age is None or age > 600):
            alerts["supervisor:stale"] = {
                "kind": "supervisor_heartbeat_stale",
                "severity": "error",
                "automatic_repair": False,
            }
        alerts = group_action_alerts(alerts, load(root / "pipeline.json", {}))
        for kind, identity in reconcile_incidents(state, alerts, now):
            event(operations, kind, incident_id=identity)
        policy = recovery_policy(root)
        save(operations / "recovery_policy.json", policy)
        local = inspect_local(
            config,
            operations,
            state,
            now,
            apply and policy["authorization_scope"]["valid"],
        )
        state.update(
            updated_at=utc(),
            heartbeat_epoch=now,
            controller_pid=os.getpid(),
            local_processes=local,
        )
        save(operations / "recovery_state.json", state)
        return {
            "state": "COMPLETE"
            if snapshot.get("state") == "COMPLETE"
            else "MONITORING",
            "incidents": {
                key: value["state"] for key, value in state["incidents"].items()
            },
            "local_processes": local,
        }


def initialize(args):
    root = args.run_root.resolve()
    require(re.fullmatch("[0-9a-f]{40}", args.commit), "Full source commit required")
    config = {
        "schema_version": 1,
        "run_root": str(root),
        "repo": str(args.repo.resolve()),
        "commit": args.commit,
        "python": str(args.python.absolute()),
        "gpu": args.gpu,
        "initial_targets": {},
    }
    source_guard(config)
    for name, pid in (("pipeline", args.pipeline_pid), ("monitor", args.monitor_pid)):
        script = (
            args.repo.resolve()
            / "experiments/fold_clothes_moving_opscale"
            / (name + ".py")
        )
        process = process_state(
            pid, script, root, commit=args.commit if name == "pipeline" else None
        )
        require(process["state"] == "RUNNING", "Initial worker identity did not match")
        config["initial_targets"][name] = {
            "pid": pid,
            "start_ticks": process["start_ticks"],
        }
    with locked(root) as operations:
        prior = load(operations / "recovery_config.json")
        require(
            prior is None or prior == config,
            "Recovery configuration already exists with different identity",
        )
        save(operations / "recovery_config.json", config)
        save(operations / "recovery_policy.json", recovery_policy(root))
        if not (operations / "recovery_state.json").exists():
            save(operations / "recovery_state.json", empty_state())
    return {"initialized": True, "automatic_restart_enabled": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("init", "reconcile", "watch", "claim", "renew", "finish"):
        item = sub.add_parser(name)
        item.add_argument("--run-root", required=True, type=Path)
        if name == "init":
            item.add_argument("--repo", required=True, type=Path)
            item.add_argument("--commit", required=True)
            item.add_argument("--python", required=True, type=Path)
            item.add_argument("--gpu", choices=("h100", "h200"), default="h200")
            item.add_argument("--pipeline-pid", required=True, type=int)
            item.add_argument("--monitor-pid", required=True, type=int)
        if name in ("watch", "reconcile"):
            item.add_argument("--allow-local-restart", action="store_true")
        if name == "watch":
            item.add_argument("--interval", type=int, default=120)
        if name in ("claim", "renew", "finish"):
            item.add_argument("--incident", required=True)
            item.add_argument("--owner", required=True)
        if name in ("claim", "renew"):
            item.add_argument("--lease-seconds", type=int, default=1800)
        if name == "finish":
            item.add_argument(
                "--outcome",
                required=True,
                choices=("fixed", "blocked", "failed", "review"),
            )
            item.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    args.run_root = args.run_root.resolve()
    if args.command == "init":
        result = initialize(args)
    elif args.command in ("watch", "reconcile"):
        require(
            getattr(args, "interval", 120) >= 60,
            "Poll interval must be at least 60 seconds",
        )
        watch_lock = None
        if args.command == "watch":
            watch_lock = (args.run_root / "operations/recovery_watch.lock").open("a")
            fcntl.flock(watch_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = None
        while True:
            try:
                result = reconcile(args.run_root, args.allow_local_restart)
                if args.command == "watch":
                    save(
                        args.run_root / "operations/recovery_watch_status.json",
                        {
                            "identity": watch_identity(),
                            "updated_at": utc(),
                            "heartbeat_epoch": time.time(),
                            "state": result["state"]
                            if result["state"] in ("COMPLETE", "BLOCKED")
                            else "RUNNING",
                            "phase": result["state"],
                        },
                    )
            except Exception as error:
                if args.command != "watch":
                    raise
                save(
                    args.run_root / "operations/recovery_watch_status.json",
                    {
                        "identity": watch_identity(),
                        "updated_at": utc(),
                        "heartbeat_epoch": time.time(),
                        "state": "BLOCKED",
                        "error_type": type(error).__name__,
                    },
                )
                result = {"state": "BLOCKED", "error_type": type(error).__name__}
            if result != previous:
                print(json.dumps(result), flush=True)
            previous = result
            if args.command == "reconcile" or result["state"] == "COMPLETE":
                break
            time.sleep(args.interval)
        if watch_lock:
            watch_lock.close()
        return
    else:
        with locked(args.run_root) as operations:
            state = load(operations / "recovery_state.json", empty_state())
            now = time.time()
            if args.command == "claim":
                result = claim_incident(
                    state, args.incident, args.owner, args.lease_seconds, now
                )
            elif args.command == "renew":
                require(60 <= args.lease_seconds <= 7200, "Invalid lease duration")
                result = owned_incident(state, args.incident, args.owner)
                require(
                    result["state"] == "CLAIMED"
                    and result["claim"]["expires_at"] > now,
                    "Expired claims require reconciliation",
                )
                result["claim"]["expires_at"] = now + args.lease_seconds
            else:
                receipt = args.receipt.resolve()
                require(
                    receipt.is_relative_to(operations.resolve()) and receipt.is_file(),
                    "Receipt must be an existing operations artifact",
                )
                result = finish_incident(
                    state, args.incident, args.owner, args.outcome, receipt, now
                )
            save(operations / "recovery_state.json", state)
            event(operations, args.command, incident_id=args.incident, owner=args.owner)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, KeyError, OSError, ValueError) as error:
        # All planned messages are fixed labels; never expose child tool stderr.
        print(
            json.dumps(
                {
                    "error_type": type(error).__name__,
                    "error": str(error)
                    if isinstance(error, RuntimeError)
                    else "Controller operation failed",
                }
            ),
            flush=True,
        )
        raise SystemExit(2)
