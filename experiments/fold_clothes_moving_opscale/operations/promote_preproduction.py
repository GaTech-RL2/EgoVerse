#!/usr/bin/env python3
"""Reviewed, resumable source promotion before any production registry exists.

Never checks out/reset/modifies the old source, cancels Slurm, or invents passing
proofs. A clean separate worktree is adopted only after exact local quiescence.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import signal
import socket
import subprocess
import time
import uuid
from pathlib import Path

if __package__:
    from .recovery_controller import current_targets, require, source_guard
    from .supervise import (
        TERMINAL,
        job_summary,
        load,
        parse_slurm,
        process_state,
        run_command,
        utc,
    )
else:
    from recovery_controller import current_targets, require, source_guard
    from supervise import (
        TERMINAL,
        job_summary,
        load,
        parse_slurm,
        process_state,
        run_command,
        utc,
    )

ALLOWED_SOURCE = {
    f"experiments/fold_clothes_moving_opscale/{name}.py"
    for name in (
        "canonicalize",
        "check_readiness",
        "launch_pace",
        "monitor",
        "pipeline",
        "run_preflight",
        "validate_setup",
    )
}
ARCHIVE = (
    "pipeline.json",
    "pipeline_status.json",
    "status.json",
    "readiness.json",
    "gate.json",
    "evaluation_smoke.json",
    "norm",
    "smoke",
    "private/canonical_zarr",
    "private/canonical_provenance",
    "private/canonical_tmp",
    "operations/process_targets.json",
    "operations/supervisor_identity.json",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
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


def command(argv):
    proc = subprocess.run(argv, text=True, capture_output=True, timeout=45)
    require(proc.returncode == 0, "Read-only identity/accounting command failed")
    return proc.stdout.strip()


def science_files(repo):
    repo = Path(repo)
    paths = set()
    for folder in (
        "egomimic/hydra_configs",
        "egomimic/models",
        "egomimic/algo",
        "egomimic/rldb/zarr",
    ):
        paths.update(
            p
            for p in (repo / folder).rglob("*")
            if p.is_file() and "__pycache__" not in p.parts
        )
    paths.update((repo / "experiments/fold_clothes_moving_opscale").glob("**/*.yaml"))
    paths.update((repo / "experiments/fold_clothes_moving_opscale").glob("**/*.yml"))
    return {str(p.relative_to(repo)): sha(p) for p in sorted(paths)}


def private_files(root):
    private = root / "private"
    paths = [
        private / "moving_opscale_design.json",
        private / "inputs/main_episodes.csv",
        private / "recovery2/report.json",
    ]
    splits = sorted((private / "splits").glob("fold_moving_b*_op*.json"))
    require(len(splits) == 36, "All36 fixed split files required")
    paths += splits
    return {str(p.relative_to(root)): sha(p) for p in paths}


def review_candidate(config, new_repo, new_commit, receipt):
    old_repo = Path(config["repo"])
    new_repo = Path(new_repo).resolve()
    require(
        new_repo != old_repo.resolve(), "Candidate must be a separate clean worktree"
    )
    require(
        not new_repo.is_relative_to(Path(config["run_root"]).parent),
        "Candidate worktree cannot live under egoverse-runs",
    )
    require(re.fullmatch("[0-9a-f]{40}", new_commit), "Full candidate commit required")
    source_guard(config)
    source_guard({**config, "repo": str(new_repo), "commit": new_commit})
    command(
        [
            "git",
            "-C",
            str(new_repo),
            "merge-base",
            "--is-ancestor",
            config["commit"],
            new_commit,
        ]
    )
    review = load(receipt)
    require(
        review
        and review.get("outcome") == "approved"
        and all(
            review.get(k) is True
            for k in ("verified", "science_unchanged", "tests_passed")
        ),
        "Independent approval with passing tests required",
    )
    require(
        review.get("candidate_repo") == str(new_repo)
        and review.get("candidate_commit") == new_commit,
        "Independent review must attest exact candidate worktree and SHA",
    )
    changed = {}
    for line in command(
        [
            "git",
            "-C",
            str(new_repo),
            "diff",
            "--name-status",
            config["commit"],
            new_commit,
        ]
    ).splitlines():
        fields = line.split("\t")
        require(
            len(fields) == 2 and fields[0] in ("A", "M"),
            "Deletion/rename/type changes require separate review",
        )
        rel = fields[1]
        require(
            rel in ALLOWED_SOURCE or rel.startswith("tests/"),
            "Source change is outside operational repair boundary",
        )
        changed[rel] = sha(new_repo / rel)
    require(bool(changed), "Candidate has no changes")
    approved = {}
    for item in review.get("candidate_files", []):
        dest = Path(item["destination"]).resolve()
        base = new_repo if dest.is_relative_to(new_repo) else old_repo.resolve()
        require(
            dest.is_relative_to(base),
            "Reviewed destination is outside source worktrees",
        )
        rel = str(dest.relative_to(base))
        require(rel not in approved, "Duplicate reviewed destination")
        require(sha(item["path"]) == item["sha256"], "Reviewed candidate bytes changed")
        approved[rel] = item["sha256"]
    require(
        approved == changed,
        "Clean candidate diff does not exactly match reviewed files",
    )
    frozen = science_files(old_repo)
    require(
        frozen == science_files(new_repo),
        "Scientific implementation/configuration changed",
    )
    return changed, frozen


def recorded_jobs(root, pipeline=None):
    require(
        not (root / "registry.json").exists(),
        "Production registry exists; preproduction promotion is forbidden",
    )
    pipeline = load(root / "pipeline.json", {}) if pipeline is None else pipeline
    jobs = {}
    for name, action in pipeline.get("actions", {}).items():
        require(
            action.get("state") not in ("SUBMITTING", "SUBMISSION_UNCERTAIN"),
            "Ambiguous prior submission",
        )
        if action.get("job_id"):
            require(str(action["job_id"]).isdigit(), "Invalid job identity")
            jobs[str(action["job_id"])] = int(action.get("array_count", 0))
    stage = load(root / "staging.json", {})
    for job in [stage.get("job_id"), *stage.get("recovery_job_ids", [])]:
        if job:
            require(str(job).isdigit(), "Invalid staging job identity")
            jobs[str(job)] = 0
    return jobs


def terminal_jobs(jobs):
    if not jobs:
        return {}
    ids = ",".join(sorted(jobs))
    rows = parse_slurm(
        command(
            [
                "sacct",
                "-n",
                "-X",
                "-P",
                "-j",
                ids,
                "--format=JobID%60,State%40,ExitCode",
            ]
        ),
        run_command(["squeue", "--array", "-h", "-j", ids, "-o", "%i|%T"]),
    )
    result = {}
    for job, count in jobs.items():
        value = job_summary(job, count, rows)
        require(
            all(row["state"] in TERMINAL for row in value["tasks"].values()),
            "A preparation job is active, pending, or unaccounted",
        )
        result[job] = value["tasks"]
    return result


def worker_specs(config, ops):
    root = Path(config["run_root"])
    targets = current_targets(config, ops)
    specs = {}
    for name in ("pipeline", "monitor"):
        specs[name] = {
            **targets[name],
            "script": str(
                Path(config["repo"])
                / "experiments/fold_clothes_moving_opscale"
                / f"{name}.py"
            ),
            "commit": config["commit"] if name == "pipeline" else None,
        }
    supervisor = load(ops / "supervisor_identity.json", {})
    require(
        supervisor.get("commit") == config["commit"],
        "Supervisor source identity differs",
    )
    pid = supervisor.get("supervisor_pid")
    require(isinstance(pid, int), "Supervisor PID missing")
    status = process_state(pid, ops / "supervise.py", root, commit=config["commit"])
    require(
        status["state"] == "RUNNING",
        "Live supervisor identity required for planned quiescence",
    )
    specs["supervisor"] = {
        "pid": pid,
        "start_ticks": status["start_ticks"],
        "script": str(ops / "supervise.py"),
        "commit": config["commit"],
    }
    for spec in specs.values():
        status = process_state(
            spec["pid"],
            Path(spec["script"]),
            root,
            commit=spec["commit"],
            expected_ticks=spec["start_ticks"],
        )
        require(
            status["state"] in ("RUNNING", "MISSING", "EXITED"),
            "Worker identity cannot be verified",
        )
    return specs


def make_plan(args):
    root = args.run_root.resolve()
    ops = root / "operations"
    config = load(ops / "recovery_config.json")
    require(
        config and config["run_root"] == str(root),
        "Initialized recovery identity required",
    )
    require(
        not load(ops / "promotion_maintenance.json", {}).get("active"),
        "Another promotion is active",
    )
    require(
        args.review_receipt.resolve().is_relative_to(ops),
        "Review receipt must be an operations artifact",
    )
    changed, frozen = review_candidate(
        config, args.candidate_repo, args.candidate_commit, args.review_receipt
    )
    pipeline = load(root / "pipeline.json", {})
    require(pipeline.get("commit") == config["commit"], "Pipeline source differs")
    require(
        pipeline.get("accepted_staging", {}).get("transfer_reached_verification")
        is True,
        "Verified transfer acceptance required",
    )
    jobs = recorded_jobs(root, pipeline)
    states = terminal_jobs(jobs)
    return {
        "schema_version": 1,
        "transaction_id": uuid.uuid4().hex,
        "created_at": utc(),
        "run_root": str(root),
        "old_repo": config["repo"],
        "old_commit": config["commit"],
        "new_repo": str(args.candidate_repo.resolve()),
        "new_commit": args.candidate_commit,
        "review_receipt": str(args.review_receipt.resolve()),
        "review_receipt_sha256": sha(args.review_receipt),
        "recovery_config": config,
        "recovery_config_sha256": sha(ops / "recovery_config.json"),
        "pipeline_sha256": sha(root / "pipeline.json"),
        "pipeline": pipeline,
        "reviewed_files": changed,
        "scientific_files": frozen,
        "private_files": private_files(root),
        "jobs": jobs,
        "terminal_job_states": states,
        "workers": worker_specs(config, ops),
        "archive_paths": list(ARCHIVE),
        "required_rebuilds": [
            "canonical_view",
            "readiness",
            "all36_normalization",
            "GPU_smoke",
            "launch_gate",
        ],
    }


def stop_exact(spec, root):
    def inspect():
        return process_state(
            spec["pid"],
            Path(spec["script"]),
            root,
            commit=spec["commit"],
            expected_ticks=spec["start_ticks"],
        )

    state = inspect()
    require(
        state["state"] in ("RUNNING", "MISSING", "EXITED"),
        "Process identity changed; refusing signal",
    )
    if state["state"] != "RUNNING":
        return
    os.kill(spec["pid"], signal.SIGTERM)
    for unused in range(100):
        time.sleep(0.1)
        state = inspect()
        if state["state"] in ("MISSING", "EXITED"):
            return
        require(state["state"] == "RUNNING", "Process identity changed while stopping")
    raise RuntimeError("Worker did not stop after SIGTERM; no forced kill performed")


def archive_one(root, archive, relative):
    source = root / relative
    dest = archive / relative
    require(
        not (source.exists() and dest.exists()),
        "Archive conflict requires reconciliation",
    )
    if source.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        source.rename(dest)


def spawn_known(tx, path, name, argv, script, commit, root):
    workers = tx.setdefault("new_workers", {})
    existing = workers.get(name)
    if existing:
        require(
            isinstance(existing.get("pid"), int)
            and isinstance(existing.get("start_ticks"), int),
            "Unconfirmed spawn identity requires reconciliation; no duplicate",
        )
        observed = process_state(
            existing["pid"],
            script,
            root,
            commit=commit,
            expected_ticks=existing.get("start_ticks"),
        )
        require(
            observed["state"] == "RUNNING",
            "Prior replacement worker is not verified alive; reconcile before retry",
        )
        existing.update(state="VERIFIED", start_ticks=observed["start_ticks"])
        save(path, tx)
        return existing
    workers[name] = {"state": "STARTING", "created_at": utc()}
    save(path, tx)
    env = dict(os.environ)
    env.update(
        PYTHONDONTWRITEBYTECODE="1",
        EGOVERSE_MECKA_PRIVATE_DIR=str(root / "private"),
        EGOVERSE_ROOT=tx["plan"]["new_repo"],
    )
    env["PYTHONPATH"] = os.pathsep.join(
        [tx["plan"]["new_repo"], str(root / "python-deps"), env.get("PYTHONPATH", "")]
    )
    with (path.parent / f"{name}.log").open("ab") as log:
        child = subprocess.Popen(
            argv,
            cwd=tx["plan"]["new_repo"],
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
    workers[name]["pid"] = child.pid
    try:
        fields = (
            (Path("/proc") / str(child.pid) / "stat")
            .read_text()
            .rsplit(")", 1)[1]
            .split()
        )
        workers[name]["start_ticks"] = int(fields[19])
    except (OSError, IndexError, ValueError):
        save(path, tx)
        raise RuntimeError("Spawn start identity unavailable; reconcile before retry")
    save(path, tx)
    for unused in range(30):
        time.sleep(0.1)
        observed = process_state(
            child.pid,
            script,
            root,
            commit=commit,
            expected_ticks=workers[name]["start_ticks"],
        )
        if observed["state"] == "RUNNING":
            workers[name].update(state="VERIFIED", start_ticks=observed["start_ticks"])
            save(path, tx)
            return workers[name]
        if child.poll() is not None:
            break
    raise RuntimeError("Replacement worker did not acquire verified process identity")


def validate_evidence(plan):
    root = Path(plan["run_root"])
    require(
        private_files(root) == plan["private_files"],
        "Private scientific inputs changed",
    )
    require(
        sha(plan["review_receipt"]) == plan["review_receipt_sha256"],
        "Review receipt changed",
    )
    changed, frozen = review_candidate(
        plan["recovery_config"],
        Path(plan["new_repo"]),
        plan["new_commit"],
        Path(plan["review_receipt"]),
    )
    require(
        changed == plan["reviewed_files"] and frozen == plan["scientific_files"],
        "Candidate or scientific source changed",
    )


def update_manifest(root, plan, workers):
    path = root / "run_manifest.json"
    manifest = load(path, {})
    require(
        manifest.get("commit") in (None, plan["old_commit"], plan["new_commit"]),
        "Run manifest source differs",
    )
    manifest.update(
        repository=plan["new_repo"],
        commit=plan["new_commit"],
        pipeline_pid=workers["pipeline"]["pid"],
        monitor_pid=workers["monitor"]["pid"],
        supervisor_pid=workers["supervisor"]["pid"],
        updated_at=utc(),
        status="PREFLIGHT_ACTIVE",
        pipeline_phase="REBUILDING_AFTER_SOURCE_PROMOTION",
    )
    monitoring = manifest.setdefault("monitoring", {})
    monitoring.update(
        pipeline_pid=workers["pipeline"]["pid"],
        run_monitor_pid=workers["monitor"]["pid"],
        supervisor_pid=workers["supervisor"]["pid"],
    )
    manifest["source_promotion_receipt"] = str(
        root / "operations/promotions" / plan["transaction_id"] / "complete.json"
    )
    save(path, manifest)
    save(root / "supervisor.pid", workers["supervisor"]["pid"])


def validate_plan_structure(plan, plan_path):
    root = Path(plan["run_root"])
    require(
        plan.get("schema_version") == 1
        and root.is_absolute()
        and root == root.resolve(),
        "Invalid plan schema/root",
    )
    require(
        re.fullmatch("[0-9a-f]{32}", str(plan.get("transaction_id", ""))),
        "Invalid transaction identity",
    )
    require(plan.get("archive_paths") == list(ARCHIVE), "Archive allowlist changed")
    require(
        plan_path.resolve().is_relative_to(root / "operations"),
        "Plan must be an operations artifact",
    )
    baseline = plan.get("recovery_config", {})
    require(
        baseline.get("run_root") == str(root)
        and baseline.get("repo") == plan.get("old_repo")
        and baseline.get("commit") == plan.get("old_commit"),
        "Plan baseline recovery identity differs",
    )
    actual = load(root / "operations/recovery_config.json", {})
    require(
        actual.get("run_root") == str(root)
        and (actual.get("repo"), actual.get("commit"))
        in [
            (plan.get("old_repo"), plan.get("old_commit")),
            (plan.get("new_repo"), plan.get("new_commit")),
        ],
        "Current recovery identity is outside planned migration",
    )


def apply_plan(plan_path):
    plan_path = plan_path.resolve()
    plan = load(plan_path)
    validate_plan_structure(plan, plan_path)
    root = Path(plan["run_root"])
    ops = root / "operations"
    require(plan_path.is_relative_to(ops), "Plan must be stored under operations")
    with (ops / "promotion.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        folder = ops / "promotions" / plan["transaction_id"]
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "transaction.json"
        tx = load(path)
        if tx is None:
            require(
                not load(ops / "promotion_maintenance.json", {}).get("active"),
                "Another promotion is active",
            )
            require(
                sha(ops / "recovery_config.json") == plan["recovery_config_sha256"],
                "Recovery identity changed since plan",
            )
            require(
                sha(root / "pipeline.json") == plan["pipeline_sha256"],
                "Pipeline actions changed since plan",
            )
            validate_evidence(plan)
            require(
                recorded_jobs(root) == plan["jobs"], "Recorded jobs changed since plan"
            )
            terminal_jobs(plan["jobs"])
            tx = {
                "schema_version": 1,
                "state": "STARTED",
                "plan": plan,
                "plan_sha256": sha(plan_path),
                "steps": {},
                "started_at": utc(),
                "deadline_epoch": time.time() + 3600,
            }
            save(path, tx)
        require(
            tx["plan_sha256"] == sha(plan_path),
            "Promotion plan changed during transaction",
        )
        if tx["state"] == "COMPLETE":
            receipt = load(folder / "complete.json")
            require(
                receipt and receipt["transaction_id"] == plan["transaction_id"],
                "Completed transaction receipt missing",
            )
            maintenance = load(ops / "promotion_maintenance.json", {})
            if (
                maintenance.get("active")
                and maintenance.get("transaction_id") == plan["transaction_id"]
            ):
                save(
                    ops / "promotion_maintenance.json",
                    {
                        "active": False,
                        "transaction_id": plan["transaction_id"],
                        "completed_at": receipt["completed_at"],
                    },
                )
            return receipt
        fields = (
            (Path("/proc") / str(os.getpid()) / "stat")
            .read_text()
            .rsplit(")", 1)[1]
            .split()
        )
        owner = {
            "pid": os.getpid(),
            "start_ticks": int(fields[19]),
            "pgid": int(fields[2]),
            "host": socket.gethostname(),
        }
        save(
            ops / "promotion_maintenance.json",
            {
                "active": True,
                "transaction_id": plan["transaction_id"],
                "plan_path": str(plan_path),
                "started_at": tx["started_at"],
                "deadline_epoch": tx["deadline_epoch"],
                "owner_identity": owner,
                **{
                    k: plan[k]
                    for k in ("old_repo", "old_commit", "new_repo", "new_commit")
                },
            },
        )
        validate_evidence(plan)
        require(
            time.time() <= tx["deadline_epoch"],
            "Promotion deadline expired; reconcile before continuing",
        )
        require(
            not (root / "registry.json").exists(),
            "Production registry appeared; stop promotion",
        )
        require(
            private_files(root) == plan["private_files"],
            "Scientific inputs changed during promotion",
        )
        source_guard(
            {
                **plan["recovery_config"],
                "repo": plan["new_repo"],
                "commit": plan["new_commit"],
            }
        )
        if not tx["steps"].get("quiesced"):
            for name in ("pipeline", "monitor", "supervisor"):
                stop_exact(plan["workers"][name], root)
            require(
                recorded_jobs(root) == plan["jobs"],
                "A job submission raced quiescence; reconcile first",
            )
            terminal_jobs(plan["jobs"])
            tx["steps"]["quiesced"] = True
            save(path, tx)
        if not tx["steps"].get("archived"):
            for relative in plan["archive_paths"]:
                archive_one(root, folder / "archive", relative)
            tx["steps"]["archived"] = True
            save(path, tx)
        if not tx["steps"].get("pipeline_reset"):
            save(
                root / "pipeline.json",
                {
                    "schema_version": 1,
                    "created_at": utc(),
                    "commit": plan["new_commit"],
                    "design_sha256": plan["pipeline"]["design_sha256"],
                    "accepted_staging": plan["pipeline"]["accepted_staging"],
                    "actions": {},
                    "promotion_transaction": plan["transaction_id"],
                },
            )
            tx["steps"]["pipeline_reset"] = True
            save(path, tx)
        cfg = plan["recovery_config"]
        python = cfg["python"]
        tools = Path(plan["new_repo"]) / "experiments/fold_clothes_moving_opscale"
        monitor = spawn_known(
            tx,
            path,
            "monitor",
            [
                python,
                "-u",
                str(tools / "monitor.py"),
                "--run-root",
                str(root),
                "--interval",
                "120",
            ],
            tools / "monitor.py",
            None,
            root,
        )
        pipeline = spawn_known(
            tx,
            path,
            "pipeline",
            [
                python,
                "-u",
                str(tools / "pipeline.py"),
                "--run-root",
                str(root),
                "--commit",
                plan["new_commit"],
                "--python",
                python,
                "--gpu",
                cfg["gpu"],
                "--interval",
                "120",
            ],
            tools / "pipeline.py",
            plan["new_commit"],
            root,
        )
        if not tx["steps"].get("identity_published"):
            new_config = {
                **cfg,
                "repo": plan["new_repo"],
                "commit": plan["new_commit"],
                "initial_targets": {
                    name: {"pid": worker["pid"], "start_ticks": worker["start_ticks"]}
                    for name, worker in [("pipeline", pipeline), ("monitor", monitor)]
                },
            }
            save(ops / "recovery_config.json", new_config)
            for name, worker in [("pipeline", pipeline), ("monitor", monitor)]:
                save(root / f"{name}.pid", worker["pid"])
            tx["steps"]["identity_published"] = True
            save(path, tx)
        spawn_known(
            tx,
            path,
            "supervisor",
            [
                python,
                "-u",
                str(ops / "supervise.py"),
                "--run-root",
                str(root),
                "--repo",
                plan["new_repo"],
                "--commit",
                plan["new_commit"],
                "--pipeline-pid",
                str(pipeline["pid"]),
                "--monitor-pid",
                str(monitor["pid"]),
                "--interval",
                "120",
            ],
            ops / "supervise.py",
            plan["new_commit"],
            root,
        )
        update_manifest(root, plan, tx["new_workers"])
        receipt = {
            "state": "COMPLETE",
            "transaction_id": plan["transaction_id"],
            "plan_sha256": tx["plan_sha256"],
            "review_receipt_sha256": plan["review_receipt_sha256"],
            **{
                k: plan[k] for k in ("old_repo", "old_commit", "new_repo", "new_commit")
            },
            "recovery_config_sha256": sha(ops / "recovery_config.json"),
            "new_processes": tx["new_workers"],
            "completed_at": utc(),
        }
        save(folder / "complete.json", receipt)
        save(ops / "promotion_complete.json", receipt)
        tx["state"] = "COMPLETE"
        save(path, tx)
        save(
            ops / "promotion_maintenance.json",
            {
                "active": False,
                "transaction_id": plan["transaction_id"],
                "completed_at": utc(),
            },
        )
        return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--run-root", required=True, type=Path)
    p.add_argument("--candidate-repo", required=True, type=Path)
    p.add_argument("--candidate-commit", required=True)
    p.add_argument("--review-receipt", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p = sub.add_parser("apply")
    p.add_argument("--plan", required=True, type=Path)
    args = parser.parse_args()
    if args.mode == "plan":
        plan = make_plan(args)
        require(
            args.output.resolve().is_relative_to(
                args.run_root.resolve() / "operations"
            ),
            "Plan output must be under operations",
        )
        save(args.output, plan)
        print(
            json.dumps(
                {
                    "state": "PLANNED",
                    "transaction_id": plan["transaction_id"],
                    "plan_sha256": sha(args.output),
                }
            )
        )
    else:
        print(json.dumps(apply_plan(args.plan)))


if __name__ == "__main__":
    main()
