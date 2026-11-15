"""Persistently queue Astra after the current Leduc comparison passes its gate.

The teacher API runs locally using the existing Keychain credential. Only frozen
public labels, evidence and source code enter the GPU payload. No API key is
written to a workflow, command line, archive, or experiment record.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

from ..data import file_sha256, write_json
from .data import leduc_gate


def utc():
    return datetime.now(timezone.utc).isoformat()


def command(args, *, cwd=None, timeout=40):
    result = subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=True, timeout=timeout)
    return json.loads(result.stdout)


def query(workflow):
    return command(["osmo", "workflow", "query", workflow, "--format-type", "json"])


def combine(leact, exit_report):
    report = dict(leact)
    report["protocol"] = {**report["protocol"], "role": "combined"}
    report["methods"] = {"leact": leact["methods"]["leact"], "exit": exit_report["methods"]["exit"]}
    return report


def bundle(source_archive, teacher, report, output):
    # Copy only allowlisted, already frozen source entries and these three
    # explicit public input files. No environment or credentials are included.
    with tarfile.open(source_archive, "r:gz") as source, tarfile.open(output, "w:gz") as target:
        for member in source.getmembers():
            if not member.isfile() or member.name.startswith("/") or ".." in Path(member.name).parts:
                raise ValueError("Unexpected frozen-source archive member")
            target.addfile(member, source.extractfile(member))
        for name, path in [("teacher.jsonl", teacher), ("teacher.meta.json", teacher.with_suffix(".meta.json")),
                           ("leduc-results.json", report)]:
            data = path.read_bytes()
            member = tarfile.TarInfo("queued-input/" + name)
            member.size, member.mode = len(data), 0o644
            target.addfile(member, io.BytesIO(data))
    return file_sha256(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    work = args.config.resolve().parent
    lock = (work / "queue.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return
    source, repo = Path(config["frozen_source"]), Path(config["repo"])
    archive = Path(config["source_archive"])
    if file_sha256(archive) != config["source_sha256"]:
        raise ValueError("Frozen queue source changed")
    runtime_path = work / "runtime.json"
    state = json.loads(runtime_path.read_text()) if runtime_path.exists() else {}
    status_path = repo / "reproductions/leact/artifacts/astra-expert/status.json"

    def progress(stage, **details):
        state.update(stage=stage, updated_utc=utc(), pid=os.getpid(), **details)
        write_json(runtime_path, state)
        status = json.loads(status_path.read_text())
        status.update(status=stage.replace("_", " "), updated_utc=utc(), queue=state,
                      training_submitted=bool(state.get("workflow")), full_teacher_collection_started=bool(state.get("collection_started")),
                      gate_passed=bool(state.get("gate", {}).get("passed")))
        write_json(status_path, status)
        print(json.dumps({"stage": stage, "updated_utc": state["updated_utc"]}), flush=True)
        try:
            from ..tracking.export import snapshot
            from ..tracking.server import ingest
            snapshot_value = snapshot(repo / "reproductions/leact")
            ingest(repo / "reproductions/leact/cluster/.build/tracking/experiments.sqlite3", snapshot_value)
        except Exception as error:
            print(json.dumps({"tracker_update_failed": type(error).__name__}), flush=True)

    def run_logged(argv, name, timeout):
        with (work / name).open("a") as output:
            subprocess.run(argv, cwd=source, stdout=output, stderr=subprocess.STDOUT,
                           stdin=subprocess.DEVNULL, check=True, timeout=timeout)

    try:
        while not state.get("gate", {}).get("passed"):
            # The collector persists the full report locally before its task
            # goes away. The two roles retain independent immutable evidence.
            left, right = Path(config["leact_report"]), Path(config["exit_report"])
            if left.exists() and right.exists():
                report = combine(json.loads(left.read_text()), json.loads(right.read_text()))
                if all(len(report["methods"].get(a, [])) == 3 for a in ("leact", "exit")):
                    try:
                        gate = leduc_gate(report, config["oracle_sha256"])
                    except ValueError as error:
                        progress("held_leduc_improvement_not_yet_demonstrated", reason=str(error))
                        return
                    write_json(work / "leduc-results.json", report)
                    progress("leduc_dependency_passed", gate=gate)
                    break
            scheduler = query(config["dependency_workflow"])
            if scheduler["status"].startswith("FAILED"):
                progress("held_dependency_workflow_failed", scheduler_status=scheduler["status"])
                return
            if scheduler["status"] == "COMPLETED" and state.get("dependency_completed_since"):
                if time.time() - state["dependency_completed_since"] > 300:
                    progress("held_dependency_report_missing")
                    return
            elif scheduler["status"] == "COMPLETED":
                state["dependency_completed_since"] = time.time()
            progress("queued_after_leduc_evaluation", dependency_workflow=config["dependency_workflow"],
                     scheduler_status=scheduler["status"])
            if args.once:
                return
            time.sleep(30)
        if args.once:
            return
        oracle = source / "reproductions/leact/artifacts/leduc6r2s/oracle.jsonl"
        teacher = work / "teacher.jsonl"
        common = ["--oracle", str(oracle), "--root", str(work / "teacher-collection"),
                  "--leduc-results", str(work / "leduc-results.json"), "--workers", "4"]
        for stage in ["actions", "summaries"]:
            if not (work / "teacher-collection" / (stage + "-complete.json")).exists():
                progress("collecting_astra_" + stage, collection_started=True)
                run_logged([sys.executable, "-m", "reproductions.leact.astra_expert", stage, *common], stage + ".log", 172800)
        if not teacher.exists():
            run_logged([sys.executable, "-m", "reproductions.leact.astra_expert", "export", *common,
                        "--include-summaries", "--output", str(teacher)], "export.log", 300)
        quality = json.loads(teacher.with_suffix(".meta.json").read_text())["quality"]
        progress("astra_teacher_frozen", teacher_quality=quality)
        if not state.get("workflow"):
            while True:
                active = command(["osmo", "workflow", "list", "--name", "leact", "--count", "100",
                                  "--status", "RUNNING", "PENDING", "WAITING", "--format-type", "json"])["workflows"]
                if not active:
                    break
                progress("waiting_for_prior_leact_allocations_to_finish", active=[v["name"] for v in active])
                time.sleep(30)
            payload = work / "launch-payload.tar.gz"
            sha = bundle(archive, teacher, work / "leduc-results.json", payload)
            workflow = source / "reproductions/leact/cluster/astra-workflow.yaml"
            run_logged(["osmo", "workflow", "validate", str(workflow), "--pool", "groot-h100-01",
                        "--set-string", "payload_sha256=" + sha], "validation.log", 60)
            # Persist an intent before the external write. A lost submit response
            # must be reconciled manually rather than risking a duplicate job.
            intent = work / "submission-intent.json"
            if intent.exists() and not (work / "submission.json").exists():
                progress("held_submission_response_requires_reconciliation")
                return
            if (work / "submission.json").exists():
                submitted = json.loads((work / "submission.json").read_text())
            else:
                write_json(intent, {"created_utc": utc(), "payload_sha256": sha})
                submitted = command(["osmo", "workflow", "submit", str(workflow), "--pool", "groot-h100-01",
                    "--priority", "HIGH", "--set-string", "payload_sha256=" + sha, "--format-type", "json"], timeout=60)
                write_json(work / "submission.json", submitted)
            write_json(work / "launch-record.json", {**submitted, "payload_sha256": sha, "priority": "HIGH",
                       "pool": "groot-h100-01", "gpu_count": 8, "phase": "Astra actions versus three-task conditioning"})
            progress("astra_workflow_submitted", workflow=submitted["name"], payload_sha256=sha)
        while True:
            scheduler = query(state["workflow"])
            write_json(work / "status-current.json", scheduler)
            if scheduler["status"] == "RUNNING":
                break
            if scheduler["status"].startswith("FAILED") or scheduler["status"] == "COMPLETED":
                progress("held_astra_workflow_ended_before_upload", scheduler_status=scheduler["status"])
                return
            progress("astra_queued_for_h100", scheduler_status=scheduler["status"])
            time.sleep(30)
        if not state.get("payload_uploaded"):
            run_logged([sys.executable, "-m", "reproductions.leact.cluster.upload_payload", state["workflow"],
                        str(work / "launch-payload.tar.gz"), "--task", "train", "--sha256", state["payload_sha256"]],
                       "upload.log", 180)
            progress("astra_training_and_evaluation_running", payload_uploaded=True)
        # The collector is kept alive in a separate process while snapshots are
        # refreshed below; all raw records also remain on persistent Lustre.
        remote = "/mnt/amlfs-07/shared/egoverse/rpunamiya/leact"
        with (work / "collector.log").open("a") as output:
            collector = subprocess.Popen([sys.executable, "-m", "reproductions.leact.cluster.collect_results",
                state["workflow"], "--task", "train", "--source-root", remote + "/runs/" + state["workflow"] + "/source",
                "--experiment-root", remote + "/studies/astra-conditioning-v2", "--output", str(work / "collected"),
                "--astra", "--watch-seconds", "172800", "--interval", "30", "--max-rollout-bytes", "1000000"],
                cwd=source, stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT)
        while True:
            scheduler = query(state["workflow"])
            write_json(work / "status-current.json", scheduler)
            if scheduler["status"] == "COMPLETED" or scheduler["status"].startswith("FAILED"):
                progress("astra_workflow_finished", scheduler_status=scheduler["status"])
                return
            local_progress = work / "collected/latest/progress.json"
            progress("astra_training_and_evaluation_running", experiment_progress=(
                json.loads(local_progress.read_text()) if local_progress.exists() else None), collector_pid=collector.pid)
            time.sleep(30)
    except Exception as error:
        progress("held_queue_error", error_type=type(error).__name__, reason=str(error)[:1000])
        raise


if __name__ == "__main__":
    main()
