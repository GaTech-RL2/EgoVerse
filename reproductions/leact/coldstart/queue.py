"""Schedule the EOD study with at most two LeAct nodes and a measured Astra gate."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

from ..data import file_sha256, write_json

REMOTE = "/mnt/amlfs-07/shared/egoverse/rpunamiya/leact"
STORAGE = "/mnt/amlfs-03/home/rpunamiya/leact"
ROLES = ("self", "frontier", "astra")


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def api_alive(root):
    runner = read(root / "runner.json", {})
    if not runner.get("pid"):
        return False
    try:
        os.kill(runner["pid"], 0)
        return True
    except ProcessLookupError:
        return False


def command(argv, *, timeout=60):
    result = subprocess.run(argv, capture_output=True, text=True, check=True, timeout=timeout)
    return json.loads(result.stdout)


def bundle(archive, input_files, output):
    with tarfile.open(archive, "r:gz") as source, tarfile.open(output, "w:gz") as target:
        for member in source.getmembers():
            if not member.isfile() or member.name.startswith("/") or ".." in Path(member.name).parts:
                raise ValueError("Unexpected source archive entry")
            target.addfile(member, source.extractfile(member))
        hashes = {}
        for name, path in input_files.items():
            data = path.read_bytes()
            member = tarfile.TarInfo("study-input/" + name)
            member.size, member.mode = len(data), 0o644
            target.addfile(member, io.BytesIO(data))
            hashes[name] = file_sha256(path)
    write_json(output.with_name("input-sha256.json"), hashes)
    return file_sha256(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = read(args.config)
    repo, source, work = (Path(config[k]) for k in ("repo", "source", "work"))
    archive = Path(config["source_archive"])
    if file_sha256(archive) != config["source_sha256"]:
        raise ValueError("Frozen study source changed")
    lock = (work / "queue.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return
    builds = {role: work.with_name(work.name + "-" + role) for role in ROLES}
    for directory in builds.values():
        directory.mkdir(exist_ok=True)
    runtime = read(work / "runtime.json", {"roles": {role: {"stage": "waiting"} for role in ROLES}})
    collectors = {}
    last_sync = 0
    env = {**os.environ, "PYTHONPATH": str(source)}

    def logged(argv, path, timeout):
        with path.open("a") as log:
            subprocess.run(argv, cwd=source, stdin=subprocess.DEVNULL, stdout=log,
                           stderr=subprocess.STDOUT, env=env, check=True, timeout=timeout)

    def publish():
        nonlocal last_sync
        runtime.update(updated_utc=utc(), pid=os.getpid())
        write_json(work / "runtime.json", runtime)
        gate = read(work / "astra/gate.json")
        api = {name: read(work / name / "progress.json", {}) for name in ("frontier", "astra")}
        studies = [
            {"label": "1 · Frontier warmstart + LeAct", "status": runtime["roles"]["frontier"]["stage"],
             "detail": f"Frontier API {api['frontier'].get('completed', 0)}/4032; full joint warmstart; two LeAct rounds"},
            {"label": "2 · Self-generated scaling", "status": runtime["roles"]["self"]["stage"],
             "detail": "128 → 4032 states; then 1 → 4 retained traces/state from a fixed N=8 pool"},
            {"label": "3 · Astra three-task reasoning", "status": runtime["roles"]["astra"]["stage"],
             "detail": f"Solver exploitability {gate['exploitability']:.6g} / threshold 0.10" if gate else f"Astra action API {api['astra'].get('completed', 0)}/4032; awaiting full-tree solver test"},
            {"label": "4 · Astra action-only expert", "status": runtime["roles"]["astra"]["stage"],
             "detail": "Same solver gate, warmstart and matched SFT budget as experiment 3"}]
        status = {"status": "Overnight Leduc studies · target 5 p.m. Pacific, October 22",
                  "updated_utc": utc(), "deadline": config["deadline"], "studies": studies,
                  "active_workflows": [v["workflow"] for v in runtime["roles"].values()
                                       if v.get("workflow") and not v.get("terminal")],
                  "astra_gate": gate, "api": api, "roles": runtime["roles"]}
        write_json(repo / "reproductions/leact/artifacts/overnight-status.json", status)
        try:
            from ..tracking.export import snapshot
            from ..tracking.server import ingest, ingest_records
            from ..tracking.traces import records
            base = repo / "reproductions/leact"
            ingest(base / "cluster/.build/tracking/experiments.sqlite3", snapshot(base))
            ingest_records(base / "cluster/.build/tracking/experiments.sqlite3", records(base))
        except Exception as error:
            print(json.dumps({"tracking_error": type(error).__name__}), flush=True)
        if time.monotonic() - last_sync >= 300:
            last_sync = time.monotonic()
            try:
                logged([sys.executable, "-m", "reproductions.leact.tracking.sync", "--state",
                        str(work / "itss-sync.json")], work / "itss-sync.log", 45)
                runtime["itss_sync"] = "succeeded"
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                runtime["itss_sync"] = "unavailable; complete local SQLite and Lustre records retained"

    print(json.dumps({"queue_started": utc(), "deadline": config["deadline"]}), flush=True)
    deadline = time.monotonic() + 30 * 3600
    while time.monotonic() < deadline:
        try:
            active = command(["osmo", "workflow", "list", "--name", "leact", "--count", "100",
                              "--status", "RUNNING", "PENDING", "WAITING", "--format-type", "json"])["workflows"]
            reserved = len(active)
            runtime["existing_allocations"] = [v["name"] for v in active]
            for role in ROLES:
                state, build = runtime["roles"][role], builds[role]
                if state.get("terminal") or state["stage"].startswith("held_"):
                    continue
                try:
                    if not state.get("workflow"):
                        inputs = {}
                        if role == "frontier":
                            if not (work / "frontier/complete.json").exists():
                                state["stage"] = "collecting frontier warmstart" if api_alive(work / "frontier") else "held_frontier_API_collection_incomplete"
                                continue
                            inputs = {name: work / "frontier" / name for name in ("frontier.jsonl", "complete.json", "collection.json")}
                        if role == "astra":
                            gate = read(work / "astra/gate.json")
                            if gate is None:
                                state["stage"] = "awaiting Astra solver-quality gate" if api_alive(work / "astra") else "held_Astra_API_collection_incomplete"
                                continue
                            if not gate["passed"]:
                                state.update(stage="skipped: Astra exploitability exceeds 0.10", terminal=True)
                                continue
                            ready = builds["frontier"] / "collected/latest/anchor-ready.json"
                            if not ready.exists() or not (work / "astra/teacher.jsonl").exists():
                                state["stage"] = "solver gate passed; waiting for shared frontier warmstart and summaries"
                                if not (work / "astra/teacher.jsonl").exists() and not api_alive(work / "astra"):
                                    state["stage"] = "held_Astra_summary_collection_incomplete"
                                continue
                            inputs = {name: work / "astra" / name for name in ("teacher.jsonl", "teacher.meta.json", "gate.json")}
                        if reserved >= 2:
                            state["stage"] = "queued for a free node within the two-node budget"
                            continue
                        payload = build / "payload.tar.gz"
                        sha = bundle(archive, inputs, payload)
                        workflow = source / "reproductions/leact/cluster/coldstart-workflow.yaml"
                        intent = build / "submission-intent.json"
                        submitted = read(build / "submission.json")
                        if not submitted:
                            if intent.exists():
                                raise RuntimeError("Submission intent requires reconciliation; no duplicate submitted")
                            logged(["osmo", "workflow", "validate", str(workflow), "--pool", "groot-h100-01",
                                    "--set", "role="+role, "--set-string", "payload_sha256="+sha], build / "validation.log", 60)
                            write_json(intent, {"created_utc": utc(), "role": role, "payload_sha256": sha})
                            submitted = command(["osmo", "workflow", "submit", str(workflow), "--pool", "groot-h100-01",
                                "--priority", "HIGH", "--set", "role="+role, "--set-string", "payload_sha256="+sha,
                                "--format-type", "json"])
                            write_json(build / "submission.json", submitted)
                        state.update(workflow=submitted["name"], stage="submitted HIGH priority", payload_sha256=sha)
                        write_json(build / "launch-record.json", {**submitted, "priority": "HIGH", "pool": "groot-h100-01",
                                   "gpu_count": 8, "payload_sha256": sha, "phase": role + " overnight study"})
                        reserved += 1
                        write_json(work / "runtime.json", runtime)
                    scheduler = command(["osmo", "workflow", "query", state["workflow"], "--format-type", "json"])
                    write_json(build / "status-current.json", scheduler)
                    state["scheduler_status"] = scheduler["status"]
                    if scheduler["status"] == "COMPLETED" or scheduler["status"].startswith("FAILED"):
                        state.update(terminal=True, stage=scheduler["status"])
                        continue
                    if scheduler["status"] != "RUNNING":
                        continue
                    if not state.get("checkpoint_staging_ready"):
                        logged([sys.executable, "-m", "reproductions.leact.cluster.prepare_staging",
                                state["workflow"], role], build / "checkpoint-staging.log", 90)
                        state["checkpoint_staging_ready"] = True
                    if not state.get("payload_uploaded"):
                        logged([sys.executable, "-m", "reproductions.leact.cluster.upload_payload", state["workflow"],
                                str(build / "payload.tar.gz"), "--task", "train", "--sha256", state["payload_sha256"]],
                               build / "upload.log", 180)
                        state["payload_uploaded"] = True
                    if role not in collectors or collectors[role].poll() is not None:
                        with (build / "collector.log").open("a") as log:
                            collectors[role] = subprocess.Popen([sys.executable, "-m", "reproductions.leact.cluster.collect_results",
                                state["workflow"], "--task", "train", "--source-root", REMOTE+"/runs/"+state["workflow"]+"/source",
                                "--experiment-root", STORAGE+"/studies/overnight-20261022-"+role,
                                "--output", str(build / "collected"), "--astra" if role == "astra" else "--coldstart",
                                "--watch-seconds", "600", "--interval", "30", "--max-rollout-bytes", "1000000"],
                                cwd=source, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                    progress = read(build / "collected/latest/progress.json", {})
                    started = scheduler.get("start_time")
                    updated = progress.get("updated_utc")
                    fresh = bool(started and updated and
                                 datetime.fromisoformat(updated).replace(tzinfo=None) >=
                                 datetime.fromisoformat(started).replace(tzinfo=None))
                    state.update(stage=progress.get("phase") if fresh else "H100 startup; resuming saved work",
                                 progress=progress, progress_current_workflow=fresh,
                                 collector_pid=collectors[role].pid)
                except Exception as error:
                    state.update(stage="held_needs_inspection", error_type=type(error).__name__, reason=str(error)[:500])
            publish()
            if all(v.get("terminal") or v["stage"].startswith("held_") for v in runtime["roles"].values()):
                break
        except Exception as error:
            runtime["query_error"] = {"type": type(error).__name__, "time": utc(), "reason": str(error)[:500]}
            publish()
        time.sleep(30)
    write_json(work / "queue-stopped.json", {"updated_utc": utc(), "runtime": runtime})


if __name__ == "__main__":
    main()
