#!/usr/bin/env python3
"""Poll the four history-ladder jobs and decide when a Codex audit is due."""

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MONITOR = ROOT / "logs/fixed_budget_opscale_hist/watchdog"
STATE = MONITOR / "state.json"
SNAPSHOT = MONITOR / "snapshot.json"


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def tail_text(path, n=500_000):
    if not path.is_file():
        return ""
    with path.open("rb") as handle:
        handle.seek(max(0, path.stat().st_size - n))
        return handle.read().decode("utf-8", "replace")


def main():
    now = int(time.time())
    state = json.loads(STATE.read_text())
    jobs = [
        str(entry["job_id"]) for entry in state["rungs"].values() if entry.get("job_id")
    ]
    cmd = ["squeue", "-h", "-j", ",".join(jobs), "-o", "%i|%T|%R"]
    query = subprocess.run(cmd, text=True, capture_output=True, check=False, timeout=30)
    active = {}
    if query.returncode == 0:
        for line in query.stdout.splitlines():
            fields = line.split("|", 2)
            if len(fields) == 3:
                active[fields[0].strip()] = {
                    "state": fields[1].strip(),
                    "reason": fields[2].strip(),
                }
    issues = []
    rungs = {}
    if query.returncode != 0:
        issues.append(f"squeue failed: {query.stderr.strip()}")
    for ops, entry in state["rungs"].items():
        job = str(entry.get("job_id", ""))
        current = active.get(job)
        log = ROOT / f"slurm-{job}.out"
        output = Path(entry.get("output_dir", ""))
        if output and not output.is_absolute():
            output = ROOT / output
        checkpoints = (
            sorted(
                (output / "checkpoints").glob("*.ckpt"),
                key=lambda p: p.stat().st_mtime,
            )
            if output.is_dir()
            else []
        )
        item = {
            "job_id": job,
            "slurm": current,
            "log": str(log),
            "log_mtime": int(log.stat().st_mtime) if log.is_file() else None,
            "latest_checkpoint": str(checkpoints[-1]) if checkpoints else None,
        }
        if entry.get("status") == "complete":
            item["assessment"] = "complete"
        elif query.returncode != 0:
            item["assessment"] = "unknown"
        elif current is None:
            end = tail_text(log)
            if "Trainer.fit stopped" in end and "max_epochs" in end and "5000" in end:
                entry["status"] = "complete"
                item["assessment"] = "complete"
            else:
                item["assessment"] = "stopped"
                issues.append(f"op{ops} job {job} left squeue before completion")
        elif current["state"] in {"PENDING", "CONFIGURING"}:
            item["assessment"] = "queued"
        elif current["state"] in {"RUNNING", "COMPLETING"}:
            item["assessment"] = "running"
            mtime = item["log_mtime"]
            if mtime is not None and now - mtime > 3 * 3600:
                item["assessment"] = "stalled"
                issues.append(
                    f"op{ops} job {job} log has not advanced for over 3 hours"
                )
            elif "Traceback (most recent call last)" in tail_text(log, 40_000):
                item["assessment"] = "error_in_log"
                issues.append(f"op{ops} job {job} has a traceback in recent output")
        else:
            item["assessment"] = current["state"].lower()
            issues.append(f"op{ops} job {job} is {current['state']}")
        entry["last_assessment"] = item["assessment"]
        rungs[ops] = item
    all_complete = all(e.get("status") == "complete" for e in state["rungs"].values())
    elapsed = now - int(state.get("last_agent_run_at", 0))
    agent_due = bool(issues) and elapsed >= 30 * 60
    if not issues and elapsed >= 30 * 60 and not all_complete:
        agent_due = True
    if agent_due:
        state["last_agent_run_at"] = now
    state["last_scan_at"] = now
    snapshot = {
        "checked_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
        "issues": issues,
        "agent_due": agent_due,
        "all_complete": all_complete,
        "rungs": rungs,
    }
    write_json(STATE, state)
    write_json(SNAPSHOT, snapshot)
    print(json.dumps(snapshot, sort_keys=True))
    if all_complete:
        return 3
    return 10 if agent_due else 0


if __name__ == "__main__":
    sys.exit(main())
