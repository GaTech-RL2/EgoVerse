"""Own one submitted study's immutable upload and local transports only."""

import argparse
import concurrent.futures
import contextlib
import fcntl
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import tarfile
import time
from pathlib import Path

from astra_reversal.osmo.demo_skill_recovery import OWNED_WORKFLOW
from astra_reversal.osmo.relay_health import ForwardHealth, healthy_relay

ROOT = REPO = PLAN = NAME = None
CHILDREN = {}
STOP = False


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def command(argv, timeout=45):
    result = subprocess.run(
        argv, capture_output=True, text=True, timeout=timeout, cwd=REPO
    )
    if result.returncode:
        raise RuntimeError(
            f"{argv[1:3]} exited {result.returncode}: {(result.stdout + result.stderr)[-1600:]}"
        )
    return result.stdout


def spawn(task, kind, argv):
    with (ROOT / task / f"{kind}.log").open("a") as log:
        cwd = ROOT / "relay_source" if kind == "relay" else REPO
        child = subprocess.Popen(
            argv, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
    CHILDREN[task, kind] = child
    write(
        ROOT / task / f"{kind}.pid.json",
        {"pid": child.pid, "started_unix": time.time()},
    )


def stop(task, kind):
    child = CHILDREN.pop((task, kind), None)
    if child is not None and child.poll() is None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(child.pid, signal.SIGKILL)


def upload(task):
    # Transfer retries do not execute a model or restart a simulator.
    for attempt in range(3):
        try:
            output = command(
                [
                    "osmo",
                    "workflow",
                    "rsync",
                    NAME,
                    task,
                    PLAN["payload"] + ":/osmo/run/workspace",
                    "--once",
                    "--timeout",
                    "180",
                ],
                210,
            )
            (ROOT / task / "upload.log").write_text(output)
            return {"status": "uploaded", "attempts": attempt + 1}
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            (ROOT / task / "upload.log").write_text(str(exc))
            if attempt < 2:
                time.sleep(5)
    return {"status": "upload_failed", "attempts": 3}


def interrupt(*_args):
    global STOP
    STOP = True


def main():
    global ROOT, REPO, PLAN, NAME
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("launch", type=Path)
    parser.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()
    ROOT, REPO = args.launch.resolve(), args.repo.resolve()
    PLAN = json.loads((ROOT / "launch_plan.json").read_text())
    NAME = json.loads((ROOT / "submission.json").read_text())["name"]
    if not re.fullmatch(OWNED_WORKFLOW, NAME):
        raise ValueError("Expected this explicitly owned pilot/full workflow")
    lock = (ROOT / "supervisor.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    identity = json.loads((ROOT / "source_identity.json").read_text())
    with Path(PLAN["payload"]).open("rb") as stream:
        assert (
            hashlib.file_digest(stream, "sha256").hexdigest()
            == identity["payload_sha256"]
        )
    assert command(["git", "rev-parse", "HEAD"]).strip() == identity["source_revision"]
    # Local model jobs use exactly the submitted source, even if later work
    # changes the development checkout. Only Python source files are needed.
    snapshot = ROOT / "relay_source"
    snapshot.mkdir(exist_ok=False)
    with tarfile.open(PLAN["payload"]) as archive:
        for member in archive.getmembers():
            path = Path(member.name)
            if (
                member.isfile()
                and path.parts[0] == "astra_reversal"
                and path.suffix == ".py"
                and ".deps" not in path.parts
            ):
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError("Unsafe source member")
                target = snapshot / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.extractfile(member).read())
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    deadline = time.time() + 28 * 3600
    state = {
        "workflow": NAME,
        "workers": {},
        "started_unix": time.time(),
        "model_job_restarts": False,
    }
    uploads = {}
    terminal = set()
    monitors = {}
    tokens = {
        r["task"]: Path(r["token_file"]).read_text().strip() for r in PLAN["workers"]
    }
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        try:
            while not STOP and time.time() < deadline:
                try:
                    raw = command(
                        ["osmo", "workflow", "query", NAME, "--format-type", "json"]
                    )
                    status = json.loads(raw[raw.index("{") :])
                    write(ROOT / "status.json", status)
                except (
                    OSError,
                    RuntimeError,
                    ValueError,
                    subprocess.SubprocessError,
                ) as exc:
                    state.update(query_error=str(exc), updated_unix=time.time())
                    write(ROOT / "supervisor_state.json", state)
                    time.sleep(10)
                    continue
                state.pop("query_error", None)
                state["workflow_status"] = status["status"]
                tasks = {t["name"]: t for g in status["groups"] for t in g["tasks"]}
                for row in PLAN["workers"]:
                    task = row["task"]
                    if task not in tasks:
                        continue
                    current = state["workers"].setdefault(task, {})
                    current["status"] = tasks[task]["status"]
                    done = current["status"] == "COMPLETED" or current[
                        "status"
                    ].startswith(("FAILED", "CANCELED"))
                    if done:
                        terminal.add(task)
                        stop(task, "relay")
                        stop(task, "forward")
                        continue
                    if current["status"] != "RUNNING":
                        continue
                    if task not in uploads:
                        uploads[task] = pool.submit(upload, task)
                    if uploads[task].done():
                        current["upload"] = uploads[task].result()
                    for kind in ("forward", "relay"):
                        child = CHILDREN.get((task, kind))
                        if child is not None and child.poll() is not None:
                            current[kind + "_exit"] = child.returncode
                            CHILDREN.pop((task, kind))
                    if (task, "forward") in CHILDREN:
                        healthy = healthy_relay(row["port"], tokens[task])
                        reason = monitors[task].observe(healthy, time.time())
                        current["transport_healthy"] = healthy
                        current["ever_healthy"] = (
                            current.get("ever_healthy", False) or healthy
                        )
                        current["health_failures"] = monitors[task].failures
                        if reason:
                            # Only renew the transport. Never restart a Codex job.
                            stop(task, "forward")
                            current["transport_renewals"] = (
                                current.get("transport_renewals", 0) + 1
                            )
                            current["last_renewal_reason"] = reason
                            current["forward_retry_after"] = time.time()
                    if (task, "forward") not in CHILDREN and time.time() >= current.get(
                        "forward_retry_after", 0
                    ):
                        spawn(
                            task,
                            "forward",
                            [
                                "osmo",
                                "workflow",
                                "port-forward",
                                NAME,
                                task,
                                "--port",
                                str(row["port"]) + ":8769",
                                "--connect-timeout",
                                "60",
                            ],
                        )
                        monitors[task] = ForwardHealth(
                            time.time(), seen_healthy=current.get("ever_healthy", False)
                        )
                        current["forward_retry_after"] = time.time() + 30
                    if not current.get("relay_started"):
                        spawn(
                            task,
                            "relay",
                            [
                                sys.executable,
                                "-m",
                                "astra_reversal.codex_relay",
                                "--url",
                                f"http://127.0.0.1:{row['port']}",
                                "--token-file",
                                row["token_file"],
                                "--directory",
                                str(ROOT / task / "jobs"),
                                "--idle-timeout",
                                "86400",
                            ],
                        )
                        current["relay_started"] = True
                state["updated_unix"] = time.time()
                write(ROOT / "supervisor_state.json", state)
                if len(terminal) == len(PLAN["workers"]):
                    state["finished_unix"] = time.time()
                    break
                time.sleep(10)
            else:
                state["stop_reason"] = (
                    "operator_signal" if STOP else "declared_deadline"
                )
                try:
                    command(
                        [
                            "osmo",
                            "workflow",
                            "cancel",
                            NAME,
                            "--message",
                            state["stop_reason"],
                        ]
                    )
                    state["owned_workflow_cancel_requested"] = True
                except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
                    state["cancel_error"] = str(exc)
        finally:
            for task, kind in list(CHILDREN):
                stop(task, kind)
            write(ROOT / "supervisor_result.json", state)


if __name__ == "__main__":
    main()
