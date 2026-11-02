"""Plan or install reviewed operations code while preserving all runtime state.

Installation never starts a process or submits a job. Stop the three operations
services first; pipeline/data jobs and every runtime manifest remain untouched.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import time
from contextlib import ExitStack
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILES = (
    "__init__.py",
    "agent_dispatcher.py",
    "recovery_controller.py",
    "supervise.py",
    "promote_preproduction.py",
    "install_operations.py",
    "AGENT_POLICY.md",
    "agent_repair_prompt.md",
    "agent_review_prompt.md",
    "agent_apply_prompt.md",
    "agent_result.schema.json",
    "README.md",
)


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def save(path, value):
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w") as stream:
        os.chmod(temp, 0o600)
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(source=HERE):
    result = {}
    for name in FILES:
        path = source / name
        if not path.is_file() or path.is_symlink():
            raise RuntimeError("Incomplete or symlinked operations package")
        result[name] = file_hash(path)
    digest = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
    return {"package_sha256": digest, "files": result}


def require_quiescent(root):
    ops = root / "operations"
    if read(root / "production_control.json", {}).get("allow_training") is not False:
        raise RuntimeError("Installation requires durable preflight-only control")
    if read(ops / "dispatcher_control.json", {}).get("enabled") is True:
        raise RuntimeError("Disable dispatch before installing operations code")
    if read(ops / "promotion_maintenance.json", {}).get("active"):
        raise RuntimeError("A source promotion is active")
    state = read(ops / "recovery_state.json", {})
    if any(
        i.get("state") in ("CLAIMED", "LEASE_EXPIRED")
        for i in state.get("incidents", {}).values()
    ):
        raise RuntimeError(
            "Reconcile active or expired repair claims before installation"
        )
    # Fail closed on remote/unknown identity; never infer that an unobservable
    # service died. A reused PID is a different process and is left untouched.
    import socket

    for name in ("dispatcher_status.json", "recovery_watch_status.json"):
        identity = read(ops / name, {}).get("identity")
        if not identity:
            continue
        if identity.get("host") != socket.gethostname():
            raise RuntimeError("Operations service identity is on another host")
        path = Path("/proc") / str(identity.get("pid")) / "stat"
        if path.exists():
            fields = path.read_text().rsplit(")", 1)[1].split()
            if fields[0] not in ("Z", "X") and int(fields[19]) == identity.get(
                "start_ticks"
            ):
                raise RuntimeError("An operations service is still running")
    supervisor = read(ops / "supervisor_identity.json", {}).get("supervisor_pid")
    if supervisor:
        path = Path("/proc") / str(supervisor) / "stat"
        if path.exists() and path.read_text().rsplit(")", 1)[1].split()[0] not in (
            "Z",
            "X",
        ):
            raise RuntimeError(
                "Stop or reconcile the recorded supervisor before installation"
            )


def install(root, expected, source=HERE):
    root = root.resolve()
    ops = root / "operations"
    ops.mkdir(parents=True, exist_ok=True)
    package = inventory(source)
    if expected != package["package_sha256"]:
        raise RuntimeError("Reviewed package checksum differs")
    with ExitStack() as stack:
        # Hold each service singleton lock through the last atomic replacement.
        # This also covers missing/stale PID receipts and racing restarts.
        for name in (
            "operations_install.lock",
            "agent_dispatcher.lock",
            "recovery_watch.lock",
            "supervision.lock",
        ):
            lock = stack.enter_context((ops / name).open("a+"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require_quiescent(root)
        intent_path = ops / "operations_install.json"
        previous = read(intent_path, {})
        if (
            previous.get("state") == "INSTALLING"
            and previous.get("package_sha256") != expected
        ):
            raise RuntimeError(
                "A different interrupted install requires reconciliation"
            )
        receipt = {"state": "INSTALLING", **package, "started_at": time.time()}
        save(intent_path, receipt)
        backup = ops / "code_history" / expected
        backup.mkdir(parents=True, exist_ok=True)
        for name, checksum in package["files"].items():
            target = ops / name
            if target.is_symlink():
                raise RuntimeError("Refusing to replace a symlinked operations file")
            if target.exists() and not (backup / name).exists():
                shutil.copy2(target, backup / name)
            temp = ops / (name + ".install.tmp")
            shutil.copyfile(source / name, temp)
            os.chmod(temp, 0o600)
            if file_hash(temp) != checksum:
                raise RuntimeError("Source package changed during installation")
            temp.replace(target)
        receipt.update(state="INSTALLED", finished_at=time.time())
        save(intent_path, receipt)
        return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--expected-package-sha")
    args = parser.parse_args()
    if args.apply:
        if not args.expected_package_sha:
            parser.error("--apply requires the reviewed --expected-package-sha")
        print(json.dumps(install(args.run_root, args.expected_package_sha), indent=2))
    else:
        print(
            json.dumps(
                {
                    "state": "PLAN_ONLY",
                    "run_root": str(args.run_root.resolve()),
                    **inventory(),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
