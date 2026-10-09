"""Restore the native Xiaomi release, run its server, and archive the screen."""

import argparse
import json
import os
import signal
import socket
import subprocess
import tarfile
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from astra_reversal.complex_manipulation.worker import (
    Publisher, archive_client, safe_relative, sha256, write_json,
)


def restore(client, stage, manifest_path, root, *, include_weights=True):
    prefix = f"experiments/astra-complex-20261006/{stage}/"
    manifest = json.loads(manifest_path.read_text())
    receipt = json.loads(client.get_object(Bucket="rldb", Key=prefix + "receipt.json")["Body"].read())
    if (receipt["workflow"] != stage or receipt["manifest_sha256"] != sha256(manifest_path)
            or receipt["status"] != "staged_not_evaluated"):
        raise ValueError("Native model stage does not match the pinned manifest")
    root.mkdir(parents=True, exist_ok=False)
    rows = receipt["uploaded_files"] + receipt["asset_files"]
    if not include_weights:
        # Reset-only CPU audits need the identical simulator, not policy weights.
        rows = [row for row in rows if not row["relative"].startswith("weights/")]
    seen = set()
    for row in rows:
        relative = str(safe_relative(row["relative"]))
        expected_prefix = (manifest["asset_receipt_key"].rsplit("/", 1)[0] + "/"
                           if relative.startswith("asset_archives/") else prefix)
        if relative in seen or row["key"] != expected_prefix + relative or not 0 < row["bytes"] < 30 * 1024**3:
            raise ValueError("Unexpected staged artifact scope or size")
        seen.add(relative)

    def fetch(row):
        path = root / row["relative"]
        path.parent.mkdir(parents=True, exist_ok=True)
        client.download_file("rldb", row["key"], str(path))
        if path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
            raise ValueError("Native artifact checksum mismatch")
        print(json.dumps({"restored": row["relative"], "bytes": row["bytes"]}), flush=True)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fetch, rows))
    # Checksummed, internally produced runtime with absolute venv interpreter links.
    with tarfile.open(root / "runtime.tar.gz", "r:gz") as tar:
        for item in tar.getmembers():
            safe_relative(item.name)
            if item.isdev():
                raise ValueError("Device file in runtime")
        tar.extractall(root, filter="fully_trusted")
    for item in manifest["asset_archives"]:
        target = root / str(safe_relative(item["extract_parent"]))
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(root / "asset_archives" / item["name"]) as archive:
            for member in archive.infolist():
                safe_relative(member.filename)
            archive.extractall(target)
    return receipt


def stop(process):
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--maximum-worker-seconds", type=int, default=10440)
    args = parser.parse_args()
    started = time.monotonic()
    workflow = os.environ["ASTRA_RUN_ID"]
    if not workflow.startswith("astra-complex-20261006-robocasa-xiaomi-native-"):
        raise ValueError("Unexpected native workflow identity")
    output, root = Path("/opt/astra-results"), Path("/opt/astra-xiaomi")
    output.mkdir(parents=True, exist_ok=False)
    client = archive_client()
    prefix = f"experiments/astra-complex-20261006/{workflow}/results"
    if client.list_objects_v2(Bucket="rldb", Prefix=prefix + "/", MaxKeys=1).get("KeyCount"):
        raise FileExistsError("Native result archive already exists")
    publisher = Publisher(client, output, prefix)
    write_json(output / "worker_started.json", {"workflow": workflow, "started_unix": time.time(),
        "source_revision": os.environ["ASTRA_SOURCE_REVISION"], "stage": args.stage,
        "worker_limit_seconds": args.maximum_worker_seconds})
    write_json(output / "protocol.json", json.loads(args.protocol.read_text()))
    event = threading.Event()

    def publish_loop():
        while not event.wait(30):
            try:
                publisher.publish()
            except Exception as exc:
                print(json.dumps({"archive_retry": type(exc).__name__}), flush=True)

    thread = threading.Thread(target=publish_loop, daemon=True)
    thread.start()
    server = evaluator = None
    returncode = 1
    try:
        receipt = restore(client, args.stage, args.manifest, root)
        write_json(output / "stage_receipt.json", receipt)
        (output / "runtime_freeze.txt").write_text((root / "runtime_freeze.txt").read_text())
        env = {k: v for k, v in os.environ.items() if not k.startswith("R2_")}
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", MIBOT_SERVER_SEED="7",
                   TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
        python = str(root / "runtime/bin/python")
        server_command = [python, "-u", str(root / "sources/xiaomi/deploy/server.py"),
                          "--model", str(root / "weights"), "--host", "127.0.0.1", "--port", "10086"]
        # Native pickle protocol is confined to loopback; only our local evaluator connects.
        with (output / "server.log").open("w") as log:
            server = subprocess.Popen(server_command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                      start_new_session=True)
        ready_deadline = min(started + args.maximum_worker_seconds - 60, time.monotonic() + 600)
        while time.monotonic() < ready_deadline:
            if server.poll() is not None:
                raise RuntimeError("Native server exited before becoming ready")
            try:
                with socket.create_connection(("127.0.0.1", 10086), timeout=1):
                    break
            except OSError:
                time.sleep(2)
        else:
            raise TimeoutError("Native server startup timed out")
        write_json(output / "server_ready.json", {"ready_seconds": time.monotonic() - started,
            "server_command": server_command, "persistent_server_rng_seed": 7})
        command = [python, "-u", "-m", "astra_reversal.complex_manipulation.xiaomi_eval",
                   "--protocol", str(args.protocol), "--output", str(output / "evaluation")]
        with (output / "evaluation.log").open("w") as log:
            evaluator = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True)
        remaining = args.maximum_worker_seconds - (time.monotonic() - started)
        if remaining < 1:
            raise TimeoutError("Restoration exhausted the worker limit")
        returncode = evaluator.wait(timeout=remaining)
        if returncode:
            raise RuntimeError("Native evaluator exited with an error")
    except Exception as exc:
        write_json(output / "worker_error.json", {"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        stop(evaluator)
        stop(server)
        event.set()
        thread.join(timeout=180)
        if thread.is_alive():
            raise RuntimeError("Publisher did not stop; archive completion remains unverified")
        write_json(output / "worker_finished.json", {"returncode": returncode,
            "elapsed_seconds": time.monotonic() - started})
        publisher.publish()


if __name__ == "__main__":
    main()
