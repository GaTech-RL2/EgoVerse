"""Restore the pinned CPU-stage artifacts and retain an OSMO pilot's evidence."""

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import tarfile
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def archive_client():
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT_URL"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        config=Config(signature_version="s3v4", connect_timeout=15, read_timeout=90),
    )


def safe_relative(value):
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("Archive path must be relative and contained")
    return path


def restore(client, *, stage, manifest_path, root):
    if not re.fullmatch(r"astra-complex-20261006-stage-[0-9]+", stage):
        raise ValueError("Restore only this study's CPU staging artifacts")
    prefix = f"experiments/astra-complex-20261006/{stage}/"
    raw = client.get_object(Bucket="rldb", Key=prefix + "receipt.json")["Body"].read(
        1024 * 1024
    )
    receipt = json.loads(raw)
    if (
        receipt["workflow"] != stage
        or receipt["status"] != "staged_not_evaluated"
        or receipt["manifest_sha256"] != sha256(manifest_path)
    ):
        raise ValueError("The staged receipt does not match the release manifest")
    root.mkdir(exist_ok=False, parents=True)
    files = receipt["uploaded_files"]
    destinations = set()
    for item in files:
        relative = safe_relative(item["relative"])
        if str(relative) in destinations or item["key"] != prefix + str(relative):
            raise ValueError("Duplicate or unscoped staged file")
        if not 0 < item["bytes"] < 30 * 1024**3:
            raise ValueError("Unexpected staged file size")
        destinations.add(str(relative))

    def download(item):
        path = root / item["relative"]
        path.parent.mkdir(parents=True, exist_ok=True)
        client.download_file("rldb", item["key"], str(path))
        if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise ValueError("Staged artifact checksum mismatch")
        print(
            json.dumps({"restored": item["relative"], "bytes": item["bytes"]}),
            flush=True,
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(download, files))
    # This is our own checksummed runtime, including absolute interpreter links
    # in the venv. Its paths and base image must be preserved on the GPU worker.
    with tarfile.open(root / "runtime.tar.gz", "r:gz") as archive:
        for entry in archive.getmembers():
            safe_relative(entry.name)
            if entry.isdev():
                raise ValueError("Device entry in runtime archive")
        archive.extractall(root, filter="fully_trusted")
    manifest = json.loads(manifest_path.read_text())
    for item in manifest["robocasa_asset_archives"]:
        parent = root / safe_relative(item["extract_parent"])
        parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(root / "asset_archives" / item["name"]) as archive:
            for entry in archive.infolist():
                safe_relative(entry.filename)
            archive.extractall(parent)
    return receipt


class Publisher:
    def __init__(self, client, directory, prefix):
        self.client, self.directory, self.prefix = client, directory, prefix
        self.uploaded = {}

    def publish(self):
        for path in sorted(self.directory.rglob("*")):
            if not path.is_file() or path.suffix == ".tmp":
                continue
            relative = path.relative_to(self.directory).as_posix()
            before = path.stat()
            stamp = (before.st_size, before.st_mtime_ns)
            if self.uploaded.get(relative, {}).get("stamp") == stamp:
                continue
            key = f"{self.prefix}/{relative}"
            self.client.upload_file(str(path), "rldb", key)
            actual_hash = sha256(path)
            after = path.stat()
            # A live JSONL/video may grow while being uploaded. Only mark a
            # stable file verified; the final flush repeats every changed file.
            if stamp == (after.st_size, after.st_mtime_ns):
                self.uploaded[relative] = {
                    "key": key,
                    "bytes": after.st_size,
                    "sha256": actual_hash,
                    "stamp": stamp,
                }
        self.client.put_object(
            Bucket="rldb",
            Key=f"{self.prefix}/archive_receipt.json",
            Body=json.dumps(
                {"files": self.uploaded, "updated_unix": time.time()}, indent=2
            ).encode(),
            ContentType="application/json",
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke-actions", type=int, default=5)
    parser.add_argument("--full-episodes", action="store_true")
    parser.add_argument(
        "--tasks", nargs="+", default=["LoadPreparedFood", "PackIdenticalLunches"]
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[9000])
    parser.add_argument("--maximum-worker-seconds", type=int, default=840)
    args = parser.parse_args()
    workflow = os.environ["ASTRA_RUN_ID"]
    if not re.fullmatch(r"astra-complex-20261006-robocasa-[a-z0-9-]+-[0-9]+", workflow):
        raise ValueError("Unexpected pilot workflow identity")
    started = time.time()
    client = archive_client()
    prefix = f"experiments/astra-complex-20261006/{workflow}/results"
    if client.list_objects_v2(Bucket="rldb", Prefix=prefix + "/", MaxKeys=1).get(
        "KeyCount"
    ):
        raise FileExistsError("Refuse to overwrite an existing run")
    args.output.mkdir(parents=True, exist_ok=False)
    publisher = Publisher(client, args.output, prefix)
    write_json(
        args.output / "worker_started.json",
        {
            "workflow": workflow,
            "started_unix": started,
            "source_revision": os.environ["ASTRA_SOURCE_REVISION"],
            "scope": "full_episodes" if args.full_episodes else "smoke_only",
            "worker_limit_seconds": args.maximum_worker_seconds,
        },
    )
    stop_publishing = threading.Event()

    def publish_loop():
        while not stop_publishing.wait(30):
            try:
                publisher.publish()
            except Exception as exc:
                print(json.dumps({"archive_retry": type(exc).__name__}), flush=True)

    thread = threading.Thread(target=publish_loop, daemon=True)
    thread.start()
    returncode = 1
    try:
        root = Path("/opt/astra-complex")
        receipt = restore(
            client, stage=args.stage, manifest_path=args.manifest, root=root
        )
        write_json(args.output / "stage_receipt.json", receipt)
        # CPU import validation found a dependency used by the upstream tokenizer
        # but absent from that fork's pyproject. Keep the original stage immutable.
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "--python",
                str(root / "runtime/bin/python"),
                "install",
                "--quiet",
                "--no-deps",
                "chex==0.1.89",
                "tqdm-loggable==0.2",
            ],
            check=True,
        )
        manifest = json.loads(args.manifest.read_text())
        checkpoint = (
            root / "weights" / manifest["checkpoints"]["robocasa_pi05"]["prefix"]
        )
        command = [
            str(root / "runtime/bin/python"),
            "-u",
            "-m",
            "astra_reversal.complex_manipulation.robocasa_eval",
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(args.output / "evaluation"),
            "--tasks",
            *args.tasks,
            "--seeds",
            *map(str, args.seeds),
        ]
        if not args.full_episodes:
            command += ["--smoke-actions", str(args.smoke_actions)]
        write_json(
            args.output / "evaluation_command.json",
            {"argv": command, "restore_seconds": time.time() - started},
        )
        remaining = args.maximum_worker_seconds - (time.time() - started)
        if remaining < 60:
            raise TimeoutError(
                "Restoration left less than one minute of the worker budget"
            )
        environment = {**os.environ, "PYTHONPATH": str(args.project)}
        # The model process requires no storage credentials.
        environment = {k: v for k, v in environment.items() if not k.startswith("R2_")}
        with (args.output / "evaluation.log").open("w") as log:
            process = subprocess.Popen(
                command,
                env=environment,
                cwd=args.project,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                returncode = process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                returncode = 124
    except Exception as exc:
        # Messages can contain signed transport URLs. Keep them out of logs;
        # the exception type and code-stage receipts identify the failed step.
        write_json(args.output / "worker_error.json", {"type": type(exc).__name__})
        print(json.dumps({"worker_error": type(exc).__name__}), flush=True)
    finally:
        stop_publishing.set()
        thread.join(timeout=100)
        write_json(
            args.output / "worker_finished.json",
            {"returncode": returncode, "worker_seconds": time.time() - started},
        )
        publisher.publish()
    print(
        json.dumps(
            {"workflow": workflow, "returncode": returncode, "results_prefix": prefix}
        ),
        flush=True,
    )
    return returncode


if __name__ == "__main__":
    sys.exit(main())
