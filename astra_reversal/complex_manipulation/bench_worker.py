"""Restore checksummed Bench2Dex releases and run two bounded native pilots."""

import argparse
import hashlib
import json
import os
import re
import signal
import socket
import subprocess
import sys
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from worker import Publisher, archive_client, safe_relative, sha256, write_json

ASSET_STAGE = "astra-complex-20261006-bench-stage-2"
ASSET_RECEIPT_SHA256 = (
    "21b6341136e64c9e206136256dee163aa3d6b4e1176fc98db0456d8f45d0103f"
)
TASKS = (
    (
        "73_jigsaw_puzzle_assembly",
        "multi_iiwa7_with_sharpa",
        54,
        1213,
        "bench2dex_jigsaw_pi05",
    ),
    (
        "34_fridge_wine_interhand_pour",
        "multi_ur5_rh5dg2_with_flange",
        38,
        1080,
        "bench2dex_fridge_pi05",
    ),
)


def receipt(client, key, *, expected_sha=None):
    raw = client.get_object(Bucket="rldb", Key=key)["Body"].read(8 * 1024 * 1024)
    if expected_sha and hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError("Immutable stage receipt checksum mismatch")
    return json.loads(raw)


def download(client, record, destination, *, prefix):
    if not record["key"].startswith(prefix) or not 0 < record["bytes"] < 40 * 1024**3:
        raise ValueError("Unexpected archive scope or size")
    destination.parent.mkdir(parents=True, exist_ok=True)
    client.download_file("rldb", record["key"], str(destination))
    if (
        destination.stat().st_size != record["bytes"]
        or sha256(destination) != record["sha256"]
    ):
        raise ValueError("Restored artifact checksum mismatch")
    print(
        json.dumps({"restored": destination.name, "bytes": record["bytes"]}), flush=True
    )


def unpack(archive_path, root):
    root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        for item in archive.getmembers():
            safe_relative(item.name)
            if item.isdev():
                raise ValueError("Device entry in owned archive")
            # Symlinks in our policy venv intentionally preserve absolute paths.
            # Only our checksummed archive, restored at its declared root, uses this.
        archive.extractall(root, filter="fully_trusted")


def stop(process):
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGINT)
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runtime-stage", required=True)
    parser.add_argument("--runtime-receipt-sha256", required=True)
    parser.add_argument("--supplement-receipt-sha256", required=True)
    parser.add_argument("--output", type=Path, default=Path("/opt/astra-results"))
    parser.add_argument("--maximum-worker-seconds", type=int, default=5040)
    args = parser.parse_args()
    workflow = os.environ["ASTRA_RUN_ID"]
    if not re.fullmatch(
        r"astra-complex-20261006-bench-native-[a-z0-9-]+-[0-9]+", workflow
    ):
        raise ValueError("Unexpected native evaluation identity")
    if not re.fullmatch(
        r"astra-complex-20261006-bench-runtime-stage-[0-9]+", args.runtime_stage
    ):
        raise ValueError("Unexpected policy runtime identity")
    began = time.monotonic()
    deadline = began + args.maximum_worker_seconds
    client = archive_client()
    prefix = f"experiments/astra-complex-20261006/{workflow}/results"
    if client.list_objects_v2(Bucket="rldb", Prefix=prefix + "/", MaxKeys=1).get(
        "KeyCount"
    ):
        raise FileExistsError("Refuse to overwrite prior experiment evidence")
    args.output.mkdir(parents=True, exist_ok=False)
    publisher = Publisher(client, args.output, prefix)
    stop_publishing = threading.Event()

    def publish_loop():
        while not stop_publishing.wait(30):
            try:
                publisher.publish()
            except Exception as exc:
                print(json.dumps({"archive_retry": type(exc).__name__}), flush=True)

    thread = threading.Thread(target=publish_loop, daemon=True)
    thread.start()
    write_json(
        args.output / "worker_started.json",
        {
            "workflow": workflow,
            "source_revision": os.environ["ASTRA_SOURCE_REVISION"],
            "scope": "one_anchor_per_task_native_development_pilot",
            "heldout_ood": False,
            "teacher_calls": 0,
            "teacher_tokens": 0,
            "policy_updates": 0,
            "worker_seconds_limit": args.maximum_worker_seconds,
        },
    )
    returncode = 1
    processes = []
    try:
        manifest = json.loads(args.manifest.read_text())
        asset_prefix = f"experiments/astra-complex-20261006/{ASSET_STAGE}/"
        assets = receipt(
            client, asset_prefix + "receipt.json", expected_sha=ASSET_RECEIPT_SHA256
        )
        if assets["manifest_sha256"]["release"] != sha256(args.manifest):
            raise ValueError("Assets and checkpoint manifest differ")
        data = Path("/opt/astra-bench-data")
        data.mkdir(exist_ok=False)

        def fetch_asset(record):
            path = data / safe_relative(record["relative"])
            if record["key"] != asset_prefix + record["relative"]:
                raise ValueError("Asset destination disagrees with its receipt")
            download(client, record, path, prefix=asset_prefix)

        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(fetch_asset, assets["uploaded_files"]))
        unpack(data / "assets.tar.gz", data)
        write_json(args.output / "asset_receipt.json", assets)
        supplement = receipt(
            client,
            asset_prefix + "anchor_supplement/receipt.json",
            expected_sha=args.supplement_receipt_sha256,
        )
        download(
            client,
            supplement["archive"],
            data / "supplement.tar.gz",
            prefix=asset_prefix + "anchor_supplement/",
        )
        unpack(data / "supplement.tar.gz", data)
        write_json(args.output / "supplement_receipt.json", supplement)

        runtime_prefix = f"experiments/astra-complex-20261006/{args.runtime_stage}/"
        runtime = receipt(
            client,
            runtime_prefix + "receipt.json",
            expected_sha=args.runtime_receipt_sha256,
        )
        if (
            runtime["source"] != manifest["sources"]["bench2dex"]
            or runtime["status"]
            != "policy_runtime_imports_validated_gpu_inference_not_run"
        ):
            raise ValueError("Runtime source or validation receipt differs")
        runtime_root = Path("/opt/astra-bench-runtime")
        if runtime_root.exists():
            raise FileExistsError("Policy runtime destination already exists")
        download(
            client, runtime["archive"], data / "runtime.tar.gz", prefix=runtime_prefix
        )
        unpack(data / "runtime.tar.gz", runtime_root)
        source = runtime_root / "Bench2Dex"
        if sha256(source / "policy/pi05/uv.lock") != runtime["uv_lock_sha256"]:
            raise ValueError("Restored frozen environment lock differs")
        (runtime_root / "dex2bench_dataset").symlink_to(
            data / "dex2bench_dataset", target_is_directory=True
        )
        (runtime_root / "teleopdata").symlink_to(
            data / "teleopdata", target_is_directory=True
        )
        write_json(args.output / "runtime_receipt.json", runtime)
        python = source / "policy/pi05/.venv/bin/python"
        helpers = Path(__file__).parent
        environment = {k: v for k, v in os.environ.items() if not k.startswith("R2_")}
        policy_environment = dict(environment)
        for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "LD_PRELOAD"):
            policy_environment.pop(key, None)
        policy_environment["LD_LIBRARY_PATH"] = ":".join(
            p
            for p in policy_environment.get("LD_LIBRARY_PATH", "").split(":")
            if p and not p.startswith(("/isaac-sim", "/workspace/isaaclab"))
        )
        policy_environment.update(
            XLA_PYTHON_CLIENT_PREALLOCATE="false",
            DEX2BENCH_DATASET_ROOT=str(data / "dex2bench_dataset"),
            PYTHONUNBUFFERED="1",
        )
        task_outcomes = []
        for task_index, (task, robot, dof, steps, checkpoint_id) in enumerate(TASKS):
            port = 9000 + task_index
            if deadline - time.monotonic() < 300:
                raise TimeoutError(
                    "Worker budget insufficient to initialize the next pilot"
                )
            task_output = args.output / task
            task_output.mkdir()
            checkpoint = (
                data / "policy_ckpt" / manifest["checkpoints"][checkpoint_id]["prefix"]
            )
            anchor = (
                data
                / "teleopdata/dataset"
                / task
                / "replay-generalization/episode_000000.hdf5"
            )
            if not anchor.is_file():
                raise FileNotFoundError("Declared native reset anchor missing")
            server_command = [
                str(python),
                "-u",
                str(helpers / "bench_policy_server.py"),
                "--source",
                str(source),
                "--checkpoint",
                str(checkpoint),
                "--robot",
                robot,
                "--active-dof",
                str(dof),
                "--output",
                str(task_output / "policy"),
                "--port",
                str(port),
            ]
            simulator_command = [
                "/isaac-sim/python.sh",
                str(helpers / "bench_simulator.py"),
                "--policy-type",
                "REMOTE",
                "--task",
                f"scenes/{task}.yaml",
                "--robot-key",
                robot,
                "--active-dof",
                "--remote-host",
                "127.0.0.1",
                "--remote-port",
                str(port),
                "--enable-rgb",
                "--headless",
                "--seed",
                "100000000",
                "--num-episodes",
                "1",
                "--episode-steps",
                str(steps),
                "--warmup-steps",
                "60",
                "--enable-generalization",
                "--generalization-profile",
                "none",
                "--anchor-hdf5",
                str(anchor),
                "--output-dir",
                str(task_output / "native_results"),
                "--record-dir",
                str(task_output / "recordings"),
                "--record-all",
            ]
            write_json(
                task_output / "commands.json",
                {
                    "server": server_command,
                    "simulator": simulator_command,
                    "restore_seconds": time.monotonic() - began,
                    "action_limit": steps,
                    "physics_steps_per_action": 3,
                    "model_horizon": 20,
                    "execute_prefix": 20,
                    "active_dof": dof,
                    "reset_anchor": str(anchor),
                    "reset_anchor_sha256": sha256(anchor),
                    "unique_anchors": 1,
                },
            )
            sim_env = dict(environment)
            sim_env.update(
                PYTHONEXE=sys.executable,
                ASTRA_BENCH_SOURCE=str(source),
                ASTRA_BENCH_AUDIT=str(task_output / "audit"),
                DEX2BENCH_DATASET_ROOT=str(data / "dex2bench_dataset"),
                DEX2BENCH_PREFLIGHT_DIR=str(task_output / "preflight"),
                DEX2BENCH_RECORD_RES="640",
                DEX2BENCH_RECORD_JPEG="90",
                DEX2BENCH_RECORD_STRIDE="1",
                DEX2BENCH_RECORD_CAMS="cam_stereo_left,cam_stereo_right,cam_wrist_left,cam_wrist_right",
                PYTHONUNBUFFERED="1",
            )
            with (
                (task_output / "policy.log").open("w") as policy_log,
                (task_output / "simulator.log").open("w") as sim_log,
            ):
                server = subprocess.Popen(
                    server_command,
                    cwd=source,
                    env=policy_environment,
                    stdout=policy_log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                processes.append(server)
                ready_deadline = min(deadline, time.monotonic() + 600)
                while True:
                    if server.poll() is not None:
                        raise RuntimeError(
                            "Native policy server exited during initialization"
                        )
                    try:
                        with socket.create_connection(("127.0.0.1", port), timeout=2):
                            break
                    except OSError:
                        if time.monotonic() >= ready_deadline:
                            raise TimeoutError(
                                "Native policy server did not become ready"
                            )
                        time.sleep(2)
                simulator = subprocess.Popen(
                    simulator_command,
                    cwd=source,
                    env=sim_env,
                    stdout=sim_log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                processes.append(simulator)
                try:
                    code = simulator.wait(timeout=max(1, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    stop(simulator)
                    raise TimeoutError(
                        "Native simulation exceeded worker budget"
                    ) from None
                finally:
                    stop(server)
                result_path = task_output / "native_results/per_episode.jsonl"
                rows = (
                    [
                        json.loads(line)
                        for line in result_path.read_text().splitlines()
                        if line.strip()
                    ]
                    if result_path.exists()
                    else []
                )
                completed = [
                    r
                    for r in rows
                    if not r.get("error")
                    and r.get("terminated_reason") in {"stable_success", "max_steps"}
                ]
                outcome = {
                    "task": task,
                    "process_exit_code": code,
                    "native_result_rows": len(rows),
                    "completed_episodes": len(completed),
                    "successes": sum(bool(r.get("success")) for r in completed),
                    "success_rate": sum(bool(r.get("success")) for r in completed)
                    / len(completed)
                    if completed
                    else None,
                    "teacher_calls": 0,
                    "teacher_tokens": 0,
                }
                write_json(task_output / "pilot_summary.json", outcome)
                task_outcomes.append(outcome)
                write_json(args.output / "pilot_summary.json", {"tasks": task_outcomes})
                if code or len(completed) != 1:
                    raise RuntimeError(
                        "Native evaluator did not complete exactly one scored episode"
                    )
        returncode = 0
    except Exception as exc:
        write_json(
            args.output / "worker_error.json",
            {
                "type": type(exc).__name__,
                "message": str(exc)
                if isinstance(
                    exc,
                    (
                        ValueError,
                        RuntimeError,
                        TimeoutError,
                        FileExistsError,
                        FileNotFoundError,
                    ),
                )
                else "See scoped task logs",
            },
        )
    finally:
        for process in reversed(processes):
            stop(process)
        write_json(
            args.output / "worker_finished.json",
            {"returncode": returncode, "wall_seconds": time.monotonic() - began},
        )
        stop_publishing.set()
        thread.join(timeout=120)
        publisher.publish()
    raise SystemExit(returncode)


if __name__ == "__main__":
    main()
