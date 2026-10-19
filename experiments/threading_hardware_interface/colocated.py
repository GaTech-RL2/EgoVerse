"""One allocated H100 node: pinned vLLM plus the pinned native image in chroot."""

import fcntl
import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path

from astra_reversal.hardware_interface.common import write_json

NATIVE_IMAGE = "docker.io/library/python@sha256:2c5a13f5b99b8c50187668f0c27157c43fef5b3615036a396fd7ee7d76c73307"
SERVER_MODULE = "experiments.libero_hardware_interface.tools.self_hosted_server"


def child_environments(base, artifacts):
    server = {
        key: value
        for key, value in base.items()
        if not key.startswith(("OPENAI_", "R2_"))
        and key not in ("HARDWARE_API_KEY_FILE", "INFERENCE_API_KEY")
    }
    server["HARDWARE_SERVER_ARTIFACTS"] = str(artifacts / "serving")
    native = {
        key: value
        for key, value in base.items()
        if key not in ("LD_LIBRARY_PATH", "LD_PRELOAD", "PYTHONPATH", "VIRTUAL_ENV")
        and not key.startswith(("VLLM_", "CUDA_", "NVIDIA_", "HF_", "PYTORCH_"))
    }
    native.update(
        HARDWARE_RENDER_BACKEND="osmesa",
        MUJOCO_GL="osmesa",
        PYOPENGL_PLATFORM="osmesa",
        CUDA_VISIBLE_DEVICES="",
        LP_NUM_THREADS="2",
        HARDWARE_BOOTSTRAP_FILE_LOG="1",
    )
    return server, native


def mirror(source, destination, seen):
    """Preserve evaluator traces without copying credentials or scratch runtimes."""
    if not source.exists():
        return
    for parent, directories, filenames in os.walk(source):
        directories[:] = [
            name
            for name in directories
            if "jail" not in name
            and name not in ("scratch-runtime", "__pycache__")
            and not (Path(parent) / name).is_symlink()
        ]
        for name in filenames:
            path = Path(parent) / name
            if path.is_symlink() or not path.is_file():
                continue
            if "api-key" in path.name:
                raise ValueError("private_credential_in_artifacts")
            relative = path.relative_to(source)
            info = path.stat()
            signature = (info.st_mtime_ns, info.st_size)
            if seen.get(str(relative)) == signature:
                continue
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".copying")
            shutil.copy2(path, temporary)
            os.replace(temporary, target)
            seen[str(relative)] = signature


def rootfs(cache, artifacts):
    print(json.dumps({"phase": "native_image_copy", "image": NATIVE_IMAGE}), flush=True)
    oci = cache / "native-image-oci"
    with (cache / "native-image.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        with (artifacts / "native-image-copy.log").open("w") as log:
            subprocess.run(
                [
                    "skopeo",
                    "--override-os",
                    "linux",
                    "--override-arch",
                    "amd64",
                    "copy",
                    "--format",
                    "oci",
                    "docker://" + NATIVE_IMAGE,
                    "oci:" + str(oci) + ":native",
                ],
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
    bundle = Path("/workspace/native-bundle")
    with (artifacts / "native-image-unpack.log").open("w") as log:
        subprocess.run(
            [
                "umoci",
                "unpack",
                "--rootless",
                "--image",
                str(oci) + ":native",
                str(bundle),
            ],
            check=True,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    root = bundle / "rootfs"
    for name, minor in (("null", 3), ("random", 8), ("urandom", 9)):
        path = root / "dev" / name
        path.parent.mkdir(exist_ok=True)
        if not path.is_char_device():
            path.unlink(missing_ok=True)
            os.mknod(path, stat.S_IFCHR | 0o666, os.makedev(1, minor))
        path.chmod(0o666)
    for filename in ("resolv.conf", "hosts", "nsswitch.conf"):
        shutil.copyfile(Path("/etc") / filename, root / "etc" / filename)
    shutil.copyfile(
        "/tmp/hardware-payload.tar.gz.b64", root / "tmp/hardware-payload.tar.gz.b64"
    )
    shutil.copyfile(
        Path(__file__).with_name("bootstrap.sh"), root / "tmp/hardware-bootstrap.sh"
    )
    if os.environ["HARDWARE_STAGE"] == "pilot":
        credential = root / "run/hardware-inference/inference_api_key"
        credential.parent.mkdir(parents=True)
        shutil.copyfile("/run/hardware-inference/inference_api_key", credential)
        credential.chmod(0o600)
    print(json.dumps({"phase": "native_image_ready"}), flush=True)
    return root


def forward_log(path, offset):
    if path.exists():
        with path.open("rb") as stream:
            stream.seek(offset)
            raw = stream.read()
        sys.stdout.buffer.write(raw)
        sys.stdout.flush()
        return offset + len(raw)
    return offset


def main():
    cache = Path(os.environ["HARDWARE_SERVING_CACHE"])
    cache.mkdir(parents=True, exist_ok=True)
    artifacts = cache / "colocated" / os.environ["HARDWARE_WORKFLOW"]
    artifacts.mkdir(parents=True, exist_ok=True, mode=0o700)
    artifacts.chmod(0o700)
    # A durable exclusive claim blocks accidental rescheduling of this cohort.
    with (artifacts / "claim.json").open("x") as stream:
        json.dump(
            {
                "workflow": os.environ["HARDWARE_WORKFLOW"],
                "source_commit": os.environ["HARDWARE_SOURCE_COMMIT"],
            },
            stream,
        )
    write_json(
        artifacts / "runtime.json",
        {
            "native_image": NATIVE_IMAGE,
            "server_image": os.environ["HARDWARE_SERVER_IMAGE"],
            "simulator_backend": "osmesa",
            "allocated_gpus": 8,
            "simulator_uses_gpu": False,
            "separate_pinned_native_rootfs": True,
            "automatic_trial_retries": False,
        },
    )
    with (artifacts / "gpu.txt").open("w") as stream:
        subprocess.run(["nvidia-smi"], check=True, stdout=stream)
    native_root = rootfs(cache, artifacts)
    server_env, native_env = child_environments(os.environ, artifacts)
    stop = threading.Event()
    snapshot_lock = threading.RLock()
    seen = {}
    native_work = native_root / "workspace/threading-study"

    def snapshot():
        with snapshot_lock:
            mirror(native_work / "artifacts", artifacts / "evaluation", seen)
            for name in ("bootstrap.log", "dependency-install.log", "apt-install.log"):
                source = native_work / name
                if source.is_file():
                    shutil.copyfile(source, artifacts / name)

    def snapshots():
        while not stop.wait(5):
            try:
                snapshot()
            except OSError as error:
                write_json(
                    artifacts / "mirror-error.json",
                    {"error_class": type(error).__name__},
                )

    snapshot_thread = threading.Thread(target=snapshots, daemon=True)
    snapshot_thread.start()
    jobs = []

    def terminate(signum, _):
        write_json(
            artifacts / "interrupted.json",
            {"signal": signum, "automatic_trial_retries": False},
        )
        for job in jobs:
            if job.poll() is None:
                os.killpg(job.pid, signal.SIGTERM)
        stop.set()
        snapshot_thread.join(timeout=10)
        snapshot()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    server = None
    if os.environ["HARDWARE_STAGE"] == "pilot":
        server = subprocess.Popen(
            [sys.executable, "-m", SERVER_MODULE],
            env=server_env,
            start_new_session=True,
        )
        jobs.append(server)
    native = subprocess.Popen(
        ["chroot", str(native_root), "/bin/bash", "/tmp/hardware-bootstrap.sh"],
        env=native_env,
        start_new_session=True,
    )
    jobs.append(native)
    native_log_offset = 0
    print(
        json.dumps(
            {
                "phase": "native_started",
                "model_server_started": server is not None,
                "allocated_gpus": 8,
                "simulator_backend": "osmesa",
            }
        ),
        flush=True,
    )
    try:
        while native.poll() is None:
            native_log_offset = forward_log(
                native_work / "bootstrap.log", native_log_offset
            )
            if server is not None and server.poll() not in (None, 0):
                raise RuntimeError("colocated_model_server_failed")
            time.sleep(2)
        forward_log(native_work / "bootstrap.log", native_log_offset)
        if native.returncode:
            raise RuntimeError(
                "colocated_native_process_failed_" + str(native.returncode)
            )
        if server is not None:
            if server.wait(timeout=60):
                raise RuntimeError("colocated_model_server_failed")
        write_json(
            artifacts / "completed.json",
            {
                "native_exit_code": 0,
                "server_exit_code": 0 if server else None,
                "scored_result": server is not None,
            },
        )
    finally:
        stop.set()
        snapshot_thread.join(timeout=10)
        snapshot()
        for job in jobs:
            if job.poll() is None:
                os.killpg(job.pid, signal.SIGTERM)
                try:
                    job.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(job.pid, signal.SIGKILL)
                    job.wait()


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        destination = (
            Path(os.environ["HARDWARE_SERVING_CACHE"])
            / "colocated"
            / os.environ["HARDWARE_WORKFLOW"]
        )
        write_json(
            destination / "supervisor-failure.json",
            {"error_class": type(error).__name__, "reason": str(error)},
        )
        raise
