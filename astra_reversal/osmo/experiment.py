"""One L40S development comparison; uploads only this run's own artifacts."""

import gc
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import numpy as np

from astra_reversal.records import file_sha256, to_numpy

ROOT = Path.cwd()
RESULTS = ROOT / "astra_reversal/artifacts/osmo"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


class Archive:
    def __init__(self):
        import boto3

        self.prefix = (
            "experiments/astra-reversal-20260924/" + os.environ["ASTRA_RUN_ID"]
        )
        self.client = boto3.client(
            "s3",
            endpoint_url=os.environ["R2_ENDPOINT_URL"],
            aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        )
        if self.client.list_objects_v2(
            Bucket="rldb", Prefix=self.prefix + "/", MaxKeys=1
        ).get("KeyCount"):
            raise FileExistsError("This run's remote artifact prefix already exists")
        self.uploaded = {}

    def sync(self, *, include_arrays=False):
        files = list(RESULTS.rglob("*")) + [ROOT / "bootstrap.log"]
        for path in files:
            if not path.is_file() or "arrays" in path.parts:
                continue
            stamp = (path.stat().st_size, path.stat().st_mtime_ns)
            if self.uploaded.get(path) == stamp:
                continue
            relative = (
                path.relative_to(RESULTS)
                if path.is_relative_to(RESULTS)
                else Path(path.name)
            )
            self.client.upload_file(str(path), "rldb", f"{self.prefix}/{relative}")
            self.uploaded[path] = stamp
        if include_arrays:
            # Thousands of per-step arrays are retained losslessly in one
            # archive, avoiding one remote object upload for each small array.
            bundle = RESULTS.parent / "artifacts.tar.gz"
            with tarfile.open(bundle, "w:gz", compresslevel=1) as archive:
                archive.add(RESULTS, arcname="results")
            self.client.upload_file(
                str(bundle), "rldb", f"{self.prefix}/artifacts.tar.gz"
            )


def main():
    import torch
    from huggingface_hub import snapshot_download

    # This import must succeed: optional native tests may otherwise skip.
    from lerobot.policies.pi05 import modeling_pi05  # noqa: F401
    from lerobot.processor import PolicyProcessorPipeline  # noqa: F401

    RESULTS.mkdir(parents=True, exist_ok=False)
    assert torch.cuda.device_count() == 1, "Expected exactly one allocated GPU"
    assert "L40S" in torch.cuda.get_device_name(0), "Expected an L40S allocation"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    runtime = {
        "gpu": torch.cuda.get_device_name(0),
        "python": platform.python_version(),
        "packages": {
            p: importlib.metadata.version(p)
            for p in (
                "torch",
                "torchvision",
                "numpy",
                "mujoco",
                "robosuite",
                "transformers",
                "lerobot",
            )
        },
        "workflow": os.environ["ASTRA_RUN_ID"],
        "payload_sha256": os.environ["PAYLOAD_SHA256"],
    }
    write_json(RESULTS / "runtime.json", runtime)
    print(json.dumps({"phase": "gpu_verified", **runtime}), flush=True)
    archive = Archive()

    def command(label, *args):
        began = time.time()
        print(json.dumps({"phase": label, "status": "started"}), flush=True)
        with (RESULTS / f"{label}.log").open("w") as log:
            result = subprocess.run(
                [sys.executable, *args], stdout=log, stderr=subprocess.STDOUT
            )
        archive.sync()
        print(
            json.dumps(
                {
                    "phase": label,
                    "returncode": result.returncode,
                    "seconds": time.time() - began,
                }
            ),
            flush=True,
        )
        if result.returncode:
            print((RESULTS / f"{label}.log").read_text()[-7000:], flush=True)
            raise RuntimeError(f"Phase {label} failed")

    try:
        inventory = json.loads(
            (
                ROOT / "astra_reversal/checkpoints/lerobot_pi05_libero_base.json"
            ).read_text()
        )
        path = ROOT / inventory["local_directory"]
        print(
            json.dumps(
                {"phase": "checkpoint_download", "revision": inventory["revision"]}
            ),
            flush=True,
        )
        snapshot_download(
            inventory["repo_id"],
            revision=inventory["revision"],
            local_dir=path,
            token=False,
        )
        for entry in inventory["files"]:
            assert file_sha256(path / entry["path"]) == entry["sha256"], entry["path"]

        command(
            "unit_tests",
            "-m",
            "pytest",
            "--confcutdir=tests/unit/astra",
            "tests/unit/astra",
            "-q",
        )
        os.environ["EGOVERSE_TEST_PI05_INPUT_ASSETS"] = str(
            ROOT / "astra_reversal/.deps/reference/pi05_libero"
        )
        command(
            "native_tests",
            "-m",
            "pytest",
            "--confcutdir=tests/integration",
            "tests/integration/test_astra_lerobot_policy.py",
            "-q",
        )
        from astra_reversal.flow import error_metrics
        from astra_reversal.lerobot_policy import FrozenLeRobotPI05

        print(json.dumps({"phase": "weight_and_sampler_verification"}), flush=True)
        policy = FrozenLeRobotPI05.load(
            path,
            "cuda",
            inventory["source_url"],
            tokenizer_path="astra_reversal/.deps/tokenizers/paligemma-3b-pt-224",
        )
        probe = np.load(
            "astra_reversal/.deps/reference/cpu_libero_probe.npz", allow_pickle=False
        )
        observation = {
            key: probe[key]
            for key in (
                "observation/image",
                "observation/wrist_image",
                "observation/state",
            )
        }
        condition = policy.prepare(
            observation, "cpu_libero_probe", str(probe["prompt"])
        )
        noise = policy.tensor(probe["noise"])
        generated = policy.sample(condition, noise, steps=10)
        native = policy.reference_actions(condition, noise, steps=10)
        report = {
            "checkpoint": policy.metadata,
            "strict_load": True,
            "all_parameters_frozen": not any(
                p.requires_grad for p in policy.policy.parameters()
            ),
            "cpu_cuda_internal_difference": error_metrics(
                probe["endpoint"], generated.value
            ),
            "cuda_native_sampler_parity": error_metrics(
                native, to_numpy(generated.value)[0, :, :7]
            ),
        }
        write_json(RESULTS / "weight_verification.json", report)
        assert report["cuda_native_sampler_parity"]["max_abs"] <= 1e-5
        print(
            json.dumps(
                {
                    "phase": "weight_verification",
                    **{k: v for k, v in report.items() if k != "checkpoint"},
                }
            ),
            flush=True,
        )
        del condition, generated, policy, noise
        gc.collect()
        torch.cuda.empty_cache()
        archive.sync()

        native_config = json.loads(
            (
                ROOT / "astra_reversal/configs/pi05_libero_native_baseline.json"
            ).read_text()
        )
        native_config["policy"]["device"] = "cuda"
        reference_config = json.loads(
            (
                ROOT / "astra_reversal/configs/pi05_libero_openpi_inputs_baseline.json"
            ).read_text()
        )
        reference_config["policy"]["device"] = "cuda"
        heun_config = json.loads(json.dumps(reference_config))
        heun_config["flow"]["integrator"] = "heun"
        for label, config in (
            ("native", native_config),
            ("openpi_inputs", reference_config),
            ("openpi_inputs_heun", heun_config),
        ):
            write_json(RESULTS / f"{label}.json", config)
        common = ["--libero-root", "astra_reversal/.deps/libero"]
        manifest = str(RESULTS / "development_manifest.json")
        command(
            "manifest",
            "-m",
            "astra_reversal",
            "make-manifest",
            "--config",
            str(RESULTS / "native.json"),
            *common,
            "--tasks",
            "0",
            "1",
            "2",
            "--trials",
            "1",
            "--output",
            manifest,
        )
        # Both profiles receive exactly the same prescribed resets and budgets.
        for label in ("native", "openpi_inputs"):
            config = str(RESULTS / f"{label}.json")
            command(
                label + "_baseline",
                "-m",
                "astra_reversal",
                "run",
                "--config",
                config,
                "--manifest",
                manifest,
                *common,
                "--output",
                str(RESULTS / (label + "_baseline")),
            )
            command(
                label + "_euler_diagnostics",
                "-m",
                "astra_reversal",
                "diagnostics",
                "--config",
                config,
                "--manifest",
                manifest,
                *common,
                "--resolutions",
                "10",
                "20",
                "50",
                "--action-atol",
                "0.02",
                "--noise-atol",
                "0.1",
                "--parity-atol",
                "0.00001",
                "--output",
                str(RESULTS / (label + "_euler_diagnostics.json")),
            )
        command(
            "openpi_inputs_heun_diagnostics",
            "-m",
            "astra_reversal",
            "diagnostics",
            "--config",
            str(RESULTS / "openpi_inputs_heun.json"),
            "--manifest",
            manifest,
            *common,
            "--resolutions",
            "50",
            "100",
            "200",
            "--action-atol",
            "0.02",
            "--noise-atol",
            "0.1",
            "--parity-atol",
            "0.00001",
            "--output",
            str(RESULTS / "openpi_inputs_heun_diagnostics.json"),
        )
        write_json(
            RESULTS / "status.json",
            {
                "status": "development_baselines_and_diagnostics_complete",
                "full_benchmark_run": False,
                "astra_calls_run": False,
            },
        )
    except Exception as exc:
        write_json(
            RESULTS / "status.json",
            {"status": "failed", "error": f"{type(exc).__name__}: {exc}"},
        )
        raise
    finally:
        archive.sync(include_arrays=True)
    print(
        json.dumps({"phase": "complete", "artifacts": "s3://rldb/" + archive.prefix}),
        flush=True,
    )


if __name__ == "__main__":
    main()
