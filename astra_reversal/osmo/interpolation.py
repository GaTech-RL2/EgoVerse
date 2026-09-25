"""L40S bank extraction and observed interpolation workers, each with its own archive."""

import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from astra_reversal.config import BenchmarkConfig
from astra_reversal.interpolation_search import (
    InterpolationSearch,
    aggregate_reports,
    load_protocol,
)
from astra_reversal.intervention_search import write_json
from astra_reversal.osmo.experiment import RESULTS, ROOT
from astra_reversal.osmo.ood_distributed import WorkerArchive
from astra_reversal.records import digest, file_sha256


def assignment(phase, worker):
    if type(worker) is not int:
        raise ValueError("Worker index must be an integer")
    if phase == "bank" and 0 <= worker < 8:
        from astra_reversal.interpolation_catalog import donor_catalog

        return {"source_ids": [row["source_id"] for row in donor_catalog()[worker::8]]}
    if phase == "development" and 0 <= worker < 3:
        return {
            "suite": "libero_goal_ood" if worker == 0 else "libero_spatial_ood",
            "task_id": (6, 2, 8)[worker],
        }
    if phase == "evaluation" and 0 <= worker < 8:
        return {
            "suite": ("libero_goal_ood", "libero_spatial_ood")[worker // 4],
            "case_shard": worker % 4,
            "case_shards": 4,
        }
    raise ValueError("Invalid interpolation phase or worker")


def native_preflight(archive):
    # Import the implementation explicitly: optional test skips cannot pass a
    # worker whose model runtime is absent.
    from lerobot.policies.pi05 import modeling_pi05  # noqa: F401

    for label, arguments in (
        ("unit", ["--confcutdir=tests/unit/astra", "tests/unit/astra"]),
        (
            "native",
            [
                "--confcutdir=tests/integration",
                "tests/integration/test_astra_lerobot_policy.py",
                "tests/integration/test_astra_interpolation_policy.py",
            ],
        ),
    ):
        environment = dict(
            os.environ,
            EGOVERSE_TEST_PI05_INPUT_ASSETS=str(
                ROOT / "astra_reversal/.deps/reference/pi05_libero"
            ),
        )
        with (RESULTS / f"{label}_tests.log").open("w") as log:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", *arguments, "-q", "-rs"],
                stdout=log,
                stderr=subprocess.STDOUT,
                env=environment,
            )
        archive.sync()
        if result.returncode:
            raise RuntimeError(f"Standalone {label} tests failed")


def load_frozen_policy():
    from huggingface_hub import snapshot_download

    from astra_reversal.policy_adapter import load_policy

    inventory = json.loads(
        (ROOT / "astra_reversal/checkpoints/lerobot_pi05_libero_base.json").read_text()
    )
    checkpoint = ROOT / inventory["local_directory"]
    snapshot_download(
        inventory["repo_id"],
        revision=inventory["revision"],
        local_dir=checkpoint,
        token=False,
    )
    for item in inventory["files"]:
        if file_sha256(checkpoint / item["path"]) != item["sha256"]:
            raise ValueError("Checkpoint inventory mismatch")
    policy = load_policy(
        checkpoint,
        device="cuda",
        provenance=inventory["repo_id"] + "@" + inventory["revision"],
        training_overlap="unknown",
        tokenizer_path=ROOT / "astra_reversal/.deps/tokenizers/paligemma-3b-pt-224",
        input_profile="openpi_libero",
        reference_assets=ROOT / "astra_reversal/.deps/reference/pi05_libero",
    )
    if any(parameter.requires_grad for parameter in policy.policy.parameters()):
        raise ValueError("Policy is not fully frozen")
    write_json(RESULTS / "checkpoint.json", policy.metadata)
    return policy


def load_banks(archive, manifest_path):
    from astra_reversal.interpolation_bank import load_bank
    from astra_reversal.interpolation_catalog import donor_catalog

    manifest = json.loads(Path(manifest_path).read_text())
    expected = {row["source_id"] for row in donor_catalog()}
    if set(manifest["donors"]) != expected:
        raise ValueError("Bank inventory does not contain the declared nine donors")
    banks = {}
    for source_id, entry in manifest["donors"].items():
        directory = ROOT / "astra_reversal/.deps/interpolation-banks" / source_id
        directory.mkdir(parents=True, exist_ok=False)
        for file in entry["files"]:
            filename = file["name"]
            if Path(filename).name != filename:
                raise ValueError("Bank filename must be a basename")
            path = directory / filename
            archive.client.download_file("rldb", file["key"], str(path))
            if file_sha256(path) != file["sha256"]:
                raise ValueError("Text latent bank artifact hash mismatch")
        banks[source_id] = load_bank(directory)
    shutil.copyfile(manifest_path, RESULTS / "bank_inventory.json")
    return banks


def main():
    import torch

    phase = os.environ["ASTRA_INTERPOLATION_PHASE"]
    worker = int(os.environ["ASTRA_WORKER_INDEX"])
    target = assignment(phase, worker)
    RESULTS.mkdir(parents=True, exist_ok=False)
    archive = WorkerArchive(worker)
    if torch.cuda.device_count() != 1 or "L40S" not in torch.cuda.get_device_name(0):
        raise RuntimeError("This experiment requires one allocated OSMO L40S")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(2)
    runtime = {
        "workflow": os.environ["ASTRA_RUN_ID"],
        "phase": phase,
        "worker": worker,
        "assignment": target,
        "gpu": torch.cuda.get_device_name(0),
        "tf32": False,
        "python": platform.python_version(),
        "payload_sha256": os.environ["PAYLOAD_SHA256"],
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "numpy",
                "mujoco",
                "robosuite",
                "transformers",
                "lerobot",
                "pyarrow",
            )
        },
    }
    write_json(RESULTS / "runtime.json", runtime)
    write_json(RESULTS / "progress.json", {"status": "preflight", **target})
    archive.sync()
    try:
        native_preflight(archive)
        if phase == "bank":
            from astra_reversal.interpolation_bank import extract_donor

            policy = load_frozen_policy()
            manifests = {}
            for source_id in target["source_ids"]:
                write_json(
                    RESULTS / "progress.json",
                    {
                        "status": "extracting",
                        "source_id": source_id,
                        "completed": list(manifests),
                        **target,
                    },
                )
                archive.sync()
                manifests[source_id] = extract_donor(
                    policy,
                    source_id,
                    output_dir=RESULTS / "banks" / source_id,
                    cache_dir=ROOT / "astra_reversal/.deps/training-donors",
                    progress_every=100,
                )
                write_json(RESULTS / "banks.json", manifests)
                archive.sync()
        else:
            from astra_reversal.interpolation_catalog import donor_catalog, oracle_for
            from astra_reversal.intervention_rollout import capture_reset_manifest
            from astra_reversal.libero_runner import configure_libero

            protocol = load_protocol()
            if phase == "development":
                protocol["seed"] = protocol["development_seed"]
            write_json(RESULTS / "protocol.json", protocol)
            root = ROOT / "astra_reversal/.deps/libero-ood/third_party/modified_libero"
            benchmark = BenchmarkConfig.preset(target["suite"])
            cases = (
                protocol["development_cases"][target["suite"]]
                if phase == "development"
                else protocol["evaluation_cases"]
            )
            manifest = capture_reset_manifest(
                root,
                benchmark,
                seed=protocol["seed"],
                cases=cases,
                output=RESULTS / "reset_manifest.json",
                split="development"
                if phase == "development"
                else "followup_adaptation",
            )
            entries = (
                [
                    row
                    for row in manifest["episodes"]
                    if row["task_id"] == target["task_id"]
                ]
                if phase == "development"
                else manifest["episodes"][target["case_shard"] :: target["case_shards"]]
            )
            bank_path = (
                ROOT / "astra_reversal/.deps/interpolation-inputs/bank_inventory.json"
            )
            write_json(
                RESULTS / "frozen_plan.json",
                {
                    "runtime": runtime,
                    "protocol_sha256": digest(protocol),
                    "manifest_sha256": manifest["sha256"],
                    "bank_inventory_sha256": file_sha256(bank_path),
                    "assigned_episodes": [row["episode_id"] for row in entries],
                },
            )
            archive.sync()
            banks = load_banks(archive, bank_path)
            _, create = configure_libero(root, benchmark, RESULTS / "runtime_libero")
            policy = load_frozen_policy()
            reports = []
            for index, entry in enumerate(entries):
                write_json(
                    RESULTS / "progress.json",
                    {
                        "status": "running",
                        "case": entry["episode_id"],
                        "complete_cases": index,
                        "assigned_cases": len(entries),
                        **target,
                    },
                )
                search = InterpolationSearch(
                    policy,
                    create,
                    benchmark,
                    entry,
                    protocol,
                    RESULTS / f"case_{entry['task_id']}_{entry['initial_state_id']}",
                    banks=banks,
                    catalog=donor_catalog(),
                    oracle=oracle_for(target["suite"], entry["task_id"]).metadata(),
                    development=phase == "development",
                    progress=archive.sync,
                )
                reports.append(search.run())
                write_json(
                    RESULTS / "aggregate.json", aggregate_reports(reports, protocol)
                )
                archive.sync()
        write_json(RESULTS / "progress.json", {"status": "complete", **target})
    except Exception as exc:
        write_json(RESULTS / "failure.json", {"type": type(exc).__name__})
        raise
    finally:
        archive.sync(include_arrays=True)


if __name__ == "__main__":
    main()
