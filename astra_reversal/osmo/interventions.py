"""Isolated OSMO L40S workers for the prespecified iterative intervention matrix."""

import importlib.metadata
import json
import os
import platform
import subprocess
import sys

from astra_reversal.config import BenchmarkConfig
from astra_reversal.intervention_search import (
    InterventionSearch,
    aggregate_reports,
    load_protocol,
    write_json,
)
from astra_reversal.osmo.experiment import RESULTS, ROOT
from astra_reversal.osmo.ood_distributed import WorkerArchive
from astra_reversal.records import digest, file_sha256


def assignment(phase, worker):
    if phase == "development" and type(worker) is int and 0 <= worker < 2:
        return {"suite": "libero_10", "case_shard": worker, "case_shards": 2}
    if phase == "evaluation" and type(worker) is int and 0 <= worker < 8:
        return {
            "suite": ("libero_goal_ood", "libero_spatial_ood")[worker // 4],
            "case_shard": worker % 4,
            "case_shards": 4,
        }
    raise ValueError("Invalid intervention phase or worker")


def main():
    import torch
    from huggingface_hub import snapshot_download

    from astra_reversal.intervention_rollout import capture_reset_manifest
    from astra_reversal.libero_runner import configure_libero
    from astra_reversal.policy_adapter import load_policy

    phase = os.environ["ASTRA_INTERVENTION_PHASE"]
    worker = int(os.environ["ASTRA_WORKER_INDEX"])
    target = assignment(phase, worker)
    protocol = load_protocol()
    RESULTS.mkdir(parents=True, exist_ok=False)
    archive = WorkerArchive(worker)
    if torch.cuda.device_count() != 1 or "L40S" not in torch.cuda.get_device_name(0):
        raise RuntimeError("This experiment requires one allocated OSMO L40S")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
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
            )
        },
    }
    write_json(RESULTS / "runtime.json", runtime)
    write_json(RESULTS / "protocol.json", protocol)
    write_json(RESULTS / "progress.json", {"status": "preflight", **target})
    archive.sync()
    reports = []
    try:
        for label, arguments in (
            ("unit", ["--confcutdir=tests/unit/astra", "tests/unit/astra"]),
            (
                "native",
                [
                    "--confcutdir=tests/integration",
                    "tests/integration/test_astra_lerobot_policy.py",
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
        inventory = json.loads(
            (
                ROOT / "astra_reversal/checkpoints/lerobot_pi05_libero_base.json"
            ).read_text()
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
        root = ROOT / (
            "astra_reversal/.deps/libero"
            if phase == "development"
            else "astra_reversal/.deps/libero-ood/third_party/modified_libero"
        )
        benchmark = BenchmarkConfig.preset(target["suite"])
        cases = (
            protocol["development_cases"]
            if phase == "development"
            else protocol["evaluation_cases"]
        )
        manifest = capture_reset_manifest(
            root,
            benchmark,
            seed=protocol["seed"],
            cases=cases,
            output=RESULTS / "reset_manifest.json",
            split="development" if phase == "development" else "followup_adaptation",
        )
        entries = manifest["episodes"][target["case_shard"] :: target["case_shards"]]
        plan = {
            "runtime": runtime,
            "protocol_sha256": digest(protocol),
            "manifest_sha256": manifest["sha256"],
            "assigned_episodes": [entry["episode_id"] for entry in entries],
        }
        write_json(RESULTS / "frozen_plan.json", plan)
        archive.sync()
        _, create = configure_libero(root, benchmark, RESULTS / "runtime_libero")
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
            search = InterventionSearch(
                policy,
                create,
                benchmark,
                entry,
                protocol,
                RESULTS / f"case_{entry['task_id']}_{entry['initial_state_id']}",
                development=phase == "development",
                progress=archive.sync,
            )
            report = search.run()
            reports.append(report)
            write_json(RESULTS / "aggregate.json", aggregate_reports(reports, protocol))
            archive.sync()
        write_json(
            RESULTS / "progress.json",
            {
                "status": "complete",
                "complete_cases": len(reports),
                "assigned_cases": len(entries),
                **target,
            },
        )
    except Exception as exc:
        write_json(RESULTS / "failure.json", {"type": type(exc).__name__})
        raise
    finally:
        archive.sync(include_arrays=True)


if __name__ == "__main__":
    main()
