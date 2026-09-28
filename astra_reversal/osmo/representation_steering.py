"""Run representation interventions and preserve each completed OOD case."""

import importlib.metadata
import os
import platform

from astra_reversal.config import BenchmarkConfig
from astra_reversal.image_donor_bank import load_library
from astra_reversal.intervention_search import write_json
from astra_reversal.osmo.experiment import RESULTS, ROOT
from astra_reversal.osmo.frs_policy_improvement import (
    frozen_parameter_receipt,
    publish_task,
)
from astra_reversal.osmo.interpolation import (
    assignment,
    load_banks,
    load_frozen_policy,
    native_preflight,
)
from astra_reversal.osmo.ood_distributed import WorkerArchive
from astra_reversal.records import digest, file_sha256
from astra_reversal.representation_search import RepresentationSearch, load_protocol


def main():
    import torch

    from astra_reversal.intervention_rollout import capture_reset_manifest
    from astra_reversal.libero_runner import configure_libero
    from astra_reversal.representation_agent import (
        PROMPT_TEMPLATE_VERSION,
        SYSTEM_PROMPT,
    )

    phase = os.environ["ASTRA_REPRESENTATION_PHASE"]
    worker = int(os.environ["ASTRA_WORKER_INDEX"])
    target = assignment(phase, worker)
    RESULTS.mkdir(parents=True, exist_ok=False)
    archive = WorkerArchive(worker)
    if torch.cuda.device_count() != 1 or "L40S" not in torch.cuda.get_device_name(0):
        raise RuntimeError("One allocated OSMO L40S is required")
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
                "pillow",
            )
        },
    }
    write_json(RESULTS / "runtime.json", runtime)
    write_json(RESULTS / "progress.json", {"status": "preflight", **target})
    archive.sync()
    try:
        native_preflight(archive)
        protocol = load_protocol(os.environ.get("ASTRA_PROTOCOL_PATH"))
        if phase == "development":
            protocol["seed"] = protocol["development_seed"]
        write_json(RESULTS / "protocol.json", protocol)
        write_json(
            RESULTS / "prompts.json",
            {
                "template_version": PROMPT_TEMPLATE_VERSION,
                "system_prompt": SYSTEM_PROMPT,
                "sampling": protocol["astra"],
            },
        )
        library = load_library(ROOT / "astra_reversal/.deps/image-perturbations/donors")
        write_json(RESULTS / "image_library.json", library.metadata())
        banks = load_banks(
            archive,
            ROOT / "astra_reversal/.deps/interpolation-inputs/bank_inventory.json",
        )
        libero_root = (
            ROOT / "astra_reversal/.deps/libero-ood/third_party/modified_libero"
        )
        benchmark = BenchmarkConfig.preset(target["suite"])
        cases = (
            protocol["development_cases"][target["suite"]]
            if phase == "development"
            else protocol["evaluation_cases"]
        )
        manifest = capture_reset_manifest(
            libero_root,
            benchmark,
            seed=protocol["seed"],
            cases=cases,
            output=RESULTS / "reset_manifest.json",
            split="development" if phase == "development" else "followup_adaptation",
        )
        entries = (
            [row for row in manifest["episodes"] if row["task_id"] == target["task_id"]]
            if phase == "development"
            else manifest["episodes"][target["case_shard"] :: target["case_shards"]]
        )
        write_json(
            RESULTS / "frozen_plan.json",
            {
                "protocol_sha256": digest(protocol),
                "reset_manifest_sha256": manifest["sha256"],
                "assigned_episodes": [e["episode_id"] for e in entries],
                "image_library_id": library.library_id,
                "bank_inventory_sha256": file_sha256(RESULTS / "bank_inventory.json"),
            },
        )
        archive.sync()
        _, create = configure_libero(libero_root, benchmark, RESULTS / "runtime_libero")
        policy = load_frozen_policy()
        before = frozen_parameter_receipt(policy)
        write_json(RESULTS / "frozen_weights_before.json", before)
        for index, entry in enumerate(entries):
            task_id = entry["task_id"]
            write_json(
                RESULTS / "progress.json",
                {
                    "status": "running",
                    "task_id": task_id,
                    "complete_tasks": index,
                    "assigned_tasks": len(entries),
                    **target,
                },
            )
            search = RepresentationSearch(
                policy,
                create,
                benchmark,
                entry,
                protocol,
                RESULTS / f"task_{task_id}",
                library=library,
                banks=banks,
                development=phase == "development",
                progress=archive.sync,
            )
            search.run()
            after = frozen_parameter_receipt(policy)
            if before != after:
                raise RuntimeError("Frozen native policy tensor bytes changed")
            directory = RESULTS / f"task_{task_id}"
            write_json(directory / "frozen_weights_after.json", after)
            write_json(
                directory / "completion_receipt.json",
                {
                    "schema_version": "representation-complete-case-1.0",
                    "episode_id": entry["episode_id"],
                    "workflow": runtime["workflow"],
                    "worker": worker,
                    "native_tensor_sha256": after["sha256"],
                    "case_files": {
                        str(p.relative_to(directory)): {
                            "sha256": file_sha256(p),
                            "bytes": p.stat().st_size,
                        }
                        for p in sorted(directory.rglob("*"))
                        if p.is_file()
                    },
                    "worker_files": {
                        name: file_sha256(RESULTS / name)
                        for name in (
                            "runtime.json",
                            "protocol.json",
                            "prompts.json",
                            "checkpoint.json",
                            "frozen_plan.json",
                            "reset_manifest.json",
                            "frozen_weights_before.json",
                            "bank_inventory.json",
                        )
                    },
                },
            )
            publish_task(archive, task_id)
        write_json(
            RESULTS / "progress.json",
            {"status": "complete", "complete_tasks": len(entries), **target},
        )
    except Exception as exc:
        write_json(
            RESULTS / "failure.json", {"type": type(exc).__name__, "message": str(exc)}
        )
        raise
    finally:
        archive.sync(include_arrays=True)


if __name__ == "__main__":
    main()
