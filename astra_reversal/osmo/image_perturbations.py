"""OSMO L40S evaluation of real camera perturbations with raw-image feedback."""

import importlib.metadata
import os
import platform

from astra_reversal.config import BenchmarkConfig
from astra_reversal.image_donor_bank import load_library
from astra_reversal.image_perturbation_search import (
    ImagePerturbationSearch,
    aggregate_reports,
    load_protocol,
)
from astra_reversal.intervention_search import write_json
from astra_reversal.osmo.experiment import RESULTS, ROOT
from astra_reversal.osmo.interpolation import (
    assignment as phase_assignment,
)
from astra_reversal.osmo.interpolation import load_frozen_policy, native_preflight
from astra_reversal.osmo.ood_distributed import WorkerArchive
from astra_reversal.records import digest


def assignment(phase, worker):
    if phase not in ("development", "evaluation"):
        raise ValueError("Image phase must be development or evaluation")
    return phase_assignment(phase, worker)


def main():
    import torch

    from astra_reversal.intervention_rollout import capture_reset_manifest
    from astra_reversal.libero_runner import configure_libero

    phase = os.environ["ASTRA_IMAGE_PHASE"]
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
                "pillow",
            )
        },
    }
    write_json(RESULTS / "runtime.json", runtime)
    write_json(RESULTS / "progress.json", {"status": "preflight", **target})
    archive.sync()
    try:
        native_preflight(archive)
        protocol = load_protocol()
        if phase == "development":
            protocol["seed"] = protocol["development_seed"]
        write_json(RESULTS / "protocol.json", protocol)
        library = load_library(ROOT / "astra_reversal/.deps/image-perturbations/donors")
        if (
            len(library.catalog()) != 45
            or library.library_id != protocol["image_library_id"]
        ):
            raise ValueError("Image library differs from the frozen protocol")
        write_json(RESULTS / "image_library.json", library.metadata())
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
                "runtime": runtime,
                "protocol_sha256": digest(protocol),
                "manifest_sha256": manifest["sha256"],
                "image_library_sha256": digest(library.metadata()),
                "assigned_episodes": [row["episode_id"] for row in entries],
            },
        )
        archive.sync()
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
            search = ImagePerturbationSearch(
                policy,
                create,
                benchmark,
                entry,
                protocol,
                RESULTS / f"case_{entry['task_id']}_{entry['initial_state_id']}",
                library=library,
                development=phase == "development",
                progress=archive.sync,
            )
            reports.append(search.run())
            write_json(RESULTS / "aggregate.json", aggregate_reports(reports, protocol))
            archive.sync()
        write_json(RESULTS / "progress.json", {"status": "complete", **target})
    except Exception as exc:
        write_json(RESULTS / "failure.json", {"type": type(exc).__name__})
        raise
    finally:
        archive.sync(include_arrays=True)


if __name__ == "__main__":
    main()
