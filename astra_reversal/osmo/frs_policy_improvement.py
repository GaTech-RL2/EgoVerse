"""OSMO L40S workers for paper-like FRS and judgment-gated auxiliary learning."""

import hashlib
import importlib.metadata
import os
import platform
import tarfile

from astra_reversal.config import BenchmarkConfig
from astra_reversal.frs_experiment import FRSTaskExperiment, load_protocol
from astra_reversal.intervention_search import write_json
from astra_reversal.osmo.experiment import RESULTS, ROOT
from astra_reversal.osmo.interpolation import load_frozen_policy, native_preflight
from astra_reversal.osmo.ood_distributed import WorkerArchive
from astra_reversal.records import digest, file_sha256


def assignment(phase, worker):
    if type(worker) is not int:
        raise ValueError("Worker index must be an integer")
    if phase == "development" and 0 <= worker < 3:
        return {
            "suite": "libero_goal_ood" if worker == 0 else "libero_spatial_ood",
            "task_ids": [[6], [2], [8]][worker],
        }
    if phase == "evaluation" and 0 <= worker < 8:
        return {
            "suite": ("libero_goal_ood", "libero_spatial_ood")[worker // 4],
            "task_ids": list(range(worker % 4, 10, 4)),
        }
    raise ValueError("Invalid FRS phase or worker")


def publish_task(archive, task_id):
    """Preserve each complete task before starting another on a borrowed GPU."""
    bundle = RESULTS.parent / f"task_{task_id}.tar.gz"
    with tarfile.open(bundle, "w:gz", compresslevel=1) as stream:
        stream.add(RESULTS / f"task_{task_id}", arcname=f"task_{task_id}")
    key = f"{archive.prefix}/task_{task_id}.tar.gz"
    receipt = {
        "task_id": task_id,
        "key": key,
        "sha256": file_sha256(bundle),
        "bytes": bundle.stat().st_size,
    }
    archive.client.upload_file(str(bundle), "rldb", key)
    write_json(RESULTS / f"task_{task_id}_archive.json", receipt)
    archive.sync()
    bundle.unlink()  # The original files and verified remote archive remain.


def frozen_parameter_receipt(policy):
    """Hash native tensor bytes before/after the full run on the allocated GPU."""
    import torch

    tensors = {}
    for name, value in policy.policy.state_dict().items():
        raw = value.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy()
        tensors[name] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": hashlib.sha256(memoryview(raw)).hexdigest(),
        }
    return {"tensors": tensors, "sha256": digest(tensors)}


def main():
    import torch

    from astra_reversal.frs_agent import prompt_manifest
    from astra_reversal.intervention_rollout import capture_reset_manifest
    from astra_reversal.libero_runner import configure_libero

    phase = os.environ["ASTRA_FRS_PHASE"]
    worker = int(os.environ["ASTRA_WORKER_INDEX"])
    target = assignment(phase, worker)
    RESULTS.mkdir(parents=True, exist_ok=False)
    archive = WorkerArchive(worker)
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("Set deterministic CUBLAS configuration before CUDA starts")
    if torch.cuda.device_count() != 1 or "L40S" not in torch.cuda.get_device_name(0):
        raise RuntimeError("FRS experiments require one allocated OSMO L40S")
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
        "source_revision": os.environ["ASTRA_SOURCE_REVISION"],
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
    write_json(RESULTS / "prompts.json", prompt_manifest())
    archive.sync()
    try:
        native_preflight(archive)
        protocol = load_protocol()
        if phase == "development":
            protocol["seed"] = protocol["development_seed"]
        write_json(RESULTS / "protocol.json", protocol)
        root = ROOT / "astra_reversal/.deps/libero-ood/third_party/modified_libero"
        benchmark = BenchmarkConfig.preset(target["suite"])
        cases = [[task, state] for task in target["task_ids"] for state in range(11)]
        manifest = capture_reset_manifest(
            root,
            benchmark,
            seed=protocol["seed"],
            cases=cases,
            output=RESULTS / "reset_manifest.json",
            split="development" if phase == "development" else "followup_adaptation",
        )
        write_json(
            RESULTS / "frozen_plan.json",
            {
                "runtime": runtime,
                "protocol_sha256": digest(protocol),
                "manifest_sha256": manifest["sha256"],
                "assigned_episodes": [
                    row["episode_id"] for row in manifest["episodes"]
                ],
                "assigned_tasks": target["task_ids"],
            },
        )
        archive.sync()
        _, create = configure_libero(root, benchmark, RESULTS / "runtime_libero")
        policy = load_frozen_policy()
        before = frozen_parameter_receipt(policy)
        write_json(RESULTS / "frozen_weights_before.json", before)
        archive.sync()
        for index, task_id in enumerate(target["task_ids"]):
            write_json(
                RESULTS / "progress.json",
                {
                    "status": "running",
                    "task_id": task_id,
                    "complete_tasks": index,
                    "assigned_tasks": len(target["task_ids"]),
                    **target,
                },
            )
            entries = [row for row in manifest["episodes"] if row["task_id"] == task_id]
            experiment = FRSTaskExperiment(
                policy,
                create,
                benchmark,
                entries,
                protocol,
                RESULTS / f"task_{task_id}",
                development=phase == "development",
                progress=archive.sync,
            )
            experiment.run()
            if any(parameter.requires_grad for parameter in policy.policy.parameters()):
                raise RuntimeError("Frozen pi05 weights became trainable")
            publish_task(archive, task_id)
        after = frozen_parameter_receipt(policy)
        write_json(RESULTS / "frozen_weights_after.json", after)
        if before != after:
            raise RuntimeError("Native pi05 tensor bytes changed during the experiment")
        write_json(
            RESULTS / "progress.json",
            {
                "status": "complete",
                "complete_tasks": len(target["task_ids"]),
                **target,
            },
        )
    except Exception as exc:
        write_json(
            RESULTS / "failure.json",
            {"type": type(exc).__name__, "message": str(exc)[:500]},
        )
        raise
    finally:
        archive.sync(include_arrays=True)


if __name__ == "__main__":
    main()
