"""Evaluate a fixed saved student; no training, teacher or collection retries."""

import json
import os
import time
from pathlib import Path

import torch

from astra_reversal.config import BenchmarkConfig
from astra_reversal.intervention_rollout import capture_reset_manifest
from astra_reversal.intervention_search import write_json
from astra_reversal.libero_runner import configure_libero
from astra_reversal.osmo.experiment import RESULTS, ROOT
from astra_reversal.osmo.interpolation import load_frozen_policy
from astra_reversal.osmo.ood_distributed import WorkerArchive
from astra_reversal.osmo.reasoning_policy_learning import load_protocol
from astra_reversal.reasoning_learning.evaluation_checkpoint import load
from astra_reversal.reasoning_learning.learning import load_adapters
from astra_reversal.reasoning_learning.native_reference import native_signature
from astra_reversal.reasoning_learning.rollout import collect
from astra_reversal.records import digest, file_sha256, to_numpy


def main():
    protocol = load_protocol()
    recipe = json.loads(
        (
            Path(__file__).parents[1]
            / "configs/reasoning_checkpoint_evaluation_v8.json"
        ).read_text()
    )
    if os.environ["ASTRA_LEARNING_PHASE"] != "checkpoint-evaluation":
        raise ValueError("Wrong frozen evaluation phase")
    if torch.cuda.device_count() != 1 or "L40S" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Allocate exactly one OSMO L40S")
    allocation = float(os.environ["ASTRA_WORKER_GPU_HOURS"])
    if not 0 < allocation <= protocol["compute"]["authorized_gpu_hours"]:
        raise ValueError("Frozen evaluation allocation exceeds study budget")
    directory = ROOT / "astra_reversal/.deps/reasoning-evaluation-inputs"
    source = load(directory, recipe["manifest_sha256"])
    for relative, expected in source["evaluation_driver_source_sha256"].items():
        if (
            Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or file_sha256(ROOT / relative) != expected
        ):
            raise ValueError("The original evaluation driver source differs")
    task = protocol["pilot"]["development_tasks"][
        int(os.environ.get("ASTRA_LEARNING_TASK_INDEX", "0"))
    ]
    seed = protocol["pilot"]["seeds"][
        int(os.environ.get("ASTRA_LEARNING_SEED_INDEX", "0"))
    ]
    if (
        task != source["source_task"]
        or seed != source["source_seed"]
        or protocol != source["source_protocol"]
        or source["source_workflow"] != recipe["source_workflow"]
        or source["policy_version"] != recipe["policy_version"]
        or recipe["evaluate_all_fixed_resets"] is not True
        or any(
            recipe[k] != 0
            for k in ("optimizer_updates", "new_teacher_calls", "new_collection_steps")
        )
    ):
        raise ValueError("Frozen evaluation recipe differs from its saved source")
    RESULTS.mkdir(parents=True, exist_ok=False)
    archive = WorkerArchive(0)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.monotonic()
    for name, value in (
        ("protocol", protocol),
        ("replay_recipe", recipe),
        ("replayed_training_data", source),
        ("collection", []),
        ("updates", []),
    ):
        write_json(RESULTS / (name + ".json"), value)
    write_json(
        RESULTS / "runtime.json",
        {
            "phase": "checkpoint-evaluation",
            "task": task,
            "seed": seed,
            "source_revision": os.environ["ASTRA_SOURCE_REVISION"],
            "payload_sha256": os.environ["PAYLOAD_SHA256"],
            "workflow": os.environ["ASTRA_RUN_ID"],
            "restored_from_workflow": source["source_workflow"],
            "gpu": torch.cuda.get_device_name(0),
            "new_collection_steps": 0,
            "new_teacher_calls": 0,
            "optimizer_updates": 0,
            "evaluation_restart": "All ten fixed reset states; prior partial evaluation remains charged, not independent samples.",
        },
    )
    try:
        policy = load_frozen_policy()
        if native_signature(policy.metadata) != source["source_policy_signature"]:
            raise ValueError("Saved student base deployment differs")
        version = load_adapters(
            policy, directory / "adapters.pt", source["files"]["adapters.pt"]
        )
        parameter_hash = digest(
            {
                n: to_numpy(p)
                for n, p in policy.model.named_parameters()
                if n.endswith(("lora_a", "lora_b"))
            }
        )
        if (
            version != source["policy_version"]
            or parameter_hash != source["adapter_parameter_sha256"]
        ):
            raise ValueError("Restored student tensors or version differ")
        write_json(
            RESULTS / "restored_checkpoint.json",
            {
                "policy_version": version,
                "checkpoint_sha256": source["files"]["adapters.pt"],
                "parameter_sha256": parameter_hash,
                "optimizer_updates": 0,
            },
        )
        write_json(RESULTS / "checkpoint.json", policy.metadata)
        benchmark = BenchmarkConfig.preset(task["suite"])
        env_root = ROOT / "astra_reversal/.deps/libero-ood/third_party/modified_libero"
        resets = source["evaluation_reset_indices"]
        manifest = capture_reset_manifest(
            env_root,
            benchmark,
            seed=seed,
            cases=[(task["task_id"], i) for i in resets],
            output=RESULTS / "resets.json",
            split="development",
        )
        _, create = configure_libero(env_root, benchmark, RESULTS / "libero_config")
        by_reset = {row["initial_state_id"]: row for row in manifest["episodes"]}
        original_resets = json.loads((directory / "resets.json").read_text())
        original = {row["initial_state_id"]: row for row in original_resets["episodes"]}
        for reset in resets:
            if any(
                by_reset[reset][k] != original[reset][k]
                for k in (
                    "episode_id",
                    "reset_state_sha256",
                    "reset_model_sha256",
                    "bddl_sha256",
                )
            ):
                raise ValueError("Evaluation restart reset identity differs")
        rows = []
        for reset in resets:
            if (time.monotonic() - started) / 3600 >= allocation:
                raise TimeoutError("Frozen evaluation allocation exhausted")
            env, _, _ = create(task["task_id"], seed)
            result, _, _ = collect(
                policy,
                None,
                env,
                by_reset[reset],
                benchmark,
                RESULTS / "evaluation_4" / f"reset_{reset}",
                version=version,
                seed=seed * 1000 + reset,
                retain_step_observations=False,
                progress=archive.sync,
            )
            rows.append(result)
            archive.sync()
        controls = sum(r["total_control_steps"] for r in rows)
        write_json(
            RESULTS / "learning_curve.json",
            [
                {
                    "collection_rollouts": len(source["episodes"]),
                    "collection_steps": source["source_collection_control_steps"],
                    "teacher_tokens_from_reused_data": source[
                        "source_teacher_total_tokens"
                    ],
                    "new_collection_steps": 0,
                    "evaluation_steps_cumulative": controls,
                    "source_evaluation_steps_retained": source[
                        "source_evaluation_control_steps_retained"
                    ],
                    "policy_version": version,
                    "successes": sum(r["success"] for r in rows),
                    "rollouts": len(rows),
                    "episodes": rows,
                }
            ],
        )
        write_json(
            RESULTS / "completion.json",
            {
                "status": "frozen_checkpoint_evaluation_complete",
                "new_collection_steps": 0,
                "evaluation_steps": controls,
                "optimizer_updates": 0,
                "new_teacher_calls": 0,
                "research_objective_met": False,
            },
        )
    finally:
        write_json(
            RESULTS / "cost.json",
            {
                "worker_gpu_hours": (time.monotonic() - started) / 3600,
                "bootstrap_gpu_hours_excluded": True,
            },
        )
        archive.sync()


if __name__ == "__main__":
    main()
