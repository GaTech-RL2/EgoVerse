"""Fixed-data native-loss ablation using audited, actually executed corrections."""

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
from astra_reversal.osmo.reasoning_policy_learning import load_protocol, preflight
from astra_reversal.reasoning_learning.native_reference import (
    native_signature,
    reuse_native_evaluation,
)
from astra_reversal.reasoning_learning.replay_data import load
from astra_reversal.reasoning_learning.rollout import collect


def main():
    protocol = load_protocol()
    recipe = json.loads(
        (Path(__file__).parents[1] / "configs/reasoning_replay_v3.json").read_text()
    )
    if torch.cuda.device_count() != 1 or "L40S" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Allocate exactly one OSMO L40S")
    allocation = float(os.environ["ASTRA_WORKER_GPU_HOURS"])
    if not 0 < allocation <= protocol["compute"]["authorized_gpu_hours"]:
        raise ValueError("Replay ablation allocation exceeds the study budget")
    task = protocol["pilot"]["development_tasks"][
        int(os.environ.get("ASTRA_LEARNING_TASK_INDEX", "0"))
    ]
    seed = protocol["pilot"]["seeds"][
        int(os.environ.get("ASTRA_LEARNING_SEED_INDEX", "0"))
    ]
    windows, observations, source = load(
        ROOT / "astra_reversal/.deps/reasoning-replay-inputs",
        recipe["manifest_sha256"],
    )
    if source["source_task"] != task:
        raise ValueError("This fixed-data ablation must use the source task")
    for key in ("checkpoint", "environment"):
        if source["source_protocol"][key] != protocol[key]:
            raise ValueError("Source collection and learner deployment differ")
    RESULTS.mkdir(parents=True, exist_ok=False)
    archive = WorkerArchive(0)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.monotonic()
    write_json(RESULTS / "protocol.json", protocol)
    write_json(RESULTS / "replay_recipe.json", recipe)
    write_json(RESULTS / "replayed_training_data.json", source)
    write_json(RESULTS / "collection.json", [])
    write_json(
        RESULTS / "runtime.json",
        {
            "phase": "replay-learning",
            "task": task,
            "seed": seed,
            "source_revision": os.environ["ASTRA_SOURCE_REVISION"],
            "payload_sha256": os.environ["PAYLOAD_SHA256"],
            "workflow": os.environ["ASTRA_RUN_ID"],
            "gpu": torch.cuda.get_device_name(0),
            "new_collection_steps": 0,
            "new_teacher_calls": 0,
        },
    )
    evaluations, updates, evaluation_steps = [], [], 0
    try:
        policy = load_frozen_policy()
        if native_signature(policy.metadata) != source["source_policy_signature"]:
            raise ValueError("Replay student does not use the same native deployment")
        first = observations[windows[0]["observation_id"]]
        learner, receipt = preflight(
            policy,
            first,
            first["prompt"],
            seed=seed,
            rank=protocol["learner"]["rank"],
            learning_rate=protocol["learner"]["learning_rate"],
            loss_action_dimensions=7,
        )
        write_json(RESULTS / "preflight.json", receipt)
        benchmark = BenchmarkConfig.preset(task["suite"])
        env_root = ROOT / "astra_reversal/.deps/libero-ood/third_party/modified_libero"
        eval_resets = protocol["pilot"]["autonomous_evaluation_reset_indices"]
        manifest = capture_reset_manifest(
            env_root,
            benchmark,
            seed=seed,
            cases=[(task["task_id"], i) for i in eval_resets],
            output=RESULTS / "resets.json",
            split="development",
        )
        _, create = configure_libero(env_root, benchmark, RESULTS / "libero_config")
        by_reset = {row["initial_state_id"]: row for row in manifest["episodes"]}

        def evaluate(optimizer_steps):
            nonlocal evaluation_steps
            rows = []
            for reset in eval_resets:
                if (time.monotonic() - started) / 3600 >= allocation:
                    raise TimeoutError("Replay ablation worker allocation exhausted")
                env, _, _ = create(task["task_id"], seed)
                result, _, _ = collect(
                    policy,
                    None,
                    env,
                    by_reset[reset],
                    benchmark,
                    RESULTS / f"evaluation_{optimizer_steps}" / f"reset_{reset}",
                    version=learner.version,
                    seed=seed * 1000 + reset,
                    retain_step_observations=False,
                    progress=archive.sync,
                )
                evaluation_steps += result["total_control_steps"]
                rows.append(result)
                archive.sync()
            evaluations.append(
                {
                    "collection_rollouts": len(source["episodes"])
                    if optimizer_steps
                    else 0,
                    "collection_steps": source["source_collection_control_steps"]
                    if optimizer_steps
                    else 0,
                    "new_collection_steps": 0,
                    "teacher_tokens_from_reused_data": source[
                        "source_teacher_total_tokens"
                    ]
                    if optimizer_steps
                    else 0,
                    "optimizer_steps_cumulative": optimizer_steps,
                    "evaluation_steps_cumulative": evaluation_steps,
                    "policy_version": learner.version,
                    "successes": sum(r["success"] for r in rows),
                    "rollouts": len(rows),
                    "episodes": rows,
                }
            )
            write_json(RESULTS / "learning_curve.json", evaluations)
            archive.sync()

        reference = protocol.get("teacher_native_evaluation_references", {}).get(
            f"{task['suite']}:{task['task_id']}:{seed}"
        )
        if reference:
            filename = reference["file"]
            if Path(filename).name != filename:
                raise ValueError("Native reference must be a bundled basename")
            point = reuse_native_evaluation(
                Path(__file__).parents[1] / "reasoning_learning" / filename,
                reference["sha256"],
                protocol=protocol,
                metadata=policy.metadata,
                preflight=receipt,
                task=task,
                seed=seed,
                manifest=manifest,
            )
            evaluations.append(point)
            write_json(RESULTS / "learning_curve.json", evaluations)
        else:
            evaluate(0)
        completed_steps = 0
        for milestone in recipe["optimizer_step_checkpoints"]:
            if completed_steps == 0:
                batches = [
                    (
                        [
                            w
                            for w in windows
                            if w["episode_id"] == episode["episode_id"]
                        ],
                        recipe["initial_updates_per_source_episode"],
                    )
                    for episode in source["episodes"]
                ]
                if sum(n for _, n in batches) != milestone:
                    raise ValueError(
                        "First replay checkpoint must match source update count"
                    )
            else:
                batches = [(windows, milestone - completed_steps)]
            for examples, count in batches:
                update = learner.update(examples, observations, updates=count)
                completed_steps += count
                update["optimizer_steps_cumulative"] = completed_steps
                update["checkpoint"] = learner.save(
                    RESULTS / "checkpoints" / f"optimizer_step_{completed_steps}"
                )
                updates.append(update)
            write_json(RESULTS / "updates.json", updates)
            archive.sync()
            evaluate(milestone)
        write_json(
            RESULTS / "completion.json",
            {
                "status": "fixed_data_learner_ablation_complete",
                "new_collection_steps": 0,
                "source_collection_control_steps": source[
                    "source_collection_control_steps"
                ],
                "source_teacher_total_tokens": source["source_teacher_total_tokens"],
                "evaluation_steps": evaluation_steps,
                "policy_updates": learner.version,
                "research_objective_met": False,
                "scope": "Development ablation using reused V3 data; not an independent collection run or confirmation.",
            },
        )
    finally:
        write_json(
            RESULTS / "cost.json",
            {
                "worker_gpu_hours": (time.monotonic() - started) / 3600,
                "bootstrap_gpu_hours_excluded": True,
                "note": "Allocation includes initialization in the external OSMO ledger.",
            },
        )
        archive.sync()


if __name__ == "__main__":
    main()
