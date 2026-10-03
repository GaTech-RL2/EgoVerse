"""Bounded DSRL baseline with the common OOD reset and evaluation schedule."""

import os
import time

import torch

from astra_reversal.config import BenchmarkConfig
from astra_reversal.intervention_rollout import capture_reset_manifest
from astra_reversal.intervention_search import write_json
from astra_reversal.libero_runner import configure_libero
from astra_reversal.osmo.experiment import RESULTS, ROOT
from astra_reversal.osmo.interpolation import load_frozen_policy
from astra_reversal.osmo.ood_distributed import WorkerArchive
from astra_reversal.osmo.reasoning_policy_learning import load_protocol
from astra_reversal.reasoning_learning.rl_rollout import collect_rl
from astra_reversal.reasoning_learning.rlinf_bridge import REVISION, import_core
from astra_reversal.reasoning_learning.rlinf_sac import DSRLLearner


def main():
    protocol = load_protocol()
    allocation = float(os.environ["ASTRA_WORKER_GPU_HOURS"])
    if not 0 < allocation <= protocol["compute"]["authorized_gpu_hours"]:
        raise ValueError("Worker allocation exceeds study budget")
    if torch.cuda.device_count() != 1 or "L40S" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Allocate one OSMO L40S")
    task = protocol["pilot"]["development_tasks"][
        int(os.environ.get("ASTRA_LEARNING_TASK_INDEX", "0"))
    ]
    seed = protocol["pilot"]["seeds"][
        int(os.environ.get("ASTRA_LEARNING_SEED_INDEX", "0"))
    ]
    RESULTS.mkdir(parents=True, exist_ok=False)
    archive = WorkerArchive(0)
    start = time.monotonic()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(2)
    write_json(RESULTS / "protocol.json", protocol)
    write_json(
        RESULTS / "runtime.json",
        {
            "method": "DSRL",
            "workflow": os.environ["ASTRA_RUN_ID"],
            "source_revision": os.environ["ASTRA_SOURCE_REVISION"],
            "payload_sha256": os.environ["PAYLOAD_SHA256"],
            "rlinf_revision": REVISION,
            "task": task,
            "seed": seed,
            "harness": "RLinf DSRL networks and SAC objectives, serial common OOD driver",
            "decoder": "strict native LeRobot checkpoint, no conversion, frozen",
            "initial_noise_policy": "released tanh Gaussian repeated across the horizon; eval uses its mean",
            "comparison_note": "The decoder weights are identical; DSRL's initial noise distribution differs from the native Gaussian sampler.",
            "overrides": {
                "generation_steps": 10,
                "action_horizon": 10,
                "executed_prefix": 5,
                "max_actions": 300,
                "train_envs": 1,
                "batch_size": 64,
                "updates_per_rollout": 200,
                "fp32_master_parameters": True,
                "reward": "duration-discounted -1+success over actual prefix",
                "finite_horizon_terminal": True,
            },
        },
    )
    try:
        import_core(ROOT / "astra_reversal/.deps/RLinf")
        policy = load_frozen_policy()
        learner = DSRLLearner(policy, seed=seed)
        benchmark = BenchmarkConfig.preset(task["suite"])
        env_root = ROOT / "astra_reversal/.deps/libero-ood/third_party/modified_libero"
        resets = protocol["pilot"]["collection_reset_indices"]
        eval_resets = protocol["pilot"]["autonomous_evaluation_reset_indices"]
        manifest = capture_reset_manifest(
            env_root,
            benchmark,
            seed=seed,
            cases=[(task["task_id"], i) for i in resets + eval_resets],
            output=RESULTS / "resets.json",
            split="development",
        )
        _, create = configure_libero(env_root, benchmark, RESULTS / "libero_config")
        by_reset = {row["initial_state_id"]: row for row in manifest["episodes"]}
        collection, evaluations, updates = [], [], []
        collection_steps, evaluation_steps = 0, 0
        for iteration in range(len(resets) + 1):
            if iteration in protocol["pilot"]["evaluation_after_collection_rollouts"]:
                rows = []
                for reset_id in eval_resets:
                    env, _, _ = create(task["task_id"], seed)
                    result, _ = collect_rl(
                        policy,
                        learner,
                        env,
                        by_reset[reset_id],
                        benchmark,
                        RESULTS / f"evaluation_{iteration}" / f"reset_{reset_id}",
                        evaluation=True,
                        seed=seed * 1000 + reset_id,
                        progress=archive.sync,
                    )
                    evaluation_steps += result["total_control_steps"]
                    rows.append(result)
                    archive.sync()
                evaluations.append(
                    {
                        "collection_rollouts": iteration,
                        "collection_steps": collection_steps,
                        "evaluation_steps_cumulative": evaluation_steps,
                        "policy_version": learner.version,
                        "successes": sum(r["success"] for r in rows),
                        "rollouts": len(rows),
                        "episodes": rows,
                    }
                )
                write_json(RESULTS / "learning_curve.json", evaluations)
            if iteration == len(resets):
                break
            if (time.monotonic() - start) / 3600 >= allocation:
                raise TimeoutError("Worker GPU-hour allocation exhausted")
            env, _, _ = create(task["task_id"], seed)
            result, transitions = collect_rl(
                policy,
                learner,
                env,
                by_reset[resets[iteration]],
                benchmark,
                RESULTS / "collection" / f"rollout_{iteration}",
                evaluation=False,
                seed=seed * 1000 + resets[iteration],
                progress=archive.sync,
            )
            collection.append(result)
            collection_steps += result["total_control_steps"]
            update = learner.update(transitions)
            updates.append(update)
            checkpoint = RESULTS / "checkpoints" / f"policy_{iteration + 1}.pt"
            checkpoint.parent.mkdir(exist_ok=True)
            learner.save(checkpoint)
            write_json(RESULTS / "collection.json", collection)
            write_json(RESULTS / "updates.json", updates)
            archive.sync()
        write_json(
            RESULTS / "completion.json",
            {
                "status": "baseline_complete",
                "collection_steps": collection_steps,
                "evaluation_steps": evaluation_steps,
                "policy_updates": learner.version,
                "research_objective_met": False,
            },
        )
    finally:
        write_json(
            RESULTS / "cost.json",
            {
                "worker_gpu_hours": (time.monotonic() - start) / 3600,
                "bootstrap_gpu_hours_excluded": True,
            },
        )
        archive.sync()


if __name__ == "__main__":
    main()
