"""Bounded DSRL baseline with the common OOD reset and evaluation schedule."""

import contextlib
import json
import os
import time

import numpy as np
import torch

from astra_reversal.config import BenchmarkConfig
from astra_reversal.intervention_rollout import capture_reset_manifest
from astra_reversal.intervention_search import write_json
from astra_reversal.libero_runner import configure_libero
from astra_reversal.osmo.experiment import RESULTS, ROOT
from astra_reversal.osmo.interpolation import load_frozen_policy
from astra_reversal.osmo.ood_distributed import WorkerArchive
from astra_reversal.osmo.reasoning_policy_learning import load_protocol
from astra_reversal.reasoning_learning.rl_recipes import collect_on_policy, settings
from astra_reversal.reasoning_learning.rl_rollout import collect_rl
from astra_reversal.reasoning_learning.rlinf_bridge import (
    REVISION,
    import_core,
    load_converted,
    match_native_gelu,
    measure,
)
from astra_reversal.reasoning_learning.rlinf_ppo import PPOLearner
from astra_reversal.reasoning_learning.rlinf_sac import DSRLLearner


def main():
    protocol = load_protocol()
    method = os.environ.get("ASTRA_LEARNING_METHOD", "dsrl")
    if method not in ("dsrl", "ppo"):
        raise ValueError("Unknown RL baseline")
    recipe = os.environ.get("ASTRA_RL_RECIPE", "standard")
    options = settings(
        recipe,
        method,
        collection_rollouts=len(protocol["pilot"]["collection_reset_indices"]),
        evaluation_schedule=protocol["pilot"]["evaluation_after_collection_rollouts"],
    )
    update_period = options.get("collection_rollouts_per_update", 1)
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
            "method": method.upper(),
            "tuning_recipe": recipe,
            "tuning_settings": options,
            "status": "initializing_before_preflight",
            "workflow": os.environ["ASTRA_RUN_ID"],
            "source_revision": os.environ["ASTRA_SOURCE_REVISION"],
            "payload_sha256": os.environ["PAYLOAD_SHA256"],
            "rlinf_revision": REVISION,
            "task": task,
            "seed": seed,
            "harness": "RLinf networks with serial common OOD driver",
            "decoder": (
                "strict native LeRobot checkpoint, no conversion, frozen"
                if method == "dsrl"
                else "strict conversion with explicit tanh GELU compatibility; parity and backward pending"
            ),
            "initial_noise_policy": (
                "released tanh Gaussian repeated across the horizon; eval uses its mean"
                if method == "dsrl"
                else "native Gaussian for evaluation; released flow-SDE exploration for training"
            ),
            "overrides": {
                "generation_steps": 10,
                "action_horizon": 10,
                "executed_prefix": 5,
                "max_actions": 300,
                "train_envs": 1,
                "batch_size": 64,
                "updates_per_rollout": options["updates"] if method == "dsrl" else None,
                "fp32_master_parameters": True,
                "reward": (
                    "duration-discounted -1+success over actual prefix"
                    if method == "dsrl"
                    else "binary success over actual prefix"
                ),
                "finite_horizon_terminal": True,
            },
        },
    )
    compatibility = contextlib.ExitStack()
    try:
        core = import_core(ROOT / "astra_reversal/.deps/RLinf")
        policy = load_frozen_policy()
        if method == "dsrl":
            learner = DSRLLearner(policy, seed=seed, **options)
        else:
            inventory = json.loads(
                (
                    ROOT / "astra_reversal/checkpoints/lerobot_pi05_libero_base.json"
                ).read_text()
            )
            converted, receipt = load_converted(
                core,
                ROOT / inventory["local_directory"],
                horizon=policy.horizon,
                device=policy.device,
            )
            compatibility.enter_context(
                match_native_gelu(core, converted, policy.policy)
            )
            with np.load(
                ROOT / "astra_reversal/.deps/reference/cpu_libero_probe.npz",
                allow_pickle=False,
            ) as data:
                observation = {
                    key: data[key]
                    for key in (
                        "observation/image",
                        "observation/wrist_image",
                        "observation/state",
                    )
                }
                prompt = str(data["prompt"].item())
            receipt["parity"] = measure(core, converted, policy, observation, prompt)
            if not receipt["parity"]["passed"]:
                raise RuntimeError("Converted PPO initial policy failed native parity")
            learner = PPOLearner(
                core,
                converted,
                policy,
                seed=seed,
                **{
                    k: v
                    for k, v in options.items()
                    if k != "collection_rollouts_per_update"
                },
            )
            receipt["ppo"] = learner.preflight(observation, prompt)
            write_json(RESULTS / "ppo_preflight.json", receipt)
            write_json(
                RESULTS / "runtime.json",
                {
                    "method": "PPO",
                    "tuning_recipe": recipe,
                    "tuning_settings": options,
                    "workflow": os.environ["ASTRA_RUN_ID"],
                    "source_revision": os.environ["ASTRA_SOURCE_REVISION"],
                    "payload_sha256": os.environ["PAYLOAD_SHA256"],
                    "rlinf_revision": REVISION,
                    "task": task,
                    "seed": seed,
                    "harness": "released RLinf Pi0RL sampler/recompute with serial clipped-PPO/GAE driver",
                    "decoder": "same checkpoint, strict conversion with explicit tanh GELU compatibility",
                    "overrides": {
                        "generation_steps": 10,
                        "action_horizon": 10,
                        "executed_prefix": 5,
                        "max_actions": 300,
                        "train_envs": 1,
                        "micro_batch": 1,
                        "optimizer_batch": options["optimizer_batch"],
                        "epochs_per_update": options["epochs"],
                        "collection_rollouts_per_update": update_period,
                        "parameters": "full action expert and value head; frozen VLM",
                        "precision": "float32",
                        "reward": "binary success summed over actual prefix",
                        "finite_horizon_terminal": True,
                    },
                },
            )
            archive.sync()
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
        pending_transitions = []
        for iteration in range(len(resets) + 1):
            if iteration in protocol["pilot"]["evaluation_after_collection_rollouts"]:
                if pending_transitions:
                    raise RuntimeError(
                        "Autonomous evaluation encountered an unfinished PPO batch"
                    )
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
                        retain_evaluation_observations=protocol["pilot"].get(
                            "record_all_evaluation_observations", True
                        ),
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
            if method == "ppo":
                pending_transitions = collect_on_policy(
                    pending_transitions, transitions, learner.version
                )
            due = (iteration + 1) % update_period == 0
            if due:
                update = learner.update(
                    pending_transitions if method == "ppo" else transitions
                )
                updates.append(update)
                pending_transitions = []
            if due and (
                method == "dsrl"
                or iteration + 1
                in protocol["pilot"]["evaluation_after_collection_rollouts"]
            ):
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
        compatibility.close()
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
