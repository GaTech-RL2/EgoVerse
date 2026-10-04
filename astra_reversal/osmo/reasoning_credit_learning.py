"""Compare three data selectors with fresh students and equal optimizer budgets."""

import copy
import gc
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
from astra_reversal.reasoning_learning.credit_data import load, variant_names
from astra_reversal.reasoning_learning.native_reference import (
    native_signature,
    reuse_native_evaluation,
)
from astra_reversal.reasoning_learning.rollout import collect
from astra_reversal.records import digest, to_numpy


def run_variant(
    variant,
    windows,
    observations,
    source,
    protocol,
    recipe,
    task,
    seed,
    archive,
    started,
    allocation,
):
    """A fresh model, adapter initialization, optimizer and RNG for every branch."""
    output = RESULTS / variant
    output.mkdir(exist_ok=False)
    attributed = copy.deepcopy(source)
    attributed.update(variant=variant, admitted_windows=len(windows))
    attributed["source_teacher_total_tokens"] += (
        source["hindsight_teacher_total_tokens"] if variant == "hindsight_gate" else 0
    )
    for filename, value in (
        ("protocol", protocol),
        ("replay_recipe", recipe),
        ("replayed_training_data", attributed),
        ("collection", []),
    ):
        write_json(output / (filename + ".json"), value)
    write_json(
        output / "runtime.json",
        {
            "phase": os.environ["ASTRA_LEARNING_PHASE"],
            "variant": variant,
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
    policy = load_frozen_policy()
    if native_signature(policy.metadata) != source["source_policy_signature"]:
        raise ValueError("Credit ablation student deployment differs")
    write_json(output / "checkpoint.json", policy.metadata)
    # Use exactly the same preflight observation across branches as well.
    first = observations[sorted(observations)[0]]
    learner, receipt = preflight(
        policy,
        first,
        first["prompt"],
        seed=seed,
        rank=protocol["learner"]["rank"],
        learning_rate=protocol["learner"]["learning_rate"],
        loss_action_dimensions=7,
    )
    receipt["initial_adapter_sha256"] = digest(
        {name: to_numpy(p) for name, p in learner.parameters.items()}
    )
    write_json(output / "preflight.json", receipt)
    benchmark = BenchmarkConfig.preset(task["suite"])
    env_root = ROOT / "astra_reversal/.deps/libero-ood/third_party/modified_libero"
    resets = protocol["pilot"]["autonomous_evaluation_reset_indices"]
    manifest = capture_reset_manifest(
        env_root,
        benchmark,
        seed=seed,
        cases=[(task["task_id"], i) for i in resets],
        output=output / "resets.json",
        split="development",
    )
    reference = (
        recipe.get("native_reference")
        or protocol["teacher_native_evaluation_references"][
            f"{task['suite']}:{task['task_id']}:{seed}"
        ]
    )
    if Path(reference["file"]).name != reference["file"]:
        raise ValueError("Shared native reference must be a bundled basename")
    initial = reuse_native_evaluation(
        Path(__file__).parents[1] / "reasoning_learning" / reference["file"],
        reference["sha256"],
        protocol=protocol,
        metadata=policy.metadata,
        preflight=receipt,
        task=task,
        seed=seed,
        manifest=manifest,
    )
    write_json(output / "learning_curve.json", [initial])
    count = recipe["optimizer_steps_per_variant"]
    update = learner.update(
        windows,
        observations,
        updates=count,
        sampling="uniform",
        allow_successful_episodes=variant == "successful_episode",
    )
    update["optimizer_steps_cumulative"] = count
    update["checkpoint"] = learner.save(
        output / "checkpoints" / f"optimizer_step_{count}"
    )
    write_json(output / "updates.json", [update])
    archive.sync()
    _, create = configure_libero(env_root, benchmark, output / "libero_config")
    by_reset = {r["initial_state_id"]: r for r in manifest["episodes"]}
    rows = []
    for reset in resets:
        if (time.monotonic() - started) / 3600 >= allocation:
            raise TimeoutError("Credit ablation worker allocation exhausted")
        env, _, _ = create(task["task_id"], seed)
        result, _, _ = collect(
            policy,
            None,
            env,
            by_reset[reset],
            benchmark,
            output / f"evaluation_{count}" / f"reset_{reset}",
            version=learner.version,
            seed=seed * 1000 + reset,
            retain_step_observations=False,
            progress=archive.sync,
        )
        rows.append(result)
        archive.sync()
    evaluation_steps = sum(r["total_control_steps"] for r in rows)
    final = {
        "collection_rollouts": len(source["episodes"]),
        "collection_steps": source["source_collection_control_steps"],
        "new_collection_steps": 0,
        "teacher_tokens_from_reused_data": attributed["source_teacher_total_tokens"],
        "optimizer_steps_cumulative": count,
        "evaluation_steps_cumulative": evaluation_steps,
        "policy_version": learner.version,
        "successes": sum(r["success"] for r in rows),
        "rollouts": len(rows),
        "episodes": rows,
    }
    write_json(output / "learning_curve.json", [initial, final])
    write_json(
        output / "completion.json",
        {
            "status": "fixed_data_credit_ablation_complete",
            "variant": variant,
            "new_collection_steps": 0,
            "evaluation_steps": evaluation_steps,
            "policy_updates": learner.version,
            "research_objective_met": False,
            "scope": recipe.get(
                "scope",
                "Development selection ablation on the same two V5 collections; no fresh collection or confirmation. The binary-success control is not evidence that every copied action was useful.",
            ),
        },
    )
    archive.sync()
    return {
        "variant": variant,
        "successes": final["successes"],
        "rollouts": len(rows),
        "initial_adapter_sha256": receipt["initial_adapter_sha256"],
        "evaluation_steps": evaluation_steps,
    }


def main():
    protocol = load_protocol()
    recipes = {
        "replay-credit": "reasoning_credit_v5.json",
        "replay-success-credit": "reasoning_credit_spatial179_native.json",
    }
    phase = os.environ["ASTRA_LEARNING_PHASE"]
    if phase not in recipes:
        raise ValueError("Wrong credit ablation phase")
    recipe = json.loads(
        (Path(__file__).parents[1] / "configs" / recipes[phase]).read_text()
    )
    if torch.cuda.device_count() != 1 or "L40S" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Allocate exactly one OSMO L40S")
    allocation = float(os.environ["ASTRA_WORKER_GPU_HOURS"])
    if not 0 < allocation <= protocol["compute"]["authorized_gpu_hours"]:
        raise ValueError("Credit ablation allocation exceeds study budget")
    task = protocol["pilot"]["development_tasks"][
        int(os.environ.get("ASTRA_LEARNING_TASK_INDEX", "0"))
    ]
    seed = protocol["pilot"]["seeds"][
        int(os.environ.get("ASTRA_LEARNING_SEED_INDEX", "0"))
    ]
    variants, observations, source = load(
        ROOT / "astra_reversal/.deps/reasoning-credit-inputs", recipe["manifest_sha256"]
    )
    if source["source_task"] != task or source["source_seed"] != seed:
        raise ValueError("This screen requires the recorded source task/seed")
    if any(
        source["source_protocol"][k] != protocol[k]
        for k in ("checkpoint", "environment")
    ):
        raise ValueError("Source and learner environments differ")
    if (
        tuple(recipe["variant_order"]) != variant_names(source)
        or recipe["window_sampling"] != "uniform"
    ):
        raise ValueError("Preregistered selectors require uniform sampling")
    RESULTS.mkdir(parents=True, exist_ok=False)
    archive = WorkerArchive(0)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.monotonic()
    summaries = []
    write_json(RESULTS / "replay_recipe.json", recipe)
    try:
        for variant in recipe["variant_order"]:
            row = run_variant(
                variant,
                variants[variant],
                observations,
                source,
                protocol,
                recipe,
                task,
                seed,
                archive,
                started,
                allocation,
            )
            if (
                summaries
                and row["initial_adapter_sha256"]
                != summaries[0]["initial_adapter_sha256"]
            ):
                raise ValueError(
                    "Branches did not start with identical adapter weights"
                )
            summaries.append(row)
            write_json(RESULTS / "variant_results.json", summaries)
            # The preceding function releases the student and optimizer together.
            gc.collect()
            torch.cuda.empty_cache()
        write_json(
            RESULTS / "completion.json",
            {
                "status": "three_way_credit_ablation_complete"
                if phase == "replay-credit"
                else "paired_credit_ablation_complete",
                "research_objective_met": False,
                "new_collection_steps": 0,
                "new_teacher_calls": 0,
                "evaluation_steps": sum(r["evaluation_steps"] for r in summaries),
                "source_collection_control_steps": source[
                    "source_collection_control_steps"
                ],
            },
        )
    finally:
        write_json(
            RESULTS / "cost.json",
            {
                "worker_gpu_hours": (time.monotonic() - started) / 3600,
                "bootstrap_gpu_hours_excluded": True,
                "note": "One allocation shared across branches; external ledger charges initialization once.",
            },
        )
        archive.sync()


if __name__ == "__main__":
    main()
