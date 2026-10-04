"""OSMO L40S pilot: collect once, update the native policy, evaluate autonomously.

This entrypoint does not claim to run the outstanding RLinf baseline integration.
The committed protocol must contain an explicit compute allocation before launch.
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from astra_reversal.codex_relay import CodexRelayClient
from astra_reversal.config import BenchmarkConfig
from astra_reversal.intervention_rollout import capture_reset_manifest
from astra_reversal.intervention_search import write_json
from astra_reversal.lerobot_policy import prepare_velocity
from astra_reversal.libero_runner import configure_libero
from astra_reversal.osmo.experiment import RESULTS, ROOT
from astra_reversal.osmo.interpolation import load_frozen_policy
from astra_reversal.osmo.ood_distributed import WorkerArchive
from astra_reversal.reasoning_learning.guidance import (
    GuidanceConfig,
    endpoint_gradient,
    generate,
)
from astra_reversal.reasoning_learning.learning import NativeLearner
from astra_reversal.reasoning_learning.native_reference import reuse_native_evaluation
from astra_reversal.reasoning_learning.rollout import collect
from astra_reversal.records import digest


def load_protocol(path=None, *, version=None):
    version = version or os.environ.get("ASTRA_LEARNING_PROTOCOL_VERSION", "v1")
    if version not in ("v1", "v2", "v3", "v4"):
        raise ValueError("Unknown learning protocol version")
    path = path or (
        Path(__file__).parents[1] / f"configs/reasoning_policy_learning_{version}.json"
    )
    protocol = json.loads(Path(path).read_text())
    compute = protocol["compute"]
    if (
        not compute["launch_allowed"]
        or type(compute["authorized_gpu_hours"]) not in (int, float)
        or compute["authorized_gpu_hours"] <= 0
    ):
        raise ValueError(
            "Record the study's authorized GPU-hour budget before launching experiments"
        )
    if (
        protocol["teacher"]["frs_action_steering"]
        or protocol["environment"]["physical_candidate_retries"]
    ):
        raise ValueError(
            "This study prohibits FRS action steering and physical candidate retries"
        )
    return protocol


def preflight(policy, observation, prompt, *, seed=173, rank=8, learning_rate=1e-4):
    """Weighted-model gradient and parity diagnostics; zero environment actions."""
    raw = {**observation, "prompt": prompt}
    condition = policy.prepare(observation, digest(raw), prompt)
    noise = policy.noise(np.random.default_rng(173))
    batch = policy._preprocess(raw)
    velocity = prepare_velocity(policy.policy, batch, differentiable=True)
    native = policy.sample(condition, noise, steps=10).value
    disabled, _ = generate(
        velocity,
        noise,
        torch.zeros_like(noise),
        torch.ones_like(noise),
        GuidanceConfig(strength=0),
    )
    native_error = float((native - disabled).abs().max())
    if native_error > 1e-6:
        raise RuntimeError("Disabled guidance differs from the native sampler")
    mask = torch.zeros_like(noise)
    mask[:, :5, 2] = 1
    target = native.detach().clone()
    target[:, :5, 2] += 0.1
    _, grad, _ = endpoint_gradient(velocity, noise, 0.5, target, mask)
    if not torch.isfinite(grad).all() or not grad.abs().max() > 0:
        raise RuntimeError("Weighted native action derivative is invalid")
    guided, timing = generate(velocity, noise, target, mask)
    learner = NativeLearner(policy, rank=rank, learning_rate=learning_rate, seed=seed)
    adapted = policy.prepare(observation, digest(raw), prompt)
    zero_adapter = policy.sample(adapted, noise, steps=10).value
    adapter_error = float((native - zero_adapter).abs().max())
    if adapter_error > 1e-6:
        raise RuntimeError("Zero adapters changed the initial policy")
    # Synthetic diagnostic ONLY: test the actual native loss/backward path.
    # Never admit this generated action as observed-useful training evidence.
    for p in learner.parameters.values():
        p.requires_grad_(True)
    t = 0.5
    model_velocity = prepare_velocity(policy.policy, batch, differentiable=True)
    x_t = t * noise + (1 - t) * native.detach()
    loss = (model_velocity(x_t, t) - (noise - native.detach())).square().mean()
    loss.backward()
    grad_norm = (
        sum(
            float(p.grad.float().square().sum())
            for p in learner.parameters.values()
            if p.grad is not None
        )
        ** 0.5
    )
    learner.optimizer.zero_grad(set_to_none=True)
    policy.policy.requires_grad_(False)
    if not np.isfinite(grad_norm) or grad_norm <= 0:
        raise RuntimeError("Native action-expert loss has no finite adapter gradient")
    receipt = {
        "native_zero_guidance_max_abs": native_error,
        "zero_adapter_max_abs": adapter_error,
        "weighted_input_gradient_norm": float(grad.norm()),
        "native_adapter_gradient_norm": grad_norm,
        "guidance_latency_seconds": timing["latency_seconds"],
        "guided_endpoint_masked_error": float(((guided - target) * mask).norm()),
        "native_endpoint_masked_error": float(((native - target) * mask).norm()),
        "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
        "environment_actions": 0,
        "policy_updates": 0,
        "synthetic_diagnostic_not_training_data": True,
        "claim": "numerical and backward-path diagnostic only; not task success",
    }
    return learner, receipt


def main():
    protocol = load_protocol()
    if torch.cuda.device_count() != 1 or "L40S" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Allocate one OSMO L40S before running this study")
    task_index, seed_index = (
        int(os.environ.get("ASTRA_LEARNING_TASK_INDEX", "0")),
        int(os.environ.get("ASTRA_LEARNING_SEED_INDEX", "0")),
    )
    task = protocol["pilot"]["development_tasks"][task_index]
    seed = protocol["pilot"]["seeds"][seed_index]
    phase = os.environ.get("ASTRA_LEARNING_PHASE", "preflight")
    if phase not in ("preflight", "pilot"):
        raise ValueError("Unknown study phase")
    allocation = float(os.environ["ASTRA_WORKER_GPU_HOURS"])
    if (
        not np.isfinite(allocation)
        or not 0 < allocation <= protocol["compute"]["authorized_gpu_hours"]
    ):
        raise ValueError("Worker allocation is outside the recorded total budget")
    RESULTS.mkdir(parents=True, exist_ok=False)
    archive = WorkerArchive(int(os.environ.get("ASTRA_WORKER_INDEX", "0")))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(2)
    start = time.monotonic()
    write_json(RESULTS / "protocol.json", protocol)
    write_json(
        RESULTS / "runtime.json",
        {
            "phase": phase,
            "task": task,
            "seed": seed,
            "gpu": torch.cuda.get_device_name(0),
            "workflow": os.environ["ASTRA_RUN_ID"],
            "source_revision": os.environ.get("ASTRA_SOURCE_REVISION"),
            "payload_sha256": os.environ["PAYLOAD_SHA256"],
        },
    )
    client = None
    if phase == "pilot":
        client = CodexRelayClient(
            model="gpt-6-astra",
            family="reasoning_learning",
            response_log=str(RESULTS / "provider.jsonl"),
            timeout=protocol["teacher"]["timeout_seconds_per_call"],
        )
        client.ensure_server()
    try:
        policy = load_frozen_policy()
        if policy.horizon != protocol["checkpoint"]["runtime_action_horizon"]:
            raise ValueError("Runtime horizon differs from the preregistered protocol")
        # Archived real observation provides a zero-interaction engineering probe.
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
            probe_prompt = str(data["prompt"].item())
        learner, receipt = preflight(
            policy,
            observation,
            probe_prompt,
            seed=seed,
            rank=protocol["learner"]["rank"],
            learning_rate=protocol["learner"]["learning_rate"],
        )
        if protocol["teacher"].get("semantic_interventions", False):
            from astra_reversal.reasoning_learning.semantic import (
                preflight as semantic_preflight,
            )

            write_json(
                RESULTS / "semantic_preflight.json",
                semantic_preflight(policy, observation, probe_prompt),
            )
        write_json(RESULTS / "preflight.json", receipt)
        archive.sync()
        if phase == "preflight":
            write_json(
                RESULTS / "completion.json",
                {"status": "preflight_complete", "research_objective_met": False},
            )
            return
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
        collection_steps, evaluation_steps = 0, 0
        collection, evaluations, updates = [], [], []
        collection_history = []
        stopped = False
        reference_key = f"{task['suite']}:{task['task_id']}:{seed}"
        reference_config = protocol.get("teacher_native_evaluation_references", {}).get(
            reference_key
        )
        if reference_config:
            filename = reference_config["file"]
            if Path(filename).name != filename:
                raise ValueError("Native reference must be a bundled basename")
            reference = reuse_native_evaluation(
                Path(__file__).parents[1] / "reasoning_learning" / filename,
                reference_config["sha256"],
                protocol=protocol,
                metadata=policy.metadata,
                preflight=receipt,
                task=task,
                seed=seed,
                manifest=manifest,
            )
            evaluations.append(reference)
            write_json(RESULTS / "native_evaluation_reference.json", reference)
            write_json(RESULTS / "learning_curve.json", evaluations)
            archive.sync()
        for iteration in range(len(resets) + 1):
            if iteration in protocol["pilot"][
                "evaluation_after_collection_rollouts"
            ] and not (iteration == 0 and reference_config):
                rows = []
                for reset_id in eval_resets:
                    entry = by_reset[reset_id]
                    env, _, _ = create(task["task_id"], seed)
                    result, _, _ = collect(
                        policy,
                        None,
                        env,
                        entry,
                        benchmark,
                        RESULTS / f"evaluation_{iteration}" / f"reset_{reset_id}",
                        version=learner.version,
                        seed=seed * 1000 + reset_id,
                        retain_step_observations=protocol["pilot"].get(
                            "record_all_evaluation_observations", True
                        ),
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
            # If the previous collection completed a scheduled checkpoint,
            # finish that evaluation before honoring an early-stop request.
            if Path("/tmp/astra-stop-after-rollout").exists():
                stopped = True
                break
            # Workflow timeout provides the hard cap; this bounds starting more
            # collection when its allocation has already expired.
            if (time.monotonic() - start) / 3600 >= allocation:
                raise TimeoutError("Worker GPU-hour allocation exhausted")
            entry = by_reset[resets[iteration]]
            env, _, _ = create(task["task_id"], seed)
            result, windows, observations = collect(
                policy,
                client,
                env,
                entry,
                benchmark,
                RESULTS / "collection" / f"rollout_{iteration}",
                version=learner.version,
                seed=seed * 1000 + resets[iteration],
                strengths=protocol["teacher"]["independent_candidate_strengths"],
                candidate_configurations=protocol["teacher"].get(
                    "candidate_configurations"
                ),
                collection_history=collection_history,
                temporal_diagnosis=protocol["teacher"].get("temporal_diagnosis", False),
                semantic_interventions=protocol["teacher"].get(
                    "semantic_interventions", False
                ),
                controller_delta_limits=protocol["teacher"].get(
                    "controller_delta_limits"
                ),
                max_episodes=protocol["teacher"]["soft_correction_episodes"],
                max_active_actions=protocol["teacher"]["max_actions_per_active_plan"],
                monitor_every=protocol["teacher"]["monitor_every_actions"],
                progress=archive.sync,
            )
            collection.append(result)
            collection_history.append(result["collection_summary"])
            collection_steps += result["total_control_steps"]
            if windows:
                update = learner.update(
                    windows,
                    observations,
                    updates=protocol["learner"]["update_steps_per_collection_rollout"],
                )
                update["checkpoint"] = learner.save(
                    RESULTS / "checkpoints" / f"policy_{learner.version}"
                )
                updates.append(update)
            write_json(RESULTS / "collection.json", collection)
            write_json(RESULTS / "updates.json", updates)
            archive.sync()
        write_json(
            RESULTS / "completion.json",
            {
                "status": "stopped_after_saved_rollout"
                if stopped
                else "pilot_complete",
                "collection_steps": collection_steps,
                "evaluation_steps": evaluation_steps,
                "policy_updates": learner.version,
                "research_objective_met": False,
                "strong_rl_comparison": "not_yet_run",
            },
        )
    finally:
        write_json(
            RESULTS / "cost.json",
            {
                "worker_gpu_hours": (time.monotonic() - start) / 3600,
                "bootstrap_gpu_hours_excluded": True,
                "note": "Use OSMO task start/end timestamps for allocation including bootstrap.",
            },
        )
        archive.sync()


if __name__ == "__main__":
    main()
