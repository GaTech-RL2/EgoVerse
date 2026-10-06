"""Trusted simulator evaluator. Candidate harnesses never receive this object."""

import time
from pathlib import Path

import numpy as np

from astra_reversal.action_adapter import ActionAdapter, ActionSpec
from astra_reversal.demo_segments import write_json
from astra_reversal.demo_skill_conditioning import InputSkillConditioner
from astra_reversal.intervention_rollout import run_rollout
from astra_reversal.records import digest, to_numpy

from .runtime import Runtime, Trace


def evaluate_episode(
    *,
    policy,
    create_env,
    benchmark,
    entry,
    compiler,
    limits,
    harness,
    worker,
    directory,
    mode,
    expected_reset=None,
    video=False,
):
    if policy.horizon != limits.prediction_horizon:
        raise ValueError("Prediction horizon differs from the fixed 50-action contract")
    if any(parameter.requires_grad for parameter in policy.policy.parameters()):
        raise ValueError("System 1 must remain frozen")
    trace = Trace(directory)
    runtime = Runtime(
        compiler=compiler,
        limits=limits,
        harness=harness,
        worker=worker,
        original_goal=entry["instruction"],
        episode_id=entry["episode_id"],
        trace=trace,
        mode=mode,
    )
    conditioner = InputSkillConditioner(policy, compiler.bank, {})
    env, task, _ = create_env(entry["task_id"], entry["seed"])
    if task.language != entry["instruction"]:
        env.close()
        runtime.close()
        raise ValueError("Original task instruction changed")
    spec = ActionSpec.from_environment(env, policy.horizon, policy.action_dim)
    adapter = ActionAdapter(spec, policy.input_transform, policy.output_transform)
    timings = []
    checked_parity = False
    original_metadata = digest(policy.metadata)

    def act(live, step):
        nonlocal checked_parity
        began = time.monotonic()
        if digest(policy.metadata) != original_metadata:
            raise ValueError("Frozen checkpoint/preprocessing identity changed")
        choice = runtime.boundary(live, step)
        raw_identity = digest(live)
        condition, receipt, effective = conditioner.prepare(
            live, entry["instruction"], choice
        )
        if digest(live) != raw_identity or not np.array_equal(
            effective["observation/state"], live["observation/state"]
        ):
            raise ValueError(
                "Input intervention changed live observations or proprioception"
            )
        rng = np.random.default_rng(
            np.random.SeedSequence(
                [entry["seed"], int(digest(entry["episode_id"])[:8], 16), step, 0]
            )
        )
        noise = policy.noise(rng)
        sample = policy.sample(
            condition, noise, solver="euler", steps=limits.solver_steps, time_power=1.0
        )
        actions, clipping = adapter.decode(sample.value, condition.state)
        if not checked_parity and choice is None:
            reference = policy.reference_actions(
                condition, noise, steps=limits.solver_steps
            )
            decoded = policy.output_transform({"actions": to_numpy(sample.value)[0]})[
                "actions"
            ]
            error = float(np.max(np.abs(np.asarray(reference) - decoded)))
            trace.event("parity", {"action": step, "native_max_abs": error})
            if error > 1e-5:
                raise ValueError("Native sampler parity failed")
            checked_parity = True
        elapsed = time.monotonic() - began
        timings.append(elapsed)
        trace.event(
            "timing",
            {
                "action": step,
                "replan_seconds": elapsed,
                "deadline_seconds": 5 / 20,
                "deadline_missed": elapsed > 5 / 20,
                "deadline_scope": "diagnostic 20Hz target; simulator is not claimed to meet it",
            },
        )
        trace.event(
            "execution",
            {
                "action": step,
                "conditioning": receipt,
                "cache_identity": conditioner.cache_identity,
                "noise_sha256": digest(to_numpy(noise)),
                "planned_actions_sha256": digest(actions),
                "execute_prefix": [0, 5],
                "predicted_actions": len(actions),
                "clipping": clipping,
            },
        )
        return actions

    try:
        result = run_rollout(
            env,
            entry,
            benchmark,
            act,
            execute_steps=5,
            action_budget=300,
            expected_reset=expected_reset,
            policy_image_size=policy.observation_image_size,
            video_path=Path(directory) / "rollout.mp4" if video else None,
        )
        result.pop(
            "snapshots"
        )  # Raw observation traces remain; no success predicate reaches runtime.
    finally:
        controller = runtime.close()
        write_json(Path(directory) / "controller.json", controller)
    result.update(
        controller=controller,
        native_parity_checked=checked_parity,
        task_key=f"{entry['suite']}:{entry['task_id']}",
        sensor_access="raw paired cameras plus live state8",
        checkpoint_identity=original_metadata,
        card_manifest_sha256=compiler.identity,
        motor_deadline_misses=sum(t > 0.25 for t in timings),
    )
    write_json(Path(directory) / "outcomes.json", result)
    return result


def metrics(results):
    if not results:
        raise ValueError("No physical rollout results")
    groups = {}
    calls, latencies, inputs, outputs, stale, invalid, misses = [], [], 0, 0, 0, 0, 0
    unknown_usage = 0
    identities = set()
    for result in results:
        identity = (result["task_key"], result["episode_id"])
        if identity in identities:
            raise ValueError("Duplicate physical episode in evaluation arm")
        identities.add(identity)
        groups.setdefault(result["task_key"], []).append(bool(result["success"]))
        controller = result["controller"]
        calls.append(controller["runtime_requests"])
        misses += result["motor_deadline_misses"]
        for receipt in controller["tool_events"]:
            error = receipt.get("validation_error")
            invalid += bool(error)
            stale += bool(
                error and (error.startswith("stale") or error == "episode_ended")
            )
        for record in controller["provider_records"]:
            latencies.append(record["latency_seconds"])
            usage = record.get("usage")
            if not usage or any(
                type(usage.get(key)) is not int
                for key in ("input_tokens", "output_tokens")
            ):
                unknown_usage += 1
            else:
                inputs += usage["input_tokens"]
                outputs += usage["output_tokens"]
    per_task = {
        task: {
            "successes": sum(rows),
            "episodes": len(rows),
            "success_rate": float(np.mean(rows)),
        }
        for task, rows in groups.items()
    }
    return {
        "episodes": len(results),
        "per_task": per_task,
        "task_macro_success": float(
            np.mean([v["success_rate"] for v in per_task.values()])
        ),
        "runtime_requests": sum(calls),
        "input_tokens_known": inputs,
        "output_tokens_known": outputs,
        "unknown_usage_records": unknown_usage,
        "invalid_calls": invalid,
        "stale_calls": stale,
        "invalid_call_rate": invalid / sum(calls) if sum(calls) else 0.0,
        "stale_call_rate": stale / sum(calls) if sum(calls) else 0.0,
        "motor_deadline_misses": misses,
        "latency_p50_seconds": float(np.percentile(latencies, 50))
        if latencies
        else None,
        "latency_p95_seconds": float(np.percentile(latencies, 95))
        if latencies
        else None,
    }


def paired_difference(left, right, *, seed=137, samples=2000):
    """Bootstrap entire task groups; retain pairing within each reset."""

    def key(row):
        return row["task_key"], row["episode_id"]

    a, b = {key(r): r for r in left}, {key(r): r for r in right}
    if set(a) != set(b) or len(a) != len(left) or len(b) != len(right):
        raise ValueError("Paired arms require identical unique task/reset cohorts")
    grouped = {}
    for identity in sorted(a):
        if a[identity]["reset_audit"] != b[identity]["reset_audit"]:
            raise ValueError("Paired arm reset audits differ")
        grouped.setdefault(identity[0], []).append(
            int(b[identity]["success"]) - int(a[identity]["success"])
        )
    values = np.array([np.mean(v) for v in grouped.values()])
    rng = np.random.default_rng(seed)
    boot = [
        rng.choice(values, len(values), replace=True).mean() for _ in range(samples)
    ]
    return {
        "right_minus_left": float(values.mean()),
        "task_bootstrap_95_percent": np.percentile(boot, [2.5, 97.5]).tolist(),
        "tasks": len(values),
        "exploratory": len(values) <= 4,
        "method": "matched reset differences; whole-task bootstrap",
    }
