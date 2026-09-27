"""Provider-free, paired rollouts for the frozen learned-correction recipe."""

import copy
import json
from pathlib import Path

from .action_adapter import ActionAdapter, ActionSpec
from .config import BenchmarkConfig
from .flow import error_metrics
from .interpolation_catalog import donor_catalog
from .intervention_rollout import run_rollout
from .intervention_search import write_json
from .recipe_selector import canonical_choice, prepare_native_features
from .records import Recorder, digest, file_sha256, to_numpy
from .representation_search import keyed_rng

ARMS = (
    "native",
    "recorded_schedule",
    "learned_selector",
    "flow_head",
    "gated_flow_head",
)


def load_protocol():
    value = json.loads(
        (
            Path(__file__).parent / "configs/learned_correction_recipe_v1.json"
        ).read_text()
    )
    if (
        value["schema_version"] != "learned-correction-recipe-1.0"
        or value["arms"] != list(ARMS)
        or value["provider_calls_allowed"]
        or value["evaluation"]["physical_rollouts"] != 240
    ):
        raise ValueError("Recipe protocol differs")
    return value


def assignment(worker):
    if type(worker) is not int or not 0 <= worker < 5:
        raise ValueError("Recipe evaluation worker must be in [0,5)")
    if worker == 4:
        return "libero_10", [0, 1, 2, 3]
    return ("libero_goal_ood" if worker < 2 else "libero_spatial_ood"), list(
        range(5 * (worker % 2), 5 * (worker % 2) + 5)
    )


def recorded_schedules(samples):
    """Select a teacher using the frozen order, never an evaluation outcome."""
    order = {
        "phase:astra_tli": 0,
        "phase:astra_tei": 1,
        "representation:astra_tli": 2,
        "representation:astra_tei": 3,
    }
    trajectories = {}
    for sample in samples:
        if sample["kind"] != "correction":
            continue
        key = sample["trajectory_id"]
        trajectories.setdefault(key, []).append(sample)
    candidates = {}
    for trajectory_id, rows in trajectories.items():
        rows.sort(key=lambda r: r["observation_step"])
        row = rows[0]
        if row["observation_step"] != 0:
            raise ValueError("Recorded schedule must start at step zero")
        rank = (
            order[row["source_study"] + ":" + row["arm"]],
            row["revision"],
            row["episode_id"],
            trajectory_id,
        )
        candidates.setdefault(row["original_prompt"], []).append((rank, rows))
    schedules = {}
    for prompt, choices in candidates.items():
        _, rows = min(choices, key=lambda pair: pair[0])
        schedules[prompt] = {
            "trajectory_id": rows[0]["trajectory_id"],
            "source_receipt_sha256": rows[0]["source_receipt_sha256"],
            "choices": [
                {"step": r["observation_step"], "choice": canonical_choice(r["choice"])}
                for r in rows
            ],
        }
    return schedules


def scheduled_choice(schedules, prompt, step):
    if prompt not in schedules:
        return {"operator": "native"}
    values = [r["choice"] for r in schedules[prompt]["choices"] if r["step"] <= step]
    if not values:
        raise ValueError("Recorded schedule does not cover its initial step")
    return copy.deepcopy(values[-1])


def condition_for_choice(policy, observation, prompt, choice, banks):
    choice = canonical_choice(choice)
    if choice["operator"] == "native":
        return policy.prepare(observation, digest(observation), prompt), None
    sources = {r["source_id"]: r["prompt"] for r in donor_catalog()}
    a, b = choice["source_a_id"], choice["source_b_id"]
    return policy.prepare_interpolated(
        observation,
        digest(observation),
        prompt,
        source_prompts=(sources[a], sources[b]),
        alpha=choice["alpha"],
        operator=choice["operator"],
        text_latents={"a": banks[a], "b": banks[b]}
        if choice["operator"] == "tli"
        else None,
    )


def rollout(
    policy,
    create_env,
    entry,
    arm,
    directory,
    *,
    selector=None,
    head=None,
    banks=None,
    schedules=None,
    expected_reset=None,
    collect=False,
    probe=False,
):
    """Run one physical trial; collect only the actually executed window prefix."""
    from .recipe_learning import ResidualActionHead, base_identity, prepare_adapted

    if arm not in ARMS:
        raise ValueError("Unknown recipe arm")
    benchmark = BenchmarkConfig.preset(entry["suite"])
    recorder = Recorder(directory)
    directory = Path(directory)
    env, _, _ = create_env(entry["task_id"], entry["seed"])
    spec = ActionSpec.from_environment(env, policy.horizon, policy.action_dim)
    adapter = ActionAdapter(spec, policy.input_transform, policy.output_transform)
    recorder.event("recipe_rollout_start", entry=entry, arm=arm, action_spec=spec)
    records, collected, decisions = [], [], []
    counters = {
        "velocity_evaluations": 0,
        "probe_velocity_evaluations": 0,
        "prefix_evaluations": 0,
        "probe_prefix_evaluations": 0,
        "clipped_predicted_values": 0,
        "provider_calls": 0,
        "provider_tokens": 0,
    }
    choice, gate = {"operator": "native"}, False
    checks = {}

    def act(observation, step):
        nonlocal choice, gate
        native_condition, feature_receipt = None, None
        if step % 25 == 0:
            if arm == "recorded_schedule":
                choice = scheduled_choice(schedules, entry["instruction"], step)
                gate = choice["operator"] != "native"
                decisions.append(
                    {
                        "step": step,
                        "choice": choice,
                        "gate_active": gate,
                        "gate_probability": None,
                    }
                )
            elif arm in ("learned_selector", "gated_flow_head"):
                native_condition, features, feature_receipt = prepare_native_features(
                    policy, observation, entry["instruction"]
                )
                counters["prefix_evaluations"] += 1
                decision = selector.predict(features)
                choice, gate = decision["choice"], decision["gate_active"]
                decisions.append(
                    {"step": step, **decision, "feature_receipt": feature_receipt}
                )
                recorder.event(
                    "recipe_selector_decision",
                    step=step,
                    features=features,
                    decision=decisions[-1],
                )
        adapted = arm == "flow_head" or (arm == "gated_flow_head" and gate)
        if adapted:
            condition, provenance = prepare_adapted(
                policy, observation, digest(observation), entry["instruction"], head
            )
            counters["prefix_evaluations"] += 1
        elif (
            arm in ("recorded_schedule", "learned_selector")
            and choice["operator"] != "native"
        ):
            condition, provenance = condition_for_choice(
                policy, observation, entry["instruction"], choice, banks
            )
            counters["prefix_evaluations"] += 1
        else:
            condition = native_condition or policy.prepare(
                observation, digest(observation), entry["instruction"]
            )
            counters["prefix_evaluations"] += int(native_condition is None)
            provenance = None
        noise = policy.noise(keyed_rng(entry, 0, step))
        sample = policy.sample(condition, noise, steps=10, solver="euler")
        actions, clipping = adapter.decode(sample.value, condition.state)
        counters["velocity_evaluations"] += sample.velocity_evaluations
        counters["clipped_predicted_values"] += clipping["count"]
        if probe and step == 0:
            if arm != "native":
                raise ValueError("Native probes require the native arm")
            reference = policy.reference_actions(condition, noise, steps=10)
            decoded = policy.output_transform({"actions": to_numpy(sample.value)[0]})[
                "actions"
            ]
            checks["upstream_native_parity"] = error_metrics(reference, decoded)
            observed, _, feature = prepare_native_features(
                policy, observation, entry["instruction"]
            )
            observed_sample = policy.sample(observed, noise, steps=10, solver="euler")
            checks["feature_capture_parity"] = error_metrics(
                sample.value, observed_sample.value
            )
            zero = ResidualActionHead(base_identity(policy), device=policy.device)
            zero_condition, _ = prepare_adapted(
                policy, observation, digest(observation), entry["instruction"], zero
            )
            zero_sample = policy.sample(zero_condition, noise, steps=10, solver="euler")
            checks["zero_head_parity"] = error_metrics(sample.value, zero_sample.value)
            counters["probe_velocity_evaluations"] += 30
            counters["probe_prefix_evaluations"] += 3
            if (
                checks["upstream_native_parity"]["max_abs"] > 1e-5
                or checks["feature_capture_parity"]["max_abs"] != 0
                or checks["zero_head_parity"]["max_abs"] != 0
            ):
                raise RuntimeError("Real checkpoint recipe parity failed")
            recorder.event(
                "recipe_native_probes", checks=checks, feature_receipt=feature
            )
        records.append(
            {
                "step": step,
                "choice": copy.deepcopy(choice),
                "gate_active": gate,
                "head_active": adapted,
                "noise_sha256": digest(to_numpy(noise)),
            }
        )
        recorder.event(
            "recipe_generation",
            observation_step=step,
            observation=observation,
            noise=noise,
            generated_actions=sample.value,
            controller_actions=actions,
            condition_id=condition.condition_id,
            choice=choice,
            gate_active=gate,
            head_active=adapted,
            provenance=provenance,
            clipping=clipping,
            velocity_evaluations=sample.velocity_evaluations,
        )
        if collect:
            collected.append(
                {
                    "observation_step": step,
                    "observation": copy.deepcopy(observation),
                    "actions": actions.copy(),
                }
            )
        return actions

    result = run_rollout(
        env,
        entry,
        benchmark,
        act,
        execute_steps=5,
        expected_reset=expected_reset,
        policy_image_size=policy.observation_image_size,
        video_path=directory / "rollout.mp4",
    )
    snapshots = result.pop("snapshots")
    if result["initial_success"]:
        raise ValueError("Initially successful recipe reset is invalid")
    result.update(
        arm=arm,
        suite=entry["suite"],
        task_id=entry["task_id"],
        seed=entry["seed"],
        initial_state_id=entry["initial_state_id"],
        instruction=entry["instruction"],
        status="complete",
        checks=checks,
        decisions=decisions,
        generations=records,
        video_sha256=file_sha256(directory / "rollout.mp4"),
        video_fps=20,
        **counters,
    )
    result["head_active_actions"] = sum(
        min(5, result["actions_executed"] - row["step"])
        for row in records
        if row["head_active"]
    )
    result["gate_active_actions"] = sum(
        min(5, result["actions_executed"] - row["step"])
        for row in records
        if row["gate_active"]
    )
    recorder.event("recipe_rollout_complete", result=result, snapshots=snapshots)
    write_json(directory / "summary.json", result)
    receipt = file_sha256(directory / "summary.json")
    samples = []
    if collect and result["success"]:
        for row in collected:
            k = min(5, result["actions_executed"] - row["observation_step"])
            samples.append(
                {
                    "sample_id": digest(
                        (entry["episode_id"], row["observation_step"], receipt)
                    ),
                    "trajectory_id": digest(entry["episode_id"]),
                    "episode_id": entry["episode_id"],
                    "original_prompt": entry["instruction"],
                    "observation_step": row["observation_step"],
                    "executed_actions": row["actions"][:k].copy(),
                    "observation": row["observation"],
                    "choice": {"operator": "native"},
                    "kind": "anchor",
                    "source_study": "fresh_native",
                    "arm": "native",
                    "revision": 0,
                    "source_receipt_sha256": receipt,
                    "action_spec": spec.as_dict(),
                }
            )
    return result, samples
