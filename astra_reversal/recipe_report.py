"""Offline, paired reporting for the frozen learned-correction recipe.

Reads simulator records; never imports a policy, fits a model, or calls a
provider. Completion seals and recorded arrays' descriptors are checked here.
This is not an independent replay of hidden states, optimization, or physics.
"""

import argparse
import copy
import csv
import html
import json
import math
import shutil
import statistics
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np

from .records import digest, file_sha256

SCHEMA = "recipe-report-1.0"
ARMS = (
    "native",
    "recorded_schedule",
    "learned_selector",
    "flow_head",
    "gated_flow_head",
)
SUITES = ("libero_goal_ood", "libero_spatial_ood", "libero_10")
LIMITATIONS = [
    "Training uses successful historical trajectories on known task compositions; new resets do not establish held-out-task or zero-shot generalization.",
    "The ID retention panel covers four of the ten standard LIBERO-10 tasks, with two prescribed states each.",
    "Every arm runs unconditionally on every evaluation reset. Successes are not inherited from native and no best-of-attempts selection is used.",
    "The selector gate imitates successful teacher intervention labels; it is not a calibrated failure probability. Its fixed threshold is strictly greater than 0.5.",
    "The gated head preserves original native conditioning; it does not compose TEI/TLI with the residual head.",
    "Executed controller prefixes supervise the head; a native-generated unexecuted suffix supplies input context, not demonstration labels.",
    "Zero new provider calls excludes the nonzero historical cost of acquiring and auditing teachers. No monetary price or amortized saving is inferred.",
    "This report verifies immutable recorded metadata, event joins, keyed noise descriptors and paired reset hashes. It does not independently recompute hidden states, optimizer updates or simulator outcomes.",
    "Head condition IDs are checked against the exact native-ID/head/config composition. Matching raw observation descriptors and prompts must share a native condition ID; non-overlapping observations are not independently rehashed from omitted array bytes.",
    "Rollout wall time includes reset, policy, simulator and video work; its components must not be added to that wall time. Parallel worker sums are resource time, not experiment elapsed time.",
    "Videos are recorded external-camera frames before each executed action at 20 fps. They omit inference pauses and the terminal post-action image.",
]


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, name, minimum=0):
    _require(type(value) is int and value >= minimum, f"Invalid {name}")
    return value


def _number(value, name):
    _require(
        type(value) in (int, float) and math.isfinite(value) and value >= 0,
        f"Invalid {name}",
    )
    return value


def _sha(value):
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value),
        "Invalid SHA256",
    )
    return value


def _safe(root, name):
    path = Path(name)
    _require(
        not path.is_absolute() and ".." not in path.parts and path.parts,
        "Unsafe artifact path",
    )
    result = Path(root) / path
    _require(
        result.resolve().is_relative_to(Path(root).resolve()),
        "Artifact escapes input directory",
    )
    return result


def _json(path):
    return json.loads(
        Path(path).read_text(),
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"Nonfinite JSON: {value}")
        ),
    )


def _write(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


class Inputs:
    """Record portable names and exact input bytes without leaking local paths."""

    def __init__(self, root, label, seal=None):
        self.root, self.label, self.seal = Path(root), label, seal
        self.files = {}

    def path(self, name):
        path = _safe(self.root, name)
        _require(path.is_file(), f"Missing required artifact: {self.label}/{name}")
        record = {"sha256": file_sha256(path), "bytes": path.stat().st_size}
        if self.seal is not None:
            _require(
                self.seal.get(name) == record,
                f"Completion seal differs: {self.label}/{name}",
            )
        self.files[name] = record
        return path

    def read(self, name):
        return _json(self.path(name))

    def provenance(self):
        return {
            "input": self.label,
            "files": copy.deepcopy(self.files),
            "inventory_sha256": digest(self.files),
        }


def _protocol(value):
    frozen = _json(Path(__file__).parent / "configs/learned_correction_recipe_v1.json")
    _require(value == frozen, "Protocol differs from the frozen recipe")
    return digest(value)


def _assignment(worker):
    _integer(worker, "worker")
    _require(worker < 5, "Unknown evaluation worker")
    return (
        ("libero_10", list(range(4)))
        if worker == 4
        else (SUITES[worker // 2], list(range(5 * (worker % 2), 5 * (worker % 2) + 5)))
    )


def _id(suite, task, state):
    return f"{suite}:seed61:task{task}:state{state}"


def _manifest(value, suite, tasks, states):
    _require(
        value["sha256"] == digest({k: v for k, v in value.items() if k != "sha256"}),
        "Reset manifest digest differs",
    )
    _require(
        value["schema_version"] == "intervention_reset_v1" and value["seed"] == 61,
        "Reset manifest namespace differs",
    )
    from dataclasses import asdict

    from .config import BenchmarkConfig

    _require(
        value["benchmark"] == asdict(BenchmarkConfig.preset(suite)),
        "Reset benchmark differs",
    )
    expected = [_id(suite, task, state) for task in tasks for state in states]
    _require(
        [r["episode_id"] for r in value["episodes"]] == expected,
        "Reset manifest assignment differs",
    )
    _require(
        value["cases"] == [[task, state] for task in tasks for state in states],
        "Reset manifest case indexes differ",
    )
    entries = {}
    for entry in value["episodes"]:
        _require(
            entry["suite"] == suite
            and entry["seed"] == 61
            and entry["episode_id"]
            == _id(suite, entry["task_id"], entry["initial_state_id"]),
            "Reset entry identity differs",
        )
        state = np.asarray(entry["reset_state"], np.float64)
        model = {
            k: np.asarray(entry["reset_model"][k], np.float64)
            for k in ("body_pos", "body_quat")
        }
        _require(
            state.ndim == 1
            and state.size
            and np.isfinite(state).all()
            and digest(state) == entry["reset_state_sha256"],
            "Reset state bytes differ",
        )
        _require(
            model["body_pos"].ndim == 2
            and model["body_pos"].shape[1] == 3
            and model["body_quat"].shape == (len(model["body_pos"]), 4)
            and all(np.isfinite(v).all() for v in model.values())
            and digest(model) == entry["reset_model_sha256"],
            "Reset model geometry differs",
        )
        _sha(entry["bddl_sha256"])
        _require(
            type(entry["initially_successful"]) is bool
            and isinstance(entry["instruction"], str)
            and entry["instruction"],
            "Invalid reset instruction or success flag",
        )
        if suite == "libero_10":
            _sha(entry["prescribed_state_asset_sha256"])
        entries[entry["episode_id"]] = entry
    return entries


def _runtime(value, phase, worker, payload=None):
    _require(
        value["phase"] == phase
        and value["worker"] == worker
        and value["provider_credential_attached"] is False
        and value["tf32"] is False
        and "L40S" in value["gpu"],
        "Runtime phase/device/provider/TF32 differs",
    )
    _sha(value["payload_sha256"])
    _require(
        payload is None or value["payload_sha256"] == payload,
        "Runtime payload differs from training",
    )
    _require(
        isinstance(value["workflow"], str)
        and value["workflow"]
        and "/" not in value["workflow"],
        "Invalid workflow identity",
    )


def _weights(inputs, expected=None):
    before, after = (
        inputs.read("frozen_weights_before.json"),
        inputs.read("frozen_weights_after.json"),
    )
    _require(
        before == after
        and before["tensors"]
        and digest(before["tensors"]) == before["sha256"],
        "Frozen native parameter receipts differ",
    )
    _require(
        expected is None or before["sha256"] == expected,
        "Native parameters differ from training",
    )
    return before["sha256"]


def _base(checkpoint, identity):
    _require(
        identity["horizon"] == 10
        and identity["model_dim"] == 32
        and identity["hidden_dim"] == 1024,
        "Base architecture differs",
    )
    _require(
        identity["native"]
        and all(checkpoint.get(k) == v for k, v in identity["native"].items()),
        "Checkpoint/preprocessing identity differs",
    )


def _no_api(value):
    _require(
        value["provider_calls"] == 0
        and type(value["provider_calls"]) is int
        and value["provider_tokens"] == 0
        and type(value["provider_tokens"]) is int,
        "Provider calls are forbidden in this recipe",
    )


def _choice(value):
    if value == {"operator": "native"}:
        return value
    _require(
        set(value) == {"operator", "source_a_id", "source_b_id", "alpha"}
        and value["operator"] in ("tei", "tli"),
        "Invalid selector choice",
    )
    _require(
        isinstance(value["source_a_id"], str)
        and isinstance(value["source_b_id"], str)
        and value["source_a_id"] <= value["source_b_id"],
        "Source pair is not canonical",
    )
    _require(
        type(value["alpha"]) in (int, float)
        and math.isfinite(value["alpha"])
        and 0 <= value["alpha"] <= 1,
        "Invalid interpolation alpha",
    )
    _require(
        value["operator"] != "tli"
        or (value["alpha"] != 0.5 and value["source_a_id"] != value["source_b_id"]),
        "Neutral TLI must be canonical native",
    )
    return value


def _noise(entry, step):
    identity = int(digest(entry["episode_id"])[:16], 16)
    rng = np.random.default_rng(
        np.random.SeedSequence([entry["seed"], identity, 0, step, 0])
    )
    return digest(rng.standard_normal((1, 10, 32)).astype(np.float32))


def _events(path):
    lines = path.read_bytes().splitlines(keepends=True)
    _require(
        lines and all(line.endswith(b"\n") for line in lines), "Incomplete event line"
    )
    rows = [json.loads(line) for line in lines]
    _require(
        [r["sequence"] for r in rows] == list(range(len(rows))),
        "Event sequence differs",
    )
    return rows


def _descriptors(value):
    if isinstance(value, dict):
        if "array" in value:
            _require(
                set(value) == {"array", "shape", "dtype", "sha256"},
                "Invalid array descriptor",
            )
            yield value
        else:
            for item in value.values():
                yield from _descriptors(item)
    elif isinstance(value, list):
        for item in value:
            yield from _descriptors(item)


def _rollout(
    inputs,
    relative,
    row,
    entry,
    *,
    probe,
    schedules,
    outer=False,
    head_sha=None,
    head_zero_effect=None,
    native_condition_bindings=None,
):
    """Join the recorded physical run to its summary and event stream."""
    summary_path = inputs.path(relative + "/summary.json")
    summary = _json(summary_path)
    expected = {
        k: v
        for k, v in row.items()
        if not outer or k not in ("relative_directory", "summary_sha256")
    }
    _require(summary == expected, "Rollout index and physical summary differ")
    if outer:
        _require(
            row["summary_sha256"] == file_sha256(summary_path),
            "Rollout summary hash differs",
        )
    _require(summary["status"] == "complete", "Incomplete physical rollout")
    for key in (
        "episode_id",
        "suite",
        "task_id",
        "initial_state_id",
        "seed",
        "instruction",
    ):
        _require(summary[key] == entry[key], f"Rollout {key} differs from frozen reset")
    arm, actions = summary["arm"], _integer(summary["actions_executed"], "actions", 1)
    _require(
        arm in ARMS
        and summary["execute_steps"] == 5
        and summary["action_budget"] == (520 if entry["suite"] == "libero_10" else 300)
        and actions <= summary["action_budget"],
        "Arm/action budget differs",
    )
    for key in (
        "success",
        "terminated",
        "initial_success",
        "captured_initial_success",
        "zero_action_success",
    ):
        _require(type(summary[key]) is bool, f"Non-boolean {key}")
    _require(
        not summary["initial_success"] and not summary["zero_action_success"],
        "Initially successful or zero-action recipe reset is invalid",
    )
    _require(
        summary["captured_initial_success"] == entry["initially_successful"],
        "Captured initial success differs",
    )
    _require(
        summary["success"]
        or summary["terminated"]
        or actions == summary["action_budget"],
        "Failure stopped before the declared budget",
    )
    _no_api(summary)
    for key in (
        "wall_seconds",
        "reset_seconds",
        "policy_seconds",
        "environment_seconds",
    ):
        _number(summary[key], key)
    _require(
        summary["wall_seconds"] + 1e-6
        >= sum(
            summary[k]
            for k in ("reset_seconds", "policy_seconds", "environment_seconds")
        ),
        "Nested rollout timing exceeds wall time",
    )
    reset = summary["reset_audit"]
    _require(
        reset["sha256"] == digest({k: v for k, v in reset.items() if k != "sha256"}),
        "Reset audit digest differs",
    )
    for key in (
        "episode_id",
        "seed",
        "reset_state_sha256",
        "reset_model_sha256",
        "bddl_sha256",
    ):
        _require(
            reset[key] == entry[key], "Reset audit does not bind the full initial scene"
        )
    _require(
        reset["initial_success"] is False and reset["stabilization_steps"] == 10,
        "Invalid stabilization/reset success",
    )
    for key in (
        "post_stabilization_state_sha256",
        "post_stabilization_model_sha256",
        "post_stabilization_observation_sha256",
    ):
        _sha(reset[key])
    events = _events(inputs.path(relative + "/events.jsonl"))
    _require(
        events[0]["kind"] == "recipe_rollout_start"
        and events[0]["entry"] == entry
        and events[0]["arm"] == arm,
        "Initial event/reset binding differs",
    )
    _require(
        events[-1]["kind"] == "recipe_rollout_complete"
        and events[-1]["result"] == summary,
        "Completion event/outcome differs",
    )
    allowed = {
        "recipe_rollout_start",
        "recipe_rollout_complete",
        "recipe_native_probes",
        "recipe_selector_decision",
        "recipe_generation",
    }
    _require(
        all(e["kind"] in allowed for e in events), "Unexpected runtime/provider event"
    )
    _require(
        sum(e["kind"] == "recipe_rollout_complete" for e in events) == 1
        and sum(e["kind"] == "recipe_rollout_start" for e in events) == 1,
        "Duplicated physical rollout event",
    )
    desc_count = 0
    for descriptor in _descriptors(events):
        _sha(descriptor["sha256"])
        name = str(Path(relative) / descriptor["array"])
        _safe(inputs.root, name)
        if inputs.seal is not None:
            _require(
                name in inputs.seal, "Array descriptor is not bound by completion seal"
            )
        desc_count += 1
    generations = [e for e in events if e["kind"] == "recipe_generation"]
    steps = list(range(0, actions, 5))
    _require(
        [g["observation_step"] for g in generations] == steps
        and [g["step"] for g in summary["generations"]] == steps
        and summary["policy_replans"] == len(steps),
        "Generation/action coverage differs",
    )
    decisions = summary["decisions"]
    expected_decision_steps = (
        list(range(0, actions, 25))
        if arm in ("recorded_schedule", "learned_selector", "gated_flow_head")
        else []
    )
    _require(
        [d["step"] for d in decisions] == expected_decision_steps,
        "Selector/schedule decision cadence differs",
    )
    selected_events = [e for e in events if e["kind"] == "recipe_selector_decision"]
    _require(
        [e["decision"] for e in selected_events]
        == (decisions if arm in ("learned_selector", "gated_flow_head") else []),
        "Selector decisions differ from events",
    )
    expected_prefixes = 0
    if native_condition_bindings is None:
        native_condition_bindings = {}
    binding_checks, repeated_bindings = 0, 0
    for record, event in zip(summary["generations"], generations, strict=True):
        step = record["step"]
        choice = _choice(record["choice"])
        gate = record["gate_active"]
        _require(
            type(gate) is bool and type(record["head_active"]) is bool,
            "Non-boolean gate/head state",
        )
        decision = next((d for d in reversed(decisions) if d["step"] <= step), None)
        if decision is None:
            _require(
                choice == {"operator": "native"} and gate is False,
                "Unconditional arm changed selector choice",
            )
        else:
            _require(
                choice == decision["choice"] and gate == decision["gate_active"],
                "Held decision lifetime differs",
            )
            if arm == "recorded_schedule":
                schedule = schedules.get(entry["instruction"])
                expected_choice = (
                    next(
                        (
                            r["choice"]
                            for r in reversed(schedule["choices"])
                            if r["step"] <= decision["step"]
                        ),
                        None,
                    )
                    if schedule
                    else {"operator": "native"}
                )
                _require(
                    choice == expected_choice
                    and gate == (choice["operator"] != "native")
                    and decision["gate_probability"] is None,
                    "Recorded schedule differs from training-only control",
                )
            else:
                probability = _number(decision["gate_probability"], "gate probability")
                _require(
                    probability <= 1
                    and gate == (probability > 0.5)
                    and (gate or choice == {"operator": "native"}),
                    "Selector threshold/tie semantics differ",
                )
        head_active = arm == "flow_head" or (arm == "gated_flow_head" and gate)
        _require(record["head_active"] == head_active, "Head activation differs")
        for key in ("choice", "gate_active", "head_active"):
            _require(
                event[key] == record[key], "Generation application differs from summary"
            )
        _require(
            event["velocity_evaluations"] == 10
            and event["noise"]["shape"] == [1, 10, 32]
            and event["noise"]["dtype"] == "float32"
            and event["noise"]["sha256"]
            == record["noise_sha256"]
            == _noise(entry, step),
            "Paired keyed Gaussian noise or Euler cost differs",
        )
        _require(
            set(event["observation"])
            == {"observation/image", "observation/wrist_image", "observation/state"},
            "Policy received privileged observation keys",
        )
        # One capture on decision steps; another prefix only when an edit/head
        # is actually applied. Other steps always require one prefix.
        feature_step = arm in ("learned_selector", "gated_flow_head") and step % 25 == 0
        edited = head_active or (
            arm == "learned_selector" and choice["operator"] != "native"
        )
        expected_prefixes += 1 + int(feature_step and edited)
        _sha(event["condition_id"])
        native_id = None
        if head_active:
            from .recipe_learning import training_config

            provenance = event["provenance"]
            native_id = _sha(provenance["original_condition_id"])
            zero = provenance["head"]["zero_effect"]
            _require(
                type(zero) is bool and zero == head_zero_effect,
                "Head zero-effect flag differs from saved checkpoint",
            )
            expected_condition_id = (
                native_id
                if zero
                else digest(
                    {
                        "native_condition_id": native_id,
                        "head": head_sha,
                        "config": training_config(),
                    }
                )
            )
            _require(
                provenance["operator"] == "bounded_final_action_expert_residual"
                and provenance["native_prefix_unchanged"] is True
                and event["condition_id"] == expected_condition_id
                and provenance["enabled"] is (not zero)
                and provenance["original_prompt_sha256"] == digest(entry["instruction"])
                and provenance["direct_rows_5_to_9_unchanged"] is True
                and provenance["direct_padding_unchanged"] is True
                and provenance["hard_clipping_used"] is False,
                "Head changed original conditioning or protected projection slots",
            )
            _require(
                provenance["head"]["parameter_sha256"] == head_sha
                and provenance["head"]["optimizer_steps"] == 1000
                and provenance["head"]["test_only"] is False,
                "Applied head checkpoint differs",
            )
            _require(
                _number(provenance["residual_max_abs"], "residual bound") <= 0.500001,
                "Applied head residual exceeds bound",
            )
        elif choice["operator"] == "native":
            native_id = event["condition_id"]
        if native_id is not None:
            # records.digest hashes raw array bytes, not their descriptor hashes.
            # Compare repeated raw conditions without pretending that separate
            # SHA256 values can reconstruct that original concatenated hash.
            observation_key = digest(
                {
                    "prompt": entry["instruction"],
                    "observation": {
                        key: {
                            field: descriptor[field]
                            for field in ("shape", "dtype", "sha256")
                        }
                        for key, descriptor in event["observation"].items()
                    },
                }
            )
            previous = native_condition_bindings.get(observation_key)
            _require(
                previous is None or previous == native_id,
                "Matching raw observation descriptors have different native condition IDs",
            )
            repeated_bindings += int(previous is not None)
            binding_checks += 1
            native_condition_bindings[observation_key] = native_id
    _require(
        summary["velocity_evaluations"] == 10 * len(steps)
        and summary["prefix_evaluations"] == expected_prefixes,
        "Rollout velocity/prefix cost differs",
    )
    _require(
        summary["clipped_predicted_values"]
        == sum(e["clipping"]["count"] for e in generations),
        "Clipping count differs",
    )
    for label in ("head", "gate"):
        _require(
            summary[f"{label}_active_actions"]
            == sum(
                min(5, actions - r["step"])
                for r in summary["generations"]
                if r[f"{label}_active"]
            ),
            "Applied action count differs",
        )
    probes = [e for e in events if e["kind"] == "recipe_native_probes"]
    _require(
        len(probes) == int(probe)
        and summary["probe_velocity_evaluations"] == 30 * int(probe)
        and summary["probe_prefix_evaluations"] == 3 * int(probe),
        "Native parity probe accounting differs",
    )
    if probe:
        _require(
            probes[0]["checks"] == summary["checks"]
            and set(summary["checks"])
            == {"upstream_native_parity", "feature_capture_parity", "zero_head_parity"},
            "Native parity probe evidence differs",
        )
        for key, tolerance in (
            ("upstream_native_parity", 1e-5),
            ("feature_capture_parity", 0),
            ("zero_head_parity", 0),
        ):
            _require(
                _number(summary["checks"][key]["max_abs"], "parity error") <= tolerance,
                "Native parity gate failed",
            )
    else:
        _require(summary["checks"] == {}, "Unexpected parity checks")
    _sha(summary["video_sha256"])
    _require(summary["video_fps"] == 20, "Video timing differs")
    video_name = relative + "/rollout.mp4"
    if inputs.seal is not None:
        _require(
            inputs.seal[video_name]["sha256"] == summary["video_sha256"],
            "Video is not bound to physical outcome",
        )
    return {
        **{
            k: summary[k]
            for k in (
                "episode_id",
                "suite",
                "task_id",
                "seed",
                "initial_state_id",
                "instruction",
                "arm",
                "success",
                "actions_executed",
                "action_budget",
                "policy_replans",
                "terminated",
                "initial_success",
                "captured_initial_success",
                "zero_action_success",
                "wall_seconds",
                "reset_seconds",
                "policy_seconds",
                "environment_seconds",
                "velocity_evaluations",
                "probe_velocity_evaluations",
                "prefix_evaluations",
                "probe_prefix_evaluations",
                "clipped_predicted_values",
                "head_active_actions",
                "gate_active_actions",
                "video_sha256",
                "video_fps",
            )
        },
        "credited_success": summary["success"]
        and actions > 0
        and not summary["initial_success"],
        "reset_audit_sha256": reset["sha256"],
        "reset_entry_sha256": digest(entry),
        "reset_audit": reset,
        "summary_sha256": file_sha256(summary_path),
        "events_sha256": file_sha256(inputs.root / relative / "events.jsonl"),
        "relative_directory": relative,
        "input": inputs.label,
        "decision_count": len(decisions),
        "decisions": decisions,
        "noise_sha256_by_step": {
            str(g["step"]): g["noise_sha256"] for g in summary["generations"]
        },
        "recorded_array_descriptors": desc_count,
        "native_condition_descriptor_binding_checks": binding_checks,
        "matched_prior_native_observation_descriptors": repeated_bindings,
    }


def _cost(rows):
    result = {"physical_rollouts": len(rows), "provider_calls": 0, "provider_tokens": 0}
    for key in (
        "actions_executed",
        "policy_replans",
        "velocity_evaluations",
        "probe_velocity_evaluations",
        "prefix_evaluations",
        "probe_prefix_evaluations",
        "head_active_actions",
        "gate_active_actions",
        "clipped_predicted_values",
        "wall_seconds",
        "reset_seconds",
        "policy_seconds",
        "environment_seconds",
    ):
        result[key] = sum(r[key] for r in rows)
    result["total_velocity_evaluations"] = (
        result["velocity_evaluations"] + result["probe_velocity_evaluations"]
    )
    result["total_prefix_evaluations"] = (
        result["prefix_evaluations"] + result["probe_prefix_evaluations"]
    )
    return result


def _training(directory, protocol):
    inputs = Inputs(directory, "training")
    receipt = inputs.read("training_receipt.json")
    _require(
        receipt["schema_version"] == "recipe-training-1.0"
        and receipt["status"] == "complete"
        and receipt["base_weights_unchanged"] is True,
        "Training has not completed its gates",
    )
    _no_api(receipt)
    _require(
        inputs.read("protocol.json") == protocol
        and receipt["protocol_sha256"] == digest(protocol),
        "Training protocol differs",
    )
    runtime = inputs.read("runtime.json")
    _runtime(runtime, "training", 0)
    _require(receipt["workflow"] == runtime["workflow"], "Training workflow differs")
    _require(
        not (inputs.root / "failure.json").exists(), "Training failure artifact exists"
    )
    _require(
        _weights(inputs) == receipt["native_parameter_sha256"],
        "Training native weights differ",
    )
    _base(inputs.read("checkpoint.json"), receipt["base_identity"])
    manifest = inputs.read("training_reset_manifest.json")
    entries = _manifest(manifest, "libero_10", list(range(4)), [0])
    rows = inputs.read("anchor_collection.json")
    _require(
        rows == receipt["fresh_native_anchor_rollouts"]
        and [r["episode_id"] for r in rows] == list(entries),
        "Training anchor coverage differs",
    )
    anchors = [
        _rollout(
            inputs,
            f"anchor_rollouts/task_{r['task_id']}",
            r,
            entries[r["episode_id"]],
            probe=i == 0,
            schedules={},
        )
        for i, r in enumerate(rows)
    ]
    _require(
        all(r["arm"] == "native" for r in anchors), "Training anchors are not native"
    )
    admission = inputs.read("admission.json")
    _no_api(admission)
    _require(
        inputs.files["admission.json"]["sha256"] == receipt["admission_sha256"]
        and admission["protocol_sha256"] == digest(protocol)
        and admission["training_reset_sha256"] == manifest["sha256"],
        "Training admission identity differs",
    )
    expected_anchors = [
        {
            "episode_id": r["episode_id"],
            "success": r["success"],
            "actions": r["actions_executed"],
            "admitted": r["success"],
        }
        for r in anchors
    ]
    _require(
        admission["fresh_anchor_rollouts"] == expected_anchors,
        "Failed fresh anchors were admitted or successful anchors omitted",
    )
    corpus = inputs.read("corpus_metadata.json")
    _require(
        corpus["status"] == "complete"
        and corpus["corpus_id"]
        == digest({k: v for k, v in corpus.items() if k != "corpus_id"}),
        "Historical corpus identity differs",
    )
    _require(
        inputs.files["corpus_metadata.json"]["sha256"]
        == admission["historical_metadata_sha256"]
        and corpus["arrays"]["sha256"] == admission["historical_array_sha256"],
        "Historical teacher bytes differ",
    )
    _require(
        len(corpus["trajectories"]) == corpus["trajectory_count"] == 19
        and len(corpus["windows"])
        == corpus["window_count"]
        == receipt["historical_windows"]
        == 413,
        "Historical training denominator differs",
    )
    _require(
        all(t["success"] is True for t in corpus["trajectories"]),
        "Unsuccessful historical trajectory admitted",
    )
    kinds = Counter(t["sample_kind"] for t in corpus["trajectories"])
    _require(
        kinds == {"astra_success": 12, "native_anchor": 7},
        "Historical teacher/anchor split differs",
    )
    _require(
        sum(w["executed_count"] for w in corpus["windows"])
        == corpus["executed_action_count"]
        == 2022
        and sum(
            w["executed_count"]
            for w in corpus["windows"]
            if w["sample_kind"] == "astra_success"
        )
        == 1248,
        "Historical actual-action coverage differs",
    )
    sample_ids = [s["sample_id"] for s in admission["samples"]]
    _require(
        len(set(sample_ids)) == len(sample_ids) == receipt["training_windows"],
        "Duplicate or missing training samples",
    )
    historical_ids = {w["sample_id"] for w in corpus["windows"]}
    _require(
        set(sample_ids[:413]) == historical_ids,
        "Historical windows omitted or reordered into fresh data",
    )
    expected_fresh = {}
    for row in anchors:
        if row["success"]:
            for step in range(0, row["actions_executed"], 5):
                sample_id = digest((row["episode_id"], step, row["summary_sha256"]))
                expected_fresh[sample_id] = {
                    "sample_id": sample_id,
                    "kind": "anchor",
                    "trajectory_id": digest(row["episode_id"]),
                    "executed_actions": min(5, row["actions_executed"] - step),
                    "source_receipt_sha256": row["summary_sha256"],
                }
    _require(
        {r["sample_id"]: r for r in admission["samples"][413:]} == expected_fresh,
        "Fresh training windows include failed anchors, evaluation data or unexecuted actions",
    )
    sources = {s["source_id"]: s for s in corpus["sources"]}
    windows = {w["sample_id"]: w for w in corpus["windows"]}
    for row in admission["samples"][:413]:
        w = windows[row["sample_id"]]
        _require(
            row
            == {
                "sample_id": w["sample_id"],
                "kind": "anchor"
                if w["sample_kind"] == "native_anchor"
                else "correction",
                "trajectory_id": w["trajectory_id"],
                "executed_actions": w["executed_count"],
                "source_receipt_sha256": sources[w["source_id"]][
                    "feedback_audit_sha256"
                ],
            },
            "Historical sample provenance differs",
        )
    selector, head = receipt["selector"], receipt["head"]
    from .recipe_learning import training_config
    from .recipe_selector import selector_config

    _require(selector["config"] == selector_config(), "Selector hyperparameters differ")
    _require(
        selector["source_sha256"]
        == head["admission_receipt_sha256"]
        == receipt["admission_sha256"],
        "Learners used different admissions",
    )
    _require(
        selector["optimizer_steps"]
        == head["optimizer_steps"]
        == head["total_optimizer_steps"]
        == 1000
        and selector["test_only"] is False
        and head["test_only"] is False
        and head["status"] == "fitted",
        "Learner update budget or test scope differs",
    )
    _require(
        isinstance(head["device"], str) and head["device"].startswith("cuda"),
        "Production head fit was not recorded on CUDA",
    )
    _require(
        selector["samples"] == len(sample_ids)
        and head["trainable_parameters"] == 7175
        and head["velocity_evaluations"] == head["base_backward_passes"] == 0,
        "Learner sample/parameter/compute scope differs",
    )
    trace = selector["loss_trace"]
    _require(
        [t["step"] for t in trace] == [*range(1, 1000, 50), 1000],
        "Selector loss trace coverage differs",
    )
    for row in trace:
        for key in ("loss", "gate_bce", "pair_ce", "alpha_mse"):
            _number(row[key], key)
    for stage in ("before", "after"):
        for key, value in head[stage].items():
            _require(
                value is False, "Unexpected hard clipping"
            ) if key == "hard_clipping_used" else _number(value, key)
    feature_rows = inputs.read("feature_receipts.json")
    _require(
        [r["sample_id"] for r in feature_rows] == sample_ids,
        "Feature extraction coverage differs",
    )
    bank_ids = {"executed": [], "anchor": []}
    feature_counts = Counter()
    for row, admitted in zip(feature_rows, admission["samples"], strict=True):
        bank, native = row["flow_bank"], row["selector"]
        p = bank["provenance"]
        _require(
            bank["bank_id"] == digest({k: v for k, v in bank.items() if k != "bank_id"})
            and p["base_identity"] == receipt["base_identity"]
            and p["test_only"] is False
            and p["base_parameter_versions_unchanged"] is True,
            "Feature bank identity/base differs",
        )
        kind = "anchor" if admitted["kind"] == "anchor" else "executed"
        _require(
            p["kind"] == kind
            and p["window"]["source_id"] == admitted["sample_id"]
            and p["window"]["source_receipt_sha256"]
            == admitted["source_receipt_sha256"],
            "Feature labels/source differ",
        )
        expected_count = 5 if kind == "anchor" else admitted["executed_actions"]
        _require(
            p["counts"]["velocity_evaluations"] == 18
            and p["counts"]["prefix_preparations"] == 2
            and p["counts"]["feature_rows"] == 8 * expected_count,
            "Feature compute or executed label count differs",
        )
        _require(
            native["prefix_evaluations"] == 1
            and native["velocity_evaluations"] == 0
            and native["raw_observation_sha256"] == p["window"]["observation_sha256"]
            and native["condition_id"] == p["window"]["condition_id"],
            "Selector features are not the same native raw condition",
        )
        bank_ids[kind].append(bank["bank_id"])
        feature_counts["velocity_evaluations"] += 18
        feature_counts["prefix_evaluations"] += 3
        feature_counts[kind + "_rows"] += 8 * expected_count
    _require(
        head["executed_bank_ids"] == bank_ids["executed"]
        and head["anchor_bank_ids"] == bank_ids["anchor"]
        and head["feature_rows"] == feature_counts["executed_rows"]
        and head["anchor_rows"] == feature_counts["anchor_rows"],
        "Head fit bank admission differs",
    )
    _require(
        len(set(bank_ids["executed"] + bank_ids["anchor"])) == len(sample_ids),
        "Repeated feature bank changes sample weighting",
    )
    _require(
        receipt["feature_velocity_evaluations"]
        == feature_counts["velocity_evaluations"]
        and receipt["feature_prefix_evaluations"]
        == feature_counts["prefix_evaluations"],
        "Feature cost differs",
    )
    bundle_receipt = inputs.read("learned_bundle.json")
    _sha(bundle_receipt["sha256"])
    seal = inputs.read("bundle/seal.json")
    bundle = Inputs(inputs.root / "bundle", "training/bundle", seal)
    _require(
        bundle.read("training_receipt.json") == receipt
        and bundle.read("protocol.json") == protocol,
        "Published bundle differs from training",
    )
    selector_meta, head_meta = (
        bundle.read("selector/metadata.json"),
        bundle.read("head/manifest.json"),
    )
    _require(
        selector_meta["training"] == selector
        and set(selector_meta["files"]) == {"weights.pt", "normalization.npz"},
        "Selector saved metadata differs",
    )
    _require(
        selector_meta["implementation_sha256"]
        == file_sha256(Path(__file__).with_name("recipe_selector.py"))
        and head_meta["source_sha256"]
        == file_sha256(Path(__file__).with_name("recipe_learning.py"))
        and head_meta["config"] == training_config(),
        "Learned implementation/config source differs",
    )
    _require(
        head_meta["history"] == [head]
        and head_meta["base_identity"] == receipt["base_identity"]
        and head_meta["parameter_sha256"] == head["after_parameter_sha256"]
        and head_meta["optimizer_steps"] == 1000
        and type(head_meta["zero_effect"]) is bool
        and head_meta["test_only"] is False,
        "Head saved metadata differs",
    )
    # Exact checkpoint bytes must be available at publication; no torch loading.
    for name, expected in selector_meta["files"].items():
        _require(
            file_sha256(bundle.path("selector/" + name)) == expected,
            "Saved selector file differs",
        )
    _require(
        file_sha256(bundle.path("head/state.pt")) == head_meta["state_file_sha256"],
        "Saved head file differs",
    )
    schedules = bundle.read("schedules.json")
    _require(len(schedules) == 8, "Recorded schedule composition coverage differs")
    # Reconstruct the control using only historical records, never eval outcomes.
    ranks = {
        "phase:astra_tli": 0,
        "phase:astra_tei": 1,
        "representation:astra_tli": 2,
        "representation:astra_tei": 3,
    }
    candidates = {}
    for trajectory in corpus["trajectories"]:
        if trajectory["sample_kind"] != "astra_success":
            continue
        rows_for_teacher = sorted(
            (
                w
                for w in corpus["windows"]
                if w["trajectory_id"] == trajectory["trajectory_id"]
            ),
            key=lambda r: r["action_start"],
        )
        first = rows_for_teacher[0]
        rank = (
            ranks[first["source_study"] + ":" + first["arm"]],
            first["revision"],
            first["episode_id"],
            first["trajectory_id"],
        )
        candidates.setdefault(first["instruction"], []).append((rank, rows_for_teacher))
    for prompt, options in candidates.items():
        _, selected = min(options, key=lambda v: v[0])
        first = selected[0]
        expected = {
            "trajectory_id": first["trajectory_id"],
            "source_receipt_sha256": sources[first["source_id"]][
                "feedback_audit_sha256"
            ],
            "choices": [
                {"step": w["action_start"], "choice": _canonical(w["choice"])}
                for w in selected
            ],
        }
        _require(
            schedules.get(prompt) == expected and expected["choices"][0]["step"] == 0,
            "Schedule teacher selection differs",
        )
    return {
        "receipt": receipt,
        "head_zero_effect": head_meta["zero_effect"],
        "runtime": runtime,
        "schedules": schedules,
        "bundle": bundle_receipt,
        "anchors": anchors,
        "manifest": manifest,
        "corpus": corpus,
        "provenance": [inputs.provenance(), bundle.provenance()],
        "cost": {
            "native_anchor_collection": _cost(anchors),
            "frozen_feature_extraction": {
                **feature_counts,
                "windows": len(sample_ids),
                "wall_seconds": _number(
                    receipt["feature_extraction_seconds"], "feature extraction seconds"
                ),
            },
            "selector_optimization": {
                "optimizer_steps": 1000,
                "velocity_evaluations": 0,
                "wall_seconds": _number(
                    selector["wall_seconds"], "selector training seconds"
                ),
            },
            "head_optimization": {
                "optimizer_steps": 1000,
                "velocity_evaluations": 0,
                "base_backward_passes": 0,
                "wall_seconds": _number(head["wall_seconds"], "head training seconds"),
            },
        },
    }


def _canonical(value):
    result = dict(value)
    if result == {"operator": "native"}:
        return result
    if result["operator"] == "tli" and (
        result["alpha"] == 0.5 or result["source_a_id"] == result["source_b_id"]
    ):
        return {"operator": "native"}
    if result["source_a_id"] > result["source_b_id"]:
        result["source_a_id"], result["source_b_id"] = (
            result["source_b_id"],
            result["source_a_id"],
        )
        result["alpha"] = 1.0 - result["alpha"]
    result["alpha"] = float(result["alpha"])
    return _choice(result)


def _worker(directory, training, protocol):
    root = Path(directory)
    seal_receipt = _json(root / "completion_receipt.json")
    _require(
        seal_receipt["schema_version"] == "recipe-evaluation-worker-1.0"
        and seal_receipt["status"] == "complete",
        "Evaluation worker has not completed",
    )
    worker = seal_receipt["worker"]
    suite, tasks = _assignment(worker)
    _no_api(seal_receipt)
    inputs = Inputs(root, f"worker_{worker}", seal_receipt["files"])
    _require(not (root / "failure.json").exists(), "Evaluation failure artifact exists")
    for name, record in inputs.seal.items():
        _safe(root, name)
        _sha(record["sha256"])
        _integer(record["bytes"], "sealed bytes")
    runtime = inputs.read("runtime.json")
    _runtime(runtime, "evaluation", worker, training["runtime"]["payload_sha256"])
    _require(
        inputs.read("protocol.json") == protocol
        and inputs.read("training_receipt.json") == training["receipt"],
        "Worker protocol/training receipt differs",
    )
    _require(
        inputs.read("progress.json")["status"] == "complete",
        "Sealed worker progress is incomplete",
    )
    _base(inputs.read("checkpoint.json"), training["receipt"]["base_identity"])
    _weights(inputs, training["receipt"]["native_parameter_sha256"])
    _require(
        seal_receipt["native_parameter_sha256"]
        == training["receipt"]["native_parameter_sha256"]
        and seal_receipt["head_parameter_sha256"]
        == training["receipt"]["head"]["after_parameter_sha256"]
        and seal_receipt["training_bundle_sha256"] == training["bundle"]["sha256"],
        "Worker frozen parameter/bundle identities differ",
    )
    manifest = inputs.read("evaluation_reset_manifest.json")
    entries = _manifest(manifest, suite, tasks, [1, 2])
    plan = inputs.read("frozen_plan.json")
    _require(
        plan
        == {
            "protocol_sha256": digest(protocol),
            "reset_manifest_sha256": manifest["sha256"],
            "training_bundle_sha256": training["bundle"]["sha256"],
            "episode_ids": list(entries),
            "arms": list(ARMS),
            "suite": suite,
            "worker": worker,
        },
        "Frozen assignment differs",
    )
    index = inputs.read("rollouts.json")
    expected = [(episode, arm) for episode in entries for arm in ARMS]
    _require(
        [(r["episode_id"], r["arm"]) for r in index] == expected
        and seal_receipt["physical_rollouts"] == len(index)
        and seal_receipt["episodes"] == len(entries)
        and seal_receipt["suite"] == suite,
        "Worker rollout coverage/order differs",
    )
    bank_inventory = inputs.read("bank_inventory.json")
    rows = []
    native_condition_bindings = {}
    for i, row in enumerate(index):
        relative = f"rollouts/task_{row['task_id']}_state_{row['initial_state_id']}/{row['arm']}"
        _require(
            row["relative_directory"] == relative, "Physical rollout directory differs"
        )
        parsed = _rollout(
            inputs,
            relative,
            row,
            entries[row["episode_id"]],
            probe=i == 0,
            schedules=training["schedules"],
            outer=True,
            head_sha=training["receipt"]["head"]["after_parameter_sha256"],
            head_zero_effect=training["head_zero_effect"],
            native_condition_bindings=native_condition_bindings,
        )
        parsed.update(worker=worker, workflow=runtime["workflow"])
        rows.append(parsed)
    for episode in entries:
        paired = [r for r in rows if r["episode_id"] == episode]
        _require(
            all(r["reset_audit"] == paired[0]["reset_audit"] for r in paired),
            "Arms do not share exact stabilized state/model/images",
        )
        keyed = {}
        for row in paired:
            for step, checksum in row["noise_sha256_by_step"].items():
                _require(
                    step not in keyed or keyed[step] == checksum,
                    "Paired arm noise differs",
                )
                keyed[step] = checksum
    provenance = inputs.provenance()
    provenance["completion_receipt"] = {
        "sha256": file_sha256(root / "completion_receipt.json"),
        "bytes": (root / "completion_receipt.json").stat().st_size,
    }
    return {
        "worker": worker,
        "runtime": runtime,
        "rows": rows,
        "manifest_sha256": manifest["sha256"],
        "bank_inventory_sha256": digest(bank_inventory),
        "provenance": provenance,
    }


def _metrics(rows):
    successful = [r for r in rows if r["credited_success"]]
    failed = [r for r in rows if not r["credited_success"]]
    return {
        "cases": len(rows),
        "successes": len(successful),
        "success_rate": len(successful) / len(rows) if rows else None,
        "failures": len(failed),
        "failure_episode_ids": [r["episode_id"] for r in failed],
        "median_actions_to_success": statistics.median(
            r["actions_executed"] for r in successful
        )
        if successful
        else None,
        "failure_actions": sum(r["actions_executed"] for r in failed),
        "budget_censored_failures": sum(
            r["actions_executed"] == r["action_budget"] for r in failed
        ),
        "early_terminal_failures": sum(
            r["terminated"] and r["actions_executed"] < r["action_budget"]
            for r in failed
        ),
        "cost": _cost(rows),
    }


def _paired(rows, reference, method):
    by_arm = {
        arm: {r["episode_id"]: r for r in rows if r["arm"] == arm}
        for arm in (reference, method)
    }
    _require(
        set(by_arm[reference]) == set(by_arm[method]),
        "Paired comparison coverage differs",
    )
    groups = {
        "both_success": [],
        "method_only_success": [],
        "reference_only_success": [],
        "both_failure": [],
    }
    for episode in sorted(by_arm[reference]):
        a, b = (
            by_arm[reference][episode]["credited_success"],
            by_arm[method][episode]["credited_success"],
        )
        key = (
            "both_success"
            if a and b
            else "method_only_success"
            if b
            else "reference_only_success"
            if a
            else "both_failure"
        )
        groups[key].append(episode)
    counts = {key: len(value) for key, value in groups.items()}
    native_fail = counts["method_only_success"] + counts["both_failure"]
    native_win = counts["both_success"] + counts["reference_only_success"]
    return {
        "reference": reference,
        "method": method,
        "cases": len(by_arm[reference]),
        **counts,
        "gain_on_reference_failures": counts["method_only_success"] / native_fail
        if native_fail
        else None,
        "harm_on_reference_successes": counts["reference_only_success"] / native_win
        if native_win
        else None,
        "net_success_difference": counts["method_only_success"]
        - counts["reference_only_success"],
        "episode_ids": groups,
    }


def _group(rows):
    return {
        "methods": {
            arm: _metrics([r for r in rows if r["arm"] == arm]) for arm in ARMS
        },
        "paired": [_paired(rows, "native", arm) for arm in ARMS[1:]]
        + [
            _paired(rows, "recorded_schedule", "learned_selector"),
            _paired(rows, "flow_head", "gated_flow_head"),
        ],
    }


def _archive_binding(value, label, provenances, runtime):
    path = Path(value) if isinstance(value, (str, Path)) else None
    receipt = _json(path) if path is not None else copy.deepcopy(value)
    _require(
        receipt["schema_version"] == "recipe-archive-audit-1.0"
        and receipt["status"] == "passed"
        and receipt["complete"] is True,
        "Archive verification is not complete/passed",
    )
    phase = "training" if label == "training" else "evaluation"
    _require(
        receipt["phase"] == phase
        and receipt["worker"] == runtime["worker"]
        and receipt["workflow"] == runtime["workflow"],
        "Archive workflow/worker/phase differs",
    )
    checks = receipt["checks"]
    _require(
        checks["full_archive_stream_verified"] is True
        and checks["all_member_bytes_verified"] is True,
        "Full archive byte verification is required",
    )
    _require(
        checks[
            "training_bundle_seal_verified"
            if phase == "training"
            else "completion_seal_verified"
        ]
        is True,
        "Archive immutable seal verification failed",
    )
    _require(
        type(checks["array_descriptors_verified"]) is bool,
        "Archive must declare array verification scope",
    )
    _sha(receipt["archive"]["sha256"])
    _integer(receipt["archive"]["bytes"], "archive bytes", 1)
    files = receipt["files"]
    for name, record in files.items():
        _require(
            not Path(name).is_absolute() and ".." not in Path(name).parts,
            "Unsafe archive member name",
        )
        _sha(record["sha256"])
        _integer(record["bytes"], "archive member bytes")
    for source in provenances:
        prefix = "bundle/" if source["input"] == "training/bundle" else ""
        for name, record in source["files"].items():
            _require(
                files.get(prefix + name) == record,
                f"Local input differs from verified archive: {label}/{prefix}{name}",
            )
        if "completion_receipt" in source:
            _require(
                files.get("completion_receipt.json") == source["completion_receipt"],
                "Completion receipt differs from streamed archive",
            )
    inventory = receipt["member_inventory"]
    _sha(inventory["sha256"])
    _require(
        _integer(inventory["members"], "archive members", 1) >= len(files)
        and _integer(inventory["bytes"], "archive uncompressed bytes", 1)
        >= sum(r["bytes"] for r in files.values()),
        "Full archive inventory is smaller than retained inputs",
    )
    return {
        "input": label,
        "status": "passed",
        "receipt_sha256": file_sha256(path) if path else None,
        "receipt_content_sha256": digest(receipt),
        "archive": {k: receipt["archive"].get(k) for k in ("sha256", "bytes", "etag")},
        "checks": checks,
        "member_inventory": inventory,
        "retained_file_inventory_sha256": digest(files),
        "retained_file_count": len(files),
        "retained_file_bytes": sum(r["bytes"] for r in files.values()),
        "auditor_source_sha256": receipt.get("source_sha256"),
    }


def build_report(
    training_directory,
    worker_directories,
    *,
    archive_receipts=None,
    require_complete=False,
):
    """Return JSON-safe evidence; missing workers withhold all efficacy tables.

    Invalid supplied complete records always raise. Incomplete workers are
    listed without adding unsealed partial records to verified cost totals.
    """
    protocol = _json(
        Path(__file__).parent / "configs/learned_correction_recipe_v1.json"
    )
    protocol_sha = _protocol(protocol)
    training = _training(training_directory, protocol)
    workers, pending = [], []
    paths = [Path(p) for p in worker_directories]
    _require(
        len({p.resolve() for p in paths}) == len(paths), "Duplicate worker input path"
    )
    for path in paths:
        if not (path / "completion_receipt.json").exists():
            runtime = _json(path / "runtime.json")
            _runtime(
                runtime,
                "evaluation",
                runtime["worker"],
                training["runtime"]["payload_sha256"],
            )
            _assignment(runtime["worker"])
            _require(
                _json(path / "protocol.json") == protocol,
                "Incomplete worker protocol differs",
            )
            pending.append(
                {
                    "worker": runtime["worker"],
                    "workflow": runtime["workflow"],
                    "status": "failed"
                    if (path / "failure.json").exists()
                    else "unsealed",
                    "unsealed_cost_excluded": True,
                }
            )
        else:
            workers.append(_worker(path, training, protocol))
    ids = [w["worker"] for w in workers] + [w["worker"] for w in pending]
    _require(len(set(ids)) == len(ids), "Duplicate logical evaluation worker")
    workers.sort(key=lambda w: w["worker"])
    _require(
        len({w["bank_inventory_sha256"] for w in workers}) <= 1,
        "Text donor bank inventories differ across workers",
    )
    rows = [r for worker in workers for r in worker["rows"]]
    unique = {(r["episode_id"], r["arm"]) for r in rows}
    _require(len(unique) == len(rows), "Duplicate physical case/arm")
    complete_coverage = [w["worker"] for w in workers] == list(range(5))
    archive_receipts = archive_receipts or {}
    _require(
        set(archive_receipts) <= {"training", *(f"worker_{i}" for i in range(5))},
        "Unexpected archive receipt assignment",
    )
    bindings = []
    if "training" in archive_receipts:
        bindings.append(
            _archive_binding(
                archive_receipts["training"],
                "training",
                training["provenance"],
                training["runtime"],
            )
        )
    for worker in workers:
        label = f"worker_{worker['worker']}"
        if label in archive_receipts:
            bindings.append(
                _archive_binding(
                    archive_receipts[label],
                    label,
                    [worker["provenance"]],
                    worker["runtime"],
                )
            )
    complete = complete_coverage and len(bindings) == 6
    _require(
        not require_complete or complete,
        "All five completed evaluation workers and six passed archive receipts are required",
    )
    if complete:
        _require(
            len(rows) == 240 and len({r["episode_id"] for r in rows}) == 48,
            "Final evaluation denominator differs",
        )
    for row in rows:
        row["cohort"] = "id_panel" if row["suite"] == "libero_10" else "ood"
        row["correction_teacher_task"] = row["instruction"] in training["schedules"]
    # No full-run claims from subset-selected successes or worker completion order.
    groups, per_task = {}, []
    if complete:
        for cohort in ("ood", "id_panel"):
            groups[cohort] = _group([r for r in rows if r["cohort"] == cohort])
        for suite in SUITES:
            groups[suite] = _group([r for r in rows if r["suite"] == suite])
        for covered in (True, False):
            subset = [
                r
                for r in rows
                if r["cohort"] == "ood" and r["correction_teacher_task"] == covered
            ]
            _require(
                len(subset) == (80 if covered else 120),
                "Teacher composition stratum differs",
            )
            groups["ood_teacher_tasks" if covered else "ood_other_tasks"] = _group(
                subset
            )
        for suite in SUITES:
            for task in range(4 if suite == "libero_10" else 10):
                subset = [
                    r for r in rows if r["suite"] == suite and r["task_id"] == task
                ]
                _require(
                    len({r["instruction"] for r in subset}) == 1 and len(subset) == 10,
                    "Per-task instruction/coverage differs",
                )
                per_task.append(
                    {
                        "suite": suite,
                        "task_id": task,
                        "instruction": subset[0]["instruction"],
                        "correction_teacher_task": subset[0]["correction_teacher_task"],
                        **_group(subset),
                    }
                )
    evaluation_cost = _cost(rows)
    training_cost = training["cost"]
    stages_total_vf = (
        training_cost["native_anchor_collection"]["total_velocity_evaluations"]
        + training_cost["frozen_feature_extraction"]["velocity_evaluations"]
        + evaluation_cost["total_velocity_evaluations"]
    )
    return {
        "schema_version": SCHEMA,
        "status": "complete" if complete else "incomplete",
        "efficacy_released": complete,
        "protocol": protocol,
        "protocol_sha256": protocol_sha,
        "coverage": {
            "expected_workers": 5,
            "completed_workers": [w["worker"] for w in workers],
            "pending_workers": pending,
            "missing_workers": sorted(set(range(5)) - set(ids)),
            "expected_cases": 48,
            "expected_physical_rollouts": 240,
            "verified_cases": len({r["episode_id"] for r in rows}),
            "verified_physical_rollouts": len(rows),
            "verified_archives": len(bindings),
            "ood_cases_per_arm": 40,
            "id_panel_cases_per_arm": 8,
            "id_tasks": "4 of 10 standard LIBERO-10 tasks",
        },
        "identities": {
            "payload_sha256": training["runtime"]["payload_sha256"],
            "training_workflow": training["runtime"]["workflow"],
            "training_receipt_sha256": file_sha256(
                Path(training_directory) / "training_receipt.json"
            ),
            "training_bundle_sha256": training["bundle"]["sha256"],
            "native_parameter_sha256": training["receipt"]["native_parameter_sha256"],
            "base_identity": training["receipt"]["base_identity"],
            "head_parameter_sha256": training["receipt"]["head"][
                "after_parameter_sha256"
            ],
            "workers": [
                {
                    "worker": w["worker"],
                    "workflow": w["runtime"]["workflow"],
                    "reset_manifest_sha256": w["manifest_sha256"],
                    "bank_inventory_sha256": w["bank_inventory_sha256"],
                }
                for w in workers
            ],
        },
        "training": {
            "historical_trajectories": 19,
            "historical_windows": 413,
            "historical_actions": training["corpus"]["executed_action_count"],
            "correction_trajectories": 12,
            "corrected_compositions": 8,
            "fresh_anchor_outcomes": training["anchors"],
            "training_windows": training["receipt"]["training_windows"],
            "selector": training["receipt"]["selector"],
            "head": training["receipt"]["head"],
            "recorded_schedules": training["schedules"],
            "historical_corpus_sha256": training["corpus"]["corpus_id"],
        },
        "cost": {
            "training": training_cost,
            "evaluation": evaluation_cost,
            "evaluation_is_lower_bound": not complete,
            "all_new_velocity_evaluations": stages_total_vf,
            "all_new_prefix_evaluations": training_cost["native_anchor_collection"][
                "total_prefix_evaluations"
            ]
            + training_cost["frozen_feature_extraction"]["prefix_evaluations"]
            + evaluation_cost["total_prefix_evaluations"],
            "new_provider_calls": 0,
            "new_provider_tokens": 0,
            "historical_teacher_acquisition": {
                "included_in_new_cost": False,
                "total_tokens": None,
                "scope": "Separate historical acquisition, validation and unsuccessful trials; no zero-cost assumption.",
            },
            "time_semantics": "Rollout wall includes reset/policy/environment/video; training extraction and two optimizer stages are disjoint. Parallel rollout wall sums are not elapsed makespan.",
        },
        "groups": groups,
        "per_task": per_task,
        "rows": rows,
        "validation": {
            "metadata_event_noise_reset_checks_passed": True,
            "exact_full_coverage": complete_coverage,
            "all_six_archives_verified": len(bindings) == 6,
            "native_weights_unchanged": True,
            "native_condition_descriptor_binding_checks": sum(
                r["native_condition_descriptor_binding_checks"] for r in rows
            ),
            "matched_prior_native_observation_descriptors": sum(
                r["matched_prior_native_observation_descriptors"] for r in rows
            ),
            "array_contents_replayed": False,
            "all_recorded_array_descriptors_verified_by_archive_auditor": bool(bindings)
            and len(bindings) == 6
            and all(b["checks"]["array_descriptors_verified"] for b in bindings),
            "model_or_physics_replayed": False,
            "new_provider_calls_observed": 0,
            "initial_success_count": sum(r["initial_success"] for r in rows),
            "zero_action_success_count": sum(r["zero_action_success"] for r in rows),
        },
        "input_provenance": training["provenance"] + [w["provenance"] for w in workers],
        "archive_verification": bindings,
        "reporter_sha256": file_sha256(__file__),
        "limitations": LIMITATIONS,
    }


def _table(report):
    rows = []
    for cohort in ("ood", "id_panel"):
        if cohort not in report["groups"]:
            continue
        group = report["groups"][cohort]
        for arm in ARMS:
            metric = group["methods"][arm]
            paired = next(
                (
                    p
                    for p in group["paired"]
                    if p["reference"] == "native" and p["method"] == arm
                ),
                None,
            )
            rows.append(
                {
                    "cohort": cohort,
                    "arm": arm,
                    "successes": metric["successes"],
                    "cases": metric["cases"],
                    "success_rate": metric["success_rate"],
                    "gain_vs_native": paired["method_only_success"] if paired else None,
                    "harm_vs_native": paired["reference_only_success"]
                    if paired
                    else None,
                    "censored_failures": metric["budget_censored_failures"],
                    "early_terminal_failures": metric["early_terminal_failures"],
                    "median_actions_success_only": metric["median_actions_to_success"],
                    "actions": metric["cost"]["actions_executed"],
                    "velocity_evaluations": metric["cost"][
                        "total_velocity_evaluations"
                    ],
                    "provider_tokens": 0,
                }
            )
    return rows


def _markdown(report):
    lines = [
        "# Learned correction recipe",
        "",
        f"Status: **{report['status']}**. Seed 61; new resets of known compositions. No Astra calls during training or evaluation.",
        "",
    ]
    if report["efficacy_released"]:
        lines += [
            "All 240 physical trials passed the recorded-data checks: 40 OOD cases and eight standard-panel cases per arm. Every arm was evaluated independently on every reset.",
            "",
            "| Cohort | Method | Success | Gains / harm vs native | Capped failures | Median actions, successes only |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for row in _table(report):
            paired = (
                "—"
                if row["gain_vs_native"] is None
                else f"{row['gain_vs_native']} / {row['harm_vs_native']}"
            )
            median = (
                "—"
                if row["median_actions_success_only"] is None
                else str(row["median_actions_success_only"])
            )
            lines.append(
                f"| {row['cohort']} | {row['arm']} | {row['successes']}/{row['cases']} | {paired} | {row['censored_failures']} | {median} |"
            )
    else:
        lines += [
            f"Efficacy tables are withheld: {report['coverage']['verified_physical_rollouts']}/240 verified trials. Unsealed partial work is not included in verified cost totals."
        ]
    cost = report["cost"]
    lines += [
        "",
        "## Physical cost",
        "",
        f"Verified evaluation: {cost['evaluation']['physical_rollouts']} rollouts; {cost['evaluation']['actions_executed']:,} actions; {cost['evaluation']['total_velocity_evaluations']:,} velocity evaluations including parity probes. These totals are {'complete' if report['efficacy_released'] else 'lower bounds'}.",
        "",
        f"Training: four native anchor collection rollouts; {report['training']['training_windows']} frozen feature windows; {cost['training']['frozen_feature_extraction']['velocity_evaluations']:,} feature-extraction velocity evaluations; 1,000 selector updates and 1,000 head updates. New provider calls/tokens: 0/0. Historical teacher acquisition is separate and was not free.",
        "",
        "## Interpretation and audit scope",
        "",
    ]
    lines += [f"- {item}" for item in report["limitations"]]
    lines += [
        "",
        "[Machine report](report.json) · [Method table](metrics.csv) · [Every physical trial](rollouts.csv) · [Per-task paired comparisons](per_task.json) · [Input hash inventory](input_provenance.json) · [Protocol](protocol.json) · [Reporter source](source/recipe_report.py.txt) · [HTML](index.html)",
        "",
    ]
    return "\n".join(lines)


def _csv(path, rows, fields):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _figures(report, output):
    if not report["efficacy_released"]:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination = output / "figures"
    destination.mkdir()
    values = {
        "schema_version": "recipe-plotted-values-1.0",
        "report_sha256": file_sha256(output / "report.json"),
        "metrics": _table(report),
        "selector_loss": report["training"]["selector"]["loss_trace"],
        "head_diagnostics": {
            key: report["training"]["head"][key] for key in ("before", "after")
        },
        "protocol_sha256": report["protocol_sha256"],
    }
    _write(destination / "values.json", values)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    colors = ["#506779", "#a98144", "#3c8a89", "#6e67a7", "#b76876"]
    for axis, cohort in zip(axes, ("ood", "id_panel"), strict=True):
        rows = [r for r in values["metrics"] if r["cohort"] == cohort]
        axis.bar(range(5), [100 * r["success_rate"] for r in rows], color=colors)
        axis.set(
            ylim=(0, 105),
            ylabel="Success (%)",
            title=f"{'OOD known compositions' if cohort == 'ood' else 'ID retention: 4/10 tasks'} · n={rows[0]['cases']} per arm",
        )
        axis.set_xticks(range(5), [a.replace("_", "\n") for a in ARMS], fontsize=8)
        for i, row in enumerate(rows):
            axis.text(
                i,
                100 * row["success_rate"] + 2,
                f"{row['successes']}/{row['cases']}",
                ha="center",
                fontsize=9,
            )
    fig.suptitle(
        "Seed 61 · matched reset/noise · unconditional evaluation · 0 new API calls"
    )
    for ext in ("png", "pdf"):
        fig.savefig(destination / f"outcomes.{ext}", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for key in ("gate_bce", "pair_ce", "alpha_mse"):
        axes[0].plot(
            [r["step"] for r in values["selector_loss"]],
            [r[key] for r in values["selector_loss"]],
            label=key,
        )
    axes[0].set(
        xlabel="Recorded optimizer update",
        ylabel="Sampled training objective",
        title="Selector training, not evaluation success",
    )
    axes[0].legend(fontsize=8)
    for key in ("masked_residual_mse", "anchor_mse"):
        axes[1].plot(
            [0, 1000],
            [values["head_diagnostics"][stage][key] for stage in ("before", "after")],
            marker="o",
            label=key,
        )
    axes[1].set(
        xlabel="Optimizer update (endpoints only)",
        ylabel="Recorded training diagnostic",
        title="Flow head: two measured endpoints",
    )
    axes[1].legend(fontsize=8)
    for ext in ("png", "pdf"):
        fig.savefig(destination / f"training.{ext}", dpi=160)
    plt.close(fig)
    return {
        "values_sha256": file_sha256(destination / "values.json"),
        "source_sha256": file_sha256(__file__),
        "files": [str(p.relative_to(output)) for p in sorted(destination.iterdir())],
    }


def _videos(report, worker_directories, output, *, maximum_bytes=40 * 1024**2):
    """Select discordants lexicographically after outcomes; never an efficacy sample."""
    if not report["efficacy_released"]:
        return {"selection": "No gallery before complete coverage", "videos": []}
    roots = {
        _json(Path(p) / "runtime.json")["worker"]: Path(p)
        for p in worker_directories
        if (Path(p) / "completion_receipt.json").exists()
    }
    by_key = {(r["episode_id"], r["arm"]): r for r in report["rows"]}
    chosen = {}
    for cohort in ("ood", "id_panel"):
        candidates = sorted(
            (
                r
                for r in report["rows"]
                if r["cohort"] == cohort and r["arm"] == "native"
            ),
            key=lambda r: r["episode_id"],
        )
        for success in (True, False):
            row = next(
                (r for r in candidates if r["credited_success"] == success), None
            )
            if row:
                chosen[(row["episode_id"], row["arm"])] = (
                    f"First lexicographic {cohort} native {'success' if success else 'failure'}"
                )
    for pair in report["groups"]["ood"]["paired"][:4]:
        for category in ("method_only_success", "reference_only_success"):
            if pair["episode_ids"][category]:
                episode = pair["episode_ids"][category][0]
                for arm in ("native", pair["method"]):
                    chosen[(episode, arm)] = (
                        f"Outcome-selected {pair['method']} vs native: {category}; first lexicographic discordant reset"
                    )
    result, used = [], 0
    for key, reason in chosen.items():
        row = by_key[key]
        path = _safe(roots[row["worker"]], row["relative_directory"] + "/rollout.mp4")
        if not path.exists():
            result.append(
                {
                    "episode_id": key[0],
                    "arm": key[1],
                    "available": False,
                    "source_sha256": row["video_sha256"],
                    "selection": reason,
                }
            )
            continue
        _require(
            file_sha256(path) == row["video_sha256"],
            "Selected video differs from sealed rollout",
        )
        if used + path.stat().st_size > maximum_bytes:
            result.append(
                {
                    "episode_id": key[0],
                    "arm": key[1],
                    "available": False,
                    "source_sha256": row["video_sha256"],
                    "selection": reason,
                    "omitted_reason": "gallery byte cap",
                }
            )
            continue
        _require(
            shutil.which("ffprobe") is not None,
            "ffprobe is required to verify selected real videos",
        )
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_frames",
                "-show_entries",
                "stream=width,height,r_frame_rate,nb_read_frames",
                "-of",
                "json",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        stream = json.loads(probe.stdout)["streams"][0]
        _require(
            stream["r_frame_rate"] == "20/1"
            and int(stream["nb_read_frames"]) == row["actions_executed"]
            and stream["width"] == stream["height"] == 224,
            "Video frame timing/count differs",
        )
        target = (
            output
            / "videos"
            / f"{row['suite']}_task{row['task_id']}_state{row['initial_state_id']}_{row['arm']}.mp4"
        )
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(path, target)
        used += target.stat().st_size
        result.append(
            {
                "episode_id": key[0],
                "arm": key[1],
                "available": True,
                "path": str(target.relative_to(output)),
                "sha256": row["video_sha256"],
                "bytes": target.stat().st_size,
                "selection": reason,
                "success": row["credited_success"],
                "actions": row["actions_executed"],
                "fps": 20,
                "speed": "original recorded action time, inference pauses omitted",
                "workflow": row["workflow"],
                "summary_sha256": row["summary_sha256"],
            }
        )
    return {
        "selection": "Outcome-selected illustrations, not a representative sample or an extra evaluation cohort; exact source video bytes.",
        "bytes": used,
        "videos": result,
    }


def _html(report, gallery, plots):
    esc = html.escape
    body = [
        "<h1>Learned correction recipe</h1>",
        f"<p class='status'>{esc(report['status'])} · Seed 61 · 0 new provider calls</p>",
        "<p>Offline distillation on known compositions, followed by unconditional matched evaluation. The gate learns intervention support, not failure probability.</p>",
    ]
    body.append(
        "<svg viewBox='0 0 900 175' role='img' aria-label='Training and evaluation diagram'><defs><marker id='arrow' markerWidth='8' markerHeight='8' refX='7' refY='4' orient='auto'><path d='M0 0L8 4L0 8' fill='#476977'/></marker></defs><g fill='#edf5f5' stroke='#476977'><rect x='10' y='20' width='255' height='60' rx='8'/><rect x='320' y='20' width='255' height='60' rx='8'/><rect x='630' y='20' width='255' height='60' rx='8'/></g><g fill='#19313d' text-anchor='middle' font-size='15'><text x='137' y='45'>Historical teachers + native anchors</text><text x='137' y='65'>Actual executed windows only</text><text x='447' y='45'>Frozen native features</text><text x='447' y='65'>Selector + bounded output head</text><text x='757' y='45'>48 new-reset cases × 5 arms</text><text x='757' y='65'>Matched noise · Astra disabled</text></g><g stroke='#476977' marker-end='url(#arrow)'><path d='M266 50H315'/><path d='M576 50H625'/></g><text x='450' y='120' text-anchor='middle' fill='#19313d'>Selector: TEI/TLI choice. Gated head: original native conditioning + residual only.</text><text x='450' y='145' text-anchor='middle' fill='#19313d'>Known-task transfer and limited ID retention; no evaluation-driven training or selection.</text></svg>"
    )
    body.append("<h2>Outcomes</h2>")
    if not report["efficacy_released"]:
        body.append(
            f"<p>Withheld until all five workers pass: {report['coverage']['verified_physical_rollouts']}/240 verified trials. No success rate is released.</p>"
        )
    else:
        body.append(
            "<table><tr><th>Cohort</th><th>Method</th><th>Success</th><th>Gain / harm vs native</th><th>Capped failures</th><th>Actions to success, median</th></tr>"
        )
        for r in _table(report):
            pair = (
                "—"
                if r["gain_vs_native"] is None
                else f"{r['gain_vs_native']} / {r['harm_vs_native']}"
            )
            median = (
                "—"
                if r["median_actions_success_only"] is None
                else str(r["median_actions_success_only"])
            )
            body.append(
                f"<tr><td>{esc(r['cohort'])}</td><td>{esc(r['arm'])}</td><td>{r['successes']}/{r['cases']}</td><td>{pair}</td><td>{r['censored_failures']}</td><td>{median}</td></tr>"
            )
        body.append(
            "</table><p>Gains are method-only successes; harm is native-only success on the same reset. Failed trials remain in every denominator. Medians condition on success and exclude censored failures.</p>"
        )
    cost = report["cost"]
    body.append(
        f"<h2>Measured cost and training</h2><p>Verified evaluation: {cost['evaluation']['actions_executed']:,} actions and {cost['evaluation']['total_velocity_evaluations']:,} velocity evaluations including probes. Training feature extraction: {cost['training']['frozen_feature_extraction']['velocity_evaluations']:,} velocity evaluations; 1,000 updates per learned module. New API usage is 0 calls / 0 tokens; historical teacher acquisition is excluded, not free.</p>"
    )
    body.append(
        "<table><tr><th>Physical stage</th><th>Rollouts / windows</th><th>Native velocity evaluations</th><th>Recorded wall seconds</th></tr>"
    )
    for label, key in (
        ("New native anchor collection (all outcomes)", "native_anchor_collection"),
        ("Frozen feature extraction", "frozen_feature_extraction"),
        ("Selector optimization", "selector_optimization"),
        ("Head optimization", "head_optimization"),
    ):
        stage = cost["training"][key]
        vf = stage.get("total_velocity_evaluations", stage["velocity_evaluations"])
        count = stage.get("physical_rollouts", stage.get("windows", "—"))
        body.append(
            f"<tr><td>{label}</td><td>{count}</td><td>{vf:,}</td><td>{stage['wall_seconds']:.2f}</td></tr>"
        )
    body.append(
        f"<tr><td>Evaluation: all arms, counted once</td><td>{cost['evaluation']['physical_rollouts']}</td><td>{cost['evaluation']['total_velocity_evaluations']:,}</td><td>{cost['evaluation']['wall_seconds']:.2f} (sum across workers)</td></tr></table><p>Rollout wall time already contains policy, reset and simulator time. Parallel sums are not makespan. Feature extraction includes recorded orchestration and artifact work; optimization times are separate measured stages.</p>"
    )
    if report["efficacy_released"]:
        body.append(
            "<h2>Every task and both evaluation resets</h2><p>Each cell counts successes out of two unconditional reset trials. Gains/harm and exact discordant episode IDs are retained in the per-task JSON; eight OOD compositions had correction teachers.</p><table><tr><th>Task</th><th>Teacher composition</th>"
            + "".join(f"<th>{esc(arm)}</th>" for arm in ARMS)
            + "</tr>"
        )
        for task in report["per_task"]:
            body.append(
                f"<tr><td>{esc(task['suite'])} · {task['task_id']}<br>{esc(task['instruction'])}</td><td>{'yes' if task['correction_teacher_task'] else 'no'}</td>"
                + "".join(
                    f"<td>{task['methods'][arm]['successes']}/2</td>" for arm in ARMS
                )
                + "</tr>"
            )
        body.append("</table>")
    if plots:
        for name in ("outcomes", "training"):
            body.append(
                f"<figure><img src='figures/{name}.png' alt='{name} measurements'><figcaption><a href='figures/{name}.pdf'>PDF</a> · <a href='figures/{name}.png'>PNG</a> · <a href='figures/values.json'>Exact plotted values and report hash</a></figcaption></figure>"
            )
    body.append(
        "<h2>Recorded rollout videos</h2><p>Outcome-selected illustrations. Original 20 fps, one pre-action frame per executed action, without inference pauses or the terminal post-action frame.</p><div class='videos'>"
    )
    for clip in gallery["videos"]:
        if not clip["available"]:
            continue
        body.append(
            f"<figure><video controls preload='metadata' src='{esc(clip['path'], quote=True)}'></video><figcaption>{esc(clip['episode_id'])}<br>{esc(clip['arm'])} · {'success' if clip['success'] else 'failure'} · {clip['actions']} actions<br>{esc(clip['selection'])}</figcaption></figure>"
        )
    body.append(
        "</div><h2>Scope and limitations</h2><ul>"
        + "".join(f"<li>{esc(v)}</li>" for v in report["limitations"])
        + "</ul>"
    )
    links = {
        "report.json": "Full machine report",
        "report.md": "Markdown",
        "metrics.csv": "Method table CSV",
        "rollouts.csv": "All physical trials CSV",
        "per_task.json": "Per-task results and discordant IDs",
        "input_provenance.json": "Input hashes",
        "protocol.json": "Frozen protocol",
        "gallery.json": "Video provenance",
        "source/recipe_report.py.txt": "Exact report source",
        "manifest.json": "Output hashes",
    }
    body.append(
        "<h2>Downloads</h2><ul>"
        + "".join(
            f"<li><a href='{path}'>{label}</a></li>" for path, label in links.items()
        )
        + "</ul>"
    )
    return (
        "<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Learned correction recipe</title><style>body{font:16px/1.6 system-ui,sans-serif;max-width:1120px;margin:32px auto;padding:0 24px;color:#19313d;background:#fbfcfc}h1,h2{line-height:1.2}h2{margin-top:2.5em}.status{font-weight:700}a{color:#176573}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:9px;border-bottom:1px solid #cbd6da;text-align:left}figure{margin:20px 0}img{max-width:100%}svg{width:100%}.videos{display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:20px}video{width:100%;max-width:420px}figcaption{font-size:13px;color:#425d68}li{margin:8px 0}</style><body>"
        + "\n".join(body)
        + "</body></html>\n"
    )


def write_report(report, output, *, worker_directories=(), include_videos=True):
    """Publish into a new directory; original records and weights stay private."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    _write(output / "report.json", report)
    _write(output / "protocol.json", report["protocol"])
    _write(output / "input_provenance.json", report["input_provenance"])
    _write(output / "per_task.json", report["per_task"])
    (output / "report.md").write_text(_markdown(report))
    metric_fields = [
        "cohort",
        "arm",
        "successes",
        "cases",
        "success_rate",
        "gain_vs_native",
        "harm_vs_native",
        "censored_failures",
        "early_terminal_failures",
        "median_actions_success_only",
        "actions",
        "velocity_evaluations",
        "provider_tokens",
    ]
    _csv(output / "metrics.csv", _table(report), metric_fields)
    _csv(
        output / "rollouts.csv",
        report["rows"],
        [
            "episode_id",
            "suite",
            "task_id",
            "initial_state_id",
            "seed",
            "instruction",
            "arm",
            "success",
            "credited_success",
            "actions_executed",
            "action_budget",
            "terminated",
            "policy_replans",
            "velocity_evaluations",
            "probe_velocity_evaluations",
            "prefix_evaluations",
            "probe_prefix_evaluations",
            "head_active_actions",
            "gate_active_actions",
            "wall_seconds",
            "worker",
            "workflow",
            "reset_audit_sha256",
            "summary_sha256",
        ],
    )
    (output / "source").mkdir()
    shutil.copyfile(__file__, output / "source/recipe_report.py.txt")
    plots = _figures(report, output)
    gallery = (
        _videos(report, worker_directories, output)
        if include_videos
        else {"selection": "Video export disabled explicitly", "videos": []}
    )
    _write(output / "gallery.json", gallery)
    (output / "index.html").write_text(_html(report, gallery, plots))
    manifest = {
        "schema_version": "recipe-report-publication-1.0",
        "report_sha256": file_sha256(output / "report.json"),
        "source_sha256": file_sha256(__file__),
        "figures": plots,
        "files": {
            str(p.relative_to(output)): {
                "sha256": file_sha256(p),
                "bytes": p.stat().st_size,
            }
            for p in sorted(output.rglob("*"))
            if p.is_file()
        },
    }
    _write(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--workers", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument(
        "--archive-receipts",
        type=Path,
        help="JSON mapping training/worker_0..4 to receipt paths; relative paths resolve beside this mapping",
    )
    parser.add_argument("--no-videos", action="store_true")
    args = parser.parse_args()
    _require(not args.output.exists(), "Report output already exists")
    receipts = {}
    if args.archive_receipts:
        mapping = _json(args.archive_receipts)
        receipts = {
            key: (
                Path(value)
                if Path(value).is_absolute()
                else args.archive_receipts.parent / value
            )
            for key, value in mapping.items()
        }
    report = build_report(
        args.training,
        args.workers,
        archive_receipts=receipts,
        require_complete=args.require_complete,
    )
    manifest = write_report(
        report,
        args.output,
        worker_directories=args.workers,
        include_videos=not args.no_videos,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "verified_rollouts": report["coverage"]["verified_physical_rollouts"],
                "report_sha256": manifest["report_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
