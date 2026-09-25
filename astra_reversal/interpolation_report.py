"""Audit and summarize completed phase-interpolation workers without model calls."""

import argparse
import base64
import csv
import io
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from .config import BenchmarkConfig
from .interpolation_agent import (
    PROMPT_TEMPLATE_VERSION,
    SCHEMA_VERSION,
    _validate_request,
    parse_proposal,
    summarize_calls,
)
from .interpolation_catalog import (
    DATASET_REVISION,
    donor_catalog,
    donor_for,
    oracle_for,
    paper_alpha,
)
from .interpolation_search import (
    ARMS,
    aggregate_reports,
    arm_summary,
    load_protocol,
    outcome_feedback,
)
from .intervention_report import (
    Inputs,
    _attempt,
    _number,
    _random_noise_check,
    _sha,
    require,
)
from .intervention_rollout import load_reset_manifest
from .interventions import noise_basis, random_noise_proposal
from .osmo.interpolation import assignment
from .records import digest, file_sha256

CONTROLS = ("known_noise", "policy_fresh")
SUITES = ("libero_goal_ood", "libero_spatial_ood")
IDENTITY = (
    "schema_version",
    "episode_id",
    "attempt_id",
    "decision_index",
    "observation_step",
    "interpolation_mode",
    "request_fingerprint",
)
ACTION_FLAGS = {
    "text_has_effect": "actions_with_nonzero_text",
    "vision_has_effect": "actions_with_changed_vision",
    "held_text_after_failed_call": "actions_with_held_text_after_failed_call",
    "native_condition_fallback": "native_condition_fallback_actions",
}
LIMITATIONS = [
    "Exploratory known-task follow-up: evaluation uses 20 previously observed OOD compositions with new seed 29 resets; development uses 3 previously inspected seed 19 cases. This is not a held-out-task or zero-shot generalization claim.",
    "Oracle source pairs/schedules use privileged manual mappings; Astra receives only the restricted nine-donor prompt library, raw observations, robot proprioception and its own feedback.",
    "This is an explicit PI05 paper-form port, including instruction-only token alignment and all-timestep demonstration means; it is not a reproduction of the released PI0 implementation.",
    "Budgets mean up to full rollouts including a common baseline: oracle stops at 2; random/Astra stop at 3. Early success reduces actual rollouts. Within-rollout decisions are a separate count.",
    "Baseline successes cost zero intervention calls. Rescue-only medians condition on baseline failure followed by a later success; censored cases remain in all-case success denominators.",
    "Provider-reported reasoning tokens are a subset of output tokens. Missing usage remains unknown; partial sums are not complete cost. Preflight failures consume decision slots but make no provider call. No dollar price is assumed.",
    "Physical case totals count the shared baseline and initialization/gates once. Standalone arm attribution repeats that common setup; its totals must not be summed across arms.",
    "One-time donor extraction is separate from case compute. Velocity evaluations count action-flow calls, not prefix-encoder forwards. Rollout wall times include baseline initialization/gates, recording and synchronization.",
    "This audit verifies frozen manifests, recorded execution, provider bindings and feedback image hashes. It does not replay simulator physics, recompute native hidden states, or verify every referenced NPY tensor.",
    "Development forces an extra rollout per arm even after baseline success. Those physical calls/actions are reported separately and do not establish intervention efficacy.",
]


def _median(values):
    return statistics.median(values) if values else None


def _operator(mode):
    return mode.removeprefix("astra_").removeprefix("oracle_").removesuffix("_vision")


def _validate_inventory(inventory):
    require(
        inventory["schema_version"] == "phase-bank-inventory-1.0"
        and isinstance(inventory["workflow"], str)
        and bool(inventory["workflow"])
        and isinstance(inventory["source_revision"], str)
        and bool(inventory["source_revision"])
        and _sha(inventory["payload_sha256"])
        and set(inventory["donors"]) == {row["source_id"] for row in donor_catalog()},
        "Invalid frozen nine-donor bank inventory",
    )
    for source_id, row in inventory["donors"].items():
        donor = donor_for(source_id)
        bank = row["bank"]
        require(
            bank["bank_id"]
            == digest({key: value for key, value in bank.items() if key != "bank_id"}),
            "Bank metadata identity is not internally consistent",
        )
        provenance = bank["provenance"]
        require(
            provenance["source_id"] == source_id
            and provenance["source_prompt"] == donor.prompt
            and provenance["dataset_revision"] == DATASET_REVISION
            and provenance["frame_count"] == donor.all_frame_count
            and [
                (ep["episode_index"], ep["frame_count"])
                for ep in provenance["episodes"]
            ]
            == list(
                zip(donor.episode_indices, donor.episode_frame_counts, strict=True)
            ),
            "Bank metadata does not describe the prescribed standard demonstrations",
        )
        require(
            type(row["worker"]) is int and row["worker"] in range(8),
            "Invalid bank extraction worker",
        )
        files = row["files"]
        require(
            len(files) == 3
            and {file["name"] for file in files}
            == {"bank.npz", "manifest.json", "frames.jsonl"},
            "Bank inventory file coverage differs",
        )
        for file in files:
            require(
                _sha(file["sha256"])
                and type(file["bytes"]) is int
                and file["bytes"] > 0
                and isinstance(file["key"], str)
                and bool(file["key"]),
                "Invalid bank artifact receipt",
            )


def _physical(report):
    return [report["baseline"], *report["controls"].values()] + [
        row for arm in ARMS for row in report["arms"][arm]["attempts"][1:]
    ]


def _physical_cost(rows, initialization, gate):
    rollout_vf = sum(row["velocity_evaluations"] for row in rows)
    return {
        "rollouts": len(rows),
        "simulated_actions": sum(row["actions_executed"] for row in rows),
        "rollout_velocity_evaluations": rollout_vf,
        "initialization_velocity_evaluations": initialization,
        "interpolation_gate_velocity_evaluations": gate,
        "velocity_evaluations": rollout_vf + initialization + gate,
        "rollout_wall_seconds": sum(row["wall_seconds"] for row in rows),
        "token_usage": summarize_calls(
            [record for row in rows for record in row["provider_records"]]
        ),
    }


def _validate_gates(report, protocol):
    initial = report["initialization"]
    errors = initial["errors"]
    for error in errors.values():
        for key in ("max_abs", "rmse"):
            _number(error[key], f"initialization {key}")
    require(
        initial["passed"] is True
        and errors["noise"]["max_abs"] <= protocol["numerical_gate"]["noise_max_abs"]
        and errors["actions"]["max_abs"] <= protocol["numerical_gate"]["action_max_abs"]
        and errors["native_parity"]["max_abs"] <= 1e-5
        and errors["zero_embedding_hook_parity"]["max_abs"] == 0
        and initial["velocity_evaluations"] == 1230
        and initial.get("development_embedding_probe") is None,
        "Initialization failed the fixed native/inversion gate",
    )
    for name in ("condition_id", "known_noise_sha256", "recovered_noise_sha256"):
        require(_sha(initial[name]), "Invalid initialization digest")
    gate = report["interpolation_gate"]
    require(
        set(gate["checks"])
        == {
            "tei_identical_sources",
            "tli_zero_residual",
            "tei_nonzero_oracle_sources",
            "tli_nonzero_oracle_banks",
        },
        "Interpolation gate has missing or undeclared checks",
    )
    require(
        gate["passed"]
        is gate["complete"]
        is gate["velocity_evaluations_complete"]
        is True
        and gate["status"] == "passed"
        and gate["velocity_evaluations"] == 50
        and gate["reference_velocity_evaluations"] == 10
        and gate["solver"] == protocol["execution_solver"]
        and gate["known_noise_sha256"] == initial["known_noise_sha256"]
        and gate["probe_kind"]
        == "fixed_weighted_numerical_probe_without_environment_actions",
        "Interpolation gate is incomplete or has wrong physical cost",
    )
    for name, operator, alpha, nonzero in (
        ("tei_identical_sources", "tei", 0.37, False),
        ("tli_zero_residual", "tli", 0.5, False),
        ("tei_nonzero_oracle_sources", "tei", 0, True),
        ("tli_nonzero_oracle_banks", "tli", 0, True),
    ):
        row = gate["checks"][name]
        require(
            row["passed"] is True
            and row["status"] == "passed"
            and row["operator"] == operator
            and row["alpha"] == alpha
            and row["velocity_evaluations"] == 10
            and row["provenance"]["has_effect"] == nonzero,
            "Interpolation identity/nonzero probe did not pass",
        )
        for field in ("errors", "controlled_channel_errors", "decoded_action_errors"):
            for metric in ("max_abs", "rmse"):
                _number(row[field][metric], f"gate {name} {field} {metric}")
        if nonzero:
            require(
                row["controlled_channel_errors"]["max_abs"] > 0
                and row["decoded_action_errors"]["max_abs"] > 0
                and row["source_ids"]
                == [report["oracle"]["source_a_id"], report["oracle"]["source_b_id"]],
                "Nonzero probe lacks a measured controlled action effect",
            )
        else:
            require(
                row["errors"]["max_abs"] == row["decoded_action_errors"]["max_abs"] == 0
                and row["condition_id"] == gate["native_condition_id"]
                and row["source_ids"] is None,
                "Native interpolation identity check failed",
            )


def _ledger(inputs, directory, row, protocol):
    path = directory / f"{row['attempt_id']}_provider.jsonl"
    records = list(inputs.lines(path)) if path.exists() else []
    if not path.exists():
        inputs.absent.add(str(path.resolve()))
    require(
        records == row["provider_records"],
        "Provider sidecar differs from recorded attempt",
    )
    decisions = row["decisions"]
    if not row["mode"].startswith("astra_"):
        require(
            not records and not decisions,
            "Control/oracle/random arm has provider decisions",
        )
    else:
        require(
            len(records) == len(decisions) == math.ceil(row["actions_executed"] / 25),
            "Missing, extra or unscheduled provider decision",
        )
    summarize_calls(records)  # Validate usage against raw provider-reported fields.
    for index, (record, decision) in enumerate(zip(records, decisions, strict=True), 1):
        require(
            record["client_schema_version"] == SCHEMA_VERSION
            and record["prompt_template_version"] == PROMPT_TEMPLATE_VERSION
            and record["requested_model"] == protocol["astra"]["model"]
            and record["attempt_id"] == row["attempt_id"]
            and record["decision_index"] == decision["decision_index"] == index
            and record["observation_step"]
            == decision["observation_step"]
            == (index - 1) * 25
            and record["interpolation_mode"] == _operator(row["mode"])
            and record["accepted"] == decision["accepted"],
            "Provider/decision identity or configured model mismatch",
        )
        if record["provider_call"]:
            require(
                record["sampling_settings"]
                == {
                    key: protocol["astra"][key]
                    for key in ("reasoning_effort", "max_completion_tokens")
                }
                and record["cache"] == {"no-cache": True}
                and record["response_format"] == {"type": "json_object"},
                "Provider sampling/cache/response settings changed",
            )
        if decision["accepted"]:
            require(
                record["provider_call"]
                and decision["proposal"] is not None
                and decision["error"] is None,
                "Accepted proposal lacks a physical response",
            )
        else:
            require(
                decision["proposal"] is None
                and isinstance(decision["error"], str)
                and bool(decision["error"])
                and decision["error"] == record["error"],
                "Rejected decision invented a proposal",
            )
    return records


def _snapshot_matches(wire, recorded):
    """Bind wire PNG/state values to the recorded raw observation descriptors."""
    for camera in ("observation/image", "observation/wrist_image"):
        with Image.open(
            io.BytesIO(base64.b64decode(wire[camera]["data"], validate=True))
        ) as image:
            array = np.asarray(image).copy()
        require(
            digest(array) == recorded[camera]["sha256"],
            "Astra feedback pixels differ from recorded raw observation",
        )
    descriptor = recorded["observation/state"]
    state = np.asarray(wire["observation/state"], dtype=descriptor["dtype"])
    require(
        digest(state) == descriptor["sha256"],
        "Astra feedback proprioception differs from raw observation",
    )


def _events(inputs, directory, report, entry, protocol, basis_id):
    physical = _physical(report)
    expected = {row["attempt_id"]: row for row in physical}
    require(len(expected) == len(physical), "Duplicate physical rollout identity")
    attempts, requests, decisions = set(), {}, set()
    snapshots, generations = {}, defaultdict(dict)
    counts = Counter()
    gate_labels = set()
    for sequence, event in enumerate(inputs.lines(directory / "events.jsonl")):
        require(event["sequence"] == sequence, "Missing or reordered event sequence")
        kind = event["kind"]
        counts[kind] += 1
        if kind == "case":
            require(
                sequence == 0
                and event["entry"] == entry
                and event["protocol"] == protocol,
                "Case event differs from frozen inputs",
            )
            require(
                event["basis_id"] == event["basis"]["sha256"] == basis_id,
                "Recorded noise basis differs from frozen seeded basis",
            )
        elif kind == "inversion_initialization":
            require(
                all(
                    event.get(key) == value
                    for key, value in report["initialization"].items()
                ),
                "Initialization event differs from summary",
            )
            for name in ("known_noise", "recovered_noise"):
                require(
                    event[name]["sha256"] == report["initialization"][f"{name}_sha256"],
                    "Initialization array hash differs",
                )
        elif kind == "interpolation_gate":
            require(
                all(
                    event.get(key) == value
                    for key, value in report["interpolation_gate"].items()
                ),
                "Interpolation gate event differs from summary",
            )
        elif kind == "interpolation_gate_reference":
            gate = report["interpolation_gate"]
            require(
                event["velocity_evaluations"] == 10
                and event["solver"] == gate["solver"]
                and event["observation_id"] == gate["observation_id"]
                and event["condition_id"] == gate["native_condition_id"]
                and event["latent"]["sha256"] == gate["known_noise_sha256"],
                "Interpolation reference event differs from measured gate",
            )
        elif kind == "interpolation_gate_check":
            label, gate = event["label"], report["interpolation_gate"]
            require(
                label not in gate_labels
                and label in gate["checks"]
                and all(
                    event.get(key) == value
                    for key, value in gate["checks"][label].items()
                )
                and event["observation_id"] == gate["observation_id"]
                and event["latent_sha256"] == gate["known_noise_sha256"],
                "Interpolation check event differs or is duplicated",
            )
            gate_labels.add(label)
        elif kind == "phase_attempt":
            row = event["attempt"]
            key = row["attempt_id"]
            require(
                key in expected and key not in attempts and row == expected[key],
                "Attempt event/summary differs or is duplicated",
            )
            attempts.add(key)
            snapshots[key] = event["snapshots"]
        elif kind == "interpolation_request":
            request = event["request"]
            _validate_request(request)
            key = (request["attempt_id"], request["decision_index"])
            require(
                key not in requests and key[0] in expected,
                "Duplicate or unassigned request",
            )
            row = expected[key[0]]
            require(row["mode"].startswith("astra_"), "Request outside an Astra arm")
            record = row["provider_records"][key[1] - 1]
            decision = row["decisions"][key[1] - 1]
            for name in IDENTITY:
                require(
                    record.get(name) == request[name],
                    f"Provider request binding mismatch: {name}",
                )
            require(
                request["episode_id"] == entry["episode_id"]
                and request["target_task"] == entry["instruction"]
                and request["source_catalog"] == donor_catalog(),
                "Request changed target task or restricted donor catalog",
            )
            require(
                request["previous_decisions"] == row["decisions"][: key[1] - 1],
                "Request differs from actual within-rollout feedback",
            )
            prior = report["arms"][row["mode"]]["attempts"][: row["iteration"] - 1]
            require(
                request["completed_rollout_feedback"]
                == [outcome_feedback(value) for value in prior],
                "Request mixes histories across arms or outcomes",
            )
            previous = request["previous_attempt"]
            require(
                previous is not None
                and previous["feedback"] == outcome_feedback(prior[-1])
                and previous["decisions"] == prior[-1]["decisions"],
                "Request previous rollout differs from recorded history",
            )
            old_snapshots = snapshots[prior[-1]["attempt_id"]]
            require(
                len(previous["snapshots"]) == len(old_snapshots),
                "Previous feedback snapshot coverage differs",
            )
            for wire, saved in zip(previous["snapshots"], old_snapshots, strict=True):
                require(
                    (wire["step"], wire["label"]) == (saved["step"], saved["label"]),
                    "Previous feedback frame identity differs",
                )
                _snapshot_matches(wire["observation"], saved["observation"])
            if record["accepted"]:
                response = record["response"]
                require(
                    response.get("model") == protocol["astra"]["model"]
                    and 200 <= record["http_status"] < 300
                    and len(response.get("choices", [])) == 1
                    and response["choices"][0].get("finish_reason") == "stop",
                    "Accepted response model/status is invalid",
                )
                parsed = parse_proposal(
                    response["choices"][0]["message"]["content"], request
                )
                require(
                    parsed == decision["proposal"]
                    and parsed["decision_id"] == record["decision_id"],
                    "Executed proposal differs from accepted provider response",
                )
            requests[key] = request
        elif kind == "interpolation_decision":
            key = event["attempt_id"], event["decision"]["decision_index"]
            require(
                key in requests
                and key not in decisions
                and event["decision"] == expected[key[0]]["decisions"][key[1] - 1],
                "Decision event differs from provider-bound summary",
            )
            decisions.add(key)
        elif kind == "phase_generation":
            key, step = event["attempt_id"], event["observation_step"]
            require(
                key in expected
                and step not in generations[key]
                and event["velocity_evaluations"] == 10,
                "Extra generation or wrong Euler call count",
            )
            row = expected[key]
            require(
                event["mode"] == row["mode"] and event["iteration"] == row["iteration"],
                "Generation identity differs from attempt",
            )
            generations[key][step] = event
        elif kind == "phase_development_validation":
            require(
                all(
                    event.get(key) == value
                    for key, value in report["development_validation"].items()
                ),
                "Development validation event differs",
            )
        else:
            raise ValueError(f"Unknown phase recording event: {kind}")
    require(
        attempts == set(expected)
        and counts["case"]
        == counts["inversion_initialization"]
        == counts["interpolation_gate"]
        == 1,
        "Incomplete physical event coverage",
    )
    require(
        counts["interpolation_gate_reference"] == 1
        and gate_labels == set(report["interpolation_gate"]["checks"]),
        "Missing measured interpolation probe records",
    )
    require(
        set(requests)
        == decisions
        == {
            (row["attempt_id"], value["decision_index"])
            for row in physical
            for value in row["decisions"]
        },
        "Incomplete request/decision coverage",
    )
    for key, request in requests.items():
        step = request["observation_step"]
        expected_steps = list(range(max(0, step - 15), step + 1, 5))
        require(
            [row["step"] for row in request["observations"]] == expected_steps,
            "Astra current frames are stale or omit the current observation",
        )
        for wire in request["observations"]:
            _snapshot_matches(
                wire["observation"], generations[key[0]][wire["step"]]["observation"]
            )
    for key, row in expected.items():
        events = generations[key]
        require(
            sorted(events) == list(range(0, row["actions_executed"], 5)),
            "Rollout generation coverage is incomplete",
        )
        applied, totals, policies = [], Counter(), Counter()
        active, active_id = None, None
        for step, event in sorted(events.items()):
            width = min(5, row["actions_executed"] - step)
            marks, held, fallback = [], False, False
            if row["mode"].startswith("astra_"):
                decision = row["decisions"][step // 25]
                if step % 25 == 0 and decision["accepted"]:
                    proposal = decision["proposal"]
                    active = {
                        name: proposal[name]
                        for name in ("source_a_id", "source_b_id", "alpha")
                    }
                    active_id = proposal["decision_id"]
                    marks = proposal["vision"]
                held, fallback = (
                    active_id is not None and not decision["accepted"],
                    active is None,
                )
            elif row["mode"].startswith("oracle_"):
                oracle = report["oracle"]
                active = {
                    "source_a_id": oracle["source_a_id"],
                    "source_b_id": oracle["source_b_id"],
                    "alpha": paper_alpha(step // 5, oracle["lambda_calls"]),
                }
            require(
                event["active_interpolation"] == active
                and event["applied_accepted_decision_id"] == active_id
                and event["vision"] == marks
                and event["held_text_after_failed_call"] == held
                and event["native_condition_fallback"] == fallback,
                "Applied interpolation/vision hold differs from accepted decisions or oracle schedule",
            )
            provenance = event["conditioning"]
            require(
                event["text_has_effect"]
                == bool(provenance and provenance["has_effect"]),
                "Text effect flag differs from model provenance",
            )
            if not marks:
                require(
                    event["vision_has_effect"] is False,
                    "Expired or absent vision marks changed the observation",
                )
            if row["mode"] != "random_noise":
                latent_kind = (
                    "known_noise"
                    if row["mode"] == "known_noise"
                    or (row["mode"] == "policy_fresh" and step == 0)
                    else "recovered_noise"
                )
                if row["mode"] != "policy_fresh" or step == 0:
                    require(
                        event["latent"]["sha256"]
                        == report["initialization"][f"{latent_kind}_sha256"],
                        "Fixed known/recovered latent changed within rollout",
                    )
            if active_id is not None:
                totals["actions_with_accepted_decision"] += width
                policies["accepted_decision_policy_calls"] += 1
                if active_id not in applied:
                    applied.append(active_id)
            for flag, field in ACTION_FLAGS.items():
                require(type(event[flag]) is bool, "Nonboolean executed effect flag")
                totals[field] += width * event[flag]
            policies["text_nonzero_policy_calls"] += event["text_has_effect"]
            policies["vision_changed_policy_calls"] += event["vision_has_effect"]
            policies["vision_active_policy_calls"] += bool(marks)
            policies["native_condition_fallback_policy_calls"] += fallback
        require(
            row["applied_accepted_decision_ids"] == applied
            and row["accepted_decisions_executed"] == len(applied),
            "Accepted/executed proposal denominator differs",
        )
        for name in ("actions_with_accepted_decision", *ACTION_FLAGS.values()):
            require(
                row[name] == totals[name],
                f"Executed action attribution mismatch: {name}",
            )
        for name, value in policies.items():
            require(
                row[name] == value, f"Executed policy-call attribution mismatch: {name}"
            )
    return dict(counts)


def _attempt_view(row):
    previous = None
    decisions = []
    for decision, record in zip(row["decisions"], row["provider_records"], strict=True):
        proposal = decision["proposal"]
        selection = (
            {key: proposal[key] for key in ("source_a_id", "source_b_id", "alpha")}
            if proposal
            else None
        )
        decisions.append(
            {
                "decision_index": decision["decision_index"],
                "observation_step": decision["observation_step"],
                "accepted": decision["accepted"],
                "provider_call": record["provider_call"],
                "error_kind": record.get("error_kind"),
                "request_fingerprint": record["request_fingerprint"],
                "decision_id": proposal["decision_id"] if proposal else None,
                "selection": selection,
                "active_selection_after_decision": selection
                if selection is not None
                else previous,
                "vision_annotations": len(proposal["vision"]) if proposal else 0,
                "source_pair_changed": previous is not None
                and selection is not None
                and (selection["source_a_id"], selection["source_b_id"])
                != (previous["source_a_id"], previous["source_b_id"]),
                "alpha_changed": previous is not None
                and selection is not None
                and selection["alpha"] != previous["alpha"],
                "token_usage": record["token_usage"],
            }
        )
        if selection is not None:
            previous = selection
    fields = (
        "attempt_id",
        "mode",
        "iteration",
        "success",
        "status",
        "actions_executed",
        "policy_replans",
        "velocity_evaluations",
        "wall_seconds",
        "accepted_decisions_executed",
        "applied_accepted_decision_ids",
        "text_nonzero_policy_calls",
        "vision_changed_policy_calls",
        "actions_with_accepted_decision",
        "actions_with_nonzero_text",
        "actions_with_changed_vision",
        "actions_with_held_text_after_failed_call",
        "native_condition_fallback_actions",
    )
    return {
        **{key: row[key] for key in fields},
        "reset_sha256": row["reset_audit"]["sha256"],
        "decisions": decisions,
        "provider": summarize_calls(row["provider_records"]),
    }


def _case(inputs, directory, entry, benchmark, protocol, phase, checkpoint):
    report = inputs.read(directory / "summary.json")
    require(
        report["status"] == "complete"
        and report["schema_version"] == "phase-interpolation-1.0"
        and report["episode_id"] == entry["episode_id"]
        and report["suite"] == benchmark.suite
        and report["task_id"] == entry["task_id"]
        and report["seed"] == protocol["seed"]
        and report["development"] == (phase == "development")
        and report["protocol_sha256"] == digest(protocol)
        and report["reset_entry_sha256"] == digest(entry)
        and report["checkpoint"] == checkpoint,
        "Case identity/status differs from frozen inputs",
    )
    require(
        set(report["arms"]) == set(ARMS) and set(report["controls"]) == set(CONTROLS),
        "Missing or undeclared arms/controls",
    )
    require(
        report["oracle"] == oracle_for(benchmark.suite, entry["task_id"]).metadata()
        and report["source_catalog"] == donor_catalog(),
        "Oracle mapping or donor library changed",
    )
    _validate_gates(report, protocol)
    baseline = report["baseline"]
    require(
        baseline["mode"] == "recovered_noise"
        and baseline["attempt_id"] == "recovered_noise_1"
        and baseline["iteration"] == 1,
        "Invalid common recovered-noise baseline",
    )
    physical = _physical(report)
    require(
        len({row["attempt_id"] for row in physical}) == len(physical),
        "Shared baseline was physically duplicated",
    )
    for row in physical:
        _attempt(
            {**row, "rollout_executed": True}, entry, benchmark, baseline["reset_audit"]
        )
        _ledger(inputs, directory, row, protocol)
    expected_sidecars = {
        f"{row['attempt_id']}_provider.jsonl"
        for row in physical
        if row["provider_records"]
    }
    require(
        {path.name for path in directory.glob("*_provider.jsonl")} == expected_sidecars,
        "Extra or omitted provider ledger",
    )
    for mode, row in report["controls"].items():
        require(
            row["mode"] == mode
            and row["attempt_id"] == f"{mode}_1"
            and row["iteration"] == 1
            and row["noise_proposal"] is None,
            "Invalid native control",
        )
    seed_words = [protocol["seed"], int(digest(entry["episode_id"])[:8], 16)]
    basis, basis_id = noise_basis(
        (1, checkpoint["horizon"], checkpoint["model_action_dim"]),
        np.random.SeedSequence(seed_words + [1]),
    )
    rng = np.random.default_rng(np.random.SeedSequence(seed_words + [3]))
    random_checks = []
    for arm in ARMS:
        value = report["arms"][arm]
        attempts = value["attempts"]
        cap = (
            protocol["oracle_attempt_budget"]
            if arm.startswith("oracle_")
            else protocol["attempt_budget"]
        )
        require(
            attempts
            and attempts[0] == baseline
            and [row["iteration"] for row in attempts]
            == list(range(1, len(attempts) + 1))
            and len(attempts) <= cap,
            "Arm does not share baseline/consecutive bounded attempts",
        )
        for row in attempts[1:]:
            require(
                row["mode"] == arm and row["attempt_id"] == f"{arm}_{row['iteration']}",
                "Intervention rollout identity differs",
            )
            prior_success = any(
                prior["success"] for prior in attempts[: row["iteration"] - 1]
            )
            require(
                not prior_success or (phase == "development" and row["iteration"] == 2),
                "Rollout continued after success outside development forcing",
            )
            if arm == "random_noise":
                random_checks.append(
                    _random_noise_check(
                        {
                            "rollout_executed": True,
                            "iteration": row["iteration"],
                            "proposal": {
                                "noise": row["noise_proposal"],
                                "language": None,
                                "vision": [],
                            },
                        },
                        random_noise_proposal(basis, rng),
                    )
                )
            else:
                require(
                    row["noise_proposal"] is None,
                    "Conditioning arm modified recovered noise",
                )
        require(
            any(row["success"] for row in attempts) or len(attempts) == cap,
            "Unsuccessful arm was censored before its allowed cap",
        )
        require(
            phase != "development" or len(attempts) >= 2,
            "Development did not execute its forced extra rollout",
        )
        computed = arm_summary(attempts, cap)
        computed["standalone_velocity_evaluations_through_success_or_cap"] = (
            computed["velocity_evaluations_through_success_or_cap"]
            + report["initialization"]["velocity_evaluations"]
            + report["interpolation_gate"]["velocity_evaluations"]
        )
        require(
            value["summary"] == computed,
            "Recorded arm cost/success/censor summary differs from attempts",
        )
    event_counts = _events(inputs, directory, report, entry, protocol, basis_id)
    cost = _physical_cost(
        physical,
        report["initialization"]["velocity_evaluations"],
        report["interpolation_gate"]["velocity_evaluations"],
    )
    require(
        all(report["physical_cost"].get(key) == value for key, value in cost.items()),
        "Physical cost does not count unique rollouts/setup exactly once",
    )
    if phase == "development":
        validation = report["development_validation"]
        require(
            validation["passed"] is True
            and all(
                validation["arms"][arm]["passed"] is True
                and sum(
                    row["accepted_decisions_executed"]
                    for row in report["arms"][arm]["attempts"][1:]
                )
                > 0
                for arm in ARMS
                if arm.startswith("astra_")
            ),
            "Development integration did not execute a genuine accepted proposal in every Astra arm",
        )
    case = {
        "episode_id": entry["episode_id"],
        "suite": benchmark.suite,
        "task_id": entry["task_id"],
        "instruction": entry["instruction"],
        "seed": protocol["seed"],
        "summary_source": inputs.source(directory / "summary.json"),
        "reset_entry_sha256": digest(entry),
        "reset_audit": baseline["reset_audit"],
        "initialization": report["initialization"],
        "interpolation_gate": report["interpolation_gate"],
        "baseline": _attempt_view(baseline),
        "controls": {
            key: _attempt_view(row) for key, row in report["controls"].items()
        },
        "arms": {
            arm: {
                "summary": report["arms"][arm]["summary"],
                "attempts": [
                    _attempt_view(row) for row in report["arms"][arm]["attempts"]
                ],
            }
            for arm in ARMS
        },
        "oracle": report["oracle"],
        "physical_cost": cost,
        "case_wall_seconds": _number(report["total_wall_seconds"], "case wall time"),
        "events_verified": event_counts,
        "random_noise_numeric_checks": random_checks,
        "development_validation": report.get("development_validation"),
    }
    return case, report


def _prefix(attempts, budget=None):
    first = next((row["iteration"] for row in attempts if row["success"]), None)
    return [
        row
        for row in attempts
        if (budget is None or row["iteration"] <= budget)
        and (first is None or row["iteration"] <= first)
    ]


def _usage(rows):
    return summarize_calls([call for row in rows for call in row["provider_records"]])


def _group(cases, originals):
    n = len(cases)
    baseline_successes = sum(row["baseline"]["success"] for row in cases)
    failed = n - baseline_successes
    arms = {}
    for arm in ARMS:
        cap = 2 if arm.startswith("oracle_") else 3
        rescues = [
            case
            for case in cases
            if not case["baseline"]["success"]
            and case["arms"][arm]["summary"]["success"]
        ]
        curves = []
        for budget in range(1, cap + 1):
            prefixes = [
                _prefix(raw["arms"][arm]["attempts"], budget) for raw in originals
            ]
            successes = sum(
                any(row["success"] for row in prefix) for prefix in prefixes
            )
            rows = [row for prefix in prefixes for row in prefix]
            curves.append(
                {
                    "up_to_attempts": budget,
                    "cases": n,
                    "successes": successes,
                    "success_rate": successes / n if n else None,
                    "rescues": successes - baseline_successes,
                    "baseline_failed_cases": failed,
                    "conditional_rescue_rate": (successes - baseline_successes) / failed
                    if failed
                    else None,
                    "attributed_rollouts": len(rows),
                    "attributed_actions": sum(row["actions_executed"] for row in rows),
                    "attributed_velocity_evaluations": sum(
                        row["velocity_evaluations"] for row in rows
                    )
                    + sum(
                        raw["initialization"]["velocity_evaluations"]
                        + raw["interpolation_gate"]["velocity_evaluations"]
                        for raw in originals
                    ),
                    "provider": _usage(rows),
                }
            )
        complete_tokens = [
            case["arms"][arm]["summary"]["tokens_to_first_success"]["tokens"][
                "total_tokens"
            ]["sum"]
            for case in rescues
            if case["arms"][arm]["summary"]["tokens_to_first_success"]["tokens"][
                "total_tokens"
            ]["complete"]
        ]
        comparisons = {
            key: []
            for key in (
                "both_succeed",
                "arm_only_succeeds",
                "random_only_succeeds",
                "both_fail",
            )
        }
        for case in cases:
            a = case["arms"][arm]["summary"]["success_by_attempt"][cap - 1]
            b = case["arms"]["random_noise"]["summary"]["success_by_attempt"][cap - 1]
            key = (
                "both_succeed"
                if a and b
                else "arm_only_succeeds"
                if a
                else "random_only_succeeds"
                if b
                else "both_fail"
            )
            comparisons[key].append(case["episode_id"])
        extras = [
            row
            for raw in originals
            for row in raw["arms"][arm]["attempts"]
            if row not in _prefix(raw["arms"][arm]["attempts"])
        ]
        arms[arm] = {
            "attempt_cap": cap,
            "cases": n,
            "successes": curves[-1]["successes"],
            "rescues": len(rescues),
            "baseline_failed_cases": failed,
            "censored_without_success": n - curves[-1]["successes"],
            "success_by_budget": curves,
            "median_first_success_attempt_among_all_successes": _median(
                [
                    case["arms"][arm]["summary"]["first_success_attempt"]
                    for case in cases
                    if case["arms"][arm]["summary"]["success"]
                ]
            ),
            "median_rollout_revisions_among_rescues": _median(
                [
                    case["arms"][arm]["summary"]["full_rollout_revisions_to_success"]
                    for case in rescues
                ]
            ),
            "median_decisions_through_success_among_rescues": _median(
                [
                    case["arms"][arm]["summary"]["decisions_through_success_or_cap"]
                    for case in rescues
                ]
            ),
            "median_decisions_in_winning_rollout_among_rescues": _median(
                [
                    case["arms"][arm]["summary"]["within_successful_rollout_decisions"]
                    for case in rescues
                ]
            ),
            "median_total_tokens_among_rescues_with_complete_usage": _median(
                complete_tokens
            ),
            "rescues_with_missing_total_token_usage": len(rescues)
            - len(complete_tokens),
            "rescue_episode_ids": [case["episode_id"] for case in rescues],
            "paired_random_at_same_cap": {
                "up_to_attempts": cap,
                "counts": {key: len(value) for key, value in comparisons.items()},
                "episode_ids": comparisons,
            },
            "development_extra_rollouts_after_success": len(extras),
            "development_extra_actions_after_success": sum(
                row["actions_executed"] for row in extras
            ),
            "development_extra_provider_after_success": _usage(extras),
            "physical_provider": _usage(
                [row for raw in originals for row in raw["arms"][arm]["attempts"][1:]]
            ),
        }
    physical = [row for raw in originals for row in _physical(raw)]
    cost = _physical_cost(
        physical,
        sum(raw["initialization"]["velocity_evaluations"] for raw in originals),
        sum(raw["interpolation_gate"]["velocity_evaluations"] for raw in originals),
    )
    cost["case_wall_seconds_sum"] = sum(raw["total_wall_seconds"] for raw in originals)
    return {
        "cases": n,
        "baseline_successes": baseline_successes,
        "baseline_failed_cases": failed,
        "controls": {
            label: {
                "successes": sum(row["success"] for row in rows),
                "cases": n,
                "actions": sum(row["actions_executed"] for row in rows),
                "velocity_evaluations": sum(
                    row["velocity_evaluations"] for row in rows
                ),
            }
            for label, rows in [
                ("recovered_noise", [raw["baseline"] for raw in originals]),
                *[
                    (mode, [raw["controls"][mode] for raw in originals])
                    for mode in CONTROLS
                ],
            ]
        },
        "arms": arms,
        "physical_cost": cost,
        "execution": {
            "accepted_decisions_executed": sum(
                row["accepted_decisions_executed"] for row in physical
            ),
            **{
                name: sum(row[name] for row in physical)
                for name in ("actions_with_accepted_decision", *ACTION_FLAGS.values())
            },
            "zero_action_successes": sum(
                row["success"] and row["actions_executed"] == 0 for row in physical
            ),
        },
    }


def build_report(directories, *, phase):
    require(
        phase in ("development", "evaluation"),
        "Specify development or evaluation explicitly",
    )
    workers_expected, cases_expected, seed = (
        (3, 3, 19) if phase == "development" else (8, 20, 29)
    )
    directories = [Path(path).resolve() for path in directories]
    require(
        len(set(directories)) == len(directories) == workers_expected,
        "Incomplete or duplicated worker folders",
    )
    inputs = Inputs()
    workers, cases, originals = [], [], []
    seen, manifests = set(), {}
    common = None
    expected_protocol = load_protocol()
    expected_protocol["seed"] = seed
    for directory in sorted(directories):
        require(
            not (directory / "failure.json").exists(),
            "Worker recorded a runtime failure",
        )
        runtime = inputs.read(directory / "runtime.json")
        index = runtime["worker"]
        require(
            type(index) is int
            and index in range(workers_expected)
            and index not in seen,
            "Invalid or duplicated worker index",
        )
        seen.add(index)
        target = assignment(phase, index)
        require(
            runtime["phase"] == phase
            and runtime["assignment"] == target
            and runtime["tf32"] is False
            and "L40S" in runtime["gpu"]
            and _sha(runtime["payload_sha256"])
            and bool(runtime["workflow"]),
            "Runtime differs from fixed worker/phase/settings",
        )
        protocol = inputs.read(directory / "protocol.json")
        checkpoint = inputs.read(directory / "checkpoint.json")
        require(
            protocol == expected_protocol,
            "Protocol differs from the declared phase experiment",
        )
        require(
            checkpoint["frozen"] is True
            and checkpoint["input_profile"] == "openpi_libero"
            and checkpoint["training_overlap"] == "unknown"
            and checkpoint["horizon"] == 10
            and checkpoint["model_action_dim"] == 32,
            "Unexpected policy/checkpoint profile",
        )
        inventory = inputs.read(directory / "bank_inventory.json")
        _validate_inventory(inventory)
        identity = (
            protocol,
            checkpoint,
            runtime["packages"],
            runtime["payload_sha256"],
            inputs.source(directory / "bank_inventory.json")["sha256"],
        )
        if common is None:
            common = identity
        require(
            identity == common,
            "Workers differ in protocol/checkpoint/runtime/payload/bank identity",
        )
        benchmark = BenchmarkConfig.preset(target["suite"])
        raw_manifest = inputs.read(directory / "reset_manifest.json")
        manifest = load_reset_manifest(directory / "reset_manifest.json", benchmark)
        fixed = (
            protocol["development_cases"][target["suite"]]
            if phase == "development"
            else protocol["evaluation_cases"]
        )
        require(
            raw_manifest == manifest
            and manifest["seed"] == seed
            and manifest["cases"] == fixed
            and manifest["split"]
            == ("development" if phase == "development" else "followup_adaptation"),
            "Reset manifest differs from fixed phase cases",
        )
        require(
            manifests.get(target["suite"], manifest["sha256"]) == manifest["sha256"],
            "Full reset manifest differs across workers in same suite",
        )
        manifests[target["suite"]] = manifest["sha256"]
        entries = (
            [row for row in manifest["episodes"] if row["task_id"] == target["task_id"]]
            if phase == "development"
            else manifest["episodes"][target["case_shard"] :: target["case_shards"]]
        )
        plan = inputs.read(directory / "frozen_plan.json")
        require(
            plan["runtime"] == runtime
            and plan["protocol_sha256"] == digest(protocol)
            and plan["manifest_sha256"] == manifest["sha256"]
            and plan["bank_inventory_sha256"] == identity[-1]
            and plan["assigned_episodes"] == [row["episode_id"] for row in entries],
            "Frozen plan differs from prescribed assignment or complete input hashes",
        )
        progress = inputs.read(directory / "progress.json")
        require(
            progress["status"] == "complete"
            and all(progress[key] == value for key, value in target.items()),
            "Worker is incomplete",
        )
        require(
            {path.name for path in directory.glob("case_*") if path.is_dir()}
            == {f"case_{row['task_id']}_{row['initial_state_id']}" for row in entries},
            "Extra or omitted fixed case directory",
        )
        worker_reports = []
        for entry in entries:
            case, original = _case(
                inputs,
                directory / f"case_{entry['task_id']}_{entry['initial_state_id']}",
                entry,
                benchmark,
                protocol,
                phase,
                checkpoint,
            )
            case.update(
                workflow=runtime["workflow"],
                worker=index,
                reset_manifest_sha256=manifest["sha256"],
                frozen_plan_source=inputs.source(directory / "frozen_plan.json"),
            )
            cases.append(case)
            originals.append(original)
            worker_reports.append(original)
        require(
            inputs.read(directory / "aggregate.json")
            == aggregate_reports(worker_reports, protocol),
            "Worker aggregate differs from complete case outcomes",
        )
        workers.append(
            {
                "runtime": runtime,
                "assigned_episodes": plan["assigned_episodes"],
                "reset_manifest_sha256": manifest["sha256"],
                "frozen_plan_source": inputs.source(directory / "frozen_plan.json"),
            }
        )
    require(
        len(cases) == len({row["episode_id"] for row in cases}) == cases_expected,
        "Incomplete or duplicate prescribed episode coverage",
    )
    order = sorted(
        range(len(cases)), key=lambda i: (cases[i]["suite"], cases[i]["task_id"])
    )
    cases, originals = [cases[i] for i in order], [originals[i] for i in order]
    inputs.finish()
    grouped = {"pooled": _group(cases, originals)}
    for suite in SUITES:
        indices = [i for i, row in enumerate(cases) if row["suite"] == suite]
        grouped[suite] = _group(
            [cases[i] for i in indices], [originals[i] for i in indices]
        )
    return {
        "schema_version": "astra-phase-interpolation-report-1",
        "status": "complete",
        "phase": phase,
        "expected_cases": cases_expected,
        "seed": seed,
        "protocol": common[0],
        "protocol_sha256": digest(common[0]),
        "checkpoint": common[1],
        "payload_sha256": common[3],
        "bank_inventory_file_sha256": common[4],
        "reset_manifest_sha256": manifests,
        "workers": workers,
        "cases": cases,
        "groups": grouped,
        "integration": {
            "recording_and_provider_bindings_verified": True,
            "native_and_nonzero_gates_passed": True,
            "development_accepted_execution_gate": True
            if phase == "development"
            else None,
            "array_replay_performed": False,
        },
        "sources": sorted(inputs.sources.values(), key=lambda row: row["path"]),
        "absent_provider_paths": sorted(inputs.absent),
        "postprocessor": {
            "path": str(Path(__file__).resolve()),
            "sha256": file_sha256(__file__),
            "dependencies": {
                name: file_sha256(Path(__file__).parent / name)
                for name in (
                    "interpolation_search.py",
                    "interpolation_agent.py",
                    "interpolation_catalog.py",
                    "intervention_report.py",
                    "intervention_rollout.py",
                    "interventions.py",
                    "records.py",
                )
            },
        },
        "limitations": LIMITATIONS,
    }


def _display(value):
    return (
        "—"
        if value is None
        else str(round(value, 3))
        if isinstance(value, float)
        else str(value)
    )


def _token_display(usage, field):
    row = usage["tokens"][field]
    return f"{row['sum']:,}" + (
        f" (partial; {row['missing_calls']} calls missing)"
        if not row["complete"]
        else ""
    )


def _markdown(report):
    pooled = report["groups"]["pooled"]
    physical = pooled["physical_cost"]
    usage = physical["token_usage"]
    text = [
        f"# Phase interpolation: {report['phase']}",
        "",
        f"Complete audited coverage: **{pooled['cases']} fixed cases**, seed {report['seed']}. Common recovered-noise baseline: **{pooled['baseline_successes']}/{pooled['cases']}**.",
        "",
        f"Physical case execution: {physical['rollouts']:,} rollouts, {physical['simulated_actions']:,} actions, {physical['velocity_evaluations']:,} velocity evaluations. Provider: {usage['provider_calls']} physical calls, {usage['failed_calls']} failed calls, {usage['preflight_failures']} preflight failures.",
        f"Physical reported tokens: input **{_token_display(usage, 'input_tokens')}**, output **{_token_display(usage, 'output_tokens')}**, total **{_token_display(usage, 'total_tokens')}**; reasoning subset **{_token_display(usage, 'reasoning_tokens')}**. Dollar cost unknown.",
        "",
        "A budget is **up to** that many full rollouts including the common baseline. Oracle caps at 2; random/Astra cap at 3. Median revisions and tokens below are conditional on rescue, so baseline successes do not dominate them. Token medians use rescues with complete reported usage; missing-usage counts remain in the JSON/CSV.",
    ]
    for label, group in report["groups"].items():
        text.extend(
            [
                "",
                f"## {label} ({group['cases']} cases; {group['baseline_failed_cases']} baseline failures)",
                "",
                "| Arm | Cap | Success | Rescues | Censored | Median rescue revisions | Median rescue decisions | Median rescue tokens |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for arm, row in group["arms"].items():
            text.append(
                f"| {arm} | {row['attempt_cap']} | {row['successes']}/{row['cases']} | {row['rescues']}/{row['baseline_failed_cases']} | {row['censored_without_success']} | {_display(row['median_rollout_revisions_among_rescues'])} | {_display(row['median_decisions_through_success_among_rescues'])} | {_display(row['median_total_tokens_among_rescues_with_complete_usage'])} |"
            )
        text.extend(
            [
                "",
                "Native controls: "
                + "; ".join(
                    f"{name} {value['successes']}/{value['cases']}"
                    for name, value in group["controls"].items()
                )
                + ".",
            ]
        )
    if report["phase"] == "development":
        extra = sum(
            row["development_extra_rollouts_after_success"]
            for row in pooled["arms"].values()
        )
        text.extend(
            [
                "",
                f"Development forced extras after first success: **{extra} rollouts**. Their costs are included in physical totals and excluded from success-prefix token curves. Passing integration demonstrates executed proposals and measured hooks, not intervention efficacy.",
            ]
        )
    text.extend(
        [
            "",
            "## Interpretation limits",
            "",
            *[f"- {row}" for row in report["limitations"]],
            "",
        ]
    )
    return "\n".join(text)


def _plots(directory, report):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    for filename, field, ylabel in (
        ("success_by_budget.png", "success_rate", "Success / all cases"),
        (
            "conditional_rescue_by_budget.png",
            "conditional_rescue_rate",
            "Rescues / baseline failures",
        ),
        ("reported_tokens_by_budget.png", "tokens", "Reported total tokens / case"),
    ):
        figure, axes = plt.subplots(1, 3, figsize=(16, 4.5), squeeze=False)
        for axis, (label, group) in zip(axes[0], report["groups"].items(), strict=True):
            for arm, result in group["arms"].items():
                rows = result["success_by_budget"]
                x = [row["up_to_attempts"] for row in rows]
                if field == "tokens":
                    y = [
                        row["provider"]["tokens"]["total_tokens"]["sum"]
                        / group["cases"]
                        for row in rows
                    ]
                    partial = any(
                        not row["provider"]["tokens"]["total_tokens"]["complete"]
                        for row in rows
                    )
                else:
                    y = [
                        float("nan") if row[field] is None else row[field]
                        for row in rows
                    ]
                    partial = False
                axis.plot(
                    x, y, marker="o", label=arm + (" (partial)" if partial else "")
                )
            axis.set_title(
                f"{label}\nN={group['cases']}, baseline failures={group['baseline_failed_cases']}"
            )
            axis.set_xticks([1, 2, 3])
            axis.set_xlabel("Up to full rollouts (baseline included)")
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.25)
            if field != "tokens":
                axis.set_ylim(-0.03, 1.03)
            if (
                field == "conditional_rescue_rate"
                and group["baseline_failed_cases"] == 0
            ):
                axis.text(
                    0.5,
                    0.5,
                    "No baseline failures; rescue rate undefined",
                    transform=axis.transAxes,
                    ha="center",
                    wrap=True,
                )
        axes[0][-1].legend(fontsize=7, loc="best")
        figure.suptitle(
            f"{report['phase']} only • exploratory known tasks • oracle cap 2; others cap 3"
        )
        figure.tight_layout()
        path = directory / filename
        figure.savefig(
            path, dpi=180, metadata={"Software": "astra_reversal.interpolation_report"}
        )
        plt.close(figure)
        paths.append(filename)
    return paths


def write_outputs(directory, report):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    (directory / "report.md").write_text(_markdown(report))
    tables = {
        "curves.csv": [],
        "cases.csv": [],
        "decisions.csv": [],
        "controls.csv": [],
    }
    for group, value in report["groups"].items():
        for arm, summary in value["arms"].items():
            for row in summary["success_by_budget"]:
                usage = row["provider"]
                tables["curves.csv"].append(
                    {
                        "phase": report["phase"],
                        "group": group,
                        "arm": arm,
                        "attempt_cap": summary["attempt_cap"],
                        **{
                            key: row[key]
                            for key in (
                                "up_to_attempts",
                                "cases",
                                "successes",
                                "success_rate",
                                "rescues",
                                "baseline_failed_cases",
                                "conditional_rescue_rate",
                                "attributed_rollouts",
                                "attributed_actions",
                                "attributed_velocity_evaluations",
                            )
                        },
                        "provider_calls": usage["provider_calls"],
                        "preflight_failures": usage["preflight_failures"],
                        **{
                            field: usage["tokens"][field]["sum"]
                            for field in (
                                "input_tokens",
                                "output_tokens",
                                "total_tokens",
                                "reasoning_tokens",
                            )
                        },
                        "total_token_missing_calls": usage["tokens"]["total_tokens"][
                            "missing_calls"
                        ],
                    }
                )
        for name, row in value["controls"].items():
            tables["controls.csv"].append(
                {"phase": report["phase"], "group": group, "control": name, **row}
            )
    for case in report["cases"]:
        for arm, value in case["arms"].items():
            summary = value["summary"]
            prefix = summary["provider_through_success_or_cap"]
            tables["cases.csv"].append(
                {
                    "episode_id": case["episode_id"],
                    "suite": case["suite"],
                    "task_id": case["task_id"],
                    "arm": arm,
                    "baseline_success": case["baseline"]["success"],
                    "success": summary["success"],
                    "rescued": not case["baseline"]["success"] and summary["success"],
                    **{
                        key: summary[key]
                        for key in (
                            "first_success_attempt",
                            "full_rollout_revisions_to_success",
                            "within_successful_rollout_decisions",
                            "decisions_through_success_or_cap",
                            "censored_without_success",
                            "development_extra_rollouts_after_success",
                        )
                    },
                    "prefix_total_tokens_reported": prefix["tokens"]["total_tokens"][
                        "sum"
                    ],
                    "prefix_total_tokens_complete": prefix["tokens"]["total_tokens"][
                        "complete"
                    ],
                    "reset_sha256": case["reset_audit"]["sha256"],
                }
            )
            for attempt in value["attempts"][1:]:
                for decision in attempt["decisions"]:
                    selection = decision["selection"] or {}
                    tables["decisions.csv"].append(
                        {
                            "episode_id": case["episode_id"],
                            "suite": case["suite"],
                            "arm": arm,
                            "iteration": attempt["iteration"],
                            "attempt_success": attempt["success"],
                            **{
                                key: decision[key]
                                for key in (
                                    "decision_index",
                                    "observation_step",
                                    "accepted",
                                    "provider_call",
                                    "error_kind",
                                    "decision_id",
                                    "source_pair_changed",
                                    "alpha_changed",
                                    "vision_annotations",
                                )
                            },
                            **{
                                key: selection.get(key)
                                for key in ("source_a_id", "source_b_id", "alpha")
                            },
                            **decision["token_usage"],
                        }
                    )
    for name, rows in tables.items():
        with (directory / name).open("w", newline="") as stream:
            fields = list(rows[0]) if rows else ["episode_id", "arm", "decision_index"]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    plots = _plots(directory, report)
    names = ["report.json", "report.md", *tables, *plots]
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "astra-phase-interpolation-report-files-1",
                "phase": report["phase"],
                "postprocessor": report["postprocessor"],
                "files": {
                    name: {
                        "sha256": file_sha256(directory / name),
                        "bytes": (directory / name).stat().st_size,
                    }
                    for name in names
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--phase", choices=("development", "evaluation"), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = build_report(args.inputs, phase=args.phase)
    write_outputs(args.output, report)
    print(
        json.dumps(
            {
                "phase": args.phase,
                "cases": len(report["cases"]),
                "baseline_successes": report["groups"]["pooled"]["baseline_successes"],
                "report_sha256": file_sha256(Path(args.output) / "report.json"),
            }
        )
    )


if __name__ == "__main__":
    main()
