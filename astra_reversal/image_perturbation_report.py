"""Offline, audit-gated image-perturbation outcomes and measured search costs.

This aggregates recorded summaries and independent audit receipts. It performs
no inference, simulation or array replay; a passing full case audit is required
before pooled efficacy or paired-comparison claims are released.
"""

import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path, PurePosixPath

from .astra_client import _strict_json
from .image_perturbation_agent import summarize_calls
from .image_perturbation_search import arm_summary
from .records import digest, file_sha256

SCHEMA_VERSION = "image-perturbation-report-1.0"
CONTROLS = ("known_noise", "policy_fresh")
PAIRS = (
    ("astra_occlusion", "random_occlusion"),
    ("astra_demo_blend", "random_demo_blend"),
)
LIMITATIONS = [
    "Exploratory adaptation on known task compositions with recorded resets; this is not a held-out-task or zero-shot claim.",
    "All arms share one recovered-noise baseline. A baseline success is not an intervention rescue. A failed capped search remains in the outcome denominator.",
    "The RGB operators keep task language and recovered noise fixed. A successful trajectory can include native fallback actions or accepted no-op decisions; acceptance is not proof of a nonzero effect or causation.",
    "The fixed training-donor catalog is not selected from evaluation outcomes. Training overlap with the policy checkpoint is not established as absent.",
    "Rescue-only timing/iteration/token statistics condition on a failed baseline and later success. Censored cases are listed separately and receive no invented time-to-success.",
    "Physical costs count the shared baseline, initialization and image gate once. Standalone arm costs repeat common setup and must not be added across arms.",
    "Physical totals cover the reconciled completed cases only. Donor preparation, transport diagnostics and any interrupted or incomplete execution require separate overhead receipts.",
    "Every physical provider call, including rejection, contributes available usage. Missing usage stays unknown; reasoning tokens are a subset of output, and no dollar price is assumed.",
    "Summed rollout/case wall time is measured work across cases, not elapsed workflow time; rollout time includes inference, recording and baseline setup.",
    "This reporter reconciles summaries and audit receipts. The bound independent audits verify recorded arrays, feedback and provider provenance without replaying simulator physics or hidden model states.",
    "Development may force one extra rollout after baseline success. These physical costs do not change zero intervention tokens to that baseline success.",
]


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _count(value, name, maximum=None):
    _require(type(value) is int and value >= 0, f"Invalid {name}")
    if maximum is not None:
        _require(value <= maximum, f"{name} exceeds its cap")
    return value


def _seconds(value, name):
    _require(
        type(value) in (float, int) and math.isfinite(value) and value >= 0,
        f"Invalid {name}",
    )
    return value


def _read(path):
    return _strict_json(Path(path).read_bytes())


def _safe_error(error):
    if isinstance(error, OSError):
        return f"Artifact read failed ({type(error).__name__})"
    return str(error)


def _same(actual, expected, name):
    _require(
        digest(actual) == digest(expected), f"{name} disagrees with recorded components"
    )


def _artifact_hashes(value):
    _require(isinstance(value, dict), "Receipt file hashes must be an object")
    for name, sha in value.items():
        _require(
            isinstance(name, str)
            and name
            and not PurePosixPath(name).is_absolute()
            and ".." not in PurePosixPath(name).parts
            and "\\" not in name
            and ":" not in name,
            "Receipt artifact identifiers must be portable relative names",
        )
        _require(
            isinstance(sha, str) and re.fullmatch("[0-9a-f]{64}", sha),
            "Invalid receipt artifact digest",
        )
    return value


def _stats(values):
    return {
        "count": len(values),
        "minimum": min(values) if values else None,
        "median": statistics.median(values) if values else None,
        "maximum": max(values) if values else None,
    }


def _tokens_value(usage):
    tokens = usage["tokens"]["total_tokens"]
    return tokens["sum"] if tokens["complete"] else None


def _physical(report, arms):
    return [
        report["baseline"],
        *(report["controls"][mode] for mode in CONTROLS),
        *(row for arm in arms for row in report["arms"][arm]["attempts"][1:]),
    ]


def _physical_cost(attempts, initialization, gate):
    calls = [call for row in attempts for call in row["provider_records"]]
    rollout_vf = sum(row["velocity_evaluations"] for row in attempts)
    return {
        "rollouts": len(attempts),
        "simulated_actions": sum(row["actions_executed"] for row in attempts),
        "rollout_velocity_evaluations": rollout_vf,
        "initialization_velocity_evaluations": initialization,
        "image_gate_velocity_evaluations": gate,
        "velocity_evaluations": rollout_vf + initialization + gate,
        "rollout_wall_seconds": sum(row["wall_seconds"] for row in attempts),
        "token_usage": summarize_calls(calls),
    }


def _validate_attempt(row, *, mode, iteration, episode_id, protocol):
    _require(isinstance(row, dict), "Attempt must be an object")
    _require(
        row["mode"] == mode
        and type(row["iteration"]) is int
        and row["iteration"] == iteration,
        "Attempt mode/iteration differs from its arm",
    )
    _require(
        row["attempt_id"] == f"{mode}_{iteration}",
        "Attempt identity differs from mode/iteration",
    )
    for name in ("success", "initial_success", "terminated"):
        _require(type(row[name]) is bool, f"Attempt {name} must be boolean")
    _require(
        not row["initial_success"], "Initially successful resets cannot be counted"
    )
    actions = _count(row["actions_executed"], "actions", protocol["action_budget"])
    _require(actions > 0, "A completed physical rollout requires actions")
    _count(row["velocity_evaluations"], "rollout velocity evaluations")
    _seconds(row["wall_seconds"], "rollout wall seconds")
    _require(
        row["status"]
        == (
            "success"
            if row["success"]
            else "terminated"
            if row["terminated"]
            else "budget_exhausted"
        ),
        "Attempt status differs from its observed outcome",
    )
    _require(
        row["success"] or row["terminated"] or actions == protocol["action_budget"],
        "An uncensored failed rollout ended before its action cap",
    )
    generations = (actions + protocol["execute_steps"] - 1) // protocol["execute_steps"]
    _require(
        row["policy_replans"] == generations,
        "Replan count differs from actual action coverage",
    )
    for key in (
        "actions_with_accepted_decision",
        "actions_with_changed_image",
        "native_condition_fallback_actions",
    ):
        _count(row[key], key, actions)
    for key in (
        "image_active_policy_calls",
        "image_changed_policy_calls",
        "accepted_decision_policy_calls",
        "native_condition_fallback_policy_calls",
    ):
        _count(row[key], key, generations)
    decisions, calls = row["decisions"], row["provider_records"]
    _require(
        isinstance(decisions, list) and isinstance(calls, list),
        "Attempt decisions/calls must be lists",
    )
    image_arm = mode in (
        "astra_occlusion",
        "astra_demo_blend",
        "random_occlusion",
        "random_demo_blend",
    )
    expected_decisions = (
        (actions - 1) // protocol["astra"]["call_interval"] + 1 if image_arm else 0
    )
    _require(
        len(decisions) == expected_decisions,
        "Missing or extra scheduled image decision",
    )
    _require(
        len(decisions) <= protocol["astra"]["max_calls_per_rollout"],
        "Decision cap exceeded",
    )
    for index, decision in enumerate(decisions, 1):
        _require(
            decision["decision_index"] == index
            and decision["observation_step"]
            == (index - 1) * protocol["astra"]["call_interval"]
            and type(decision["accepted"]) is bool,
            "Decision ordering or schedule differs",
        )
        _require(
            (decision["proposal"] is not None and decision["error"] is None)
            if decision["accepted"]
            else (decision["proposal"] is None and isinstance(decision["error"], str)),
            "Decision acceptance/proposal/error disagree",
        )
    accepted_ids = [
        decision["proposal"]["decision_id"]
        for decision in decisions
        if decision["accepted"]
    ]
    _require(len(accepted_ids) == len(set(accepted_ids)), "Duplicate decision identity")
    _same(
        row["applied_accepted_decision_ids"],
        accepted_ids,
        "Executed accepted decision IDs",
    )
    _require(
        row["accepted_decisions_executed"] == len(accepted_ids),
        "Executed decision count differs",
    )
    if mode.startswith("astra_"):
        _require(
            len(calls) == len(decisions),
            "Each Astra slot requires exactly one provider/preflight ledger row",
        )
        for decision, call in zip(decisions, calls, strict=True):
            for name, expected in (
                ("episode_id", episode_id),
                ("attempt_id", row["attempt_id"]),
                ("decision_index", decision["decision_index"]),
                ("observation_step", decision["observation_step"]),
                ("image_mode", mode.removeprefix("astra_")),
                ("requested_model", protocol["astra"]["model"]),
                ("accepted", decision["accepted"]),
            ):
                _require(call.get(name) == expected, f"Provider {name} binding differs")
            if call["provider_call"]:
                _same(
                    call["sampling_settings"],
                    {
                        key: protocol["astra"][key]
                        for key in ("reasoning_effort", "max_completion_tokens")
                    },
                    "Provider sampling",
                )
                _same(call["cache"], {"no-cache": True}, "Provider cache")
                _same(
                    call["response_format"],
                    {"type": "json_object"},
                    "Provider JSON mode",
                )
            if decision["accepted"]:
                proposal = decision["proposal"]
                _require(
                    call["provider_call"] is True
                    and call["response"]["model"] == protocol["astra"]["model"],
                    "Accepted decision lacks exact physical provider identity",
                )
                _require(
                    call["decision_id"] == proposal["decision_id"]
                    and call["request_fingerprint"] == proposal["request_fingerprint"],
                    "Accepted decision/provider binding differs",
                )
                wire = _strict_json(
                    call["response"]["choices"][0]["message"]["content"]
                )
                _same(
                    wire,
                    {
                        key: value
                        for key, value in proposal.items()
                        if key != "request_fingerprint"
                    },
                    "Accepted provider response",
                )
        _require(
            row["actions_with_accepted_decision"]
            + row["native_condition_fallback_actions"]
            == actions,
            "Astra accepted/fallback action coverage differs",
        )
        _require(
            row["actions_with_changed_image"] <= row["actions_with_accepted_decision"],
            "Changed images lack accepted steering",
        )
    else:
        _require(not calls, "A control cannot contain Astra calls")
        _require(
            row["native_condition_fallback_actions"] == 0,
            "A control cannot claim provider fallback",
        )
    return summarize_calls(calls)


def _audit_binding(receipt, report, summary_sha, physical, *, receipt_sha=None):
    _require(isinstance(receipt, dict), "Audit receipt must be an object")
    _require(
        receipt.get("schema_version") == "image-perturbation-audit-1.0"
        and receipt.get("status") == "passed"
        and receipt.get("complete") is True
        and receipt.get("all_arrays_verified") is True
        and receipt.get("provider_bindings_verified") is True
        and receipt.get("reset_pairing_verified") is True
        and receipt.get("decision_lifetimes_verified") is True,
        "Independent case audit did not pass completely",
    )
    expected = {
        "episode_id": report["episode_id"],
        "protocol_sha256": report["protocol_sha256"],
        "library_id": report["image_library"]["library_id"],
        "summary_sha256": summary_sha,
        "reset_entry_sha256": report["reset_entry_sha256"],
    }
    for name, value in expected.items():
        _require(receipt.get(name) == value, f"Audit {name} binding differs")
    _same(receipt["provider"], physical["token_usage"], "Audited provider accounting")
    expected_counts = {
        "physical_rollouts": physical["rollouts"],
        "actions": physical["simulated_actions"],
        "velocity_evaluations": physical["velocity_evaluations"],
        "provider_calls": physical["token_usage"]["provider_calls"],
        "preflight_failures": physical["token_usage"]["preflight_failures"],
        "accepted_proposals": physical["token_usage"]["accepted_proposals"],
        "executed_decisions": sum(
            row["accepted_decisions_executed"]
            for row in _physical(report, list(report["arms"]))
        ),
    }
    for name, value in expected_counts.items():
        _same(receipt["counts"][name], value, f"Audited {name}")
    inputs = _artifact_hashes(receipt["input_file_sha256"])
    sources = _artifact_hashes(receipt["source_sha256"])
    return {
        "status": "passed",
        "receipt_digest": digest(receipt),
        "receipt_sha256": receipt_sha,
        "summary_sha256": summary_sha,
        "events_sha256": receipt.get("events_sha256"),
        "array_inventory_sha256": receipt.get("array_inventory_sha256"),
        "all_arrays_verified": True,
        "counts": receipt.get("counts"),
        "input_file_inventory_digest": digest(inputs),
        "input_file_count": len(inputs),
        "source_sha256": sources,
    }


def _case(directory, protocol, receipt):
    path = Path(directory) / "summary.json"
    report, summary_sha = _read(path), file_sha256(path)
    _require(
        report["schema_version"] == "image-perturbations-1.0", "Wrong summary schema"
    )
    _require(report["status"] == "complete", "Case is not complete")
    _require(
        report["protocol_sha256"] == digest(protocol),
        "Case protocol fingerprint differs",
    )
    _require(report["seed"] == protocol["seed"], "Case protocol seed differs")
    _require(
        report["image_library"]["library_id"] == protocol["image_library_id"],
        "Case donor library differs",
    )
    _require(type(report["development"]) is bool, "Development flag must be boolean")
    _require(
        report["initialization"]["passed"] is True
        and report["image_gate"]["passed"] is True,
        "Native/inversion/image gate failed",
    )
    if report["development"]:
        _require(
            report["development_validation"]["passed"] is True,
            "Development provider integration gate failed",
        )
    arms = protocol["arms"]
    _require(
        set(report["arms"]) == set(arms) and set(report["controls"]) == set(CONTROLS),
        "Case arm/control coverage differs",
    )
    baseline = report["baseline"]
    _validate_attempt(
        baseline,
        mode="recovered_noise",
        iteration=1,
        episode_id=report["episode_id"],
        protocol=protocol,
    )
    for control in CONTROLS:
        _validate_attempt(
            report["controls"][control],
            mode=control,
            iteration=1,
            episode_id=report["episode_id"],
            protocol=protocol,
        )
    initialization = _count(
        report["initialization"]["velocity_evaluations"], "initialization VF"
    )
    gate = _count(report["image_gate"]["velocity_evaluations"], "image gate VF")
    arm_rows = []
    for arm in arms:
        value = report["arms"][arm]
        attempts = value["attempts"]
        _require(
            isinstance(attempts, list)
            and 1 <= len(attempts) <= protocol["attempt_budget"],
            "Invalid rollout attempt budget",
        )
        _same(attempts[0], baseline, "Common baseline copy")
        for iteration, attempt in enumerate(attempts[1:], 2):
            _validate_attempt(
                attempt,
                mode=arm,
                iteration=iteration,
                episode_id=report["episode_id"],
                protocol=protocol,
            )
        first = next((row for row in attempts if row["success"]), None)
        if baseline["success"]:
            _require(
                len(attempts) == (2 if report["development"] else 1),
                "Baseline stopping/forced development rule differs",
            )
        elif first is not None:
            _require(attempts[-1] is first, "Search continued after its first success")
        else:
            _require(
                len(attempts) == protocol["attempt_budget"],
                "Unsuccessful search did not exhaust its declared cap",
            )
        summary = arm_summary(attempts, protocol["attempt_budget"])
        summary["standalone_velocity_evaluations_through_success_or_cap"] = (
            summary["velocity_evaluations_through_success_or_cap"]
            + initialization
            + gate
        )
        _same(value["summary"], summary, "Arm outcome/cost summary")
        winning = first or {}
        arm_rows.append(
            {
                "episode_id": report["episode_id"],
                "suite": report["suite"],
                "task_id": report["task_id"],
                "seed": report["seed"],
                "arm": arm,
                "baseline_success": baseline["success"],
                "success": first is not None,
                "rescued": not baseline["success"] and first is not None,
                "censored_without_success": first is None,
                "first_success_attempt": summary["first_success_attempt"],
                "full_rollout_revisions_to_success": summary[
                    "full_rollout_revisions_to_success"
                ],
                "within_successful_rollout_decisions": summary[
                    "within_successful_rollout_decisions"
                ],
                "online_decisions_through_success_or_cap": summary[
                    "decisions_through_success_or_cap"
                ],
                "actions_through_success_or_cap": summary[
                    "actions_through_success_or_cap"
                ],
                "rollout_seconds_through_success_or_cap": summary[
                    "rollout_seconds_through_success_or_cap"
                ],
                "standalone_velocity_evaluations_through_success_or_cap": summary[
                    "standalone_velocity_evaluations_through_success_or_cap"
                ],
                "provider_through_success_or_cap": summary[
                    "provider_through_success_or_cap"
                ],
                "complete_tokens_to_success": _tokens_value(
                    summary["tokens_to_first_success"]
                )
                if first is not None
                else None,
                "physical_provider": summary["provider"],
                "development_extra_rollouts_after_success": summary[
                    "development_extra_rollouts_after_success"
                ],
                "success_by_attempt": summary["success_by_attempt"],
                "winning_attempt_effects": {
                    key: winning.get(key)
                    for key in (
                        "attempt_id",
                        "actions_executed",
                        "accepted_decisions_executed",
                        "actions_with_accepted_decision",
                        "actions_with_changed_image",
                        "native_condition_fallback_actions",
                    )
                }
                if first is not None
                else None,
            }
        )
    physical_attempts = _physical(report, arms)
    _require(
        len({row["attempt_id"] for row in physical_attempts}) == len(physical_attempts),
        "Duplicate physical rollout identity",
    )
    physical = _physical_cost(physical_attempts, initialization, gate)
    for name, value in physical.items():
        _same(report["physical_cost"][name], value, f"Physical {name}")
    audit = {"status": "pending"}
    if receipt is not None:
        receipt_sha = None
        if isinstance(receipt, (str, Path)):
            receipt_sha = file_sha256(receipt)
            receipt = _read(receipt)
        audit = _audit_binding(
            receipt, report, summary_sha, physical, receipt_sha=receipt_sha
        )
    calls = [call for row in physical_attempts for call in row["provider_records"]]
    case = {
        "episode_id": report["episode_id"],
        "suite": report["suite"],
        "task_id": report["task_id"],
        "seed": report["seed"],
        "development": report["development"],
        "baseline_success": baseline["success"],
        "summary_sha256": summary_sha,
        "protocol_sha256": report["protocol_sha256"],
        "library_id": report["image_library"]["library_id"],
        "checkpoint_digest": digest(report["checkpoint"]),
        "reset_entry_sha256": report["reset_entry_sha256"],
        "audit": audit,
        "physical_cost": physical,
        "arm_rows": arm_rows,
        "case_wall_seconds": _seconds(report["total_wall_seconds"], "case wall time"),
        "controls": {
            mode: {
                "success": report["controls"][mode]["success"],
                "actions": report["controls"][mode]["actions_executed"],
            }
            for mode in CONTROLS
        },
        "rejected_call_usage": summarize_calls(
            [call for call in calls if not call["accepted"]]
        ),
        "fallback_actions": sum(
            row["native_condition_fallback_actions"] for row in physical_attempts
        ),
        "changed_image_actions": sum(
            row["actions_with_changed_image"] for row in physical_attempts
        ),
        "accepted_decision_actions": sum(
            row["actions_with_accepted_decision"] for row in physical_attempts
        ),
        "physical_attempt_rows": [
            {
                "episode_id": report["episode_id"],
                "suite": report["suite"],
                "attempt_id": row["attempt_id"],
                "mode": row["mode"],
                "iteration": row["iteration"],
                "success": row["success"],
                "actions": row["actions_executed"],
                "velocity_evaluations": row["velocity_evaluations"],
                "wall_seconds": row["wall_seconds"],
                "decisions": len(row["decisions"]),
                "accepted_decisions_executed": row["accepted_decisions_executed"],
                "accepted_decision_actions": row["actions_with_accepted_decision"],
                "changed_image_actions": row["actions_with_changed_image"],
                "fallback_actions": row["native_condition_fallback_actions"],
                "provider": summarize_calls(row["provider_records"]),
            }
            for row in physical_attempts
        ],
    }
    return case, calls


def _arm_group(rows, budget):
    rescued = [row for row in rows if row["rescued"]]
    complete_tokens = [
        row["complete_tokens_to_success"]
        for row in rescued
        if row["complete_tokens_to_success"] is not None
    ]
    return {
        "cases": len(rows),
        "successes": sum(row["success"] for row in rows),
        "baseline_successes": sum(row["baseline_success"] for row in rows),
        "rescues": len(rescued),
        "censored_without_success": sum(
            row["censored_without_success"] for row in rows
        ),
        "successes_by_attempt": [
            sum(row["success_by_attempt"][index] for row in rows)
            for index in range(budget)
        ],
        "rescue_only": {
            "first_success_attempt": _stats(
                [row["first_success_attempt"] for row in rescued]
            ),
            "full_rollout_revisions": _stats(
                [row["full_rollout_revisions_to_success"] for row in rescued]
            ),
            "online_decisions_through_success": _stats(
                [row["online_decisions_through_success_or_cap"] for row in rescued]
            ),
            "within_winning_rollout_decisions": _stats(
                [row["within_successful_rollout_decisions"] for row in rescued]
            ),
            "physical_provider_calls_through_success": _stats(
                [
                    row["provider_through_success_or_cap"]["provider_calls"]
                    for row in rescued
                ]
            ),
            "rollout_seconds_through_success": _stats(
                [row["rollout_seconds_through_success_or_cap"] for row in rescued]
            ),
            "complete_total_tokens_to_success": _stats(complete_tokens),
            "rescues_with_incomplete_token_usage": len(rescued) - len(complete_tokens),
        },
        "censored_episode_ids": [
            row["episode_id"] for row in rows if row["censored_without_success"]
        ],
        "rescue_episode_ids": [row["episode_id"] for row in rescued],
    }


def _paired(cases, astra, random):
    rows = []
    for case in cases:
        arms = {row["arm"]: row for row in case["arm_rows"]}
        a, b = arms[astra], arms[random]
        rows.append(
            {
                "episode_id": case["episode_id"],
                "baseline_success": case["baseline_success"],
                "astra_success": a["success"],
                "random_success": b["success"],
                "astra_first_success_attempt": a["first_success_attempt"],
                "random_first_success_attempt": b["first_success_attempt"],
                "astra_complete_tokens_to_success": a["complete_tokens_to_success"],
                "astra_winning_attempt_effects": a["winning_attempt_effects"],
            }
        )
    failures = [row for row in rows if not row["baseline_success"]]
    return {
        "astra_arm": astra,
        "matched_random_arm": random,
        "paired_cases": len(rows),
        "baseline_failures": len(failures),
        "on_failed_baselines": {
            "both_rescue": sum(
                row["astra_success"] and row["random_success"] for row in failures
            ),
            "astra_only_rescue": sum(
                row["astra_success"] and not row["random_success"] for row in failures
            ),
            "random_only_rescue": sum(
                not row["astra_success"] and row["random_success"] for row in failures
            ),
            "neither_rescues": sum(
                not row["astra_success"] and not row["random_success"]
                for row in failures
            ),
        },
        "rows": rows,
    }


def _group(cases, calls, protocol, release):
    physical = {
        name: sum(case["physical_cost"][name] for case in cases)
        for name in (
            "rollouts",
            "simulated_actions",
            "rollout_velocity_evaluations",
            "initialization_velocity_evaluations",
            "image_gate_velocity_evaluations",
            "velocity_evaluations",
            "rollout_wall_seconds",
        )
    }
    physical.update(
        token_usage=summarize_calls(calls),
        rejected_call_usage=summarize_calls(
            [call for call in calls if not call["accepted"]]
        ),
        case_wall_seconds_sum=sum(case["case_wall_seconds"] for case in cases),
        fallback_actions=sum(case["fallback_actions"] for case in cases),
        changed_image_actions=sum(case["changed_image_actions"] for case in cases),
        accepted_decision_actions=sum(
            case["accepted_decision_actions"] for case in cases
        ),
    )
    return {
        "recorded_completed_cases": len(cases),
        "physical_cost": physical,
        "efficacy_released": release,
        "arms": {
            arm: _arm_group(
                [
                    row
                    for case in cases
                    for row in case["arm_rows"]
                    if row["arm"] == arm
                ],
                protocol["attempt_budget"],
            )
            for arm in protocol["arms"]
        }
        if release
        else None,
        "paired_vs_random": {
            astra: _paired(cases, astra, random) for astra, random in PAIRS
        }
        if release
        else None,
    }


def build_report(
    case_directories, *, expected_episode_ids, protocol, audit_receipts=None
):
    """Reconcile completed cases; publish efficacy only with exact audited coverage.

    ``audit_receipts`` maps episode ID to a safe ``audit_case`` JSON receipt or
    its path. Missing/failed audits and incomplete coverage remain explicit;
    they never silently reduce the denominator or establish a paired win.
    """
    expected = list(expected_episode_ids)
    _require(
        expected
        and all(isinstance(value, str) and value for value in expected)
        and len(set(expected)) == len(expected),
        "Expected episode IDs must be explicit and unique",
    )
    _require(
        set(protocol["arms"]) >= {value for pair in PAIRS for value in pair},
        "Both matched image comparisons must be declared",
    )
    receipts = audit_receipts or {}
    _require(
        isinstance(receipts, dict) and not set(receipts) - set(expected),
        "Audit receipt inventory includes unexpected episodes",
    )
    cases, calls_by_case, seen, problems = [], {}, set(), []
    for directory in case_directories:
        path = Path(directory) / "summary.json"
        try:
            preliminary = _read(path)
            episode_id = preliminary["episode_id"]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            problems.append(
                {
                    "case_directory": Path(directory).name,
                    "status": "missing_or_invalid_summary",
                    "error_type": type(exc).__name__,
                }
            )
            continue
        _require(
            episode_id in expected and episode_id not in seen,
            "Unexpected or duplicate case episode ID",
        )
        seen.add(episode_id)
        if preliminary.get("status") != "complete":
            problems.append(
                {
                    "episode_id": episode_id,
                    "status": "incomplete_case",
                    "recorded_status": preliminary.get("status"),
                }
            )
            continue
        try:
            case, calls = _case(directory, protocol, None)
        except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
            problems.append(
                {
                    "episode_id": episode_id,
                    "status": "invalid_case",
                    "error": _safe_error(exc),
                }
            )
            continue
        receipt = receipts.get(episode_id)
        if receipt is not None:
            try:
                receipt_sha = (
                    file_sha256(receipt) if isinstance(receipt, (str, Path)) else None
                )
                value = _read(receipt) if isinstance(receipt, (str, Path)) else receipt
                case["audit"] = _audit_binding(
                    value,
                    preliminary,
                    case["summary_sha256"],
                    case["physical_cost"],
                    receipt_sha=receipt_sha,
                )
            except (OSError, ValueError, KeyError, TypeError) as exc:
                case["audit"] = {"status": "failed", "error": _safe_error(exc)}
                problems.append(
                    {
                        "episode_id": episode_id,
                        "status": "audit_failed",
                        "error": _safe_error(exc),
                    }
                )
        cases.append(case)
        calls_by_case[episode_id] = calls
    cases.sort(key=lambda row: expected.index(row["episode_id"]))
    complete_ids = {case["episode_id"] for case in cases}
    coverage = complete_ids == set(expected)
    passed = [
        case["episode_id"] for case in cases if case["audit"]["status"] == "passed"
    ]
    if (
        len({case["checkpoint_digest"] for case in cases}) > 1
        or len({case["development"] for case in cases}) > 1
    ):
        problems.append(
            {
                "status": "incompatible_case_group",
                "error": "Checkpoint or development/evaluation identity differs across cases",
            }
        )
    release = coverage and len(passed) == len(expected) and not problems
    failed = any(
        row["status"] in ("invalid_case", "audit_failed", "incompatible_case_group")
        for row in problems
    )
    status = (
        "complete"
        if release
        else "audit_failed"
        if failed
        else "partial"
        if not coverage
        else "audit_pending"
    )
    calls = [call for case in cases for call in calls_by_case[case["episode_id"]]]
    groups = {"pooled": _group(cases, calls, protocol, release)}
    for suite in sorted({case["suite"] for case in cases}):
        selected = [case for case in cases if case["suite"] == suite]
        groups[suite] = _group(
            selected,
            [call for case in selected for call in calls_by_case[case["episode_id"]]],
            protocol,
            release,
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "complete": release,
        "protocol_sha256": digest(protocol),
        "image_library_id": protocol["image_library_id"],
        "expected_episode_ids": expected,
        "expected_cases": len(expected),
        "completed_cases": len(cases),
        "complete_episode_coverage": coverage,
        "audited_cases": len(passed),
        "missing_or_incomplete_episode_ids": [
            value for value in expected if value not in complete_ids
        ],
        "audit_pending_episode_ids": [
            case["episode_id"] for case in cases if case["audit"]["status"] == "pending"
        ],
        "efficacy_released": release,
        "problems": problems,
        "groups": groups,
        "cases": cases,
        "case_arm_rows": [row for case in cases for row in case["arm_rows"]],
        "physical_attempt_rows": [
            row for case in cases for row in case["physical_attempt_rows"]
        ],
        "provider_error_kinds": dict(
            Counter(
                call.get("error_kind", "unspecified")
                for call in calls
                if not call["accepted"]
            )
        ),
        "limitations": LIMITATIONS,
        "source_sha256": {
            name: file_sha256(Path(__file__).with_name(name))
            for name in (
                "image_perturbation_report.py",
                "image_perturbation_search.py",
                "image_perturbation_agent.py",
            )
        },
    }


def _flat_rows(rows):
    return [
        {
            key: json.dumps(value, sort_keys=True, allow_nan=False)
            if isinstance(value, (dict, list))
            else value
            for key, value in row.items()
        }
        for row in rows
    ]


def write_outputs(directory, report):
    """Write safe JSON, complete row tables and a small audit-status-first report."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    for name, rows in (
        ("case_arms.csv", report["case_arm_rows"]),
        ("physical_attempts.csv", report["physical_attempt_rows"]),
    ):
        with (directory / name).open("w", newline="") as stream:
            if rows:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(_flat_rows(rows))
    cost = report["groups"]["pooled"]["physical_cost"]
    usage = cost["token_usage"]
    token_count = _tokens_value(usage)
    token_text = (
        f"{token_count:,}"
        if token_count is not None
        else f"at least {usage['tokens']['total_tokens']['sum']:,}; {usage['tokens']['total_tokens']['missing_calls']} calls have unknown totals"
    )
    lines = [
        f"Image perturbation report: {report['status']}",
        "",
        f"Completed {report['completed_cases']}/{report['expected_cases']} cases; independent audits passed for {report['audited_cases']}. Efficacy released: {report['efficacy_released']}.",
        "",
        f"Recorded completed-case physical work: {cost['rollouts']} rollouts, {cost['simulated_actions']:,} actions, {cost['velocity_evaluations']:,} velocity evaluations; {usage['provider_calls']} provider calls ({usage['failed_calls']} rejected), {token_text} tokens. Preflight failures: {usage['preflight_failures']}.",
        "",
    ]
    if report["efficacy_released"]:
        lines += [
            "| Arm | Successes | Baseline wins | Rescues | Censored | Rescue revisions median [min,max] | Rescue tokens median [min,max] |",
            "|---|---:|---:|---:|---:|---|---|",
        ]
        for arm, row in report["groups"]["pooled"]["arms"].items():
            revision = row["rescue_only"]["full_rollout_revisions"]
            tokens = row["rescue_only"]["complete_total_tokens_to_success"]
            lines.append(
                f"| {arm} | {row['successes']}/{row['cases']} | {row['baseline_successes']} | {row['rescues']} | {row['censored_without_success']} | {revision['median']} [{revision['minimum']},{revision['maximum']}] | {tokens['median']} [{tokens['minimum']},{tokens['maximum']}] |"
            )
        lines += [
            "",
            "Matched comparisons use every failed-baseline pair, including capped failures:",
            "",
        ]
        for pair in report["groups"]["pooled"]["paired_vs_random"].values():
            counts = pair["on_failed_baselines"]
            lines.append(
                f"- {pair['astra_arm']} vs {pair['matched_random_arm']}: both rescue {counts['both_rescue']}, Astra only {counts['astra_only_rescue']}, random only {counts['random_only_rescue']}, neither {counts['neither_rescues']}."
            )
    else:
        lines += [
            "Pooled efficacy and paired comparisons are withheld until expected coverage and independent case audits pass."
        ]
    lines += ["", *[f"- {note}" for note in LIMITATIONS]]
    (directory / "README.md").write_text("\n".join(lines) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_directories", nargs="+")
    parser.add_argument("--protocol", required=True)
    parser.add_argument(
        "--expected-episodes", required=True, help="JSON list of exact episode IDs"
    )
    parser.add_argument(
        "--audit-receipts",
        help="JSON object mapping episode IDs to receipts or receipt paths",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    report = build_report(
        args.case_directories,
        expected_episode_ids=_read(args.expected_episodes),
        protocol=_read(args.protocol),
        audit_receipts=_read(args.audit_receipts) if args.audit_receipts else None,
    )
    write_outputs(args.output, report)
    return 0 if report["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
