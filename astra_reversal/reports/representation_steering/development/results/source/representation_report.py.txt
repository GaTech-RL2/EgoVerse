"""Offline, audit-gated representation outcomes and physical costs.

No model, simulator or provider calls are made. Complete numerical/archive and
provider audits must bind every prescribed case before aggregate efficacy is
released. Partial inputs retain recorded costs, with unrecorded work unknown.
"""

import argparse
import copy
import csv
import hashlib
import html
import json
import math
import statistics
from collections import Counter
from pathlib import Path

from .astra_client import _strict_json
from .intervention_agent import normalize_usage
from .records import digest, file_sha256
from .representation_agent import summarize_calls
from .representation_search import ARMS, load_protocol

SCHEMA_VERSION = "representation-report-1.0"
ACTION_COUNTS = (
    "accepted_decision_actions",
    "explicit_native_actions",
    "nonzero_intervention_actions",
    "provider_failure_fallback_actions",
)
TIMES = (
    "wall_seconds",
    "policy_seconds",
    "environment_seconds",
    "condition_seconds",
    "donor_capture_seconds",
)
LIMITATIONS = [
    "Exploratory assisted correction on known task compositions and prescribed resets; no autonomous learning or held-out-task claim.",
    "Every arm shares one native baseline. Success at revision zero is not an intervention rescue; failures remain in the fixed denominator.",
    "Revision budgets count full reset rollouts. An online decision every 25 actions is a different unit; both are reported.",
    "Policy noise is matched across arms at a case/revision/step and changes across revisions. Native retries measure gains available from that new noise alone.",
    "Random controls use the same source catalogs and continuous alpha bounds. Astra's explicit deferral distribution is not reproduced by random controls.",
    "Recorded representation changes and accepted proposals are not proofs of action causation. A success can include neutral choices or native fallback after a failed call.",
    "Physical totals count the shared baseline and numerical probes once. Standalone arm attribution repeats shared setup and must never be summed across arms.",
    "Rollout wall time includes provider waits, donor captures, probes, recording and simulator work. Policy, environment, conditioning, capture and provider times are nested diagnostics, not extra wall time.",
    "All physical calls, including rejected calls, contribute available provider usage. Missing usage stays unknown; known sums are lower bounds. Reasoning tokens are a subset of output. No dollar cost is assumed.",
    "Rescue-only medians exclude baseline successes and censored failures. Failed searches retain their full recorded token and rollout costs.",
    "Development may force a revision after baseline success. Those extra physical costs are reported separately and do not improve the success curve.",
    "Partial costs cover recorded work only. In-flight, interrupted, setup and donor-extraction work are not silently assigned zero cost or pooled into the completed cohort.",
    "Independent audits bind archives, arrays, provider feedback, resets and frozen weights. This report does not independently replay model hidden states or simulator physics.",
    "Videos contain raw external-camera frames before each executed action, at 20 fps; they omit inference pauses, stabilization and the terminal post-action image.",
]


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _read(path):
    return _strict_json(Path(path).read_bytes())


def _same(actual, expected, label):
    _require(digest(actual) == digest(expected), f"{label} differs")


def _count(value, label, maximum=None):
    _require(type(value) is int and value >= 0, f"Invalid {label}")
    _require(maximum is None or value <= maximum, f"{label} exceeds its bound")
    return value


def _seconds(value, label):
    _require(
        type(value) in (int, float) and math.isfinite(value) and value >= 0,
        f"Invalid {label}",
    )
    return value


def _stats(values):
    return {
        "count": len(values),
        "median": statistics.median(values) if values else None,
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
    }


def expected_episodes(protocol, phase):
    """Return the fixed cohort, not an outcome-dependent selected subset."""
    _require(phase in ("development", "evaluation"), "Phase must be explicit")
    _require(protocol["arms"] == list(ARMS), "Representation arms differ")
    _require(
        protocol["schema_version"] == "representation-steering-1.0"
        and protocol["revisions"] == 2
        and protocol["execute_steps"] == 5
        and protocol["action_budget"] == 300
        and protocol["learning"]["enabled"] is False,
        "Unsupported representation protocol",
    )
    seed = 19 if phase == "development" else 47
    _require(protocol["seed"] == seed, "Phase seed differs from the prescribed cohort")
    pairs = (
        [("libero_goal_ood", 6), ("libero_spatial_ood", 2), ("libero_spatial_ood", 8)]
        if phase == "development"
        else [(suite, task) for suite in protocol["suites"] for task in range(10)]
    )
    _require(
        set(protocol["suites"]) == {"libero_goal_ood", "libero_spatial_ood"}
        and protocol["evaluation_cases"] == [[i, 0] for i in range(10)]
        and protocol["development_cases"]
        == {"libero_goal_ood": [[6, 0]], "libero_spatial_ood": [[2, 0], [8, 0]]},
        "Prescribed task/reset inventory differs",
    )
    return [f"{suite}:seed{seed}:task{task}:state0" for suite, task in pairs]


def _calls(path):
    if not path.exists():
        return []
    return [
        _strict_json(line) for line in path.read_bytes().splitlines() if line.strip()
    ]


def _validate_attempt(row, episode_id, protocol):
    arm, revision = row["arm"], row["revision"]
    _require(arm == "native" or arm in ARMS, "Unknown physical arm")
    _count(revision, "revision", protocol["revisions"])
    _require((arm == "native") == (revision == 0), "Native baseline identity differs")
    _require(row["attempt_id"] == f"{arm}_revision{revision}", "Attempt ID differs")
    _require(row["episode_id"] == episode_id, "Attempt episode differs")
    for name in ("success", "initial_success", "terminated"):
        _require(type(row[name]) is bool, f"Attempt {name} must be boolean")
    _require(not row["initial_success"], "Initially successful reset is invalid")
    actions = _count(row["actions_executed"], "actions", protocol["action_budget"])
    _require(actions > 0, "Completed rollout must execute actions")
    _require(
        row["success"] or row["terminated"] or actions == protocol["action_budget"],
        "Unterminated failed rollout ended before its cap",
    )
    status = (
        "success"
        if row["success"]
        else "terminated"
        if row["terminated"]
        else "budget_exhausted"
    )
    _require(row["status"] == status, "Recorded status differs from outcome")
    count = math.ceil(actions / protocol["execute_steps"])
    _require(row["policy_replans"] == count, "Policy replan count differs")
    _require(
        row["velocity_evaluations"] == count * protocol["solver"]["steps"],
        "Flow cost differs",
    )
    for name in (
        "parity_velocity_evaluations",
        "probe_velocity_evaluations",
        "donor_captures",
    ):
        _count(row[name], name)
    if revision:
        _require(
            row["parity_velocity_evaluations"]
            == row["probe_velocity_evaluations"]
            == 0,
            "Repeated setup cost",
        )
    for name in TIMES:
        _seconds(row[name], name)
    decisions, calls, generations = (
        row["decisions"],
        row["provider_records"],
        row["generations"],
    )
    expected_decisions = (
        0
        if arm in ("native", "native_retry")
        else math.ceil(actions / protocol["astra"]["call_interval"])
    )
    _require(
        len(decisions) == expected_decisions, "Scheduled decision coverage differs"
    )
    _require(len(generations) == count, "Generation coverage differs")
    for index, decision in enumerate(decisions, 1):
        _require(
            decision["decision_index"] == index
            and decision["observation_step"]
            == (index - 1) * protocol["astra"]["call_interval"],
            "Decision cadence differs",
        )
        _require(
            type(decision["accepted"]) is bool, "Decision accepted must be boolean"
        )
        _require(
            (decision["proposal"] is not None and decision["error"] is None)
            if decision["accepted"]
            else (decision["proposal"] is None and isinstance(decision["error"], str)),
            "Decision acceptance/error differs",
        )
    if arm.startswith("astra_"):
        _require(
            len(calls) == len(decisions), "Astra decisions require one ledger row each"
        )
        for decision, call in zip(decisions, calls, strict=True):
            for name, value in {
                "episode_id": episode_id,
                "attempt_id": row["attempt_id"],
                "decision_index": decision["decision_index"],
                "observation_step": decision["observation_step"],
                "representation_mode": arm.removeprefix("astra_"),
                "requested_model": protocol["astra"]["model"],
                "accepted": decision["accepted"],
            }.items():
                _require(call.get(name) == value, f"Provider {name} differs")
            _same(
                call["token_usage"],
                normalize_usage(call.get("response", {}).get("usage")),
                "Provider usage",
            )
            if call["provider_call"]:
                _same(
                    call["sampling_settings"],
                    {
                        k: protocol["astra"][k]
                        for k in ("reasoning_effort", "max_completion_tokens")
                    },
                    "Sampling settings",
                )
                _same(call["cache"], {"no-cache": True}, "Provider cache")
            if decision["accepted"]:
                _require(
                    call["provider_call"] is True
                    and call["response"]["model"] == protocol["astra"]["model"],
                    "Accepted provider identity differs",
                )
                for name in ("decision_id", "request_fingerprint"):
                    _require(
                        call[name] == decision["proposal"][name],
                        f"Accepted {name} differs",
                    )
    else:
        _require(
            not calls and all(d["accepted"] for d in decisions),
            "Control has provider calls/rejections",
        )
    summarize_calls(calls)
    counts = Counter({name: 0 for name in ACTION_COUNTS})
    accepted_ids, effective_ids = set(), set()
    for index, generation in enumerate(generations):
        step = index * protocol["execute_steps"]
        _require(generation["step"] == step, "Generation steps differ")
        for flag in ("has_effect", "provider_failure_fallback"):
            _require(
                type(generation[flag]) is bool, f"Generation {flag} must be boolean"
            )
        decision = (
            decisions[step // protocol["astra"]["call_interval"]] if decisions else None
        )
        expected_choice = decision["proposal"] if decision else None
        _same(generation["choice"], expected_choice, "Applied decision lifetime")
        fallback = bool(
            arm.startswith("astra_") and decision and not decision["accepted"]
        )
        _require(
            generation["provider_failure_fallback"] == fallback,
            "Failed decision did not clear immediately",
        )
        size = min(protocol["execute_steps"], actions - step)
        choice = generation["choice"]
        if choice is None or choice["mode"] == "native":
            _require(
                not generation["has_effect"],
                "Native choice claims representation effect",
            )
        if arm.startswith("astra_") and choice is not None:
            accepted_ids.add(choice["decision_id"])
            if generation["has_effect"]:
                effective_ids.add(choice["decision_id"])
            counts["accepted_decision_actions"] += size
            counts["explicit_native_actions"] += size * (choice["mode"] == "native")
        counts["nonzero_intervention_actions"] += size * generation["has_effect"]
        counts["provider_failure_fallback_actions"] += size * fallback
    for name in ACTION_COUNTS:
        _require(
            _count(row[name], name, actions) == counts[name], f"Executed {name} differs"
        )
    _require(
        row["accepted_decisions_executed"] == len(accepted_ids),
        "Executed accepted count differs",
    )
    return {
        **dict(counts),
        "accepted_decisions_executed": len(accepted_ids),
        "accepted_decisions_with_recorded_effect": len(effective_ids),
        "accepted_decisions_without_recorded_effect": len(accepted_ids - effective_ids),
        "explicit_native_decisions": sum(
            d["accepted"] and d["proposal"]["mode"] == "native" for d in decisions
        ),
        "interpolation_decisions": sum(
            d["accepted"] and d["proposal"]["mode"] == "interpolate" for d in decisions
        ),
        "rejected_decisions": sum(not d["accepted"] for d in decisions),
    }


def _cost(rows, calls=None):
    calls = (
        calls if calls is not None else [c for r in rows for c in r["provider_records"]]
    )
    return {
        "rollouts": len(rows),
        "actions": sum(r["actions_executed"] for r in rows),
        "flow_velocity_evaluations": sum(r["velocity_evaluations"] for r in rows),
        "native_parity_velocity_evaluations": sum(
            r["parity_velocity_evaluations"] for r in rows
        ),
        "probe_velocity_evaluations": sum(
            r["probe_velocity_evaluations"] for r in rows
        ),
        "velocity_evaluations": sum(
            r["velocity_evaluations"]
            + r["parity_velocity_evaluations"]
            + r["probe_velocity_evaluations"]
            for r in rows
        ),
        "provider": summarize_calls(calls),
        "rejected_call_usage": summarize_calls([c for c in calls if not c["accepted"]]),
        "provider_error_taxonomy": {
            "budget_exceeded": sum(
                c.get("provider_error", {}).get("type") == "budget_exceeded"
                for c in calls
            ),
            "http_429": sum(c.get("http_status") == 429 for c in calls),
            "transport_error": sum(
                c.get("error_kind") == "transport_error" for c in calls
            ),
            "proposal_rejected": sum(
                c.get("error_kind") == "proposal_rejected" for c in calls
            ),
        },
        "donor_captures": sum(r["donor_captures"] for r in rows),
        **{name: sum(r[name] for r in rows) for name in TIMES},
        **{name: sum(r[name] for r in rows) for name in ACTION_COUNTS},
        "accepted_decisions_executed": sum(
            r["accepted_decisions_executed"] for r in rows
        ),
    }


def _arm_rows(summary, protocol):
    baseline, physical = summary["baseline"], summary["physical_rollouts"]
    result = []
    for arm in ARMS:
        revisions = [r for r in physical if r["arm"] == arm]
        attempts = [baseline, *revisions]
        expected_revisions = []
        succeeded = baseline["success"]
        for revision in range(1, protocol["revisions"] + 1):
            if succeeded and not (summary["development"] and revision == 1):
                break
            expected_revisions.append(revision)
            if len(revisions) >= len(expected_revisions):
                succeeded = (
                    succeeded or revisions[len(expected_revisions) - 1]["success"]
                )
        _require(
            [r["revision"] for r in revisions] == expected_revisions,
            "Revision coverage/stopping differs",
        )
        first = next((r for r in attempts if r["success"]), None)
        prefix = [
            r for r in attempts if first is None or r["revision"] <= first["revision"]
        ]
        costs = _cost(prefix)
        provider = _cost(attempts)["provider"]
        curve = [
            any(r["success"] and r["revision"] <= budget for r in attempts)
            for budget in range(3)
        ]
        expected_summary = {
            "success": first is not None,
            "first_success_revision": first["revision"] if first else None,
            "censored": first is None,
            "success_by_revision": curve,
            "physical_attempt_ids": [r["attempt_id"] for r in attempts],
            "actions_through_success_or_cap": costs["actions"],
            "decisions_through_success_or_cap": sum(
                len(r["decisions"]) for r in prefix
            ),
            "provider": provider,
            "provider_through_success_or_cap": costs["provider"],
            "development_extra_rollouts": len(attempts) - len(prefix),
        }
        _same(summary["arms"][arm], expected_summary, "Arm summary")
        budget_costs = [
            _cost([r for r in prefix if r["revision"] <= budget]) for budget in range(3)
        ]
        token_total = costs["provider"]["tokens"]["total_tokens"]
        result.append(
            {
                "episode_id": summary["episode_id"],
                "suite": summary["suite"],
                "task_id": summary["task_id"],
                "arm": arm,
                "baseline_success": baseline["success"],
                "success": first is not None,
                "rescued": not baseline["success"] and first is not None,
                "censored": first is None,
                "first_success_revision": first["revision"] if first else None,
                "success_by_revision": curve,
                "winning_attempt_id": first["attempt_id"] if first else None,
                "decisions_through_success_or_cap": expected_summary[
                    "decisions_through_success_or_cap"
                ],
                "within_winning_rollout_decisions": len(first["decisions"])
                if first
                else None,
                "provider_through_success_or_cap": costs["provider"],
                "complete_tokens_to_success": token_total["sum"]
                if first and token_total["complete"]
                else None,
                "standalone_cost_through_success_or_cap": costs,
                "standalone_cost_by_revision_budget": budget_costs,
                "physical_cost_with_shared_baseline": _cost(attempts),
                "development_extra_rollouts": len(attempts) - len(prefix),
                "development_extra_cost": _cost(
                    [r for r in attempts if r not in prefix]
                ),
            }
        )
    return result


def _receipt(value):
    if isinstance(value, (str, Path)):
        return _read(value), file_sha256(value)
    _require(isinstance(value, dict), "Audit receipt must be an object or path")
    return value, None


def _source_sha(value, *, entire=False):
    if isinstance(value, dict):
        if entire:
            _require(
                value.get("entire_file") is True, "Audit covers only a file prefix"
            )
        value = value["sha256"]
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value),
        "Invalid audit file digest",
    )
    return value


def _metadata(directory, summary, protocol, phase, *, sealed=True):
    worker = directory.parent
    if not (worker / "runtime.json").is_file():
        worker = worker / "metadata"
    names = (
        "runtime.json",
        "protocol.json",
        "prompts.json",
        "checkpoint.json",
        "frozen_plan.json",
        "reset_manifest.json",
        "frozen_weights_before.json",
        "bank_inventory.json",
    )
    values = {name: _read(worker / name) for name in names}
    hashes = {name: file_sha256(worker / name) for name in names}
    runtime, plan = values["runtime.json"], values["frozen_plan.json"]
    _same(values["protocol.json"], protocol, "Worker protocol")
    _require(
        runtime["phase"] == phase
        and runtime["tf32"] is False
        and "L40S" in runtime["gpu"],
        "Worker runtime differs",
    )
    _require(
        plan["protocol_sha256"] == digest(protocol)
        and summary["episode_id"] in plan["assigned_episodes"],
        "Frozen assignment differs",
    )
    _require(
        plan["image_library_id"] == summary["image_library_id"]
        and plan["bank_inventory_sha256"] == hashes["bank_inventory.json"],
        "Donor inventory differs",
    )
    manifest = values["reset_manifest.json"]
    _require(
        plan["reset_manifest_sha256"] == manifest["sha256"],
        "Reset manifest identity differs",
    )
    entries = [
        e for e in manifest["episodes"] if e["episode_id"] == summary["episode_id"]
    ]
    _require(
        len(entries) == 1 and digest(entries[0]) == summary["reset_entry_sha256"],
        "Frozen reset entry differs",
    )
    before = values["frozen_weights_before.json"]
    result = {
        "workflow": runtime["workflow"],
        "worker": runtime["worker"],
        "worker_metadata_sha256": hashes,
        "reset_manifest_sha256": manifest["sha256"],
        "native_tensor_sha256": before["sha256"],
        "assignment": plan["assigned_episodes"],
        "weight_check_scope": "completed_seal"
        if sealed
        else "before_only_no_final_check",
        "common_identity": {
            "payload_sha256": runtime["payload_sha256"],
            "checkpoint_sha256": hashes["checkpoint.json"],
            "prompt_sha256": hashes["prompts.json"],
            "bank_inventory_sha256": hashes["bank_inventory.json"],
            "image_library_id": summary["image_library_id"],
            "native_tensor_sha256": before["sha256"],
            "packages": runtime["packages"],
            "python": runtime["python"],
        },
    }
    if not sealed:
        return result
    _same(
        _read(directory / "frozen_weights_after.json"), before, "Frozen native weights"
    )
    seal = _read(directory / "completion_receipt.json")
    _require(
        seal["schema_version"] == "representation-complete-case-1.0"
        and seal["episode_id"] == summary["episode_id"]
        and seal["workflow"] == runtime["workflow"]
        and seal["worker"] == runtime["worker"],
        "Completion seal identity differs",
    )
    _require(
        seal["native_tensor_sha256"] == before["sha256"],
        "Completion seal tensor differs",
    )
    _same(seal["worker_files"], hashes, "Sealed worker inputs")
    for name in ("summary.json", "events.jsonl", "frozen_weights_after.json"):
        _require(
            seal["case_files"][name]["sha256"] == file_sha256(directory / name),
            f"Sealed {name} differs",
        )
    if (directory / "provider.jsonl").exists():
        _require(
            seal["case_files"]["provider.jsonl"]["sha256"]
            == file_sha256(directory / "provider.jsonl"),
            "Sealed provider ledger differs",
        )
    return result


def _recorded_work(arrays, provider, physical, *, partial):
    """Keep completed rows and independently verified interrupted bounds distinct."""
    counts = arrays["counts"]
    for key, value in {
        "physical_rollouts": physical["rollouts"],
        "actions": physical["actions"],
        "native_parity_velocity_evaluations": physical[
            "native_parity_velocity_evaluations"
        ],
        "probe_velocity_evaluations": physical["probe_velocity_evaluations"],
    }.items():
        _require(counts[key] == value, f"Recorded-work {key} differs")
    flow = _count(counts["flow_velocity_evaluations"], "recorded flow evaluations")
    total = _count(counts["velocity_evaluations"], "recorded total evaluations")
    setup = (
        physical["native_parity_velocity_evaluations"]
        + physical["probe_velocity_evaluations"]
    )
    _require(
        flow >= physical["flow_velocity_evaluations"] and total == flow + setup,
        "Recorded compute bound differs",
    )
    incomplete = (
        provider["incomplete_attempts"]
        if partial
        else provider.get("incomplete_attempts", [])
    )
    _require(isinstance(incomplete, list), "Incomplete attempt bounds must be a list")
    _require(
        len({r["attempt_id"] for r in incomplete}) == len(incomplete),
        "Duplicate incomplete attempt bound",
    )
    if partial:
        _require(
            len(incomplete) == counts["inflight_attempts"],
            "Incomplete attempt coverage differs",
        )
    else:
        _require(
            not incomplete and total == physical["velocity_evaluations"],
            "Complete case contains interrupted work",
        )
    lower = sum(
        _count(r["executed_actions_lower_bound"], "unfinished actions", 300)
        for r in incomplete
    )
    upper = sum(
        _count(r["executed_actions_upper_bound"], "unfinished cap", 300)
        for r in incomplete
    )
    for row in incomplete:
        _require(
            row["executed_actions_lower_bound"]
            <= row["executed_actions_upper_bound"]
            == 300,
            "Unfinished action bounds differ",
        )
        _require(
            row["outcome"] == "unknown_no_completed_rollout_record",
            "Unfinished outcome is not unknown",
        )
    unfinished_flow = sum(
        _count(r["flow_velocity_evaluations_recorded"], "unfinished flow evaluations")
        for r in incomplete
    )
    _require(
        unfinished_flow == flow - physical["flow_velocity_evaluations"],
        "Unfinished recorded compute differs",
    )
    array_lower = counts.get("recorded_action_lower_bound", physical["actions"])
    _count(array_lower, "array action lower bound")
    _require(
        physical["actions"] <= array_lower <= physical["actions"] + lower,
        "Provider-request bound does not cover array bound",
    )
    for name, value in {
        "physical_completed_rollouts": physical["rollouts"],
        "physical_completed_actions": physical["actions"],
        "physical_completed_velocity_evaluations": physical["velocity_evaluations"],
        "recorded_flow_velocity_evaluations": flow,
        "recorded_total_velocity_evaluations_lower_bound": total,
    }.items():
        if partial or name in provider:
            _require(provider[name] == value, f"Provider {name} differs")
    if partial or "all_recorded_actions_bounds" in provider:
        _same(
            provider["all_recorded_actions_bounds"],
            {
                "lower_bound": physical["actions"] + lower,
                "upper_bound": physical["actions"] + upper,
            },
            "Provider action bounds",
        )
    return {
        "completed_rollouts": physical["rollouts"],
        "completed_actions": physical["actions"],
        "completed_velocity_evaluations": physical["velocity_evaluations"],
        "recorded_velocity_evaluations_lower_bound": total,
        "setup_velocity_evaluations": setup,
        "unfinished_recorded_flow_velocity_evaluations": unfinished_flow,
        "unfinished_attempts": len(incomplete),
        "unfinished_actions_lower_bound": lower,
        "unfinished_attempts_protocol_cap_sum": upper,
        "all_actions_lower_bound": physical["actions"] + lower,
        "array_only_actions_lower_bound": array_lower,
        "arrays": _count(counts["arrays"], "verified arrays"),
        "generations": _count(counts["generations"], "recorded generations"),
        "incomplete_attempts": copy.deepcopy(incomplete),
        "action_bound_basis": "Completed rows plus the latest archive-bound request/generation step for each known unfinished attempt. This stronger bound includes, rather than adds to, the array-only generation-step bound. Unknown unsynchronized work is not assigned zero cost.",
    }


def _audit(directory, summary, physical, metadata, array_value, provider_value):
    if array_value is None or provider_value is None:
        return {
            "status": "pending",
            "array_receipt_present": array_value is not None,
            "provider_receipt_present": provider_value is not None,
        }
    try:
        arrays, array_sha = _receipt(array_value)
        provider, provider_sha = _receipt(provider_value)
        _require(
            arrays["schema_version"] == "representation-case-array-audit-1"
            and arrays["status"] == "passed"
            and arrays["checks"]["complete_sealed_case"] is True,
            "Array audit did not pass",
        )
        _require(
            provider["schema_version"] == "representation-provider-audit-1.0"
            and provider["status"] == "complete"
            and provider["complete"] is True,
            "Provider audit is incomplete",
        )
        for receipt in (arrays, provider):
            _require(
                receipt["episode_id"] == summary["episode_id"], "Audit episode differs"
            )
        for name in (
            "summary.json",
            "events.jsonl",
            "completion_receipt.json",
            "frozen_weights_after.json",
        ):
            _require(
                arrays["input_file_sha256"][name] == file_sha256(directory / name),
                f"Array audit {name} differs",
            )
        for name in ("summary.json", "events.jsonl"):
            _require(
                _source_sha(
                    provider["source_file_sha256"][name], entire=name.endswith(".jsonl")
                )
                == file_sha256(directory / name),
                f"Provider audit {name} differs",
            )
        if (directory / "provider.jsonl").exists():
            current = file_sha256(directory / "provider.jsonl")
            _require(
                arrays["input_file_sha256"]["provider.jsonl"]
                == _source_sha(
                    provider["source_file_sha256"]["provider.jsonl"], entire=True
                )
                == current,
                "Audited provider ledger differs",
            )
        for name, value in metadata["worker_metadata_sha256"].items():
            _require(
                arrays["worker_metadata_sha256"][name] == value,
                f"Audited worker {name} differs",
            )
        identities = {
            "protocol_sha256": summary["protocol_sha256"],
            "reset_manifest_sha256": metadata["reset_manifest_sha256"],
            "image_library_id": summary["image_library_id"],
            **{
                k: metadata["common_identity"][k]
                for k in (
                    "bank_inventory_sha256",
                    "native_tensor_sha256",
                    "payload_sha256",
                )
            },
        }
        for name, value in identities.items():
            _require(arrays["identities"][name] == value, f"Array audit {name} differs")
        _require(
            provider["protocol_sha256"] == summary["protocol_sha256"]
            and provider["image_library_id"] == summary["image_library_id"],
            "Provider audit input identity differs",
        )
        for key, value in {
            "physical_rollouts": physical["rollouts"],
            **{
                k: physical[k]
                for k in (
                    "actions",
                    "flow_velocity_evaluations",
                    "native_parity_velocity_evaluations",
                    "probe_velocity_evaluations",
                    "velocity_evaluations",
                )
            },
        }.items():
            _require(arrays["counts"][key] == value, f"Array audit {key} differs")
        _require(
            arrays["archive"]["gzip_crc_verified"] is True,
            "Archive verification is incomplete",
        )
        _same(provider["provider"], physical["provider"], "Audited provider usage")
        if array_sha is not None:
            _require(
                provider["archive_audit"]["receipt_sha256"] == array_sha,
                "Provider sealed-archive receipt binding differs",
            )
        return {
            "status": "passed",
            "array_receipt_sha256": array_sha,
            "array_receipt_digest": digest(arrays),
            "provider_receipt_sha256": provider_sha,
            "provider_receipt_digest": digest(provider),
            "archive": arrays["archive"],
            "counts": arrays["counts"],
            "recorded_work": _recorded_work(arrays, provider, physical, partial=False),
            "provider": provider["provider"],
            "array_source_sha256": arrays.get("source_code_sha256"),
            "array_postprocessor": arrays.get("postprocessor"),
            "provider_source_sha256": provider.get("source_code_sha256"),
        }
    except (KeyError, TypeError, ValueError, OSError) as exc:
        return {
            "status": "failed",
            "error": f"Audit binding failed: {type(exc).__name__}: {str(exc) if not isinstance(exc, OSError) else 'input unavailable'}",
        }


def _partial_audit(
    directory,
    summary,
    rows,
    physical,
    metadata,
    array_value,
    provider_value,
    event_hashes,
):
    """Release observations from archived completed events, never cohort efficacy."""
    if array_value is None or provider_value is None:
        return {"status": "incomplete_case"}
    try:
        arrays, array_sha = _receipt(array_value)
        provider, provider_sha = _receipt(provider_value)
        _require(
            arrays["schema_version"] == "representation-partial-array-audit-1"
            and arrays["status"] == "verified_partial_archive"
            and arrays["checks"]["complete_sealed_case"] is False,
            "Partial archive scope differs",
        )
        for name in (
            "full_archive_stream_verified",
            "all_available_npy_references_verified",
            "published_small_files_match_archive",
        ):
            _require(arrays["checks"][name] is True, f"Partial archive {name} failed")
        _require(
            provider["schema_version"] == "representation-provider-audit-1.0"
            and provider["status"] == "preserved_prefix"
            and provider["complete"] is False,
            "Partial provider scope differs",
        )
        for receipt in (arrays, provider):
            _require(
                receipt["episode_id"] == summary["episode_id"],
                "Partial audit episode differs",
            )
        for name in ("summary.json", "events.jsonl", "provider.jsonl"):
            if name == "provider.jsonl" and not (directory / name).exists():
                continue
            current = file_sha256(directory / name)
            _require(
                arrays["input_file_sha256"][name]
                == _source_sha(
                    provider["source_file_sha256"][name], entire=name.endswith(".jsonl")
                )
                == current,
                f"Partial audit {name} differs",
            )
        for name, current in metadata["worker_metadata_sha256"].items():
            _require(
                arrays["worker_metadata_sha256"][name] == current,
                f"Partial worker {name} differs",
            )
        for name in ("protocol_sha256", "image_library_id"):
            _require(
                arrays["identities"][name] == provider[name] == summary[name],
                f"Partial {name} differs",
            )
        for name, value in {
            "reset_manifest_sha256": metadata["reset_manifest_sha256"],
            "native_tensor_sha256_before": metadata["common_identity"][
                "native_tensor_sha256"
            ],
            "native_tensor_sha256_after": None,
            **{
                key: metadata["common_identity"][key]
                for key in (
                    "bank_inventory_sha256",
                    "payload_sha256",
                )
            },
        }.items():
            _require(arrays["identities"][name] == value, f"Partial {name} differs")
        _same(provider["provider"], physical["provider"], "Partial provider usage")
        if array_sha is not None:
            _require(
                provider["archive_audit"]["receipt_sha256"] == array_sha,
                "Provider partial-archive receipt binding differs",
            )
        verified = arrays["verified_completed_attempts"]
        _require(
            len(verified) == len(rows), "Partial completed attempt coverage differs"
        )
        for proof, row in zip(verified, rows, strict=True):
            for name in ("attempt_id", "arm", "revision", "success"):
                _require(proof[name] == row[name], f"Partial completed {name} differs")
            _require(
                proof["actions"] == row["actions_executed"]
                and proof["attempt_sha256"] == digest(row)
                and proof["reset_audit_sha256"] == digest(row["reset_audit"]),
                "Partial attempt/reset digest differs",
            )
            _source_sha(proof["completed_event_sha256"])
            _require(
                proof["completed_event_sha256"] == event_hashes[row["attempt_id"]],
                "Partial completed event digest differs",
            )
        return {
            "status": "verified_partial",
            "complete_sealed_case": False,
            "array_receipt_sha256": array_sha,
            "array_receipt_digest": digest(arrays),
            "provider_receipt_sha256": provider_sha,
            "provider_receipt_digest": digest(provider),
            "archive": arrays["archive"],
            "counts": arrays["counts"],
            "recorded_work": _recorded_work(arrays, provider, physical, partial=True),
            "verified_completed_attempts": verified,
            "array_source_sha256": arrays.get("source_code_sha256"),
            "array_postprocessor": arrays.get("postprocessor"),
            "provider_source_sha256": provider.get("source_code_sha256"),
            "limitations": "Completed-event observations only. No completion seal/final native-weight proof; no cohort efficacy or absent-cost claim.",
        }
    except (KeyError, TypeError, ValueError, OSError) as exc:
        return {
            "status": "failed",
            "error": f"Partial audit binding failed: {type(exc).__name__}: {str(exc) if not isinstance(exc, OSError) else 'input unavailable'}",
        }


def _case(directory, protocol, phase, arrays, providers):
    directory = Path(directory)
    summary = _read(directory / "summary.json")
    episode = summary["episode_id"]
    _require(
        summary["schema_version"] == protocol["schema_version"]
        and summary["protocol_sha256"] == digest(protocol),
        "Summary protocol differs",
    )
    _require(
        summary["development"] is (phase == "development")
        and summary["seed"] == protocol["seed"]
        and summary["initial_state_id"] == 0,
        "Summary phase/reset differs",
    )
    _require(
        episode
        == f"{summary['suite']}:seed{summary['seed']}:task{summary['task_id']}:state0",
        "Episode identity differs",
    )
    rows, counts = list(summary["physical_rollouts"]), []
    event_hashes = {}
    if summary["status"] != "complete" and arrays is not None:
        completed_events = []
        with (directory / "events.jsonl").open("rb") as stream:
            for line in stream:
                event = _strict_json(line)
                if event.get("kind") == "representation_rollout_complete":
                    completed_events.append(event["attempt"])
                    event_hashes[event["attempt"]["attempt_id"]] = hashlib.sha256(
                        line
                    ).hexdigest()
        _same(
            completed_events[: len(rows)],
            rows,
            "Interrupted summary completed-event prefix",
        )
        rows = completed_events
    _require(
        len({r["attempt_id"] for r in rows}) == len(rows), "Duplicate physical rollout"
    )
    for row in rows:
        counts.append(_validate_attempt(row, episode, protocol))
    if rows:
        _require(
            rows[0]["arm"] == "native" and sum(r["arm"] == "native" for r in rows) == 1,
            "Shared baseline must occur exactly once",
        )
        for row in rows:
            _same(row["reset_audit"], rows[0]["reset_audit"], "Paired reset audit")
    recorded_calls = [c for row in rows for c in row["provider_records"]]
    calls = _calls(directory / "provider.jsonl")
    _same(
        calls[: len(recorded_calls)], recorded_calls, "Physical provider ledger prefix"
    )
    physical = _cost(rows, calls)
    complete = summary["status"] == "complete"
    arm_rows, metadata, audit = [], None, {"status": "incomplete_case"}
    if complete:
        _same(calls, recorded_calls, "Complete provider ledger")
        _same(summary["baseline"], rows[0], "Shared baseline record")
        _require(set(summary["arms"]) == set(ARMS), "Complete arm coverage differs")
        _require(
            summary["checks"]["native_parity"]["max_abs"] <= 1e-5
            and summary["checks"]["weighted_vision"]["status"] == "passed",
            "Numerical gate failed",
        )
        _require(
            rows[0]["parity_velocity_evaluations"] == 10
            and rows[0]["probe_velocity_evaluations"]
            == summary["checks"]["weighted_vision"]["velocity_evaluations"],
            "Numerical gate cost differs",
        )
        arm_rows = _arm_rows(summary, protocol)
        old = summary["physical_cost"]
        for name in (
            "rollouts",
            "actions",
            "velocity_evaluations",
            "provider",
            "donor_captures",
            "donor_capture_seconds",
            "policy_seconds",
            "environment_seconds",
            "condition_seconds",
        ):
            _same(old[name], physical[name], f"Physical {name}")
        _same(
            old["rollout_wall_seconds"], physical["wall_seconds"], "Physical wall time"
        )
        metadata = _metadata(directory, summary, protocol, phase)
        audit = _audit(directory, summary, physical, metadata, arrays, providers)
    elif arrays is not None and providers is not None:
        metadata = _metadata(directory, summary, protocol, phase, sealed=False)
        audit = _partial_audit(
            directory,
            summary,
            rows,
            physical,
            metadata,
            arrays,
            providers,
            event_hashes,
        )
    return (
        {
            "episode_id": episode,
            "suite": summary["suite"],
            "task_id": summary["task_id"],
            "instruction": summary["instruction"],
            "seed": summary["seed"],
            "complete": complete,
            "summary_sha256": file_sha256(directory / "summary.json"),
            "protocol_sha256": summary["protocol_sha256"],
            "reset_entry_sha256": summary["reset_entry_sha256"],
            "image_library_id": summary["image_library_id"],
            "physical_cost": physical,
            "metadata": metadata,
            "audit": audit,
            "arm_rows": arm_rows,
            "physical_attempt_rows": [
                {
                    "episode_id": episode,
                    "suite": summary["suite"],
                    "attempt_id": r["attempt_id"],
                    "arm": r["arm"],
                    "revision": r["revision"],
                    "forced_after_baseline_success": bool(
                        summary["development"]
                        and rows[0]["success"]
                        and r["revision"] > 0
                    ),
                    "recorded_success": r["success"],
                    "actions": r["actions_executed"],
                    "policy_replans": r["policy_replans"],
                    "velocity_evaluations": r["velocity_evaluations"]
                    + r["parity_velocity_evaluations"]
                    + r["probe_velocity_evaluations"],
                    "provider": summarize_calls(r["provider_records"]),
                    "video_sha256": r["video_sha256"],
                    "video_fps": r["video_fps"],
                    "video_file": Path(r["video_path"]).name,
                    **{k: r[k] for k in TIMES},
                    **c,
                }
                for r, c in zip(rows, counts, strict=True)
            ],
        },
        rows,
        calls,
    )


def _paired(cases, arm, control):
    counts = Counter(
        {name: 0 for name in ("both_success", "arm_only", "control_only", "both_fail")}
    )
    discordant = []
    for case in cases:
        rows = {r["arm"]: r for r in case["arm_rows"]}
        a, b = rows[arm]["success"], rows[control]["success"]
        key = (
            "both_success"
            if a and b
            else "arm_only"
            if a
            else "control_only"
            if b
            else "both_fail"
        )
        counts[key] += 1
        if a != b:
            discordant.append({"episode_id": case["episode_id"], "outcome": key})
    return {
        "arm": arm,
        "control": control,
        "counts": dict(counts),
        "discordant": discordant,
    }


def _group(cases, physical, calls, release):
    verified_work = [
        c["audit"]["recorded_work"]
        for c in cases
        if c["audit"]["status"] in ("passed", "verified_partial")
    ]
    result = {
        "cases": len(cases),
        "physical_cost": _cost(physical, calls),
        "verified_recorded_work": {
            "cases": len(verified_work),
            **{
                name: sum(row[name] for row in verified_work)
                for name in (
                    "completed_rollouts",
                    "completed_actions",
                    "completed_velocity_evaluations",
                    "recorded_velocity_evaluations_lower_bound",
                    "setup_velocity_evaluations",
                    "unfinished_recorded_flow_velocity_evaluations",
                    "unfinished_attempts",
                    "unfinished_actions_lower_bound",
                    "unfinished_attempts_protocol_cap_sum",
                    "all_actions_lower_bound",
                    "array_only_actions_lower_bound",
                    "arrays",
                    "generations",
                )
            },
            "incomplete_attempts": [
                {"episode_id": c["episode_id"], **row}
                for c in cases
                if c["audit"]["status"] in ("passed", "verified_partial")
                for row in c["audit"]["recorded_work"]["incomplete_attempts"]
            ],
        },
        "arms": None,
        "paired": None,
        "operator_coverage": {
            arm: {
                "completed_rollouts": sum(r["arm"] == arm for r in physical),
                "provider": summarize_calls(
                    [c for c in calls if c["attempt_id"].startswith(arm + "_revision")]
                ),
                **{
                    name: sum(r[name] for r in physical if r["arm"] == arm)
                    for name in (*ACTION_COUNTS, "accepted_decisions_executed")
                },
            }
            for arm in ARMS
        },
    }
    if not release:
        return result
    arms = {}
    for arm in ARMS:
        rows = [r for c in cases for r in c["arm_rows"] if r["arm"] == arm]
        rescued = [r for r in rows if r["rescued"]]
        complete_tokens = [
            r["complete_tokens_to_success"]
            for r in rescued
            if r["complete_tokens_to_success"] is not None
        ]
        arms[arm] = {
            "cases": len(rows),
            "baseline_successes": sum(r["baseline_success"] for r in rows),
            "successes": sum(r["success"] for r in rows),
            "rescues": len(rescued),
            "censored": sum(r["censored"] for r in rows),
            "successes_by_revision_budget": [
                sum(r["success_by_revision"][b] for r in rows) for b in range(3)
            ],
            "rescue_episode_ids": [r["episode_id"] for r in rescued],
            "censored_episode_ids": [r["episode_id"] for r in rows if r["censored"]],
            "rescue_only": {
                "revisions_to_success": _stats(
                    [r["first_success_revision"] for r in rescued]
                ),
                "decisions_to_success": _stats(
                    [r["decisions_through_success_or_cap"] for r in rescued]
                ),
                "complete_tokens_to_success": _stats(complete_tokens),
                "rescues_with_missing_usage": len(rescued) - len(complete_tokens),
            },
            "development_extra_rollouts": sum(
                r["development_extra_rollouts"] for r in rows
            ),
            "physical_provider": summarize_calls(
                [c for r in physical if r["arm"] == arm for c in r["provider_records"]]
            ),
            "executed_coverage": {
                name: sum(r[name] for r in physical if r["arm"] == arm)
                for name in (*ACTION_COUNTS, "accepted_decisions_executed")
            },
            "provider_by_revision_budget": [
                summarize_calls(
                    [
                        c
                        for case in cases
                        for r in physical
                        if r["episode_id"] == case["episode_id"]
                        and r["arm"] == arm
                        and r["revision"] <= budget
                        and not (
                            next(x for x in case["arm_rows"] if x["arm"] == arm)[
                                "baseline_success"
                            ]
                        )
                        for c in r["provider_records"]
                    ]
                )
                for budget in range(3)
            ],
        }
    result["arms"] = arms
    result["paired"] = {
        "vs_native_retry": {
            arm: _paired(cases, arm, "native_retry")
            for arm in ARMS
            if arm != "native_retry"
        },
        "vs_same_operator_random": {
            arm: _paired(cases, arm, f"random_{arm.removeprefix('astra_')}")
            for arm in ("astra_tli", "astra_vei", "astra_vli")
        },
    }
    return result


def build_report(
    case_directories, *, phase, protocol, array_audits=None, provider_audits=None
):
    expected = expected_episodes(protocol, phase)
    array_audits, provider_audits = array_audits or {}, provider_audits or {}
    cases, physical, calls, seen = [], [], [], set()
    for directory in case_directories:
        identity = _read(Path(directory) / "summary.json")["episode_id"]
        _require(
            identity in expected and identity not in seen,
            "Unexpected or duplicate case",
        )
        seen.add(identity)
        case, rows, ledger = _case(
            directory,
            protocol,
            phase,
            array_audits.get(identity),
            provider_audits.get(identity),
        )
        cases.append(case)
        physical.extend(rows)
        calls.extend(ledger)
    cases.sort(key=lambda c: expected.index(c["episode_id"]))
    completed = [c for c in cases if c["complete"]]
    identified = [c for c in cases if c["metadata"] is not None]
    if identified:
        for case in identified[1:]:
            _same(
                case["metadata"]["common_identity"],
                identified[0]["metadata"]["common_identity"],
                "Across-case checkpoint/runtime/prompt identity",
            )
        for suite in protocol["suites"]:
            selected = [c for c in identified if c["suite"] == suite]
            if selected:
                for case in selected[1:]:
                    _same(
                        case["metadata"]["worker_metadata_sha256"][
                            "reset_manifest.json"
                        ],
                        selected[0]["metadata"]["worker_metadata_sha256"][
                            "reset_manifest.json"
                        ],
                        "Full suite reset manifest",
                    )
    audited = sum(c["audit"]["status"] == "passed" for c in cases)
    failed = any(c["audit"]["status"] == "failed" for c in cases)
    release = len(completed) == len(expected) == audited
    status = (
        "complete"
        if release
        else "audit_failed"
        if failed
        else "audit_pending"
        if len(completed) == len(expected)
        else "partial"
    )
    groups = {"pooled": _group(cases, physical, calls, release)}
    for suite in protocol["suites"]:
        groups[suite] = _group(
            [c for c in cases if c["suite"] == suite],
            [r for r in physical if r["episode_id"].startswith(suite + ":")],
            [c for c in calls if c["episode_id"].startswith(suite + ":")],
            release,
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "complete": release,
        "efficacy_released": release,
        "phase": phase,
        "seed": protocol["seed"],
        "expected_cases": len(expected),
        "received_cases": len(cases),
        "completed_cases": len(completed),
        "audited_cases": audited,
        "expected_episode_ids": expected,
        "missing_episode_ids": [e for e in expected if e not in seen],
        "protocol": copy.deepcopy(protocol),
        "protocol_sha256": digest(protocol),
        "common_identity": identified[0]["metadata"]["common_identity"]
        if identified
        else None,
        "groups": groups,
        "cases": cases,
        "case_arm_rows": [r for c in cases for r in c["arm_rows"]] if release else [],
        "physical_attempt_rows": [r for c in cases for r in c["physical_attempt_rows"]],
        "verified_completed_rollout_observations": [
            {
                **r,
                "audit_scope": c["audit"]["status"],
                "summary_sha256": c["summary_sha256"],
            }
            for c in cases
            if c["audit"]["status"] in ("passed", "verified_partial")
            for r in c["physical_attempt_rows"]
        ],
        "development_stability_rows": [
            r
            for c in cases
            if c["audit"]["status"] in ("passed", "verified_partial")
            for r in c["physical_attempt_rows"]
            if r["forced_after_baseline_success"]
        ],
        "limitations": LIMITATIONS,
        "source_sha256": {"representation_report.py": file_sha256(__file__)},
        "related_studies_pooled": False,
        "hidden_state_or_physics_replay_performed": False,
    }


def _json_write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def _csv(path, rows):
    with path.open("w", newline="") as handle:
        if not rows:
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    k: json.dumps(v, sort_keys=True)
                    if isinstance(v, (list, dict))
                    else v
                    for k, v in row.items()
                }
            )


def _budget_rows(report):
    if not report["efficacy_released"]:
        return []
    return [
        {
            "group": group,
            "arm": arm,
            "revision_budget": budget,
            "cases": row["cases"],
            "successes": row["successes_by_revision_budget"][budget],
            "provider_calls": row["provider_by_revision_budget"][budget][
                "provider_calls"
            ],
            **{
                f"{name}_{suffix}": token[field]
                for name, token in row["provider_by_revision_budget"][budget][
                    "tokens"
                ].items()
                for suffix, field in (("known_sum", "sum"), ("complete", "complete"))
            },
        }
        for group, values in report["groups"].items()
        for arm, row in values["arms"].items()
        for budget in range(3)
    ]


def _application_label(row):
    if not row["arm"].startswith("astra_"):
        return (
            "native policy"
            if row["arm"] in ("native", "native_retry")
            else "random representation choice"
        )
    if row["accepted_decisions_executed"] == 0:
        return "native fallback only; no accepted Astra choice executed"
    if row["nonzero_intervention_actions"] == 0:
        return "accepted native/neutral choices; no recorded edit effect"
    return "accepted representation choices; native portions may also occur"


def _figures(directory, report):
    """Export measured counts only, binding plotted values to the exact report."""
    if not report["efficacy_released"]:
        return []
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pooled = report["groups"]["pooled"]
    values = {
        "report_sha256": file_sha256(directory / "report.json"),
        "source_sha256": file_sha256(__file__),
        "phase": report["phase"],
        "seed": report["seed"],
        "cases": report["expected_cases"],
        "successes_by_revision_budget": {
            arm: row["successes_by_revision_budget"]
            for arm, row in pooled["arms"].items()
        },
        "censored": {arm: row["censored"] for arm, row in pooled["arms"].items()},
        "physical_tokens": {
            arm: row["physical_provider"]["tokens"]
            for arm, row in pooled["arms"].items()
        },
        "development_extra_rollouts": {
            arm: row["development_extra_rollouts"]
            for arm, row in pooled["arms"].items()
        },
        "scope": "Measured cumulative reset-search outcomes, not learning. Costs include failed and forced development calls; missing-usage sums are lower bounds.",
    }
    _json_write(directory / "plotted_values.json", values)
    title = f"{report['phase'].capitalize()} · seed {report['seed']} · {report['expected_cases']} known-task resets"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for axis, selected, label in zip(
        axes,
        (
            [a for a in ARMS if a.startswith("astra_")],
            [a for a in ARMS if not a.startswith("astra_")],
        ),
        ("Astra choices", "Matched controls"),
        strict=True,
    ):
        for arm in selected:
            row = pooled["arms"][arm]
            axis.plot(
                range(3),
                row["successes_by_revision_budget"],
                marker="o",
                label=f"{arm.removeprefix('astra_')} (censored {row['censored']})",
            )
        axis.set(
            title=label,
            xlabel="Up-to full-rollout revisions after shared baseline",
            xticks=[0, 1, 2],
            ylim=(-0.1, report["expected_cases"] + 0.2),
        )
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8, loc="best")
    axes[0].set_ylabel("Cumulative successful cases (fixed denominator)")
    fig.suptitle(title)
    fig.tight_layout()
    names = []
    for extension in ("png", "pdf"):
        name = f"success_by_revision.{extension}"
        fig.savefig(
            directory / name, dpi=170, metadata={"Creator": "representation_report"}
        )
        names.append(name)
    plt.close(fig)
    selected = [a for a in ARMS if a.startswith("astra_")]
    inputs = [
        pooled["arms"][a]["physical_provider"]["tokens"]["input_tokens"]["sum"]
        for a in selected
    ]
    outputs = [
        pooled["arms"][a]["physical_provider"]["tokens"]["output_tokens"]["sum"]
        for a in selected
    ]
    complete = all(
        pooled["arms"][a]["physical_provider"]["tokens"][name]["complete"]
        for a in selected
        for name in ("input_tokens", "output_tokens")
    )
    fig, axis = plt.subplots(figsize=(10, 4.8))
    axis.bar(range(len(selected)), inputs, label="Reported input")
    axis.bar(
        range(len(selected)),
        outputs,
        bottom=inputs,
        label="Reported output (includes reasoning)",
    )
    axis.set_xticks(range(len(selected)), [a.removeprefix("astra_") for a in selected])
    axis.set(
        ylabel="Known provider tokens",
        title=title
        + (" · complete usage" if complete else " · lower bounds; missing usage"),
    )
    axis.legend()
    fig.text(
        0.5,
        0.015,
        "All physical calls, including rejection and development extras. Failed capped searches remain included.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    for extension in ("png", "pdf"):
        name = f"physical_token_cost.{extension}"
        fig.savefig(
            directory / name, dpi=170, metadata={"Creator": "representation_report"}
        )
        names.append(name)
    plt.close(fig)
    return names


def write_outputs(directory, report):
    """Create a new portable result bundle; never overwrite frozen evidence."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    _json_write(directory / "report.json", report)
    _json_write(directory / "protocol.json", report["protocol"])
    _csv(directory / "case_arms.csv", report["case_arm_rows"])
    _csv(directory / "physical_rollouts.csv", report["physical_attempt_rows"])
    _csv(directory / "revision_budgets.csv", _budget_rows(report))
    _csv(directory / "development_stability.csv", report["development_stability_rows"])
    _csv(
        directory / "verified_completed_rollouts.csv",
        report["verified_completed_rollout_observations"],
    )
    figures = _figures(directory, report)
    cost = report["groups"]["pooled"]["physical_cost"]
    recorded = report["groups"]["pooled"]["verified_recorded_work"]
    _csv(directory / "unfinished_attempts.csv", recorded["incomplete_attempts"])
    usage = cost["provider"]
    token_text = "; ".join(
        f"{name}: {v['sum']:,}{' (lower bound; missing usage)' if not v['complete'] else ''}"
        for name, v in usage["tokens"].items()
    )
    text = [
        f"# Representation steering — {report['phase']}",
        "",
        f"Status: **{report['status']}**. Seed {report['seed']}; {report['audited_cases']}/{report['expected_cases']} prescribed cases independently audited. Efficacy released: {report['efficacy_released']}.",
        "",
        f"Recorded completed-rollout work: {cost['rollouts']} rollouts, {cost['actions']:,} actions, {cost['velocity_evaluations']:,} velocity evaluations. Separately, the full recorded ledger contains {usage['provider_calls']} provider calls and {usage['preflight_failures']} preflight failures, including any unfinished rollout. Shared baseline and numerical probes counted once."
        + (
            " Incomplete-run rollout/action/compute totals are lower bounds; in-flight work is not assigned zero cost."
            if not report["complete"]
            else ""
        ),
        "",
        f"Provider-reported tokens, including rejected calls: {token_text}. Reasoning is a subset of output.",
        "",
        f"Summed rollout wall time: {cost['wall_seconds']:.3f}s. This is measured work across cases, not elapsed workflow time; all other timings are nested.",
        "",
    ]
    if recorded["cases"]:
        text += [
            f"Independent archive/provider proofs cover {recorded['cases']} cases: {recorded['arrays']:,} arrays and {recorded['generations']:,} recorded generations. They establish **{recorded['recorded_velocity_evaluations_lower_bound']:,} recorded velocity evaluations**, including {recorded['setup_velocity_evaluations']:,} setup/probe evaluations once per case. Of these, {recorded['completed_velocity_evaluations']:,} belong to completed rollouts and {recorded['unfinished_recorded_flow_velocity_evaluations']:,} to unfinished rollouts; these are different scopes, not extra additive totals.",
            "",
        ]
        if recorded["unfinished_attempts"]:
            text += [
                f"The {recorded['unfinished_attempts']} known unfinished attempts executed **at least {recorded['unfinished_actions_lower_bound']:,} additional actions** beyond the exact {recorded['completed_actions']:,} completed-row actions: at least {recorded['all_actions_lower_bound']:,} actions overall. This uses the latest verified request/generation step. The weaker array-only bound ({recorded['array_only_actions_lower_bound']:,} overall) overlaps that evidence and is not added again. Outcomes and unsynchronized tails remain unknown; the per-attempt protocol cap does not make missing work complete.",
                "",
                "| Case | Unfinished attempt | Additional actions lower bound | Recorded flow evaluations | Outcome |",
                "|---|---|---:|---:|---|",
            ]
            for row in recorded["incomplete_attempts"]:
                text.append(
                    f"| {row['episode_id']} | {row['attempt_id']} | {row['executed_actions_lower_bound']} | {row['flow_velocity_evaluations_recorded']} | unknown |"
                )
            text.append("")
    if report["efficacy_released"]:
        text += [
            "| Arm | Success after 0 / ≤1 / ≤2 revisions | Rescues | Censored | Rescue-only median revisions |",
            "|---|---|---:|---:|---:|",
        ]
        for arm, row in report["groups"]["pooled"]["arms"].items():
            curve = " / ".join(
                f"{x}/{row['cases']}" for x in row["successes_by_revision_budget"]
            )
            text.append(
                f"| {arm} | {curve} | {row['rescues']} | {row['censored']} | {row['rescue_only']['revisions_to_success']['median']} |"
            )
        text += [
            "",
            "Recorded application coverage (accepted can include explicit native or neutral choices):",
            "",
            "| Astra arm | Calls / accepted | Executed accepted decisions | Actions with recorded edit effect | Failed-call native fallback actions |",
            "|---|---:|---:|---:|---:|",
        ]
        for arm, row in report["groups"]["pooled"]["arms"].items():
            if arm.startswith("astra_"):
                usage, coverage = row["physical_provider"], row["executed_coverage"]
                text.append(
                    f"| {arm} | {usage['provider_calls']} / {usage['accepted_proposals']} | {coverage['accepted_decisions_executed']} | {coverage['nonzero_intervention_actions']} | {coverage['provider_failure_fallback_actions']} |"
                )
        if report["development_stability_rows"]:
            text += [
                "",
                "Forced development stability/coverage checks after a successful shared baseline. Each row is the actual extra rollout outcome; these are not rescues and do not improve the inherited success curve. The success-gated evaluation does not measure unconditional deployment robustness.",
                "",
                "| Case | Extra arm / revision | Actual outcome | Actions |",
                "|---|---|---|---:|",
            ]
            for row in report["development_stability_rows"]:
                text.append(
                    f"| {row['episode_id']} | {row['arm']} / {row['revision']} | {'success' if row['recorded_success'] else 'failure'} | {row['actions']} |"
                )
        text += [
            "",
            "![Cumulative success by revision budget](success_by_revision.png)",
            "",
            "![Known physical provider token costs](physical_token_cost.png)",
            "",
            "Scientific exports: "
            + " · ".join(f"[{name}]({name})" for name in figures)
            + " · [Exact plotted values and source hashes](plotted_values.json)",
        ]
    else:
        text.append(
            "Success curves and paired comparisons are withheld until the full prescribed cohort and both audits are complete. Recorded costs remain available; unrecorded or in-flight work is unknown."
        )
        if cost["provider_error_taxonomy"]["budget_exceeded"]:
            text += [
                "",
                "The recorded provider classified calls as **budget_exceeded**. This is a spending-cap failure, not evidence that the unexecuted Astra operators worked. Available token sums are lower bounds; missing usage and unfinished work remain unknown.",
            ]
        if report["verified_completed_rollout_observations"]:
            text += [
                "",
                "Individually verified completed-rollout observations from preserved archives. This table is not a complete-cohort success rate or a causal steering claim. A forced rollout after a successful baseline is a stability check, not a rescue.",
                "",
                "| Case | Actual physical rollout | Recorded outcome | Actions | Accepted / edit-effect / fallback actions | Application scope |",
                "|---|---|---|---:|---|---|",
            ]
            for row in report["verified_completed_rollout_observations"]:
                text.append(
                    f"| {row['episode_id']} | {row['attempt_id']}{' (forced stability)' if row['forced_after_baseline_success'] else ''} | {'success' if row['recorded_success'] else 'failure'} | {row['actions']} | {row['accepted_decision_actions']} / {row['nonzero_intervention_actions']} / {row['provider_failure_fallback_actions']} | {_application_label(row)} |"
                )
        if report["development_stability_rows"]:
            text += [
                "",
                "Forced development stability checks, shown separately. These are the same physical rows listed above, not extra cost or rescues.",
                "",
                "| Case | Extra arm | Actual outcome | Actions |",
                "|---|---|---|---:|",
            ]
            for row in report["development_stability_rows"]:
                text.append(
                    f"| {row['episode_id']} | {row['arm']} | {'success' if row['recorded_success'] else 'failure'} | {row['actions']} |"
                )
        text += [
            "",
            "Recorded operator coverage. Accepted calls during unfinished rollouts are not counted as executed decisions. Zero executed accepted choices means a configured Astra arm did not exercise its requested steering in the completed evidence.",
            "",
            "| Astra arm | Recorded calls / accepted | Completed rollouts | Executed accepted decisions | Edit-effect actions | Native fallback actions |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for arm, row in report["groups"]["pooled"]["operator_coverage"].items():
            if arm.startswith("astra_"):
                usage = row["provider"]
                text.append(
                    f"| {arm} | {usage['provider_calls']} / {usage['accepted_proposals']} | {row['completed_rollouts']} | {row['accepted_decisions_executed']} | {row['nonzero_intervention_actions']} | {row['provider_failure_fallback_actions']} |"
                )
    text += [
        "",
        "[Machine report](report.json) · [Case/arm CSV](case_arms.csv) · [Physical rollout CSV](physical_rollouts.csv) · [Verified completed-rollout CSV](verified_completed_rollouts.csv) · [Unfinished-work CSV](unfinished_attempts.csv) · [Budget/token CSV](revision_budgets.csv) · [Forced development CSV](development_stability.csv) · [Protocol](protocol.json)",
        "",
        *[f"- {line}" for line in LIMITATIONS],
    ]
    (directory / "report.md").write_text("\n".join(text) + "\n")
    from markdown_it import MarkdownIt

    body = (
        MarkdownIt("commonmark", {"html": False})
        .enable("table")
        .render("\n".join(text))
    )
    (directory / "index.html").write_text(
        f'<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Representation steering — {html.escape(report["phase"])}</title><style>body{{font:16px/1.6 system-ui;max-width:1200px;margin:35px auto;padding:0 22px;color:#183344}}table{{border-collapse:collapse;display:block;overflow:auto}}td,th{{padding:10px;border-bottom:1px solid #ccd9e0;text-align:left}}a{{color:#086984}}code{{overflow-wrap:anywhere}}</style><main>{body}</main></html>\n'
    )
    _json_write(
        directory / "manifest.json",
        {
            "schema_version": "representation-report-publication-1.0",
            "files": {
                p.name: {"sha256": file_sha256(p), "bytes": p.stat().st_size}
                for p in sorted(directory.iterdir())
                if p.is_file()
            },
        },
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--phase", choices=("development", "evaluation"), required=True)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--array-audits", nargs="*", type=Path, default=[])
    parser.add_argument("--provider-audits", nargs="*", type=Path, default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args(argv)
    protocol = _read(args.protocol) if args.protocol else load_protocol()
    if args.phase == "development" and args.protocol is None:
        protocol["seed"] = protocol["development_seed"]

    def receipts(paths):
        result = {}
        for path in paths:
            episode = _read(path)["episode_id"]
            _require(episode not in result, "Duplicate audit receipt")
            result[episode] = path
        return result

    report = build_report(
        args.inputs,
        phase=args.phase,
        protocol=protocol,
        array_audits=receipts(args.array_audits),
        provider_audits=receipts(args.provider_audits),
    )
    if args.require_complete and not report["complete"]:
        raise ValueError(
            f"Report is {report['status']}; complete audited coverage is required"
        )
    write_outputs(args.output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "phase": report["phase"],
                "audited_cases": report["audited_cases"],
            }
        )
    )


if __name__ == "__main__":
    main()
