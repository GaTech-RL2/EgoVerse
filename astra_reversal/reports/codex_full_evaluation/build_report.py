#!/usr/bin/env python3
"""Build a local, whitelisted report from independently audited sealed tasks.

This is a presentation layer, not an auditor or an experiment runner. It reads
existing audit receipts, checks their bindings, and never fetches or runs jobs.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import statistics
import tempfile
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ASTRA = HERE.parents[1]
DEFAULT_OPS = ASTRA / ".deps/codex-bridge-20260928"
SOURCE = {
    "source_revision": "64bb16495cad3315185965c44f77ef430f3f4cc6",
    "payload_sha256": "2c341508a6a3845a4991cd6cda9a4ed789c6df75da7a60cfd8f66d9e5c6e56ac",
    "payload_bytes": 50202537,
}
RECOVERY_SOURCE = {
    "source_revision": "9ab4cb57120c2ad0dd8919a6dcb32f3f0dad4064",
    "payload_sha256": "5c0a8715d44c9b16a48483476c417244b12b035af983d3d4f67b1c488d7df170",
    "payload_bytes": 51637817,
}
EQUIVALENCE_SHA = "5ce1a0580f872efcbc8eee921bef8eadc9693d32068dc29821b1689e73e1e537"
FRS_RECOVERY_PLAN_SHA = (
    "f919278c12618412c6affdc4c959f28f6d786dea511e38f788341d9e0a998654"
)
VISION_RECOVERY_PLAN_SHA = (
    "6a4a8eb2ee3fe78b11b3165f11c509b0e4c1f86ff81df264ba7981d86e968f5c"
)
TASK_INVENTORY_SHA = "99bd8640be6dabe68fc534147490b2086ebcd4c073257e44a0cf1ec33ec4bbeb"
APPROVED_RECOVERY_PLANS = {
    "frs": {
        FRS_RECOVERY_PLAN_SHA: "frs_recovery1/recovery_plan.json",
        "0a1081520d100cf8978e3b5db794ffd1283293eb97bc59ed6d4162bb87ad63c2": "frs_recovery2/recovery_plan.json",
    },
    "vision": {
        VISION_RECOVERY_PLAN_SHA: "vision_recovery2/recovery_plan.json",
        "025b29e3b712d4301b0b6833cc200a7f3f8f2c1e61302117c1ab9e08d0213f86": "vision_recovery3/recovery_plan.json",
    },
}
WEIGHTS = "c0c8d17e2c875a7f60c919e3c8f4545c29ed97f20570ff4d75bad5b184e244d6"
FRS_PROTOCOL = "56cf1cf89fe8503417f5f3e5a86a8f723ddc0877d273fe4d9696eb87dec89947"
VISION_PROTOCOL = "4a9a478d90b03d25ae72fc65ceb76a470a2958cffe5ee9216d7e1c8ef325fd81"
SUITES = ("libero_goal_ood", "libero_spatial_ood")
IDENTITIES = [(suite, task) for suite in SUITES for task in range(10)]
FRS_METHODS = ("native_euler10", "native_repeated_noise", "astra_frs")
VISION_METHODS = (
    "native",
    "native_retry",
    "random_vei",
    "random_vli",
    "astra_vei",
    "astra_vli",
)
LABELS = {
    "native_euler10": "Native Euler-10",
    "native_repeated_noise": "Repeated-noise control",
    "astra_frs": "Astra FRS",
    "native": "Shared native baseline",
    "native_retry": "Native retry",
    "random_vei": "Random VEI",
    "random_vli": "Random VLI",
    "astra_vei": "Astra VEI",
    "astra_vli": "Astra VLI",
}
TOKEN_FIELDS = ("input_tokens", "output_tokens", "reasoning_tokens", "total_tokens")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path, expected=None):
    data = Path(path).read_bytes()
    require(expected is None or sha(data) == expected, "Bound JSON digest changed")
    return json.loads(data), sha(data)


def inside(root, relative):
    relative = Path(relative)
    require(
        not relative.is_absolute() and ".." not in relative.parts,
        "Unsafe evidence path",
    )
    result = (root / relative).resolve()
    require(result.is_relative_to(root.resolve()), "Evidence path escaped its root")
    return result


def number(value, integer=False):
    require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        "Expected a numeric metric",
    )
    require(math.isfinite(value) and value >= 0, "Invalid metric")
    if integer:
        require(int(value) == value, "Expected an integer metric")
        return int(value)
    return value


def nullable_number(value, integer=False):
    return None if value is None else number(value, integer)


def identity(row):
    key = (row["suite"], row["task_id"])
    require(key in IDENTITIES, "Unexpected suite/task identity")
    return key


def task_label(key):
    return f"{'Goal' if key[0] == SUITES[0] else 'Spatial'}-OOD {key[1]}"


def benchmark_text(value):
    require(
        isinstance(value, str) and 0 < len(value) <= 500,
        "Invalid benchmark instruction",
    )
    require(
        not re.search(r"[\x00-\x1f]|https?://|/Users/|/workspace/|\.deps/", value),
        "Unsafe public instruction",
    )
    return value


def accepted_source(value, equivalence):
    require(
        value == SOURCE
        or (
            value == RECOVERY_SOURCE
            and equivalence is not None
            and equivalence.get("proof_sha256") == EQUIVALENCE_SHA
        ),
        "Unapproved or unproven producer identity",
    )
    return dict(value)


def source_equivalence(ops):
    path = ops / "recovery_bundle/source_equivalence.json"
    if not path.is_file():
        return None
    proof, digest = read_json(path, EQUIVALENCE_SHA)
    require(
        proof["schema_version"] == "routing-recovery-source-equivalence-1.0"
        and proof["status"] == "passed_exact_reviewed_delta"
        and proof["baseline_source_identity"] == SOURCE
        and proof["recovery_source_identity"] == RECOVERY_SOURCE,
        "Recovery source equivalence mismatch",
    )
    scientific_files = (
        "astra_reversal/frs_experiment.py",
        "astra_reversal/frs_noise_policy.py",
        "astra_reversal/frs_operators.py",
        "astra_reversal/frs_agent.py",
        "astra_reversal/frs_guide.py",
        "astra_reversal/representation_search.py",
        "astra_reversal/representation_agent.py",
        "astra_reversal/codex_executor.py",
        "astra_reversal/configs/frs_codex_frozen_evaluation_v1.json",
        "astra_reversal/configs/vision_codex_representation_screen_v1.json",
    )
    unchanged = proof["unchanged_source_files"]
    require(
        all(name in unchanged for name in scientific_files), "Scientific source changed"
    )
    runtime = [
        item["path"]
        for item in proof["source_changes"]
        if not item["path"].startswith(("astra_reversal/reports/", "tests/"))
    ]
    require(
        set(runtime)
        == {
            "astra_reversal/codex_relay.py",
            "astra_reversal/osmo/frs_policy_improvement.py",
            "astra_reversal/osmo/representation_steering.py",
            "astra_reversal/osmo/task_selection.py",
        },
        "Unexpected recovery runtime delta",
    )
    return {
        "proof_sha256": digest,
        "reviewed_transition_policy_sha256": proof["reviewed_transition_policy_sha256"],
        "baseline_source_identity": SOURCE,
        "recovery_source_identity": RECOVERY_SOURCE,
        "unchanged_scientific_sources_sha256": {
            name: unchanged[name]["sha256"] for name in scientific_files
        },
        "unchanged_source_file_count": len(unchanged),
        "unchanged_payload_file_count": len(
            proof["unchanged_payload_files_including_assets"]
        ),
        "changed_runtime_files": runtime,
    }


def task_inventory():
    path = HERE.parent / "smart_system2_results/tasks.json"
    inventory, digest = read_json(path, TASK_INVENTORY_SHA)
    require(
        len(inventory["tasks"]) == 20
        and {identity(row) for row in inventory["tasks"]} == set(IDENTITIES),
        "Task inventory identity mismatch",
    )
    tasks = [
        {
            "suite": row["suite"],
            "task_id": row["task_id"],
            "label": task_label(identity(row)),
            "instruction": benchmark_text(row["instruction"]),
            "bddl_sha256": row["bddl_sha256"],
        }
        for row in sorted(
            inventory["tasks"], key=lambda r: IDENTITIES.index(identity(r))
        )
    ]
    require(
        all(re.fullmatch(r"[0-9a-f]{64}", row["bddl_sha256"]) for row in tasks),
        "Task inventory BDDL digest malformed",
    )
    return {
        "repository_file": "astra_reversal/reports/smart_system2_results/tasks.json",
        "repository_revision": RECOVERY_SOURCE["source_revision"],
        "file_sha256": digest,
        "release_revision": inventory["release"]["revision"],
        "tasks": tasks,
    }


def recovery_plan(ops, family, digest):
    approved = APPROVED_RECOVERY_PLANS[family]
    require(digest in approved, "Recovery plan has not been reviewed")
    plan, _ = read_json(inside(ops, approved[digest]), digest)
    require(
        plan["source_equivalence_proof_sha256"] == EQUIVALENCE_SHA
        and plan["new_source_identity"] == RECOVERY_SOURCE
        and plan["protocol_changes"] is False
        and plan["model_weight_changes"] is False,
        "Recovery plan changed scientific scope",
    )
    roots = plan["source_roots_in_predeclared_order"]
    require(len(roots) == len(set(roots)), "Recovery plan repeats a source root")
    original_roots = (
        {"frs_full"} if family == "frs" else {"vision_full", "vision_recovery1"}
    )
    return plan, {
        root: SOURCE if root in original_roots else RECOVERY_SOURCE for root in roots
    }


def median(values):
    return statistics.median(values) if values else None


def tokens_from_usage(usage):
    result = {}
    for field in TOKEN_FIELDS:
        entry = usage["tokens"][field]
        known = number(entry["sum"], True)
        complete = entry["complete"] is True
        missing = number(entry.get("missing_calls", 0), True)
        lower = number(entry.get("lower_bound_calls", 0), True)
        require(
            not complete or missing + lower == 0, "Contradictory token completeness"
        )
        result[field] = {
            "known_sum": known,
            "exact_total": known if complete else None,
            "unknown_usage_jobs": missing,
            "lower_bound_usage_jobs": lower,
        }
    return result


def sum_tokens(rows):
    return {
        field: {
            "known_sum": sum(row[field]["known_sum"] for row in rows),
            "exact_total": (
                sum(row[field]["exact_total"] for row in rows)
                if all(row[field]["exact_total"] is not None for row in rows)
                else None
            ),
            "unknown_usage_jobs": sum(row[field]["unknown_usage_jobs"] for row in rows),
            "lower_bound_usage_jobs": sum(
                row[field]["lower_bound_usage_jobs"] for row in rows
            ),
        }
        for field in TOKEN_FIELDS
    }


def acquisition_summary(row):
    """Deliberate whitelist: never include job IDs, origins, journals or strings."""
    counts = (
        "unique_invocation_ids",
        "confirmed_started_codex_jobs",
        "completed_started_jobs",
        "confirmed_not_started_claims",
        "unknown_launch_claims",
        "unfinished_started_claims",
        "completed_jobs_without_any_usage",
        "unknown_job_wall_count",
    )
    result = {key: number(row[key], True) for key in counts}
    result["known_job_wall_seconds"] = number(row["known_job_wall_seconds"])
    result["median_known_job_wall_seconds"] = nullable_number(
        row["median_known_job_wall_seconds"]
    )
    result["tokens"] = {
        field: {
            key: nullable_number(row["tokens"][field][key], True)
            for key in (
                "known_sum",
                "exact_total",
                "unknown_usage_jobs",
                "lower_bound_usage_jobs",
                "known_completed_job_sum",
                "known_unfinished_job_sum",
            )
        }
        for field in TOKEN_FIELDS
    }
    for token in result["tokens"].values():
        require(
            token["exact_total"] is None
            or (
                token["exact_total"] == token["known_sum"]
                and token["unknown_usage_jobs"] == token["lower_bound_usage_jobs"] == 0
            ),
            "Acquisition usage completeness mismatch",
        )
    return result


def coverage(rows):
    require(
        len(rows) == 20 and {identity(row) for row in rows} == set(IDENTITIES),
        "Coverage must name all 20 tasks once",
    )
    result = []
    for row in sorted(rows, key=lambda r: IDENTITIES.index(identity(r))):
        status = row["status"]
        public_status = (
            "audited"
            if status == "passed"
            else (
                "blocked" if "fail" in status or "block" in status else "awaiting_audit"
            )
        )
        result.append(
            {
                "suite": row["suite"],
                "task_id": row["task_id"],
                "label": task_label(identity(row)),
                "status": public_status,
            }
        )
    return result


def load_frs(
    progress_path,
    campaign,
    acquisition_path,
    expected_sha=None,
    *,
    ops=None,
    equivalence=None,
):
    progress, progress_sha = read_json(progress_path, expected_sha)
    is_merge = progress["schema_version"] == "codex-frs-private-recovery-merge-1.0"
    require(
        is_merge
        or progress["schema_version"] == "codex-frs-private-campaign-progress-1.0",
        "Unsupported FRS progress",
    )
    require(progress["expected_tasks"] == 20, "FRS expected task cohort changed")
    source_progress = {}
    if is_merge:
        require(
            ops is not None and equivalence is not None,
            "FRS recovery proof unavailable",
        )
        plan, expected_sources = recovery_plan(
            ops, "frs", progress["recovery_plan_sha256"]
        )
        require(
            progress["source_equivalence_proof_sha256"] == EQUIVALENCE_SHA
            and progress["source_roots_in_predeclared_order"]
            == plan["source_roots_in_predeclared_order"],
            "FRS recovery selection plan mismatch",
        )
        states = {row["source_root"]: row for row in progress["source_states"]}
        require(
            len(progress["source_states"]) == len(expected_sources)
            and set(states) == set(expected_sources),
            "FRS source state mismatch",
        )
        for root, row in states.items():
            expected_source = expected_sources[root]
            require(row["source_identity"] == expected_source, "FRS source mislabeled")
            accepted_source(expected_source, equivalence)
            if row["audit_progress_sha256"] is not None:
                doc, _ = read_json(
                    inside(ops, row["audit_progress_path"]),
                    row["audit_progress_sha256"],
                )
                require(
                    doc["source_identity"] == expected_source,
                    "FRS source progress identity mismatch",
                )
                source_progress[root] = doc
    else:
        accepted_source(progress["source_identity"], equivalence)
    public_coverage = coverage(progress["task_coverage"])
    tasks, contexts, all_rollouts = [], {}, []
    protocols = set()
    for task in progress["tasks"]:
        key = identity(task)
        producer = accepted_source(task["source_identity"], equivalence)
        task_progress_path, task_campaign = progress_path, campaign
        if is_merge:
            root = task["source_root"]
            require(
                root in source_progress,
                "Selected FRS task has no bound source progress",
            )
            state = states[root]
            require(
                task["source_identity"] == state["source_identity"]
                and task["audit_progress_path"] == state["audit_progress_path"]
                and task["audit_progress_sha256"] == state["audit_progress_sha256"],
                "FRS selected source binding mismatch",
            )
            matches = [
                row for row in source_progress[root]["tasks"] if identity(row) == key
            ]
            require(
                len(matches) == 1
                and all(task.get(k) == v for k, v in matches[0].items()),
                "FRS merge changed audited task evidence",
            )
            task_progress_path = inside(ops, task["audit_progress_path"])
            task_campaign = inside(ops, root)
        else:
            require(
                producer == progress["source_identity"],
                "FRS task/producer identity mismatch",
            )
        require(key not in contexts, "Duplicate FRS task")
        require(
            task["phase"] == "evaluation"
            and task["evaluated_reset_ids"] == list(range(1, 11)),
            "FRS reset cohort mismatch",
        )
        proof = inside(
            task_progress_path.parent,
            f"by_archive_sha256/{task['archive_sha256']}/{task['auditor_id']}",
        )
        binding, _ = read_json(proof / "binding.json")
        for field in (
            "archive_sha256",
            "archive_audit_sha256",
            "scientific_audit_sha256",
            "source_identity",
            "suite",
            "task_id",
            "worker",
            "evaluated_reset_ids",
        ):
            require(binding[field] == task[field], "FRS case binding mismatch")
        audit, _ = read_json(proof / "frs_audit.json", task["scientific_audit_sha256"])
        archive, _ = read_json(
            proof / "archive_audit.json", task["archive_audit_sha256"]
        )
        require(
            audit["status"] == archive["status"] == "passed", "FRS audit did not pass"
        )
        require(
            archive["archive_sha256"] == task["archive_sha256"],
            "FRS archive audit mismatch",
        )
        task_dir = inside(
            task_campaign, f"extracted/worker_{task['worker']}/results/task_{key[1]}"
        )
        summary, summary_sha = read_json(
            task_dir / "summary.json", audit["summary_sha256"]
        )
        require(
            summary["status"] == "complete" and identity(summary) == key,
            "FRS task is not complete",
        )
        require(
            summary["protocol_sha256"] == FRS_PROTOCOL, "FRS frozen protocol mismatch"
        )
        protocols.add(summary["protocol_sha256"])
        weight, _ = read_json(
            task_dir / "frozen_weights_after.json",
            audit["input_file_sha256"]["frozen_weights_after.json"],
        )
        require(weight["sha256"] == WEIGHTS, "FRS frozen weight mismatch")
        rows = summary["physical_rollouts"]
        expected = {(method, reset) for method in FRS_METHODS for reset in range(1, 11)}
        seen = set()
        public_rows = []
        proof_rows = {r["physical_run_id"]: r for r in task["rollouts"]}
        for row in rows:
            method = row["method"]
            match = re.fullmatch(
                rf"{re.escape(method)}_state(\d+)_round0", row["attempt_id"]
            )
            require(
                method in FRS_METHODS and match is not None,
                "Unexpected FRS method or attempt",
            )
            reset = int(match[1])
            require(
                (method, reset) in expected and (method, reset) not in seen,
                "FRS paired reset mismatch",
            )
            seen.add((method, reset))
            proof_row = proof_rows[row["physical_run_id"]]
            require(
                proof_row["success"] == row["success"]
                and proof_row["method"] == method,
                "FRS summary/progress mismatch",
            )
            require(
                proof_row["reset_sha256"] == row["reset_audit"]["sha256"],
                "FRS reset binding mismatch",
            )
            require(
                row["evaluation"] is True and row["round_index"] == 0,
                "FRS includes adaptation",
            )
            usage = row["provider_usage"]
            public = {
                "suite": key[0],
                "task_id": key[1],
                "method": method,
                "reset_id": reset,
                "success": row["success"] is True,
                "actions": number(row["actions_executed"], True),
                "rollout_seconds": number(row["wall_seconds"]),
                "reasoner_wait_seconds": number(usage["latency_seconds"]),
                "codex_jobs": number(usage["provider_calls"], True),
                "tokens": tokens_from_usage(usage),
            }
            public_rows.append(public)
        require(seen == expected and len(proof_rows) == 30, "Incomplete FRS task")
        for reset in range(1, 11):
            require(
                len(
                    {
                        row["reset_audit"]["sha256"]
                        for row in rows
                        if row["attempt_id"].endswith(f"_state{reset}_round0")
                    }
                )
                == 1,
                "FRS methods have unmatched resets",
            )
        waits = []
        provider = task_dir / "provider.jsonl"
        require(
            file_sha(provider) == audit["provider_sha256"],
            "FRS provider ledger changed",
        )
        with provider.open() as stream:
            for line in stream:
                row = json.loads(line)
                if row.get("provider_call"):
                    waits.append(number(row["latency_seconds"]))
        contexts[key] = {
            "task_dir": task_dir,
            "events_sha": audit["events_sha256"],
            "members": archive["members"],
            "summary": summary,
            "waits": waits,
            "producer_identity": producer,
        }
        all_rollouts.extend(public_rows)
        tasks.append(
            {
                "suite": key[0],
                "task_id": key[1],
                "label": task_label(key),
                "instruction": benchmark_text(summary["instruction"]),
                "producer_identity": producer,
                "archive_sha256": task["archive_sha256"],
                "summary_sha256": summary_sha,
                "scientific_audit_sha256": task["scientific_audit_sha256"],
                "archive_audit_sha256": task["archive_audit_sha256"],
                "methods": [
                    {
                        "method": method,
                        "successes": sum(
                            r["success"] for r in public_rows if r["method"] == method
                        ),
                        "denominator": 10,
                    }
                    for method in FRS_METHODS
                ],
            }
        )
    count = len(tasks)
    require(
        count
        == progress["audited_completed_tasks"]
        == sum(c["status"] == "audited" for c in public_coverage),
        "FRS coverage count mismatch",
    )
    require(
        {identity(t) for t in tasks}
        == {identity(c) for c in public_coverage if c["status"] == "audited"},
        "FRS audited identities mismatch",
    )
    complete = count == 20
    require(
        progress["full_campaign_complete"] is complete, "FRS full-cohort gate mismatch"
    )
    methods = []
    baseline = sum(r["success"] for r in all_rollouts if r["method"] == FRS_METHODS[0])
    for method in FRS_METHODS:
        rows = [r for r in all_rollouts if r["method"] == method]
        successes = sum(r["success"] for r in rows)
        require(
            progress["methods"][method]["simulator_successes"] == successes
            and progress["methods"][method]["evaluated_rollouts"] == len(rows),
            "FRS aggregate mismatch",
        )
        methods.append(
            {
                "method": method,
                "label": LABELS[method],
                "successes": successes,
                "denominator": len(rows),
                "full_success_rate": successes / 200 if complete else None,
                "successes_vs_baseline": successes - baseline,
                "physical_rollouts": len(rows),
                "actions": sum(r["actions"] for r in rows),
                "codex_jobs": sum(r["codex_jobs"] for r in rows),
                "tokens": sum_tokens([r["tokens"] for r in rows]),
                "rollout_seconds": sum(r["rollout_seconds"] for r in rows),
                "median_rollout_seconds": median([r["rollout_seconds"] for r in rows]),
                "reasoner_wait_seconds": sum(r["reasoner_wait_seconds"] for r in rows),
                "median_reasoner_wait_per_call_seconds": median(
                    [w for ctx in contexts.values() for w in ctx["waits"]]
                )
                if method == "astra_frs"
                else None,
                "success_revision_histogram": {
                    "0": successes,
                    "1": 0,
                    "2": 0,
                    "none": len(rows) - successes,
                },
            }
        )
    acquisition = None
    acquisition_status = "unavailable"
    acquisition_sha = None
    if is_merge:
        ledger = progress["acquisition_cost_ledger"]
        require(not ledger["integrity_errors"], "FRS acquisition integrity failure")
        acquisition = {
            "audited": acquisition_summary(ledger["canonical_acquisition"]),
            "other": acquisition_summary(
                ledger["interrupted_or_duplicate_acquisition"]
            ),
            "all": acquisition_summary(ledger["all_local_acquisition"]),
        }
        canonical_tokens = sum_tokens([row["tokens"] for row in methods])[
            "total_tokens"
        ]
        require(
            all(
                acquisition["audited"]["tokens"]["total_tokens"][key]
                == canonical_tokens[key]
                for key in ("known_sum", "exact_total")
            ),
            "FRS canonical acquisition mismatch",
        )
        acquisition_sha, acquisition_status = progress_sha, "bound"
    elif acquisition_path.is_file():
        cost, acquisition_sha = read_json(acquisition_path)
        if cost.get("audit_progress_sha256") != progress_sha:
            acquisition_status = "stale_audit_binding"
        else:
            require(
                cost["source_identity"] == progress["source_identity"]
                and not cost["integrity_errors"],
                "FRS acquisition integrity failure",
            )
            acquisition = {
                "audited": acquisition_summary(
                    cost["audited_completed_task_acquisition"]
                ),
                "other": acquisition_summary(cost["ongoing_or_unsealed_acquisition"]),
                "all": acquisition_summary(cost["all_local_acquisition"]),
            }
            require(
                acquisition["audited"]["tokens"]["total_tokens"]["exact_total"]
                == methods[-1]["tokens"]["total_tokens"]["exact_total"],
                "FRS cost/score cohort mismatch",
            )
            acquisition_status = "bound"
    return {
        "audited_tasks": count,
        "expected_tasks": 20,
        "full_campaign_complete": complete,
        "coverage": public_coverage,
        "tasks": tasks,
        "methods": methods,
        "rollouts": all_rollouts,
        "physical_rollouts": len(all_rollouts),
        "progress_sha256": progress_sha,
        "protocol_sha256": FRS_PROTOCOL,
        "acquisition": acquisition,
        "acquisition_status": acquisition_status,
        "acquisition_receipt_sha256": acquisition_sha,
        "evidence_kind": "recovery_merge" if is_merge else "single_campaign_audit",
        "recovery_plan_sha256": progress["recovery_plan_sha256"] if is_merge else None,
        "source_equivalence_proof_sha256": EQUIVALENCE_SHA if is_merge else None,
        "source_audit_progress_sha256": [
            row["audit_progress_sha256"]
            for row in progress["source_states"]
            if row["audit_progress_sha256"] is not None
        ]
        if is_merge
        else [progress_sha],
    }, contexts


def load_vision(merge_path, ops, expected_sha=None, *, equivalence=None):
    merged, merge_sha = read_json(merge_path, expected_sha)
    require(
        merged["schema_version"] == "codex-vision-private-recovery-merge-1.0",
        "Unsupported vision merge",
    )
    require(
        merged["protocol_sha256"] == VISION_PROTOCOL, "Vision frozen protocol mismatch"
    )
    multi = merged.get("multi_source_proof")
    sources = {"vision_full": SOURCE, "vision_recovery1": SOURCE}
    if multi is not None:
        require(equivalence is not None, "Vision recovery proof unavailable")
        _, sources = recovery_plan(ops, "vision", multi["recovery_plan_sha256"])
        require(
            multi["source_equivalence_proof_sha256"] == EQUIVALENCE_SHA
            and merged["payload_identity"] is None
            and merged["baseline_payload_identity"] == SOURCE,
            "Vision multi-source transition mismatch",
        )
        require(
            merged["source_identities"] == sources,
            "Vision producer source map mismatch",
        )
    else:
        require(
            merged["payload_identity"] == SOURCE, "Vision producer identity mismatch"
        )
    require(
        merged["source_roots_in_predeclared_order"] == list(sources),
        "Vision selection order changed",
    )
    source_progress = {}
    seen_sources = set()
    for source in merged["source_audit_progress"]:
        root = source["source_root"]
        require(
            root in sources and root not in seen_sources,
            "Vision source audit map mismatch",
        )
        seen_sources.add(root)
        producer = source.get("source_identity", SOURCE)
        require(producer == sources[root], "Vision source progress mislabeled")
        accepted_source(producer, equivalence)
        if source["sha256"] is not None:
            doc, _ = read_json(inside(ops, source["path"]), source["sha256"])
            require(
                doc["source_identity"] == producer,
                "Vision source progress identity changed",
            )
            source_progress[root] = doc
    require(merged["native_tensor_sha256"] == WEIGHTS, "Vision frozen weight mismatch")
    require(
        not merged["acquisition_cost_ledger"]["integrity_errors"],
        "Vision acquisition integrity failure",
    )
    public_coverage = coverage(merged["case_coverage"])
    contexts, tasks, case_rows = {}, [], []
    selected = {
        identity(row): row
        for row in merged["case_coverage"]
        if row["status"] == "passed"
    }
    groups = {}
    for row in merged["case_method_rows"]:
        key = identity(row)
        require(
            key in selected
            and row["archive_sha256"] == selected[key]["archive_sha256"],
            "Vision selected case mismatch",
        )
        require(row["method"] in VISION_METHODS, "Unapproved vision arm")
        groups.setdefault(key, []).append(row)
    for key, rows in groups.items():
        require(
            len(rows) == 6 and {r["method"] for r in rows} == set(VISION_METHODS),
            "Incomplete vision arm comparison",
        )
        first = rows[0]
        root = first["source_root"]
        require(
            root in source_progress, "Selected vision case has no bound source progress"
        )
        producer = accepted_source(first.get("source_identity", SOURCE), equivalence)
        require(producer == sources[root], "Vision selected producer mismatch")
        if multi is not None:
            require(
                selected[key]["source_identity"] == producer,
                "Vision coverage producer mismatch",
            )
        require(
            all(
                r["independent_audit_directory"] == first["independent_audit_directory"]
                and r["source_root"] == first["source_root"]
                and r.get("source_identity", SOURCE) == producer
                for r in rows
            ),
            "Vision method evidence mismatch",
        )
        proof = inside(ops, first["independent_audit_directory"])
        require(
            proof.is_relative_to(inside(ops, root)),
            "Vision proof escaped selected source",
        )
        binding, binding_sha = read_json(proof / "binding.json")
        require(
            binding["complete"] is True
            and binding["source_identity"] == producer
            and identity(binding) == key,
            "Vision binding incomplete",
        )
        require(
            binding["archive_sha256"] == first["archive_sha256"],
            "Vision archive binding mismatch",
        )
        array_audit, _ = read_json(
            proof / "array_audit.json", binding["array_audit_sha256"]
        )
        codex_audit, _ = read_json(
            proof / "codex_audit.json", binding["codex_audit_sha256"]
        )
        require(
            array_audit["status"] == "passed"
            and codex_audit["status"] == "complete"
            and codex_audit["complete"] is True,
            "Vision audits incomplete",
        )
        require(
            array_audit["identities"]["protocol_sha256"] == VISION_PROTOCOL
            and array_audit["identities"]["payload_sha256"]
            == producer["payload_sha256"]
            and array_audit["identities"]["source_revision"]
            == producer["source_revision"],
            "Vision audit protocol mismatch",
        )
        task_dir = inside(
            ops,
            f"{first['source_root']}/extracted/worker_{first['worker']}/results/task_{key[1]}",
        )
        summary, summary_sha = read_json(
            task_dir / "summary.json", array_audit["input_file_sha256"]["summary.json"]
        )
        require(
            summary_sha == codex_audit["source_file_sha256"]["summary.json"]["sha256"],
            "Vision audits bind different summaries",
        )
        require(
            summary["status"] == "complete"
            and identity(summary) == key
            and summary["seed"] == 47
            and summary["initial_state_id"] == 0,
            "Vision cohort mismatch",
        )
        task_rows = []
        for row in rows:
            revision = row["earliest_success_revision"]
            require(
                revision in (None, 0, 1, 2)
                and (revision is not None) == row["success"],
                "Vision success revision mismatch",
            )
            public = {
                "suite": key[0],
                "task_id": key[1],
                "method": row["method"],
                "success": row["success"] is True,
                "earliest_success_revision": revision,
                "physical_rollouts": number(row["actual_physical_rollouts"], True),
                "codex_jobs": number(row["actual_calls"], True),
                "tokens": tokens_from_usage(row["provider_usage"]),
                "rollout_seconds": number(row["actual_rollout_wall_seconds"]),
                "reasoner_wait_seconds": number(row["actual_reasoner_wait_seconds"]),
            }
            require(
                public["tokens"]["total_tokens"]["exact_total"]
                == row["actual_total_tokens"],
                "Vision case token claim mismatch",
            )
            case_rows.append(public)
            task_rows.append(public)
        tasks.append(
            {
                "suite": key[0],
                "task_id": key[1],
                "label": task_label(key),
                "instruction": benchmark_text(summary["instruction"]),
                "archive_sha256": first["archive_sha256"],
                "producer_identity": producer,
                "summary_sha256": summary_sha,
                "binding_sha256": binding_sha,
                "array_audit_sha256": binding["array_audit_sha256"],
                "codex_audit_sha256": binding["codex_audit_sha256"],
                "methods": [
                    {
                        "method": row["method"],
                        "successes": int(row["success"]),
                        "denominator": 1,
                        "earliest_success_revision": row["earliest_success_revision"],
                    }
                    for row in task_rows
                ],
            }
        )
        contexts[key] = {
            "task_dir": task_dir,
            "summary": summary,
            "events_sha": array_audit["input_file_sha256"]["events.jsonl"],
            "rows": rows,
            "array_audit": array_audit,
            "producer_identity": producer,
        }
    count = len(tasks)
    require(
        set(groups) == set(selected) and count == merged["canonical_audited_cases"],
        "Vision audited identities mismatch",
    )
    complete = count == 20
    require(
        merged["full_campaign_complete"] is complete, "Vision full-cohort gate mismatch"
    )
    methods = []
    baseline = sum(row["success"] for row in case_rows if row["method"] == "native")
    require(
        {m["method"] for m in merged["method_rows"]} == set(VISION_METHODS),
        "Vision merged arm set mismatch",
    )
    for method in VISION_METHODS:
        source = next(m for m in merged["method_rows"] if m["method"] == method)
        rows = [row for row in case_rows if row["method"] == method]
        successes = sum(r["success"] for r in rows)
        require(
            source["audited_successes"] == successes
            and source["audited_case_denominator"] == count,
            "Vision aggregate mismatch",
        )
        tokens = sum_tokens([r["tokens"] for r in rows])
        require(
            source["actual_total_tokens"] == tokens["total_tokens"]["exact_total"],
            "Vision aggregate token mismatch",
        )
        histogram = {
            str(i): sum(r["earliest_success_revision"] == i for r in rows)
            for i in range(3)
        }
        histogram["none"] = sum(r["earliest_success_revision"] is None for r in rows)
        require(
            histogram == source["earliest_success_revision_histogram"],
            "Vision attempt histogram mismatch",
        )
        methods.append(
            {
                "method": method,
                "label": LABELS[method],
                "successes": successes,
                "denominator": count,
                "full_success_rate": successes / 20 if complete else None,
                "successes_vs_baseline": successes - baseline,
                "physical_rollouts": sum(r["physical_rollouts"] for r in rows),
                "actions": number(source["actual_actions"], True),
                "codex_jobs": sum(r["codex_jobs"] for r in rows),
                "tokens": tokens,
                "rollout_seconds": sum(r["rollout_seconds"] for r in rows),
                "median_rollout_seconds": nullable_number(
                    source["median_physical_rollout_seconds"]
                ),
                "reasoner_wait_seconds": sum(r["reasoner_wait_seconds"] for r in rows),
                "median_reasoner_wait_per_call_seconds": nullable_number(
                    source["median_reasoner_wait_seconds_per_call"]
                ),
                "success_revision_histogram": histogram,
            }
        )
    ledger = merged["acquisition_cost_ledger"]
    acquisition = {
        "audited": acquisition_summary(ledger["canonical_case_acquisition"]),
        "other": acquisition_summary(ledger["noncanonical_research_overhead"]),
        "all": acquisition_summary(ledger["all_local_acquisition"]),
    }
    canonical_tokens = sum_tokens([m["tokens"] for m in methods])["total_tokens"]
    require(
        all(
            acquisition["audited"]["tokens"]["total_tokens"][field]
            == canonical_tokens[field]
            for field in ("known_sum", "exact_total")
        ),
        "Vision canonical acquisition mismatch",
    )
    return {
        "audited_tasks": count,
        "expected_tasks": 20,
        "full_campaign_complete": complete,
        "coverage": public_coverage,
        "tasks": tasks,
        "methods": methods,
        "case_rows": case_rows,
        "physical_rollouts": sum(m["physical_rollouts"] for m in methods),
        "merge_sha256": merge_sha,
        "protocol_sha256": VISION_PROTOCOL,
        "recovery_plan_sha256": multi["recovery_plan_sha256"]
        if multi is not None
        else merged["recovery_plan_sha256"],
        "source_equivalence_proof_sha256": EQUIVALENCE_SHA
        if multi is not None
        else None,
        "acquisition": acquisition,
        "acquisition_status": "bound",
        "audit_progress_sha256": [
            row["sha256"]
            for row in merged["source_audit_progress"]
            if row["sha256"] is not None
        ],
    }, contexts


class Media:
    def __init__(self, output, limit_bytes):
        self.output, self.limit_bytes, self.used = output, limit_bytes, 0
        self.inventory = {}

    def write(self, data, extension):
        digest = sha(data)
        name = f"media/{digest[:24]}.{extension}"
        if name not in self.inventory:
            require(
                self.used + len(data) <= self.limit_bytes,
                "Selected media exceeds configured byte limit",
            )
            atomic_write(self.output / name, data)
            self.used += len(data)
            self.inventory[name] = {"sha256": digest, "bytes": len(data)}
        return {"path": name, **self.inventory[name]}

    def video(self, path, expected_sha, expected_bytes=None):
        require(path.suffix == ".mp4", "Unexpected video type")
        data = path.read_bytes()
        require(
            sha(data) == expected_sha
            and (expected_bytes is None or len(data) == expected_bytes),
            "Sealed video changed",
        )
        return self.write(data, "mp4")

    def frame(self, task_dir, ref):
        import numpy as np
        from PIL import Image

        require(
            re.fullmatch(r"arrays/\d+\.npy", ref["array"]) is not None,
            "Unsafe frame array reference",
        )
        path = inside(task_dir, ref["array"])
        frame = np.load(path, allow_pickle=False)
        require(
            frame.shape == (224, 224, 3) and frame.dtype == np.uint8,
            "Unexpected policy image layout",
        )
        require(
            list(frame.shape) == ref["shape"] and str(frame.dtype) == ref["dtype"],
            "Image reference mismatch",
        )
        array_sha = sha(
            b"array:"
            + str((frame.dtype.str, frame.shape)).encode()
            + np.ascontiguousarray(frame).tobytes()
        )
        require(array_sha == ref["sha256"], "Image pixels changed after audit")
        buffer = io.BytesIO()
        Image.fromarray(frame).save(buffer, format="PNG")
        return {
            **self.write(buffer.getvalue(), "png"),
            "source_array_sha256": array_sha,
            "width": 224,
            "height": 224,
        }


def atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".report-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def initial_frame_event(context, attempt, frs):
    path = context["task_dir"] / "events.jsonl"
    require(file_sha(path) == context["events_sha"], "Audited event trace changed")
    with path.open() as stream:
        for line in stream:
            event = json.loads(line)
            if (
                event.get("kind")
                == ("generation" if frs else "representation_generation")
                and event.get("attempt_id") == attempt
                and event.get("step" if frs else "observation_step") == 0
            ):
                return event
    raise ValueError("Audited initial observation unavailable")


def select_media(
    frs, frs_contexts, vision, vision_contexts, ops, media, examples, videos
):
    galleries = {"frs": [], "vision": []}
    if examples == 0:
        return galleries
    # Deterministic selection: first audited identity/reset, then first baseline
    # failure. Selection never uses an assisted method's result.
    frs_candidates = sorted(
        [r for r in frs["rollouts"] if r["method"] == "native_euler10"],
        key=lambda r: (IDENTITIES.index(identity(r)), r["reset_id"]),
    )
    vision_candidates = sorted(
        [r for r in vision["case_rows"] if r["method"] == "native"],
        key=lambda r: IDENTITIES.index(identity(r)),
    )
    for family, candidates, contexts in (
        ("frs", frs_candidates, frs_contexts),
        ("vision", vision_candidates, vision_contexts),
    ):
        if not candidates:
            continue
        selected = [candidates[0]]
        failed = next((r for r in candidates[1:] if not r["success"]), None)
        if failed:
            selected.append(failed)
        for baseline in selected[:examples]:
            key = identity(baseline)
            context = contexts[key]
            is_frs = family == "frs"
            reset = baseline.get("reset_id", 0)
            attempt = (
                f"native_euler10_state{reset}_round0" if is_frs else "native_revision0"
            )
            event = initial_frame_event(context, attempt, is_frs)
            item = {
                "suite": key[0],
                "task_id": key[1],
                "label": task_label(key),
                "reset_id": reset,
                "instruction": benchmark_text(context["summary"]["instruction"]),
                "producer_identity": context["producer_identity"],
                "selection": "first_audited_identity"
                if baseline is selected[0]
                else "first_native_failure",
                "external": media.frame(
                    context["task_dir"], event["observation"]["observation/image"]
                ),
                "wrist": media.frame(
                    context["task_dir"], event["observation"]["observation/wrist_image"]
                ),
                "videos": [],
            }
            if videos and is_frs:
                for method in FRS_METHODS:
                    name = f"{method}_state{reset}_round0.mp4"
                    member = context["members"][f"task_{key[1]}/{name}"]
                    rollout = next(
                        r
                        for r in frs["rollouts"]
                        if identity(r) == key
                        and r["reset_id"] == reset
                        and r["method"] == method
                    )
                    item["videos"].append(
                        {
                            "method": method,
                            "revision": 0,
                            "success": rollout["success"],
                            **media.video(
                                context["task_dir"] / name,
                                member["sha256"],
                                member["bytes"],
                            ),
                        }
                    )
            elif videos:
                for row in sorted(
                    context["rows"], key=lambda r: VISION_METHODS.index(r["method"])
                ):
                    for video in row["physical_videos"]:
                        item["videos"].append(
                            {
                                "method": row["method"],
                                "revision": video["revision"],
                                "success": video["success"],
                                **media.video(
                                    inside(ops, video["path"]),
                                    video["sha256"],
                                    video["bytes"],
                                ),
                            }
                        )
            galleries[family].append(item)
    return galleries


def public_scan(value):
    """A secondary leak guard, in addition to construction by field whitelist."""
    forbidden_keys = {
        "jobs",
        "origins",
        "request",
        "prompt",
        "events",
        "raw_usage_events",
        "invocation_id",
        "workflow",
        "source_root",
        "independent_audit_directory",
        "justification",
        "reasoning",
    }
    if isinstance(value, dict):
        require(not set(value) & forbidden_keys, "Private field in public export")
        for child in value.values():
            public_scan(child)
    elif isinstance(value, list):
        for child in value:
            public_scan(child)
    elif isinstance(value, str):
        require(
            not re.search(
                r"https?://|file://|/Users/|/workspace/|\.deps/|Bearer\s|X-Amz-|[?&](?:token|signature)=",
                value,
                re.I,
            ),
            "Private path or URL in public export",
        )


def monitored_input(
    ops, state, family, explicit, fallback, recovery_state=None, after_dns_state=None
):
    if explicit is not None:
        return explicit.resolve(), None
    for candidate in (after_dns_state, recovery_state):
        entry = (candidate or {}).get("merges", {}).get(family, {})
        if entry.get("status") == "verified_report":
            require(entry.get("report_path"), "Verified monitor input has no path")
            path_key, sha_key = "report_path", "report_sha256"
            break
    else:
        if family == "frs":
            entry = state.get("families", {}).get("frs_full", {})
            path_key, sha_key = "audit_progress_path", "audit_progress_sha256"
        else:
            entry = state.get("vision_merge", {})
            path_key, sha_key = "report_path", "report_sha256"
    if not entry.get(path_key):
        return fallback.resolve(), None
    path = Path(entry[path_key])
    path = path.resolve() if path.is_absolute() else inside(ops, path)
    require(path.is_relative_to(ops), "Monitor input escaped private operations root")
    digest = entry.get(sha_key)
    require(
        isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest),
        "Monitor input has no valid receipt hash",
    )
    return path, digest


def build(args):
    ops = args.ops_root.resolve()
    state, state_sha = {}, None
    if (ops / "postprocess_state.json").is_file():
        state, state_sha = read_json(ops / "postprocess_state.json")
        require(
            state.get("schema_version") == "codex-private-postprocessing-1.0",
            "Unsupported monitor state schema",
        )
    recovery_state, recovery_state_sha = {}, None
    if (ops / "recovery_postprocess_state.json").is_file():
        recovery_state, recovery_state_sha = read_json(
            ops / "recovery_postprocess_state.json"
        )
        require(
            recovery_state.get("schema_version")
            == "codex-routing-recovery-monitor-1.0",
            "Unsupported recovery monitor state schema",
        )
    after_dns_state, after_dns_state_sha = {}, None
    if (ops / "recovery_after_dns_postprocess_state.json").is_file():
        after_dns_state, after_dns_state_sha = read_json(
            ops / "recovery_after_dns_postprocess_state.json"
        )
        require(
            after_dns_state.get("schema_version")
            == "codex-routing-recovery-monitor-1.0",
            "Unsupported post-DNS recovery monitor state schema",
        )
    progress, progress_sha = monitored_input(
        ops,
        state,
        "frs",
        getattr(args, "frs_merge", None) or args.frs_progress,
        ops / "frs_full/offline_frs_audits/progress.json",
        recovery_state,
        after_dns_state,
    )
    merge, merge_sha = monitored_input(
        ops,
        state,
        "vision",
        args.vision_merge,
        ops / "vision_recovery1/merged_vision/provisional.json",
        recovery_state,
        after_dns_state,
    )
    acquisition = (
        args.frs_acquisition or ops / "frs_full/acquisition_cost.json"
    ).resolve()
    campaign = (args.frs_campaign or ops / "frs_full").resolve()
    output = args.output.resolve()
    require(
        output.is_relative_to(HERE),
        "Report output must remain within this report directory",
    )
    require(
        0 <= args.examples_per_cohort <= 2 and 0 <= args.max_media_mib <= 256,
        "Media limits out of bounds",
    )
    equivalence = source_equivalence(ops)
    frs, frs_contexts = load_frs(
        progress, campaign, acquisition, progress_sha, ops=ops, equivalence=equivalence
    )
    vision, vision_contexts = load_vision(
        merge, ops, merge_sha, equivalence=equivalence
    )
    inventory = task_inventory()
    by_identity = {identity(row): row for row in inventory["tasks"]}
    for cohort, contexts in ((frs, frs_contexts), (vision, vision_contexts)):
        for task in cohort["tasks"]:
            key = identity(task)
            require(
                task["instruction"] == by_identity[key]["instruction"],
                "Audited instruction differs from committed task inventory",
            )
            require(
                all(
                    row["reset_audit"]["bddl_sha256"] == by_identity[key]["bddl_sha256"]
                    for row in contexts[key]["summary"]["physical_rollouts"]
                ),
                "Audited task BDDL differs from committed task inventory",
            )
        for row in cohort["coverage"]:
            row["instruction"] = by_identity[identity(row)]["instruction"]
    producers = [
        {
            "label": label,
            "identity": producer,
            "frs_audited_tasks": sum(
                task["producer_identity"] == producer for task in frs["tasks"]
            ),
            "vision_audited_tasks": sum(
                task["producer_identity"] == producer for task in vision["tasks"]
            ),
        }
        for label, producer in (
            ("Original producer", SOURCE),
            ("Routing/transport recovery producer", RECOVERY_SOURCE),
        )
        if producer == SOURCE or equivalence is not None
    ]
    media = Media(output, int(args.max_media_mib * 1024 * 1024))
    galleries = select_media(
        frs,
        frs_contexts,
        vision,
        vision_contexts,
        ops,
        media,
        args.examples_per_cohort,
        not args.no_videos,
    )
    report = {
        "schema_version": "codex-full-evaluation-public-1.1",
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "producer_identities": producers,
        "source_equivalence": equivalence,
        "task_inventory": inventory,
        "frozen_weight_sha256": WEIGHTS,
        "configured_model": "gpt-6-astra",
        "reasoning_effort": "medium",
        "backend": "Codex CLI",
        "hardware": "NVIDIA L40S on OSMO",
        "frs": frs,
        "vision": vision,
        "media": galleries,
        "media_bytes": media.used,
        "media_inventory": media.inventory,
        "builder_sha256": file_sha(Path(__file__)),
        "monitor_state_sha256": state_sha,
        "recovery_monitor_state_sha256": recovery_state_sha,
        "after_dns_recovery_monitor_state_sha256": after_dns_state_sha,
    }
    public_scan(report)
    encoded = (
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    )
    inline = json.dumps(report, separators=(",", ":"), allow_nan=False).replace(
        "<", "\\u003c"
    )
    template = (HERE / "report.html").read_text()
    require(
        template.count("__REPORT_DATA__") == 1, "Report template placeholder mismatch"
    )
    html = template.replace("__REPORT_DATA__", inline)
    for diagram in ("frs-flow.svg", "vision-flow.svg"):
        html = html.replace(f"__{diagram}__", (HERE / diagram).read_text())
    atomic_write(output / "results.json", encoded)
    atomic_write(output / "index.html", html.encode())
    # Remove only stale, builder-owned content-addressed media after publishing.
    for path in (output / "media").glob("*"):
        if (
            re.fullmatch(r"[0-9a-f]{24}\.(png|mp4)", path.name)
            and f"media/{path.name}" not in media.inventory
        ):
            path.unlink()
    print(
        json.dumps(
            {
                "frs_audited_tasks": frs["audited_tasks"],
                "vision_audited_tasks": vision["audited_tasks"],
                "frs_full_complete": frs["full_campaign_complete"],
                "vision_full_complete": vision["full_campaign_complete"],
                "public_media_bytes": media.used,
                "frs_acquisition_status": frs["acquisition_status"],
            },
            sort_keys=True,
        )
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ops-root", type=Path, default=DEFAULT_OPS)
    frs_input = parser.add_mutually_exclusive_group()
    frs_input.add_argument("--frs-progress", type=Path)
    frs_input.add_argument("--frs-merge", type=Path)
    parser.add_argument("--frs-campaign", type=Path)
    parser.add_argument("--frs-acquisition", type=Path)
    parser.add_argument("--vision-merge", type=Path)
    parser.add_argument("--output", type=Path, default=HERE)
    parser.add_argument("--examples-per-cohort", type=int, default=2)
    parser.add_argument("--max-media-mib", type=float, default=120)
    parser.add_argument("--no-videos", action="store_true")
    args = parser.parse_args()
    try:
        build(args)
    except (ValueError, KeyError, OSError, TypeError) as error:
        # Do not echo exception values from private inputs/paths into logs.
        print(
            f"Report build failed ({type(error).__name__}); no new report index published."
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
