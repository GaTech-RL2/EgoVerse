"""Summarize audited development behavior without any new model or simulator call.

The caller supplies the two completed recovery case directories. Public audit
receipts must already exist in the adjacent audits directory. Full raw provider
envelopes and request image data are never copied into the output.
"""

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

from astra_reversal.interpolation_agent import parse_proposal, summarize_calls
from astra_reversal.records import file_sha256

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[3]
MODES = ("astra_tei", "astra_tli", "astra_tli_vision")
MODEL = "azure/openai/gpt-6-astra"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text())


def load_case(case, worker):
    """Reject incomplete cases or a changed artifact before creating public files."""
    audits = HERE.parent / "audits"
    receipt_path = audits / f"worker_{worker}_receipt.json"
    receipt = read_json(receipt_path)
    require(receipt["status"] == "passed", "Archive receipt must pass")
    arrays_path = audits / receipt["array_audit"]["file"]
    require(
        file_sha256(arrays_path) == receipt["array_audit"]["sha256"],
        "Array receipt hash",
    )
    arrays = read_json(arrays_path)
    require(
        arrays["status"] == "passed" and arrays["complete"],
        "Full array audit must pass",
    )
    metadata = next(row for row in receipt["feedback"] if row["case"] == case.name)
    feedback_path = audits / metadata["file"]
    require(file_sha256(feedback_path) == metadata["sha256"], "Feedback receipt hash")
    raw = gzip.decompress(feedback_path.read_bytes())
    require(
        hashlib.sha256(raw).hexdigest() == metadata["uncompressed_sha256"],
        "Feedback decompression hash",
    )
    feedback = json.loads(raw)
    require(feedback["status"] == "passed", "Full feedback audit must pass")
    source_files = {
        name: sha
        for name, sha in feedback["input_file_sha256"].items()
        if not name.startswith("arrays/")
    }
    for name, sha in source_files.items():
        require(file_sha256(case / name) == sha, f"Changed audited input: {name}")
    summary = read_json(case / "summary.json")
    require(
        summary["status"] == "complete" and summary["development"],
        "Complete development case required",
    )
    require(summary["episode_id"] == feedback["episode_id"], "Episode identity")
    require(
        summary["protocol_sha256"] == feedback["protocol_sha256"], "Protocol identity"
    )
    require(
        summary["baseline"]["success"] is False, "This example requires failed baseline"
    )
    events = [
        json.loads(line) for line in (case / "events.jsonl").read_text().splitlines()
    ]
    requests = {
        row["request"]["request_fingerprint"]: row
        for row in events
        if row["kind"] == "interpolation_request"
    }
    require(len(requests) == feedback["counts"]["requests"], "Request coverage")
    return (
        summary,
        feedback,
        requests,
        {
            "workflow": receipt["workflow"],
            "archive_sha256": receipt["archive"]["sha256"],
            "payload_sha256": arrays["payload_sha256"],
            "audit_file_sha256": {
                "../audits/" + p.name: file_sha256(p)
                for p in (receipt_path, arrays_path, feedback_path)
            },
            "case_file_sha256": source_files,
        },
    )


def summarize_case(case, worker):
    summary, feedback, requests, sources = load_case(case, worker)
    bindings = {
        row["request_fingerprint"]: row for row in feedback["provider_bindings"]
    }
    arms, calls, provider_records = {}, [], []
    for mode in MODES:
        arm = summary["arms"][mode]
        attempts = []
        for attempt in arm["attempts"][1:]:
            attempt_id = attempt["attempt_id"]
            records = [
                json.loads(line)
                for line in (case / f"{attempt_id}_provider.jsonl")
                .read_text()
                .splitlines()
            ]
            require(
                len(records) == len(attempt["decisions"]),
                "Every decision slot must have a ledger row",
            )
            provider_records.extend(records)
            accepted, rejected = 0, 0
            for decision, record in zip(attempt["decisions"], records, strict=True):
                fingerprint = record["request_fingerprint"]
                event = requests[fingerprint]
                request = event["request"]
                require(request["attempt_id"] == attempt_id, "Per-arm attempt identity")
                require(record["requested_model"] == MODEL, "Requested model")
                require(record["cache"] == {"no-cache": True}, "Cache")
                require(
                    record["sampling_settings"]
                    == {"reasoning_effort": "medium", "max_completion_tokens": 8192},
                    "Sampling settings",
                )
                require(
                    record["accepted"] == decision["accepted"], "Decision acceptance"
                )
                binding = bindings[fingerprint]
                require(
                    binding["accepted"] == record["accepted"],
                    "Audited provider acceptance",
                )
                content = (
                    record.get("response", {})
                    .get("choices", [{}])[0]
                    .get("message", {})
                    .get("content")
                )
                proposal = None
                if content is not None:
                    require(record["response"]["model"] == MODEL, "Actual model")
                    try:
                        candidate = json.loads(content)
                        proposal = candidate if isinstance(candidate, dict) else None
                    except json.JSONDecodeError:
                        pass
                if decision["accepted"]:
                    require(
                        parse_proposal(content, request) == decision["proposal"],
                        "Exact accepted proposal",
                    )
                    require(
                        binding["response_text_sha256"]
                        == hashlib.sha256(content.encode()).hexdigest(),
                        "Accepted response text hash",
                    )
                    accepted += 1
                else:
                    rejected += 1
                current = request["observations"][-1]
                require(
                    current["step"] == request["observation_step"],
                    "Current snapshot freshness",
                )
                previous = request["previous_attempt"]
                selected = (
                    None
                    if proposal is None
                    else {
                        name: proposal.get(name)
                        for name in (
                            "source_a_id",
                            "source_b_id",
                            "alpha",
                            "observed_phase",
                            "rationale",
                            "vision",
                        )
                    }
                )
                calls.append(
                    {
                        "mode": mode,
                        "attempt_id": attempt_id,
                        "full_rollout_revision": attempt["iteration"] - 1,
                        "decision_index": request["decision_index"],
                        "step": request["observation_step"],
                        "request_fingerprint": fingerprint,
                        "request_event_sequence": event["sequence"],
                        "physical_provider_call": record["provider_call"],
                        "accepted": decision["accepted"],
                        "decision_error": decision["error"],
                        "http_status": record.get("http_status"),
                        "response_text_sha256": None
                        if content is None
                        else hashlib.sha256(content.encode()).hexdigest(),
                        "proposal_fields": selected,
                        "proposal_fields_are_applied": decision["accepted"],
                        "active_text_before_call": request["active_interpolation"],
                        "current_snapshot_steps": [
                            row["step"] for row in request["observations"]
                        ],
                        "completed_rollout_feedback": request[
                            "completed_rollout_feedback"
                        ],
                        "previous_attempt_feedback": None
                        if previous is None
                        else previous["feedback"],
                        "previous_attempt_snapshot_steps": []
                        if previous is None
                        else [row["step"] for row in previous["snapshots"]],
                        "previous_attempt_decision_count": 0
                        if previous is None
                        else len(previous["decisions"]),
                        "same_attempt_history_count": len(
                            request["previous_decisions"]
                        ),
                        "token_usage": binding["token_usage"],
                    }
                )
            attempts.append(
                {
                    "attempt_id": attempt_id,
                    "full_rollout_revision": attempt["iteration"] - 1,
                    "recorded_success": attempt["success"],
                    "actions": attempt["actions_executed"],
                    "accepted_decisions": accepted,
                    "rejected_decisions": rejected,
                    "accepted_decisions_executed": attempt[
                        "accepted_decisions_executed"
                    ],
                    "actions_with_nonzero_text": attempt["actions_with_nonzero_text"],
                    "actions_with_changed_vision": attempt[
                        "actions_with_changed_vision"
                    ],
                    "actions_with_held_text_after_failed_call": attempt[
                        "actions_with_held_text_after_failed_call"
                    ],
                    "native_condition_fallback_actions": attempt[
                        "native_condition_fallback_actions"
                    ],
                }
            )
        arms[mode] = {"summary": arm["summary"], "attempts": attempts}
    costs = summarize_calls(provider_records)
    require(
        costs == feedback["provider"], "Full physical call and token reconciliation"
    )
    require(
        len(calls) == feedback["counts"]["client_attempts"],
        "All client slots represented",
    )
    outcomes = {
        name: {
            "success": row["summary"]["success"],
            "attempt_successes": [a["success"] for a in row["attempts"]],
        }
        for name, row in summary["arms"].items()
    }
    return {
        "status": "verified",
        "episode_id": summary["episode_id"],
        "target_task": next(iter(requests.values()))["request"]["target_task"],
        "protocol_sha256": summary["protocol_sha256"],
        "source_catalog": summary["source_catalog"],
        "oracle_mapping_supplied_to_astra": False,
        "baseline": {
            "recorded_success": False,
            "actions": summary["baseline"]["actions_executed"],
        },
        "recorded_outcomes": outcomes,
        "astra_arms": arms,
        "provider": costs,
        "rejected_decision_reasons": dict(
            Counter(row["decision_error"] for row in calls if not row["accepted"])
        ),
        "calls": calls,
        "sources": sources,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wine-case", type=Path, required=True)
    parser.add_argument("--milk-case", type=Path, required=True)
    args = parser.parse_args()
    cases = {
        "wine": summarize_case(args.wine_case, 0),
        "milk": summarize_case(args.milk_case, 1),
    }
    require(
        cases["wine"]["episode_id"] == "libero_goal_ood:seed19:task6:state0",
        "Wine identity",
    )
    require(
        cases["milk"]["episode_id"] == "libero_spatial_ood:seed19:task2:state0",
        "Milk identity",
    )
    result = {
        "schema_version": "phase-interpolation-development-behavior-1",
        "status": "verified",
        "scope": "two development cases, recovery-v2 only",
        "cases": cases,
        "source_file_sha256": {
            str(p.relative_to(PROJECT)): file_sha256(p)
            for p in (
                Path(__file__).resolve(),
                PROJECT / "interpolation_agent.py",
                PROJECT / "intervention_agent.py",
                PROJECT / "astra_client.py",
                PROJECT / "interpolation_conditioning.py",
                PROJECT / "records.py",
            )
        },
        "limitations": [
            "observed_phase and rationale are explicit public response fields, not hidden reasoning or independently measured progress.",
            "Behavior is in-context adaptation; model weights are frozen and there is no demonstration of persistent learning.",
            "Rejected proposal fields describe an intention and were not applied. The previous valid text decision remained active; fresh vision expired.",
            "Numerical and feedback audits bind recorded execution; task outcomes were not independently rerun.",
            "Development examples cannot establish generalization or isolate the causal contribution of language versus vision.",
            "Interrupted original-workflow overhead and the separate transport smoke are excluded; their costs are retained in separate receipts.",
        ],
    }
    target = HERE / "behavior.json"
    target.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "behavior_sha256": file_sha256(target),
                "provider_calls": {
                    name: row["provider"]["provider_calls"]
                    for name, row in cases.items()
                },
            }
        )
    )


if __name__ == "__main__":
    main()
