"""Reconcile sealed development ledgers and publish the selected raw-frame review.

CPU only. Requires retained development artifacts and the pinned LIBERO-OOD
checkout; does not make provider, policy, simulator, or network calls.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from astra_reversal.frs_agent import summarize_calls
from astra_reversal.records import digest

REPORT_SHA = "7b69978c4cc5653fec9fe780a731e23f88777236562a44df2c38d01ede7c0be6"
OOD_REVISION = "587a6cbf64f16c7b87fa5805dc0ed934192239a4"
TASKS = ((0, 6), (1, 2), (2, 8))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def independent_tokens(rows):
    """Sum original HTTP usage, separately from the client's normalized ledger."""
    total, missing = Counter(), Counter()
    for row in rows:
        usage = (row.get("response") or {}).get("usage") or {}
        fields = {
            "input_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get(
                "reasoning_tokens"
            ),
        }
        for key, value in fields.items():
            if value is None:
                missing[key] += 1
            else:
                assert type(value) is int and value >= 0
                total[key] += value
    return {
        key: {"sum": total[key], "missing_calls": missing[key]}
        for key in ("input_tokens", "output_tokens", "total_tokens", "reasoning_tokens")
    }


def choices(rows):
    result = defaultdict(Counter)
    for row in rows:
        role = row["role"]
        if not row["accepted"]:
            result[role]["rejected"] += 1
            continue
        response = json.loads(row["response"]["choices"][0]["message"]["content"])
        if role == "action_edit":
            value = response["mode"]
        elif role == "paper_direction":
            value = "native_defer" if response["fine"] else "direction"
        elif role == "judge":
            value = response["verdict"]
        else:
            value = "rules_returned"
        result[role][value] += 1
    return {key: dict(value) for key, value in result.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--ood-source", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    base, output = args.run_root, args.output
    output.mkdir(parents=True, exist_ok=True)
    report_path = base / "compiled/report.json"
    assert sha(report_path) == REPORT_SHA
    report = read(report_path)
    assert report["status"] == "complete" and report["phase"] == "development"
    assert report["producer_evidence"]["missing_task_keys"] == []
    providers_all, evaluations, bindings = [], [], []
    physical = Counter()
    physical_runs = {}
    empty_rule_defer_count = 0
    judges = Counter()
    unchanged_actor_generations = 0
    fallback_contexts = []
    for worker, task in TASKS:
        label = f"worker_{worker}/task_{task}"
        audit_dir = base / "audits" / label
        task_dir = base / f"extracted/worker_{worker}/results/task_{task}"
        seal_path = base / "task_archive_inputs" / label / "completion_receipt.json"
        audit_path, receipt_path = audit_dir / "audit.json", audit_dir / "receipt.json"
        audit, receipt, seal = read(audit_path), read(receipt_path), read(seal_path)
        assert audit["status"] == receipt["status"] == "passed"
        assert all(audit["validations"].values())
        assert sha(audit_path) == receipt["audit_sha256"]
        assert sha(seal_path) == receipt["completion_receipt_sha256"]
        archive = base / receipt["archive"]["local_path"]
        assert sha(archive) == receipt["archive"]["sha256"]
        assert archive.stat().st_size == receipt["archive"]["bytes"]
        for name in ("summary.json", "events.jsonl", "provider.jsonl"):
            assert sha(task_dir / name) == seal["task_files_sha256"][name]
            assert sha(task_dir / name) == audit["input_file_sha256"][name]
        summary = read(task_dir / "summary.json")
        assert summary["status"] == "complete" and summary["development"]
        events, providers = (
            lines(task_dir / "events.jsonl"),
            lines(task_dir / "provider.jsonl"),
        )
        assert [row["sequence"] for row in events] == list(range(len(events)))
        assert [row["request_index"] for row in providers] == list(
            range(1, len(providers) + 1)
        )
        decisions = {
            row["request_index"]: row
            for row in events
            if row["kind"] == "astra_decision"
        }
        assert len(decisions) == len(providers)
        for row in providers:
            decision = decisions[row["request_index"]]
            assert decision["request_fingerprint"] == row["request_fingerprint"]
            assert (decision["response"] is not None) == row["accepted"]
            if row["accepted"]:
                parsed = json.loads(row["response"]["choices"][0]["message"]["content"])
                parsed["request_fingerprint"] = row["request_fingerprint"]
                assert parsed == decision["response"]
        usage = summarize_calls(providers)
        assert (
            usage
            == summary["provider_usage"]
            == audit["provider_usage"]
            == receipt["provider_usage"]
        )
        for key, value in independent_tokens(providers).items():
            assert value["sum"] == usage["tokens"][key]["sum"]
            assert value["missing_calls"] == usage["tokens"][key]["missing_calls"]
        prefix_dir = base / f"visual-review/development_prefix_usage/worker_{worker}"
        assert all(
            sha(prefix_dir / name) == seal["task_files_sha256"][name]
            for name in ("events.jsonl", "provider.jsonl")
        )
        starts = {
            row["attempt_id"]: row for row in events if row["kind"] == "rollout_start"
        }
        ends = {
            row["result"]["attempt_id"]: row["result"]
            for row in events
            if row["kind"] == "rollout_end"
        }
        assert starts.keys() == ends.keys()
        for name, result in ends.items():
            physical_runs[(result["episode_id"], name)] = result
        for arm in summary["adaptation"].values():
            assert len(arm["rounds"]) == 3 and arm["final_rules"] == []
            for round_record in arm["rounds"]:
                assert round_record["promoted"] is False
                assert ends[round_record["candidate_attempt_id"]]["success"] is False
                judges[round_record["judge"]["verdict"]] += 1
        for event in events:
            if event["kind"] != "generation":
                continue
            start = starts[event["attempt_id"]]
            online = event["method"].startswith("critique_frs_") or event["method"] in (
                "astra_direction_direct",
                "astra_frs",
            )
            if online and event["proposal"] is None:
                result = ends[event["attempt_id"]]
                assert event["generation_kind"] == "native_defer"
                fallback_contexts.append(
                    {
                        "episode_id": result["episode_id"],
                        "attempt_id": result["attempt_id"],
                        "step": event["step"],
                        "executed_actions": min(
                            10, result["actions_executed"] - event["step"]
                        ),
                        "recorded_run_success": result["success"],
                    }
                )
            if event["method"] == "critique_frs_no_learning" and start["evaluation"]:
                assert start["rules"] == []
                assert event["proposal"]["mode"] == "defer"
                assert event["generation_kind"] == "native_defer"
                assert event["reference"] is None and event["reversed_noise"] is None
                empty_rule_defer_count += 1
            if event["method"] == "learned_noise":
                assert (
                    event["actor_receipt"]["source"] == "exact_untrained_base_fallback"
                )
                assert event["actor_receipt"]["trained_rounds"] == 0
                unchanged_actor_generations += 1
        assert summary["physical_cost"]["accepted_policy_updates"] == 0
        assert summary["physical_cost"]["optimizer_steps"] == 0
        assert summary["physical_cost"]["auxiliary_inferences"] == 0
        for key in (
            "rollouts",
            "actions",
            "velocity_evaluations",
            "accepted_policy_updates",
            "optimizer_steps",
            "auxiliary_inferences",
        ):
            physical[key] += summary["physical_cost"][key]
        for row in summary["evaluation"]:
            assert row["episode_id"].endswith(":state1")
            assert row["success"] == ends[row["attempt_id"]]["success"]
            evaluations.append(
                {
                    key: row[key]
                    for key in ("episode_id", "method", "round_index", "success")
                }
            )
        bindings.append(
            {
                "artifact_id": label,
                "suite": summary["suite"],
                "task_id": task,
                "archive_sha256": receipt["archive"]["sha256"],
                "audit_receipt_sha256": sha(receipt_path),
                "full_audit_sha256": sha(audit_path),
                "completion_seal_sha256": sha(seal_path),
                "source_sha256": {
                    name: sha(task_dir / name)
                    for name in ("summary.json", "events.jsonl", "provider.jsonl")
                },
                "preserved_live_prefix_is_complete": True,
                "provider_event_joins": len(providers),
                "all_prior_full_audit_validations_passed": True,
            }
        )
        providers_all.extend(providers)
    for cohort in report["cohorts"]:
        for round_record in cohort["rounds"]:
            for row in round_record["episodes"]:
                result = physical_runs[(row["episode_id"], row["attempt_id"])]
                assert row["success"] == row["recorded_success"] == result["success"]
                assert row["actions"] == result["actions_executed"]
                assert row["audit"]["status"] == "passed"
    tokens = independent_tokens(providers_all)
    expected = report["producer_evidence"]["physical_provider_usage"]
    assert tokens == expected["tokens"]
    assert len(providers_all) == expected["calls"] == 775
    assert (
        sum(row["accepted"] for row in providers_all)
        == expected["accepted_calls"]
        == 771
    )
    assert expected["failed_calls"] == 4 and expected["preflight_failures"] == 0
    assert tokens["total_tokens"] == {"sum": 3229704, "missing_calls": 3}
    assert judges == {"same": 18} and empty_rule_defer_count == 90
    outcomes = []
    for method, round_index in sorted(
        {(row["method"], row["round_index"]) for row in evaluations}
    ):
        subset = [
            row
            for row in evaluations
            if (row["method"], row["round_index"]) == (method, round_index)
        ]
        assert len(subset) == 3 and len({row["episode_id"] for row in subset}) == 3
        wins = [row["episode_id"] for row in subset if row["success"]]
        assert len(wins) == (
            2 if method in ("astra_direction_direct", "astra_frs") else 0
        )
        outcomes.append(
            {
                "method": method,
                "checkpoint_round": round_index,
                "successes": len(wins),
                "episodes": len(subset),
                "successful_episode_ids": wins,
            }
        )
    per_role = {
        role: summarize_calls([row for row in providers_all if row["role"] == role])
        for role in sorted({row["role"] for row in providers_all})
    }
    final = {
        "schema_version": "frs-development-independent-reconciliation-1.0",
        "status": "passed",
        "coverage": "Three completed development tasks; adaptation reset0 and held-out reset1 only. This is not the 20-task evaluation.",
        "compiled_report_sha256": REPORT_SHA,
        "source_bindings": bindings,
        "provider_usage": summarize_calls(providers_all),
        "independent_http_usage_arithmetic": tokens,
        "per_role_costs": per_role,
        "provider_choices": choices(providers_all),
        "rejected_call_usage": summarize_calls(
            [row for row in providers_all if not row["accepted"]]
        ),
        "rejected_call_native_fallback": {
            "generations": len(fallback_contexts),
            "executed_actions": sum(
                row["executed_actions"] for row in fallback_contexts
            ),
            "contexts": fallback_contexts,
        },
        "physical_totals": dict(physical),
        "evaluation_outcomes": outcomes,
        "judgments": dict(judges),
        "promotions": 0,
        "heldout_no_learning_empty_rule_deferrals": empty_rule_defer_count,
        "learned_noise_untrained_fallback_generations": unchanged_actor_generations,
        "interpretation": "All18 judges returned same, so no rules or noise-policy weights were promoted. The held-out no-learning arm received empty rules and deferred on all90 calls; no held-out action editing occurred. The auxiliary actor remained untrained and used its exact native-noise fallback.",
        "checks": {
            "all_seals_and_audit_hashes_match": True,
            "all_archives_rehashed": True,
            "all775_provider_decisions_join": True,
            "all_report_outcome_rows_match_events": True,
            "failed_call_costs_retained": True,
            "live_prefixes_match_completed_files": True,
        },
        "limitations": [
            "This reconciliation rechecks ledgers, seals, outcomes and costs and binds the prior full array audits; it does not repeat every array replay or hidden model/physics computation.",
            "Three missing usage reports make3,229,704 a lower bound on tokens. No dollar price is assumed. Reasoning tokens are included in output tokens.",
            "Previous interrupted workflows and the transport smoke remain separate costs.",
            "The three development resets cannot establish generalization or superiority of FRS over direct steering.",
        ],
        "builder_sha256": sha(Path(__file__)),
    }
    write(output / "independent_reconciliation.json", final)
    # Bind the selected guide/reference check and raw figures to the sealed archive.
    selected = base / "visual-review/worker_0_direction"
    sample = read(selected / "selected_direction_verification.json")
    assert (
        sample["status"] == "passed"
        and sample["archive_sha256"] == bindings[0]["archive_sha256"]
    )
    source = base / "visual-review/verify_direction_sample.py"
    assert sha(source) == sample["verification_helper_sha256"]
    for filename in (
        "selected_direction_verification.json",
        "direction_final_raw_frames.png",
        "direction_guide.png",
    ):
        if filename.endswith(".png"):
            assert sha(selected / filename) == sample["figure_sha256"][filename]
        shutil.copyfile(selected / filename, output / filename)
    (output / "source").mkdir(exist_ok=True)
    shutil.copyfile(source, output / "source/verify_direction_sample.py")
    # Read exact source blobs by the pinned revision, independent of worktree changes.
    source_paths = (
        "third_party/modified_libero/libero/libero/bddl_files/libero_goal_ood/put_the_wine_bottle_in_the_bowl.bddl",
        "third_party/modified_libero/libero/libero/envs/predicates/base_predicates.py",
        "third_party/modified_libero/libero/libero/envs/object_states/base_object_states.py",
    )
    predicate_hashes = {}
    for name in source_paths:
        content = subprocess.check_output(
            ["git", "-C", str(args.ood_source), "show", f"{OOD_REVISION}:{name}"]
        )
        predicate_hashes[name] = hashlib.sha256(content).hexdigest()
        if name.endswith(".bddl"):
            entry = lines(base / "extracted/worker_0/results/task_6/events.jsonl")[0][
                "entries"
            ][0]
            assert predicate_hashes[name] == entry["bddl_sha256"]
            assert "(And (On wine_bottle_1 akita_black_bowl_1))" in content.decode()
    goal_events = lines(base / "extracted/worker_0/results/task_6/events.jsonl")
    first_review = base / "visual-review/worker_0"
    first_generation = next(
        row
        for row in goal_events
        if row["kind"] == "generation"
        and (row.get("proposal") or {}).get("request_index") == 11
    )
    first_arrays = {}
    for name, descriptor in (
        ("native_actions", first_generation["native_actions"]),
        ("reference_actions", first_generation["reference"]["target_actions"]),
        ("executed_actions", first_generation["actions"]),
    ):
        value = np.load(first_review / descriptor["array"], allow_pickle=False)
        assert digest(value) == descriptor["sha256"]
        first_arrays[name] = value
    assert np.all(first_arrays["native_actions"][:, 6] == -1)
    assert np.all(first_arrays["reference_actions"][:, 6] == 1)
    assert np.all(first_arrays["executed_actions"][:6, 6] > 0)
    assert np.all(first_arrays["executed_actions"][6:, 6] < 0)
    for name in ("events", "provider"):
        complete = base / f"extracted/worker_0/results/task_6/{name}.jsonl"
        assert complete.read_bytes().startswith(
            (first_review / f"{name}_prefix.jsonl").read_bytes()
        )
    write(
        output / "first_action_edit_review.json",
        {
            "schema_version": "frs-selected-edit-fidelity-1.0",
            "status": "passed_selected_reconciliation",
            "episode_id": "libero_goal_ood:seed19:task6:state0",
            "attempt_id": first_generation["attempt_id"],
            "step": first_generation["step"],
            "request_index": 11,
            "request_fingerprint": first_generation["proposal"]["request_fingerprint"],
            "archive_sha256": bindings[0]["archive_sha256"],
            "full_task_audit_sha256": bindings[0]["full_audit_sha256"],
            "original_selected_review_sha256": sha(
                first_review / "first_record_review.json"
            ),
            "preserved_prefixes_match_complete_archive": True,
            "gripper_commands": {
                name: value[:, 6].tolist() for name, value in first_arrays.items()
            },
            "action_array_sha256": {
                name: digest(value) for name, value in first_arrays.items()
            },
            "interpretation": "The reference closes for all10 rows, while the post-FRS executed chunk closes for6 rows and reopens for4. The reference edit is applied correctly; finite flow reversal with the declared noise transform does not enforce a hard execution constraint. This is not an identity-roundtrip test.",
        },
    )
    control_mix = []
    for method in ("astra_direction_direct", "astra_frs"):
        result = next(
            event["result"]
            for event in goal_events
            if event["kind"] == "rollout_end" and event["result"]["method"] == method
        )
        counts = Counter()
        for event in goal_events:
            if event["kind"] != "generation" or event["method"] != method:
                continue
            count = min(10, result["actions_executed"] - event["step"])
            proposal = event["proposal"]
            category = (
                "rejected_native_fallback"
                if proposal is None
                else "accepted_native_defer"
                if proposal["fine"]
                else "accepted_direction"
            )
            counts[category + "_chunks"] += 1
            counts[category + "_actions"] += count
        control_mix.append({"method": method, **counts})
    visual = {
        "schema_version": "frs-development-direction-visual-review-1.0",
        "status": "completed_selected_review",
        "episode_id": "libero_goal_ood:seed19:task6:state1",
        "selected_request_index": 204,
        "selected_verification_sha256": sha(
            output / "selected_direction_verification.json"
        ),
        "figure_sha256": sample["figure_sha256"],
        "recorded_control_mix": control_mix,
        "predicate_source_revision": OOD_REVISION,
        "predicate_source_sha256": predicate_hashes,
        "recorded_instruction": "put the wine bottle in the bowl",
        "exact_bddl_language": "Put the wine bottle on the bowl",
        "exact_goal": "(On wine_bottle_1 akita_black_bowl_1)",
        "predicate_meaning": "On checks bottle center above bowl center, object contact and XY center distance below0.1. It has no release, unsupported stability or dwell condition.",
        "visual_finding": "Both final paired raw views show the wine bottle positioned in/over the bowl with the gripper still very close around the bottle. This supports the recorded bowl-contact relation but does not establish release or stable unsupported placement.",
        "guide_finding": "The exact projected robot end-effector marker and plumb line replay correctly with camera signs[1,-1,1]. The marker denotes the recorded end-effector/palm reference, not a visually estimated bottle contact point. The guide is a separate reasoner image; raw policy pixels are unchanged.",
        "interpretation": "Direct steering and FRS both satisfy the benchmark predicate on this development reset. This selected example does not isolate an FRS benefit or show a learned-policy improvement.",
        "source_scope": "Raw final images and selected generation verified against sealed worker0 archive; predicate source inspected at the pinned commit, not executed again.",
    }
    write(output / "visual_review.json", visual)
    print(
        json.dumps(
            {
                "status": "passed",
                "provider_calls": 775,
                "known_tokens": 3229704,
                "reconciliation_sha256": sha(
                    output / "independent_reconciliation.json"
                ),
                "visual_review_sha256": sha(output / "visual_review.json"),
            }
        )
    )


if __name__ == "__main__":
    main()
