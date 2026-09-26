"""Synthetic recording/receipt fixtures; no model, simulator or provider calls."""

import copy
import hashlib
import json
import shutil

import pytest

from astra_reversal.frs_agent import SCHEMA_VERSION as CLIENT_VERSION
from astra_reversal.frs_agent import prompt_manifest, summarize_calls
from astra_reversal.frs_experiment import load_protocol
from astra_reversal.frs_html_report import TASKS, derived_results, validate_report
from astra_reversal.frs_report import (
    AUDIT_VERSION,
    REQUIRED_VALIDATIONS,
    _assignment,
    _compact_usage,
    build_report,
    write_report,
)
from astra_reversal.intervention_agent import normalize_usage
from astra_reversal.records import digest, file_sha256


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def lines(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def checkpoint(task_key, trained=0):
    policy = {
        "task_id": task_key,
        "parameter_sha256": digest(["synthetic parameters", task_key, trained]),
        "optimizer_sha256": digest(["synthetic optimizer", trained]),
        "replay_sha256": digest(["synthetic replay", trained]),
        "history_sha256": digest(["synthetic history", trained]),
        "trained": bool(trained),
        "rounds": trained,
        "replay_samples": trained,
        "synthetic_cpu_test_only": False,
        "test_only_cpu_steps": None,
    }
    policy["state_id"] = digest(policy)
    manifest = {
        "schema_version": "frs-noise-checkpoint-1.0",
        "policy": policy,
        "files": {
            "state.pt": {
                "sha256": digest(["synthetic serialized state", task_key, trained]),
                "bytes": 1,
            }
        },
    }
    raw = json.dumps(manifest, sort_keys=True, indent=2) + "\n"
    return {**manifest, "manifest_sha256": hashlib.sha256(raw.encode()).hexdigest()}


def _task(root, worker_values, task_id, *, initial_success=False, missing_usage=False):
    protocol = worker_values["protocol.json"]
    runtime = worker_values["runtime.json"]
    suite = runtime["assignment"]["suite"]
    task_key = f"{suite}:{task_id}"
    entries = [
        row
        for row in worker_values["reset_manifest.json"]["episodes"]
        if row["task_id"] == task_id
    ]
    directory = root / f"task_{task_id}"
    directory.mkdir()
    events, providers, physical, evaluation, adaptations = [], [], [], [], {}
    by_id = {}
    snapshots = []

    def event(kind, **data):
        value = {
            "sequence": len(events),
            "timestamp": float(len(events)),
            "kind": kind,
            **copy.deepcopy(data),
        }
        events.append(value)
        return value

    event(
        "task",
        entries=entries,
        protocol=protocol,
        checkpoint=worker_values["checkpoint.json"],
    )
    initial = checkpoint(task_key)
    event("noise_policy_initial", checkpoint=initial)
    write(
        directory / "noise_policy_initial/manifest.json",
        {k: v for k, v in initial.items() if k != "manifest_sha256"},
    )

    def call(role, attempt, entry, response, *, failed=False, unknown=False):
        index = len(providers)
        usage = (
            None
            if unknown
            else {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "completion_tokens_details": {"reasoning_tokens": 2},
            }
        )
        row = {
            "client_schema_version": CLIENT_VERSION,
            "prompt_template_version": worker_values["prompts.json"][
                "prompt_template_version"
            ],
            "requested_model": protocol["astra"]["model"],
            "system_prompt_sha256": worker_values["prompts.json"]["roles"][role][
                "sha256"
            ],
            "request_index": index + 1,
            "request_fingerprint": digest([task_key, index]),
            "episode_id": entry["episode_id"],
            "attempt_id": attempt,
            "role": role,
            "provider_call": True,
            "accepted": not failed,
            "token_usage": normalize_usage(usage),
            "response": {"model": protocol["astra"]["model"], "usage": usage},
            "latency_seconds": 1.0,
            "http_status": 200,
        }
        if failed:
            row["error_kind"] = "proposal_rejected"
            row["error"] = "Synthetic rejection"
        providers.append(row)
        event(
            "astra_decision",
            request_index=index + 1,
            role=role,
            request_fingerprint=row["request_fingerprint"],
            response=None if failed else response,
            error="Synthetic rejection" if failed else None,
            provider_record_index=index,
        )
        return None if failed else response

    def rollout(method, state, index):
        entry = entries[state]
        attempt = f"{method}_state{state}_round{index}"
        event(
            "rollout_start",
            attempt_id=attempt,
            method=method,
            episode_id=entry["episode_id"],
            round_index=index,
            reset_entry_sha256=digest(entry),
            evaluation=state != 0,
        )
        provider_begin = len(providers)
        initial = initial_success and state == 0
        if not initial and method.startswith(("astra_", "critique_")):
            role = "paper_direction" if method.startswith("astra_") else "action_edit"
            call(role, attempt, entry, {"mode": "synthetic"})
        reset = {
            key: entry[key]
            for key in (
                "episode_id",
                "seed",
                "reset_state_sha256",
                "reset_model_sha256",
                "bddl_sha256",
            )
        }
        reset["initial_success"] = initial
        reset["post_stabilization_observation_sha256"] = digest(
            [task_key, state, "pixels"]
        )
        reset["sha256"] = digest(reset)
        success = (
            initial
            or (
                state == 0
                and (
                    (method == "critique_frs_no_learning" and index == 2)
                    or (method == "critique_frs_learning" and index == 3)
                )
            )
            or (
                state > 0
                and (
                    method == "native_repeated_noise"
                    or (method == "learned_noise" and index == 2)
                )
            )
        )
        result = {
            "attempt_id": attempt,
            "method": method,
            "episode_id": entry["episode_id"],
            "round_index": index,
            "evaluation": state != 0,
            "success": success,
            "actions_executed": 0 if initial else 10,
            "initial_success": initial,
            "zero_action_success": initial,
            "reset_audit": reset,
            "action_budget": 300,
            "execute_steps": 10,
            "counters": {
                "velocity_evaluations": 0 if initial else 10,
                "auxiliary_inferences": int(
                    method == "learned_noise" and index > 1 and not initial
                ),
                "auxiliary_request_seconds": 0.0,
            },
            "wall_seconds": 10.0,
            "provider_record_indexes": list(range(provider_begin, len(providers))),
            "provider_usage": summarize_calls(providers[provider_begin:]),
            "snapshots": snapshots,
            "executed_actions": [],
            "video_path": "synthetic.mp4",
        }
        event("rollout_end", result=result)
        row = {
            k: v
            for k, v in result.items()
            if k not in ("snapshots", "executed_actions", "video_path")
        }
        row["physical_run_id"] = f"{entry['episode_id']}:{attempt}"
        physical.append(row)
        by_id[attempt] = row
        return row

    baseline = rollout("native_repeated_noise", 0, 0)
    updates = []
    states = [1] if runtime["phase"] == "development" else list(range(1, 11))
    for method in protocol["adaptation_methods"]:
        rounds, attempts = [], [baseline]
        latest = incumbent = baseline
        for index in range(1, 4):
            failed = method == "critique_frs_no_learning" and index == 1
            critique = call(
                "critique",
                latest["attempt_id"],
                entries[0],
                {"rules": []},
                failed=failed,
                unknown=missing_usage and failed,
            )
            candidate = rollout(method, 0, index)
            judge = call(
                "judge",
                candidate["attempt_id"],
                entries[0],
                {"verdict": "better" if index == 1 else "same"},
            )
            promoted = critique is not None and judge["verdict"] == "better"
            update, saved = None, None
            if method == "critique_frs_learning":
                if promoted:
                    update = {
                        "status": "updated",
                        "optimizer_updates": 1000,
                        "judge_sha256": digest(judge),
                        "rollout_id": candidate["attempt_id"],
                        "wall_seconds": 5.0,
                    }
                    updates.append(update)
                saved = checkpoint(task_key, 1)
                write(
                    directory / f"noise_policy_round{index}/manifest.json",
                    {k: v for k, v in saved.items() if k != "manifest_sha256"},
                )
            record = {
                "round_index": index,
                "critique": critique,
                "rules": [],
                "candidate_attempt_id": candidate["attempt_id"],
                "incumbent_attempt_id": incumbent["attempt_id"],
                "judge": judge,
                "promoted": promoted,
                "update": update,
                "checkpoint": saved,
                "incumbent_success_evaluation_only": incumbent["success"],
                "candidate_success_evaluation_only": candidate["success"],
            }
            event("adaptation_round", method=method, **record)
            rounds.append(record)
            if promoted:
                incumbent = candidate
            latest = candidate
            attempts.append(candidate)
            if method == "critique_frs_learning":
                for state in states:
                    row = rollout("learned_noise", state, index)
                    evaluation.append(
                        {
                            key: row[key]
                            for key in (
                                "method",
                                "round_index",
                                "physical_run_id",
                                "attempt_id",
                                "episode_id",
                                "success",
                            )
                        }
                    )
        first = next((i for i, row in enumerate(attempts) if row["success"]), None)
        adaptations[method] = {
            "rounds": rounds,
            "baseline_attempt_id": baseline["attempt_id"],
            "final_rules": [],
            "first_success": {
                "first_success_round": first,
                "censored": first is None,
                "observed_rounds": 3,
                "success_by_round": [
                    any(row["success"] for row in attempts[: i + 1]) for i in range(4)
                ],
            },
        }
    for method in protocol["evaluation_methods"]:
        if method != "learned_noise":
            for state in states:
                row = rollout(method, state, 3)
                evaluation.append(
                    {
                        key: row[key]
                        for key in (
                            "method",
                            "round_index",
                            "physical_run_id",
                            "attempt_id",
                            "episode_id",
                            "success",
                        )
                    }
                )
    provider_usage = summarize_calls(providers)
    cost = {
        "rollouts": len(physical),
        "actions": sum(row["actions_executed"] for row in physical),
        "velocity_evaluations": sum(
            row["counters"]["velocity_evaluations"] for row in physical
        ),
        "rollout_wall_seconds": sum(row["wall_seconds"] for row in physical),
        "provider_calls": provider_usage["provider_calls"],
        "client_attempts": len(providers),
        "accepted_policy_updates": len(updates),
        "optimizer_steps": 1000 * len(updates),
        "training_seconds": 5.0 * len(updates),
        "auxiliary_inferences": sum(
            row["counters"]["auxiliary_inferences"] for row in physical
        ),
        "auxiliary_request_seconds": 0.0,
    }
    summary = {
        "schema_version": "frs-task-1.0",
        "status": "complete",
        "development": runtime["phase"] == "development",
        "suite": suite,
        "task_id": task_id,
        "instruction": TASKS[task_key],
        "protocol_sha256": digest(protocol),
        "entries_sha256": digest(entries),
        "checkpoint": worker_values["checkpoint.json"],
        "initial_noise_policy_checkpoint": initial,
        "adaptation": adaptations,
        "evaluation": evaluation,
        "physical_rollouts": physical,
        "provider_usage": provider_usage,
        "physical_cost": cost,
    }
    write(directory / "summary.json", summary)
    lines(directory / "events.jsonl", events)
    lines(directory / "provider.jsonl", providers)
    hashes = {
        str(path.relative_to(directory)): file_sha256(path)
        for path in directory.rglob("*")
        if path.is_file()
    }
    worker_hashes = {
        name: file_sha256(root / name)
        for name in worker_values
        if name != "prompts.json"
    }
    initials = {
        "manifest_sha256": initial["manifest_sha256"],
        "state_file_sha256": initial["files"]["state.pt"]["sha256"],
        "state_id": initial["policy"]["state_id"],
        "parameter_sha256": initial["policy"]["parameter_sha256"],
    }
    receipt = {
        "schema_version": AUDIT_VERSION,
        "status": "passed",
        "suite": suite,
        "task_id": task_id,
        "development": summary["development"],
        "summary_sha256": hashes["summary.json"],
        "events_sha256": hashes["events.jsonl"],
        "provider_sha256": hashes["provider.jsonl"],
        "input_file_sha256": hashes,
        "worker": {
            "runtime": runtime,
            "input_file_sha256": worker_hashes,
            "native_tensor_sha256": worker_values["frozen_weights_before.json"][
                "sha256"
            ],
            "reset_manifest_sha256": worker_values["reset_manifest.json"]["sha256"],
            "checkpoint_metadata_sha256": digest(worker_values["checkpoint.json"]),
        },
        "provider_usage": provider_usage,
        "physical_cost": cost,
        "physical_runs": [
            {
                "episode_id": row["episode_id"],
                "method": row["method"],
                "round_index": row["round_index"],
                "physical_run_id": row["physical_run_id"],
                "attempt_id": row["attempt_id"],
                "evaluation": row["evaluation"],
                "success": row["success"],
                "actions": row["actions_executed"],
                "velocity_evaluations": row["counters"]["velocity_evaluations"],
                "reset_sha256": row["reset_audit"]["sha256"],
            }
            for row in physical
        ],
        "counts": {
            "physical_rollouts": cost["rollouts"],
            "actions": cost["actions"],
            "velocity_evaluations": cost["velocity_evaluations"],
            "provider_bindings": len(providers),
            "accepted_policy_updates": len(updates),
            "optimizer_steps": cost["optimizer_steps"],
        },
        "initial_checkpoint": initials,
        "checkpoints": [
            {
                "round_index": i,
                "manifest_sha256": checkpoint(task_key, 1)["manifest_sha256"],
                "state_file_sha256": checkpoint(task_key, 1)["files"]["state.pt"][
                    "sha256"
                ],
                "state_id": checkpoint(task_key, 1)["policy"]["state_id"],
                "parameter_sha256": checkpoint(task_key, 1)["policy"][
                    "parameter_sha256"
                ],
            }
            for i in range(1, 4)
        ],
        "validations": dict.fromkeys(REQUIRED_VALIDATIONS, True),
        "postprocessor_sources": {"frs_audit.py": "a" * 64},
        "limitations": ["Synthetic aggregate-test receipt; no empirical evidence."],
    }
    return receipt


def fixture_tree(
    root, phase="development", *, initial_success=False, missing_usage=False
):
    protocol = load_protocol()
    if phase == "development":
        protocol["seed"] = protocol["development_seed"]
    workers, receipts = [], []
    for worker in range(3 if phase == "development" else 8):
        directory = root / f"worker_{worker}"
        target = _assignment(phase, worker)
        runtime = {
            "workflow": "synthetic-no-compute",
            "worker": worker,
            "phase": phase,
            "assignment": target,
            "tf32": False,
            "gpu": "Synthetic L40S record fixture",
            "python": "test",
            "payload_sha256": "1" * 64,
            "source_revision": "2" * 40,
            "packages": {"numpy": "test"},
        }
        entries = [
            {
                "episode_id": f"{target['suite']}:seed{protocol['seed']}:task{task}:state{state}",
                "seed": protocol["seed"],
                "task_id": task,
                "initial_state_id": state,
                "instruction": TASKS[f"{target['suite']}:{task}"],
                "reset_state_sha256": digest([target["suite"], task, state, "state"]),
                "reset_model_sha256": digest([target["suite"], task, state, "model"]),
                "bddl_sha256": digest([target["suite"], task, "bddl"]),
            }
            for task in target["task_ids"]
            for state in range(11)
        ]
        manifest = {"episodes": entries}
        manifest["sha256"] = digest(manifest)
        tensors = {
            "synthetic-native-tensor": {
                "shape": [1],
                "dtype": "test",
                "sha256": "3" * 64,
            }
        }
        weights = {"tensors": tensors, "sha256": digest(tensors)}
        values = {
            "runtime.json": runtime,
            "protocol.json": protocol,
            "frozen_plan.json": {
                "runtime": runtime,
                "protocol_sha256": digest(protocol),
                "manifest_sha256": manifest["sha256"],
                "assigned_episodes": [row["episode_id"] for row in entries],
                "assigned_tasks": target["task_ids"],
            },
            "reset_manifest.json": manifest,
            "checkpoint.json": {
                "input_profile": "openpi_libero",
                "horizon": 10,
                "model_action_dim": 32,
                "fixture": "Synthetic aggregate records",
            },
            "prompts.json": prompt_manifest(),
            "frozen_weights_before.json": weights,
            "frozen_weights_after.json": weights,
        }
        for name, value in values.items():
            write(directory / name, value)
        workers.append(directory)
        for task in target["task_ids"]:
            receipts.append(
                _task(
                    directory,
                    values,
                    task,
                    initial_success=initial_success and worker == 0,
                    missing_usage=missing_usage and worker == 0,
                )
            )
    return workers, receipts


def test_complete_development_conserves_every_call_and_physical_run(tmp_path):
    workers, receipts = fixture_tree(tmp_path)
    report = build_report(workers, phase="development", audit_receipts=receipts)
    assert report["status"] == "complete"
    checked = validate_report(report)["physical_cost"]
    assert checked["unique_rollouts"] == 45
    assert checked["actions"] == 450
    assert checked["velocity_evaluations"] == 450
    assert checked["training_steps"] == 3000
    assert checked["provider_usage"]["calls"] == 63
    assert checked["provider_usage"]["failed_calls"] == 3
    assert checked["provider_usage"]["tokens"]["total_tokens"]["sum"] == 945
    # Online latency is within rollouts; only12 critique/judge calls and5s training
    # per task add wall time. No pooled elapsed-time claim is made.
    assert checked["summed_wall_seconds"] == 450 + 36 + 15
    results = derived_results(report)
    assert [
        p["successes"] for p in results["evaluation"]["learned_noise"]["points"]
    ] == [0, 3, 0]
    assert len(results["evaluation"]["native_euler10"]["points"]) == 1
    rescue = results["adaptation"]["critique_frs_no_learning"]["rescue"]
    assert rescue["median_revision_among_rescues"] == 2
    assert rescue["median_tokens_among_rescues_with_complete_usage"] == 90
    assert rescue["rescued"] == 3
    assert all(t["audited"] for t in report["producer_evidence"]["tasks"])
    supporting = report["supporting_results"]
    assert len(supporting["adaptation_attempts"]) == 6
    native_pair = next(
        row
        for row in supporting["paired_evaluation"]
        if row["method_id"] == "learned_noise"
        and row["checkpoint_round"] == 2
        and row["reference_method_id"] == "native_repeated_noise"
    )
    assert native_pair["counts"] == {
        "both_succeed": 3,
        "method_only": 0,
        "native_only": 0,
        "both_fail": 0,
    }
    assert native_pair["reference_is_reused_measurement"] is True
    learned_final = [
        row
        for row in supporting["per_suite"]
        if row["cohort_id"] == "evaluation"
        and row["method_id"] == "learned_noise"
        and row["round_index"] == 3
    ]
    assert [row["episodes"] for row in learned_final] == [1, 2]
    assert sum(row["successes"] for row in learned_final) == 0


def test_all20_tasks_and200_evaluation_resets_are_required(tmp_path):
    workers, receipts = fixture_tree(tmp_path, "evaluation")
    report = build_report(workers, phase="evaluation", audit_receipts=receipts)
    assert report["status"] == "complete"
    assert len(report["producer_evidence"]["expected_task_keys"]) == 20
    assert validate_report(report)["physical_cost"]["unique_rollouts"] == 1740
    assert len(report["cohorts"][1]["expected_episode_ids"]) == 200
    points = derived_results(report)["evaluation"]["learned_noise"]["points"]
    assert [(row["successes"], row["episodes"]) for row in points] == [
        (0, 200),
        (200, 200),
        (0, 200),
    ]
    partial = build_report(
        workers[:-1], phase="evaluation", audit_receipts=receipts[:-2]
    )
    assert partial["status"] == "partial"
    assert derived_results(partial) == {}


def test_missing_audit_withholds_efficacy_without_erasing_failed_cost(tmp_path):
    workers, receipts = fixture_tree(tmp_path)
    report = build_report(workers, phase="development", audit_receipts=receipts[:2])
    assert report["status"] == "partial"
    assert derived_results(report) == {}
    assert report["producer_evidence"]["physical_provider_usage"]["failed_calls"] == 3
    assert report["producer_evidence"]["unaudited_task_keys"] == [
        "libero_spatial_ood:8"
    ]


def test_initial_success_is_flagged_not_credited_and_denominator_is_preserved(tmp_path):
    workers, receipts = fixture_tree(tmp_path, initial_success=True)
    report = build_report(workers, phase="development", audit_receipts=receipts)
    assert report["status"] == "complete"
    assert report["producer_evidence"]["initial_success_physical_runs"] == 7
    results = derived_results(report)["adaptation"]["critique_frs_no_learning"]
    assert results["points"][-1]["episodes"] == 3
    assert results["points"][-1]["successes"] == 2
    assert results["rescue"]["censored"] == 1
    initial_rows = [
        row
        for rnd in report["cohorts"][0]["rounds"]
        for row in rnd["episodes"]
        if row["initial_success"]
    ]
    assert all(row["recorded_success"] and not row["success"] for row in initial_rows)


def test_missing_failed_call_usage_is_never_filled_with_zero(tmp_path):
    workers, receipts = fixture_tree(tmp_path, missing_usage=True)
    report = build_report(workers, phase="development", audit_receipts=receipts)
    total = report["producer_evidence"]["physical_provider_usage"]["tokens"][
        "total_tokens"
    ]
    assert total == {"sum": 930, "missing_calls": 1}
    rescue = derived_results(report)["adaptation"]["critique_frs_no_learning"]["rescue"]
    assert rescue["rescues_missing_usage"] == 1
    assert rescue["median_tokens_among_rescues_with_complete_usage"] == 90


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("summary_byte", "different source bytes"),
        ("event_outcome", "summary differs"),
        ("physical_duplicate", "repeats a physical"),
        ("provider_cost", "provider usage differs"),
        ("prompt", "prompt text/hash"),
        ("native_weight", "tensor bytes changed"),
        ("reset", "reset audit digest"),
        ("audit_cost", "Audit physical totals"),
        ("audit_initial", "initial auxiliary checkpoint"),
    ],
)
def test_changed_evidence_or_accounting_cannot_publish_as_complete(
    tmp_path, mutation, message
):
    workers, receipts = fixture_tree(tmp_path)
    directory = workers[0] / "task_6"
    if mutation == "summary_byte":
        (directory / "summary.json").write_text(
            (directory / "summary.json").read_text() + "\n"
        )
    elif mutation == "event_outcome":
        values = [
            json.loads(row)
            for row in (directory / "events.jsonl").read_text().splitlines()
        ]
        next(row for row in values if row["kind"] == "rollout_end")["result"][
            "success"
        ] = True
        lines(directory / "events.jsonl", values)
    elif mutation == "physical_duplicate":
        value = json.loads((directory / "summary.json").read_text())
        value["physical_rollouts"].append(copy.deepcopy(value["physical_rollouts"][0]))
        write(directory / "summary.json", value)
    elif mutation == "provider_cost":
        values = [
            json.loads(row)
            for row in (directory / "provider.jsonl").read_text().splitlines()
        ]
        values[1]["response"]["usage"]["prompt_tokens"] = 11
        values[1]["response"]["usage"]["total_tokens"] = 16
        values[1]["token_usage"] = normalize_usage(values[1]["response"]["usage"])
        lines(directory / "provider.jsonl", values)
    elif mutation == "prompt":
        value = json.loads((workers[0] / "prompts.json").read_text())
        value["roles"]["judge"]["system_prompt"] += " changed"
        write(workers[0] / "prompts.json", value)
    elif mutation == "native_weight":
        value = json.loads((workers[0] / "frozen_weights_after.json").read_text())
        value["sha256"] = "f" * 64
        write(workers[0] / "frozen_weights_after.json", value)
    elif mutation == "reset":
        values = [
            json.loads(row)
            for row in (directory / "events.jsonl").read_text().splitlines()
        ]
        end = next(row for row in values if row["kind"] == "rollout_end")
        end["result"]["reset_audit"]["sha256"] = "f" * 64
        summary = json.loads((directory / "summary.json").read_text())
        summary["physical_rollouts"][0]["reset_audit"]["sha256"] = "f" * 64
        lines(directory / "events.jsonl", values)
        write(directory / "summary.json", summary)
    elif mutation == "audit_cost":
        receipts[0]["physical_cost"]["actions"] += 1
    elif mutation == "audit_initial":
        receipts[0]["initial_checkpoint"]["state_id"] = "f" * 64
    with pytest.raises(ValueError, match=message):
        build_report(workers, phase="development", audit_receipts=receipts)


def test_duplicate_worker_and_cross_phase_records_are_rejected(tmp_path):
    workers, receipts = fixture_tree(tmp_path)
    with pytest.raises(ValueError, match="Duplicate logical worker"):
        build_report([workers[0], workers[0]], phase="development")
    with pytest.raises(ValueError, match="Worker phase"):
        build_report(workers, phase="evaluation", audit_receipts=receipts)


def test_stable_partial_tail_preserves_unassigned_provider_cost(tmp_path):
    workers, _ = fixture_tree(tmp_path)
    directory = workers[0] / "task_6"
    summary = json.loads((directory / "summary.json").read_text())
    events = [
        json.loads(row) for row in (directory / "events.jsonl").read_text().splitlines()
    ]
    providers = [
        json.loads(row)
        for row in (directory / "provider.jsonl").read_text().splitlines()
    ]
    # Stable prefix ends during the first candidate after its recorded online call.
    stop = next(
        i
        for i, row in enumerate(events)
        if row["kind"] == "rollout_end"
        and row["result"]["method"] == "critique_frs_no_learning"
    )
    events = events[:stop]
    summary["status"] = "running"
    summary["physical_rollouts"] = summary["physical_rollouts"][:1]
    summary["evaluation"] = []
    summary["adaptation"] = {
        "critique_frs_no_learning": {
            "rounds": [],
            "baseline_attempt_id": "native_repeated_noise_state0_round0",
        }
    }
    write(directory / "summary.json", summary)
    lines(directory / "events.jsonl", events)
    lines(directory / "provider.jsonl", providers[:2])
    report = build_report([workers[0]], phase="development")
    assert report["status"] == "partial"
    assert validate_report(report)["physical_cost"]["unique_rollouts"] == 1
    assert report["producer_evidence"]["physical_provider_usage"]["calls"] == 2
    assert report["producer_evidence"]["physical_provider_usage"]["failed_calls"] == 1
    assert report["producer_evidence"]["tasks"][0]["unfinished_rollout_ids"] == [
        "critique_frs_no_learning_state0_round1"
    ]
    assert len(report["overheads"]) == 1


def test_json_markdown_csv_and_html_bind_same_complete_data(tmp_path):
    workers, receipts = fixture_tree(tmp_path / "inputs")
    receipt_paths = []
    for i, receipt in enumerate(receipts):
        path = tmp_path / f"audit_{i}.json"
        write(path, receipt)
        receipt_paths.append(path)
    report = build_report(workers, phase="development", audit_receipts=receipt_paths)
    output = tmp_path / "report"
    manifest = write_report(report, output)
    assert manifest["report_sha256"] == file_sha256(output / "report.json")
    assert (output / "html/report.json").read_bytes() == (
        output / "report.json"
    ).read_bytes()
    assert "945" in (output / "report.md").read_text()
    assert "Checkpoint success is not accumulated" in (output / "report.md").read_text()
    assert "recorded_success" in (output / "episodes.csv").read_text()
    assert all(
        file_sha256(output / name) == row["sha256"]
        for name, row in manifest["files"].items()
    )
    assert str(tmp_path) not in (output / "report.json").read_text()
    with pytest.raises(ValueError, match="overwrite"):
        write_report(report, output)


def test_zero_call_preflight_is_separate_and_not_missing_physical_usage():
    record = {
        "client_schema_version": CLIENT_VERSION,
        "role": "judge",
        "provider_call": False,
        "accepted": False,
        "token_usage": normalize_usage(None),
        "error_kind": "preflight_error",
        "error": "Synthetic missing key",
        "latency_seconds": 0.1,
    }
    usage = _compact_usage([record])
    assert usage["calls"] == 0 and usage["preflight_failures"] == 1
    assert all(
        row == {"sum": 0, "missing_calls": 0} for row in usage["tokens"].values()
    )


def test_audited_task_seals_survive_later_worker_interruption(tmp_path):
    workers, receipts = fixture_tree(tmp_path)
    for worker, receipt in zip(workers, receipts, strict=True):
        directory = worker / f"task_{receipt['task_id']}"
        before = json.loads((worker / "frozen_weights_before.json").read_text())
        write(directory / "frozen_weights_after.json", before)
        runtime = json.loads((worker / "runtime.json").read_text())
        worker_names = (
            "runtime.json",
            "checkpoint.json",
            "protocol.json",
            "frozen_plan.json",
            "reset_manifest.json",
            "prompts.json",
            "frozen_weights_before.json",
        )
        seal = {
            "schema_version": "frs-completed-task-1.0",
            "task_id": receipt["task_id"],
            "workflow": runtime["workflow"],
            "worker": runtime["worker"],
            "native_tensor_sha256": before["sha256"],
            "task_files_sha256": {
                name: file_sha256(directory / name)
                for name in (
                    "summary.json",
                    "events.jsonl",
                    "provider.jsonl",
                    "frozen_weights_after.json",
                )
            },
            "worker_metadata_sha256": {
                name: file_sha256(worker / name) for name in worker_names
            },
        }
        write(directory / "completion_receipt.json", seal)
        (worker / "frozen_weights_after.json").unlink()
        receipt["input_file_sha256"].update(
            {
                name: file_sha256(directory / name)
                for name in ("completion_receipt.json", "frozen_weights_after.json")
            }
        )
        receipt["worker"].update(
            input_file_sha256=seal["worker_metadata_sha256"],
            weight_check_scope="completed_task",
            completion_receipt_sha256=file_sha256(
                directory / "completion_receipt.json"
            ),
            task_frozen_weights_after_sha256=file_sha256(
                directory / "frozen_weights_after.json"
            ),
        )
    report = build_report(workers, phase="development", audit_receipts=receipts)
    assert report["status"] == "complete"
    assert all(
        row["native_weight_scope"] == "completed_task_seal"
        for row in report["producer_evidence"]["tasks"]
    )
    # A stale/changed summary cannot borrow the seal or audit of the original task.
    path = workers[0] / "task_6/completion_receipt.json"
    seal = json.loads(path.read_text())
    seal["task_files_sha256"]["summary.json"] = "f" * 64
    write(path, seal)
    with pytest.raises(ValueError, match="Task seal recording bytes"):
        build_report(workers, phase="development", audit_receipts=receipts)


def test_disjoint_whole_tasks_can_be_selected_across_recovery_workflows(tmp_path):
    workers, receipts = fixture_tree(tmp_path / "original", "evaluation")
    recovered = tmp_path / "recovery/worker_0"
    shutil.copytree(workers[0], recovered)
    runtime = json.loads((recovered / "runtime.json").read_text())
    runtime["workflow"] = "synthetic-recovery-no-compute"
    plan = json.loads((recovered / "frozen_plan.json").read_text())
    plan["runtime"] = runtime
    write(recovered / "runtime.json", runtime)
    write(recovered / "frozen_plan.json", plan)
    for receipt in receipts:
        if receipt["suite"] == "libero_goal_ood" and receipt["task_id"] in (4, 8):
            receipt["worker"]["runtime"] = runtime
            for name in ("runtime.json", "frozen_plan.json"):
                receipt["worker"]["input_file_sha256"][name] = file_sha256(
                    recovered / name
                )
    inputs = [
        workers[0] / "task_0",
        recovered / "task_4",
        recovered / "task_8",
        *workers[1:],
    ]
    report = build_report(inputs, phase="evaluation", audit_receipts=receipts)
    assert report["status"] == "complete"
    assert validate_report(report)["physical_cost"]["unique_rollouts"] == 1740
    selected = {
        row["task_key"]: row["workflow"] for row in report["producer_evidence"]["tasks"]
    }
    assert selected["libero_goal_ood:0"] == "synthetic-no-compute"
    assert selected["libero_goal_ood:4"] == "synthetic-recovery-no-compute"
    with pytest.raises(ValueError, match="Duplicate task input"):
        build_report([workers[0], recovered], phase="evaluation")
    manifest = json.loads((recovered / "reset_manifest.json").read_text())
    manifest["episodes"][0]["reset_state_sha256"] = "f" * 64
    manifest["sha256"] = digest({k: v for k, v in manifest.items() if k != "sha256"})
    plan["manifest_sha256"] = manifest["sha256"]
    write(recovered / "reset_manifest.json", manifest)
    write(recovered / "frozen_plan.json", plan)
    with pytest.raises(ValueError, match="Recovery changed"):
        build_report(inputs, phase="evaluation")


def test_completed_summary_cannot_omit_a_heldout_reference(tmp_path):
    workers, receipts = fixture_tree(tmp_path)
    path = workers[0] / "task_6/summary.json"
    summary = json.loads(path.read_text())
    summary["evaluation"].pop()
    write(path, summary)
    with pytest.raises(ValueError, match="Evaluation index is incomplete"):
        build_report(workers, phase="development", audit_receipts=receipts)
