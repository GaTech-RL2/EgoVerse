"""Synthetic accounting fixtures; never experiment evidence or provider calls."""

import copy
import hashlib
import json

import pytest

from astra_reversal.intervention_agent import normalize_usage
from astra_reversal.records import digest, file_sha256
from astra_reversal.representation_agent import summarize_calls
from astra_reversal.representation_report import build_report, write_outputs
from astra_reversal.representation_search import ARMS, load_protocol, summarize_attempts


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def attempt(entry, arm, revision, success, protocol):
    decisions, calls, choice = [], [], None
    failed = (entry["task_id"], arm, revision) in (
        (6, "astra_tei", 1),
        (2, "astra_vli", 1),
    )
    if arm not in ("native", "native_retry"):
        choice = {"mode": "native", "language": None, "vision": None}
        if arm.startswith("astra_"):
            choice.update(
                decision_id=digest([entry["episode_id"], arm, revision]),
                request_fingerprint=digest(
                    [entry["episode_id"], arm, revision, "request"]
                ),
            )
            usage = {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "completion_tokens_details": {"reasoning_tokens": 2},
            }
            if failed and entry["task_id"] == 2:
                usage = None
            call = {
                "client_schema_version": "representation-1.0",
                "provider_call": True,
                "accepted": not failed,
                "episode_id": entry["episode_id"],
                "attempt_id": f"{arm}_revision{revision}",
                "decision_index": 1,
                "observation_step": 0,
                "representation_mode": arm.removeprefix("astra_"),
                "requested_model": protocol["astra"]["model"],
                "sampling_settings": {
                    k: protocol["astra"][k]
                    for k in ("reasoning_effort", "max_completion_tokens")
                },
                "cache": {"no-cache": True},
                "response": {"model": protocol["astra"]["model"], "usage": usage},
                "token_usage": normalize_usage(usage),
                "latency_seconds": 0.25,
                "decision_id": choice["decision_id"],
                "request_fingerprint": choice["request_fingerprint"],
            }
            if failed:
                call.update(
                    error_kind="http_error",
                    error="Synthetic HTTP error",
                    http_status=503,
                )
            calls.append(call)
        if failed:
            choice = None
        decisions.append(
            {
                "decision_index": 1,
                "observation_step": 0,
                "accepted": not failed,
                "proposal": choice,
                "error": "Synthetic HTTP error" if failed else None,
            }
        )
    accepted = arm.startswith("astra_") and not failed
    return {
        "episode_id": entry["episode_id"],
        "attempt_id": f"{arm}_revision{revision}",
        "arm": arm,
        "revision": revision,
        "success": success,
        "initial_success": False,
        "terminated": not success,
        "status": "success" if success else "terminated",
        "actions_executed": 3,
        "policy_replans": 1,
        "velocity_evaluations": 10,
        "parity_velocity_evaluations": 10 if arm == "native" else 0,
        "probe_velocity_evaluations": 90 if arm == "native" else 0,
        "donor_captures": 0,
        "wall_seconds": 1.0,
        "policy_seconds": 0.7,
        "environment_seconds": 0.1,
        "condition_seconds": 0.05,
        "donor_capture_seconds": 0.0,
        "decisions": decisions,
        "provider_records": calls,
        "generations": [
            {
                "step": 0,
                "choice": choice,
                "has_effect": False,
                "provider_failure_fallback": failed,
            }
        ],
        "accepted_decision_actions": 3 * accepted,
        "explicit_native_actions": 3 * accepted,
        "nonzero_intervention_actions": 0,
        "provider_failure_fallback_actions": 3 * failed,
        "accepted_decisions_executed": int(accepted),
        "reset_audit": {"state": digest(entry)},
        "video_path": f"/private/remote/{arm}_revision{revision}.mp4",
        "video_sha256": "e" * 64,
        "video_fps": 20,
    }


def seal(path, summary, protocol, worker, manifest):
    write(path / "summary.json", summary)
    (path / "events.jsonl").write_text('{"synthetic":true}\n')
    calls = [c for r in summary["physical_rollouts"] for c in r["provider_records"]]
    (path / "provider.jsonl").write_text("".join(json.dumps(c) + "\n" for c in calls))
    metadata = {
        "runtime.json": {
            "workflow": "synthetic-workflow",
            "worker": worker,
            "phase": "development",
            "tf32": False,
            "gpu": "NVIDIA L40S",
            "payload_sha256": "a" * 64,
            "packages": {"numpy": "1.26.4"},
            "python": "3.11",
        },
        "protocol.json": protocol,
        "prompts.json": {"system_prompt": "Synthetic fixture"},
        "checkpoint.json": {"checkpoint_sha256": "b" * 64},
        "reset_manifest.json": manifest,
        "frozen_weights_before.json": {"sha256": "c" * 64},
        "bank_inventory.json": {"donors": "synthetic"},
    }
    for name, value in metadata.items():
        write(path.parent / name, value)
    metadata["frozen_plan.json"] = {
        "protocol_sha256": digest(protocol),
        "reset_manifest_sha256": manifest["sha256"],
        "assigned_episodes": [summary["episode_id"]],
        "image_library_id": "d" * 64,
        "bank_inventory_sha256": file_sha256(path.parent / "bank_inventory.json"),
    }
    write(path.parent / "frozen_plan.json", metadata["frozen_plan.json"])
    write(path / "frozen_weights_after.json", metadata["frozen_weights_before.json"])
    receipt = {
        "schema_version": "representation-complete-case-1.0",
        "episode_id": summary["episode_id"],
        "workflow": "synthetic-workflow",
        "worker": worker,
        "native_tensor_sha256": "c" * 64,
        "worker_files": {name: file_sha256(path.parent / name) for name in metadata},
        "case_files": {
            name: {
                "sha256": file_sha256(path / name),
                "bytes": (path / name).stat().st_size,
            }
            for name in (
                "summary.json",
                "events.jsonl",
                "provider.jsonl",
                "frozen_weights_after.json",
            )
        },
    }
    write(path / "completion_receipt.json", receipt)
    physical = summary["physical_cost"]
    arrays = {
        "schema_version": "representation-case-array-audit-1",
        "status": "passed",
        "checks": {"complete_sealed_case": True},
        "episode_id": summary["episode_id"],
        "archive": {
            "key": "synthetic/task.tar.gz",
            "sha256": "f" * 64,
            "bytes": 100,
            "gzip_crc_verified": True,
        },
        "input_file_sha256": {
            name: file_sha256(path / name)
            for name in (*receipt["case_files"], "completion_receipt.json")
        },
        "worker_metadata_sha256": receipt["worker_files"],
        "identities": {
            "protocol_sha256": digest(protocol),
            "reset_manifest_sha256": manifest["sha256"],
            "image_library_id": "d" * 64,
            "bank_inventory_sha256": file_sha256(path.parent / "bank_inventory.json"),
            "native_tensor_sha256": "c" * 64,
            "payload_sha256": "a" * 64,
        },
        "counts": {
            "physical_rollouts": physical["rollouts"],
            "actions": physical["actions"],
            "flow_velocity_evaluations": 10 * physical["rollouts"],
            "native_parity_velocity_evaluations": 10,
            "probe_velocity_evaluations": 90,
            "velocity_evaluations": physical["velocity_evaluations"],
            "arrays": 1,
            "generations": physical["rollouts"],
        },
    }
    providers = {
        "schema_version": "representation-provider-audit-1.0",
        "status": "complete",
        "complete": True,
        "episode_id": summary["episode_id"],
        "protocol_sha256": digest(protocol),
        "image_library_id": "d" * 64,
        "source_file_sha256": arrays["input_file_sha256"],
        "provider": physical["provider"],
        "source_code_sha256": {"synthetic.py": "0" * 64},
    }
    return arrays, providers


@pytest.fixture
def cohort(tmp_path):
    protocol = load_protocol()
    protocol["seed"] = 19
    entries = [
        {
            "episode_id": f"{suite}:seed19:task{task}:state0",
            "suite": suite,
            "task_id": task,
            "seed": 19,
            "initial_state_id": 0,
            "instruction": f"Synthetic task {task}",
        }
        for suite, task in [
            ("libero_goal_ood", 6),
            ("libero_spatial_ood", 2),
            ("libero_spatial_ood", 8),
        ]
    ]
    directories, summaries, arrays, providers = [], {}, {}, {}
    for worker, entry in enumerate(entries):
        baseline = attempt(entry, "native", 0, entry["task_id"] == 8, protocol)
        physical, arms = [baseline], {}
        for arm in ARMS:
            attempts = [baseline]
            for revision in (1, 2):
                if revision == 2 and any(r["success"] for r in attempts):
                    break
                success = entry["task_id"] == 6 and arm == "astra_tli" and revision == 1
                row = attempt(entry, arm, revision, success, protocol)
                physical.append(row)
                attempts.append(row)
            arms[arm] = summarize_attempts(attempts, 2)
        cost = {
            "rollouts": len(physical),
            "actions": 3 * len(physical),
            "velocity_evaluations": 10 * len(physical) + 100,
            "provider": summarize_calls(
                [c for r in physical for c in r["provider_records"]]
            ),
            "donor_captures": 0,
            "donor_capture_seconds": 0.0,
            "rollout_wall_seconds": sum(r["wall_seconds"] for r in physical),
            "policy_seconds": sum(r["policy_seconds"] for r in physical),
            "environment_seconds": sum(r["environment_seconds"] for r in physical),
            "condition_seconds": sum(r["condition_seconds"] for r in physical),
        }
        summary = {
            **entry,
            "schema_version": protocol["schema_version"],
            "status": "complete",
            "development": True,
            "protocol_sha256": digest(protocol),
            "reset_entry_sha256": digest(entry),
            "image_library_id": "d" * 64,
            "baseline": baseline,
            "physical_rollouts": physical,
            "arms": arms,
            "physical_cost": cost,
            "checks": {
                "native_parity": {"max_abs": 0},
                "weighted_vision": {"status": "passed", "velocity_evaluations": 90},
            },
        }
        directory = tmp_path / f"worker_{worker}" / f"task_{entry['task_id']}"
        selected = [e for e in entries if e["suite"] == entry["suite"]]
        manifest = {"episodes": selected, "sha256": digest(selected)}
        a, p = seal(directory, summary, protocol, worker, manifest)
        directories.append(directory)
        summaries[entry["episode_id"]] = summary
        arrays[entry["episode_id"]] = a
        providers[entry["episode_id"]] = p
    return {
        "directories": directories,
        "summaries": summaries,
        "protocol": protocol,
        "arrays": arrays,
        "providers": providers,
    }


def report(cohort):
    return build_report(
        cohort["directories"],
        phase="development",
        protocol=cohort["protocol"],
        array_audits=cohort["arrays"],
        provider_audits=cohort["providers"],
    )


def test_shared_baseline_known_missing_usage_and_rescue_denominators(cohort, tmp_path):
    value = report(cohort)
    assert value["complete"] and value["audited_cases"] == 3
    pooled = value["groups"]["pooled"]
    cost = pooled["physical_cost"]
    assert (cost["rollouts"], cost["actions"], cost["velocity_evaluations"]) == (
        52,
        156,
        820,
    )
    assert cost["wall_seconds"] == 52
    assert cost["provider"]["provider_calls"] == 29
    assert cost["provider"]["tokens"]["total_tokens"]["sum"] == 420
    assert cost["provider"]["tokens"]["total_tokens"]["complete"] is False
    assert cost["rejected_call_usage"]["provider_calls"] == 2
    assert cost["rejected_call_usage"]["tokens"]["total_tokens"]["sum"] == 15
    assert cost["provider_failure_fallback_actions"] == 6
    tli = pooled["arms"]["astra_tli"]
    assert tli["successes_by_revision_budget"] == [1, 2, 2]
    assert tli["rescues"] == 1 and tli["censored"] == 1
    assert tli["rescue_only"]["revisions_to_success"]["median"] == 1
    assert tli["rescue_only"]["complete_tokens_to_success"]["median"] == 15
    assert tli["development_extra_rollouts"] == 1
    assert tli["provider_by_revision_budget"][0]["provider_calls"] == 0
    assert tli["provider_by_revision_budget"][2]["provider_calls"] == 3
    stability = value["development_stability_rows"]
    assert len(stability) == 10 and all(not r["recorded_success"] for r in stability)
    assert all(r["episode_id"].endswith("task8:state0") for r in stability)
    output = tmp_path / "public"
    write_outputs(output, value)
    assert "lower bound" in (output / "report.md").read_text()
    assert "/private/remote" not in (output / "report.json").read_text()
    plotted = json.loads((output / "plotted_values.json").read_text())
    assert plotted["report_sha256"] == file_sha256(output / "report.json")
    assert plotted["successes_by_revision_budget"]["astra_tli"] == [1, 2, 2]
    assert (output / "physical_token_cost.pdf").is_file()
    with pytest.raises(FileExistsError):
        write_outputs(output, value)


@pytest.mark.parametrize("missing", ["case", "array", "provider"])
def test_partial_efficacy_withheld_costs_preserved(cohort, missing):
    if missing == "case":
        cohort["directories"].pop()
    else:
        cohort["arrays" if missing == "array" else "providers"].pop(
            next(iter(cohort["summaries"]))
        )
    value = report(cohort)
    assert not value["efficacy_released"] and not value["complete"]
    assert value["groups"]["pooled"]["arms"] is None
    assert value["groups"]["pooled"]["paired"] is None
    assert value["groups"]["pooled"]["physical_cost"]["rollouts"] > 0
    assert value["case_arm_rows"] == []


@pytest.mark.parametrize("mutation", ["array_summary", "provider_usage", "archive_crc"])
def test_audit_binding_failure_suppresses_results(cohort, mutation):
    identity = next(iter(cohort["arrays"]))
    if mutation == "array_summary":
        cohort["arrays"][identity]["input_file_sha256"]["summary.json"] = "0" * 64
    elif mutation == "archive_crc":
        cohort["arrays"][identity]["archive"]["gzip_crc_verified"] = False
    else:
        cohort["providers"][identity] = copy.deepcopy(cohort["providers"][identity])
        cohort["providers"][identity]["provider"]["tokens"]["total_tokens"]["sum"] += 1
    value = report(cohort)
    assert value["status"] == "audit_failed" and not value["efficacy_released"]


@pytest.mark.parametrize(
    "mutation",
    [
        "reset",
        "cost",
        "initial_success",
        "missed_revision",
        "active_after_failure",
        "usage",
        "duplicate_case",
    ],
)
def test_corrupt_summary_never_releases_efficacy(cohort, mutation):
    directory = cohort["directories"][0]
    summary = json.loads((directory / "summary.json").read_text())
    if mutation == "reset":
        summary["physical_rollouts"][1]["reset_audit"] = {"state": "wrong"}
    elif mutation == "cost":
        summary["physical_cost"]["actions"] += 1
    elif mutation == "initial_success":
        summary["physical_rollouts"][0]["initial_success"] = True
    elif mutation == "missed_revision":
        summary["physical_rollouts"].pop()
    elif mutation == "active_after_failure":
        row = next(
            r
            for r in summary["physical_rollouts"]
            if r["arm"] == "astra_tei" and r["revision"] == 1
        )
        row["generations"][0]["choice"] = {"mode": "native"}
    elif mutation == "usage":
        row = next(r for r in summary["physical_rollouts"] if r["provider_records"])
        row["provider_records"][0]["token_usage"]["total_tokens"] += 1
    else:
        cohort["directories"].append(directory)
    write(directory / "summary.json", summary)
    with pytest.raises(ValueError):
        report(cohort)


def test_wrong_phase_and_seed_rejected(cohort):
    with pytest.raises(ValueError, match="seed"):
        build_report(
            cohort["directories"], phase="evaluation", protocol=cohort["protocol"]
        )


def test_archive_layout_with_sibling_metadata(cohort):
    for directory in cohort["directories"]:
        target = directory.parent / "metadata"
        target.mkdir()
        for path in list(directory.parent.glob("*.json")):
            path.rename(target / path.name)
    assert report(cohort)["complete"]


def test_partial_outputs_do_not_plot_or_hide_costs(cohort, tmp_path):
    cohort["directories"].pop()
    output = tmp_path / "partial"
    write_outputs(output, report(cohort))
    assert not list(output.glob("*.png")) and not list(output.glob("*.pdf"))
    assert "withheld" in (output / "report.md").read_text()


@pytest.mark.parametrize("bad_event_digest", [False, True])
@pytest.mark.parametrize("bad_bound", [None, "weaker_actions", "missing_compute"])
def test_interrupted_archive_releases_only_bound_completed_observations(
    cohort, tmp_path, bad_event_digest, bad_bound
):
    directory = cohort["directories"][0]
    summary = json.loads((directory / "summary.json").read_text())
    episode = summary["episode_id"]
    summary["status"] = "running"
    summary["physical_rollouts"] = summary["physical_rollouts"][:2]
    summary["arms"] = {}
    summary.pop("physical_cost")
    write(directory / "summary.json", summary)
    lines = [
        (
            json.dumps({"kind": "representation_rollout_complete", "attempt": row})
            + "\n"
        ).encode()
        for row in summary["physical_rollouts"]
    ]
    (directory / "events.jsonl").write_bytes(b"".join(lines))
    call = copy.deepcopy(
        next(
            c
            for s in cohort["summaries"].values()
            for r in s["physical_rollouts"]
            for c in r["provider_records"]
        )
    )
    call.update(
        accepted=False,
        http_status=429,
        error_kind="http_error",
        error="Inference endpoint returned HTTP 429",
        response={"model": cohort["protocol"]["astra"]["model"]},
        token_usage=normalize_usage(None),
        provider_error={"type": "budget_exceeded", "message": "sensitive-key-alias"},
    )
    (directory / "provider.jsonl").write_text(json.dumps(call) + "\n")
    (directory / "completion_receipt.json").unlink()
    (directory / "frozen_weights_after.json").unlink()
    hashes = {
        name: file_sha256(directory / name)
        for name in ("summary.json", "events.jsonl", "provider.jsonl")
    }
    arrays = cohort["arrays"][episode]
    arrays.update(
        schema_version="representation-partial-array-audit-1",
        status="verified_partial_archive",
        checks={
            "complete_sealed_case": False,
            "full_archive_stream_verified": True,
            "all_available_npy_references_verified": True,
            "published_small_files_match_archive": True,
        },
        input_file_sha256=hashes,
    )
    arrays["counts"].update(
        physical_rollouts=2,
        actions=6,
        flow_velocity_evaluations=70,
        velocity_evaluations=170,
        inflight_attempts=1,
        recorded_action_lower_bound=26,
        generations=7,
    )
    arrays["identities"]["native_tensor_sha256_before"] = arrays["identities"].pop(
        "native_tensor_sha256"
    )
    arrays["identities"]["native_tensor_sha256_after"] = None
    arrays["verified_completed_attempts"] = [
        {
            "attempt_id": r["attempt_id"],
            "arm": r["arm"],
            "revision": r["revision"],
            "success": r["success"],
            "actions": r["actions_executed"],
            "attempt_sha256": digest(r),
            "reset_audit_sha256": digest(r["reset_audit"]),
            "completed_event_sha256": hashlib.sha256(line).hexdigest(),
        }
        for r, line in zip(summary["physical_rollouts"], lines, strict=True)
    ]
    if bad_event_digest:
        arrays["verified_completed_attempts"][0]["completed_event_sha256"] = "0" * 64
    provider = cohort["providers"][episode]
    provider.update(
        status="preserved_prefix",
        complete=False,
        provider=summarize_calls([call]),
        source_file_sha256={
            name: {"sha256": sha, "entire_file": True} for name, sha in hashes.items()
        },
        physical_completed_rollouts=2,
        physical_completed_actions=6,
        physical_completed_velocity_evaluations=120,
        recorded_flow_velocity_evaluations=70,
        recorded_total_velocity_evaluations_lower_bound=170,
        all_recorded_actions_bounds={"lower_bound": 31, "upper_bound": 306},
        incomplete_attempts=[
            {
                "attempt_id": "astra_vli_revision2",
                "arm": "astra_vli",
                "executed_actions_lower_bound": 25,
                "executed_actions_upper_bound": 300,
                "flow_velocity_evaluations_recorded": 50,
                "outcome": "unknown_no_completed_rollout_record",
            }
        ],
    )
    if bad_bound == "weaker_actions":
        provider["incomplete_attempts"][0]["executed_actions_lower_bound"] = 19
        provider["all_recorded_actions_bounds"]["lower_bound"] = 25
    elif bad_bound == "missing_compute":
        provider["incomplete_attempts"][0]["flow_velocity_evaluations_recorded"] = 40
    value = report(cohort)
    assert not value["complete"] and not value["efficacy_released"]
    observed = [
        r
        for r in value["verified_completed_rollout_observations"]
        if r["episode_id"] == episode
    ]
    assert len(observed) == (0 if bad_event_digest or bad_bound else 2)
    if not bad_event_digest and not bad_bound:
        bound = value["cases"][0]["audit"]["recorded_work"]
        assert bound["completed_actions"] == 6
        assert bound["all_actions_lower_bound"] == 31
        assert bound["array_only_actions_lower_bound"] == 26
        assert bound["completed_velocity_evaluations"] == 120
        assert bound["recorded_velocity_evaluations_lower_bound"] == 170
    assert (
        value["groups"]["pooled"]["physical_cost"]["provider_error_taxonomy"][
            "budget_exceeded"
        ]
        == 1
    )
    output = tmp_path / "interrupted"
    write_outputs(output, value)
    assert "sensitive-key-alias" not in (output / "report.json").read_text()
    assert "budget_exceeded" in (output / "report.md").read_text()
    assert not list(output.glob("*.png"))
