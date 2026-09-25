"""Offline feedback provenance checks over real runner records and fake inference."""

import copy
import json

import numpy as np
import pytest

from astra_reversal import interpolation_agent
from astra_reversal.interpolation_catalog import donor_catalog
from astra_reversal.interpolation_feedback_audit import (
    _Inputs,
    _snapshot_binding,
    audit_case,
)

from .test_interpolation_search import ObservedClient
from .test_interpolation_search import phase_search_factory as phase_search_factory


def events(directory):
    return [
        json.loads(line)
        for line in (directory / "events.jsonl").read_text().splitlines()
    ]


def write_events(directory, rows):
    (directory / "events.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )


@pytest.fixture
def recorded_case(request, monkeypatch):
    search_factory = request.getfixturevalue("phase_search_factory")

    def make(*, preflight=False, all_provider_fail=False, baseline_success=False):
        original_propose = ObservedClient.propose

        def donor_proposal(self, request):
            result = original_propose(self, request)
            result.update(source_a_id="10", source_b_id="13")
            return result

        monkeypatch.setattr(ObservedClient, "propose", donor_proposal)
        search, requests, _ = search_factory(
            all_provider_fail=all_provider_fail, baseline_success=baseline_success
        )
        search.catalog = donor_catalog()
        search.sources = {row["source_id"]: row["prompt"] for row in search.catalog}
        search.banks = {row["source_id"]: None for row in search.catalog}
        search.oracle.update(source_a_id="10", source_b_id="13")
        search.report["source_catalog"] = search.catalog
        initialize = search.initialize_noise

        def initialize_with_recording(observation, spec):
            initialize(observation, spec)
            search.recorder.event(
                "inversion_initialization", action_spec=spec.as_dict()
            )

        monkeypatch.setattr(search, "initialize_noise", initialize_with_recording)
        prepare = search.policy.prepare_interpolated

        def prepare_with_provenance(observation, observation_id, prompt, **kwargs):
            condition, provenance = prepare(
                observation, observation_id, prompt, **kwargs
            )
            provenance.update(
                operator=kwargs["operator"],
                alpha=kwargs["alpha"],
                source_prompts=list(kwargs["source_prompts"]),
            )
            return condition, provenance

        search.policy.prepare_interpolated = prepare_with_provenance
        if preflight:
            client_type = interpolation_agent.InterpolationClient

            class MissingCredentialOnce(client_type):
                def propose(self, request):
                    if (
                        request["attempt_id"] == "astra_tei_2"
                        and request["decision_index"] == 1
                    ):
                        with monkeypatch.context() as patch:
                            patch.delenv("NVIDIA_INFERENCE_API_KEY")
                            return super().propose(request)
                    return super().propose(request)

            monkeypatch.setattr(
                interpolation_agent, "InterpolationClient", MissingCredentialOnce
            )
        search.run()
        return search.directory, search.protocol, requests

    return make


def test_actual_raw_arrays_bind_pngs_provider_calls_and_expiring_decisions(
    recorded_case,
):
    directory, protocol, _ = recorded_case()
    result = audit_case(directory, expected_protocol=protocol)
    assert result["status"] == "passed"
    assert result["counts"]["physical_rollouts"] == 8
    assert (
        result["counts"]["requests"]
        == result["counts"]["physical_provider_calls"]
        == 10
    )
    assert result["counts"]["accepted_decisions_executed"] == 9
    assert result["provider"]["failed_calls"] == 1
    assert result["provider"]["tokens"]["total_tokens"]["sum"] == 130
    assert result["counts"]["verified_raw_arrays"] > 120
    assert (
        result["counts"]["decoded_camera_bindings"]
        == 2 * result["counts"]["raw_snapshot_bindings"]
    )
    assert result["actions"]["actions_with_held_text_after_failed_call"] == 5
    assert result["actions"]["actions_with_changed_vision"] == 20
    assert result["actions"]["native_condition_fallback_actions"] == 0
    assert {row["scope"] for row in result["snapshot_bindings"]} == {
        "current_attempt",
        "previous_attempt",
    }
    assert (
        result["input_file_sha256"]["summary.json"]
        and result["input_file_sha256"]["events.jsonl"]
    )
    assert not result["separate_transport_smoke_included"]
    serialized = json.dumps(result)
    assert "base64_png" not in serialized and 'message": {' not in serialized


def test_preflight_is_a_consumed_slot_but_not_a_physical_provider_call(recorded_case):
    directory, protocol, _ = recorded_case(preflight=True)
    result = audit_case(directory, expected_protocol=protocol)
    assert result["counts"]["requests"] == result["provider"]["client_attempts"] == 10
    assert result["counts"]["physical_provider_calls"] == 9
    assert result["counts"]["preflight_failures"] == 1
    assert result["provider"]["tokens"]["total_tokens"]["sum"] == 117
    assert result["actions"]["native_condition_fallback_actions"] == 25


def test_intact_provider_failures_remain_valid_negative_evidence(recorded_case):
    directory, protocol, _ = recorded_case(all_provider_fail=True)
    result = audit_case(directory, expected_protocol=protocol)
    assert result["provider"]["failed_calls"] == 10
    assert result["counts"]["accepted_decisions_executed"] == 0
    assert result["actions"]["native_condition_fallback_actions"] == 150
    assert result["provider"]["tokens"]["total_tokens"]["sum"] == 130


def test_raw_array_corruption_is_rejected(recorded_case):
    directory, protocol, _ = recorded_case()
    generation = next(
        row for row in events(directory) if row["kind"] == "phase_generation"
    )
    path = directory / generation["observation"]["observation/image"]["array"]
    array = np.load(path, allow_pickle=False)
    array[0, 0, 0] ^= 1
    np.save(path, array, allow_pickle=False)
    with pytest.raises(ValueError, match="raw array integrity"):
        audit_case(directory, expected_protocol=protocol)


def test_png_comparison_reads_array_bytes_instead_of_trusting_descriptors(
    recorded_case,
):
    directory, _, _ = recorded_case()
    rows = events(directory)
    request_event = next(row for row in rows if row["kind"] == "interpolation_request")
    request = request_event["request"]
    generation = next(
        row
        for row in rows
        if row["kind"] == "phase_generation"
        and row["attempt_id"] == request["attempt_id"]
        and row["observation_step"] == 0
    )
    wire = copy.deepcopy(request["observations"][0])
    wire["observation"]["observation/image"] = wire["observation"][
        "observation/wrist_image"
    ]
    saved = {
        "label": wire["label"],
        "step": 0,
        "observation": generation["observation"],
    }
    with pytest.raises(ValueError, match="PNG pixels differ"):
        _snapshot_binding(
            _Inputs(directory),
            wire,
            saved,
            fingerprint=request["request_fingerprint"],
            scope="current_attempt",
            attempt_id=request["attempt_id"],
            sequence=generation["sequence"],
        )


@pytest.mark.parametrize(
    "mutation", ["stale_vision", "lost_hold", "silent_text", "fallback_count"]
)
def test_applied_decision_or_effect_count_tampering_is_rejected(
    recorded_case, mutation
):
    directory, protocol, _ = recorded_case()
    rows = events(directory)
    if mutation == "fallback_count":
        summary = json.loads((directory / "summary.json").read_text())
        summary["controls"]["known_noise"]["accepted_decision_policy_calls"] = 9
        attempt = next(
            row
            for row in rows
            if row["kind"] == "phase_attempt"
            and row["attempt"]["attempt_id"] == "known_noise_1"
        )
        attempt["attempt"]["accepted_decision_policy_calls"] = 9
        (directory / "summary.json").write_text(json.dumps(summary))
    else:
        mode, step = (
            ("astra_tli_2", 25)
            if mutation == "lost_hold"
            else ("astra_tli_vision_2", 5)
        )
        target = next(
            row
            for row in rows
            if row["kind"] == "phase_generation"
            and row["attempt_id"] == mode
            and row["observation_step"] == step
        )
        if mutation == "stale_vision":
            initial = next(
                row
                for row in rows
                if row["kind"] == "phase_generation"
                and row["attempt_id"] == mode
                and row["observation_step"] == 0
            )
            target["vision"] = initial["vision"]
        elif mutation == "lost_hold":
            target["active_interpolation"] = None
        else:
            target["conditioning"]["source_prompts"] = [
                "unreported source",
                "unreported source",
            ]
    write_events(directory, rows)
    with pytest.raises(
        ValueError,
        match="Applied text/vision|Conditioning provenance|attribution count",
    ):
        audit_case(directory, expected_protocol=protocol)


def test_unassigned_sidecar_and_extra_physical_call_are_rejected(recorded_case):
    directory, protocol, _ = recorded_case()
    (directory / "hidden_provider.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="unassigned provider sidecar"):
        audit_case(directory, expected_protocol=protocol)


def test_future_current_frame_event_order_is_rejected(recorded_case):
    directory, protocol, _ = recorded_case()
    rows = events(directory)
    request_index = next(
        index
        for index, row in enumerate(rows)
        if row["kind"] == "interpolation_request"
    )
    request = rows[request_index]["request"]
    generation_index = next(
        index
        for index, row in enumerate(rows)
        if row["kind"] == "phase_generation"
        and row["attempt_id"] == request["attempt_id"]
        and row["observation_step"] == 0
    )
    rows.insert(request_index, rows.pop(generation_index))
    for index, row in enumerate(rows):
        row["sequence"] = index
    write_events(directory, rows)
    with pytest.raises(ValueError, match="request/decision/generation order"):
        audit_case(directory, expected_protocol=protocol)


def test_oracle_mapping_cannot_be_added_to_request(recorded_case):
    directory, protocol, _ = recorded_case()
    rows = events(directory)
    request = next(
        row["request"] for row in rows if row["kind"] == "interpolation_request"
    )
    request["oracle_source_pair"] = ["10", "13"]
    write_events(directory, rows)
    with pytest.raises(ValueError, match="exactly"):
        audit_case(directory, expected_protocol=protocol)


def test_other_arm_feedback_cannot_be_substituted(recorded_case):
    directory, protocol, _ = recorded_case()
    rows = events(directory)
    request = next(
        row["request"]
        for row in rows
        if row["kind"] == "interpolation_request"
        and row["request"]["attempt_id"] == "astra_tli_vision_3"
    )
    request["completed_rollout_feedback"][-1]["attempt_id"] = "astra_tei_2"
    write_events(directory, rows)
    with pytest.raises(ValueError, match="feedback|fingerprint"):
        audit_case(directory, expected_protocol=protocol)


def test_shared_successful_baseline_has_zero_provider_or_smoke_cost(recorded_case):
    directory, protocol, _ = recorded_case(baseline_success=True)
    result = audit_case(directory, expected_protocol=protocol)
    assert result["counts"]["physical_rollouts"] == 3
    assert (
        result["counts"]["requests"]
        == result["counts"]["accepted_decisions_executed"]
        == 0
    )
    assert result["provider"]["tokens"]["total_tokens"]["sum"] == 0
    assert result["snapshot_bindings"] == []
    assert not result["separate_transport_smoke_included"]


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("model", "substituted"),
        ("cache", "sampling/cache"),
        ("fingerprint", "different request identity"),
        ("usage", "disagrees"),
        ("extra_call", "hidden retry"),
    ],
)
def test_provider_binding_and_no_hidden_retry_even_when_summary_is_altered(
    recorded_case, mutation, match
):
    directory, protocol, _ = recorded_case()
    path = directory / "astra_tei_2_provider.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    if mutation == "model":
        records[0]["response"]["model"] = "synthetic-unconfigured-model"
    elif mutation == "cache":
        records[0]["cache"] = {"enabled": False}
    elif mutation == "fingerprint":
        records[0]["request_fingerprint"] = "0" * 64
    elif mutation == "usage":
        records[0]["token_usage"]["total_tokens"] += 1
    else:
        records.append(copy.deepcopy(records[-1]))
    path.write_text("".join(json.dumps(row) + "\n" for row in records))
    report = json.loads((directory / "summary.json").read_text())
    report["arms"]["astra_tei"]["attempts"][1]["provider_records"] = records
    (directory / "summary.json").write_text(json.dumps(report))
    rows = events(directory)
    for event in rows:
        if (
            event["kind"] == "phase_attempt"
            and event["attempt"]["attempt_id"] == "astra_tei_2"
        ):
            event["attempt"]["provider_records"] = records
    write_events(directory, rows)
    with pytest.raises(ValueError, match=match):
        audit_case(directory, expected_protocol=protocol)
