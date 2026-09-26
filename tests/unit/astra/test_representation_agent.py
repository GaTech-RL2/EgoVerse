"""CPU contract and transport checks; synthetic images, no provider calls."""

import base64
import copy
import hashlib
import io
import json
import urllib.error
from functools import lru_cache

import numpy as np
import pytest
from PIL import Image

from astra_reversal.agent import CAMERAS
from astra_reversal.astra_client import ClientError
from astra_reversal.image_perturbation_agent import CONTACT_SHEET_LAYOUT
from astra_reversal.records import digest
from astra_reversal.representation_agent import (
    LANGUAGE_MODES,
    PROMPT_TEMPLATE_VERSION,
    REPRESENTATION_MODES,
    SCHEMA_VERSION,
    SYSTEM_PROMPT,
    VISION_MODES,
    RepresentationClient,
    build_payload,
    build_request,
    parse_proposal,
    response_schema,
    summarize_calls,
)

MODEL = "synthetic-provider/representation-contract-test"
TEST_KEY = "synthetic-representation-key-not-a-real-credential"


def snapshot(step=0, *, value=17, label="current raw observation"):
    return {
        "label": label,
        "step": step,
        "observation": {
            **{
                camera: np.full((224, 224, 3), value + index, np.uint8)
                for index, camera in enumerate(CAMERAS)
            },
            "observation/state": np.asarray(
                [0.3, 0.2, 0.1, 0, 0, 0, 0.02, -0.02], np.float32
            ),
        },
    }


@lru_cache(maxsize=1)
def library():
    sources = [
        {"source_id": "10", "prompt": "Pick up the synthetic cup."},
        {"source_id": "13", "prompt": "Put the cup in the synthetic bowl."},
    ]
    rows = []
    for index, source in enumerate(sources):
        row = {
            "donor_id": f"std{source['source_id']}-e379-f{index}",
            "library_id": "a" * 64,
            **source,
            "episode_index": 379,
            "frame_index": index,
            "phase": {"numerator": index, "denominator": 4},
            "preview_position": {"row": index, "column": 0},
            "cameras": {
                camera: {
                    "shape": [224, 224, 3],
                    "dtype": "uint8",
                    "pixels_sha256": digest(
                        np.full((224, 224, 3), 31 + index, np.uint8)
                    ),
                }
                for camera in CAMERAS
            },
        }
        row["sample_sha256"] = digest(
            {k: v for k, v in row.items() if k != "library_id"}
        )
        rows.append(row)
    sheets = []
    for index, camera in enumerate(CAMERAS):
        pixels = np.full(CONTACT_SHEET_LAYOUT["canvas_shape"], 51 + index, np.uint8)
        stream = io.BytesIO()
        Image.fromarray(pixels).save(stream, format="PNG")
        sheets.append(
            {
                "camera": camera,
                "image": pixels,
                "sha256": digest(pixels),
                "file_sha256": hashlib.sha256(stream.getvalue()).hexdigest(),
                "layout": copy.deepcopy(CONTACT_SHEET_LAYOUT),
                "library_id": "a" * 64,
            }
        )
    return sources, rows, sheets


def request_for(**overrides):
    sources, catalog, sheets = library()
    values = {
        "episode_id": "synthetic-episode",
        "attempt_id": "synthetic-tli-vli-r1",
        "decision_index": 1,
        "observation_step": 0,
        "representation_mode": "tli_vli",
        "target_task": "Put the cup in the bowl.",
        "source_catalog": sources,
        "donor_catalog": catalog,
        "contact_sheets": sheets,
        "observations": [snapshot()],
    }
    return build_request(**(values | overrides))


def proposal_for(request, **overrides):
    return {
        "request_id": request["request_id"],
        "mode": "interpolate",
        "language": {"source_a_id": "10", "source_b_id": "13", "alpha": 0.25}
        if request["representation_mode"] in LANGUAGE_MODES
        else None,
        "vision": {"donor_id": "std13-e379-f1", "alpha": 0.75}
        if request["representation_mode"] in VISION_MODES
        else None,
        "observed_phase": "cup remains on table",
        "rationale": "The cup is still on the table, so test the acquisition donor.",
        **overrides,
    }


def decision_row(request, *, accepted=True):
    return {
        "decision_index": request["decision_index"],
        "observation_step": request["observation_step"],
        "accepted": accepted,
        "error": None if accepted else "Inference endpoint returned HTTP 503",
        "proposal": parse_proposal(proposal_for(request), request)
        if accepted
        else None,
    }


def feedback(attempt_id="synthetic-baseline", *, executed_actions=300):
    return {
        "attempt_id": attempt_id,
        "success": False,
        "executed_actions": executed_actions,
        "termination": "action_budget",
        "error": None,
    }


def rehash(request):
    request["request_fingerprint"] = digest(
        {
            k: v
            for k, v in request.items()
            if k not in ("request_fingerprint", "request_id")
        }
    )
    request["request_id"] = request["request_fingerprint"][:16]
    return request


def envelope_for(request, *, usage=True, proposal=None):
    result = {
        "id": "synthetic-completion",
        "model": MODEL,
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": json.dumps(proposal or proposal_for(request)),
                },
            }
        ],
    }
    if usage:
        result["usage"] = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "completion_tokens_details": {"reasoning_tokens": 20},
        }
    return result


class FakeOpener:
    def __init__(self, *, envelope=None, error=None):
        self.envelope, self.error, self.calls = envelope, error, []

    def open(self, request, timeout):
        self.calls.append((request, timeout))
        if self.error is not None:
            raise self.error
        response = io.BytesIO(json.dumps(self.envelope).encode())
        response.status = 200
        return response


@pytest.fixture
def client_factory(monkeypatch, tmp_path):
    monkeypatch.setenv("NVIDIA_INFERENCE_API_KEY", TEST_KEY)

    def make(*, envelope=None, error=None, **kwargs):
        client = RepresentationClient(
            model=MODEL, response_log=tmp_path / "provider.jsonl", **kwargs
        )
        client.opener = FakeOpener(envelope=envelope, error=error)
        return client

    return make


@pytest.mark.parametrize("mode", REPRESENTATION_MODES)
def test_modes_select_only_their_declared_channels_and_bind_identity_locally(mode):
    request = request_for(representation_mode=mode)
    wire = proposal_for(request)
    result = parse_proposal(json.dumps(wire), request)
    assert set(wire) == {
        "request_id",
        "mode",
        "language",
        "vision",
        "observed_phase",
        "rationale",
    }
    assert result["request_fingerprint"] == request["request_fingerprint"]
    assert result["decision_id"] == result["request_id"]
    assert result["representation_mode"] == mode
    assert result["attempt_id"] == request["attempt_id"]
    assert set(response_schema(request)["properties"]) == set(wire)
    assert (result["language"] is not None) == (mode in LANGUAGE_MODES)
    assert (result["vision"] is not None) == (mode in VISION_MODES)
    native = parse_proposal(
        proposal_for(request, mode="native", language=None, vision=None), request
    )
    assert native["mode"] == "native"


def test_text_and_visual_neutral_coefficients_remain_distinct():
    combined = request_for()
    zero = proposal_for(
        combined,
        language={"source_a_id": "10", "source_b_id": "13", "alpha": 0.5},
        vision={"donor_id": "std13-e379-f1", "alpha": 0},
    )
    assert parse_proposal(zero, combined)["language"]["alpha"] == 0.5
    assert parse_proposal(zero, combined)["vision"]["alpha"] == 0
    tei = request_for(representation_mode="tei")
    source_a = proposal_for(
        tei, language={"source_a_id": "10", "source_b_id": "13", "alpha": 0}
    )
    assert parse_proposal(source_a, tei)["mode"] == "interpolate"
    assert "Alpha 0 is NOT" in SYSTEM_PROMPT
    assert "(1-2alpha)*(T_A-T_B)" in SYSTEM_PROMPT
    assert "INDEPENDENT alpha" in SYSTEM_PROMPT
    assert "clears ALL previous choices" in SYSTEM_PROMPT
    assert "do not repeat native or identical choices" in SYSTEM_PROMPT


@pytest.mark.parametrize(
    "change",
    [
        {"request_id": "0" * 16},
        {"request_fingerprint": "0" * 64},
        {"mode": "native"},
        {"mode": "other"},
        {"language": None},
        {"vision": None},
        {"language": {"source_a_id": "oracle", "source_b_id": "13", "alpha": 0.5}},
        {"vision": {"donor_id": "unlisted-frame", "alpha": 0.4}},
        {"vision": {"donor_id": "std13-e379-f1", "alpha": True}},
        {"vision": {"donor_id": "std13-e379-f1", "alpha": -0.01}},
        {"language": {"source_a_id": "10", "source_b_id": "13", "alpha": 1.01}},
        {"rationale": "x" * 321},
    ],
)
def test_rejects_stale_incomplete_unbounded_or_extra_response_fields(change):
    request = request_for()
    with pytest.raises(ClientError):
        parse_proposal(proposal_for(request, **change), request)


def test_channel_injection_and_duplicate_json_fields_are_rejected():
    request = request_for(representation_mode="vei")
    with pytest.raises(ClientError, match="Language is disabled"):
        parse_proposal(
            proposal_for(
                request,
                language={"source_a_id": "10", "source_b_id": "13", "alpha": 0.5},
            ),
            request,
        )
    raw = json.dumps(proposal_for(request))
    with pytest.raises(ClientError, match="duplicate"):
        parse_proposal(raw[:-1] + ',"mode":"native"}', request)


def test_raw_current_prior_and_training_pixels_are_separate_and_exact():
    prior_feedback = feedback()
    prior = snapshot(300, value=71, label="prior final raw frame")
    current = snapshot(value=91)
    request = request_for(
        completed_rollout_feedback=[prior_feedback],
        previous_attempt={
            "feedback": prior_feedback,
            "decisions": [],
            "snapshots": [prior],
        },
        observations=[current],
    )
    payload = build_payload(request, MODEL)
    assert payload["reasoning_effort"] == "medium"
    assert payload["max_completion_tokens"] == 8192
    assert payload["cache"] == {"no-cache": True}
    assert payload["response_format"] == {"type": "json_object"}
    content = payload["messages"][1]["content"]
    described = json.loads(content[0]["text"])
    assert "json" in described["response_instructions"]
    assert described["prompt_template_version"] == PROMPT_TEMPLATE_VERSION
    assert described["request"]["previous_attempt"]["feedback"]["success"] is False
    assert described["request"]["source_catalog"] == library()[0]
    expected = (
        [sheet["image"] for sheet in library()[2]]
        + [prior["observation"][camera] for camera in CAMERAS]
        + [current["observation"][camera] for camera in CAMERAS]
    )
    images = [row for row in content if row["type"] == "image_url"]
    assert len(images) == len(expected) == 6
    for row, array in zip(images, expected, strict=True):
        png = base64.b64decode(row["image_url"]["url"].split(",", 1)[1])
        np.testing.assert_array_equal(np.asarray(Image.open(io.BytesIO(png))), array)
    labels = " ".join(row["text"] for row in content[1:] if row["type"] == "text")
    assert "TRAINING DONOR PREVIEWS ONLY" in labels
    assert "COMPLETED previous attempt synthetic-baseline" in labels
    assert "CURRENT attempt synthetic-tli-vli-r1" in labels


def test_compact_wire_catalog_keeps_choices_and_preview_positions_without_mutation():
    request = request_for()
    frozen = copy.deepcopy(request)
    payload = build_payload(request, MODEL)
    wire = json.loads(payload["messages"][1]["content"][0]["text"])
    context = wire["request"]
    assert request == frozen
    assert context["request_fingerprint"] == frozen["request_fingerprint"]
    assert "Full validated catalog provenance" in wire["wire_projection"]
    assert context["source_catalog"] == frozen["source_catalog"]
    assert len(context["donor_catalog"]) == len(frozen["donor_catalog"])
    expected_fields = {
        "donor_id",
        "source_id",
        "episode_index",
        "frame_index",
        "phase",
        "preview_position",
    }
    for described, original in zip(
        context["donor_catalog"], frozen["donor_catalog"], strict=True
    ):
        assert set(described) == expected_fields
        assert described == {name: original[name] for name in expected_fields}
        source = next(
            row
            for row in context["source_catalog"]
            if row["source_id"] == described["source_id"]
        )
        assert source["prompt"] == original["prompt"]
        assert "sample_sha256" in original and "cameras" in original
    assert "library_id" not in json.dumps(context["donor_catalog"])
    allowed = wire["response_schema"]["properties"]["vision"]["anyOf"][1]["properties"][
        "donor_id"
    ]["enum"]
    assert allowed == [row["donor_id"] for row in frozen["donor_catalog"]]
    assert all(
        item["image_url"]["url"].startswith("data:image/png;base64,")
        for item in payload["messages"][1]["content"]
        if item["type"] == "image_url"
    )


def test_latest_two_slots_include_failure_and_proposals_cannot_cross_arms_or_attempts():
    first = request_for()
    second = request_for(
        decision_index=2,
        observation_step=25,
        observations=[snapshot(0), snapshot(25)],
        previous_decisions=[decision_row(first)],
    )
    third = request_for(
        decision_index=3,
        observation_step=50,
        observations=[snapshot(50)],
        previous_decisions=[decision_row(first), decision_row(second, accepted=False)],
    )
    fourth = request_for(
        decision_index=4,
        observation_step=75,
        observations=[snapshot(0), snapshot(25), snapshot(50), snapshot(75)],
        previous_decisions=[decision_row(second, accepted=False), decision_row(third)],
    )
    assert [row["decision_index"] for row in fourth["previous_decisions"]] == [2, 3]
    assert fourth["previous_decisions"][0]["proposal"] is None
    assert len(fourth["observations"]) == 4
    for field, value in (
        ("attempt_id", "other-attempt"),
        ("episode_id", "other-episode"),
        ("representation_mode", "tei"),
    ):
        bad = copy.deepcopy(fourth)
        bad["previous_decisions"][-1]["proposal"][field] = value
        with pytest.raises(
            ClientError, match="different episode, arm, attempt or step"
        ):
            build_payload(rehash(bad), MODEL)
    bad = copy.deepcopy(fourth)
    bad["previous_decisions"] = [decision_row(third)]
    with pytest.raises(ClientError, match="latest contiguous"):
        build_payload(rehash(bad), MODEL)


def test_failed_rollout_feedback_is_bound_to_latest_attempt_and_raw_time():
    first = request_for()
    outcome = feedback(first["attempt_id"], executed_actions=25)
    values = {
        "attempt_id": "synthetic-tli-vli-r2",
        "completed_rollout_feedback": [outcome],
        "previous_attempt": {
            "feedback": outcome,
            "decisions": [decision_row(first)],
            "snapshots": [snapshot(25)],
        },
    }
    request = request_for(**values)
    assert (
        request["previous_attempt"]["decisions"][0]["proposal"]["attempt_id"]
        == first["attempt_id"]
    )
    for mutation in ("future_frame", "other_feedback", "online_goal", "oracle_catalog"):
        bad = copy.deepcopy(request)
        if mutation == "future_frame":
            bad["previous_attempt"]["snapshots"][0]["step"] = 26
        elif mutation == "other_feedback":
            bad["previous_attempt"]["feedback"]["attempt_id"] = "another-arm"
        elif mutation == "online_goal":
            bad["observations"][0]["observation"]["success"] = True
        else:
            bad["source_catalog"][0]["oracle_target_mapping"] = "target"
        with pytest.raises(ClientError):
            build_payload(rehash(bad), MODEL)


def test_catalog_hash_frame_freshness_state_and_budget_tampering_are_rejected():
    for mutation in (
        "donor_pixels",
        "sheet_pixels",
        "source_prompt",
        "step",
        "state",
        "cap",
        "alignment",
        "fifth_frame",
    ):
        request = request_for()
        if mutation == "donor_pixels":
            request["donor_catalog"][0]["cameras"][CAMERAS[0]]["pixels_sha256"] = (
                "0" * 64
            )
        elif mutation == "sheet_pixels":
            request["contact_sheets"][0]["sha256"] = "0" * 64
        elif mutation == "source_prompt":
            request["source_catalog"][0]["prompt"] = "Different source task"
        elif mutation == "step":
            request["observations"][0]["step"] = 25
        elif mutation == "state":
            request["observations"][0]["observation"]["observation/state"] = [0] * 7
        elif mutation == "cap":
            request["limits"]["max_calls"] = 13
        elif mutation == "alignment":
            request["limits"]["execute_steps"] = 4
        else:
            request["observations"] *= 5
        with pytest.raises(ClientError):
            build_payload(rehash(request), MODEL)
    request = request_for()
    request["target_task"] = "Unbound change"
    with pytest.raises(ClientError, match="fingerprint"):
        build_payload(request, MODEL)


def test_client_single_call_keeps_exact_prompt_wire_and_local_identity(client_factory):
    request = request_for()
    client = client_factory(envelope=envelope_for(request))
    proposal = client.propose(request)
    assert len(client.opener.calls) == len(client.records) == 1
    http_request, timeout = client.opener.calls[0]
    assert timeout == 170
    record = client.records[0]
    assert record["payload_sha256"] == hashlib.sha256(http_request.data).hexdigest()
    assert (
        record["system_prompt_sha256"]
        == hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()
    )
    assert record["request_fingerprint"] == proposal["request_fingerprint"]
    assert record["source_catalog_sha256"] == digest(request["source_catalog"])
    assert record["donor_catalog_sha256"] == digest(request["donor_catalog"])
    assert record["accepted"] is True and record["provider_call"] is True
    assert record["client_schema_version"] == SCHEMA_VERSION
    assert TEST_KEY not in client.response_log.read_text()
    assert "Authorization" not in client.response_log.read_text()
    summary = summarize_calls(client.records)
    assert summary["provider_calls"] == summary["accepted_proposals"] == 1
    assert summary["tokens"]["total_tokens"]["sum"] == 150
    assert summary["tokens"]["reasoning_tokens"]["sum"] == 20
    assert summary["reasoning_tokens_are_subset_of_output_tokens"]
    assert summarize_calls(client.response_log) == summary


@pytest.mark.parametrize("kind", ("stale", "model", "truncated", "bad_envelope"))
def test_usage_from_rejected_completions_is_retained_without_retry(
    client_factory, kind
):
    request = request_for()
    envelope = envelope_for(request)
    if kind == "stale":
        envelope["choices"][0]["message"]["content"] = json.dumps(
            proposal_for(request, request_id="0" * 16)
        )
    elif kind == "model":
        envelope["model"] = "different-model"
    elif kind == "truncated":
        envelope["choices"][0]["finish_reason"] = "length"
    else:
        envelope["choices"] = []
    client = client_factory(envelope=envelope)
    with pytest.raises(ClientError):
        client.propose(request)
    assert len(client.opener.calls) == len(client.records) == 1
    summary = summarize_calls(client.records)
    assert summary["accepted_proposals"] == 0
    assert summary["failed_calls"] == 1
    assert summary["tokens"]["total_tokens"]["sum"] == 150


def test_http_error_records_available_usage_and_redacts_error_without_retry(
    client_factory,
):
    request = request_for()
    error_body = {
        "model": MODEL,
        "usage": envelope_for(request)["usage"],
        "error": {
            "code": "synthetic",
            "type": "unavailable",
            "message": f"Do not disclose {TEST_KEY}",
        },
    }
    error = urllib.error.HTTPError(
        "https://synthetic.invalid",
        503,
        "unavailable",
        {},
        io.BytesIO(json.dumps(error_body).encode()),
    )
    client = client_factory(error=error)
    with pytest.raises(ClientError, match="HTTP 503"):
        client.propose(request)
    assert len(client.opener.calls) == 1
    assert TEST_KEY not in client.response_log.read_text()
    summary = summarize_calls(client.records)
    assert summary["provider_calls"] == 1
    assert summary["tokens"]["total_tokens"]["sum"] == 150
    assert summary["http_status"] == {"503": 1}


def test_transport_and_preflight_do_not_invent_usage_or_actions(
    client_factory, monkeypatch
):
    request = request_for()
    client = client_factory(error=TimeoutError("synthetic"))
    with pytest.raises(ClientError, match="transport"):
        client.propose(request)
    assert len(client.opener.calls) == 1
    monkeypatch.delenv("NVIDIA_INFERENCE_API_KEY")
    with pytest.raises(ClientError, match="not set"):
        client.propose(request)
    assert len(client.opener.calls) == 1
    assert len(client.records) == 2
    summary = summarize_calls(client.records)
    assert summary["client_attempts"] == 2
    assert summary["provider_calls"] == 1
    assert summary["preflight_failures"] == 1
    assert summary["usage_unavailable_calls"] == 1
    assert summary["tokens"]["total_tokens"]["complete"] is False
    assert summary["monetary_cost"] is None
    assert all(not record["accepted"] for record in client.records)


def test_invalid_request_preflight_is_ledgered_and_sampling_conflicts_are_explicit(
    client_factory,
):
    request = request_for()
    request["request_id"] = "stale"
    client = client_factory()
    with pytest.raises(ClientError):
        client.propose(request)
    assert len(client.opener.calls) == 0
    assert summarize_calls(client.records)["preflight_failures"] == 1
    with pytest.raises(ClientError, match="conflicts"):
        client_factory(sampling={"reasoning_effort": "low"})
    with pytest.raises(ClientError, match="incompatible"):
        summarize_calls([{**client.records[0], "client_schema_version": "old"}])
