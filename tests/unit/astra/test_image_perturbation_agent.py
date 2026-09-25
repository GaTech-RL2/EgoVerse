"""Synthetic CPU contract/transport tests; no model calls or success evidence."""

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
from astra_reversal.image_perturbation_agent import (
    CONTACT_SHEET_LAYOUT,
    PROMPT_TEMPLATE_VERSION,
    SCHEMA_VERSION,
    SYSTEM_PROMPT,
    ImagePerturbationClient,
    build_payload,
    build_request,
    parse_proposal,
    response_schema,
    summarize_calls,
)
from astra_reversal.records import digest

MODEL = "synthetic-provider/image-perturbation-test"
TEST_KEY = "synthetic-image-perturbation-key-not-a-real-credential"


def snapshot(step=0, *, label="current raw observation", value=17):
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
def synthetic_library():
    library_id = "a" * 64
    rows = []
    for index in range(2):
        row = {
            "donor_id": f"std10-e379-f{index}",
            "library_id": library_id,
            "source_id": "10",
            "prompt": "Pick up the synthetic cup.",
            "episode_index": 379,
            "frame_index": index,
            "phase": {"numerator": index, "denominator": 4},
            "preview_position": {"row": 0, "column": index},
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
                "library_id": library_id,
            }
        )
    return rows, sheets


def request_for(**overrides):
    catalog, sheets = synthetic_library()
    values = {
        "episode_id": "synthetic-episode",
        "attempt_id": "synthetic-occlusion-1",
        "decision_index": 1,
        "observation_step": 0,
        "image_mode": "occlusion",
        "target_task": "Put the cup in the bowl.",
        "donor_catalog": catalog,
        "contact_sheets": sheets,
        "observations": [snapshot()],
    }
    return build_request(**(values | overrides))


def occlusion(**overrides):
    return {
        "kind": "occlusion",
        "camera": CAMERAS[0],
        "box_xyxy": [0, 0, 112, 224],
        "fill_rgb": [127, 127, 127],
        "strength": 0.4,
        **overrides,
    }


def blend(**overrides):
    return {
        "kind": "demo_blend",
        "camera": CAMERAS[0],
        "donor_id": "std10-e379-f1",
        "alpha": 0.15,
        **overrides,
    }


def proposal_for(request, **overrides):
    return {
        **{
            field: request[field]
            for field in (
                "schema_version",
                "episode_id",
                "attempt_id",
                "decision_index",
                "observation_step",
                "image_mode",
                "request_id",
            )
        },
        "decision_id": f"{request['attempt_id']}-d{request['decision_index']}",
        "observed_phase": "cup remains ungrasped",
        "rationale": "Use the current raw scene to reconsider the distraction region.",
        "image_perturbations": [occlusion()]
        if request["image_mode"] == "occlusion"
        else [blend()],
        **overrides,
    }


def decision_row(request, *, accepted=True, **overrides):
    return {
        "decision_index": request["decision_index"],
        "observation_step": request["observation_step"],
        "accepted": accepted,
        "error": None if accepted else "Inference endpoint returned HTTP 503",
        "proposal": parse_proposal(proposal_for(request, **overrides), request)
        if accepted
        else None,
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


def envelope_for(request, *, usage=True, **overrides):
    envelope = {
        "id": "synthetic-image-completion",
        "model": MODEL,
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": json.dumps(proposal_for(request, **overrides)),
                },
            }
        ],
    }
    if usage:
        envelope["usage"] = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "completion_tokens_details": {"reasoning_tokens": 20},
        }
    return envelope


class FakeOpener:
    def __init__(self, *, envelope=None, error=None):
        self.envelope, self.error, self.calls = envelope, error, []

    def open(self, request, timeout):
        self.calls.append((request, timeout))
        if self.error is not None:
            raise self.error
        result = io.BytesIO(json.dumps(self.envelope).encode())
        result.status = 200
        return result


@pytest.fixture
def client_factory(monkeypatch, tmp_path):
    monkeypatch.setenv("NVIDIA_INFERENCE_API_KEY", TEST_KEY)

    def make(*, envelope=None, error=None, **kwargs):
        client = ImagePerturbationClient(
            model=MODEL, response_log=tmp_path / "provider.jsonl", **kwargs
        )
        client.opener = FakeOpener(envelope=envelope, error=error)
        return client

    return make


def test_request_payload_preserves_pixels_and_separates_donors_current_and_prior():
    first = request_for()
    feedback = {
        "attempt_id": first["attempt_id"],
        "success": False,
        "executed_actions": 300,
        "termination": "action_budget",
        "error": None,
    }
    prior = snapshot(300, label="prior final raw frame", value=71)
    current = snapshot(value=91)
    request = request_for(
        attempt_id="synthetic-occlusion-2",
        completed_rollout_feedback=[feedback],
        previous_attempt={
            "feedback": feedback,
            "decisions": [decision_row(first)],
            "snapshots": [prior],
        },
        observations=[current],
    )
    payload = build_payload(request, MODEL)
    content = payload["messages"][1]["content"]
    text = json.loads(content[0]["text"])
    context = text["request"]
    assert "json" in text["response_instructions"]
    assert text["prompt_template_version"] == PROMPT_TEMPLATE_VERSION
    assert payload["cache"] == {"no-cache": True}
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["reasoning_effort"] == "medium"
    assert payload["max_completion_tokens"] == 8192
    assert context["donor_catalog"] == request["donor_catalog"]
    images = [item for item in content if item["type"] == "image_url"]
    expected = [sheet["image"] for sheet in synthetic_library()[1]]
    expected += [prior["observation"][camera] for camera in CAMERAS]
    expected += [current["observation"][camera] for camera in CAMERAS]
    assert len(images) == len(expected) == 6
    for item, pixels in zip(images, expected, strict=True):
        data = base64.b64decode(item["image_url"]["url"].split(",", 1)[1])
        np.testing.assert_array_equal(np.asarray(Image.open(io.BytesIO(data))), pixels)
    assert "TRAINING DONOR PREVIEWS" in content[1]["text"]
    assert "COMPLETED previous attempt" in content[5]["text"]
    assert "CURRENT attempt" in content[9]["text"]
    assert context["observations"][0]["observation"][CAMERAS[0]]["image_index"] == 4
    assert (
        context["observations"][0]["observation"]["observation/state"]
        == current["observation"]["observation/state"].tolist()
    )
    assert "five native policy chunks" in SYSTEM_PROMPT
    assert "clears ALL prior edits" in SYSTEM_PROMPT
    assert "fixed" in SYSTEM_PROMPT and "noise remain fixed" in SYSTEM_PROMPT


def test_short_wire_identity_is_exact_and_full_hash_attached_only_locally():
    request = request_for()
    wire = proposal_for(request)
    proposal = parse_proposal(json.dumps(wire), request)
    assert proposal == {**wire, "request_fingerprint": request["request_fingerprint"]}
    schema = response_schema(request)
    assert "request_fingerprint" not in schema["properties"]
    assert schema["properties"]["request_id"] == {
        "const": request["request_fingerprint"][:16]
    }
    with pytest.raises(ClientError, match="exactly"):
        parse_proposal(proposal, request)
    changed_pixels = request_for(observations=[snapshot(value=19)])
    with pytest.raises(ClientError, match="request_id"):
        parse_proposal(wire, changed_pixels)


@pytest.mark.parametrize(
    "mode,operations",
    [
        ("occlusion", []),
        ("demo_blend", []),
        ("occlusion", [occlusion(strength=0)]),
        ("demo_blend", [blend(alpha=1), blend(camera=CAMERAS[1], alpha=0)]),
    ],
)
def test_valid_noop_and_full_rgb_operators_preserve_absolute_parameters(
    mode, operations
):
    request = request_for(image_mode=mode)
    assert (
        parse_proposal(proposal_for(request, image_perturbations=operations), request)[
            "image_perturbations"
        ]
        == operations
    )


@pytest.mark.parametrize(
    "mode,operations",
    [
        ("demo_blend", [blend(donor_id="not-in-library")]),
        ("demo_blend", [blend(alpha=True)]),
        ("demo_blend", [blend(alpha=1.001)]),
        ("demo_blend", [occlusion()]),
        ("occlusion", [blend()]),
        ("occlusion", [occlusion(strength="0.2")]),
        ("occlusion", [occlusion(box_xyxy=[0, 0, 113, 224])]),
        ("occlusion", [occlusion(box_xyxy=[0, 0, 0, 224])]),
        ("occlusion", [occlusion(box_xyxy=[0, 0, 20.0, 224])]),
        ("occlusion", [occlusion(box_xyxy=[0, 0, 20, 225])]),
        ("occlusion", [occlusion(fill_rgb=[128, 128, 128])]),
        ("occlusion", [occlusion(), occlusion()]),
        (
            "occlusion",
            [
                {
                    "kind": "box",
                    "camera": CAMERAS[0],
                    "coordinates": [0, 0, 20, 20],
                    "gain": 0.5,
                }
            ],
        ),
    ],
)
def test_rejects_cross_arm_annotations_unknown_donors_and_unsafe_geometry(
    mode, operations
):
    request = request_for(image_mode=mode)
    with pytest.raises(ClientError, match="Invalid image_perturbations"):
        parse_proposal(proposal_for(request, image_perturbations=operations), request)


@pytest.mark.parametrize(
    "field,value",
    [
        ("episode_id", "another-case"),
        ("attempt_id", "another-attempt"),
        ("decision_index", True),
        ("observation_step", 25),
        ("image_mode", "demo_blend"),
        ("request_id", "0" * 16),
        ("request_id", "0" * 64),
        ("schema_version", "1.0"),
    ],
)
def test_rejects_stale_response_identities(field, value):
    request = request_for()
    with pytest.raises(ClientError):
        parse_proposal(proposal_for(request, **{field: value}), request)


def test_history_keeps_failed_slots_and_does_not_carry_an_active_edit():
    first = request_for()
    accepted = decision_row(first)
    second = request_for(
        decision_index=2,
        observation_step=25,
        observations=[snapshot(25)],
        previous_decisions=[accepted],
    )
    failed = decision_row(second, accepted=False)
    third = request_for(
        decision_index=3,
        observation_step=50,
        observations=[snapshot(25), snapshot(50)],
        previous_decisions=[accepted, failed],
    )
    assert third["previous_decisions"][-1]["error"] == failed["error"]
    assert third["limits"]["failure_policy"] == "clear_to_raw"
    assert not any(key.startswith("active_") for key in third)
    revised = proposal_for(third, image_perturbations=[occlusion(strength=0.1)])
    assert parse_proposal(revised, third)["image_perturbations"][0]["strength"] == 0.1
    with pytest.raises(ClientError, match="fresh decision_id"):
        parse_proposal(
            proposal_for(third, decision_id=accepted["proposal"]["decision_id"]), third
        )
    with pytest.raises(ClientError, match="every previous"):
        request_for(
            decision_index=3,
            observation_step=50,
            observations=[snapshot(50)],
            previous_decisions=[accepted],
        )
    stale = copy.deepcopy(accepted)
    stale["proposal"]["image_mode"] = "demo_blend"
    with pytest.raises(ClientError, match="image_mode"):
        request_for(
            decision_index=2,
            observation_step=25,
            observations=[snapshot(25)],
            previous_decisions=[stale],
        )


def test_completed_attempt_feedback_and_current_frame_freshness_are_strict():
    first = request_for()
    row = decision_row(first)
    feedback = {
        "attempt_id": first["attempt_id"],
        "success": False,
        "executed_actions": 25,
        "termination": "action_budget",
        "error": None,
    }
    args = {
        "attempt_id": "synthetic-occlusion-2",
        "completed_rollout_feedback": [feedback],
        "previous_attempt": {
            "feedback": feedback,
            "decisions": [row],
            "snapshots": [snapshot(25)],
        },
    }
    request_for(**args)
    bad = copy.deepcopy(args)
    bad["previous_attempt"]["decisions"][0]["proposal"]["attempt_id"] = "other-arm"
    with pytest.raises(ClientError, match="attempt_id"):
        request_for(**bad)
    bad = copy.deepcopy(args)
    bad["previous_attempt"]["snapshots"][0]["step"] = 26
    with pytest.raises(ClientError, match="later than"):
        request_for(**bad)
    with pytest.raises(ClientError, match="latest completed"):
        request_for(**(args | {"completed_rollout_feedback": []}))
    with pytest.raises(ClientError, match="current observation_step"):
        request_for(observations=[snapshot(1)])


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("extra_catalog_field", "exactly"),
        ("sample_hash", "sample digest"),
        ("camera_shape", "same raw camera"),
        ("missing_camera", "exactly"),
        ("sheet_pixels", "pixels disagree"),
        ("sheet_layout", "fixed 9x5"),
        ("sheet_library", "identities differ"),
        ("duplicate_donor", "unique"),
        ("fingerprint", "fingerprint"),
        ("limits", "limits were altered"),
    ],
)
def test_request_catalog_sheet_and_provenance_tampering_is_rejected(mutation, match):
    request = request_for()
    if mutation == "extra_catalog_field":
        request["donor_catalog"][0]["oracle_goal_pair"] = ["10", "13"]
    elif mutation == "sample_hash":
        request["donor_catalog"][0]["prompt"] = "Changed donor prompt."
    elif mutation == "camera_shape":
        request["donor_catalog"][0]["cameras"][CAMERAS[0]]["shape"] = [112, 112, 3]
    elif mutation == "missing_camera":
        del request["donor_catalog"][0]["cameras"][CAMERAS[1]]
    elif mutation == "sheet_pixels":
        request["contact_sheets"][0]["sha256"] = "0" * 64
    elif mutation == "sheet_layout":
        request["contact_sheets"][0]["layout"]["oracle_pair"] = [1, 2]
    elif mutation == "sheet_library":
        request["contact_sheets"][0]["library_id"] = "b" * 64
    elif mutation == "duplicate_donor":
        request["donor_catalog"].append(copy.deepcopy(request["donor_catalog"][0]))
    elif mutation == "limits":
        request["limits"]["max_occlusion_fraction"] = 0.75
    else:
        request["target_task"] = "Altered target task."
    if mutation != "fingerprint":
        rehash(request)
    with pytest.raises(ClientError, match=match):
        build_payload(request, MODEL)


def test_duplicate_json_and_unlisted_language_fields_are_rejected():
    request = request_for()
    duplicate = json.dumps(proposal_for(request))[:-1] + ',"image_perturbations":[]}'
    with pytest.raises(ClientError, match="duplicate"):
        parse_proposal(duplicate, request)
    with pytest.raises(ClientError, match="exactly"):
        parse_proposal(proposal_for(request, language={"target": "New task"}), request)


def test_one_physical_call_exact_wire_binding_and_reasoning_subset_ledger(
    client_factory,
):
    request = request_for()
    client = client_factory(envelope=envelope_for(request))
    proposal = client.propose(request)
    assert proposal["request_fingerprint"] == request["request_fingerprint"]
    assert len(client.opener.calls) == 1
    http_request, _ = client.opener.calls[0]
    row = json.loads(client.response_log.read_text())
    assert row["payload_sha256"] == hashlib.sha256(http_request.data).hexdigest()
    assert row["request_fingerprint"] == request["request_fingerprint"]
    assert row["donor_catalog_sha256"] == digest(request["donor_catalog"])
    assert row["contact_sheet_sha256"] == {
        sheet["camera"]: sheet["sha256"] for sheet in request["contact_sheets"]
    }
    summary = summarize_calls(client.response_log)
    assert (
        summary["client_attempts"]
        == summary["provider_calls"]
        == summary["accepted_proposals"]
        == 1
    )
    assert summary["tokens"]["total_tokens"]["sum"] == 150
    assert summary["tokens"]["reasoning_tokens"]["sum"] == 20
    assert summary["reasoning_tokens_are_subset_of_output_tokens"]
    assert summary["actual_models"] == {MODEL: 1}
    assert summary["monetary_cost"] is None
    assert TEST_KEY not in client.response_log.read_text()


@pytest.mark.parametrize(
    "mutation", ["model", "stale", "truncated", "refusal", "wrong_operator"]
)
def test_paid_rejections_keep_usage_and_never_retry(client_factory, mutation):
    request = request_for()
    envelope = envelope_for(request)
    choice = envelope["choices"][0]
    if mutation == "model":
        envelope["model"] = "synthetic-other-provider"
    elif mutation == "stale":
        choice["message"]["content"] = json.dumps(
            proposal_for(request, request_id="0" * 16)
        )
    elif mutation == "truncated":
        choice["finish_reason"] = "length"
    elif mutation == "refusal":
        choice["message"]["refusal"] = "Synthetic refusal"
    else:
        choice["message"]["content"] = json.dumps(
            proposal_for(request, image_perturbations=[blend()])
        )
    client = client_factory(envelope=envelope)
    with pytest.raises(ClientError):
        client.propose(request)
    assert len(client.opener.calls) == 1
    summary = summarize_calls(client.response_log)
    assert summary["failed_calls"] == 1 and summary["accepted_proposals"] == 0
    assert summary["tokens"]["total_tokens"]["sum"] == 150


def test_http_failure_retains_usage_and_allowlisted_sanitized_error(client_factory):
    request = request_for()
    body = {
        "model": MODEL,
        "usage": envelope_for(request)["usage"],
        "error": {
            "code": "synthetic_error",
            "type": "invalid_request",
            "message": TEST_KEY + " failure",
            "internal_credentials": TEST_KEY,
        },
        "headers": {"Authorization": TEST_KEY},
    }
    error = urllib.error.HTTPError(
        "https://unused.invalid",
        400,
        "synthetic",
        {},
        io.BytesIO(json.dumps(body).encode()),
    )
    client = client_factory(error=error)
    with pytest.raises(ClientError, match="HTTP 400"):
        client.propose(request)
    assert len(client.opener.calls) == 1
    text = client.response_log.read_text()
    assert (
        TEST_KEY not in text
        and "internal_credentials" not in text
        and "headers" not in text
    )
    row = json.loads(text)
    assert row["error_kind"] == "http_error" and row["http_status"] == 400
    assert set(row["provider_error"]) == {"code", "type", "message"}
    assert summarize_calls(client.response_log)["tokens"]["total_tokens"]["sum"] == 150


def test_unknown_transport_usage_is_missing_not_zero_and_preflight_is_separate(
    client_factory, monkeypatch
):
    request = request_for()
    client = client_factory(error=urllib.error.URLError("synthetic no response"))
    with pytest.raises(ClientError, match="transport"):
        client.propose(request)
    monkeypatch.delenv("NVIDIA_INFERENCE_API_KEY")
    with pytest.raises(ClientError, match="not set"):
        client.propose(request)
    assert len(client.opener.calls) == 1
    summary = summarize_calls(client.response_log)
    assert summary["client_attempts"] == 2
    assert summary["provider_calls"] == summary["preflight_failures"] == 1
    assert summary["usage_unavailable_calls"] == 1
    assert summary["tokens"]["total_tokens"] == {
        "sum": 0,
        "available_calls": 0,
        "missing_calls": 1,
        "complete": False,
    }


def test_malformed_request_preflight_is_recorded_without_call_or_fallback(
    client_factory,
):
    client = client_factory()
    with pytest.raises(ClientError):
        client.propose({"episode_id": "malformed"})
    assert client.opener.calls == []
    summary = summarize_calls(client.response_log)
    assert summary["client_attempts"] == summary["preflight_failures"] == 1
    assert summary["provider_calls"] == summary["accepted_proposals"] == 0


def test_ledger_rejects_tampered_usage_and_does_not_change_original_record(
    client_factory,
):
    request = request_for()
    client = client_factory(envelope=envelope_for(request))
    client.propose(request)
    row = json.loads(client.response_log.read_text())
    summarize_calls([row])
    assert row["client_schema_version"] == SCHEMA_VERSION
    row["token_usage"]["total_tokens"] += 1
    with pytest.raises(ClientError, match="disagrees"):
        summarize_calls([row])
