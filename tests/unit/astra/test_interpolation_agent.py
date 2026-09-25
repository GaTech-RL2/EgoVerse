"""CPU-only interpolation contract tests; synthetic replies are not model evidence."""

import base64
import copy
import io
import json
import urllib.error

import numpy as np
import pytest
from PIL import Image

from astra_reversal.agent import CAMERAS
from astra_reversal.astra_client import ClientError
from astra_reversal.interpolation_agent import (
    PROMPT_TEMPLATE_VERSION,
    SCHEMA_VERSION,
    SYSTEM_PROMPT,
    InterpolationClient,
    build_payload,
    build_request,
    parse_proposal,
    response_schema,
    summarize_calls,
)
from astra_reversal.records import digest

from .conftest import SyntheticEnvironment

MODEL = "synthetic-provider/interpolation-test"
TEST_KEY = "synthetic-interpolation-key-not-a-real-credential"


def snapshot(step=0, *, label="current raw observation", value=17):
    observation = SyntheticEnvironment().observe()
    for index, camera in enumerate(CAMERAS):
        observation[camera][:] = value + index
    return {"label": label, "step": step, "observation": observation}


def request_for(**overrides):
    args = {
        "episode_id": "synthetic-episode",
        "attempt_id": "synthetic-attempt-1",
        "decision_index": 1,
        "observation_step": 0,
        "interpolation_mode": "tei",
        "target_task": "Place the cup in the bowl.",
        "source_catalog": [
            {"source_id": "10", "prompt": "Pick up the cup."},
            {"source_id": "13", "prompt": "Place the cup in the bowl."},
        ],
        "observations": [snapshot()],
    }
    return build_request(**(args | overrides))


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
                "interpolation_mode",
                "request_fingerprint",
            )
        },
        "decision_id": f"{request['attempt_id']}-decision-{request['decision_index']}",
        "source_a_id": "10",
        "source_b_id": "13",
        "alpha": 0.75,
        "observed_phase": "repositioning the cup",
        "rationale": "The current image suggests the object still needs alignment.",
        "vision": [],
        **overrides,
    }


def decision_row(request, *, accepted=True, **proposal_overrides):
    return {
        "decision_index": request["decision_index"],
        "observation_step": request["observation_step"],
        "accepted": accepted,
        "proposal": proposal_for(request, **proposal_overrides) if accepted else None,
        "error": None if accepted else "Inference endpoint returned HTTP 503",
    }


def active(row):
    return {
        name: row["proposal"][name] for name in ("source_a_id", "source_b_id", "alpha")
    }


def rehash(request):
    request["request_fingerprint"] = digest(
        {k: v for k, v in request.items() if k != "request_fingerprint"}
    )
    return request


def envelope_for(request, *, usage=True, **proposal_overrides):
    envelope = {
        "id": "synthetic-completion",
        "model": MODEL,
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": json.dumps(proposal_for(request, **proposal_overrides)),
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
        client = InterpolationClient(
            model=MODEL, response_log=tmp_path / "provider.jsonl", **kwargs
        )
        client.opener = FakeOpener(envelope=envelope, error=error)
        return client

    return make


def test_payload_preserves_pixels_state_catalog_and_explicit_json_instruction(spec):
    raw = snapshot()
    request = request_for(observations=[raw], action_spec=spec.as_dict())
    payload = build_payload(
        request,
        MODEL,
        sampling={"reasoning_effort": "medium", "max_completion_tokens": 4096},
    )
    user = payload["messages"][1]["content"]
    text = json.loads(user[0]["text"])
    assert "json" in text["response_instructions"]
    assert text["prompt_template_version"] == PROMPT_TEMPLATE_VERSION
    assert text["request"]["source_catalog"] == request["source_catalog"]
    assert text["request"]["action_spec"] == json.loads(json.dumps(spec.as_dict()))
    assert payload["reasoning_effort"] == "medium"
    assert payload["cache"] == {"no-cache": True}
    assert payload["response_format"] == {"type": "json_object"}
    images = [row for row in user if row["type"] == "image_url"]
    assert len(images) == 2
    for camera, row in zip(CAMERAS, images, strict=True):
        data = base64.b64decode(row["image_url"]["url"].split(",", 1)[1])
        np.testing.assert_array_equal(
            np.asarray(Image.open(io.BytesIO(data))), raw["observation"][camera]
        )
    assert (
        text["request"]["observations"][0]["observation"]["observation/state"]
        == raw["observation"]["observation/state"].tolist()
    )
    assert "(1-2alpha)*(T_A-T_B)" in SYSTEM_PROMPT
    assert "five-action native policy chunk" in SYSTEM_PROMPT


@pytest.mark.parametrize("mode", ["tei", "tli", "tei_tli"])
def test_operator_and_absolute_alpha_can_move_backward_after_failure(mode):
    first = request_for(interpolation_mode=mode)
    row = decision_row(first, alpha=0.9)
    second = request_for(
        interpolation_mode=mode,
        decision_index=2,
        observation_step=25,
        observations=[snapshot(25)],
        previous_decisions=[row],
        active_interpolation=active(row),
    )
    proposal = proposal_for(second, alpha=0.1)
    assert parse_proposal(proposal, second) == proposal
    neutral = proposal_for(second, source_a_id="10", source_b_id="10", alpha=0.5)
    assert parse_proposal(neutral, second) == neutral
    assert response_schema(second)["properties"]["source_a_id"] == {
        "enum": ["10", "13"]
    }


def test_failure_history_preserves_text_and_reaches_next_scheduled_slot():
    first = request_for(vision_enabled=True)
    accepted = decision_row(
        first,
        vision=[
            {"camera": CAMERAS[0], "kind": "point", "coordinates": [3, 4], "gain": 0.6}
        ],
    )
    second = request_for(
        vision_enabled=True,
        decision_index=2,
        observation_step=25,
        observations=[snapshot(25)],
        previous_decisions=[accepted],
        active_interpolation=active(accepted),
    )
    failed = decision_row(second, accepted=False)
    third = request_for(
        vision_enabled=True,
        decision_index=3,
        observation_step=50,
        observations=[snapshot(25), snapshot(50)],
        previous_decisions=[accepted, failed],
        active_interpolation=active(accepted),
    )
    assert third["previous_decisions"][1]["error"] == failed["error"]
    assert third["active_interpolation"] == active(accepted)
    assert parse_proposal(proposal_for(third), third)["vision"] == []
    altered = copy.deepcopy(third)
    altered["active_interpolation"]["alpha"] = 0.5
    with pytest.raises(ClientError, match="last accepted"):
        build_payload(rehash(altered), MODEL)


def test_prior_attempt_frames_are_separate_and_fingerprinted():
    first = request_for()
    feedback = {
        "attempt_id": first["attempt_id"],
        "success": False,
        "executed_actions": 300,
        "termination": "action_budget",
        "error": None,
        "visual_summary": "Object remained beside bowl.",
    }
    previous = {
        "feedback": feedback,
        "decisions": [decision_row(first)],
        "snapshots": [
            snapshot(0, label="prior start", value=31),
            snapshot(300, label="prior end", value=33),
        ],
    }
    request = request_for(
        attempt_id="synthetic-attempt-2",
        previous_attempt=previous,
        completed_rollout_feedback=[feedback],
    )
    payload = build_payload(request, MODEL)
    content = payload["messages"][1]["content"]
    context = json.loads(content[0]["text"])["request"]
    assert len([row for row in content if row["type"] == "image_url"]) == 6
    assert context["observations"][0]["step"] == 0
    assert context["observations"][0]["observation"][CAMERAS[0]]["image_index"] == 4
    assert context["previous_attempt"]["snapshots"][-1]["step"] == 300
    assert "Completed previous attempt" in content[1]["text"]
    assert "CURRENT attempt" in content[9]["text"]
    altered = copy.deepcopy(request)
    altered["previous_attempt"]["feedback"]["visual_summary"] = "Changed summary"
    altered["completed_rollout_feedback"][0]["visual_summary"] = "Changed summary"
    with pytest.raises(ClientError, match="fingerprint"):
        build_payload(altered, MODEL)
    with pytest.raises(ClientError, match="attempt_id"):
        parse_proposal(proposal_for(first), request)


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"max_calls": 13}, "max_calls"),
        ({"action_budget": 301}, "action_budget"),
        ({"call_interval": 26}, "call_interval"),
        ({"decision_index": 0}, "decision_index"),
        ({"observation_step": 1}, "schedule"),
        ({"vision_enabled": 1}, "boolean"),
        ({"interpolation_mode": "TEI"}, "interpolation_mode"),
        (
            {
                "source_catalog": [
                    {"source_id": "10", "prompt": "A"},
                    {"source_id": "10", "prompt": "B"},
                ]
            },
            "unique",
        ),
        (
            {
                "source_catalog": [
                    {"source_id": "10", "prompt": "A", "oracle": True},
                    {"source_id": "13", "prompt": "B"},
                ]
            },
            "exactly",
        ),
        ({"observations": [snapshot(1)]}, "current observation_step"),
        ({"observations": [snapshot(), snapshot()]}, "increasing"),
    ],
)
def test_request_rejects_protocol_drift(overrides, match):
    with pytest.raises(ClientError, match=match):
        request_for(**overrides)


def test_lowered_development_limits_and_missing_history():
    request = request_for(call_interval=5, max_calls=2, action_budget=10)
    assert request["limits"]["call_interval"] == 5
    with pytest.raises(ClientError, match="every previous decision slot"):
        request_for(decision_index=2, observation_step=25, observations=[snapshot(25)])
    row = decision_row(request, accepted=False)
    next_request = request_for(
        call_interval=5,
        max_calls=2,
        action_budget=10,
        decision_index=2,
        observation_step=5,
        observations=[snapshot(5)],
        previous_decisions=[row],
    )
    assert next_request["active_interpolation"] is None
    row["error"] = None
    with pytest.raises(ClientError, match="previous call error"):
        request_for(
            call_interval=5,
            max_calls=2,
            action_budget=10,
            decision_index=2,
            observation_step=5,
            observations=[snapshot(5)],
            previous_decisions=[row],
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("episode_id", "another-episode"),
        ("attempt_id", "another-attempt"),
        ("decision_index", 2),
        ("decision_index", True),
        ("observation_step", 25),
        ("interpolation_mode", "tli"),
        ("request_fingerprint", "0" * 64),
        ("source_a_id", "unknown"),
        ("source_b_id", 13),
        ("alpha", True),
        ("alpha", -0.1),
        ("alpha", 1.01),
        ("alpha", float("nan")),
    ],
)
def test_stale_and_invalid_proposals_are_rejected_without_repair(field, value):
    request = request_for()
    proposal = proposal_for(request, **{field: value})
    with pytest.raises(ClientError):
        parse_proposal(proposal, request)


def test_extra_fields_duplicate_json_and_reused_decision_id_rejected():
    request = request_for()
    with pytest.raises(ClientError, match="exactly"):
        parse_proposal(proposal_for(request, success=True), request)
    duplicate = json.dumps(proposal_for(request))[:-1] + ',"alpha":0.2}'
    with pytest.raises(ClientError):
        parse_proposal(duplicate, request)
    row = decision_row(request)
    next_request = request_for(
        decision_index=2,
        observation_step=25,
        observations=[snapshot(25)],
        previous_decisions=[row],
        active_interpolation=active(row),
    )
    with pytest.raises(ClientError, match="fresh decision_id"):
        parse_proposal(
            proposal_for(next_request, decision_id=row["proposal"]["decision_id"]),
            next_request,
        )


@pytest.mark.parametrize(
    "annotation",
    [
        {"camera": CAMERAS[0], "kind": "point", "coordinates": [16, 3], "gain": 0.3},
        {"camera": CAMERAS[0], "kind": "box", "coordinates": [7, 1, 3, 8], "gain": 0.3},
        {"camera": CAMERAS[0], "kind": "point", "coordinates": [2, 3], "gain": 1.1},
        {"camera": "future-frame", "kind": "point", "coordinates": [2, 3], "gain": 0.3},
    ],
)
def test_vision_invalid_geometry_rejected(annotation):
    request = request_for(vision_enabled=True)
    with pytest.raises(ClientError):
        parse_proposal(proposal_for(request, vision=[annotation]), request)


def test_vision_is_optional_fresh_bounded_and_disabled_when_requested():
    annotation = {
        "camera": CAMERAS[0],
        "kind": "box",
        "coordinates": [0, 0, 15, 15],
        "gain": 0.3,
    }
    request = request_for(vision_enabled=True)
    assert parse_proposal(proposal_for(request, vision=[annotation]), request)[
        "vision"
    ] == [annotation]
    assert parse_proposal(proposal_for(request), request)["vision"] == []
    with pytest.raises(ClientError, match="four annotations"):
        parse_proposal(proposal_for(request, vision=[annotation] * 5), request)
    disabled = request_for()
    with pytest.raises(ClientError, match="disabled"):
        parse_proposal(proposal_for(disabled, vision=[annotation]), disabled)


def test_exact_model_one_physical_call_and_reasoning_subset_accounting(client_factory):
    request = request_for()
    client = client_factory(envelope=envelope_for(request))
    assert client.propose(request) == proposal_for(request)
    assert len(client.opener.calls) == 1
    summary = summarize_calls(client.response_log)
    assert summary["client_attempts"] == summary["provider_calls"] == 1
    assert summary["accepted_proposals"] == 1
    assert summary["tokens"]["total_tokens"]["sum"] == 150
    assert summary["tokens"]["reasoning_tokens"]["sum"] == 20
    assert summary["reasoning_tokens_are_subset_of_output_tokens"]
    assert summary["actual_models"] == {MODEL: 1}
    assert summary["monetary_cost"] is None
    assert TEST_KEY not in client.response_log.read_text()


@pytest.mark.parametrize("mutation", ["model", "stale", "truncated", "refusal"])
def test_rejected_provider_responses_keep_billed_usage_and_do_not_retry(
    client_factory, mutation
):
    request = request_for()
    envelope = envelope_for(request)
    if mutation == "model":
        envelope["model"] = "synthetic-other-model"
    elif mutation == "stale":
        envelope["choices"][0]["message"]["content"] = json.dumps(
            proposal_for(request, attempt_id="stale")
        )
    elif mutation == "truncated":
        envelope["choices"][0]["finish_reason"] = "length"
    else:
        envelope["choices"][0]["message"]["refusal"] = "synthetic refusal"
    client = client_factory(envelope=envelope)
    with pytest.raises(ClientError):
        client.propose(request)
    assert len(client.opener.calls) == 1
    summary = summarize_calls(client.response_log)
    assert summary["accepted_proposals"] == 0
    assert summary["failed_calls"] == 1
    assert summary["tokens"]["total_tokens"]["sum"] == 150


def test_http_error_usage_sanitization_and_unknown_transport_usage(client_factory):
    request = request_for()
    body = {
        "model": MODEL,
        "usage": envelope_for(request)["usage"],
        "error": {
            "code": "invalid_request",
            "type": "synthetic",
            "message": TEST_KEY + " synthetic failure",
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
    client.opener = FakeOpener(error=urllib.error.URLError(TEST_KEY))
    with pytest.raises(ClientError, match="URLError"):
        client.propose(request)
    assert len(client.opener.calls) == 1
    text = client.response_log.read_text()
    assert TEST_KEY not in text
    assert "Authorization" not in text
    rows = [json.loads(line) for line in text.splitlines()]
    assert rows[0]["provider_error"]["code"] == "invalid_request"
    assert rows[0]["provider_error"]["message"].startswith("[REDACTED]")
    summary = summarize_calls(rows)
    assert summary["provider_calls"] == summary["failed_calls"] == 2
    assert summary["tokens"]["total_tokens"] == {
        "sum": 150,
        "available_calls": 1,
        "missing_calls": 1,
        "complete": False,
    }


def test_preflight_failures_have_no_physical_cost_or_hidden_call(
    client_factory, monkeypatch
):
    request = request_for()
    client = client_factory(envelope=envelope_for(request))
    monkeypatch.delenv("NVIDIA_INFERENCE_API_KEY")
    with pytest.raises(ClientError, match="not set"):
        client.propose(request)
    request["source_catalog"].append({"oracle_pair": ["10", "13"]})
    with pytest.raises(ClientError):
        client.propose(request)
    assert client.opener.calls == []
    summary = summarize_calls(client.response_log)
    assert summary["client_attempts"] == summary["preflight_failures"] == 2
    assert summary["provider_calls"] == summary["failed_calls"] == 0
    assert summary["tokens"]["total_tokens"] == {
        "sum": 0,
        "available_calls": 0,
        "missing_calls": 0,
        "complete": True,
    }


def test_ledger_rejects_incompatible_versions_or_tampered_usage(client_factory):
    request = request_for()
    client = client_factory(envelope=envelope_for(request))
    client.propose(request)
    row = json.loads(client.response_log.read_text())
    unchanged = copy.deepcopy(row)
    assert summarize_calls([row])["provider_calls"] == 1
    assert row == unchanged and row["client_schema_version"] == SCHEMA_VERSION
    row["token_usage"]["total_tokens"] = 1
    with pytest.raises(ClientError, match="disagrees"):
        summarize_calls([row])
    row = copy.deepcopy(unchanged)
    row["client_schema_version"] = "intervention-1.0"
    with pytest.raises(ClientError, match="incompatible"):
        summarize_calls([row])
    with pytest.raises(FileNotFoundError):
        summarize_calls(client.response_log.with_name("missing-ledger.jsonl"))
