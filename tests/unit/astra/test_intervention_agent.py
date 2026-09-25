"""Hermetic contract/accounting checks; synthetic replies are not Astra evidence."""

import copy
import io
import json
import urllib.error

import numpy as np
import pytest
from PIL import Image

from astra_reversal.agent import CAMERAS
from astra_reversal.astra_client import ClientError
from astra_reversal.intervention_agent import (
    ARM_CHANNELS,
    SCHEMA_VERSION,
    InterventionClient,
    build_payload,
    build_request,
    normalize_usage,
    parse_proposal,
    summarize_calls,
)

from .conftest import SyntheticEnvironment

MODEL = "synthetic-provider/intervention-fixture"
TEST_KEY = "synthetic-intervention-credential-never-a-real-secret"


def request_for(spec, arm="joint", **overrides):
    kwargs = {
        "episode_id": "synthetic-episode",
        "iteration": 2,
        "arm": arm,
        "task_instruction": "Place the cup in the bowl.",
        "action_spec": spec.as_dict(),
        "observations": [
            {
                "label": "baseline first",
                "step": 0,
                "observation": SyntheticEnvironment().observe(),
            },
            {
                "label": "baseline last",
                "step": 100,
                "observation": SyntheticEnvironment().observe(),
            },
        ],
        "prior_candidates": [
            {
                "candidate_id": "baseline",
                "iteration": 1,
                "proposal": None,
                "outcome": {
                    "success": False,
                    "executed_steps": 100,
                    "termination": "horizon",
                    "error": None,
                },
            }
        ],
        "basis_id": "synthetic-fixed-basis",
        "incumbent_candidate_id": "baseline",
    }
    return build_request(**(kwargs | overrides))


def proposal_for(request):
    enabled = ARM_CHANNELS[request["arm"]]
    return {
        **{
            key: request[key]
            for key in (
                "schema_version",
                "episode_id",
                "iteration",
                "arm",
                "request_fingerprint",
            )
        },
        "candidate_id": f"candidate-{request['iteration']}",
        "best_candidate_id": "baseline",
        "rationale": "Observed baseline did not finish; test this declared candidate.",
        "language": {
            "target_text": "Place the cup fully inside the bowl.",
            "scale": 0.5,
        }
        if "language" in enabled
        else None,
        "vision": [
            {
                "camera": CAMERAS[0],
                "kind": "box",
                "coordinates": [1, 2, 13, 14],
                "gain": 0.3,
            }
        ]
        if "vision" in enabled
        else [],
        "noise": {
            "basis_id": request["basis_id"],
            "coefficients": [0.1] * 8,
            "perturbation_scale": 0.3,
        }
        if "noise" in enabled
        else None,
    }


def envelope_for(request, usage=None):
    return {
        "id": "synthetic-response",
        "object": "chat.completion",
        "model": MODEL,
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": json.dumps(proposal_for(request)),
                },
            }
        ],
        "usage": usage
        or {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "completion_tokens_details": {"reasoning_tokens": 20},
        },
    }


class FakeOpener:
    def __init__(self, *, envelope=None, error=None):
        self.envelope, self.error, self.calls = envelope, error, []

    def open(self, request, timeout):
        self.calls.append((request, timeout))
        if self.error:
            raise self.error
        result = io.BytesIO(json.dumps(self.envelope).encode())
        result.status = 200
        return result


@pytest.fixture
def client_factory(monkeypatch, tmp_path):
    monkeypatch.setenv("NVIDIA_INFERENCE_API_KEY", TEST_KEY)

    def make(*, envelope=None, error=None, **kwargs):
        client = InterventionClient(
            model=MODEL, response_log=tmp_path / "provider.jsonl", **kwargs
        )
        client.opener = FakeOpener(envelope=envelope, error=error)
        return client

    return make


def read_log(client):
    return json.loads(client.response_log.read_text().splitlines()[-1])


def test_raw_frames_exact_controller_and_outcomes_bound_to_request(spec):
    request = request_for(spec)
    payload = build_payload(request, MODEL)
    assert payload["cache"] == {"no-cache": True}
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["reasoning_effort"] == "low"
    assert payload["max_completion_tokens"] == 8192
    assert "temperature" not in payload
    assert payload["stream"] is False
    parts = payload["messages"][1]["content"]
    context = json.loads(parts[0]["text"])
    for field in (
        "action_spec",
        "prior_candidates",
        "basis_id",
        "request_fingerprint",
        "task_instruction",
    ):
        assert context["request"][field] == request[field]
    assert "base64_png" not in parts[0]["text"]
    images = [part for part in parts if part["type"] == "image_url"]
    assert len(images) == 4
    for index, snapshot in enumerate(request["observations"]):
        for camera_index, camera in enumerate(CAMERAS):
            assert (
                images[2 * index + camera_index]["image_url"]["url"]
                == "data:image/png;base64," + snapshot["observation"][camera]["data"]
            )
    changed = copy.deepcopy(request)
    changed["prior_candidates"][0]["outcome"]["success"] = True
    with pytest.raises(ClientError, match="fingerprint"):
        build_payload(changed, MODEL)


def test_input_encoding_preserves_pixels_and_does_not_alias_arrays(spec):
    import base64

    observation = SyntheticEnvironment().observe()
    observation[CAMERAS[0]][2, 3] = [42, 92, 17]
    request = request_for(
        spec,
        observations=[{"label": "baseline", "step": 0, "observation": observation}],
    )
    wire = request["observations"][0]["observation"][CAMERAS[0]]
    with Image.open(io.BytesIO(base64.b64decode(wire["data"]))) as image:
        assert np.array_equal(np.array(image), observation[CAMERAS[0]])
    observation[CAMERAS[0]][2, 3] = 0
    with Image.open(io.BytesIO(base64.b64decode(wire["data"]))) as image:
        assert list(np.array(image)[2, 3]) == [42, 92, 17]


@pytest.mark.parametrize("arm", ARM_CHANNELS)
def test_all_seven_arms_accept_exact_declared_channels(spec, arm):
    request = request_for(spec, arm)
    proposal = proposal_for(request)
    assert parse_proposal(json.dumps(proposal), request) == proposal
    schema = json.loads(
        build_payload(request, MODEL)["messages"][1]["content"][0]["text"]
    )["response_schema"]
    assert set(schema["required"]) == set(proposal)
    assert schema["additionalProperties"] is False


@pytest.mark.parametrize(
    "arm,channel",
    [("noise_only", "language"), ("language_only", "vision"), ("vision_only", "noise")],
)
def test_disabled_channels_are_not_silently_applied(spec, arm, channel):
    request = request_for(spec, arm)
    proposal = proposal_for(request)
    proposal[channel] = proposal_for(request_for(spec))[channel]
    with pytest.raises(ClientError, match="disabled"):
        parse_proposal(proposal, request)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", "1.0"),
        ("episode_id", "another-episode"),
        ("iteration", True),
        ("iteration", 3),
        ("arm", "noise_only"),
        ("request_fingerprint", "unbound"),
        ("candidate_id", "baseline"),
        ("best_candidate_id", "imagined"),
        ("rationale", ""),
    ],
)
def test_proposal_identity_and_assessment_cannot_drift(spec, field, value):
    request = request_for(spec)
    proposal = proposal_for(request)
    proposal[field] = value
    with pytest.raises(ClientError):
        parse_proposal(proposal, request)


@pytest.mark.parametrize(
    "channel,field,value",
    [
        ("noise", "basis_id", "new-basis"),
        ("noise", "coefficients", [0.5] * 8),
        ("noise", "coefficients", [0.1] * 7),
        ("noise", "coefficients", [True] + [0] * 7),
        ("noise", "perturbation_scale", 0.500001),
        ("noise", "perturbation_scale", -0.1),
        ("noise", "perturbation_scale", True),
        ("language", "target_text", "x" * 161),
        ("language", "scale", 1.01),
        ("language", "scale", False),
    ],
)
def test_noise_and_actual_language_bounds_reject_without_repair(
    spec, channel, field, value
):
    request = request_for(spec)
    proposal = proposal_for(request)
    proposal[channel][field] = value
    original = copy.deepcopy(proposal)
    with pytest.raises(ClientError):
        parse_proposal(proposal, request)
    assert proposal == original


@pytest.mark.parametrize(
    "field,value",
    [
        ("coordinates", [0, 0, 16, 15]),
        ("coordinates", [5, 5, 2, 2]),
        ("coordinates", [False, 0, 5, 5]),
        ("coordinates", [1, 2]),
        ("camera", "imaginary_camera"),
        ("gain", 1.01),
        ("gain", True),
    ],
)
def test_static_annotations_use_exact_raw_camera_bounds(spec, field, value):
    request = request_for(spec)
    proposal = proposal_for(request)
    proposal["vision"][0][field] = value
    with pytest.raises(ClientError):
        parse_proposal(proposal, request)


def test_annotation_counts_and_unknown_fields_are_bounded(spec):
    request = request_for(spec)
    proposal = proposal_for(request)
    proposal["vision"] *= 5
    with pytest.raises(ClientError, match="four annotations"):
        parse_proposal(proposal, request)
    proposal = proposal_for(request)
    proposal["noise"]["seed"] = 91
    with pytest.raises(ClientError, match="declared fields"):
        parse_proposal(proposal, request)


@pytest.mark.parametrize(
    "raw", ['{"schema_version":NaN}', '{"a":1,"a":2}', "```json\n{}\n```", "{} {}"]
)
def test_non_json_and_duplicate_or_nonfinite_values_are_rejected(spec, raw):
    with pytest.raises(ClientError):
        parse_proposal(raw, request_for(spec))


def test_failed_prior_attempts_remain_visible_and_bound(spec):
    previous = request_for(spec)["prior_candidates"]
    previous.append(
        {
            "candidate_id": "failed-2",
            "iteration": 2,
            "proposal": None,
            "outcome": {
                "success": None,
                "executed_steps": 0,
                "termination": "proposal_error",
                "error": "Inference endpoint returned HTTP 503",
            },
        }
    )
    request = request_for(spec, iteration=3, prior_candidates=previous)
    assert request["prior_candidates"][1]["outcome"]["success"] is None
    assert request["iteration"] == 3
    for kwargs in (
        {"iteration": 6},
        {"max_iterations": 6},
        {"iteration": 1},
        {"incumbent_candidate_id": "unseen"},
        {"iteration": 3},
    ):
        with pytest.raises(ClientError):
            request_for(spec, **kwargs)


def test_runner_baseline_and_failure_feedback_validate_without_special_cases(spec):
    from astra_reversal.intervention_search import feedback_row

    baseline = {
        "candidate_id": "reversal_identity",
        "iteration": 1,
        "proposal": None,
        "success": False,
        "rollout_executed": True,
        "actions_executed": 100,
        "status": "budget_exhausted",
    }
    rejected = {
        "candidate_id": "joint_error_2",
        "iteration": 2,
        "proposal": None,
        "success": False,
        "rollout_executed": False,
        "actions_executed": 0,
        "status": "proposal_error",
        "error": "ClientError",
    }
    request = request_for(
        spec,
        iteration=3,
        prior_candidates=[feedback_row(baseline), feedback_row(rejected)],
        incumbent_candidate_id="reversal_identity",
    )
    payload = build_payload(request, MODEL)
    context = json.loads(payload["messages"][1]["content"][0]["text"])
    assert (
        context["request"]["prior_candidates"][0]["candidate_id"] == "reversal_identity"
    )
    assert context["request"]["prior_candidates"][1]["outcome"]["success"] is None
    assert context["request"]["prior_candidates"][1]["outcome"]["executed_steps"] == 0


def test_baseline_and_at_most_four_consistent_snapshots_are_required(spec):
    snapshots = [
        {
            "label": "frame",
            "step": index,
            "observation": SyntheticEnvironment().observe(),
        }
        for index in range(5)
    ]
    with pytest.raises(ClientError, match="one to four"):
        request_for(spec, observations=snapshots)
    snapshots = snapshots[:2]
    snapshots[1]["observation"][CAMERAS[0]] = np.zeros((18, 16, 3), dtype=np.uint8)
    with pytest.raises(ClientError, match="consistent"):
        request_for(spec, observations=snapshots)
    with pytest.raises(ClientError, match="baseline"):
        request_for(spec, prior_candidates=[])


def test_real_transport_shape_and_exact_response_usage_preserved(spec, client_factory):
    request = request_for(spec)
    envelope = envelope_for(request)
    client = client_factory(envelope=envelope)
    assert client.propose(request) == proposal_for(request)
    assert len(client.opener.calls) == 1
    http_request, timeout = client.opener.calls[0]
    assert http_request.get_header("Authorization") == "Bearer " + TEST_KEY
    assert http_request.method == "POST"
    assert timeout == 170
    assert json.loads(http_request.data) == build_payload(request, MODEL)
    row = read_log(client)
    assert row["response"] == envelope
    assert row["requested_model"] == MODEL
    assert row["accepted"] is True
    assert row["token_usage"] == {
        "input_tokens": 100,
        "output_tokens": 50,
        "reasoning_tokens": 20,
        "total_tokens": 150,
    }
    assert row["iteration"] == 2
    assert row["request_fingerprint"] == request["request_fingerprint"]
    assert TEST_KEY not in client.response_log.read_text()


@pytest.mark.parametrize(
    "failure", ["model", "truncated", "refusal", "bad_proposal", "multiple_choices"]
)
def test_rejected_completions_keep_real_usage_and_never_retry(
    spec, client_factory, failure
):
    request = request_for(spec)
    envelope = envelope_for(request)
    if failure == "model":
        envelope["model"] = "synthetic-substitute"
    elif failure == "truncated":
        envelope["choices"][0]["finish_reason"] = "length"
    elif failure == "refusal":
        envelope["choices"][0]["message"]["refusal"] = "no"
    elif failure == "bad_proposal":
        envelope["choices"][0]["message"]["content"] = "{}"
    else:
        envelope["choices"] *= 2
    client = client_factory(envelope=envelope)
    with pytest.raises(ClientError):
        client.propose(request)
    assert len(client.opener.calls) == 1
    row = read_log(client)
    assert row["accepted"] is False
    assert row["token_usage"]["total_tokens"] == 150
    assert summarize_calls(client.response_log)["tokens"]["total_tokens"]["sum"] == 150


def test_http_failure_usage_is_counted_without_logging_body_or_headers(
    spec, client_factory
):
    body = {
        "id": "failed-call",
        "model": MODEL,
        "usage": {"prompt_tokens": 71, "completion_tokens": 5, "total_tokens": 76},
        "error": {"message": "sensitive provider message " + TEST_KEY},
        "headers": {"Authorization": TEST_KEY},
    }
    error = urllib.error.HTTPError(
        "https://fixture.invalid",
        503,
        "error " + TEST_KEY,
        {"private": TEST_KEY},
        io.BytesIO(json.dumps(body).encode()),
    )
    client = client_factory(error=error)
    with pytest.raises(ClientError, match="HTTP 503") as caught:
        client.propose(request_for(spec))
    assert TEST_KEY not in str(caught.value)
    row = read_log(client)
    assert row["http_status"] == 503
    assert row["token_usage"]["total_tokens"] == 76
    assert row["response"] == {key: body[key] for key in ("id", "model", "usage")}
    assert "sensitive provider message" not in client.response_log.read_text()
    assert TEST_KEY not in client.response_log.read_text()
    assert (
        summarize_calls(client.response_log)["tokens"]["reasoning_tokens"][
            "missing_calls"
        ]
        == 1
    )


@pytest.mark.parametrize(
    "error",
    [
        urllib.error.URLError("untrusted error detail"),
        TimeoutError("untrusted timeout detail"),
        urllib.error.HTTPError(
            "https://fixture.invalid",
            500,
            "failure",
            {},
            io.BytesIO(b"invalid provider body"),
        ),
    ],
)
def test_unknown_failed_call_usage_stays_unavailable(spec, client_factory, error):
    client = client_factory(error=error)
    with pytest.raises(ClientError):
        client.propose(request_for(spec))
    summary = summarize_calls(client.response_log)
    assert summary["provider_calls"] == summary["failed_calls"] == 1
    assert summary["usage_unavailable_calls"] == 1
    assert summary["tokens"]["total_tokens"] == {
        "sum": 0,
        "available_calls": 0,
        "missing_calls": 1,
        "complete": False,
    }
    assert summary["monetary_cost"] is None
    assert "untrusted" not in client.response_log.read_text()


def test_missing_credential_and_invalid_requests_make_no_provider_calls(
    spec, client_factory, monkeypatch
):
    request = request_for(spec)
    client = client_factory(envelope=envelope_for(request))
    monkeypatch.delenv("NVIDIA_INFERENCE_API_KEY")
    with pytest.raises(ClientError, match="not set"):
        client.propose(request)
    monkeypatch.setenv("NVIDIA_INFERENCE_API_KEY", TEST_KEY)
    request["iteration"] = 9
    with pytest.raises(ClientError):
        client.propose(request)
    assert not client.opener.calls
    assert not client.response_log.exists()


def test_usage_aliases_conflicts_and_reasoning_subset_are_explicit():
    assert normalize_usage(
        {
            "input_tokens": 10,
            "output_tokens": 20,
            "output_tokens_details": {"reasoning_tokens": 4},
            "total_tokens": 30,
        }
    ) == {
        "input_tokens": 10,
        "output_tokens": 20,
        "reasoning_tokens": 4,
        "total_tokens": 30,
    }
    usage = normalize_usage(
        {
            "prompt_tokens": 10,
            "input_tokens": 11,
            "completion_tokens": True,
            "total_tokens": -1,
            "completion_tokens_details": {"reasoning_tokens": "4"},
        }
    )
    assert all(value is None for value in usage.values())
    assert (
        normalize_usage(
            {
                "completion_tokens": 3,
                "completion_tokens_details": {"reasoning_tokens": 4},
            }
        )["reasoning_tokens"]
        is None
    )


def test_total_accounting_includes_failed_calls_without_double_counting_reasoning(
    spec, client_factory
):
    request = request_for(spec)
    client = client_factory(envelope=envelope_for(request))
    client.propose(request)
    client.opener = FakeOpener(
        envelope={**envelope_for(request), "model": "wrong-model"}
    )
    with pytest.raises(ClientError):
        client.propose(request)
    client.opener = FakeOpener(error=TimeoutError())
    with pytest.raises(ClientError):
        client.propose(request)
    summary = summarize_calls(client.response_log)
    assert summary["provider_calls"] == 3
    assert summary["accepted_proposals"] == 1
    assert summary["failed_calls"] == 2
    assert summary["tokens"]["input_tokens"]["sum"] == 200
    assert summary["tokens"]["output_tokens"]["sum"] == 100
    assert summary["tokens"]["reasoning_tokens"]["sum"] == 40
    assert summary["tokens"]["total_tokens"]["sum"] == 300
    assert summary["tokens"]["total_tokens"]["missing_calls"] == 1
    assert summary["reasoning_tokens_are_subset_of_output_tokens"] is True
    rows = [json.loads(line) for line in client.response_log.read_text().splitlines()]
    rows[0]["token_usage"]["total_tokens"] = 1
    with pytest.raises(ClientError, match="disagrees"):
        summarize_calls(rows)
    with pytest.raises(ClientError, match="incompatible"):
        summarize_calls([{"client_schema_version": SCHEMA_VERSION}])


def test_missing_ledger_is_not_reported_as_zero_cost(tmp_path):
    with pytest.raises(FileNotFoundError):
        summarize_calls(tmp_path / "missing.jsonl")
    assert summarize_calls([])["provider_calls"] == 0


def test_malformed_provider_model_still_has_accountable_failed_usage(
    spec, client_factory
):
    request = request_for(spec)
    envelope = envelope_for(request)
    envelope["model"] = {"invalid": "model identity"}
    client = client_factory(envelope=envelope)
    with pytest.raises(ClientError, match="different model"):
        client.propose(request)
    summary = summarize_calls(client.response_log)
    assert summary["actual_models"] == {"unavailable": 1}
    assert summary["tokens"]["total_tokens"]["sum"] == 150
