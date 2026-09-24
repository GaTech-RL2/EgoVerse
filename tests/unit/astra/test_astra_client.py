"""Synthetic HTTP fixtures test transport contracts, not Astra performance."""

import copy
import io
import json
import urllib.error

import pytest

from astra_reversal.agent import _wire_value, build_request
from astra_reversal.astra_client import (
    AstraHTTPClient,
    ClientError,
    build_payload,
    main,
)

from .conftest import SyntheticBackend, SyntheticEnvironment

MODEL = "synthetic-provider/astra-test-fixture"
TEST_KEY = "synthetic-credential-for-unit-tests-only"


def request_for(spec, stage=1):
    return _wire_value(
        build_request(
            episode_id="test-episode",
            step=7,
            observation=SyntheticEnvironment().observe(),
            instruction="move cup",
            history=[
                {
                    "observation_step": 6,
                    "action": [0.1] * 7,
                    "subgoal_id": "reach",
                    "fallback_reason": None,
                }
            ],
            active=None,
            spec=spec,
            stage=stage,
            completion_types=("eef_position", "gripper_width"),
            model_version=MODEL,
        )
    )


def envelope_for(request):
    return {
        "id": "synthetic-completion-1",
        "object": "chat.completion",
        "model": MODEL,
        "created": 1,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": SyntheticBackend().generate(request),
                },
            }
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 34, "total_tokens": 46},
    }


class FakeOpener:
    def __init__(self, envelope=None, error=None):
        self.envelope, self.error, self.calls = envelope, error, []

    def open(self, request, timeout):
        self.calls.append((request, timeout))
        if self.error:
            raise self.error
        response = io.BytesIO(json.dumps(self.envelope).encode())
        response.status = 200
        return response


@pytest.fixture
def client_factory(monkeypatch, tmp_path):
    monkeypatch.setenv("NVIDIA_INFERENCE_API_KEY", TEST_KEY)

    def make(envelope=None, error=None, **kwargs):
        client = AstraHTTPClient(
            model=MODEL,
            response_log=tmp_path / "responses.jsonl",
            **kwargs,
        )
        client.opener = FakeOpener(envelope, error)
        return client

    return make


def read_log(client):
    return json.loads(client.response_log.read_text().splitlines()[-1])


def test_native_images_exact_state_history_and_spec_are_sent(spec):
    request = request_for(spec)
    original = copy.deepcopy(request)
    payload, decoded = build_payload(
        request, MODEL, sampling={"reasoning_effort": "low"}
    )
    assert request == original
    assert payload["model"] == MODEL
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["cache"] == {"no-cache": True}
    assert payload["reasoning_effort"] == "low"
    assert "temperature" not in payload
    assert payload["stream"] is False
    content = payload["messages"][1]["content"]
    context = json.loads(content[0]["text"])
    for field in ("action_spec", "history", "task_instruction", "request_fingerprint"):
        assert context["request"][field] == request[field]
    assert (
        context["request"]["observation"]["observation/state"]
        == request["observation"]["observation/state"]
    )
    for index, camera in enumerate(("observation/image", "observation/wrist_image")):
        assert content[1 + index * 2] == {
            "type": "text",
            "text": f"Current camera: {camera}",
        }
        assert (
            content[2 + index * 2]["image_url"]["url"]
            == "data:image/png;base64," + request["observation"][camera]["data"]
        )
        assert decoded["observation"][camera].shape == (16, 16, 3)
    schema = context["response_schema"]["properties"]["action_chunk"]
    assert schema["minItems"] == schema["maxItems"] == spec.horizon
    assert schema["items"]["minItems"] == schema["items"]["maxItems"] == 7
    # Pixels occur in image parts rather than as opaque base64 text to the model.
    assert "base64_png" not in content[0]["text"]


@pytest.mark.parametrize("stage", [1, 2])
def test_original_response_and_provider_provenance_are_preserved(
    spec, client_factory, stage
):
    request = request_for(spec, stage)
    envelope = envelope_for(request)
    raw = "  " + envelope["choices"][0]["message"]["content"] + "\n"
    envelope["choices"][0]["message"]["content"] = raw
    client = client_factory(envelope, sampling={"reasoning_effort": "low"})
    assert client.generate(request) == raw
    http_request, timeout = client.opener.calls[0]
    assert http_request.method == "POST"
    assert http_request.get_header("Authorization") == "Bearer " + TEST_KEY
    assert (
        http_request.full_url == "https://inference-api.nvidia.com/v1/chat/completions"
    )
    assert timeout == client.timeout
    log = read_log(client)
    assert log["accepted"] is True
    assert log["response"] == envelope
    assert log["request_fingerprint"] == request["request_fingerprint"]
    assert log["requested_model"] == MODEL
    assert log["response"]["usage"]["total_tokens"] == 46
    assert log["latency_seconds"] >= 0
    assert log["response_time"] >= log["request_time"]
    assert TEST_KEY not in client.response_log.read_text()
    assert "Authorization" not in client.response_log.read_text()


@pytest.mark.parametrize(
    "change",
    [
        lambda data: data["action_chunk"].pop(),
        lambda data: data["action_chunk"][0].__setitem__(0, 2),
        lambda data: data["action_chunk"][0].__setitem__(0, True),
        lambda data: data.update(observation_step=8),
        lambda data: data.update(action_spec_id="wrong"),
        lambda data: data.update(extra_field="not in schema"),
    ],
)
def test_invalid_actions_are_recorded_and_rejected_without_repair(
    spec, client_factory, change
):
    request = request_for(spec)
    envelope = envelope_for(request)
    data = json.loads(envelope["choices"][0]["message"]["content"])
    change(data)
    raw = json.dumps(data)
    envelope["choices"][0]["message"]["content"] = raw
    client = client_factory(envelope)
    with pytest.raises(ClientError, match="Proposal rejected"):
        client.generate(request)
    assert len(client.opener.calls) == 1  # The agent, not the transport, owns retries.
    log = read_log(client)
    assert log["accepted"] is False
    assert log["response"]["choices"][0]["message"]["content"] == raw


@pytest.mark.parametrize(
    "raw",
    [
        '{"schema_version":"1.0","schema_version":"1.0"}',
        '{"action_chunk":NaN}',
        "```json\n{}\n```",
        "{} {}",
    ],
)
def test_noncanonical_json_is_rejected(spec, client_factory, raw):
    request = request_for(spec)
    envelope = envelope_for(request)
    envelope["choices"][0]["message"]["content"] = raw
    client = client_factory(envelope)
    with pytest.raises(ClientError):
        client.generate(request)
    assert read_log(client)["accepted"] is False


def test_truncated_proposal_is_rejected_even_if_content_parses(spec, client_factory):
    request = request_for(spec)
    envelope = envelope_for(request)
    envelope["choices"][0]["finish_reason"] = "length"
    client = client_factory(envelope)
    with pytest.raises(ClientError, match="truncated"):
        client.generate(request)


def test_provider_model_substitution_is_rejected(spec, client_factory):
    request = request_for(spec)
    envelope = envelope_for(request)
    envelope["model"] = "synthetic-provider/different-model"
    client = client_factory(envelope)
    with pytest.raises(ClientError, match="different model"):
        client.generate(request)
    log = read_log(client)
    assert log["accepted"] is False
    assert log["response"]["model"] == envelope["model"]


def test_stage2_uses_source_image_bounds(spec, client_factory):
    request = request_for(spec, stage=2)
    envelope = envelope_for(request)
    data = json.loads(envelope["choices"][0]["message"]["content"])
    data["annotations"][0]["coordinates"] = [0, 0, 16, 16]
    envelope["choices"][0]["message"]["content"] = json.dumps(data)
    with pytest.raises(ClientError, match="outside the source image"):
        client_factory(envelope).generate(request)


def test_http_error_does_not_expose_headers_provider_body_or_credentials(
    spec, client_factory
):
    error = urllib.error.HTTPError(
        "https://example.invalid",
        401,
        "unauthorized " + TEST_KEY,
        {"Authorization": TEST_KEY},
        io.BytesIO(TEST_KEY.encode()),
    )
    client = client_factory(error=error)
    with pytest.raises(ClientError, match="HTTP 401") as caught:
        client.generate(request_for(spec))
    assert TEST_KEY not in str(caught.value)
    assert TEST_KEY not in client.response_log.read_text()
    assert "headers" not in read_log(client)
    assert read_log(client)["http_status"] == 401


def test_response_logging_filters_credential_echo_and_headers(spec, client_factory):
    request = request_for(spec)
    envelope = envelope_for(request)
    envelope["headers"] = {"Authorization": TEST_KEY}
    envelope["usage"]["api_key"] = TEST_KEY
    envelope["choices"][0]["message"]["content"] = TEST_KEY
    client = client_factory(envelope)
    with pytest.raises(ClientError):
        client.generate(request)
    text = client.response_log.read_text()
    assert TEST_KEY not in text
    assert "headers" not in text
    assert "api_key" not in text
    assert "[REDACTED]" in text


@pytest.mark.parametrize("extra", ["object_poses", "success", "goal_truth"])
def test_oracle_observation_fields_are_never_transmitted(spec, client_factory, extra):
    request = request_for(spec)
    request["observation"][extra] = "forbidden fixture"
    client = client_factory()
    with pytest.raises(ClientError, match="Only the two declared cameras"):
        client.generate(request)
    assert not client.opener.calls


def test_configured_model_must_match_request(spec):
    with pytest.raises(ClientError, match="differs"):
        build_payload(request_for(spec), "another-model")


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://remote.invalid/v1",
        "https://user:password@example.invalid/v1",
        "https://example.invalid/v1?api_key=secret",
        "https://example.invalid/v1#fragment",
    ],
)
def test_endpoint_cannot_embed_credentials_or_send_them_over_remote_http(
    tmp_path, endpoint
):
    with pytest.raises(ClientError):
        AstraHTTPClient(
            model=MODEL, response_log=tmp_path / "responses.jsonl", endpoint=endpoint
        )


def test_cli_stdout_is_only_the_command_backend_proposal(
    spec, monkeypatch, tmp_path, capsys
):
    request = request_for(spec)
    envelope = envelope_for(request)
    opener = FakeOpener(envelope)
    monkeypatch.setenv("NVIDIA_INFERENCE_API_KEY", TEST_KEY)
    monkeypatch.setattr("urllib.request.build_opener", lambda *args: opener)
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(request)))
    status = main(
        [
            "--model",
            MODEL,
            "--response-log",
            str(tmp_path / "api.jsonl"),
            "--reasoning-effort",
            "low",
        ]
    )
    captured = capsys.readouterr()
    assert status == 0
    assert captured.err == ""
    assert captured.out.strip() == envelope["choices"][0]["message"]["content"]


def test_missing_key_fails_before_network(spec, client_factory, monkeypatch):
    client = client_factory()
    monkeypatch.delenv("NVIDIA_INFERENCE_API_KEY")
    with pytest.raises(ClientError, match="NVIDIA_INFERENCE_API_KEY is not set"):
        client.generate(request_for(spec))
    assert not client.opener.calls
