"""CPU-only audit corruption checks for model request families."""

import copy
import json
import subprocess
from pathlib import Path

import pytest

from astra_reversal.codex_executor import execute_request
from astra_reversal.codex_provider_audit import verify_codex_provider
from astra_reversal.frs_audit import _physical_schedule
from astra_reversal.meta_harness.relay_agent import wire_request

from .test_codex_executor import proposal
from .test_codex_executor import request as frs_request
from .test_meta_harness import request_fixture
from .test_representation_agent import proposal_for, request_for

SETTINGS = {
    "backend": "codex_relay",
    "model": "gpt-6-astra",
    "reasoning_effort": "medium",
    "max_completion_tokens": None,
    "stop_on_provider_unavailable": True,
    "retries": 0,
}


@pytest.fixture(params=["frs", "vei", "vli", "meta"])
def job(tmp_path, monkeypatch, request):
    if request.param == "frs":
        req = frs_request()
        response = proposal(req)
    elif request.param == "meta":
        req = wire_request(request_fixture())
        response = {
            "tool": "clear_policy_program",
            "arguments": {"observation_id": req["binding"]["observation_id"]},
        }
    else:
        req = request_for(representation_mode=request.param)
        response = proposal_for(req)

    def run(args, **kwargs):
        if args[-1] == "--version":
            return subprocess.CompletedProcess(args, 0, "codex-cli 0.test", "")
        raw = json.dumps(response)
        Path(args[args.index("--output-last-message") + 1]).write_text(raw)
        events = [
            {"type": "item.completed", "item": {"type": "agent_message", "text": raw}},
            {
                "type": "turn.completed",
                "usage": {"input_tokens": 20, "output_tokens": 10},
            },
        ]
        return subprocess.CompletedProcess(
            args, 0, "\n".join(map(json.dumps, events)), ""
        )

    monkeypatch.setattr("astra_reversal.codex_executor.subprocess.run", run)
    result = execute_request(req, tmp_path)
    receipt = result["receipt"]
    row = {
        **req,
        "backend": "codex_relay",
        "requested_model": "gpt-6-astra",
        "reasoning_effort": "medium",
        "accepted": True,
        "provider_call": True,
        "codex_receipt": receipt,
        "token_usage": receipt["token_usage"],
    }
    return req, row, result["proposal"], tmp_path


def test_original_job_verified_without_fabricating_served_model(job):
    req, row, value, directory = job
    result = verify_codex_provider(req, row, value, SETTINGS, job_directory=directory)
    assert result["status"] == "passed" and result["local_job_verified"]
    assert result["returned_model"] is None
    assert result["token_usage"]["total_tokens"] == 30
    assert "text" not in result


def test_missing_job_bytes_are_unverified(job):
    req, row, value, _ = job
    result = verify_codex_provider(req, row, value, SETTINGS)
    assert result["status"] == "unverified" and not result["local_job_verified"]
    assert "events.jsonl" in result["missing_files"]


@pytest.mark.parametrize("job", ["meta"], indirect=True)
def test_reloaded_runtime_request_keeps_original_prompt_proof(job):
    _, row, value, directory = job
    req = json.loads((directory / "request.json").read_text())
    result = verify_codex_provider(req, row, value, SETTINGS, job_directory=directory)
    assert result["status"] == "passed"
    assert not result["unverified_input_hashes"]
    if req["schema_version"] == "meta-harness-runtime-profile-1":
        assert result["prompt_key_order_recovered_from_original_artifact"]


@pytest.mark.parametrize("job", ["meta"], indirect=True)
def test_runtime_original_prompt_required_for_order_proof(job):
    _, row, value, directory = job
    req = json.loads((directory / "request.json").read_text())
    (directory / "prompt.txt").unlink()
    result = verify_codex_provider(req, row, value, SETTINGS, job_directory=directory)
    assert result["status"] == "unverified"
    assert result["missing_files"] == ["prompt.txt"]
    assert result["unverified_input_hashes"] == ["prompt_sha256"]


@pytest.mark.parametrize("job", ["meta"], indirect=True)
@pytest.mark.parametrize("changed_action", [999, False])
def test_runtime_prompt_order_recovery_cannot_change_context(job, changed_action):
    req, row, value, directory = job
    path = directory / "prompt.txt"
    lines = path.read_text().splitlines(keepends=True)
    index = next(i for i, line in enumerate(lines) if line.startswith('{"request":'))
    projection = json.loads(lines[index])
    projection["request"]["current"]["action"] = changed_action
    lines[index] = json.dumps(projection) + "\n"
    path.write_text("".join(lines))
    with pytest.raises(ValueError, match="context/schema differs"):
        verify_codex_provider(req, row, value, SETTINGS, job_directory=directory)


@pytest.mark.parametrize(
    "filename",
    ["image_0.png", "prompt.txt", "schema.json", "request.json", "final.json"],
)
def test_changed_artifact_fails(job, filename):
    req, row, value, directory = job
    path = directory / filename
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="differ"):
        verify_codex_provider(req, row, value, SETTINGS, job_directory=directory)


def test_changed_usage_event_fails(job):
    req, row, value, directory = job
    path = directory / "events.jsonl"
    events = [json.loads(x) for x in path.read_text().splitlines()]
    events[-1]["usage"]["input_tokens"] += 1
    path.write_text("\n".join(map(json.dumps, events)))
    with pytest.raises(ValueError, match="usage differs"):
        verify_codex_provider(req, row, value, SETTINGS, job_directory=directory)


def test_accepted_tool_use_cannot_be_hidden(job):
    req, row, value, directory = job
    path = directory / "events.jsonl"
    with path.open("a") as stream:
        stream.write('\n{"type":"item.completed","item":{"type":"command_execution"}}')
    with pytest.raises(ValueError, match="tool audit"):
        verify_codex_provider(req, row, value, SETTINGS, job_directory=directory)


def test_wrong_applied_proposal_fails(job):
    req, row, value, directory = job
    if "tool" in value:
        value = {
            **value,
            "tool": "keep_policy_program",
            "arguments": {**value["arguments"], "program_id": "another-program"},
        }
    else:
        value = (
            {**value, "justification": "Another valid observation."}
            if "fine" in value
            else {**value, "rationale": "Another valid observation."}
        )
    with pytest.raises(ValueError, match="applied decision"):
        verify_codex_provider(req, row, value, SETTINGS, job_directory=directory)


def frozen_protocol():
    return json.loads(
        (
            Path(__file__).resolve().parents[3]
            / "astra_reversal/configs/frs_codex_frozen_evaluation_v1.json"
        ).read_text()
    )


def test_frozen_pilot_vs_full_evaluation_schedule():
    protocol = frozen_protocol()
    pilot = _physical_schedule(protocol, True)
    full = _physical_schedule(protocol, False)
    assert pilot == {(method, 1, 0) for method in protocol["evaluation_methods"]}
    assert len(full) == 30 and pilot < full
    assert all(state in range(1, 11) and index == 0 for _, state, index in full)


@pytest.mark.parametrize(
    "field,value",
    [
        ("rounds", 3),
        ("adaptation_methods", ["critique_frs_learning"]),
        ("evaluation_methods", ["astra_frs"]),
        ("evaluation_states", [1]),
    ],
)
def test_frozen_schedule_changes_rejected(field, value):
    protocol = copy.deepcopy(frozen_protocol())
    protocol[field] = value
    with pytest.raises(ValueError):
        _physical_schedule(protocol, False)


def test_cli_reordered_images_rejected(job):
    req, row, value, directory = job
    path = directory / "invocation.json"
    invocation = json.loads(path.read_text())
    argv = invocation["argv"]
    indexes = [index + 1 for index, arg in enumerate(argv) if arg == "--image"]
    if len(indexes) < 2:
        pytest.skip("Single image has no image ordering ambiguity")
    argv[indexes[0]], argv[indexes[1]] = argv[indexes[1]], argv[indexes[0]]
    path.write_text(json.dumps(invocation))
    with pytest.raises(ValueError, match="image order"):
        verify_codex_provider(req, row, value, SETTINGS, job_directory=directory)
