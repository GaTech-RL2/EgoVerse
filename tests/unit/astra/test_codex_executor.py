"""Hermetic Codex job receipts, schema compatibility, and crash recovery."""

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from astra_reversal import codex_executor as executor
from astra_reversal.frs_agent import direction_request


def request():
    return direction_request(
        episode_id="cpu",
        attempt_id="attempt",
        request_index=1,
        observation_step=0,
        target_task="Move to the bowl",
        external_image=np.zeros((16, 16, 3), dtype=np.uint8),
    )


def proposal(req):
    return {
        **{
            key: req[key]
            for key in (
                "schema_version",
                "role",
                "episode_id",
                "attempt_id",
                "request_index",
                "observation_step",
                "request_id",
            )
        },
        "response_id": "new-response",
        "fine": True,
        "coords": [0, 0, 0],
        "motion_amount": "less",
        "justification": "The target is occluded.",
    }


def mock_cli(
    monkeypatch,
    req,
    *,
    returncode=0,
    tool=False,
    malformed=False,
    invalid=False,
    incomplete=False,
    timeout=False,
    error_message=None,
):
    calls = []

    def run(args, **kwargs):
        calls.append((args, kwargs))
        if args[-1] == "--version":
            return subprocess.CompletedProcess(args, 0, "codex-cli 0.test\n", "")
        assert not Path(kwargs["cwd"]).is_relative_to(Path.cwd())
        assert kwargs.get("shell") is None
        assert "--ignore-user-config" in args and "--ignore-rules" in args
        assert "--ephemeral" in args and args[-1] == "-"
        if timeout:
            raise subprocess.TimeoutExpired(args, 1, output=b"", stderr=b"timeout")
        output = proposal(req)
        if invalid:
            output["request_id"] = "0" * 16
        final = Path(args[args.index("--output-last-message") + 1])
        final.write_text("broken" if malformed else json.dumps(output))
        events = [{"type": "thread.started", "thread_id": "synthetic"}]
        if error_message is not None:
            events.append(
                {
                    "type": "item.completed",
                    "item": {"type": "error", "message": error_message},
                }
            )
        if tool:
            events.append(
                {
                    "type": "item.completed",
                    "item": {"type": "command_execution", "command": "PRIVATE"},
                }
            )
        if not incomplete:
            events.append(
                {
                    "type": "turn.completed",
                    "usage": {
                        "input_tokens": 100,
                        "output_tokens": 20,
                        "reasoning_output_tokens": 5,
                        "cached_input_tokens": 30,
                    },
                }
            )
        return subprocess.CompletedProcess(
            args, returncode, "\n".join(map(json.dumps, events)), ""
        )

    monkeypatch.setattr(executor.subprocess, "run", run)
    return calls


def test_success_usage_and_recovery(tmp_path, monkeypatch):
    req = request()
    calls = mock_cli(monkeypatch, req)
    result = executor.execute_request(req, tmp_path / "job")
    receipt = result["receipt"]
    assert (
        receipt["accepted"]
        and result["proposal"]["request_fingerprint"] == req["request_fingerprint"]
    )
    assert receipt["token_usage"] == {
        "input_tokens": 100,
        "output_tokens": 20,
        "reasoning_tokens": 5,
        "total_tokens": 120,
    }
    assert (
        receipt["total_tokens_derived"]
        and receipt["raw_usage"]["cached_input_tokens"] == 30
    )
    assert receipt["returned_model"] is None and receipt["internal_retry_count"] is None
    assert len(receipt["image_hashes"]) == 1
    assert executor.execute_request(req, tmp_path / "job") == result
    assert len(calls) == 2


@pytest.mark.parametrize(
    "options,reason",
    [
        ({"returncode": 1}, "cli_nonzero_exit"),
        ({"tool": True}, "cli_tool_use"),
        ({"malformed": True}, "cli_final_malformed"),
        ({"incomplete": True}, "cli_incomplete_or_ambiguous_turn"),
        ({"timeout": True}, "cli_timeout"),
        ({"error_message": "Unrecognized CLI failure"}, "cli_error_item"),
    ],
)
def test_unavailable(tmp_path, monkeypatch, options, reason):
    req = request()
    mock_cli(monkeypatch, req, **options)
    result = executor.execute_request(req, tmp_path / "job")
    assert result["proposal"] is None
    assert result["receipt"]["provider_unavailable"]
    assert result["receipt"]["availability_reason"] == reason
    assert "PRIVATE" not in json.dumps(result)


def test_known_ignored_cli_startup_warning_is_audited_without_becoming_tool_use(
    tmp_path, monkeypatch
):
    req = request()
    mock_cli(
        monkeypatch,
        req,
        error_message="Ignoring unknown `features` requirement `ultrafast_mode` from requirements layers: managed fixture",
    )
    result = executor.execute_request(req, tmp_path / "job")
    assert result["receipt"]["accepted"]
    assert result["receipt"]["tool_items"] == result["receipt"]["error_items"] == []
    assert result["receipt"]["diagnostic_items"][0]["feature"] == "ultrafast_mode"


def test_schema_rejection_is_not_unavailable(tmp_path, monkeypatch):
    req = request()
    mock_cli(monkeypatch, req, invalid=True)
    result = executor.execute_request(req, tmp_path / "job")
    assert result["proposal"] is None
    assert result["receipt"]["error_kind"] == "proposal_rejected"
    assert not result["receipt"]["provider_unavailable"]


def test_ambiguous_job_never_restarts(tmp_path, monkeypatch):
    (tmp_path / "started.json").write_text("{}")

    def forbidden(*args, **kwargs):
        pytest.fail("An ambiguous job was restarted")

    monkeypatch.setattr(executor.subprocess, "run", forbidden)
    result = executor.execute_request(request(), tmp_path)
    assert (
        result["receipt"]["availability_reason"] == "already_started_job_indeterminate"
    )


def test_completed_job_different_binding_never_reused(tmp_path, monkeypatch):
    req = request()
    calls = mock_cli(monkeypatch, req)
    executor.execute_request(req, tmp_path)
    result = executor.execute_request(req, tmp_path, model="other")
    assert result["receipt"]["availability_reason"] == "job_directory_binding_mismatch"
    assert len(calls) == 2


def test_generation_schema_normalization():
    original = {
        "type": "object",
        "properties": {
            "identity": {"const": "abc"},
            "coords": {"items": {"enum": [-1, 0, 1]}},
            "choice": {"anyOf": [{"type": "null"}, {"enum": ["a", "b"]}]},
        },
        "oneOf": [{"properties": {"identity": {"const": "abc"}}}],
    }
    actual = executor.generation_schema(original)
    assert "oneOf" not in actual and "oneOf" in original
    assert actual["properties"]["identity"] == {"enum": ["abc"], "type": "string"}
    assert actual["properties"]["coords"]["items"]["type"] == "integer"
    assert actual["properties"]["choice"]["anyOf"][1]["type"] == "string"


@pytest.mark.parametrize("timed_out", [False, True])
def test_all_known_usage_survives_ambiguous_completion(
    tmp_path, monkeypatch, timed_out
):
    req = request()

    def run(args, **kwargs):
        if args[-1] == "--version":
            return subprocess.CompletedProcess(args, 0, "codex-cli 0.test", "")
        event = json.dumps(
            {
                "type": "turn.completed",
                "usage": {"input_tokens": 100, "output_tokens": 20},
            }
        )
        output = event + "\n" + event + "\n"
        if timed_out:
            raise subprocess.TimeoutExpired(
                args, 1, output=(output + '{"partial"').encode()
            )
        return subprocess.CompletedProcess(args, 0, output, "")

    monkeypatch.setattr(executor.subprocess, "run", run)
    result = executor.execute_request(req, tmp_path)
    assert result["receipt"]["provider_unavailable"]
    assert result["receipt"]["token_usage"]["input_tokens"] == 200
    assert result["receipt"]["token_usage"]["total_tokens"] == 240
    assert result["receipt"]["token_usage_is_lower_bound"]
    assert len(result["receipt"]["raw_usage_events"]) == 2


def test_unapproved_settings_fail_without_cli(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("CLI called with unapproved settings")

    monkeypatch.setattr(executor.subprocess, "run", forbidden)
    result = executor.execute_request(request(), tmp_path, reasoning_effort="high")
    assert result["receipt"]["provider_unavailable"]
    assert not result["receipt"]["provider_call"]
