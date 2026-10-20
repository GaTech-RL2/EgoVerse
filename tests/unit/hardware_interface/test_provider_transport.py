"""Native Python 3.8 timeouts must use the same bounded retries as newer Python."""

import io
import json
import socket

import pytest

from astra_reversal.hardware_interface import provider
from astra_reversal.hardware_interface.common import Events
from astra_reversal.hardware_interface.threading_protocol import load, validate


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("phase", ["open", "read"])
@pytest.mark.parametrize("exhausted", [False, True])
def test_socket_timeout_retry_keeps_request_history_and_usage(
    tmp_path, monkeypatch, legacy, phase, exhausted
):
    # Emulate the native runtime even when the local interpreter aliases
    # socket.timeout to TimeoutError. No live network or provider is involved.
    timeout_type = type("timeout", (OSError,), {}) if legacy else socket.timeout
    monkeypatch.setattr(provider.socket, "timeout", timeout_type)
    waits = []
    monkeypatch.setattr(provider.time, "sleep", waits.append)
    manifest = load()
    model = manifest["models"]["astra"]
    expected = {
        "model": model["identifier"],
        "status": "completed",
        "usage": {"input_tokens": 100, "output_tokens": 5},
        "output": [
            {
                "type": "function_call",
                "name": "read_latest",
                "call_id": "one-response",
                "arguments": "{}",
            }
        ],
    }
    requests = []

    class Reply(io.BytesIO):
        def read(self, *args):
            if phase == "read" and (exhausted or len(requests) == 1):
                raise timeout_type("private provider detail")
            return super().read(*args)

    class Opener:
        def open(self, request, timeout):
            requests.append(request.data)
            if phase == "open" and (exhausted or len(requests) == 1):
                raise timeout_type("private provider detail")
            return Reply(json.dumps(expected).encode())

    post = provider.HTTP("https://unit.example.invalid/v1", key="test-only")
    post.opener = Opener()
    meter = provider.Meter(None)
    events = Events(tmp_path / "trial", {"trial_id": "transport-only"})
    session = provider.Session(model, validate(manifest), meter, events, post=post)
    initial = {"role": "user", "content": "unchanged observation"}
    session.history = [initial]
    if exhausted:
        with pytest.raises(provider.ModelFailure, match="^provider_TimeoutError$"):
            session.request("instructions", [])
        assert len(requests) == model["retries"] + 1
        assert waits == [2 ** (attempt + 1) for attempt in range(model["retries"])]
        assert session.history == [initial]
        assert meter.total == 0
    else:
        assert session.request("instructions", []) == expected
        assert len(requests) == 2 and waits == [2]
        assert session.history == [initial] + expected["output"]
        assert meter.total == 105
    assert len(set(requests)) == 1
    failures = len(requests) - int(not exhausted)
    assert meter.unknown == failures
    assert meter.usage["actor"]["failed_transport_requests"] == failures
    assert meter.usage["actor"]["generation_attempts"] == len(requests)
    events.close()
    log = (events.directory / "events.jsonl").read_text()
    assert "private provider detail" not in log and "test-only" not in log
    errors = [r for r in map(json.loads, log.splitlines()) if r["event"] == "error"]
    assert len(errors) == failures
    assert all(r["reason"] == "provider_TimeoutError" for r in errors)
    assert errors[-1]["will_retry"] is (not exhausted)
