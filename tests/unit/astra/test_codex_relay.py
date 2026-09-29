"""Hermetic mailbox/HTTP-stream tests; no sockets, provider, GPU, or CLI calls."""

import io
import json
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest

from astra_reversal import codex_relay as relay
from astra_reversal import frs_agent
from astra_reversal.astra_client import ClientError
from astra_reversal.intervention_agent import normalize_usage

TOKEN = "synthetic-relay-token-never-published"


def request_for():
    return frs_agent.direction_request(
        episode_id="synthetic-relay-case",
        attempt_id="trial-1",
        request_index=1,
        observation_step=0,
        target_task="Place the cup in the bowl.",
        external_image=np.full((16, 16, 3), 17, np.uint8),
    )


def envelope_for(request=None):
    request = request or request_for()
    return {
        "protocol": relay.PROTOCOL_VERSION,
        "invocation_id": "a" * 32,
        "request_fingerprint": request["request_fingerprint"],
        "family": "frs",
        "request": request,
        "model": relay.MODEL,
        "reasoning_effort": "medium",
        "timeout": 3,
        "expires_at": time.time() + 3,
    }


def result_for(envelope, *, unavailable=False):
    request = envelope["request"]
    proposal = frs_agent.parse_proposal(
        {
            **{key: request[key] for key in frs_agent._IDENTITY_FIELDS},
            "response_id": "synthetic-response",
            "fine": True,
            "coords": [0, 0, 0],
            "motion_amount": "less",
            "justification": "Grasp alignment is uncertain.",
        },
        request,
    )
    return {
        "request_fingerprint": request["request_fingerprint"],
        "proposal": None if unavailable else proposal,
        "receipt": {
            "backend": "codex_exec",
            "configured_model": relay.MODEL,
            "requested_model": relay.MODEL,
            "returned_model": None,
            "reasoning_effort": "medium",
            "request_fingerprint": request["request_fingerprint"],
            "provider_call": True,
            "accepted": not unavailable,
            "provider_unavailable": unavailable,
            "status": "provider_unavailable" if unavailable else "accepted",
            "observed_completed_turns": 1,
            "exit_code": 0,
            "tool_items": [],
            "raw_usage": {
                "input_tokens": 100,
                "output_tokens": 23,
                "reasoning_output_tokens": 8,
            },
            "token_usage": {
                "input_tokens": 100,
                "output_tokens": 23,
                "reasoning_tokens": 8,
                "total_tokens": 123,
            },
        },
    }


def submission(envelope, result=None):
    return {key: envelope[key] for key in ("invocation_id", "request_fingerprint")} | {
        "result": result if result is not None else result_for(envelope)
    }


def client_for(monkeypatch, tmp_path, exchange=result_for):
    monkeypatch.setenv("ASTRA_CODEX_RELAY_TOKEN", TOKEN)
    monkeypatch.setenv("NVIDIA_INFERENCE_API_KEY", "provider-key-not-for-this-relay")
    mailbox = SimpleNamespace(exchange=lambda envelope, timeout: exchange(envelope))
    monkeypatch.setattr(
        relay, "ensure_server", lambda **kwargs: SimpleNamespace(mailbox=mailbox)
    )
    return relay.CodexRelayClient(
        model=relay.MODEL, response_log=tmp_path / "calls.jsonl", family="frs"
    )


def wait_pending(mailbox):
    with mailbox._condition:
        assert mailbox._condition.wait_for(
            lambda: mailbox._pending is not None, timeout=2
        )
    return mailbox.pending()


def handler(mailbox, *, token=TOKEN, path="/health", body=b"", size=None):
    value = object.__new__(relay.RelayHandler)
    value.server = SimpleNamespace(mailbox=mailbox, relay_token=TOKEN)
    value.path = path
    value.headers = {
        "Authorization": "Bearer " + token,
        "Content-Type": "application/json",
        "Content-Length": str(len(body) if size is None else size),
    }
    value.rfile, value.wfile = io.BytesIO(body), io.BytesIO()
    value.responses = []
    value._send = lambda status, body: value.responses.append((status, body))
    return value


def test_client_revalidates_bound_proposal_and_retains_actual_usage_once(
    monkeypatch, tmp_path
):
    seen = []

    def execute(envelope):
        seen.append(envelope)
        return result_for(envelope)

    client = client_for(monkeypatch, tmp_path, execute)
    request = request_for()
    assert (
        client.propose(request)["request_fingerprint"] == request["request_fingerprint"]
    )
    assert len(client.records) == 1
    record = client.records[0]
    assert record["client_schema_version"] == frs_agent.SCHEMA_VERSION
    assert record["backend"] == "codex_relay" and record["provider_call"] is True
    assert record["token_usage"]["total_tokens"] == 123
    assert record["raw_usage"]["reasoning_output_tokens"] == 8
    assert "http_status" not in record and "response" not in record
    assert (
        seen[0]["request"] == request
        and seen[0]["request"]["inputs"]["external_image"]["encoding"] == "base64_png"
    )
    persisted = (tmp_path / "calls.jsonl").read_text()
    assert len(persisted.splitlines()) == 1 and json.loads(persisted) == record
    assert (
        TOKEN not in persisted
        and "provider-key-not-for-this-relay" not in json.dumps(seen)
    )
    summary = frs_agent.summarize_calls(client.records)
    assert summary["provider_calls"] == 1 and summary["accepted_proposals"] == 1


@pytest.mark.parametrize(
    "change",
    [
        lambda r: r["proposal"].update(request_fingerprint="b" * 64),
        lambda r: r["proposal"].update(coords=[2, 0, 0]),
        lambda r: r["receipt"].update(configured_model="different-model"),
        lambda r: r["receipt"].update(observed_completed_turns=0),
        lambda r: r["receipt"].pop("status"),
    ],
)
def test_invalid_result_fails_closed_and_preserves_receipt(
    monkeypatch, tmp_path, change
):
    def execute(envelope):
        result = result_for(envelope)
        change(result)
        return result

    client = client_for(monkeypatch, tmp_path, execute)
    with pytest.raises(ClientError):
        client.propose(request_for())
    assert len(client.records) == 1
    record = client.records[0]
    assert record["provider_unavailable"] is True and record["accepted"] is False
    assert record["codex_receipt"] and record["token_usage"]["input_tokens"] == 100


def test_unavailable_and_rejected_jobs_are_distinct(monkeypatch, tmp_path):
    client = client_for(
        monkeypatch, tmp_path, lambda e: result_for(e, unavailable=True)
    )
    with pytest.raises(ClientError, match="unavailable"):
        client.propose(request_for())
    assert client.records[0]["provider_unavailable"] is True

    def rejected(envelope):
        result = result_for(envelope)
        result["proposal"] = None
        result["receipt"].update(accepted=False, status="proposal_rejected")
        return result

    client = client_for(monkeypatch, tmp_path, rejected)
    with pytest.raises(ClientError, match="original request contract"):
        client.propose(request_for())
    assert client.records[0]["provider_unavailable"] is False
    assert client.records[0]["error_kind"] == "proposal_rejected"


def test_missing_token_and_bad_request_retain_sanitized_preflight(
    monkeypatch, tmp_path
):
    monkeypatch.delenv("ASTRA_CODEX_RELAY_TOKEN", raising=False)
    monkeypatch.delenv("ASTRA_CODEX_RELAY_TOKEN_FILE", raising=False)
    client = relay.CodexRelayClient(
        model=relay.MODEL, response_log=tmp_path / "calls.jsonl", family="frs"
    )
    with pytest.raises(ClientError):
        client.propose({"episode_id": "malformed"})
    assert len(client.records) == 1
    record = client.records[0]
    assert (
        record["provider_call"] is False and record["error_kind"] == "preflight_error"
    )
    assert record["token_usage"] == normalize_usage(None)
    assert "http_status" not in record


def test_token_file_and_shared_server(monkeypatch, tmp_path):
    path = tmp_path / "private-token"
    path.write_text(TOKEN + "\n")
    monkeypatch.delenv("ASTRA_CODEX_RELAY_TOKEN", raising=False)
    monkeypatch.setenv("ASTRA_CODEX_RELAY_TOKEN_FILE", str(path))
    assert relay._token() == TOKEN
    instances = []

    class FakeServer:
        def __init__(self, host, port, token):
            self.pid = relay.os.getpid()
            self.args = host, port, token
            instances.append(self)

        def matches(self, *args):
            return args == self.args

    monkeypatch.setattr(relay, "_SERVER", None)
    monkeypatch.setattr(relay, "RelayServer", FakeServer)
    assert relay.ensure_server() is relay.ensure_server()
    assert len(instances) == 1
    with pytest.raises(ClientError, match="different"):
        relay.ensure_server(port=8770)


@pytest.mark.parametrize(
    "path,method",
    [("/health", "do_GET"), ("/pending", "do_GET"), ("/response", "do_POST")],
)
def test_http_rejects_bad_auth_before_mailbox_access(path, method):
    value = handler(SimpleNamespace(), token="wrong-token", path=path)
    getattr(value, method)()
    assert value.responses == [(401, {"error": "unauthorized"})]


def test_http_health_reveals_only_status_and_oversized_body_is_not_read():
    value = handler(relay.RelayMailbox())
    value.do_GET()
    assert value.responses == [
        (200, {"protocol": relay.PROTOCOL_VERSION, "status": "idle"})
    ]
    value = handler(
        relay.RelayMailbox(),
        path="/response",
        body=b"unread",
        size=relay.MAX_RESPONSE_BYTES + 1,
    )
    value.do_POST()
    assert value.responses[0][0] == 413 and value.rfile.tell() == 0


def test_mailbox_rejects_wrong_binding_and_replay():
    mailbox = relay.RelayMailbox()
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(mailbox.exchange, envelope_for(), 2)
        envelope = wait_pending(mailbox)
        wrong = submission(envelope)
        wrong["invocation_id"] = "b" * 32
        with pytest.raises(ClientError, match="invocation_id_mismatch"):
            mailbox.submit(wrong)
        wrong = submission(envelope)
        wrong["result"]["request_fingerprint"] = "b" * 64
        with pytest.raises(ClientError, match="request_fingerprint_mismatch"):
            mailbox.submit(wrong)
        body = submission(envelope)
        value = handler(mailbox, path="/response", body=json.dumps(body).encode())
        value.do_POST()
        assert value.responses[0][0] == 200
        assert future.result(timeout=2) == body["result"]
        mailbox.submit(body)  # Exact replay only acknowledges the consumed result.
        changed = json.loads(json.dumps(body))
        changed["result"]["receipt"]["exit_code"] = 1
        with pytest.raises(ClientError, match="accepted_invocation_result_mismatch"):
            mailbox.submit(changed)


def test_expired_mailbox_does_not_accept_late_result():
    mailbox = relay.RelayMailbox()
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(mailbox.exchange, envelope_for(), 0.04)
        envelope = wait_pending(mailbox)
        with pytest.raises(ClientError, match="relay_response_timeout"):
            future.result(timeout=2)
        assert mailbox.pending() is None
        with pytest.raises(ClientError):
            mailbox.submit(submission(envelope))


def test_client_timeout_records_unknown_job_start_once(monkeypatch, tmp_path):
    client = client_for(monkeypatch, tmp_path)
    mailbox = relay.RelayMailbox()
    monkeypatch.setattr(
        relay, "ensure_server", lambda **kwargs: SimpleNamespace(mailbox=mailbox)
    )
    client.timeout = 0.01
    with pytest.raises(ClientError, match="relay_response_timeout"):
        client.propose(request_for())
    record = client.records[0]
    assert (
        record["job_start_unknown"] is True
        and record["provider_call_count_is_lower_bound"] is True
    )
    assert record["provider_call"] is False and record["provider_unavailable"] is True
    assert len((tmp_path / "calls.jsonl").read_text().splitlines()) == 1
    assert frs_agent.summarize_calls(client.records)["provider_calls"] == 0


def test_representation_binding_is_checked_before_stripping_extras(monkeypatch):
    module = SimpleNamespace(
        _IDENTITY_FIELDS=("request_id", "episode_id", "decision_index")
    )
    module.parse_proposal = lambda value, request: value
    request = {
        "request_id": "a" * 16,
        "episode_id": "case",
        "decision_index": 1,
        "request_fingerprint": "a" * 64,
    }
    bound = {**request, "decision_id": request["request_id"], "mode": "native"}
    assert relay._proposal(bound, request, module, "representation") == {
        "request_id": "a" * 16,
        "mode": "native",
    }
    bound["decision_index"] = True
    with pytest.raises(ClientError, match="identity"):
        relay._proposal(bound, request, module, "representation")


def test_local_relay_posts_unavailable_result_then_stops(monkeypatch, tmp_path):
    from astra_reversal import codex_executor

    envelope = envelope_for()
    envelope["expires_at"] = time.time() + 30
    calls = []
    executions = []

    def http(opener, url, token, path, timeout, body=None):
        calls.append((path, body))
        assert token == TOKEN
        if path == "/health":
            return {"protocol": relay.PROTOCOL_VERSION, "status": "pending"}
        if path == "/pending":
            return envelope
        return {"protocol": relay.PROTOCOL_VERSION, "accepted": True}

    def execute(request, directory, **kwargs):
        executions.append((request, directory, kwargs))
        return result_for(envelope, unavailable=True)

    monkeypatch.setattr(relay, "_http", http)
    monkeypatch.setattr(codex_executor, "execute_request", execute)
    with pytest.raises(ClientError, match="stop receipt was delivered"):
        relay.run_relay("http://127.0.0.1:8769", TOKEN, tmp_path, max_requests=3)
    assert [path for path, _ in calls] == ["/health", "/pending", "/response"]
    assert (
        len(executions) == 1
        and executions[0][1] == tmp_path / envelope["invocation_id"]
    )
    assert TOKEN not in json.dumps(calls[-1][1])


def test_local_relay_rejects_path_invocation_ids():
    envelope = envelope_for()
    envelope["invocation_id"] = "../../elsewhere"
    with pytest.raises(ClientError, match="invocation ID"):
        relay._pending_request(envelope)


@pytest.mark.parametrize("path", ["/health", "/pending"])
@pytest.mark.parametrize("status", [502, 503, 504])
def test_gateway_poll_retry_runs_executor_once(monkeypatch, tmp_path, path, status):
    from astra_reversal import codex_executor

    envelope = envelope_for()
    envelope["expires_at"] = time.time() + 30
    attempts, executions, errors = [], [], []

    def http(opener, url, token, endpoint, timeout, body=None):
        assert token == TOKEN
        attempts.append(endpoint)
        if endpoint == path and attempts.count(path) == 1:
            error = relay.urllib.error.HTTPError(
                url, status, "gateway", {}, io.BytesIO(b"private gateway details")
            )
            errors.append(error)
            raise error
        if endpoint == "/health":
            return {"protocol": relay.PROTOCOL_VERSION, "status": "pending"}
        if endpoint == "/pending":
            return envelope
        return {"protocol": relay.PROTOCOL_VERSION, "accepted": True}

    def execute(*args, **kwargs):
        executions.append(args)
        return result_for(envelope)

    monkeypatch.setattr(relay, "_http", http)
    monkeypatch.setattr(codex_executor, "execute_request", execute)
    assert (
        relay.run_relay(
            "http://127.0.0.1:8769",
            TOKEN,
            tmp_path,
            max_requests=1,
            poll_interval=0.001,
        )
        == 1
    )
    assert attempts.count(path) == 2 and len(executions) == 1
    assert errors[0].fp.closed


@pytest.mark.parametrize("status", [502, 503, 504])
@pytest.mark.parametrize("first_delivery_reached_worker", [False, True])
def test_gateway_ambiguous_post_reuses_bound_result_without_execution(
    monkeypatch, tmp_path, status, first_delivery_reached_worker
):
    from astra_reversal import codex_executor

    envelope = envelope_for()
    envelope["expires_at"] = time.time() + 30
    executions, posted = [], []
    mailbox = relay.RelayMailbox()
    with ThreadPoolExecutor(max_workers=1) as pool:
        waiting = pool.submit(mailbox.exchange, envelope, 10)
        active = wait_pending(mailbox)

        def http(opener, url, token, endpoint, timeout, body=None):
            assert token == TOKEN
            if endpoint == "/health":
                return {"protocol": relay.PROTOCOL_VERSION, "status": "pending"}
            if endpoint == "/pending":
                return active
            posted.append(body)
            if len(posted) == 1:
                if first_delivery_reached_worker:
                    mailbox.submit(body)
                    waiting.result(timeout=2)  # ACK lost after worker consumed it.
                raise relay.urllib.error.HTTPError(
                    url, status, "gateway", {}, io.BytesIO()
                )
            try:
                mailbox.submit(body)
            except relay._MailboxError as exc:
                raise relay.urllib.error.HTTPError(
                    url, exc.status, "rejected", {}, io.BytesIO()
                ) from None
            return {"protocol": relay.PROTOCOL_VERSION, "accepted": True}

        def execute(*args, **kwargs):
            executions.append(args)
            return result_for(active)

        monkeypatch.setattr(relay, "_http", http)
        monkeypatch.setattr(codex_executor, "execute_request", execute)

        def run():
            return relay.run_relay(
                "http://127.0.0.1:8769",
                TOKEN,
                tmp_path,
                max_requests=1,
                poll_interval=0.001,
            )

        assert run() == 1
        assert waiting.result(timeout=2) == result_for(active)
    assert len(executions) == 1 and len(posted) == 2
    assert posted[0] is posted[1]
    assert posted[0]["invocation_id"] == active["invocation_id"]
    assert posted[0]["request_fingerprint"] == active["request_fingerprint"]


@pytest.mark.parametrize("status", [401, 403, 409, 410, 500])
@pytest.mark.parametrize("path", ["/health", "/pending", "/response"])
def test_non_gateway_errors_remain_fatal(monkeypatch, tmp_path, status, path):
    from astra_reversal import codex_executor

    envelope = envelope_for()
    envelope["expires_at"] = time.time() + 30
    calls, executions = [], []

    def http(opener, url, token, endpoint, timeout, body=None):
        calls.append(endpoint)
        if endpoint == path:
            raise relay.urllib.error.HTTPError(
                url, status, TOKEN, {}, io.BytesIO(TOKEN.encode())
            )
        if endpoint == "/health":
            return {"protocol": relay.PROTOCOL_VERSION, "status": "pending"}
        if endpoint == "/pending":
            return envelope
        pytest.fail("Unexpected successful result delivery")

    def execute(*args, **kwargs):
        executions.append(args)
        return result_for(envelope)

    monkeypatch.setattr(relay, "_http", http)
    monkeypatch.setattr(codex_executor, "execute_request", execute)
    with pytest.raises(ClientError, match=f"HTTP {status}") as error:
        relay.run_relay(
            "http://127.0.0.1:8769",
            TOKEN,
            tmp_path,
            max_requests=1,
            poll_interval=0.001,
        )
    assert calls.count(path) == 1
    assert len(executions) == (1 if path == "/response" else 0)
    assert TOKEN not in str(error.value)


def test_ack_replay_cannot_change_next_pending_invocation():
    mailbox = relay.RelayMailbox()
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(mailbox.exchange, envelope_for(), 2)
        old = wait_pending(mailbox)
        body = submission(old)
        mailbox.submit(body)
        assert first.result(timeout=2) == body["result"]
        fresh = envelope_for()
        fresh["invocation_id"] = "c" * 32
        second = pool.submit(mailbox.exchange, fresh, 2)
        pending = wait_pending(mailbox)
        # Canonical digest permits object key reordering, but no value mutation.
        reordered = dict(reversed(list(body.items())))
        mailbox.submit(reordered)
        assert mailbox.pending() == pending and not second.done()
        changed = json.loads(json.dumps(body))
        changed["request_fingerprint"] = "d" * 64
        with pytest.raises(ClientError, match="result_mismatch"):
            mailbox.submit(changed)
        assert mailbox.pending() == pending and not second.done()
        mailbox.submit(submission(pending))
        assert second.result(timeout=2) == result_for(pending)
    with pytest.raises(ClientError, match="accepted_invocation_id_reused"):
        mailbox.exchange(old, 0.01)


def test_ack_cache_is_bounded_and_contains_only_digests():
    mailbox = relay.RelayMailbox()
    original = envelope_for()
    with ThreadPoolExecutor(max_workers=1) as pool:
        for index in range(257):
            envelope = {**original, "invocation_id": f"{index:032x}"}
            future = pool.submit(mailbox.exchange, envelope, 2)
            active = wait_pending(mailbox)
            mailbox.submit(submission(active))
            future.result(timeout=2)
    assert len(mailbox._acknowledged) == 256
    assert "0" * 32 not in mailbox._acknowledged
    assert all(
        isinstance(value, str) and len(value) == 64
        for value in mailbox._acknowledged.values()
    )
    with pytest.raises(ClientError, match="no_pending_invocation"):
        mailbox.submit(submission({**original, "invocation_id": "0" * 32}))


def wire_response(value, *, truncate=False, declared=None):
    """Real HTTPResponse framing over a hermetic in-memory socket."""
    body = relay._json_bytes(value)
    length = str(len(body)) if declared is None else declared
    wire = (
        b"HTTP/1.1 200 OK\r\nContent-Length: "
        + length.encode()
        + b"\r\n\r\n"
        + (body[:10] if truncate else body)
    )
    response = relay.http.client.HTTPResponse(
        SimpleNamespace(makefile=lambda *args: io.BytesIO(wire))
    )
    response.begin()
    return response


@pytest.mark.parametrize("endpoint", ["/health", "/pending", "/response"])
def test_truncated_framing_retries_transport_not_executor(
    monkeypatch, tmp_path, endpoint
):
    from astra_reversal import codex_executor

    mailbox = relay.RelayMailbox()
    executions, posts, attempts = [], [], []
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(mailbox.exchange, envelope_for(), 10)
        active = wait_pending(mailbox)

        def open_request(request, timeout):
            path = relay.urllib.parse.urlsplit(request.full_url).path
            attempts.append(path)
            if path == "/health":
                value = {"protocol": relay.PROTOCOL_VERSION, "status": "pending"}
            elif path == "/pending":
                value = active
            else:
                posts.append(request.data)
                mailbox.submit(json.loads(request.data))
                future.result(timeout=2)  # Worker consumes before the ACK is lost.
                value = {"protocol": relay.PROTOCOL_VERSION, "accepted": True}
            return wire_response(
                value, truncate=path == endpoint and attempts.count(path) == 1
            )

        def execute(*args, **kwargs):
            executions.append(args)
            return result_for(active)

        monkeypatch.setattr(
            relay.urllib.request,
            "build_opener",
            lambda *args: SimpleNamespace(open=open_request),
        )
        monkeypatch.setattr(codex_executor, "execute_request", execute)
        assert (
            relay.run_relay(
                "http://127.0.0.1:8769",
                TOKEN,
                tmp_path,
                max_requests=1,
                poll_interval=0.001,
            )
            == 1
        )
        assert future.result(timeout=2) == result_for(active)
    assert len(executions) == 1
    assert attempts.count(endpoint) == 2
    if endpoint == "/response":
        assert len(posts) == 2 and posts[0] == posts[1]
        assert len(mailbox._acknowledged) == 1


@pytest.mark.parametrize("kind", ["malformed_json", "oversized", "invalid_length"])
def test_complete_response_contract_errors_remain_fatal(kind):
    response = wire_response({"private": TOKEN})
    if kind == "malformed_json":
        response = wire_response({"private": TOKEN}, truncate=True, declared="10")
    elif kind == "oversized":
        response = wire_response({}, declared=str(relay.MAX_REQUEST_BYTES + 1))
    else:
        response = wire_response({}, declared="invalid")
    opener = SimpleNamespace(open=lambda *args, **kwargs: response)
    with pytest.raises(ClientError) as error:
        relay._http(opener, "http://127.0.0.1", TOKEN, "/pending", 1)
    assert TOKEN not in str(error.value)


def test_cli_diagnostic_never_prints_exception_secrets(monkeypatch, tmp_path, capsys):
    token_file = tmp_path / "token"
    token_file.write_text(TOKEN)

    def fail(*args, **kwargs):
        raise RuntimeError(TOKEN + " private response payload")

    monkeypatch.setattr(relay, "run_relay", fail)
    with pytest.raises(SystemExit) as error:
        relay.main(
            ["--directory", str(tmp_path / "jobs"), "--token-file", str(token_file)]
        )
    assert error.value.code == 1
    diagnostic = capsys.readouterr().err
    assert json.loads(diagnostic) == {
        "status": "relay_stopped",
        "error_kind": "unexpected_error",
    }
    assert TOKEN not in diagnostic and "payload" not in diagnostic
