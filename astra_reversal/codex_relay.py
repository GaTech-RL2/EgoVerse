"""Authenticated worker mailbox for locally authenticated, one-shot Codex jobs.

The worker exposes only health, a validated pending request, and bound response
submission. It cannot execute commands or read arbitrary files. The local relay
keeps Codex authentication and durable job artifacts off the GPU worker.
"""

import argparse
import hashlib
import hmac
import http.client
import json
import math
import os
import re
import secrets
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .astra_client import (
    ClientError,
    _append_record,
    _NoRedirect,
    _redact,
    _strict_json,
)
from .intervention_agent import normalize_usage

PROTOCOL_VERSION = "astra-codex-relay-1"
MODEL = "gpt-6-astra"
REASONING_EFFORT = "medium"
DEFAULT_PORT = 8769
MAX_RESPONSE_BYTES = 4 * 1024 * 1024
MAX_REQUEST_BYTES = 128 * 1024 * 1024
_SERVER = None
_SERVER_LOCK = threading.Lock()


def _json_bytes(value):
    return json.dumps(value, allow_nan=False, separators=(",", ":")).encode()


def _copy(value):
    return _strict_json(_json_bytes(value))


def _module(family):
    if family == "frs":
        from . import frs_agent

        return frs_agent
    if family == "representation":
        from . import representation_agent

        return representation_agent
    raise ClientError("Unknown Codex relay request family")


def _positive(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ClientError(f"{name} must be positive and finite")
    return value


def _token(value=None):
    if value is None:
        value = os.environ.get("ASTRA_CODEX_RELAY_TOKEN", "").strip()
        if not value and os.environ.get("ASTRA_CODEX_RELAY_TOKEN_FILE"):
            try:
                with Path(os.environ["ASTRA_CODEX_RELAY_TOKEN_FILE"]).open() as stream:
                    value = stream.read(4097).strip()
            except (OSError, UnicodeError):
                raise ClientError("Codex relay token file is unavailable") from None
    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= 4096
        or any(character.isspace() or ord(character) < 33 for character in value)
    ):
        raise ClientError(
            "ASTRA_CODEX_RELAY_TOKEN or its private token file is required"
        )
    return value


def _address(host=None, port=None):
    host = (
        host
        if host is not None
        else os.environ.get("ASTRA_CODEX_RELAY_HOST", "127.0.0.1")
    )
    if (
        not isinstance(host, str)
        or not host
        or any(character.isspace() for character in host)
    ):
        raise ClientError("Codex relay host must be nonempty")
    try:
        port = (
            int(os.environ.get("ASTRA_CODEX_RELAY_PORT", DEFAULT_PORT))
            if port is None
            else port
        )
    except (TypeError, ValueError):
        raise ClientError("Codex relay port must be an integer") from None
    if type(port) is not int or not 0 <= port <= 65535:
        raise ClientError("Codex relay port is outside its allowed range")
    return host, port


class _MailboxError(ClientError):
    def __init__(self, reason, status=409):
        self.reason, self.status = reason, status
        super().__init__(reason)


class RelayMailbox:
    """One pending invocation, with a deadline covering queueing and execution."""

    def __init__(self):
        self._serial = threading.Lock()
        self._condition = threading.Condition()
        self._pending = None
        # Bounded ACKs contain only IDs and full-submission digests, never payloads.
        self._acknowledged = {}

    def exchange(self, envelope, timeout):
        deadline = time.monotonic() + _positive(timeout, "Relay timeout")
        if not self._serial.acquire(timeout=timeout):
            raise _MailboxError("relay_queue_timeout", 410)
        try:
            with self._condition:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise _MailboxError("relay_queue_timeout", 410)
                envelope = _copy(envelope)
                if envelope["invocation_id"] in self._acknowledged:
                    raise _MailboxError("accepted_invocation_id_reused")
                envelope["expires_at"] = time.time() + remaining
                self._pending = {
                    "envelope": envelope,
                    "deadline": deadline,
                    "result": None,
                }
                self._condition.notify_all()
                while self._pending["result"] is None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise _MailboxError("relay_response_timeout", 410)
                    self._condition.wait(timeout=remaining)
                return _copy(self._pending["result"])
        finally:
            with self._condition:
                self._pending = None
                self._condition.notify_all()
            self._serial.release()

    def pending(self):
        with self._condition:
            item = self._pending
            if (
                item is None
                or item["result"] is not None
                or time.monotonic() >= item["deadline"]
            ):
                return None
            return _copy(item["envelope"])

    def submit(self, value):
        if not isinstance(value, dict) or set(value) != {
            "invocation_id",
            "request_fingerprint",
            "result",
        }:
            raise _MailboxError("invalid_response_envelope", 400)
        result = value["result"]
        if not isinstance(result, dict) or set(result) != {
            "request_fingerprint",
            "proposal",
            "receipt",
        }:
            raise _MailboxError("invalid_result_envelope", 400)
        invocation_id = value["invocation_id"]
        if (
            not isinstance(invocation_id, str)
            or re.fullmatch(r"[0-9a-f]{32}", invocation_id) is None
        ):
            raise _MailboxError("invalid_invocation_id", 400)
        try:
            submission_digest = hashlib.sha256(
                json.dumps(
                    value, sort_keys=True, separators=(",", ":"), allow_nan=False
                ).encode()
            ).hexdigest()
        except (ValueError, TypeError):
            raise _MailboxError("invalid_result_json", 400) from None
        with self._condition:
            accepted_digest = self._acknowledged.get(invocation_id)
            if accepted_digest is not None:
                if not hmac.compare_digest(accepted_digest, submission_digest):
                    raise _MailboxError("accepted_invocation_result_mismatch")
                return  # Lost ACK: do not touch a newer pending invocation.
            item = self._pending
            if item is None:
                raise _MailboxError("no_pending_invocation")
            if time.monotonic() >= item["deadline"]:
                raise _MailboxError("expired_invocation", 410)
            if item["result"] is not None:
                raise _MailboxError("invocation_already_completed")
            expected = item["envelope"]
            if value["invocation_id"] != expected["invocation_id"]:
                raise _MailboxError("invocation_id_mismatch")
            if any(
                value != expected["request_fingerprint"]
                for value in (
                    value["request_fingerprint"],
                    result["request_fingerprint"],
                )
            ):
                raise _MailboxError("request_fingerprint_mismatch")
            self._pending["result"] = _copy(result)
            self._acknowledged[invocation_id] = submission_digest
            if len(self._acknowledged) > 256:
                del self._acknowledged[next(iter(self._acknowledged))]
            self._condition.notify_all()


class RelayHandler(BaseHTTPRequestHandler):
    """No request headers, tokens, proposals, or HTTP access logs are printed."""

    def setup(self):
        super().setup()
        self.connection.settimeout(5)

    def log_message(self, *_args):
        pass

    def _send(self, status, value):
        body = _json_bytes(value)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self):
        actual = self.headers.get("Authorization", "").encode()
        expected = ("Bearer " + self.server.relay_token).encode()
        if not hmac.compare_digest(actual, expected):
            self._send(401, {"error": "unauthorized"})
            return False
        return True

    def do_GET(self):
        if not self._authorized():
            return
        pending = self.server.mailbox.pending()
        if self.path == "/health":
            self._send(
                200,
                {
                    "protocol": PROTOCOL_VERSION,
                    "status": "pending" if pending else "idle",
                },
            )
        elif self.path == "/pending":
            self._send(200, pending or {"protocol": PROTOCOL_VERSION, "pending": None})
        else:
            self._send(404, {"error": "not_found"})

    def do_POST(self):
        if not self._authorized():
            return
        if self.path != "/response":
            self._send(404, {"error": "not_found"})
            return
        size = self.headers.get("Content-Length", "")
        if (
            self.headers.get("Transfer-Encoding")
            or len(size) > 10
            or not size.isdecimal()
        ):
            self._send(400, {"error": "content_length_required"})
            return
        size = int(size)
        if not 0 < size <= MAX_RESPONSE_BYTES:
            self._send(413, {"error": "response_too_large"})
            return
        if self.headers.get("Content-Type", "").split(";", 1)[0] != "application/json":
            self._send(415, {"error": "json_required"})
            return
        try:
            body = self.rfile.read(size)
            if len(body) != size:
                raise ClientError("incomplete_body")
            self.server.mailbox.submit(_strict_json(body))
        except _MailboxError as exc:
            self._send(exc.status, {"error": exc.reason})
        except (ValueError, TypeError, OSError, OverflowError, RecursionError):
            self._send(400, {"error": "invalid_json_response"})
        else:
            self._send(200, {"protocol": PROTOCOL_VERSION, "accepted": True})


class RelayServer:
    def __init__(self, host, port, token):
        server_type = type(
            "MailboxHTTPServer",
            (ThreadingHTTPServer,),
            {
                "address_family": socket.AF_INET6 if ":" in host else socket.AF_INET,
                "daemon_threads": True,
            },
        )
        self._http = server_type((host, port), RelayHandler)
        self._http.relay_token = token
        self.mailbox = self._http.mailbox = RelayMailbox()
        self.host, self.requested_port, self.pid = host, port, os.getpid()
        self.port = self._http.server_address[1]
        visible_host = "127.0.0.1" if host == "0.0.0.0" else host
        visible_host = f"[{visible_host}]" if ":" in visible_host else visible_host
        self.url = f"http://{visible_host}:{self.port}"
        self._thread = threading.Thread(
            target=self._http.serve_forever, daemon=True, name="astra-codex-mailbox"
        )
        self._thread.start()

    def matches(self, host, port, token):
        return (self.host, self.requested_port, self.pid) == (
            host,
            port,
            os.getpid(),
        ) and hmac.compare_digest(self._http.relay_token.encode(), token.encode())

    def close(self):
        self._http.shutdown()
        self._http.server_close()
        self._thread.join(timeout=5)


def ensure_server(*, host=None, port=None, token=None):
    """Start the one process-wide worker mailbox, including before GPU setup."""
    global _SERVER
    host, port = _address(host, port)
    token = _token(token)
    with _SERVER_LOCK:
        if _SERVER is None or _SERVER.pid != os.getpid():
            _SERVER = RelayServer(host, port, token)
        elif not _SERVER.matches(host, port, token):
            raise ClientError(
                "A different Codex relay server is already configured in this process"
            )
        return _SERVER


def _receipt(value, envelope):
    if not isinstance(value, dict):
        raise ClientError("Codex receipt must be an object")
    expected = {
        "backend": "codex_exec",
        "configured_model": envelope["model"],
        "request_fingerprint": envelope["request_fingerprint"],
        "reasoning_effort": envelope["reasoning_effort"],
    }
    if any(
        value.get(key) != expected_value for key, expected_value in expected.items()
    ):
        raise ClientError("Codex receipt does not match the configured invocation")
    if "requested_model" in value and value["requested_model"] != envelope["model"]:
        raise ClientError("Codex receipt requested a different model")
    if value.get("returned_model") not in (None, envelope["model"]):
        raise ClientError("Codex reported a different served model")
    if any(
        type(value.get(key)) is not bool
        for key in ("provider_call", "accepted", "provider_unavailable")
    ):
        raise ClientError(
            "Codex receipt is missing explicit job and availability status"
        )
    status = value.get("status")
    if status not in ("accepted", "proposal_rejected", "provider_unavailable") or (
        value["accepted"] != (status == "accepted")
        or value["provider_unavailable"] != (status == "provider_unavailable")
    ):
        raise ClientError("Codex receipt has inconsistent completion status")
    if status != "provider_unavailable" and (
        not value["provider_call"]
        or type(value.get("observed_completed_turns")) is not int
        or value["observed_completed_turns"] != 1
        or type(value.get("exit_code")) is not int
        or value["exit_code"] != 0
        or value.get("tool_items")
    ):
        raise ClientError("Codex receipt does not establish one completed job")
    if value.get("raw_usage") is not None and not isinstance(value["raw_usage"], dict):
        raise ClientError("Codex usage must be an object or unknown")
    if not value["provider_call"] and value.get("raw_usage") is not None:
        raise ClientError("Codex receipt claims usage without a job")
    return value


def _proposal(value, request, module, family):
    if not isinstance(value, dict):
        raise ClientError("Codex proposal must be an object")
    bound_fields = {"request_fingerprint": request["request_fingerprint"]}
    if family == "representation":
        bound_fields.update(
            {
                key: request[key]
                for key in module._IDENTITY_FIELDS
                if key != "request_id"
            }
        )
        bound_fields["decision_id"] = request["request_id"]
    if any(
        key in value
        and (type(value[key]) is not type(expected) or value[key] != expected)
        for key, expected in bound_fields.items()
    ):
        raise ClientError("Codex proposal identity differs from its original request")
    return module.parse_proposal(
        {key: item for key, item in value.items() if key not in bound_fields}, request
    )


class CodexRelayClient:
    """FRS/representation-compatible client; exactly one retained row per call."""

    def __init__(
        self,
        *,
        model,
        response_log,
        family,
        endpoint=None,
        timeout=300.0,
        reasoning_effort="medium",
        max_completion_tokens=None,
        sampling=None,
        host=None,
        port=None,
        token=None,
    ):
        self.module = _module(family)
        if model != MODEL or reasoning_effort != REASONING_EFFORT:
            raise ClientError("Codex relay requires gpt-6-astra with medium reasoning")
        if not response_log:
            raise ClientError("A response_log path is required for Codex provenance")
        if max_completion_tokens is not None or (
            sampling is not None
            and (
                not isinstance(sampling, dict)
                or any(
                    key not in ("reasoning_effort", "max_completion_tokens")
                    or value
                    != {
                        "reasoning_effort": reasoning_effort,
                        "max_completion_tokens": None,
                    }[key]
                    for key, value in sampling.items()
                )
            )
        ):
            raise ClientError(
                "Codex relay does not support HTTP sampling or a completion-token cap"
            )
        self.family, self.model, self.reasoning_effort = family, model, reasoning_effort
        self.timeout = _positive(timeout, "Codex timeout")
        self.host, self.port = _address(host, port)
        self._token = token
        self.response_log, self.records = response_log, []
        self.endpoint = "codex_relay"

    def ensure_server(self):
        return ensure_server(host=self.host, port=self.port, token=self._token)

    def propose(self, request):
        started, monotonic = time.time(), time.perf_counter()
        record = {
            "client_schema_version": self.module.SCHEMA_VERSION,
            "prompt_template_version": self.module.PROMPT_TEMPLATE_VERSION,
            "backend": "codex_relay",
            "endpoint": self.endpoint,
            "requested_model": self.model,
            "configured_model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "provider_call": False,
            "provider_call_kind": "codex_exec",
            "provider_call_semantics": "CLI job started; not a raw HTTP request count",
            "provider_unavailable": False,
            "accepted": False,
            "job_start_unknown": False,
            "provider_call_count_is_lower_bound": False,
            "request_time": started,
            "token_usage": normalize_usage(None),
        }
        token = ""
        if isinstance(request, dict):
            record.update(
                {
                    key: request[key]
                    for key in (*self.module._IDENTITY_FIELDS, "request_fingerprint")
                    if key in request and type(request[key]) in (str, int)
                }
            )
        try:
            token = _token(self._token)
            self.module._validate_request(request)
            request = _copy(request)
            server = ensure_server(host=self.host, port=self.port, token=token)
            invocation = secrets.token_hex(16)
            envelope = {
                "protocol": PROTOCOL_VERSION,
                "invocation_id": invocation,
                "request_fingerprint": request["request_fingerprint"],
                "family": self.family,
                "request": request,
                "model": self.model,
                "reasoning_effort": self.reasoning_effort,
                "timeout": self.timeout,
            }
            body = _json_bytes(envelope)
            if len(body) > MAX_REQUEST_BYTES:
                raise ClientError("Codex relay request exceeds its size limit")
            system = (
                self.module.SYSTEM_PROMPTS[request["role"]]
                if self.family == "frs"
                else self.module.SYSTEM_PROMPT
            )
            record.update(
                invocation_id=invocation,
                relay_payload_sha256=hashlib.sha256(body).hexdigest(),
                system_prompt_sha256=hashlib.sha256(system.encode()).hexdigest(),
            )
            result = server.mailbox.exchange(envelope, self.timeout)
            record["codex_receipt"] = result.get("receipt")
            receipt = result.get("receipt")
            if (
                isinstance(receipt, dict)
                and receipt.get("backend") == "codex_exec"
                and receipt.get("provider_call") is True
            ):
                from .codex_accounting import normalize_receipt_usage

                record["provider_call"] = True
                record["raw_usage"] = receipt.get("raw_usage")
                record["raw_usage_events"] = receipt.get("raw_usage_events")
                record["token_usage"], record["token_usage_is_lower_bound"] = (
                    normalize_receipt_usage(receipt)
                )
            receipt = _receipt(receipt, envelope)
            if result.get("request_fingerprint") != request["request_fingerprint"]:
                raise ClientError("Codex result has a different request fingerprint")
            if receipt["provider_unavailable"]:
                record["availability_reason"] = receipt.get(
                    "availability_reason", "codex_job_unavailable"
                )
                raise _MailboxError("codex_provider_unavailable")
            if not receipt["accepted"]:
                if result.get("proposal") is not None:
                    raise ClientError(
                        "Rejected Codex result unexpectedly contains a proposal"
                    )
                record.update(
                    error_kind="proposal_rejected",
                    error="Completed Codex proposal failed the original request contract",
                )
                raise ClientError(record["error"])
            proposal = _proposal(
                result.get("proposal"), request, self.module, self.family
            )
            record.update(accepted=True)
            record["response_id" if self.family == "frs" else "decision_id"] = proposal[
                "response_id" if self.family == "frs" else "decision_id"
            ]
            return proposal
        except _MailboxError as exc:
            record.update(
                provider_unavailable=True,
                availability_reason=record.get("availability_reason", exc.reason),
                error_kind="provider_unavailable"
                if record["provider_call"]
                else "preflight_error",
                error=f"Codex relay unavailable ({exc.reason})",
            )
            if exc.reason == "relay_response_timeout":
                record.update(
                    job_start_unknown=True,
                    provider_call_count_is_lower_bound=True,
                    error_kind="transport_error",
                )
            raise ClientError(record["error"]) from None
        except (
            OSError,
            TimeoutError,
            urllib.error.URLError,
            http.client.HTTPException,
        ) as exc:
            record.update(
                provider_unavailable=True,
                availability_reason="relay_transport_failed",
                error_kind="provider_unavailable"
                if record["provider_call"]
                else "preflight_error",
                error=f"Codex relay transport failed ({type(exc).__name__})",
            )
            raise ClientError(record["error"]) from None
        except (ValueError, TypeError, KeyError, OverflowError, RecursionError) as exc:
            if record.get("error_kind") != "proposal_rejected":
                record.update(
                    provider_unavailable=True,
                    availability_reason="relay_contract_failed",
                    error_kind="provider_unavailable"
                    if record["provider_call"]
                    else "preflight_error",
                    error=_redact(
                        f"Codex relay rejected invalid evidence ({type(exc).__name__})",
                        token,
                    ),
                )
            raise ClientError(record["error"]) from None
        finally:
            record.update(
                response_time=time.time(),
                latency_seconds=time.perf_counter() - monotonic,
            )
            safe_record = _redact(record, token)
            self.records.append(safe_record)
            _append_record(self.response_log, safe_record)


def _relay_url(url):
    parsed = urllib.parse.urlsplit(url)
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path not in ("", "/")
    ):
        raise ClientError("Relay URL must be a credential-free HTTP(S) origin")
    if parsed.scheme == "http" and parsed.hostname not in (
        "localhost",
        "127.0.0.1",
        "::1",
    ):
        raise ClientError("Use a local port-forward or HTTPS for the Codex relay")
    return url.rstrip("/")


def _http(opener, url, token, path, timeout, body=None):
    request = urllib.request.Request(
        url + path,
        data=None if body is None else _json_bytes(body),
        headers={
            "Authorization": "Bearer " + token,
            "Content-Type": "application/json",
        },
        method="GET" if body is None else "POST",
    )
    with opener.open(request, timeout=timeout) as response:
        declared = response.headers.get("Content-Length")
        if declared is not None:
            if not re.fullmatch(r"[0-9]+", declared):
                raise ClientError("Worker relay Content-Length is invalid")
            declared = int(declared)
            if declared > MAX_REQUEST_BYTES:
                raise ClientError("Worker relay response exceeds its size limit")
        raw = response.read(MAX_REQUEST_BYTES + 1)
        if len(raw) > MAX_REQUEST_BYTES:
            raise ClientError("Worker relay response exceeds its size limit")
        if declared is not None and len(raw) < declared:
            # HTTPResponse.read(amt) permits premature EOF without raising.
            # Treat framing truncation as transport failure, not invalid JSON.
            # Never attach the private partial response to the exception.
            raise http.client.IncompleteRead(b"", declared - len(raw))
        return _strict_json(raw)


def _pending_request(value):
    if not isinstance(value, dict) or value.get("protocol") != PROTOCOL_VERSION:
        raise ClientError("Worker relay protocol is incompatible")
    if value.get("pending", False) is None:
        return None
    if (
        not isinstance(value.get("invocation_id"), str)
        or re.fullmatch(r"[0-9a-f]{32}", value["invocation_id"]) is None
    ):
        raise ClientError("Worker relay invocation ID is invalid")
    if value.get("model") != MODEL or value.get("reasoning_effort") != REASONING_EFFORT:
        raise ClientError(
            "Worker requested an unsupported Codex model or reasoning effort"
        )
    _positive(value.get("timeout"), "Worker job timeout")
    _positive(value.get("expires_at"), "Worker job expiry")
    module = _module(value.get("family"))
    request = value.get("request")
    try:
        module._validate_request(request)
    except (ValueError, TypeError, KeyError, OverflowError):
        raise ClientError("Worker relay request failed its original contract") from None
    if value.get("request_fingerprint") != request["request_fingerprint"]:
        raise ClientError("Worker relay request fingerprint differs from its envelope")
    return value


def run_relay(
    url,
    token,
    directory,
    *,
    max_requests=None,
    poll_interval=0.5,
    connection_timeout=5.0,
    idle_timeout=None,
    stop_event=None,
):
    """Poll a port-forward and execute each invocation once in its own directory.

    Connection polling and response delivery never relaunch a Codex job. Existing
    durable executor directories resolve any repeated pending invocation.
    Gateway 502/503/504 retries affect only relay transport, never the executor
    or the ChatGPT provider. Result delivery repeats the same bound submission.
    """
    from .codex_executor import execute_request

    url, token = _relay_url(url), _token(token)
    _positive(connection_timeout, "Connection timeout")
    _positive(poll_interval, "Poll interval")
    if max_requests is not None and (
        type(max_requests) is not int or max_requests <= 0
    ):
        raise ClientError("max_requests must be a positive integer")
    if idle_timeout is not None:
        _positive(idle_timeout, "Idle timeout")
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    stop_event = stop_event or threading.Event()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
    completed, last_activity = 0, time.monotonic()
    healthy = False
    while not stop_event.is_set() and (
        max_requests is None or completed < max_requests
    ):
        if (
            idle_timeout is not None
            and time.monotonic() - last_activity >= idle_timeout
        ):
            raise ClientError("Codex relay idle timeout")
        try:
            if not healthy:
                health = _http(opener, url, token, "/health", connection_timeout)
                if (
                    not isinstance(health, dict)
                    or health.get("protocol") != PROTOCOL_VERSION
                    or health.get("status") not in ("idle", "pending")
                ):
                    raise ClientError("Worker relay health protocol is incompatible")
                healthy = True
            pending = _pending_request(
                _http(opener, url, token, "/pending", connection_timeout)
            )
        except urllib.error.HTTPError as exc:
            status = exc.code
            exc.close()
            if status in (502, 503, 504):
                healthy = False
                stop_event.wait(poll_interval)
                continue
            raise ClientError(
                f"Worker relay rejected authentication or polling (HTTP {status})"
            ) from None
        except (urllib.error.URLError, OSError, http.client.HTTPException):
            healthy = False
            stop_event.wait(poll_interval)
            continue
        if pending is None:
            stop_event.wait(poll_interval)
            continue
        remaining = pending["expires_at"] - time.time() - 2.0
        if remaining <= 0:
            raise ClientError(
                "Worker invocation expired before a Codex job could start"
            )
        try:
            result = execute_request(
                pending["request"],
                directory / pending["invocation_id"],
                model=pending["model"],
                reasoning_effort=pending["reasoning_effort"],
                timeout=min(pending["timeout"], remaining),
            )
        except (OSError, ValueError, TypeError, KeyError, OverflowError):
            result = {
                "request_fingerprint": pending["request_fingerprint"],
                "proposal": None,
                "receipt": {
                    "backend": "codex_exec",
                    "configured_model": pending["model"],
                    "reasoning_effort": pending["reasoning_effort"],
                    "request_fingerprint": pending["request_fingerprint"],
                    "status": "provider_unavailable",
                    "provider_call": False,
                    "accepted": False,
                    "provider_unavailable": True,
                    "availability_reason": "local_executor_failed",
                    "raw_usage": None,
                },
            }
        submission = {
            "invocation_id": pending["invocation_id"],
            "request_fingerprint": pending["request_fingerprint"],
            "result": result,
        }
        while True:
            remaining = pending["expires_at"] - time.time()
            if remaining <= 0:
                raise ClientError("Worker invocation expired before result delivery")
            try:
                response = _http(
                    opener,
                    url,
                    token,
                    "/response",
                    min(connection_timeout, remaining),
                    submission,
                )
                if (
                    not isinstance(response, dict)
                    or response.get("protocol") != PROTOCOL_VERSION
                    or response.get("accepted") is not True
                ):
                    raise ClientError(
                        "Worker relay did not acknowledge its bound result"
                    )
                break
            except urllib.error.HTTPError as exc:
                status = exc.code
                exc.close()
                if status in (502, 503, 504):
                    stop_event.wait(min(poll_interval, max(0, remaining)))
                    continue
                raise ClientError(
                    f"Worker relay rejected result delivery (HTTP {status})"
                ) from None
            except (urllib.error.URLError, OSError, http.client.HTTPException):
                stop_event.wait(min(poll_interval, max(0, remaining)))
        completed += 1
        last_activity = time.monotonic()
        if result.get("receipt", {}).get("provider_unavailable"):
            raise ClientError(
                "Codex provider became unavailable; its stop receipt was delivered"
            )
    return completed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=f"http://127.0.0.1:{DEFAULT_PORT}")
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--token-file", type=Path)
    parser.add_argument("--max-requests", type=int)
    parser.add_argument("--idle-timeout", type=float)
    parser.add_argument("--connection-timeout", type=float, default=5.0)
    parser.add_argument("--poll-interval", type=float, default=0.5)
    args = parser.parse_args(argv)
    try:
        token = (
            _token()
            if args.token_file is None
            else _token(args.token_file.read_text().strip())
        )
        run_relay(
            args.url,
            token,
            args.directory,
            max_requests=args.max_requests,
            idle_timeout=args.idle_timeout,
            connection_timeout=args.connection_timeout,
            poll_interval=args.poll_interval,
        )
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        error_kind = next(
            (
                name
                for kind, name in (
                    (ClientError, "contract_or_provider_error"),
                    (OSError, "transport_os_error"),
                    (ValueError, "value_error"),
                    (TypeError, "type_error"),
                )
                if isinstance(exc, kind)
            ),
            "unexpected_error",
        )
        parser.exit(
            1, json.dumps({"status": "relay_stopped", "error_kind": error_kind}) + "\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
