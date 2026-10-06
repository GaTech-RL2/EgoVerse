"""Local, one-shot Codex jobs; credentials never leave the local CLI.

``provider_call`` means a CLI job was started, NOT one raw HTTP request. CLI
internal retries/turns are unknown. Private event files may contain reasoning;
only structured receipts, never event text, should be published in reports.
"""

import base64
import hashlib
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

from .intervention_agent import normalize_usage


def _bytes(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _hash(value):
    return hashlib.sha256(value).hexdigest()


def _write(path, data):
    """Atomic, durable, private file replacement on the local filesystem."""
    temporary = path.with_name(path.name + ".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def generation_schema(value):
    """Normalize const/enum typing; only root cross-field oneOf is omitted.

    The unchanged domain parser enforces the full original response contract.
    """

    def visit(node):
        if isinstance(node, list):
            return [visit(item) for item in node]
        if not isinstance(node, dict):
            return node
        result = {key: visit(item) for key, item in node.items()}
        if "const" in result:
            result["enum"] = [result.pop("const")]
        if "enum" in result and "type" not in result:

            def kind(item):
                if item is None:
                    return "null"
                return {
                    bool: "boolean",
                    int: "integer",
                    float: "number",
                    str: "string",
                }[type(item)]

            kinds = list(dict.fromkeys(kind(item) for item in result["enum"]))
            result["type"] = kinds[0] if len(kinds) == 1 else kinds
        return result

    schema = visit(value)
    schema.pop("oneOf", None)
    return schema


def _module(request):
    from . import demo_skill_agent, frs_agent, representation_agent
    from .meta_harness import relay_agent

    for module in (frs_agent, representation_agent, demo_skill_agent, relay_agent):
        if request.get("schema_version") == module.SCHEMA_VERSION:
            return module
    raise ValueError("Unsupported request schema_version")


def execute_request(
    request,
    directory,
    *,
    model="gpt-6-astra",
    reasoning_effort="medium",
    timeout=170,
    executable="codex",
):
    """Execute or recover one request in a dedicated durable job directory.

    Any existing started marker without a completed result is indeterminate and
    is NEVER retried. Use a new directory only for a genuinely new request.
    """
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    identity = _hash(_bytes(request))
    settings = {
        "model": model,
        "reasoning_effort": reasoning_effort,
        "executable": str(executable),
    }
    binding = {"request_sha256": identity, "settings": settings}
    completed = directory / "completed.json"
    started_path = directory / "started.json"
    receipt = {
        "backend": "codex_exec",
        "requested_model": model,
        "configured_model": model,
        "returned_model": None,
        "reasoning_effort": reasoning_effort,
        "request_fingerprint": request.get("request_fingerprint"),
        "request_sha256": identity,
        "request_time": time.time(),
        "provider_call": False,
        "provider_call_semantics": "CLI job started; not a raw HTTP request count",
        "internal_retry_count": None,
        "internal_turn_count": None,
        "accepted": False,
        "provider_unavailable": False,
        "status": "provider_unavailable",
        "token_usage": normalize_usage(None),
        "raw_usage": None,
        "tool_items": [],
        "cli_version": None,
    }
    result = {
        "request_fingerprint": request.get("request_fingerprint"),
        "proposal": None,
        "receipt": receipt,
    }
    clock = time.perf_counter()

    def unavailable(reason):
        receipt.update(
            status="provider_unavailable",
            provider_unavailable=True,
            availability_reason=reason,
            error_kind="provider_unavailable",
        )

    if completed.exists():
        saved = json.loads(completed.read_text())
        if saved.get("binding") != binding:
            unavailable("job_directory_binding_mismatch")
            return result
        return saved["result"]
    # Exclusive creation claims the job before any process or mutable evidence.
    try:
        fd = os.open(started_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        unavailable("already_started_job_indeterminate")
        return result
    with os.fdopen(fd, "wb") as stream:
        stream.write(_bytes(binding))
        stream.flush()
        os.fsync(stream.fileno())
    directory_fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    try:
        if model != "gpt-6-astra" or reasoning_effort != "medium":
            raise ValueError("Only gpt-6-astra with medium effort is authorized")
        module = _module(request)
        payload = module.build_payload(
            request, model, sampling={"reasoning_effort": reasoning_effort}
        )
        original_schema = module.response_schema(request)
        schema = generation_schema(original_schema)
        _write(directory / "request.json", _bytes(request))
        _write(directory / "schema.json", _bytes(schema))
        receipt["schema_sha256"] = _hash(_bytes(schema))
        receipt["original_schema_sha256"] = _hash(_bytes(original_schema))
        receipt["prompt_template_version"] = module.PROMPT_TEMPLATE_VERSION
        system = payload["messages"][0]["content"]
        lines = [
            system,
            "Do not call tools or inspect files, shell, network, or workspace. "
            "Use only this prompt and the attached images. Return the requested JSON.",
        ]
        images = []
        labels = []
        pending_label = ""
        for part in payload["messages"][1]["content"]:
            if part["type"] == "text":
                pending_label = part["text"]
                lines.append(pending_label)
            elif part["type"] == "image_url":
                prefix, encoded = part["image_url"]["url"].split(",", 1)
                if prefix != "data:image/png;base64":
                    raise ValueError("Expected PNG attachment")
                raw = base64.b64decode(encoded, validate=True)
                name = f"image_{len(images)}.png"
                _write(directory / name, raw)
                labels.append(
                    {
                        "image_index": len(images),
                        "file": name,
                        "label": pending_label,
                        "sha256": _hash(raw),
                    }
                )
                images.append(directory / name)
        lines.append(
            "Attached images in CLI order (image_index is zero based):\n"
            + json.dumps(labels)
        )
        prompt = "\n\n".join(lines) + "\n"
        _write(directory / "prompt.txt", prompt.encode())
        receipt.update(
            prompt_sha256=_hash(prompt.encode()),
            system_prompt_sha256=_hash(system.encode()),
            image_hashes=labels,
        )
        environment = dict(os.environ)
        for name in (
            "OPENAI_API_KEY",
            "CODEX_API_KEY",
            "OPENAI_BASE_URL",
            "OPENAI_ORG_ID",
            "OPENAI_PROJECT_ID",
        ):
            environment.pop(name, None)
        with tempfile.TemporaryDirectory(prefix="astra-codex-job-") as isolated:
            version = subprocess.run(
                [str(executable), "--version"],
                cwd=isolated,
                capture_output=True,
                text=True,
                timeout=min(timeout, 10),
                check=False,
                env=environment,
            )
            if version.returncode:
                unavailable("cli_version_unavailable")
            else:
                receipt["cli_version"] = version.stdout.strip()
                args = [
                    str(executable),
                    "exec",
                    "--ignore-user-config",
                    "--ignore-rules",
                    "--ephemeral",
                    "--skip-git-repo-check",
                    "--sandbox",
                    "read-only",
                    "--json",
                    "-m",
                    model,
                    "-c",
                    f'model_reasoning_effort="{reasoning_effort}"',
                    "--output-schema",
                    str(directory / "schema.json"),
                    "--output-last-message",
                    str(directory / "final.json"),
                ]
                for path in images:
                    args.extend(["--image", str(path)])
                args.append("-")
                _write(
                    directory / "invocation.json",
                    _bytes({"argv": args, "cwd": isolated}),
                )
                receipt["provider_call"] = True
                run = subprocess.run(
                    args,
                    input=prompt,
                    cwd=isolated,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    check=False,
                    env=environment,
                )
                _write(directory / "events.jsonl", run.stdout.encode())
                _write(directory / "stderr.log", run.stderr.encode())
                receipt["exit_code"] = run.returncode
                _consume(run, directory, request, module, result)
    except subprocess.TimeoutExpired as exc:
        # Partial private event output is retained for audit, never exposed in receipt.
        for name, output in (("events.jsonl", exc.stdout), ("stderr.log", exc.stderr)):
            if output:
                _write(
                    directory / name,
                    output if isinstance(output, bytes) else output.encode(),
                )
        partial = exc.stdout or ""
        if isinstance(partial, bytes):
            partial = partial.decode("utf-8", errors="replace")
        if receipt["provider_call"]:
            _consume(
                subprocess.CompletedProcess([], -1, partial, ""),
                directory,
                request,
                module,
                result,
            )
        unavailable("cli_timeout")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        unavailable("executor_" + type(exc).__name__)
    finally:
        receipt["status"] = (
            "accepted"
            if receipt["accepted"]
            else (
                "provider_unavailable"
                if receipt["provider_unavailable"]
                else "proposal_rejected"
            )
        )
        receipt.update(
            response_time=time.time(), latency_seconds=time.perf_counter() - clock
        )
        _write(completed, _bytes({"binding": binding, "result": result}))
    return result


def _consume(run, directory, request, module, result):
    receipt = result["receipt"]

    def fail(reason):
        receipt.update(
            status="provider_unavailable",
            provider_unavailable=True,
            availability_reason=reason,
            error_kind="provider_unavailable",
        )

    events = []
    malformed_events = False
    for line in run.stdout.splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
            if not isinstance(event, dict):
                raise ValueError("Malformed event")
            events.append(event)
        except (ValueError, TypeError):
            malformed_events = True
    completed = [event for event in events if event.get("type") == "turn.completed"]
    receipt["observed_completed_turns"] = len(completed)
    receipt["tool_items"] = [
        {"event_type": event.get("type"), "item_type": event["item"].get("type")}
        for event in events
        if isinstance(event.get("item"), dict)
        and event["item"].get("type") not in ("agent_message", "reasoning")
    ]
    if completed:
        usages = [event.get("usage") for event in completed]
        receipt["raw_usage_events"] = usages
        # A single turn preserves the provider's actual raw usage. Ambiguous
        # multi-turn jobs retain all receipts and report known sums as lower bounds.
        usage = usages[0] if len(usages) == 1 else None
        receipt["raw_usage"] = usage
        normalized_rows = []
        for raw in usages:
            mapped = dict(raw) if isinstance(raw, dict) else {}
            if "reasoning_output_tokens" in mapped:
                mapped["output_tokens_details"] = {
                    "reasoning_tokens": mapped["reasoning_output_tokens"]
                }
            normalized = normalize_usage(mapped)
            if normalized["total_tokens"] is None and all(
                normalized[k] is not None for k in ("input_tokens", "output_tokens")
            ):
                normalized["total_tokens"] = (
                    normalized["input_tokens"] + normalized["output_tokens"]
                )
                receipt["total_tokens_derived"] = True
            normalized_rows.append(normalized)
        receipt["token_usage"] = {
            key: (
                sum(row[key] for row in normalized_rows if row[key] is not None)
                if any(row[key] is not None for row in normalized_rows)
                else None
            )
            for key in normalized_rows[0]
        }
        receipt["token_usage_is_lower_bound"] = len(usages) != 1 or malformed_events
    if malformed_events:
        fail("malformed_cli_events")
    elif run.returncode:
        fail("cli_nonzero_exit")
    elif receipt["tool_items"]:
        fail("cli_tool_use")
    elif len(completed) != 1 or any(
        e.get("type") in ("turn.failed", "error") for e in events
    ):
        fail("cli_incomplete_or_ambiguous_turn")
    elif not (directory / "final.json").is_file():
        fail("cli_final_missing")
    else:
        raw = (directory / "final.json").read_text()
        receipt["final_sha256"] = _hash(raw.encode())
        try:
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                raise ValueError("Expected JSON object")
        except (ValueError, TypeError):
            fail("cli_final_malformed")
            return
        try:
            result["proposal"] = module.parse_proposal(raw, request)
            receipt["accepted"] = True
        except (ValueError, TypeError, KeyError):
            receipt["error_kind"] = "proposal_rejected"
            receipt["error"] = "Completed proposal failed the original request contract"
