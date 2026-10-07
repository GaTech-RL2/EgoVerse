"""Offline Codex request/job proofs. Never invent HTTP metadata or served models.

Private CLI events are read only to verify completion, tool absence and usage;
raw text/reasoning is never copied into the public audit receipt.
"""

import base64
import hashlib
import json
from pathlib import Path

from .astra_client import ClientError, _strict_json
from .codex_accounting import normalize_receipt_usage
from .codex_executor import generation_schema
from .records import digest, file_sha256


def require(value, message):
    if not value:
        raise ValueError(message)


def _encoded(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def reconstruct_inputs(request):
    """Reconstruct the exact executor projection from the original request."""
    from . import frs_agent, representation_agent
    from .meta_harness import relay_agent

    modules = {
        m.SCHEMA_VERSION: m for m in (frs_agent, representation_agent, relay_agent)
    }
    module = modules.get(request.get("schema_version"))
    require(module is not None, "Unsupported Codex request schema")
    payload = module.build_payload(
        request, "gpt-6-astra", sampling={"reasoning_effort": "medium"}
    )
    system = payload["messages"][0]["content"]
    lines = [
        system,
        "Do not call tools or inspect files, shell, network, or workspace. "
        "Use only this prompt and the attached images. Return the requested JSON.",
    ]
    images, labels = {}, []
    label = ""
    for part in payload["messages"][1]["content"]:
        if part["type"] == "text":
            label = part["text"]
            lines.append(label)
        else:
            require(part["type"] == "image_url", "Unexpected Codex input part")
            prefix, encoded = part["image_url"]["url"].split(",", 1)
            require(prefix == "data:image/png;base64", "Non-PNG Codex image")
            raw = base64.b64decode(encoded, validate=True)
            name = f"image_{len(images)}.png"
            labels.append(
                {
                    "image_index": len(images),
                    "file": name,
                    "label": label,
                    "sha256": _sha(raw),
                }
            )
            images[name] = raw
    lines.append(
        "Attached images in CLI order (image_index is zero based):\n"
        + json.dumps(labels)
    )
    schema = module.response_schema(request)
    prompt = ("\n\n".join(lines) + "\n").encode()
    files = {
        "request.json": _encoded(request),
        "schema.json": _encoded(generation_schema(schema)),
        "prompt.txt": prompt,
        **images,
    }
    hashes = {
        "request_sha256": _sha(files["request.json"]),
        "schema_sha256": _sha(files["schema.json"]),
        "original_schema_sha256": _sha(_encoded(schema)),
        "prompt_sha256": _sha(prompt),
        "system_prompt_sha256": _sha(system.encode()),
        "image_hashes": labels,
        "prompt_template_version": module.PROMPT_TEMPLATE_VERSION,
    }
    return module, files, hashes


def verify_codex_provider(request, row, proposal, settings, *, job_directory=None):
    """Verify ledger binding and, when supplied, the original local job bytes.

    Missing local job artifacts yield ``unverified`` with explicit missing files.
    Present but inconsistent artifacts always fail. This never certifies physics.
    """
    require(
        settings.get("backend") == "codex_relay"
        and settings["model"] == "gpt-6-astra"
        and settings["reasoning_effort"] == "medium"
        and settings.get("max_completion_tokens") is None,
        "Codex audit requires the declared medium-effort harness",
    )
    from .meta_harness import relay_agent

    # Profile v1 serialized the context in insertion order, while request.json
    # canonicalized keys. Recover only that order from the original prompt;
    # every context/schema value must still equal the bound request exactly.
    recovered_order = False
    runtime_profile = request.get("schema_version") == relay_agent.SCHEMA_VERSION
    directory = Path(job_directory).resolve() if job_directory is not None else None
    if runtime_profile and directory is not None:
        prompt_path = directory / "prompt.txt"
        if prompt_path.is_file():
            require(
                not prompt_path.is_symlink()
                and prompt_path.resolve().is_relative_to(directory),
                "Codex job artifact escapes directory",
            )
            projections = [
                _strict_json(line)
                for line in prompt_path.read_text().splitlines()
                if line.startswith('{"request":')
            ]
            require(
                len(projections) == 1, "Missing or ambiguous runtime prompt projection"
            )
            projection = projections[0]
            require(
                digest(projection)
                == digest(
                    {
                        "request": request["context"],
                        "response_schema": relay_agent.response_schema(request),
                    }
                ),
                "Runtime prompt context/schema differs from the bound request",
            )
            request = {**request, "context": projection["request"]}
            recovered_order = True
    module, files, hashes = reconstruct_inputs(request)
    require(
        row.get("backend") == "codex_relay"
        and row.get("requested_model") == "gpt-6-astra"
        and row.get("reasoning_effort") == "medium",
        "Codex ledger backend/model differs",
    )
    for key in (*module._IDENTITY_FIELDS, "request_fingerprint"):
        require(
            type(row.get(key)) is type(request[key]) and row[key] == request[key],
            f"Codex request identity differs: {key}",
        )
    require(
        type(row.get("accepted")) is bool and row["accepted"] == (proposal is not None),
        "Codex decision acceptance differs",
    )
    receipt = row.get("codex_receipt")
    require(isinstance(receipt, dict), "Codex execution receipt missing")
    require(
        receipt.get("backend") == "codex_exec"
        and receipt.get("configured_model") == settings["model"]
        and receipt.get("reasoning_effort") == "medium"
        and receipt.get("request_fingerprint") == request["request_fingerprint"],
        "Codex execution settings/request differ",
    )
    for key, value in hashes.items():
        if key == "prompt_sha256" and runtime_profile and not recovered_order:
            # No original prompt means ordering cannot be proven. The missing
            # prompt artifact below must keep the result unverified.
            continue
        require(receipt.get(key) == value, f"Codex reconstructed input differs: {key}")
    counts, lower_bound = normalize_receipt_usage(receipt)
    require(
        row.get("token_usage") == counts, "Codex ledger usage differs from job receipt"
    )
    require(
        row.get("provider_call") is receipt.get("provider_call"),
        "Codex job-start ledger differs",
    )
    if row["accepted"]:
        require(
            receipt.get("accepted") is True
            and receipt.get("status") == "accepted"
            and receipt.get("provider_unavailable") is False
            and receipt.get("exit_code") == 0
            and receipt.get("observed_completed_turns") == 1
            and receipt.get("tool_items") == [],
            "Accepted Codex job lacks normal tool-free completion",
        )
    require(
        receipt.get("returned_model") in (None, "gpt-6-astra"),
        "Codex receipt reports another served model",
    )
    if proposal is not None:
        bound = {"request_fingerprint": request["request_fingerprint"]}
        if request["schema_version"] == "representation-1.0":
            bound.update(
                {
                    key: request[key]
                    for key in module._IDENTITY_FIELDS
                    if key != "request_id"
                }
            )
            bound["decision_id"] = request["request_id"]
        elif request["schema_version"] == relay_agent.SCHEMA_VERSION:
            bound["decision_id"] = request["request_id"]
        require(
            all(proposal.get(key) == value for key, value in bound.items()),
            "Applied Codex proposal has incorrect local identity binding",
        )
        parsed = module.parse_proposal(
            {key: value for key, value in proposal.items() if key not in bound}, request
        )
        require(
            digest(parsed) == digest(proposal),
            "Applied Codex proposal fails its original contract",
        )
    report = {
        "schema_version": "codex-provider-audit-1.0",
        "status": "unverified",
        "request_fingerprint": request["request_fingerprint"],
        "request_sha256": hashes["request_sha256"],
        "ledger_binding_verified": True,
        "local_job_verified": False,
        "prompt_key_order_recovered_from_original_artifact": recovered_order,
        "unverified_input_hashes": (
            ["prompt_sha256"] if runtime_profile and not recovered_order else []
        ),
        "configured_model": settings["model"],
        "returned_model": receipt.get("returned_model"),
        "token_usage": counts,
        "token_usage_is_lower_bound": lower_bound,
        "files_sha256": {},
        "missing_files": [],
        "auditor_sources_sha256": {
            path.name: file_sha256(path)
            for path in (
                Path(__file__),
                Path(__file__).with_name("codex_executor.py"),
                Path(__file__).with_name("codex_accounting.py"),
                Path(module.__file__),
            )
        },
    }
    required = [
        *files,
        "completed.json",
        "started.json",
        "invocation.json",
        "events.jsonl",
        "final.json",
    ]
    missing = [
        name
        for name in required
        if directory is None or not (directory / name).is_file()
    ]
    report["missing_files"] = missing
    for name in required:
        if name in missing:
            continue
        path = directory / name
        require(
            not path.is_symlink() and path.resolve().is_relative_to(directory),
            "Codex job artifact escapes directory",
        )
        report["files_sha256"][name] = file_sha256(path)
        if name in files:
            require(
                path.read_bytes() == files[name],
                f"Codex original artifact differs: {name}",
            )
    if missing:
        return report

    def read(name):
        return _strict_json((directory / name).read_bytes())

    completed, started = read("completed.json"), read("started.json")
    require(
        completed["binding"] == started
        and started["request_sha256"] == hashes["request_sha256"],
        "Codex durable job binding differs",
    )
    require(
        started["settings"]["model"] == "gpt-6-astra"
        and started["settings"]["reasoning_effort"] == "medium",
        "Codex journal settings differ",
    )
    result = completed["result"]
    require(
        result["receipt"] == receipt
        and result["request_fingerprint"] == request["request_fingerprint"],
        "Codex completed receipt differs from worker ledger",
    )
    require(
        digest(result["proposal"]) == digest(proposal),
        "Codex completed proposal differs from applied decision",
    )
    invocation = read("invocation.json")
    argv = invocation["argv"]
    require(
        isinstance(argv, list) and all(isinstance(x, str) for x in argv),
        "Invalid CLI argv",
    )
    # Original paths may have moved during archival; compare file names and order.
    require(
        argv[0] == started["settings"]["executable"], "Codex executable binding differs"
    )
    expected_prefix = [
        argv[0],
        "exec",
        "--ignore-user-config",
        "--ignore-rules",
        "--ephemeral",
        "--skip-git-repo-check",
        "--sandbox",
        "read-only",
        "--json",
        "-m",
        "gpt-6-astra",
        "-c",
        'model_reasoning_effort="medium"',
        "--output-schema",
    ]
    require(
        argv[: len(expected_prefix)] == expected_prefix,
        "Codex invocation harness differs",
    )
    tail = argv[len(expected_prefix) :]
    require(
        len(tail) == 4 + 2 * len(hashes["image_hashes"])
        and Path(tail[0]).name == "schema.json"
        and tail[1] == "--output-last-message"
        and Path(tail[2]).name == "final.json"
        and tail[-1] == "-",
        "Codex output invocation differs",
    )
    for index, image in enumerate(hashes["image_hashes"]):
        require(
            tail[3 + index * 2] == "--image"
            and Path(tail[4 + index * 2]).name == image["file"],
            "Codex CLI image order differs",
        )
    events = [
        _strict_json(line)
        for line in (directory / "events.jsonl").read_bytes().splitlines()
        if line.strip()
    ]
    turns = [event for event in events if event.get("type") == "turn.completed"]
    require(
        len(turns) == receipt.get("observed_completed_turns"),
        "Codex event completion count differs",
    )
    require(
        [event.get("usage") for event in turns] == receipt.get("raw_usage_events"),
        "Codex raw event usage differs",
    )
    from .codex_executor import classify_cli_items

    classified = classify_cli_items(events)
    tool_items = classified["tool_items"]
    for kind, values in classified.items():
        label = "tool" if kind == "tool_items" else "item"
        require(
            values == receipt.get(kind, []),
            f"Codex {label} audit differs from raw events",
        )
    raw = (directory / "final.json").read_bytes()
    require(_sha(raw) == receipt.get("final_sha256"), "Codex final bytes differ")
    if row["accepted"]:
        require(
            not tool_items
            and not classified["error_items"]
            and not any(
                event.get("type") in ("error", "turn.failed") for event in events
            ),
            "Accepted Codex job contains error/tool events",
        )
        parsed = module.parse_proposal(raw, request)
        require(
            digest(parsed) == digest(proposal),
            "Codex final proposal differs from applied proposal",
        )
        messages = [
            e["item"]["text"]
            for e in events
            if e.get("type") == "item.completed"
            and isinstance(e.get("item"), dict)
            and e["item"].get("type") == "agent_message"
        ]
        require(
            messages
            and digest(_strict_json(messages[-1])) == digest(_strict_json(raw)),
            "Codex final output differs from final agent message",
        )
    elif receipt.get("status") == "proposal_rejected":
        try:
            module.parse_proposal(raw, request)
        except (ClientError, ValueError, TypeError, KeyError):
            pass
        else:
            raise ValueError(
                "Rejected Codex proposal actually passes the request contract"
            )
    report.update(status="passed", local_job_verified=True)
    return report
