"""Existing authenticated Astra relay for an explicitly uncapped latency pilot.

Codex CLI does not expose a hard completion-token cap. These measurements must
not be pooled with the 8192/256-token main protocol or selected as its results.
"""

import base64
import copy
import io
import time

from PIL import Image

from astra_reversal.records import digest

from .astra_worker import SYSTEM
from .schema import exact, tool_schemas

SCHEMA_VERSION = "meta-harness-runtime-profile-1"
PROMPT_TEMPLATE_VERSION = "meta-harness-runtime-1"
SYSTEM_PROMPT = SYSTEM
_IDENTITY_FIELDS = ("schema_version", "request_id")


def wire_request(request):
    value = copy.deepcopy(request)
    value["schema_version"] = SCHEMA_VERSION
    for frame in value["frames"]:
        for camera, array in frame["images"].items():
            buffer = io.BytesIO()
            Image.fromarray(array).save(buffer, format="PNG")
            frame["images"][camera] = base64.b64encode(buffer.getvalue()).decode()
    value["request_fingerprint"] = digest(value)
    return value


def _validate_request(request):
    exact(
        request,
        (
            "schema_version",
            "request_id",
            "binding",
            "context",
            "frames",
            "request_fingerprint",
        ),
    )
    unbound = {k: v for k, v in request.items() if k != "request_fingerprint"}
    if (
        request["schema_version"] != SCHEMA_VERSION
        or digest(unbound) != request["request_fingerprint"]
    ):
        raise ValueError("Changed runtime request identity")
    if (
        not 1 <= len(request["frames"]) <= 2
        or not 1 <= len(request["context"]["cards"]) <= 6
    ):
        raise ValueError("Runtime context exceeds its fixed frame/card limits")
    if (
        request["context"]["current"]["observation_id"]
        != request["binding"]["observation_id"]
    ):
        raise ValueError("Runtime observation identity mismatch")


def response_schema(request):
    _validate_request(request)
    tools = tool_schemas(request["context"]["cards"])
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["tool", "arguments"],
        "properties": {
            "tool": {"type": "string", "enum": [t["name"] for t in tools]},
            "arguments": {"anyOf": [t["parameters"] for t in tools]},
        },
    }


def parse_proposal(value, request):
    from astra_reversal.astra_client import _strict_json

    _validate_request(request)
    if isinstance(value, (str, bytes)):
        value = _strict_json(value)
    exact(value, ("tool", "arguments"))
    tools = {t["name"]: t for t in tool_schemas(request["context"]["cards"])}
    if value["tool"] not in tools:
        raise ValueError("Unknown runtime policy tool")
    exact(value["arguments"], tools[value["tool"]]["parameters"]["required"])
    if value["arguments"]["observation_id"] != request["binding"]["observation_id"]:
        raise ValueError("Wrong request observation")
    return {
        **copy.deepcopy(value),
        "decision_id": request["request_id"],
        "request_fingerprint": request["request_fingerprint"],
    }


def build_payload(request, model, *, sampling=None):
    import json

    _validate_request(request)
    content = [
        {
            "type": "text",
            "text": json.dumps(
                {
                    "request": request["context"],
                    "response_schema": response_schema(request),
                }
            ),
        }
    ]
    for frame in request["frames"]:
        for camera, encoded in frame["images"].items():
            with Image.open(
                io.BytesIO(base64.b64decode(encoded, validate=True))
            ) as image:
                if image.mode != "RGB" or max(image.size) > 1024:
                    raise ValueError("Invalid live frame")
            content.extend(
                [
                    {
                        "type": "text",
                        "text": f"RAW LIVE {frame['observation_id']} action {frame['action']} {camera}",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64," + encoded},
                    },
                ]
            )
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        **(sampling or {}),
    }


class ProfileRelayWorker:
    def __init__(self, log, timeout=240):
        from astra_reversal.codex_relay import CodexRelayClient

        self.client = CodexRelayClient(
            family="meta_harness_runtime",
            model="gpt-6-astra",
            reasoning_effort="medium",
            response_log=log,
            timeout=timeout,
        )
        self.records = []

    def __call__(self, request):
        began = time.monotonic()
        call, error = None, None
        try:
            result = self.client.propose(wire_request(request))
            call = {key: result[key] for key in ("tool", "arguments")}
        except Exception as exc:
            error = type(exc).__name__
        provider = copy.deepcopy(self.client.records[-1]) if self.client.records else {}
        record = {
            "request_id": request["request_id"],
            "latency_seconds": time.monotonic() - began,
            "accepted": call is not None,
            "error": error,
            "provider": provider,
            "usage": provider.get("token_usage"),
            "hard_token_cap_enforced": False,
            "scope": "latency/interface pilot only; excluded from main selection",
        }
        self.records.append(record)
        return {"call": call, "record": record}
