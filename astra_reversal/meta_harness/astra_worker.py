"""One-shot Responses tool calls with exact input counting and no retries.

This transport does not reuse a Codex login as an API key. Credentials belong
to the serving process, never candidate bundles or runtime context.
"""

import base64
import copy
import io
import json
import math
import os
import time
import urllib.request
from dataclasses import asdict, dataclass

from PIL import Image

from astra_reversal.astra_client import _NoRedirect, _strict_json
from astra_reversal.records import digest

from .schema import tool_schemas

SYSTEM = """You are runtime Astra steering a frozen pi0.5 policy. Use only the
original task, raw live paired cameras, live proprioception, public execution
receipts and retrieved standard-demo cards. Donor examples are not the current
scene. Hypotheses in memory are not verified facts. A closed gripper is not proof
of a grasp. Preserve achieved subgoals. Return exactly one supplied policy tool.
Native conditioning is language=null, vision=null or clear_policy_program.
TEI(alpha=0) selects source A; it is NOT neutral. VEI(alpha=0) is native vision.
TEI and VEI cannot run together. Sources and exclusive frame bounds come from
retrieved cards. max_actions bounds conditioning; pi0.5 still predicts 50 actions
and executes FIVE before replanning with live state and cameras. Keep retains
the donor cursor and original expiry. Do not assume success or inspect anything
outside this request. Cite observation IDs in any remembered uncertainty.
"""


@dataclass(frozen=True)
class ModelSettings:
    model: str = "gpt-6-astra"
    reasoning_effort: str = "medium"
    input_tokens: int = 8192
    output_tokens: int = 256
    timeout: float = 240.0

    def __post_init__(self):
        if not (self.model == "gpt-6-astra" or self.model.startswith("gpt-6-astra-")):
            raise ValueError("This study requires Astra without model substitution")
        if (
            self.reasoning_effort != "medium"
            or self.input_tokens != 8192
            or self.output_tokens != 256
        ):
            raise ValueError("Changing frozen runtime settings requires a new protocol")
        if (
            type(self.timeout) not in (int, float)
            or not math.isfinite(self.timeout)
            or not 0 < self.timeout <= 240
        ):
            raise ValueError("Runtime request timeout must be within 240 seconds")


def png_url(raw):
    stream = io.BytesIO()
    Image.fromarray(raw).save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()


def build_payload(request, settings):
    context = copy.deepcopy(request["context"])
    content = [{"type": "input_text", "text": json.dumps(context, allow_nan=False)}]
    for frame in request["frames"]:
        for camera, image in frame["images"].items():
            content.extend(
                [
                    {
                        "type": "input_text",
                        "text": f"RAW LIVE {frame['observation_id']} action {frame['action']} camera {camera}",
                    },
                    {
                        "type": "input_image",
                        "image_url": png_url(image),
                        "detail": "high",
                    },
                ]
            )
    return {
        "model": settings.model,
        "reasoning": {"effort": settings.reasoning_effort},
        "instructions": SYSTEM,
        "input": [{"role": "user", "content": content}],
        "tools": tool_schemas(context["cards"]),
        "tool_choice": "required",
        "parallel_tool_calls": False,
        "max_output_tokens": settings.output_tokens,
        "store": False,
    }


class AstraWorker:
    def __init__(self, settings=None, *, post=None):
        self.settings = settings or ModelSettings()
        self.post = post or self._post
        self.records = []
        if post is None and not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is required for the bounded Responses transport"
            )

    @staticmethod
    def _post(path, payload, timeout):
        # Do not follow redirects with the authorization header, or auto-retry.
        request = urllib.request.Request(
            "https://api.openai.com/v1/" + path,
            data=json.dumps(payload, allow_nan=False).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + os.environ["OPENAI_API_KEY"],
            },
            method="POST",
        )
        with urllib.request.build_opener(_NoRedirect()).open(
            request, timeout=timeout
        ) as response:
            raw = response.read(4 * 1024 * 1024 + 1)
        if len(raw) > 4 * 1024 * 1024:
            raise ValueError("Provider response exceeds four MB")
        return _strict_json(raw)

    def __call__(self, request):
        started = time.monotonic()
        record = {
            "settings": asdict(self.settings),
            "request_id": request["request_id"],
            "generation_requests": 0,
            "count_requests": 0,
            "accepted": False,
            "usage": None,
            "response": None,
            "error": None,
        }
        call = None
        try:
            payload = build_payload(request, self.settings)
            record["payload_sha256"] = digest(payload)
            count_body = {
                key: payload[key] for key in ("model", "instructions", "input", "tools")
            }
            record["count_requests"] = 1
            count = self.post(
                "responses/input_tokens", count_body, self.settings.timeout
            )
            tokens = count.get("input_tokens")
            record["counted_input_tokens"] = tokens
            if type(tokens) is not int or not 0 < tokens <= self.settings.input_tokens:
                raise ValueError("input_token_budget")
            remaining = self.settings.timeout - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError("Token counting consumed the request deadline")
            record["generation_requests"] = 1
            response = self.post("responses", payload, remaining)
            record["response"] = response
            record["usage"] = response.get("usage")
            if response.get("model") != self.settings.model:
                raise ValueError("served_model_identity_changed")
            usage = response.get("usage") or {}
            if (
                type(usage.get("input_tokens")) is not int
                or type(usage.get("output_tokens")) is not int
                or not 0 < usage["input_tokens"] <= self.settings.input_tokens
                or not 0 <= usage["output_tokens"] <= self.settings.output_tokens
            ):
                raise ValueError("unknown_or_exceeded_token_budget")
            if response.get("status") != "completed":
                raise ValueError("incomplete_response")
            outputs = response.get("output", [])
            if any(
                item.get("type") not in ("reasoning", "function_call", "message")
                for item in outputs
            ):
                raise ValueError("unsupported_output_tool")
            calls = [item for item in outputs if item.get("type") == "function_call"]
            if len(calls) != 1:
                raise ValueError("Expected exactly one policy tool call")
            call = {
                "tool": calls[0]["name"],
                "arguments": _strict_json(calls[0]["arguments"].encode()),
            }
            record["accepted"] = True
        except Exception as error:
            # Retain bounded classification, never credentials or HTTP headers.
            record["error"] = type(error).__name__
            if isinstance(error, ValueError):
                record["validation_error"] = str(error)[:200]
        record["latency_seconds"] = time.monotonic() - started
        self.records.append(record)
        return {"call": call, "record": record}
