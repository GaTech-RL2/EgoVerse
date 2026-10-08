"""Metered, tool-only Astra sessions using the Responses API; no silent fallback."""

import copy
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request

from .common import digest, encoded, strict_json
from .protocol import OBSERVER_PROMPT


class ModelFailure(RuntimeError):
    pass


class BudgetEnd(RuntimeError):
    pass


class Meter:
    def __init__(self, limit):
        self.limit = limit
        self.total = 0
        self.unknown = 0
        self.usage = {
            role: {
                "input_tokens": 0,
                "cached_input_tokens": 0,
                "output_tokens": 0,
                "reasoning_tokens": 0,
                "image_tokens": 0,
                "calls": 0,
                "generation_attempts": 0,
                "token_count_requests": 0,
                "failed_transport_requests": 0,
                "provider_wall_seconds": 0.0,
            }
            for role in ("actor", "observer")
        }

    def record(self, role, usage):
        if not isinstance(usage, dict) or any(
            type(usage.get(k)) is not int or usage[k] < 0
            for k in ("input_tokens", "output_tokens")
        ):
            self.unknown += 1
            raise ModelFailure("usage_unavailable")
        target = self.usage[role]
        target["input_tokens"] += usage["input_tokens"]
        target["output_tokens"] += usage["output_tokens"]
        for field, container, name in (
            ("cached_input_tokens", "input_tokens_details", "cached_tokens"),
            ("reasoning_tokens", "output_tokens_details", "reasoning_tokens"),
            ("image_tokens", "input_tokens_details", "image_tokens"),
        ):
            details = usage.get(container) or {}
            value = details.get(name) if isinstance(details, dict) else None
            if type(value) is not int or value < 0:
                target[field] = None
            elif target[field] is not None:
                target[field] += value
        target["calls"] += 1
        self.total += usage["input_tokens"] + usage["output_tokens"]
        if self.total > self.limit:
            raise ModelFailure("provider_usage_exceeded_reserved_budget")


class HTTP:
    def __init__(self, base_url, *, key=None):
        parsed = urllib.parse.urlparse(base_url)
        if (
            parsed.scheme != "https"
            or parsed.username
            or parsed.password
            or parsed.query
        ):
            raise ValueError("credential_free_https_endpoint_required")
        self.base_url = base_url.rstrip("/")
        self.key = key or os.environ.get("OPENAI_API_KEY")
        if not self.key:
            raise ModelFailure("api_key_unavailable")

    def __call__(self, path, body, timeout):
        request = urllib.request.Request(
            self.base_url + "/" + path,
            data=encoded(body),
            headers={
                "Authorization": "Bearer " + self.key,
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return strict_json(response.read())
        except urllib.error.HTTPError as error:
            # Provider error bodies may echo credential fragments; never return them.
            raise ModelFailure("provider_http_" + str(error.code)) from None
        except Exception as error:
            raise ModelFailure("provider_" + type(error).__name__) from None


def observation_content(value):
    """Keep images out of token-expensive JSON strings; attach the exact pixels."""
    frames = []

    def visit(node, path):
        if (
            isinstance(node, dict)
            and node.get("encoding") == "image/png"
            and "base64" in node
        ):
            frames.append((path, node["base64"]))
            return {
                "image_attachment": path,
                "shape": node["shape"],
                "encoding": "image/png",
            }
        if isinstance(node, dict):
            return {k: visit(v, path + "/" + k) for k, v in node.items()}
        if isinstance(node, list):
            return [visit(v, path + "/" + str(i)) for i, v in enumerate(node)]
        return node

    text = json.dumps(visit(value, "result"), allow_nan=False)
    content = []
    for label, frame in frames:
        content.extend(
            [
                {"type": "input_text", "text": "Current tool image: " + label},
                {"type": "input_image", "image_url": "data:image/png;base64," + frame},
            ]
        )
    return text, content


class Session:
    def __init__(self, model, limits, meter, events, *, post=None, role="actor"):
        self.model, self.limits, self.meter, self.events, self.role = (
            model,
            limits,
            meter,
            events,
            role,
        )
        self.post = post or HTTP(model["base_url"])
        self.history = []

    def _post(self, path, body, timeout):
        usage = self.meter.usage[self.role]
        field = (
            "token_count_requests"
            if path.endswith("input_tokens")
            else "generation_attempts"
        )
        usage[field] += 1
        start, failed = time.monotonic(), False
        try:
            return self.post(path, body, timeout)
        except Exception:
            failed = True
            usage["failed_transport_requests"] += 1
            raise
        finally:
            elapsed = time.monotonic() - start
            usage["provider_wall_seconds"] += elapsed
            self.events.emit(
                "provider_transport",
                role=self.role,
                endpoint=path,
                failed=failed,
                latency_seconds=elapsed,
            )

    def request(self, system, tools, *, text_format=None, timeout=None):
        cap = (
            self.limits.actor_output_tokens
            if self.role == "actor"
            else self.limits.observer_output_tokens
        )
        body = {
            "model": self.model["identifier"],
            "instructions": system,
            "input": copy.deepcopy(self.history),
            "tools": tools,
            "reasoning": {"effort": self.model["reasoning_effort"]},
            "store": False,
            "truncation": "disabled",
            "parallel_tool_calls": False,
            "include": ["reasoning.encrypted_content"],
        }
        if text_format:
            body["text"] = {"format": text_format}
        if tools:
            body["tool_choice"] = "required"
        count_body = {
            k: body[k]
            for k in (
                "model",
                "instructions",
                "input",
                "tools",
                "text",
                "reasoning",
                "parallel_tool_calls",
                "tool_choice",
                "truncation",
            )
            if k in body
        }
        timeout = min(
            self.limits.response_seconds, timeout or self.limits.response_seconds
        )
        start = time.monotonic()
        counts = self._post("responses/input_tokens", count_body, timeout)
        count = counts.get("input_tokens")
        if type(count) is not int or count < 0:
            raise ModelFailure("input_token_count_unavailable")
        if (
            count > self.limits.context_tokens
            or self.meter.total + count + cap > self.meter.limit
        ):
            raise BudgetEnd("token_limit")
        remaining_time = timeout - (time.monotonic() - start)
        if remaining_time <= 0:
            raise BudgetEnd("wall_limit")
        body["max_output_tokens"] = cap
        # Evaluator-owned events are private; this includes context/response
        # bytes for audit. No credentials or provider request headers are logged.
        self.events.emit(
            "model_request",
            role=self.role,
            request=body,
            request_sha256=digest(body),
            input_tokens_counted=count,
        )
        try:
            response = self._post("responses", body, remaining_time)
        except ModelFailure:
            self.meter.unknown += 1
            self.events.emit(
                "error", role=self.role, error_class="ModelFailure", usage_unknown=True
            )
            raise
        self.events.emit(
            "model_response",
            role=self.role,
            response=response,
            response_sha256=digest(response),
            latency_seconds=time.monotonic() - start,
        )
        self.meter.record(self.role, response.get("usage"))
        if response.get("model") != self.model["identifier"]:
            raise ModelFailure("returned_model_mismatch")
        if (
            response["usage"]["input_tokens"] != count
            or response["usage"]["output_tokens"] > cap
        ):
            raise ModelFailure("provider_token_contract_mismatch")
        if response.get("status") != "completed":
            raise ModelFailure("incomplete_response")
        self.history.extend(response.get("output", []))
        return response

    def tool_result(self, call_id, result):
        text, images = observation_content(result)
        self.history.append(
            {"type": "function_call_output", "call_id": call_id, "output": text}
        )
        if images:
            self.history.append({"role": "user", "content": images})


class Observer:
    REQUESTS = {
        "scene": "Describe visible scene evidence.",
        "gripper_and_objects": "Describe the visible gripper and objects.",
        "occlusions": "Describe visible occlusions and uncertainty.",
    }

    def __init__(self, session):
        self.session, self.calls = session, 0

    def describe(self, request_kind, observation, *, timeout=None):
        if request_kind not in self.REQUESTS:
            raise ValueError("observer_request_not_allowed")
        if self.calls >= self.session.limits.observer_calls:
            raise BudgetEnd("observer_call_limit")
        step = observation["simulator_step"]
        visible = {
            "frame_step": step,
            "timestamp": observation["timestamp"],
            "cameras": observation["observations"],
            "request": self.REQUESTS[request_kind],
        }
        text, images = observation_content(visible)
        self.session.history = [
            {"role": "user", "content": [{"type": "input_text", "text": text}, *images]}
        ]
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["frame_step", "visible_facts", "uncertainties", "occlusions"],
            "properties": {
                "frame_step": {"type": "integer", "enum": [step]},
                **{
                    key: {"type": "array", "items": {"type": "string"}, "maxItems": 8}
                    for key in ("visible_facts", "uncertainties", "occlusions")
                },
            },
        }
        self.calls += 1
        response = self.session.request(
            OBSERVER_PROMPT,
            [],
            timeout=timeout,
            text_format={
                "type": "json_schema",
                "name": "visible_evidence",
                "strict": True,
                "schema": schema,
            },
        )
        parts = [
            part["text"]
            for item in response["output"]
            if item.get("type") == "message"
            for part in item.get("content", [])
            if part.get("type") == "output_text"
        ]
        value = strict_json("".join(parts))
        import jsonschema

        jsonschema.validate(value, schema)
        words = " ".join(
            s
            for key in ("visible_facts", "uncertainties", "occlusions")
            for s in value[key]
        )
        if re.search(
            r"\b(should|recommend|next action|move the|rotate the|task complete|successfully completed)\b",
            words,
            re.I,
        ):
            raise ValueError("observer_advice_rejected")
        return value
