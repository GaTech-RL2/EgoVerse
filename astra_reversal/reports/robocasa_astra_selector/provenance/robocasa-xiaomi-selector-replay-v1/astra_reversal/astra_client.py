"""A genuine vision-model transport for :class:`agent.CommandBackend`.

Reads one ``_wire_value(build_request(...))`` JSON object on stdin and writes
only the unmodified, validated model proposal on stdout. Credentials come from
``NVIDIA_INFERENCE_API_KEY``; never put credentials in the command arguments.
The separate response JSONL retains the provider model, usage, content and
timing for accepted and rejected calls. This module does not generate actions.

The protocol is documented by NVIDIA's multimodal-input guide and API backend:
https://docs.nvidia.com/nim/large-language-models/latest/advanced-use-cases/multimodal-input.html
https://nvidia-isaac.github.io/video_to_data/video_ingestion_agent/main/pages/model_backends.html
Internal/public gateway and key pairing:
https://github.com/isaac-for-healthcare/i4h-workflows/blob/main/local-agent/README.md

The model must be selected explicitly from the authenticated gateway catalog.
There is no model fallback, trajectory repair, image rescaling or interpolation.
"""

import argparse
import base64
import binascii
import io
import json
import math
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

from .agent import CAMERAS, SCHEMA_VERSION, parse_proposal

DEFAULT_ENDPOINT = "https://inference-api.nvidia.com/v1/chat/completions"
PROMPT_TEMPLATE_VERSION = "astra-reversal-http-1"
SYSTEM_PROMPT = """You are the vision and motion-planning agent for a simulated
LIBERO Panda robot. Solve the supplied instruction from the two CURRENT camera
images, robot proprioception and bounded observed history. This is zero-shot:
no demonstrations, simulator object poses, hidden goals, success predicates or
policy-generated proposed actions are available. Do not assume access to them.

Treat the versioned action_spec in the request as authoritative. In Stage 1 you
must personally generate every numeric value of an H by 7 action_chunk, where H
is action_spec.horizon. Each row applies for action_spec.timestep_seconds. The
seven channels are in action_spec.channels order, in environment controller
input space BEFORE checkpoint normalization and padding. For scaled_delta
OSC_POSE, translations and axis-angle rotations use the declared world frame;
the values are dimensionless controller inputs, not absolute positions, meters
or radians. Use the declared input/output scaling, per-channel bounds and
gripper convention. Do not return joint angles, sparse waypoints, code, nulls,
ellipses or instructions for another controller to fill in missing actions.

observation/state is [world eef x,y,z in meters, eef axis-angle rx,ry,rz in
radians, left finger qpos, right finger qpos in meters]. This is the CURRENT
observed pose, not the delta-action coordinate space. Completion eef_position
targets are absolute observed world positions; gripper_width is
abs(state[6] - state[7]) in meters. Choose only a supported completion type,
with positive tolerance and a meaningful reachable subgoal. Maintaining or
revising the current subgoal from fresh observations does not itself complete
it. Retain its subgoal_id while it remains active; assign a unique plan_id to
each response. Respect completed_subgoals, timeout and any validation_error.

The images are already in the experiment's camera orientation. Use their
actual pixels; do not infer world geometry from hidden calibration. Stage 2
returns only the requested subgoal, constraints and annotations, without
continuous actions. Pixel x increases right, y increases down. Annotations must
refer to the named source camera and remain inside its exact image dimensions.

Return exactly one JSON object satisfying the supplied response_schema and
request instructions. Echo schema_version, episode_id, observation_step and
(Stage 1) action_spec_id exactly. No markdown fences or surrounding prose.
"""

_REQUEST_FIELDS = {
    "schema_version",
    "episode_id",
    "observation_step",
    "observation_id",
    "task_instruction",
    "observation",
    "history",
    "completed_subgoals",
    "active_subgoal",
    "max_timeout_env_steps",
    "requested_model_version",
    "sampling_settings",
    "stage",
    "action_spec",
    "supported_completion_types",
    "instructions",
    "validation_error",
    "request_fingerprint",
}
_SAMPLING_FIELDS = {
    "max_completion_tokens",
    "temperature",
    "top_p",
    "reasoning_effort",
    "seed",
}
_RESPONSE_FIELDS = {
    "id",
    "object",
    "created",
    "model",
    "choices",
    "usage",
    "system_fingerprint",
    "service_tier",
}
_SECRET_FIELDS = {
    "authorization",
    "api_key",
    "apikey",
    "x-api-key",
    "headers",
    "request_headers",
    "response_headers",
}


class ClientError(ValueError):
    """A sanitized transport/contract failure; bounded retries belong to AstraAgent."""


def _strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ClientError("JSON contains a duplicate object key")
            result[key] = value
        return result

    def constant(_):
        raise ClientError("JSON contains a nonfinite number")

    try:
        return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ClientError("Response is not a single valid JSON object") from exc


def _endpoint(value):
    parsed = urllib.parse.urlsplit(value)
    if (
        parsed.scheme not in ("https", "http")
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ClientError("Endpoint must be a credential-free HTTP(S) URL")
    if parsed.scheme == "http" and parsed.hostname not in (
        "localhost",
        "127.0.0.1",
        "::1",
    ):
        raise ClientError("Remote inference requires an HTTPS endpoint")
    path = parsed.path.rstrip("/")
    if not path:
        path = "/v1/chat/completions"
    elif path.endswith("/v1"):
        path += "/chat/completions"
    elif not path.endswith("/chat/completions"):
        raise ClientError("Endpoint must name a /v1 base or /chat/completions resource")
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def response_schema(request):
    """Describe the existing schema; parse_proposal remains the authority."""

    def obj(properties):
        return {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        }

    def array(items, count):
        return {"type": "array", "items": items, "minItems": count, "maxItems": count}

    completion = []
    for kind in request["supported_completion_types"]:
        if kind == "eef_position":
            params = {
                "target": array({"type": "number"}, 3),
                "tolerance": {"type": "number", "exclusiveMinimum": 0},
            }
        elif kind == "gripper_width":
            params = {
                "target": {"type": "number"},
                "tolerance": {"type": "number", "exclusiveMinimum": 0},
            }
        elif kind == "image_progress":
            params = {"criterion": {"type": "string", "minLength": 1}}
        else:
            raise ClientError("Request names an unsupported completion checker")
        completion.append(obj({"type": {"const": kind}, "parameters": obj(params)}))
    if not completion:
        raise ClientError("Request must enable at least one completion checker")
    properties = {
        "schema_version": {"const": SCHEMA_VERSION},
        "episode_id": {"const": request["episode_id"]},
        "plan_id": {"type": "string", "minLength": 1},
        "observation_step": {"const": request["observation_step"]},
        "subgoal_id": {"type": "string", "minLength": 1},
        "subgoal_instruction": {"type": "string", "minLength": 1},
        "completion": {"anyOf": completion},
        "timeout_env_steps": {
            "type": "integer",
            "minimum": 1,
            "maximum": request["max_timeout_env_steps"],
        },
    }
    if request["stage"] == 1:
        spec = request["action_spec"]
        properties.update(
            action_spec_id={"const": spec["action_spec_id"]},
            action_chunk=array(array({"type": "number"}, 7), spec["horizon"]),
        )
    else:
        annotation_kinds = []
        for kind, size in (("point", 2), ("box", 4)):
            annotation_kinds.append(
                obj(
                    {
                        "camera": {"type": "string", "enum": list(CAMERAS)},
                        "kind": {"const": kind},
                        "coordinates": array({"type": "number", "minimum": 0}, size),
                        "label": {"type": "string", "minLength": 1},
                    }
                )
            )
        properties.update(
            annotations={
                "type": "array",
                "maxItems": 32,
                "items": {"anyOf": annotation_kinds},
            },
            constraints={
                "type": "array",
                "maxItems": 32,
                "items": {"type": "string", "minLength": 1},
            },
        )
    return obj(properties)


def _observations(request):
    if not isinstance(request, dict) or set(request) != _REQUEST_FIELDS:
        raise ClientError("Input must match build_request schema 1.0 exactly")
    if (
        request["schema_version"] != SCHEMA_VERSION
        or type(request["stage"]) is not int
        or request["stage"] not in (1, 2)
    ):
        raise ClientError("Unsupported request schema or experiment stage")
    observation = request["observation"]
    if not isinstance(observation, dict) or set(observation) != {
        *CAMERAS,
        "observation/state",
    }:
        raise ClientError(
            "Only the two declared cameras and robot proprioception are permitted"
        )
    state = observation["observation/state"]
    if (
        not isinstance(state, list)
        or len(state) != 8
        or any(type(x) not in (int, float) or not math.isfinite(x) for x in state)
    ):
        raise ClientError("Expected eight finite robot proprioception values")
    decoded, descriptions = {"observation/state": np.asarray(state)}, {}
    for camera in CAMERAS:
        wire = observation[camera]
        if (
            not isinstance(wire, dict)
            or set(wire) != {"encoding", "data"}
            or wire["encoding"] != "base64_png"
        ):
            raise ClientError("Camera images must be CommandBackend base64 PNGs")
        try:
            data = base64.b64decode(wire["data"], validate=True)
            with Image.open(io.BytesIO(data)) as frame:
                if frame.format != "PNG" or frame.mode != "RGB":
                    raise ClientError("Expected RGB PNG camera observations")
                decoded[camera] = np.asarray(frame).copy()
                descriptions[camera] = {
                    "width": frame.width,
                    "height": frame.height,
                    "encoding": "attached_png",
                    "image_index": len(descriptions),
                }
        except (ValueError, TypeError, binascii.Error, OSError) as exc:
            raise ClientError("Invalid PNG camera observation") from exc
    return decoded, descriptions


def build_payload(request, model, *, sampling=None):
    """Attach actual pixels, keeping the exact action spec/state/history in text."""
    decoded, descriptions = _observations(request)
    if request["requested_model_version"] not in (None, model):
        raise ClientError("Configured model differs from requested_model_version")
    settings = request["sampling_settings"]
    if not isinstance(settings, dict) or set(settings) - _SAMPLING_FIELDS:
        raise ClientError("Unsupported sampling_settings for the HTTP backend")
    settings = {"max_completion_tokens": 8192, **settings, **(sampling or {})}
    if set(settings) - _SAMPLING_FIELDS:
        raise ClientError("Unsupported HTTP sampling override")
    if (
        type(settings["max_completion_tokens"]) is not int
        or settings["max_completion_tokens"] <= 0
    ):
        raise ClientError("max_completion_tokens must be a positive integer")
    context = {key: value for key, value in request.items() if key != "observation"}
    context["observation"] = {
        "observation/state": request["observation"]["observation/state"],
        **descriptions,
    }
    content = [
        {
            "type": "text",
            "text": json.dumps(
                {
                    "prompt_template_version": PROMPT_TEMPLATE_VERSION,
                    "request": context,
                    "response_schema": response_schema(request),
                },
                allow_nan=False,
            ),
        }
    ]
    for camera in CAMERAS:
        content.extend(
            [
                {"type": "text", "text": f"Current camera: {camera}"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,"
                        + request["observation"][camera]["data"],
                    },
                },
            ]
        )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        "stream": False,
        "response_format": {"type": "json_object"},
        "cache": {"no-cache": True},
        **settings,
    }
    return payload, {**request, "observation": decoded}


def _redact(value, secret):
    if isinstance(value, str):
        return value.replace(secret, "[REDACTED]") if secret else value
    if isinstance(value, dict):
        return {
            key: _redact(item, secret)
            for key, item in value.items()
            if key.lower() not in _SECRET_FIELDS
        }
    if isinstance(value, list):
        return [_redact(item, secret) for item in value]
    return value


def _append_record(path, record):
    # Each command subprocess can share a sidecar without interleaving records.
    import fcntl

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with os.fdopen(
        os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600), "a"
    ) as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.write(json.dumps(record, allow_nan=False) + "\n")
        stream.flush()
        fcntl.flock(stream, fcntl.LOCK_UN)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Do not forward the credential to a redirected host.
        return None


class AstraHTTPClient:
    def __init__(
        self,
        *,
        model,
        response_log,
        endpoint=DEFAULT_ENDPOINT,
        timeout=110.0,
        sampling=None,
    ):
        if not isinstance(model, str) or not model.strip():
            raise ClientError("An explicitly configured model ID is required")
        if not response_log:
            raise ClientError("A response_log path is required for API provenance")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ClientError("HTTP timeout must be positive and finite")
        self.model, self.endpoint, self.timeout = model, _endpoint(endpoint), timeout
        self.response_log, self.sampling = response_log, sampling
        self.opener = urllib.request.build_opener(_NoRedirect())

    def generate(self, request):
        key = os.environ.get("NVIDIA_INFERENCE_API_KEY", "").strip()
        if not key:
            raise ClientError("NVIDIA_INFERENCE_API_KEY is not set")
        payload, decoded_request = build_payload(
            request, self.model, sampling=self.sampling
        )
        http_request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload, allow_nan=False).encode("utf-8"),
            headers={
                "Authorization": "Bearer " + key,
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        started, monotonic = time.time(), time.perf_counter()
        record = {
            "client_schema_version": "1.0",
            "prompt_template_version": PROMPT_TEMPLATE_VERSION,
            "request_fingerprint": request["request_fingerprint"],
            "episode_id": request["episode_id"],
            "observation_step": request["observation_step"],
            "stage": request["stage"],
            "endpoint": self.endpoint,
            "requested_model": self.model,
            "sampling_settings": {
                key: payload[key] for key in _SAMPLING_FIELDS if key in payload
            },
            "cache": payload["cache"],
            "response_format": payload["response_format"],
            "request_time": started,
            "accepted": False,
        }
        try:
            with self.opener.open(http_request, timeout=self.timeout) as response:
                record["http_status"] = response.status
                envelope = _strict_json(response.read())
            if not isinstance(envelope, dict):
                raise ClientError("API response envelope must be a JSON object")
            record["response"] = {
                key: value for key, value in envelope.items() if key in _RESPONSE_FIELDS
            }
            choices = envelope.get("choices")
            if not isinstance(choices, list) or len(choices) != 1:
                raise ClientError("Expected exactly one model response choice")
            choice = choices[0]
            if not isinstance(choice, dict) or choice.get("finish_reason") != "stop":
                raise ClientError(
                    "Model response was truncated or did not finish normally"
                )
            message = choice.get("message")
            if (
                not isinstance(message, dict)
                or message.get("role") != "assistant"
                or message.get("refusal")
            ):
                raise ClientError("Model did not return an assistant proposal")
            raw = message.get("content")
            if not isinstance(raw, str) or not raw.strip():
                raise ClientError("Model response content must be nonempty JSON text")
            _strict_json(raw)
            model_version = envelope.get("model")
            if not isinstance(model_version, str) or not model_version.strip():
                raise ClientError("API response lacks model identity")
            if model_version != self.model:
                raise ClientError(
                    "API returned a different model than the configured experiment"
                )
            parse_proposal(
                raw,
                decoded_request,
                model_version=model_version,
                request_time=started,
                response_time=time.time(),
                max_timeout=request["max_timeout_env_steps"],
                completion_types=request["supported_completion_types"],
            )
            record["accepted"] = True
            return raw
        except urllib.error.HTTPError as exc:
            record["http_status"] = exc.code
            record["error"] = f"Inference endpoint returned HTTP {exc.code}"
            raise ClientError(record["error"]) from None
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            record["error"] = f"Inference transport failed ({type(exc).__name__})"
            raise ClientError(record["error"]) from None
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            record["error"] = _redact(f"Proposal rejected: {exc}", key)
            raise ClientError(record["error"]) from None
        finally:
            record["response_time"] = time.time()
            record["latency_seconds"] = time.perf_counter() - monotonic
            _append_record(self.response_log, _redact(record, key))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--endpoint", default=os.environ.get("ASTRA_ENDPOINT", DEFAULT_ENDPOINT)
    )
    parser.add_argument("--model", default=os.environ.get("ASTRA_MODEL"))
    parser.add_argument("--response-log", default=os.environ.get("ASTRA_RESPONSE_LOG"))
    parser.add_argument("--timeout", type=float, default=110.0)
    parser.add_argument("--max-completion-tokens", type=int)
    parser.add_argument("--reasoning-effort")
    parser.add_argument("--temperature", type=float)
    args = parser.parse_args(argv)
    sampling = {
        name: value
        for name, value in {
            "max_completion_tokens": args.max_completion_tokens,
            "reasoning_effort": args.reasoning_effort,
            "temperature": args.temperature,
        }.items()
        if value is not None
    }
    try:
        client = AstraHTTPClient(
            model=args.model,
            endpoint=args.endpoint,
            response_log=args.response_log,
            timeout=args.timeout,
            sampling=sampling,
        )
        raw = client.generate(_strict_json(sys.stdin.read()))
    except (ClientError, ValueError, TypeError, OSError) as exc:
        # No traceback, request headers or provider error body on the command channel.
        error = _redact(
            str(exc), os.environ.get("NVIDIA_INFERENCE_API_KEY", "").strip()
        )
        print(f"astra_client: {error}", file=sys.stderr)
        return 1
    print(raw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
