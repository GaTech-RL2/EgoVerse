"""Bounded Astra proposals for observed, iterative intervention experiments.

This client proposes candidate parameters, never policy actions. Application,
episode resets, observed success, incumbent selection and iteration budgets
belong to the experiment runner. One ``propose`` call makes at most one physical
provider request. A failed request consumes a runner iteration; there is no
retry, response repair, model substitution or generated fallback here.

The inherited transport uses the already verified NVIDIA OpenAI-compatible
endpoint, environment credential and no-redirect opener in ``astra_client``.
Token ledgers retain available usage from rejected completions and HTTP errors.
Missing usage stays missing; reasoning tokens are part of output tokens.
"""

import base64
import binascii
import http.client
import io
import json
import math
import os
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

from PIL import Image

from .agent import CAMERAS, _wire_value
from .astra_client import (
    DEFAULT_ENDPOINT,
    AstraHTTPClient,
    ClientError,
    _append_record,
    _redact,
    _strict_json,
)
from .records import digest

SCHEMA_VERSION = "intervention-1.0"
PROMPT_TEMPLATE_VERSION = "astra-intervention-http-2"
ARM_CHANNELS = {
    "noise_only": ("noise",),
    "language_only": ("language",),
    "vision_only": ("vision",),
    "noise_language": ("noise", "language"),
    "noise_vision": ("noise", "vision"),
    "language_vision": ("language", "vision"),
    "joint": ("noise", "language", "vision"),
}
LIMITS = {
    "noise_rank": 8,
    "noise_coefficient_l2_max": 1.0,
    "max_noise_scale": 0.5,
    "max_language_chars": 160,
    "max_language_scale": 1.0,
    "language_embedding_delta_norm_fraction": 0.25,
    "language_guidance_token_budget": 200,
    "max_annotations": 8,
    "max_annotations_per_camera": 4,
    "max_vision_gain": 1.0,
    "max_observations": 4,
}
SYSTEM_PROMPT = """You propose bounded interventions for a fixed pretrained
robot policy, using only the original task, raw camera images, proprioception,
and the supplied observed rollout feedback. No demonstrations, simulator object
poses, goal predicates or hidden state are available. Binary observed success
is the only objective outcome. You may assess visual progress in your rationale
but must not invent a measured progress score or claim success without feedback.

Each candidate is tested in a full closed-loop episode reset to the same initial
state. Attempt 1 is the common unmodified baseline. Attempts 2 through 5 may
revise a candidate based on observations. Provider failures consume an attempt;
they do not trigger hidden retries. Compare previous candidates with the last
observed rollout; do not assume that a numerically larger perturbation helps.

The enabled intervention channels are fixed by arm. Supply all enabled channels
and no disabled channels. Zero scale or gain explicitly represents a neutral
intervention. Parameters are ABSOLUTE relative to the original condition and
recovered original noise z0, never cumulative edits of a previous candidate.
To repeat a prior candidate, explicitly return the same intervention parameters
with a new candidate_id. best_candidate_id is your assessment of the strongest
previous or newly proposed candidate, not a switch that applies hidden edits.

NOISE: supply eight coefficients with Euclidean norm <=1 and a perturbation
scale <=0.5. The runner uses one fixed orthogonal Gaussian basis per episode,
whose full-state vectors have unit RMS. Delta noise is scale*sum(coeff[i]*B[i]),
added to recovered original z0. The basis axes have NO guaranteed physical
motion semantics. You cannot equate an axis to XYZ, rotation or gripper action.
Use observed behavior to revise coefficients. Do not change the basis or seed.

LANGUAGE: supply a short target_text and scale alpha in [0,1]. The runner uses
actual policy text embeddings: alpha*(mean E(original+'\\nGuidance: '+target_text)
-mean E(original)), broadcast only over original valid text slots. The residual
is bounded to 25% of the original text embedding norm. Original text masks,
padding and vision tokens stay fixed. Guidance must fit 200 model tokens and
160 characters; an over-budget target is rejected, never silently truncated.

VISION: supply source-camera point or ordered box annotations with gain [0,1].
Pixel x increases right and y down; remain inside the exact raw image dimensions.
The runner repeats fixed magenta marks at these static pixel coordinates each
step (3px box outline, 4px point radius, blended by gain). There is no tracker.
Favor stable goal/receptacle regions in observation/image when useful; wrist
camera content changes rapidly. Frames sent to you are always unmodified.

Use the exact versioned action_spec only as controller context. Do not produce
continuous actions or reinterpret latent coefficients as controller deltas.
observation/state is world eef XYZ, eef axis-angle rotation, then two finger
positions. Return exactly one JSON object satisfying response_schema, echoing
the request identity fields. No additional fields, markdown, code or prose.
"""

_REQUEST_FIELDS = {
    "schema_version",
    "episode_id",
    "iteration",
    "max_iterations",
    "arm",
    "task_instruction",
    "action_spec",
    "observations",
    "prior_candidates",
    "basis_id",
    "incumbent_candidate_id",
    "limits",
    "request_fingerprint",
}
_IDENTITY_FIELDS = (
    "schema_version",
    "episode_id",
    "iteration",
    "arm",
    "request_fingerprint",
)
_PROPOSAL_FIELDS = set(_IDENTITY_FIELDS) | {
    "candidate_id",
    "best_candidate_id",
    "rationale",
    "language",
    "vision",
    "noise",
}
_TOKEN_FIELDS = ("input_tokens", "output_tokens", "reasoning_tokens", "total_tokens")
_SAMPLING_FIELDS = {
    "max_completion_tokens",
    "reasoning_effort",
    "temperature",
    "top_p",
    "seed",
}


def _object(value, fields, name):
    if not isinstance(value, dict) or set(value) != set(fields):
        raise ClientError(f"{name} must contain exactly its declared fields")


def _text(value, name, maximum=256):
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ClientError(
            f"{name} must be nonempty text of at most {maximum} characters"
        )
    return value


def _number(value, name, lower, upper):
    if (
        type(value) not in (int, float)
        or not lower <= value <= upper
        or not math.isfinite(value)
    ):
        raise ClientError(f"{name} must be finite and in [{lower}, {upper}]")
    return value


def _integer(value, name, lower, upper):
    if type(value) is not int or not lower <= value <= upper:
        raise ClientError(f"{name} must be an integer in [{lower}, {upper}]")


def _json_copy(value, name):
    try:
        return _strict_json(json.dumps(_wire_value(value), allow_nan=False))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ClientError(f"{name} must be finite JSON data") from exc


def _camera_dimensions(observations):
    if (
        not isinstance(observations, list)
        or not 1 <= len(observations) <= LIMITS["max_observations"]
    ):
        raise ClientError("Supply one to four paired raw observation snapshots")
    dimensions = {}
    for snapshot in observations:
        _object(snapshot, ("label", "step", "observation"), "snapshot")
        _text(snapshot["label"], "snapshot label", 128)
        _integer(snapshot["step"], "snapshot step", 0, 1000000)
        observation = snapshot["observation"]
        _object(observation, (*CAMERAS, "observation/state"), "observation")
        state = observation["observation/state"]
        if not isinstance(state, list) or len(state) != 8:
            raise ClientError("observation/state must contain eight numeric values")
        for value in state:
            _number(value, "proprioception", -1e10, 1e10)
        for camera in CAMERAS:
            wire = observation[camera]
            _object(wire, ("encoding", "data"), "camera")
            if wire["encoding"] != "base64_png" or not isinstance(wire["data"], str):
                raise ClientError("Cameras must be unmodified RGB PNG observations")
            if len(wire["data"]) > 8 * 1024 * 1024:
                raise ClientError("Camera PNG exceeds the input size limit")
            try:
                data = base64.b64decode(wire["data"], validate=True)
                with Image.open(io.BytesIO(data)) as image:
                    if image.format != "PNG" or image.mode != "RGB":
                        raise ClientError("Cameras must be RGB PNG observations")
                    width, height = image.size
                    if not 1 <= width <= 4096 or not 1 <= height <= 4096:
                        raise ClientError("Camera dimensions exceed the input limit")
                    image.load()
            except (ValueError, OSError, binascii.Error) as exc:
                raise ClientError("Invalid camera PNG") from exc
            if camera in dimensions and dimensions[camera] != (width, height):
                raise ClientError(
                    "Static annotations require consistent camera dimensions"
                )
            dimensions[camera] = (width, height)
    return dimensions


def _validate_request(request):
    _object(request, _REQUEST_FIELDS, "intervention request")
    if request["schema_version"] != SCHEMA_VERSION or request["limits"] != LIMITS:
        raise ClientError("Unsupported intervention contract or altered limits")
    _text(request["episode_id"], "episode_id")
    _integer(request["max_iterations"], "max_iterations", 2, 5)
    _integer(request["iteration"], "iteration", 2, request["max_iterations"])
    if not isinstance(request["arm"], str) or request["arm"] not in ARM_CHANNELS:
        raise ClientError("Unknown intervention arm")
    _text(request["task_instruction"], "task_instruction", 10000)
    _text(request["basis_id"], "basis_id")
    spec = request["action_spec"]
    if not isinstance(spec, dict) or spec.get("schema_version") != "1.0":
        raise ClientError("A versioned exact action_spec is required")
    _text(spec.get("action_spec_id"), "action_spec_id")
    prior = request["prior_candidates"]
    if not isinstance(prior, list) or len(prior) != request["iteration"] - 1:
        raise ClientError("Provide the baseline and one result for every prior attempt")
    ids, iterations = set(), set()
    for row in prior:
        _object(
            row, ("candidate_id", "iteration", "proposal", "outcome"), "prior candidate"
        )
        candidate_id = _text(row["candidate_id"], "prior candidate_id", 128)
        _integer(row["iteration"], "prior iteration", 1, request["iteration"] - 1)
        if candidate_id in ids or row["iteration"] in iterations:
            raise ClientError("Prior candidate IDs and iterations must be unique")
        ids.add(candidate_id)
        iterations.add(row["iteration"])
        if row["proposal"] is not None and not isinstance(row["proposal"], dict):
            raise ClientError("Prior proposal must be an object or null")
        outcome = row["outcome"]
        required = {"success", "executed_steps", "termination", "error"}
        if not isinstance(outcome, dict) or not required <= set(outcome) <= required | {
            "visual_progress"
        }:
            raise ClientError(
                "Outcome must contain observed success, steps, termination and error"
            )
        if outcome["success"] is not None and type(outcome["success"]) is not bool:
            raise ClientError("Observed success must be boolean or unavailable")
        _integer(outcome["executed_steps"], "executed_steps", 0, 1000000)
        _text(outcome["termination"], "termination", 128)
        if outcome["error"] is not None:
            _text(outcome["error"], "outcome error", 2048)
        if "visual_progress" in outcome:
            _text(outcome["visual_progress"], "visual_progress", 2048)
    if 1 not in iterations:
        raise ClientError("Observed baseline attempt 1 is required")
    if request["incumbent_candidate_id"] is not None:
        _text(request["incumbent_candidate_id"], "incumbent_candidate_id", 128)
        if request["incumbent_candidate_id"] not in ids:
            raise ClientError("Incumbent must reference a supplied prior candidate")
    if (
        digest({k: v for k, v in request.items() if k != "request_fingerprint"})
        != request["request_fingerprint"]
    ):
        raise ClientError(
            "Intervention request fingerprint does not match its contents"
        )
    return _camera_dimensions(request["observations"])


def build_request(
    *,
    episode_id,
    iteration,
    arm,
    task_instruction,
    action_spec,
    observations,
    prior_candidates,
    basis_id,
    incumbent_candidate_id=None,
    max_iterations=5,
):
    """Freeze raw paired snapshots and observed history into a fingerprinted request.

    ``observations`` entries are ``{label, step, observation}``; observation has
    both ``CAMERAS`` and the eight-value ``observation/state``. NumPy RGB arrays
    are losslessly PNG encoded. Prior entries are ``{candidate_id, iteration,
    proposal, outcome}``; outcome contains ``success`` (bool or unavailable),
    ``executed_steps``, ``termination``, ``error`` and optional ``visual_progress``.
    """
    request = _json_copy(
        {
            "schema_version": SCHEMA_VERSION,
            "episode_id": episode_id,
            "iteration": iteration,
            "max_iterations": max_iterations,
            "arm": arm,
            "task_instruction": task_instruction,
            "action_spec": action_spec,
            "observations": observations,
            "prior_candidates": prior_candidates,
            "basis_id": basis_id,
            "incumbent_candidate_id": incumbent_candidate_id,
            "limits": LIMITS,
        },
        "intervention request",
    )
    request["request_fingerprint"] = digest(request)
    _validate_request(request)
    return request


def response_schema(request):
    """Describe the same contract enforced by ``parse_proposal``."""

    def obj(properties):
        return {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        }

    def number(upper=1.0, lower=0.0):
        return {"type": "number", "minimum": lower, "maximum": upper}

    def array(items, count):
        return {"type": "array", "items": items, "minItems": count, "maxItems": count}

    enabled = ARM_CHANNELS[request["arm"]]
    properties = {field: {"const": request[field]} for field in _IDENTITY_FIELDS}
    properties.update(
        candidate_id={"type": "string", "minLength": 1, "maxLength": 128},
        best_candidate_id={"type": "string", "minLength": 1, "maxLength": 128},
        rationale={"type": "string", "minLength": 1, "maxLength": 2048},
        language=obj(
            {
                "target_text": {"type": "string", "minLength": 1, "maxLength": 160},
                "scale": number(),
            }
        )
        if "language" in enabled
        else {"type": "null"},
        noise=obj(
            {
                "basis_id": {"const": request["basis_id"]},
                "coefficients": array(number(lower=-1), 8),
                "perturbation_scale": number(0.5),
            }
        )
        if "noise" in enabled
        else {"type": "null"},
        vision={
            "type": "array",
            "minItems": 1 if "vision" in enabled else 0,
            "maxItems": 8 if "vision" in enabled else 0,
            "items": {
                "anyOf": [
                    obj(
                        {
                            "camera": {"enum": list(CAMERAS)},
                            "kind": {"const": kind},
                            "coordinates": array(
                                {"type": "number", "minimum": 0}, count
                            ),
                            "gain": number(),
                        }
                    )
                    for kind, count in (("point", 2), ("box", 4))
                ]
            },
        },
    )
    return obj(properties)


def parse_proposal(raw, request):
    """Validate without clipping, truncating, repairing or selecting a fallback."""
    dimensions = _validate_request(request)
    proposal = (
        _strict_json(raw)
        if isinstance(raw, (str, bytes))
        else _json_copy(raw, "proposal")
    )
    _object(proposal, _PROPOSAL_FIELDS, "proposal")
    for field in _IDENTITY_FIELDS:
        if (
            type(proposal[field]) is not type(request[field])
            or proposal[field] != request[field]
        ):
            raise ClientError(f"Proposal does not echo {field}")
    candidate_id = _text(proposal["candidate_id"], "candidate_id", 128)
    prior_ids = {row["candidate_id"] for row in request["prior_candidates"]}
    if candidate_id in prior_ids:
        raise ClientError("Each proposal requires a new candidate_id")
    best_id = _text(proposal["best_candidate_id"], "best_candidate_id", 128)
    if best_id not in prior_ids | {candidate_id}:
        raise ClientError(
            "best_candidate_id must reference a supplied or proposed candidate"
        )
    _text(proposal["rationale"], "rationale", 2048)
    enabled = ARM_CHANNELS[request["arm"]]
    for channel in ("language", "noise", "vision"):
        if channel not in enabled and proposal[channel] != (
            [] if channel == "vision" else None
        ):
            raise ClientError(f"The {channel} channel is disabled for this arm")
    if "language" in enabled:
        language = proposal["language"]
        _object(language, ("target_text", "scale"), "language")
        _text(language["target_text"], "target_text", 160)
        _number(language["scale"], "language scale", 0, 1)
    if "noise" in enabled:
        noise = proposal["noise"]
        _object(noise, ("basis_id", "coefficients", "perturbation_scale"), "noise")
        if noise["basis_id"] != request["basis_id"]:
            raise ClientError("Noise basis_id differs from the fixed episode basis")
        coefficients = noise["coefficients"]
        if not isinstance(coefficients, list) or len(coefficients) != 8:
            raise ClientError("Noise requires exactly eight coefficients")
        for value in coefficients:
            _number(value, "noise coefficient", -1, 1)
        if math.sqrt(math.fsum(value * value for value in coefficients)) > 1.0:
            raise ClientError("Noise coefficient L2 norm exceeds one")
        _number(noise["perturbation_scale"], "noise perturbation_scale", 0, 0.5)
    vision = proposal["vision"]
    if (
        not isinstance(vision, list)
        or len(vision) > 8
        or ("vision" in enabled and not vision)
    ):
        raise ClientError("Vision requires one to eight annotations when enabled")
    counts = Counter()
    for annotation in vision:
        _object(annotation, ("camera", "kind", "coordinates", "gain"), "annotation")
        camera, kind = annotation["camera"], annotation["kind"]
        if camera not in CAMERAS or kind not in ("point", "box"):
            raise ClientError("Unknown source camera or annotation kind")
        counts[camera] += 1
        if counts[camera] > 4:
            raise ClientError("At most four annotations are allowed per camera")
        coordinates = annotation["coordinates"]
        if not isinstance(coordinates, list) or len(coordinates) != (
            2 if kind == "point" else 4
        ):
            raise ClientError("Annotation coordinates have the wrong shape")
        width, height = dimensions[camera]
        for index, value in enumerate(coordinates):
            _number(value, "pixel coordinate", 0, width if index % 2 == 0 else height)
            if value >= (width if index % 2 == 0 else height):
                raise ClientError("Annotation lies outside its raw source image")
        if kind == "box" and (
            coordinates[0] >= coordinates[2] or coordinates[1] >= coordinates[3]
        ):
            raise ClientError("Box corners must have positive ordered extent")
        _number(annotation["gain"], "vision gain", 0, 1)
    return proposal


def build_payload(request, model, *, sampling=None):
    dimensions = _validate_request(request)
    settings = {
        "max_completion_tokens": 8192,
        "reasoning_effort": "low",
        **(sampling or {}),
    }
    if set(settings) - _SAMPLING_FIELDS:
        raise ClientError("Unsupported HTTP sampling setting")
    _integer(settings["max_completion_tokens"], "max_completion_tokens", 1, 1000000)
    if settings["reasoning_effort"] not in ("low", "medium", "high"):
        raise ClientError("Unsupported reasoning_effort")
    for name, upper in (("temperature", 2), ("top_p", 1)):
        if name in settings:
            _number(settings[name], name, 0, upper)
    if "seed" in settings:
        _integer(settings["seed"], "seed", 0, 2**63 - 1)
    context = {key: value for key, value in request.items() if key != "observations"}
    context["observations"] = []
    image_parts = []
    for index, snapshot in enumerate(request["observations"]):
        observation = snapshot["observation"]
        described = {"observation/state": observation["observation/state"]}
        for camera in CAMERAS:
            width, height = dimensions[camera]
            described[camera] = {
                "width": width,
                "height": height,
                "encoding": "attached_png",
                "image_index": len(image_parts) // 2,
            }
            image_parts.extend(
                [
                    {
                        "type": "text",
                        "text": f"Raw snapshot {index}: {snapshot['label']}; step {snapshot['step']}; camera {camera}",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,"
                            + observation[camera]["data"]
                        },
                    },
                ]
            )
        context["observations"].append({**snapshot, "observation": described})
    content = [
        {
            "type": "text",
            "text": json.dumps(
                {
                    "prompt_template_version": PROMPT_TEMPLATE_VERSION,
                    # The gateway validates converted user input separately
                    # from system instructions before accepting json_object.
                    "response_instructions": "Return exactly one valid json object matching response_schema.",
                    "request": context,
                    "response_schema": response_schema(request),
                },
                allow_nan=False,
            ),
        },
        *image_parts,
    ]
    return {
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


def normalize_usage(usage):
    """Return reported nonnegative integer counts; unknown/conflicting stays null."""
    usage = usage if isinstance(usage, dict) else {}

    def count(*values):
        present = [value for value in values if value is not None]
        if not present or any(type(value) is not int or value < 0 for value in present):
            return None
        return present[0] if all(value == present[0] for value in present) else None

    output = count(usage.get("completion_tokens"), usage.get("output_tokens"))
    details = [
        usage.get(name)
        for name in ("completion_tokens_details", "output_tokens_details")
    ]
    reasoning = count(
        *(value.get("reasoning_tokens") for value in details if isinstance(value, dict))
    )
    if output is not None and reasoning is not None and reasoning > output:
        reasoning = None
    return {
        "input_tokens": count(usage.get("prompt_tokens"), usage.get("input_tokens")),
        "output_tokens": output,
        "reasoning_tokens": reasoning,
        "total_tokens": count(usage.get("total_tokens")),
    }


class InterventionClient(AstraHTTPClient):
    """One genuine request per candidate, with exact provider model identity."""

    def __init__(
        self,
        *,
        model,
        response_log,
        endpoint=DEFAULT_ENDPOINT,
        timeout=170.0,
        sampling=None,
    ):
        super().__init__(
            model=model,
            response_log=response_log,
            endpoint=endpoint,
            timeout=timeout,
            sampling=sampling,
        )

    def propose(self, request):
        payload = build_payload(request, self.model, sampling=self.sampling)
        key = os.environ.get("NVIDIA_INFERENCE_API_KEY", "").strip()
        if not key:
            raise ClientError("NVIDIA_INFERENCE_API_KEY is not set")
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
            "client_schema_version": SCHEMA_VERSION,
            "prompt_template_version": PROMPT_TEMPLATE_VERSION,
            **{field: request[field] for field in _IDENTITY_FIELDS},
            "endpoint": self.endpoint,
            "requested_model": self.model,
            "sampling_settings": {
                name: payload[name] for name in _SAMPLING_FIELDS if name in payload
            },
            "cache": payload["cache"],
            "response_format": payload["response_format"],
            "request_time": started,
            "provider_call": True,
            "accepted": False,
            "token_usage": normalize_usage(None),
        }
        try:
            with self.opener.open(http_request, timeout=self.timeout) as response:
                record["http_status"] = response.status
                envelope = _strict_json(response.read())
            if not isinstance(envelope, dict):
                raise ClientError("API response envelope must be a JSON object")
            record["response"] = {
                field: envelope[field]
                for field in (
                    "id",
                    "object",
                    "created",
                    "model",
                    "choices",
                    "usage",
                    "system_fingerprint",
                    "service_tier",
                )
                if field in envelope
            }
            record["token_usage"] = normalize_usage(envelope.get("usage"))
            if envelope.get("model") != self.model:
                raise ClientError(
                    "API returned a different model than the configured experiment"
                )
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
            proposal = parse_proposal(raw, request)
            record["candidate_id"] = proposal["candidate_id"]
            record["accepted"] = True
            return proposal
        except urllib.error.HTTPError as exc:
            record["http_status"] = exc.code
            # Failed calls can still be billable. Retain standard usage/identity
            # fields only, never provider error bodies, headers or credentials.
            try:
                envelope = _strict_json(exc.read(1024 * 1024))
                if isinstance(envelope, dict):
                    record["response"] = {
                        field: envelope[field]
                        for field in ("id", "model", "usage")
                        if field in envelope
                    }
                    record["token_usage"] = normalize_usage(envelope.get("usage"))
            except (ValueError, OSError, TypeError, http.client.HTTPException):
                pass
            finally:
                exc.close()
            record["error_kind"] = "http_error"
            record["error"] = f"Inference endpoint returned HTTP {exc.code}"
            raise ClientError(record["error"]) from None
        except (
            urllib.error.URLError,
            OSError,
            TimeoutError,
            http.client.HTTPException,
        ) as exc:
            record["error_kind"] = "transport_error"
            record["error"] = f"Inference transport failed ({type(exc).__name__})"
            raise ClientError(record["error"]) from None
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            record["error_kind"] = "proposal_rejected"
            record["error"] = _redact(f"Proposal rejected: {exc}", key)
            raise ClientError(record["error"]) from None
        finally:
            record["response_time"] = time.time()
            record["latency_seconds"] = time.perf_counter() - monotonic
            _append_record(self.response_log, _redact(record, key))


def summarize_calls(records_or_path):
    """Aggregate real attempts, including failed calls, without guessing usage/rates.

    Each token field reports its sum over available records and missing-call
    count. A partial sum is not the complete experiment token cost. Summing input
    and output is valid; adding reasoning again would double count it.
    """
    if isinstance(records_or_path, (str, Path)):
        path = Path(records_or_path)
        records = [
            _strict_json(line) for line in path.read_text().splitlines() if line.strip()
        ]
    else:
        records = list(records_or_path)
    for row in records:
        if (
            not isinstance(row, dict)
            or row.get("client_schema_version") != SCHEMA_VERSION
            or row.get("provider_call") is not True
            or type(row.get("accepted")) is not bool
        ):
            raise ClientError(
                "Token ledger contains a non-provider or incompatible record"
            )
        response = row.get("response", {})
        if not isinstance(response, dict):
            raise ClientError("Recorded provider response must be an object")
        if row.get("token_usage") != normalize_usage(response.get("usage")):
            raise ClientError(
                "Normalized token ledger disagrees with recorded provider usage"
            )
        _number(row.get("latency_seconds"), "recorded latency", 0, 1e10)
    total, accepted = len(records), sum(row["accepted"] for row in records)
    tokens = {}
    for field in _TOKEN_FIELDS:
        available = [
            row["token_usage"][field]
            for row in records
            if row["token_usage"][field] is not None
        ]
        tokens[field] = {
            "sum": sum(available),
            "available_calls": len(available),
            "missing_calls": total - len(available),
            "complete": len(available) == total,
        }
    usage_available = sum(
        any(value is not None for value in row["token_usage"].values())
        for row in records
    )
    return {
        "provider_calls": total,
        "accepted_proposals": accepted,
        "failed_calls": total - accepted,
        "usage_available_calls": usage_available,
        "usage_unavailable_calls": total - usage_available,
        "tokens": tokens,
        "reasoning_tokens_are_subset_of_output_tokens": True,
        "latency_seconds": sum(row["latency_seconds"] for row in records),
        "actual_models": dict(
            Counter(
                model
                if isinstance(model := row.get("response", {}).get("model"), str)
                else "unavailable"
                for row in records
            )
        ),
        "http_status": dict(
            Counter(str(row.get("http_status", "unavailable")) for row in records)
        ),
        "errors": dict(
            Counter(
                row.get("error", "unclassified_failure")
                for row in records
                if not row["accepted"]
            )
        ),
        "monetary_cost": None,
        "monetary_cost_unavailable_reason": "No verified provider price was supplied.",
    }
