"""Observed phase decisions for fixed donor-prompt interpolation experiments.

One ``propose`` invocation makes at most one physical provider request. The
runner applies accepted parameters, preserves valid text on failure, expires
vision, and measures success. This module never executes actions or retries.
It reuses the verified endpoint, credential handling and usage accounting of
the existing clients, without changing their frozen protocols.
"""

import http.client
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

from .agent import CAMERAS
from .astra_client import (
    DEFAULT_ENDPOINT,
    AstraHTTPClient,
    ClientError,
    _append_record,
    _redact,
    _strict_json,
)
from .intervention_agent import (
    SCHEMA_VERSION as _USAGE_LEDGER_VERSION,
)
from .intervention_agent import (
    _camera_dimensions,
    _integer,
    _json_copy,
    _number,
    _object,
    _text,
    normalize_usage,
)
from .intervention_agent import (
    summarize_calls as _summarize_provider_calls,
)
from .records import digest

SCHEMA_VERSION = "interpolation-1.0"
PROMPT_TEMPLATE_VERSION = "astra-interpolation-http-1"
INTERPOLATION_MODES = ("tei", "tli", "tei_tli")
LIMITS = {
    "max_observations": 4,
    "max_previous_observations": 4,
    "max_annotations": 8,
    "max_annotations_per_camera": 4,
    "max_vision_gain": 1.0,
    "vision_chunk_actions": 5,
}
SYSTEM_PROMPT = """You choose phase-dependent conditioning for a fixed pretrained
robot policy. Use only the target task, supplied donor-prompt catalog, raw camera
images, robot proprioception, past decisions/errors and observed rollout outcomes.
No oracle source pair, demonstrations, simulator object poses, hidden predicates
or privileged goal state are supplied. Binary observed success is the only
objective outcome. observed_phase and rationale are your visual assessments;
do not invent a measured progress score or treat an assessment as task success.

Choose source_a_id and source_b_id from source_catalog and alpha in [0,1]. They
are ABSOLUTE parameters for this decision, not cumulative edits. Source A and B
may be identical. You may change the pair and move alpha backward after a failed
grasp or when recovery is needed. Infer the useful phase from CURRENT observations;
previous_attempt images describe a completed rollout, never the current state.

The experiment fixes interpolation_mode; you cannot change it:
tei: actual text embedding interpolation (1-alpha)*E_A + alpha*E_B, so alpha is
the weight of source B. The runner preserves the declared conditioning layout.
tli: the paper's text-latent residual update (1-2alpha)*(T_A-T_B). Alpha still
transitions from the A side to the B side, but this is NOT a convex hidden-state
overwrite. Alpha 0.5 gives zero residual; identical sources also give zero.
tei_tli: apply both declared operators with the same source pair and alpha.
Do not output actions, noise coefficients, new source prompts or other edits.

Each attempt resets to its recorded initial state. Within an attempt, calls
occur at the supplied call_interval until max_calls or action_budget. Every
call slot, including a failure, is recorded in previous_decisions. There are no
hidden retries. A rejected/failed call leaves the previous valid text parameters
active. Prior vision annotations expire at the next native policy replan. The current
active_interpolation describes only the last accepted text decision, if any.

When vision_enabled, you may supply fresh point or box annotations in the raw
CURRENT camera pixel coordinates. Use [] for no annotations. Pixel x increases
right and y down. Magenta marks (3px box outline, 4px point radius) are blended
by gain [0,1] for ONLY the current five-action native policy chunk. They expire
at the next policy replan, before the next 25-action provider refresh, and are
not restored between provider calls. There is no tracker. Prefer stable
goal/receptacle regions in observation/image
when useful; wrist camera content changes rapidly. A failed next call clears
vision instead of reusing stale coordinates. Frames shown to you are unmodified.
When vision is disabled, vision must be [].

observation/state contains world eef XYZ, eef axis-angle rotation, and two finger
positions. action_spec, when supplied, is controller context only. Return exactly
one JSON object matching response_schema and echoing every request identity
field. No additional fields, markdown, code, prose, or success command.
"""

_IDENTITY_FIELDS = (
    "schema_version",
    "episode_id",
    "attempt_id",
    "decision_index",
    "observation_step",
    "interpolation_mode",
    "request_fingerprint",
)
_REQUEST_FIELDS = set(_IDENTITY_FIELDS) | {
    "target_task",
    "source_catalog",
    "observations",
    "previous_decisions",
    "completed_rollout_feedback",
    "previous_attempt",
    "active_interpolation",
    "vision_enabled",
    "action_spec",
    "limits",
}
_PROPOSAL_FIELDS = set(_IDENTITY_FIELDS) | {
    "decision_id",
    "source_a_id",
    "source_b_id",
    "alpha",
    "observed_phase",
    "rationale",
    "vision",
}
_INTERPOLATION_FIELDS = ("source_a_id", "source_b_id", "alpha")
_SAMPLING_FIELDS = {
    "max_completion_tokens",
    "reasoning_effort",
    "temperature",
    "top_p",
    "seed",
}


def _fingerprint(value):
    if not isinstance(value, str) or not re.fullmatch("[0-9a-f]{64}", value):
        raise ClientError("request_fingerprint must be a SHA256 digest")


def _snapshots(snapshots, *, last_step=None, maximum_step=None, optional=False):
    if optional and snapshots == []:
        return None
    dimensions = _camera_dimensions(snapshots)
    steps = [row["step"] for row in snapshots]
    if steps != sorted(set(steps)):
        raise ClientError("Raw snapshots must have distinct increasing steps")
    if last_step is not None and steps[-1] != last_step:
        raise ClientError("Last raw snapshot must be the current observation_step")
    if maximum_step is not None and steps[-1] > maximum_step:
        raise ClientError("Raw snapshot is later than the recorded rollout")
    return dimensions


def _feedback(row):
    required = {"attempt_id", "success", "executed_actions", "termination", "error"}
    if not isinstance(row, dict) or not required <= set(row) <= required | {
        "visual_summary"
    }:
        raise ClientError("Rollout feedback contains unsupported or missing fields")
    _text(row["attempt_id"], "feedback attempt_id")
    if type(row["success"]) is not bool:
        raise ClientError("Completed rollout success must be an observed boolean")
    _integer(row["executed_actions"], "executed_actions", 0, 300)
    _text(row["termination"], "termination", 128)
    if row["error"] is not None:
        _text(row["error"], "rollout error", 2048)
    if "visual_summary" in row:
        _text(row["visual_summary"], "visual_summary", 2048)


def _interpolation(value, source_ids):
    for name in ("source_a_id", "source_b_id"):
        if not isinstance(value[name], str) or value[name] not in source_ids:
            raise ClientError(f"{name} must reference the supplied source_catalog")
    _number(value["alpha"], "alpha", 0, 1)


def _vision(annotations, dimensions, enabled):
    if not isinstance(annotations, list) or len(annotations) > 8:
        raise ClientError("Vision must contain zero to eight annotations")
    if not enabled and annotations:
        raise ClientError("Vision annotations are disabled for this arm")
    counts = Counter()
    for annotation in annotations:
        _object(annotation, ("camera", "kind", "coordinates", "gain"), "annotation")
        camera, kind = annotation["camera"], annotation["kind"]
        if camera not in CAMERAS or kind not in ("point", "box"):
            raise ClientError("Unknown raw camera or annotation kind")
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
            bound = width if index % 2 == 0 else height
            _number(value, "pixel coordinate", 0, bound)
            if value >= bound:
                raise ClientError("Annotation lies outside its raw source image")
        if kind == "box" and (
            coordinates[0] >= coordinates[2] or coordinates[1] >= coordinates[3]
        ):
            raise ClientError("Box corners must have positive ordered extent")
        _number(annotation["gain"], "vision gain", 0, 1)


def _proposal(proposal, request, dimensions, *, attempt_id, decision_index, step):
    _object(proposal, _PROPOSAL_FIELDS, "interpolation proposal")
    expected = {
        "schema_version": SCHEMA_VERSION,
        "episode_id": request["episode_id"],
        "attempt_id": attempt_id,
        "decision_index": decision_index,
        "observation_step": step,
        "interpolation_mode": request["interpolation_mode"],
    }
    for name, value in expected.items():
        if type(proposal[name]) is not type(value) or proposal[name] != value:
            raise ClientError(f"Proposal does not echo {name}")
    _fingerprint(proposal["request_fingerprint"])
    _text(proposal["decision_id"], "decision_id", 128)
    _interpolation(proposal, {row["source_id"] for row in request["source_catalog"]})
    _text(proposal["observed_phase"], "observed_phase", 256)
    _text(proposal["rationale"], "rationale", 2048)
    _vision(proposal["vision"], dimensions, request["vision_enabled"])


def _decisions(rows, request, dimensions, *, attempt_id, count=None, end_step=None):
    if not isinstance(rows, list) or len(rows) > request["limits"]["max_calls"]:
        raise ClientError("Too many or invalid previous_decisions")
    if count is not None and len(rows) != count:
        raise ClientError("Supply every previous decision slot, including failures")
    latest, seen_ids, indices = None, set(), []
    for row in rows:
        _object(
            row,
            ("decision_index", "observation_step", "proposal", "accepted", "error"),
            "previous decision",
        )
        index, step = row["decision_index"], row["observation_step"]
        _integer(index, "previous decision_index", 1, request["limits"]["max_calls"])
        _integer(
            step, "previous observation_step", 0, request["limits"]["action_budget"] - 1
        )
        if step != (index - 1) * request["limits"]["call_interval"]:
            raise ClientError("Previous decision step differs from the call schedule")
        if end_step is not None and step > end_step:
            raise ClientError("Previous decision follows the completed rollout")
        if type(row["accepted"]) is not bool:
            raise ClientError("Previous decision accepted must be boolean")
        indices.append(index)
        if row["accepted"]:
            if row["error"] is not None:
                raise ClientError("Accepted previous decisions cannot have errors")
            _proposal(
                row["proposal"],
                request,
                dimensions,
                attempt_id=attempt_id,
                decision_index=index,
                step=step,
            )
            proposal_id = row["proposal"]["decision_id"]
            if proposal_id in seen_ids:
                raise ClientError("Previous decision_id values must be unique")
            seen_ids.add(proposal_id)
            latest = {name: row["proposal"][name] for name in _INTERPOLATION_FIELDS}
        else:
            if row["proposal"] is not None:
                raise ClientError("Rejected previous decisions must have null proposal")
            _text(row["error"], "previous call error", 2048)
    if indices != sorted(set(indices)) or (
        count is not None and indices != list(range(1, count + 1))
    ):
        raise ClientError(
            "Previous decisions must have increasing, complete slot indices"
        )
    return latest, seen_ids


def _validate_request(request):
    _object(request, _REQUEST_FIELDS, "interpolation request")
    if request["schema_version"] != SCHEMA_VERSION:
        raise ClientError("Unsupported interpolation schema")
    for name in ("episode_id", "attempt_id"):
        _text(request[name], name)
    if request["interpolation_mode"] not in INTERPOLATION_MODES:
        raise ClientError("Unknown interpolation_mode")
    if type(request["vision_enabled"]) is not bool:
        raise ClientError("vision_enabled must be boolean")
    _text(request["target_task"], "target_task", 10000)
    limits = request["limits"]
    _object(limits, (*LIMITS, "call_interval", "max_calls", "action_budget"), "limits")
    if any(
        type(limits[name]) is not type(value) or limits[name] != value
        for name, value in LIMITS.items()
    ):
        raise ClientError("Fixed interpolation limits were altered")
    for name, maximum in (
        ("call_interval", 25),
        ("max_calls", 12),
        ("action_budget", 300),
    ):
        _integer(limits[name], name, 1, maximum)
    _integer(request["decision_index"], "decision_index", 1, limits["max_calls"])
    _integer(
        request["observation_step"], "observation_step", 0, limits["action_budget"] - 1
    )
    if (
        request["observation_step"]
        != (request["decision_index"] - 1) * limits["call_interval"]
    ):
        raise ClientError("observation_step differs from the declared call schedule")
    catalog = request["source_catalog"]
    if not isinstance(catalog, list) or not 2 <= len(catalog) <= 128:
        raise ClientError("Supply a bounded source_catalog of donor IDs and prompts")
    ids = set()
    for row in catalog:
        _object(row, ("source_id", "prompt"), "source catalog entry")
        source_id = _text(row["source_id"], "source_id", 128)
        _text(row["prompt"], "donor prompt", 2048)
        if source_id in ids:
            raise ClientError("source_catalog IDs must be unique")
        ids.add(source_id)
    spec = request["action_spec"]
    if spec is not None:
        if not isinstance(spec, dict) or spec.get("schema_version") != "1.0":
            raise ClientError("action_spec must be a versioned specification or null")
        _text(spec.get("action_spec_id"), "action_spec_id")
    dimensions = _snapshots(
        request["observations"], last_step=request["observation_step"]
    )
    active, _ = _decisions(
        request["previous_decisions"],
        request,
        dimensions,
        attempt_id=request["attempt_id"],
        count=request["decision_index"] - 1,
    )
    if request["active_interpolation"] is not None:
        _object(
            request["active_interpolation"],
            _INTERPOLATION_FIELDS,
            "active_interpolation",
        )
        _interpolation(request["active_interpolation"], ids)
    if request["active_interpolation"] != active:
        raise ClientError(
            "active_interpolation must equal the last accepted current-attempt decision"
        )
    feedback = request["completed_rollout_feedback"]
    if not isinstance(feedback, list) or len(feedback) > 16:
        raise ClientError("Supply at most sixteen completed rollout feedback rows")
    prior_ids = set()
    for row in feedback:
        _feedback(row)
        if row["attempt_id"] == request["attempt_id"] or row["attempt_id"] in prior_ids:
            raise ClientError("Completed feedback requires distinct prior attempt IDs")
        prior_ids.add(row["attempt_id"])
    previous = request["previous_attempt"]
    if previous is not None:
        _object(previous, ("feedback", "decisions", "snapshots"), "previous_attempt")
        _feedback(previous["feedback"])
        prior_id = previous["feedback"]["attempt_id"]
        if prior_id == request["attempt_id"]:
            raise ClientError(
                "previous_attempt must refer to a completed different attempt"
            )
        if feedback and previous["feedback"] != feedback[-1]:
            raise ClientError(
                "previous_attempt feedback must match the latest completed feedback"
            )
        previous_dimensions = _snapshots(
            previous["snapshots"],
            maximum_step=previous["feedback"]["executed_actions"],
            optional=True,
        )
        _decisions(
            previous["decisions"],
            request,
            previous_dimensions or dimensions,
            attempt_id=prior_id,
            end_step=previous["feedback"]["executed_actions"],
        )
    _fingerprint(request["request_fingerprint"])
    if (
        digest(
            {
                name: value
                for name, value in request.items()
                if name != "request_fingerprint"
            }
        )
        != request["request_fingerprint"]
    ):
        raise ClientError(
            "Interpolation request fingerprint does not match its contents"
        )
    return dimensions


def build_request(
    *,
    episode_id,
    attempt_id,
    decision_index,
    observation_step,
    interpolation_mode,
    target_task,
    source_catalog,
    observations,
    previous_decisions=(),
    completed_rollout_feedback=(),
    previous_attempt=None,
    active_interpolation=None,
    vision_enabled=False,
    action_spec=None,
    call_interval=25,
    max_calls=12,
    action_budget=300,
):
    """Freeze current observations and bounded prior feedback; no oracle pair.

    Snapshots are ``{label,step,observation}`` with both raw RGB cameras and 8D
    state. Current snapshots end at observation_step. Previous decision rows
    contain ``decision_index,observation_step,proposal,accepted,error``: accepted
    full responses or null proposal plus a failure reason. Every current-attempt
    prior slot is required. ``previous_attempt`` optionally contains ``feedback,
    decisions,snapshots``; its decisions may be an ordered subset. Feedback is
    ``{attempt_id,success,executed_actions,termination,error}`` with an optional
    subjective ``visual_summary``. No previous images become current images.
    """
    request = _json_copy(
        {
            "schema_version": SCHEMA_VERSION,
            "episode_id": episode_id,
            "attempt_id": attempt_id,
            "decision_index": decision_index,
            "observation_step": observation_step,
            "interpolation_mode": interpolation_mode,
            "target_task": target_task,
            "source_catalog": source_catalog,
            "observations": observations,
            "previous_decisions": previous_decisions,
            "completed_rollout_feedback": completed_rollout_feedback,
            "previous_attempt": previous_attempt,
            "active_interpolation": active_interpolation,
            "vision_enabled": vision_enabled,
            "action_spec": action_spec,
            "limits": {
                **LIMITS,
                "call_interval": call_interval,
                "max_calls": max_calls,
                "action_budget": action_budget,
            },
        },
        "interpolation request",
    )
    request["request_fingerprint"] = digest(request)
    _validate_request(request)
    return request


def response_schema(request):
    """JSON schema for the locally enforced proposal (additional fields forbidden)."""

    def obj(properties):
        return {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        }

    properties = {name: {"const": request[name]} for name in _IDENTITY_FIELDS}
    sources = [row["source_id"] for row in request["source_catalog"]]
    properties.update(
        decision_id={"type": "string", "minLength": 1, "maxLength": 128},
        source_a_id={"enum": sources},
        source_b_id={"enum": sources},
        alpha={"type": "number", "minimum": 0, "maximum": 1},
        observed_phase={"type": "string", "minLength": 1, "maxLength": 256},
        rationale={"type": "string", "minLength": 1, "maxLength": 2048},
        vision={
            "type": "array",
            "maxItems": 8 if request["vision_enabled"] else 0,
            "items": {
                "anyOf": [
                    obj(
                        {
                            "camera": {"enum": list(CAMERAS)},
                            "kind": {"const": kind},
                            "coordinates": {
                                "type": "array",
                                "minItems": count,
                                "maxItems": count,
                                "items": {"type": "number", "minimum": 0},
                            },
                            "gain": {"type": "number", "minimum": 0, "maximum": 1},
                        }
                    )
                    for kind, count in (("point", 2), ("box", 4))
                ]
            },
        },
    )
    return obj(properties)


def parse_proposal(raw, request):
    """Reject stale/invalid responses without repair or generated fallback."""
    dimensions = _validate_request(request)
    proposal = (
        _strict_json(raw)
        if isinstance(raw, (str, bytes))
        else _json_copy(raw, "proposal")
    )
    _proposal(
        proposal,
        request,
        dimensions,
        attempt_id=request["attempt_id"],
        decision_index=request["decision_index"],
        step=request["observation_step"],
    )
    if proposal["request_fingerprint"] != request["request_fingerprint"]:
        raise ClientError("Proposal does not echo request_fingerprint")
    prior_ids = {
        row["proposal"]["decision_id"]
        for row in request["previous_decisions"]
        if row["accepted"]
    }
    if request["previous_attempt"] is not None:
        prior_ids.update(
            row["proposal"]["decision_id"]
            for row in request["previous_attempt"]["decisions"]
            if row["accepted"]
        )
    if proposal["decision_id"] in prior_ids:
        raise ClientError("Each proposal requires a fresh decision_id")
    return proposal


def build_payload(request, model, *, sampling=None):
    _validate_request(request)
    _text(model, "model")
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
    image_parts = []

    def attach(snapshots, scope):
        described = []
        if not snapshots:
            return described
        dimensions = _camera_dimensions(snapshots)
        for snapshot in snapshots:
            observation = {
                "observation/state": snapshot["observation"]["observation/state"]
            }
            for camera in CAMERAS:
                width, height = dimensions[camera]
                observation[camera] = {
                    "width": width,
                    "height": height,
                    "encoding": "attached_png",
                    "image_index": len(image_parts) // 2,
                }
                image_parts.extend(
                    [
                        {
                            "type": "text",
                            "text": f"{scope}: {snapshot['label']}; step {snapshot['step']}; raw camera {camera}",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,"
                                + snapshot["observation"][camera]["data"]
                            },
                        },
                    ]
                )
            described.append({**snapshot, "observation": observation})
        return described

    context = {
        name: value
        for name, value in request.items()
        if name not in ("observations", "previous_attempt")
    }
    previous = request["previous_attempt"]
    context["previous_attempt"] = (
        None
        if previous is None
        else {
            **previous,
            "snapshots": attach(
                previous["snapshots"],
                f"Completed previous attempt {previous['feedback']['attempt_id']}",
            ),
        }
    )
    context["observations"] = attach(
        request["observations"], f"CURRENT attempt {request['attempt_id']}"
    )
    content = [
        {
            "type": "text",
            "text": json.dumps(
                {
                    "prompt_template_version": PROMPT_TEMPLATE_VERSION,
                    # The verified gateway requires 'json' in user text, not only system text.
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


class InterpolationClient(AstraHTTPClient):
    """Exactly configured model, no retry, and a ledger row for every invocation."""

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
        key = os.environ.get("NVIDIA_INFERENCE_API_KEY", "").strip()
        started, monotonic = time.time(), time.perf_counter()
        record = {
            "client_schema_version": SCHEMA_VERSION,
            "prompt_template_version": PROMPT_TEMPLATE_VERSION,
            "endpoint": self.endpoint,
            "requested_model": self.model,
            "request_time": started,
            "provider_call": False,
            "accepted": False,
            "token_usage": normalize_usage(None),
        }
        # Retain usable identity even if a malformed request fails preflight.
        if isinstance(request, dict):
            record.update(
                {
                    name: request[name]
                    for name in _IDENTITY_FIELDS
                    if name in request and type(request[name]) in (str, int)
                }
            )
        try:
            payload = build_payload(request, self.model, sampling=self.sampling)
            record.update({name: request[name] for name in _IDENTITY_FIELDS})
            record.update(
                sampling_settings={
                    name: payload[name] for name in _SAMPLING_FIELDS if name in payload
                },
                cache=payload["cache"],
                response_format=payload["response_format"],
            )
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
            record["provider_call"] = True
            with self.opener.open(http_request, timeout=self.timeout) as response:
                record["http_status"] = response.status
                envelope = _strict_json(response.read())
            if not isinstance(envelope, dict):
                raise ClientError("API response envelope must be a JSON object")
            record["response"] = {
                name: envelope[name]
                for name in (
                    "id",
                    "object",
                    "created",
                    "model",
                    "choices",
                    "usage",
                    "system_fingerprint",
                    "service_tier",
                )
                if name in envelope
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
            record.update(decision_id=proposal["decision_id"], accepted=True)
            return proposal
        except urllib.error.HTTPError as exc:
            record["http_status"] = exc.code
            try:
                envelope = _strict_json(exc.read(1024 * 1024))
                if isinstance(envelope, dict):
                    record["response"] = {
                        name: envelope[name]
                        for name in ("id", "model", "usage")
                        if name in envelope
                    }
                    record["token_usage"] = normalize_usage(envelope.get("usage"))
                    error = envelope.get("error")
                    if isinstance(error, dict):
                        record["provider_error"] = {
                            name: _redact(error[name], key)[:1024]
                            for name in ("code", "type", "message")
                            if isinstance(error.get(name), str)
                        }
            except (ValueError, OSError, TypeError, http.client.HTTPException):
                pass
            finally:
                exc.close()
            record.update(
                error_kind="http_error",
                error=f"Inference endpoint returned HTTP {exc.code}",
            )
            raise ClientError(record["error"]) from None
        except (
            urllib.error.URLError,
            OSError,
            TimeoutError,
            http.client.HTTPException,
        ) as exc:
            kind = "transport_error" if record["provider_call"] else "preflight_error"
            record.update(
                error_kind=kind,
                error=f"Inference transport failed ({type(exc).__name__})",
            )
            raise ClientError(record["error"]) from None
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            kind = "proposal_rejected" if record["provider_call"] else "preflight_error"
            record.update(error_kind=kind, error=_redact(f"{kind}: {exc}", key))
            raise ClientError(record["error"]) from None
        finally:
            record.update(
                response_time=time.time(),
                latency_seconds=time.perf_counter() - monotonic,
            )
            _append_record(self.response_log, _redact(record, key))


def summarize_calls(records_or_path):
    """Reuse physical usage accounting; separate no-network preflight failures.

    Reasoning is a subset of output tokens. Failed calls retain available usage;
    unavailable usage stays unknown. No dollar rate is assumed. The internal
    adapter changes the version label only on copies passed to the shared ledger
    accountant; recorded interpolation records and their identity remain intact.
    """
    records = (
        [
            _strict_json(line)
            for line in Path(records_or_path).read_text().splitlines()
            if line.strip()
        ]
        if isinstance(records_or_path, (str, Path))
        else list(records_or_path)
    )
    physical, preflight = [], []
    for row in records:
        if (
            not isinstance(row, dict)
            or row.get("client_schema_version") != SCHEMA_VERSION
            or type(row.get("provider_call")) is not bool
        ):
            raise ClientError(
                "Token ledger contains an incompatible interpolation record"
            )
        if row["provider_call"]:
            physical.append({**row, "client_schema_version": _USAGE_LEDGER_VERSION})
        else:
            if (
                row.get("accepted") is not False
                or row.get("token_usage") != normalize_usage(None)
                or row.get("response")
                or "http_status" in row
                or row.get("error_kind") != "preflight_error"
            ):
                raise ClientError(
                    "Preflight ledger row claims a physical response or usage"
                )
            _number(row.get("latency_seconds"), "recorded preflight latency", 0, 1e10)
            _text(row.get("error"), "preflight error", 4096)
            preflight.append(row)
    return {
        **_summarize_provider_calls(physical),
        "client_attempts": len(records),
        "preflight_failures": len(preflight),
        "preflight_errors": dict(Counter(row["error"] for row in preflight)),
        "preflight_latency_seconds": sum(row["latency_seconds"] for row in preflight),
    }
