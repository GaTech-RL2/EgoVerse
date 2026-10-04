"""Retrospective credit labels on one completed real collection trajectory.

This is a data-selection diagnostic, never an action proposal or rollout retry.
Future observations may inform training labels, but never become student inputs.
"""

import base64
import copy
import json

import numpy as np

from astra_reversal.records import digest

from .teacher import _check, _object

SCHEMA_VERSION = "reasoning-hindsight-credit-1"
PROMPT_TEMPLATE_VERSION = "reasoning-completed-trajectory-credit-1"
SYSTEM_PROMPT = """Review this completed REAL robot collection trajectory.
Use only its original task, actual controller actions, camera images, robot state,
and final binary environment outcome. Do not call tools or invent object poses,
counterfactual executions, missing observations, or actions. No evaluation data
is supplied. You are assigning retrospective training credit, not steering now.

For EVERY supplied segment, judge whether ALL its actual actions contributed to
the original task: useful setup, task-directed transport, correct placement, or
necessary recovery. Use the later observations to resolve uncertainty about an
earlier grasp or movement. A successful episode does not make every action good;
a failed episode can still contain useful setup. Do not mark lifting/holding as
useful if the fuller context shows unproductive repetition, movement to the wrong
destination, or loss of the required object. Distinguish visible task progress
from plausible intention. If a segment mixes useful and clearly bad actions, or
its usefulness remains unsupported, mark it ambiguous; do not invent finer timing.

Return observed_useful, failed, or ambiguous for each fixed segment, a confidence
between zero and one, a phase, and concise visible evidence. Confidence reflects
evidence for that label. Training may admit only observed_useful with confidence
at least 0.8, and only complete actually executed windows. An actual intervention
also retains its original pre-execution relative-improvement requirement; this
review cannot override it. Return only the requested JSON.
"""


def build_request(*, workflow, episode_id, instruction, frames, segments, success):
    request = {
        "schema_version": SCHEMA_VERSION,
        "role": "review_completed_collection",
        "source_workflow": workflow,
        "episode_id": episode_id,
        "instruction": instruction,
        "frames": copy.deepcopy(frames),
        "segments": copy.deepcopy(segments),
        "environment_success": success,
    }
    request["request_fingerprint"] = digest(request)
    _validate_request(request)
    return request


def _validate_request(request):
    value = copy.deepcopy(request)
    fingerprint = value.pop("request_fingerprint", None)
    if fingerprint != digest(value) or set(value) != {
        "schema_version",
        "role",
        "source_workflow",
        "episode_id",
        "instruction",
        "frames",
        "segments",
        "environment_success",
    }:
        raise ValueError("Hindsight request fields or fingerprint differ")
    if (
        value["schema_version"] != SCHEMA_VERSION
        or value["role"] != "review_completed_collection"
        or type(value["environment_success"]) is not bool
        or any(
            not isinstance(value[k], str) or not value[k].strip()
            for k in ("source_workflow", "episode_id", "instruction")
        )
    ):
        raise ValueError("Completed collection identity and outcome are required")
    frames, segments = value["frames"], value["segments"]
    if not 1 <= len(segments) <= 60 or len(frames) != len(segments) + 1:
        raise ValueError("Each segment requires its real boundary frames")
    end = 0
    for index, row in enumerate(segments):
        if set(row) != {"segment_id", "start_step", "end_step_exclusive", "actions"}:
            raise ValueError("Unexpected segment metadata")
        start, stop = row["start_step"], row["end_step_exclusive"]
        if (
            type(row["segment_id"]) is not int
            or row["segment_id"] != index
            or type(start) is not int
            or type(stop) is not int
            or start != end
            or not 1 <= stop - start <= 10
        ):
            raise ValueError("Segments must cover one contiguous actual trajectory")
        actions = np.asarray(row["actions"], dtype=float)
        if actions.shape != (stop - start, 7) or not np.isfinite(actions).all():
            raise ValueError("Actual seven-channel controller actions are required")
        end = stop
    expected_steps = [0] + [r["end_step_exclusive"] for r in segments]
    if (
        any(type(f["step"]) is not int for f in frames)
        or [f["step"] for f in frames] != expected_steps
    ):
        raise ValueError("Frames must be exact chronological segment boundaries")
    for frame in frames:
        if set(frame) != {"step", "state", "images"}:
            raise ValueError("Only actual camera images and robot state are allowed")
        state = np.asarray(frame["state"], dtype=float)
        if (
            state.shape != (8,)
            or not np.isfinite(state).all()
            or len(frame["images"]) != 2
        ):
            raise ValueError("Two real cameras and finite state8 are required")
        for wire in frame["images"]:
            if (
                wire["encoding"] != "base64_png"
                or min(wire["width"], wire["height"]) < 1
            ):
                raise ValueError("Real PNG observations required")
            base64.b64decode(wire["data"], validate=True)


def response_schema(request):
    return _object(
        {
            "episode_assessment": {"type": "string", "minLength": 1, "maxLength": 900},
            "segments": {
                "type": "array",
                "minItems": len(request["segments"]),
                "maxItems": len(request["segments"]),
                "items": _object(
                    {
                        "segment_id": {
                            "type": "integer",
                            "enum": list(range(len(request["segments"]))),
                        },
                        "outcome": {
                            "type": "string",
                            "enum": ["observed_useful", "failed", "ambiguous"],
                        },
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "phase": {
                            "type": "string",
                            "enum": ["setup", "transport", "placement", "recovery"],
                        },
                        "evidence": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 350,
                        },
                    }
                ),
            },
        }
    )


def parse_proposal(raw, request):
    _validate_request(request)
    value = json.loads(raw) if isinstance(raw, str) else copy.deepcopy(raw)
    _check(value, response_schema(request))
    if sorted(r["segment_id"] for r in value["segments"]) != list(
        range(len(request["segments"]))
    ):
        raise ValueError("Each actual segment needs exactly one credit judgment")
    return {**value, "request_fingerprint": request["request_fingerprint"]}


def build_payload(request, model, *, sampling=None):
    _validate_request(request)
    context = {k: v for k, v in request.items() if k != "frames"}
    context["frame_states"] = [
        {k: f[k] for k in ("step", "state")} for f in request["frames"]
    ]
    content = [
        {
            "type": "text",
            "text": json.dumps(
                {"request": context, "response_schema": response_schema(request)}
            ),
        }
    ]
    for frame in request["frames"]:
        for camera, wire in zip(("external", "wrist"), frame["images"], strict=True):
            content.extend(
                [
                    {
                        "type": "text",
                        "text": f"REAL {camera} camera at control step {frame['step']}",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64," + wire["data"]},
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
