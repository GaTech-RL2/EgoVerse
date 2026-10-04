"""Offline feasibility check for image-space guidance from ordinary observations.

The VLM labels visible pixels without seeing robot poses. A separate fit relates
those labels to recorded proprioception. No simulator geometry, physical probes,
policy changes or action generation occur in this module.
"""

import base64
import copy
import json

import numpy as np

from astra_reversal.records import digest

from .teacher import _check, _object

SCHEMA_VERSION = "reasoning-visual-grounding-1"
PROMPT_TEMPLATE_VERSION = "reasoning-visible-pixel-labels-1"
_IDENTITY_FIELDS = (
    "schema_version",
    "role",
    "episode_id",
    "attempt_id",
    "request_index",
    "observation_step",
    "request_id",
)
SYSTEM_PROMPT = """Label visible points in these real robot camera images.
Use images only. Do not call tools, infer hidden coordinates, or invent motion.
For each frame, locate the midpoint between the robot's two fingertip contact
surfaces (the grasp center), consistently across frames. Do not label the wrist,
arm joint, or held object's center instead. Coordinates uv are normalized to
[0,1]: u increases rightwards, v downwards, origin at the top-left image corner.
If that point is not visually supported, set visible=false, confidence=0, uv=[0,0].
Report honest confidence and a short explanation of the visible evidence.
Also locate the center of the destination receptacle's opening in the LAST frame,
using the original task instruction. This is a visible 2D annotation, not a 3D
waypoint, a grasp assertion, or proof of task success. No robot pose, depth,
camera calibration or simulated future is available to you. Return only JSON.
"""


def build_request(frames, instruction, *, identity=None):
    request = {
        "schema_version": SCHEMA_VERSION,
        "role": "localize_visible_points",
        "instruction": instruction,
        "frames": copy.deepcopy(frames),
    }
    if identity is not None:
        if set(identity) != {"episode_id", "request_index", "observation_step"}:
            raise ValueError("Online pixel labels require an explicit rollout identity")
        episode = identity["episode_id"]
        index, step = identity["request_index"], identity["observation_step"]
        if (
            not isinstance(episode, str)
            or not episode
            or any(type(x) is not int or x < 0 for x in (index, step))
        ):
            raise ValueError("Invalid online pixel-label identity")
        request.update(
            **identity,
            attempt_id=episode,
            request_id=f"{episode}:{step}:localize_visible_points:{index}",
        )
    request["request_fingerprint"] = digest(request)
    validate_request(request)
    return request


def validate_request(request):
    value = copy.deepcopy(request)
    fingerprint = value.pop("request_fingerprint", None)
    if fingerprint != digest(value) or value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Visual grounding request identity differs")
    base = {"schema_version", "role", "instruction", "frames"}
    online = {
        "episode_id",
        "attempt_id",
        "request_index",
        "observation_step",
        "request_id",
    }
    if (
        set(value) not in (base, base | online)
        or value["role"] != "localize_visible_points"
    ):
        raise ValueError("Unexpected visual grounding request fields or role")
    frames = value["frames"]
    if not 6 <= len(frames) <= 12:
        raise ValueError("Use six to twelve recorded frames")
    steps = [row["step"] for row in frames]
    if any(type(step) is not int or step < 0 for step in steps) or steps != sorted(
        set(steps)
    ):
        raise ValueError("Frame steps must be unique and chronological")
    if "episode_id" in value:
        if (
            not isinstance(value["episode_id"], str)
            or not value["episode_id"]
            or any(
                type(value[k]) is not int or value[k] < 0
                for k in ("observation_step", "request_index")
            )
            or value["attempt_id"] != value["episode_id"]
            or value["request_id"]
            != f"{value['episode_id']}:{value['observation_step']}:localize_visible_points:{value['request_index']}"
            or max(steps) >= value["observation_step"]
        ):
            raise ValueError("Online projection uses only its own earlier prefix")
    for row in frames:
        if set(row) != {"step", "image"}:
            raise ValueError("The pixel labeler receives no poses or other state")
        wire = row["image"]
        if wire["encoding"] != "base64_png" or min(wire["width"], wire["height"]) < 1:
            raise ValueError("Real PNG frames required")
        base64.b64decode(wire["data"], validate=True)


_validate_request = validate_request


def response_schema(request):
    point = {
        "visible": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "uv": {
            "type": "array",
            "minItems": 2,
            "maxItems": 2,
            "items": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "evidence": {"type": "string", "minLength": 1, "maxLength": 240},
    }
    return _object(
        {
            "frames": {
                "type": "array",
                "minItems": len(request["frames"]),
                "maxItems": len(request["frames"]),
                "items": _object(
                    {
                        "step": {
                            "type": "integer",
                            "enum": [r["step"] for r in request["frames"]],
                        },
                        **point,
                    }
                ),
            },
            "destination": _object(point),
        }
    )


def parse_proposal(raw, request):
    validate_request(request)
    value = json.loads(raw) if isinstance(raw, str) else copy.deepcopy(raw)
    _check(value, response_schema(request))
    if sorted(row["step"] for row in value["frames"]) != [
        r["step"] for r in request["frames"]
    ]:
        raise ValueError("Each actual frame needs exactly one label")
    for row in [*value["frames"], value["destination"]]:
        if not row["visible"] and (row["confidence"] != 0 or row["uv"] != [0, 0]):
            raise ValueError("Invisible points cannot be numeric pseudo-labels")
    return {
        **value,
        "request_fingerprint": request["request_fingerprint"],
        **(
            {
                **{k: request[k] for k in _IDENTITY_FIELDS if k != "request_id"},
                "decision_id": request["request_id"],
            }
            if "episode_id" in request
            else {}
        ),
    }


def build_payload(request, model, *, sampling=None):
    validate_request(request)
    context = {k: v for k, v in request.items() if k != "frames"}
    context["frame_steps"] = [r["step"] for r in request["frames"]]
    content = [
        {
            "type": "text",
            "text": json.dumps(
                {"request": context, "response_schema": response_schema(request)}
            ),
        }
    ]
    for row in request["frames"]:
        content.extend(
            [
                {
                    "type": "text",
                    "text": f"REAL external-camera frame at control step {row['step']}",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64," + row["image"]["data"]
                    },
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


def fit_projection(
    positions, pixels, *, maximum_loo_pixels=8.0, maximum_condition=100.0
):
    """Cross-validated local affine projection; never a dynamics/success model."""
    xyz, uv = np.asarray(positions, dtype=float), np.asarray(pixels, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or uv.shape != (len(xyz), 2) or len(xyz) < 6:
        raise ValueError("At least six paired real XYZ and image pixel labels required")
    if not np.isfinite(xyz).all() or not np.isfinite(uv).all():
        raise ValueError("Finite measured positions and labels required")
    center, scale = xyz.mean(0), xyz.std(0)
    if np.any(scale < 1e-4):
        return {"accepted": False, "reason": "insufficient_motion_excitation"}
    design = np.column_stack(((xyz - center) / scale, np.ones(len(xyz))))
    condition = float(np.linalg.cond(design))
    if np.linalg.matrix_rank(design) != 4 or condition > maximum_condition:
        return {
            "accepted": False,
            "reason": "rank_or_condition_failure",
            "condition": condition,
        }
    coefficients = np.linalg.lstsq(design, uv, rcond=None)[0]
    errors = []
    for i in range(len(xyz)):
        training = np.arange(len(xyz)) != i
        if np.linalg.matrix_rank(design[training]) != 4:
            return {"accepted": False, "reason": "leave_one_out_rank_failure"}
        fit = np.linalg.lstsq(design[training], uv[training], rcond=None)[0]
        errors.append(float(np.linalg.norm(design[i] @ fit - uv[i])))
    loo = float(np.sqrt(np.mean(np.square(errors))))
    return {
        "accepted": loo <= maximum_loo_pixels,
        "reason": "passed"
        if loo <= maximum_loo_pixels
        else "inaccurate_pixel_projection",
        "observations": len(xyz),
        "condition": condition,
        "loo_rms_pixels": loo,
        "loo_errors_pixels": errors,
        "maximum_loo_pixels": maximum_loo_pixels,
        "jacobian_pixels_per_meter": (coefficients[:3] / scale[:, None]).T.tolist(),
        "offset_pixels": (
            coefficients[3] - center @ (coefficients[:3] / scale[:, None])
        ).tolist(),
        "claim": "Local geometry fit to VLM pixel labels; not independently verified camera calibration or action-outcome prediction.",
    }
