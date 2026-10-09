"""Versioned Astra contracts, strict validation, and observation-bound replay."""

import base64
import io
import json
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .records import digest

SCHEMA_VERSION = "1.0"
CAMERAS = ("observation/image", "observation/wrist_image")
COMMON_FIELDS = {
    "schema_version",
    "episode_id",
    "plan_id",
    "observation_step",
    "subgoal_id",
    "subgoal_instruction",
    "completion",
    "timeout_env_steps",
}


def require_text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")
    return value


def validate_completion(value, supported):
    if not isinstance(value, dict) or set(value) != {"type", "parameters"}:
        raise ValueError("completion requires type and parameters")
    kind, params = value["type"], value["parameters"]
    if kind not in supported or not isinstance(params, dict):
        raise ValueError(f"Unsupported completion checker: {kind}")
    expected = {
        "eef_position": {"target", "tolerance"},
        "gripper_width": {"target", "tolerance"},
        "image_progress": {"criterion"},
    }[kind]
    if set(params) != expected:
        raise ValueError("Invalid completion parameters")
    if kind == "image_progress":
        require_text(params["criterion"], "criterion")
    else:
        target = np.asarray(params["target"])
        expected_shape = (3,) if kind == "eef_position" else ()
        if (
            target.dtype.kind not in "ifu"
            or target.shape != expected_shape
            or not np.isfinite(target).all()
        ):
            raise ValueError(
                "Completion target must be finite and match the observed state"
            )
        tolerance = params["tolerance"]
        if (
            isinstance(tolerance, bool)
            or not isinstance(tolerance, (int, float))
            or not np.isfinite(tolerance)
            or tolerance <= 0
        ):
            raise ValueError("Completion tolerance must be positive and finite")
    return value


@dataclass
class AgentProposal:
    schema_version: str
    episode_id: str
    plan_id: str
    observation_step: int
    subgoal_id: str
    subgoal_instruction: str
    completion: dict
    timeout_env_steps: int
    action_spec_id: str | None
    action_chunk: np.ndarray | None
    annotations: list[dict]
    constraints: list[str]
    raw_response: str
    model_version: str
    request_fingerprint: str
    request_time: float
    response_time: float
    prompt_template_version: str = "astra-reversal-1"


def validate_annotations(annotations, observation):
    if not isinstance(annotations, list) or len(annotations) > 32:
        raise ValueError("annotations must be a list of at most 32 items")
    for item in annotations:
        if not isinstance(item, dict) or set(item) != {
            "camera",
            "kind",
            "coordinates",
            "label",
        }:
            raise ValueError(
                "Each annotation requires camera, kind, coordinates, label"
            )
        camera, kind = item["camera"], item["kind"]
        if camera not in CAMERAS or kind not in ("point", "box"):
            raise ValueError(
                "Annotations must use an existing camera and point/box kind"
            )
        coords = np.asarray(item["coordinates"])
        if (
            coords.dtype.kind not in "ifu"
            or coords.shape != ((2,) if kind == "point" else (4,))
            or not np.isfinite(coords).all()
        ):
            raise ValueError("Invalid annotation coordinates")
        height, width = observation[camera].shape[:2]
        if (
            np.any(coords < 0)
            or np.any(coords[0::2] >= width)
            or np.any(coords[1::2] >= height)
        ):
            raise ValueError("Annotation coordinates are outside the source image")
        if kind == "box" and (coords[0] >= coords[2] or coords[1] >= coords[3]):
            raise ValueError("Box corners must be ordered")
        require_text(item["label"], "annotation label")
    return annotations


def build_request(
    *,
    episode_id,
    step,
    observation,
    instruction,
    history,
    active,
    spec,
    stage,
    completion_types,
    error=None,
    completed_subgoals=(),
    max_timeout=60,
    model_version=None,
    sampling_settings=None,
):
    request = {
        "schema_version": SCHEMA_VERSION,
        "episode_id": episode_id,
        "observation_step": step,
        "observation_id": digest(observation),
        "task_instruction": instruction,
        "observation": {
            key: observation[key] for key in (*CAMERAS, "observation/state")
        },
        "history": history,
        "completed_subgoals": list(completed_subgoals),
        "active_subgoal": active,
        "max_timeout_env_steps": max_timeout,
        "requested_model_version": model_version,
        "sampling_settings": sampling_settings or {},
        "stage": stage,
        "action_spec": spec.as_dict(),
        "supported_completion_types": list(completion_types),
        "instructions": (
            "Return one JSON object. Maintain or revise the current subgoal using this fresh observation. "
            "New local actions alone do not mean the subgoal is completed. Never use simulator object poses or predicates. "
            "Use unique plan_id for each accepted response. Echo episode_id and observation_step exactly. "
            "Completion is {type, parameters}: eef_position {target: [x,y,z], tolerance}, "
            "gripper_width {target, tolerance}, or image_progress {criterion} only if supported. "
            "Pose completion targets use observed world position; action values use the declared controller convention. "
            "Required fields: schema_version, episode_id, plan_id, observation_step, subgoal_id, "
            "subgoal_instruction, completion, timeout_env_steps. "
            + (
                "Also return action_spec_id and exactly H rows of seven continuous numeric action_chunk values. "
                "Supply every timestep; do not return sparse waypoints or missing numbers."
                if stage == 1
                else "Also return constraints (list of strings) and annotations (list of {camera, kind, coordinates, label}). "
                "Use pixel coordinates in the supplied images; kind is point [x,y] or box [x1,y1,x2,y2]. "
                "Camera must be observation/image or observation/wrist_image. Empty annotations are allowed. "
                "Do not return continuous actions in Stage 2."
            )
        ),
        "validation_error": error,
    }
    request["request_fingerprint"] = digest(request)
    return request


def parse_proposal(
    raw,
    request,
    *,
    model_version,
    request_time,
    response_time,
    max_timeout,
    completion_types,
):
    data = json.loads(raw)
    required = COMMON_FIELDS | (
        {"action_spec_id", "action_chunk"}
        if request["stage"] == 1
        else {"annotations", "constraints"}
    )
    if not isinstance(data, dict) or set(data) != required:
        raise ValueError(f"Response must have exactly these fields: {sorted(required)}")
    if (
        data["schema_version"] != SCHEMA_VERSION
        or data["episode_id"] != request["episode_id"]
    ):
        raise ValueError("Wrong schema or episode")
    if (
        type(data["observation_step"]) is not int
        or data["observation_step"] != request["observation_step"]
    ):
        raise ValueError("Stale or future proposal")
    for name in ("plan_id", "subgoal_id", "subgoal_instruction"):
        require_text(data[name], name)
    timeout = data["timeout_env_steps"]
    if type(timeout) is not int or not 0 < timeout <= max_timeout:
        raise ValueError("Subgoal timeout exceeds the configured bound")
    validate_completion(data["completion"], completion_types)
    if request["stage"] == 1:
        from .action_adapter import ActionSpec

        spec = ActionSpec(**request["action_spec"])
        if data["action_spec_id"] != spec.action_spec_id:
            raise ValueError("Wrong controller action specification")
        data["action_chunk"] = spec.validate_actions(data["action_chunk"])
        data.update(annotations=[], constraints=[])
    else:
        validate_annotations(data["annotations"], request["observation"])
        if not isinstance(data["constraints"], list) or len(data["constraints"]) > 32:
            raise ValueError("constraints must be a bounded list")
        for value in data["constraints"]:
            require_text(value, "constraint")
        data.update(action_spec_id=None, action_chunk=None)
    return AgentProposal(
        **data,
        raw_response=raw,
        model_version=model_version,
        request_fingerprint=request["request_fingerprint"],
        request_time=request_time,
        response_time=response_time,
    )


def _wire_value(value):
    if isinstance(value, np.ndarray):
        if value.dtype == np.uint8 and value.ndim == 3:
            from PIL import Image

            buffer = io.BytesIO()
            Image.fromarray(value).save(buffer, format="PNG")
            return {
                "encoding": "base64_png",
                "data": base64.b64encode(buffer.getvalue()).decode(),
            }
        return value.tolist()
    if isinstance(value, dict):
        return {k: _wire_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_wire_value(v) for v in value]
    return value


class CommandBackend:
    """Transport to an explicitly configured Astra client; no guessed API/model."""

    def __init__(self, command: list[str], model_version: str, timeout=120.0):
        self.command, self.timeout = command, timeout
        self.model_version = require_text(model_version, "model_version")
        self.sampling_settings = {
            "transport": "json_stdin_stdout",
            "command": command,
            "timeout_seconds": timeout,
        }

    def generate(self, request):
        result = subprocess.run(
            self.command,
            input=json.dumps(_wire_value(request), allow_nan=False),
            capture_output=True,
            text=True,
            timeout=self.timeout,
            check=True,
        )
        return result.stdout.strip()


class ReplayBackend:
    """Replays only identical recorded inputs, including the original images."""

    def __init__(self, path):
        self.records = [
            json.loads(line)
            for line in Path(path).read_text().splitlines()
            if line.strip()
        ]
        self.model_version = "recorded_astra_responses"
        self.sampling_settings = {"replay_file": str(path)}

    def generate(self, request):
        if not self.records:
            raise ValueError("No recorded response remains")
        record = self.records.pop(0)
        if record["request_fingerprint"] != request["request_fingerprint"]:
            raise ValueError(
                "Replay request differs from recorded observations/conditioning"
            )
        self.model_version = record["model_version"]
        return record["raw_response"]


class AstraAgent:
    def __init__(self, backend, config, recorder):
        self.backend, self.config, self.recorder = backend, config, recorder
        self.calls = self.invalid_responses = self.retries = 0

    def propose(self, request_args, *, accept=None):
        error = None
        for attempt in range(self.config.invalid_response_retries + 1):
            request = build_request(
                **request_args,
                error=error,
                max_timeout=self.config.subgoal_timeout_env_steps,
                model_version=self.config.model_version,
                sampling_settings=self.config.sampling_settings,
            )
            began = time.time()
            raw = None
            self.calls += 1
            self.retries += int(attempt > 0)
            try:
                raw = self.backend.generate(request)
                ended = time.time()
                proposal = parse_proposal(
                    raw,
                    request,
                    model_version=self.backend.model_version,
                    request_time=began,
                    response_time=ended,
                    max_timeout=self.config.subgoal_timeout_env_steps,
                    completion_types=request_args["completion_types"],
                )
                if accept is not None:
                    accept(proposal)
            except (
                ValueError,
                TypeError,
                KeyError,
                OverflowError,
                subprocess.SubprocessError,
                OSError,
            ) as exc:
                self.invalid_responses += 1
                error = f"{type(exc).__name__}: {exc}"
                self.recorder.event(
                    "agent_response",
                    accepted=False,
                    attempt=attempt,
                    request=request,
                    raw_response=raw,
                    error=error,
                    model_version=self.backend.model_version,
                    latency_seconds=time.time() - began,
                    sampling_settings=getattr(self.backend, "sampling_settings", {}),
                )
                continue
            self.recorder.event(
                "agent_response",
                accepted=True,
                attempt=attempt,
                request=request,
                proposal=asdict(proposal),
                latency_seconds=ended - began,
                sampling_settings=getattr(self.backend, "sampling_settings", {}),
            )
            return proposal
        raise ValueError(f"Agent proposal failed after bounded retries: {error}")
