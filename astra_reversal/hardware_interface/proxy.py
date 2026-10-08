"""Both interfaces terminate in the same deterministic, private execution proxy."""

import base64
import copy
import io
import math
import time
from dataclasses import dataclass

import numpy as np
from PIL import Image

from .common import digest, exact, utc

# Only native robot sensors and RGB cameras. No object-state aggregates.
SENSORS = {
    "robot.joint_position": ("robot0_joint_pos", "rad", "joint", [7]),
    "robot.joint_velocity": ("robot0_joint_vel", "rad/s", "joint", [7]),
    "robot.eef_position": ("robot0_eef_pos", "m", "world", [3]),
    "robot.eef_orientation": ("robot0_eef_quat", "unitless", "world_xyzw", [4]),
    "robot.gripper_position": ("robot0_gripper_qpos", "m", "joint", [2]),
    "robot.gripper_velocity": ("robot0_gripper_qvel", "m/s", "joint", [2]),
    "camera.front": ("agentview_image", "RGB_uint8", "agentview", None),
    "camera.wrist": (
        "robot0_eye_in_hand_image",
        "RGB_uint8",
        "robot0_eye_in_hand",
        None,
    ),
}
NATIVE_KEYS = {values[0] for values in SENSORS.values()}


@dataclass(frozen=True)
class Limits:
    steps: int = 1000
    wall_seconds: float = 1200
    workflow_tokens: int = 100000
    tool_calls: int = 1000
    max_repeat: int = 10
    observer_calls: int = 50
    actor_output_tokens: int = 2048
    observer_output_tokens: int = 256
    context_tokens: int = 32768
    response_seconds: float = 180


class CommandError(ValueError):
    def __init__(self, reason, *, safety=False):
        super().__init__(reason)
        self.reason, self.safety = reason, safety


def validate_action(action, repeat, observation_step, current_step, bounds, limits):
    if type(action) is not list or len(action) != 7:
        raise CommandError("action_shape")
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in action):
        raise CommandError("action_finite_numbers_required")
    if type(repeat) is not int or not 1 <= repeat <= limits.max_repeat:
        raise CommandError("repeat_limit")
    if type(observation_step) is not int or observation_step != current_step:
        raise CommandError("stale_observation_step")
    if any(not low <= x <= high for x, low, high in zip(action, *bounds)):
        raise CommandError("controller_input_bounds", safety=True)
    if repeat > limits.steps - current_step:
        raise CommandError("remaining_step_budget")
    return [float(x) for x in action]


class Proxy:
    """The agent receives JSON from this object, never this object or its env."""

    def __init__(
        self,
        environment,
        observation,
        *,
        episode_id,
        controller,
        events,
        limits=Limits(),
        clock=time.monotonic,
    ):
        self.env = environment
        self.observation = observation
        self.episode_id = episode_id
        self.controller = copy.deepcopy(controller)
        self.events, self.limits, self.clock = events, limits, clock
        self.step_count = 0
        self.wall_start = clock()
        self.captured_at = clock()
        self.captured_utc = utc()
        self.terminal = None
        self.success = False
        self.simulator_seconds = 0.0
        self.invalid_actions = self.safety_attempts = self.applied_violations = 0
        self.bounds = (controller["input_min"], controller["input_max"])
        if any(key not in observation for key in NATIVE_KEYS):
            raise ValueError("required_sensor_unavailable")
        if any(len(v) != 7 for v in self.bounds):
            raise ValueError("controller_shape")

    def available(self):
        if self.terminal:
            raise CommandError("episode_ended")
        if self.clock() - self.wall_start >= self.limits.wall_seconds:
            self.terminal = "TIMEOUT_WALL"
            raise CommandError("wall_limit")

    def _sample(self, key):
        array = np.asarray(self.observation[key])
        if key.endswith("_image"):
            # Both arms see the identical upright camera convention.
            image = np.ascontiguousarray(array[::-1])
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError("invalid_camera")
            buffer = io.BytesIO()
            Image.fromarray(image).save(buffer, format="PNG")
            raw = buffer.getvalue()
            return {
                "encoding": "image/png",
                "base64": base64.b64encode(raw).decode(),
                "shape": list(image.shape),
            }, "image"
        if not np.isfinite(array).all():
            raise ValueError("nonfinite_sensor")
        return array.astype(float).tolist(), "float64[]"

    def observe(self, keys):
        self.available()
        if (
            type(keys) is not list
            or not keys
            or len(keys) != len(set(keys))
            or not set(keys) <= NATIVE_KEYS
        ):
            raise CommandError("unknown_observation_key")
        self.observation = self.env.read_sensors()
        self.captured_at, self.captured_utc = self.clock(), utc()
        result = {
            "episode_id": self.episode_id,
            "simulator_step": self.step_count,
            "timestamp": self.captured_utc,
            "observations": {},
        }
        for key in keys:
            result["observations"][key] = self._sample(key)[0]
        self.events.emit(
            "observation_returned",
            simulator_step=self.step_count,
            keys=keys,
            response_sha256=digest(result),
        )
        return result

    def describe(self):
        self.available()
        channels = []
        for channel, (key, unit, frame, shape) in SENSORS.items():
            shape = (
                list(np.asarray(self.observation[key]).shape)
                if shape is None
                else shape
            )
            channels.append(
                {
                    "channel": channel,
                    "type": "image" if key.endswith("_image") else "float64[]",
                    "description": channel.replace(".", " "),
                    "units": unit,
                    "shape": shape,
                    "frequency_hz": self.controller["frequency_hz"],
                    "metadata": {
                        "frame": frame,
                        "episode_id": self.episode_id,
                        "simulator_step": self.step_count,
                        "timestamp": self.captured_utc,
                        "valid": True,
                        "image_encoding": "PNG; upright RGB"
                        if key.endswith("_image")
                        else None,
                    },
                }
            )
        return {
            "schema_version": "hardware-1",
            "device_id": "libero-panda",
            "observations": channels,
            "actions": [
                {
                    "channel": "robot.controller_command",
                    "type": "float64[]",
                    "shape": [7],
                    "description": "Atomic normalized delta position xyz, rotation axis-angle xyz, gripper",
                    "units": "normalized_controller_input",
                    "frequency_hz": self.controller["frequency_hz"],
                    "control_frequency_hz": self.controller["frequency_hz"],
                    "limits": {"min": self.bounds[0], "max": self.bounds[1]},
                    "safe_range": {"min": self.bounds[0], "max": self.bounds[1]},
                    "metadata": {
                        **self.controller,
                        "episode_id": self.episode_id,
                        "max_duration_steps": self.limits.max_repeat,
                        "freshness": "observation_step must equal current simulator_step; physics pauses between calls",
                    },
                }
            ],
        }

    def read(self, channel, max_age_ms):
        self.available()
        if (
            type(max_age_ms) not in (int, float)
            or not math.isfinite(max_age_ms)
            or max_age_ms < 0
        ):
            raise CommandError("max_age_ms")
        if channel not in SENSORS:
            raise CommandError("unknown_channel")
        key, unit, frame, _ = SENSORS[channel]
        # Re-sample current paused sensors. A newly requested image is current
        # in simulation even when an API request took minutes of wall time.
        self.observation = self.env.read_sensors()
        self.captured_at, self.captured_utc = self.clock(), utc()
        value, kind = self._sample(key)
        now = utc()
        envelope = {
            "channel": channel,
            "type": kind,
            "value": value,
            "timestamp": now,
            "units": unit,
            "valid": True,
            "metadata": {
                "episode_id": self.episode_id,
                "simulator_step": self.step_count,
                "frame": frame,
                "age_ms": 0,
                "controller_version": self.controller["version"],
            },
        }
        self.events.emit(
            "observation_returned",
            simulator_step=self.step_count,
            keys=[key],
            response_sha256=digest(envelope),
        )
        return envelope

    def act(self, envelope):
        try:
            exact(envelope, ("channel", "value", "duration_steps", "metadata"))
            exact(
                envelope["metadata"],
                ("mode", "units", "episode_id", "observation_step"),
            )
            if envelope["channel"] != "robot.controller_command":
                raise CommandError("unknown_channel")
            m = envelope["metadata"]
            if (
                m["mode"] != "configured_controller"
                or m["units"] != "normalized_controller_input"
            ):
                raise CommandError("mode_or_units")
            if m["episode_id"] != self.episode_id:
                raise CommandError("wrong_episode")
        except (KeyError, TypeError, ValueError) as error:
            return self.reject(
                envelope,
                error.reason if isinstance(error, CommandError) else "invalid_fields",
            )
        return self.step(
            envelope["value"],
            envelope["duration_steps"],
            m["observation_step"],
            requested=envelope,
        )

    def reject(self, request, reason, *, safety=False):
        self.invalid_actions += 1
        self.safety_attempts += int(safety)
        self.events.emit(
            "action_rejected",
            requested=request,
            rejection_reason=reason,
            simulator_step=self.step_count,
            safety_attempt=safety,
        )
        return {
            "accepted": False,
            "applied_action": None,
            "rejection_reason": reason,
            "simulator_step": self.step_count,
            "timestamp": utc(),
        }

    def step(self, action, repeat_steps, observation_step, *, requested=None):
        request = (
            requested
            if requested is not None
            else {
                "action": action,
                "repeat_steps": repeat_steps,
                "observation_step": observation_step,
            }
        )
        self.events.emit(
            "action_requested", requested=request, simulator_step=self.step_count
        )
        try:
            self.available()
            action = validate_action(
                action,
                repeat_steps,
                observation_step,
                self.step_count,
                self.bounds,
                self.limits,
            )
        except CommandError as error:
            return self.reject(request, error.reason, safety=error.safety)
        applied = 0
        for _ in range(repeat_steps):
            if self.clock() - self.wall_start >= self.limits.wall_seconds:
                self.terminal = "TIMEOUT_WALL"
                break
            before = self.step_count
            started = time.monotonic()
            self.observation, _, _, _ = self.env.step(action)
            self.step_count += 1
            applied += 1
            self.captured_at, self.captured_utc = self.clock(), utc()
            self.events.emit(
                "action_applied",
                requested_action=action,
                applied_action=action,
                simulator_step_before=before,
                simulator_step_after=self.step_count,
                controller_sha256=digest(self.controller),
            )
            # Private evaluator check on EVERY step, never relay reward/done/info.
            self.success = bool(self.env.check_success())
            self.simulator_seconds += time.monotonic() - started
            self.events.emit(
                "evaluator_result", simulator_step=self.step_count, success=self.success
            )
            if self.success:
                self.terminal = "SUCCESS"
                break
        if not self.terminal and self.step_count >= self.limits.steps:
            self.terminal = "TIMEOUT_STEPS"
        return {
            "accepted": True,
            "applied_action": action,
            "applied_steps": applied,
            "simulator_step": self.step_count,
            "timestamp": utc(),
            "episode_ended": self.terminal is not None,
        }

    def finish(self):
        self.available()
        self.terminal = "TASK_FAILURE"
        return {"episode_ended": True}
