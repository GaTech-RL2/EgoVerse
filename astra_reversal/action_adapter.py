"""Controller validation and exact checkpoint transforms; no motion synthesis."""

import copy
import inspect
from dataclasses import asdict, dataclass
from importlib.metadata import version

import numpy as np

from .records import digest, to_numpy


@dataclass(frozen=True)
class ActionSpec:
    action_spec_id: str
    horizon: int
    model_action_dim: int
    timestep_seconds: float
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    semantics: dict
    channels: tuple[str, ...] = (
        "delta_x",
        "delta_y",
        "delta_z",
        "delta_rx",
        "delta_ry",
        "delta_rz",
        "gripper",
    )
    schema_version: str = "1.0"

    def __post_init__(self):
        if (
            type(self.horizon) is not int
            or type(self.model_action_dim) is not int
            or self.horizon < 1
            or self.model_action_dim < 7
            or len(self.lower) != 7
            or len(self.upper) != 7
            or len(self.channels) != 7
            or not np.isfinite(self.timestep_seconds)
            or self.timestep_seconds <= 0
        ):
            raise ValueError("Invalid action dimensions")
        bounds = np.asarray([self.lower, self.upper])
        if not np.isfinite(bounds).all() or np.any(bounds[0] >= bounds[1]):
            raise ValueError("Invalid controller bounds")

    @classmethod
    def from_environment(cls, env, horizon, model_action_dim):
        robosuite_version = version("robosuite")
        if robosuite_version != "1.4.1":
            raise ValueError(
                "Controller semantics are verified for the bundled protocol's robosuite 1.4.1 only"
            )
        robot = env.robots[0]
        ctrl, gripper = robot.controller, robot.gripper
        if (
            len(env.robots) != 1
            or ctrl.name != "OSC_POSE"
            or not ctrl.use_delta
            or ctrl.impedance_mode != "fixed"
            or ctrl.control_dim != 6
            or type(gripper).__name__ != "PandaGripper"
        ):
            raise ValueError(
                "Only verified fixed-impedance delta OSC_POSE/PandaGripper is supported"
            )
        # Newer robosuite conventions must be checked rather than guessed.
        if type(ctrl).__module__ != "robosuite.controllers.osc":
            raise ValueError(
                "Unverified OSC implementation; resolve its coordinate frame first"
            )
        lower, upper = (np.asarray(x) for x in env.env.action_spec)
        semantics = {
            "robosuite_version": robosuite_version,
            "coordinate_space": "environment_controller_input_before_checkpoint_normalization",
            "command_type": "scaled_delta",
            "translation_frame": "world",
            "rotation_frame": "world",
            "rotation_representation": "axis_angle_delta",
            "units": "dimensionless controller inputs, not meters or radians",
            "controller_input_min": np.asarray(ctrl.input_min).tolist(),
            "controller_input_max": np.asarray(ctrl.input_max).tolist(),
            "controller_output_min": np.asarray(ctrl.output_min).tolist(),
            "controller_output_max": np.asarray(ctrl.output_max).tolist(),
            "gripper": "negative opens; positive closes; zero holds; continuous input in bounds",
            "controller_source_sha256": digest(inspect.getsource(type(ctrl))),
            "gripper_source_sha256": digest(inspect.getsource(type(gripper))),
            "checkpoint_padding": "provided by checkpoint input transform after normalization",
        }
        payload = {
            "horizon": horizon,
            "model_action_dim": model_action_dim,
            "timestep_seconds": 1.0 / env.env.control_freq,
            "lower": tuple(lower.tolist()),
            "upper": tuple(upper.tolist()),
            "semantics": semantics,
        }
        return cls(action_spec_id=digest(payload), **payload)

    def validate_actions(self, actions):
        raw = np.asarray(actions)
        if raw.dtype.kind not in "fiu" or raw.shape != (self.horizon, 7):
            raise ValueError(
                f"action_chunk must contain exactly [{self.horizon}, 7] numeric values"
            )
        # Reject booleans even when numpy would coerce a mixed list to numbers.
        if isinstance(actions, list) and any(
            isinstance(x, bool) for row in actions if isinstance(row, list) for x in row
        ):
            raise ValueError("Booleans are not action values")
        array = np.array(raw, dtype=np.float32, copy=True)
        if not np.isfinite(raw).all() or not np.isfinite(array).all():
            raise ValueError("Actions must be finite")
        if np.any(raw < self.lower) or np.any(raw > self.upper):
            raise ValueError(
                "Proposal violates controller bounds; clipping proposals is forbidden"
            )
        return array

    def as_dict(self):
        return asdict(self)


class ActionAdapter:
    def __init__(self, spec: ActionSpec, input_transform, output_transform):
        self.spec = spec
        self.input_transform = input_transform
        self.output_transform = output_transform

    def encode(self, controller_actions, observation: dict):
        actions = self.spec.validate_actions(controller_actions)
        # Includes conversion -> normalization -> checkpoint padding, in that order.
        inputs = self.input_transform(
            {**copy.deepcopy(observation), "actions": actions}
        )
        model_actions = np.asarray(inputs["actions"], dtype=np.float32)
        expected = (self.spec.horizon, self.spec.model_action_dim)
        if model_actions.shape != expected or not np.isfinite(model_actions).all():
            raise ValueError(f"Checkpoint action transform must produce {expected}")
        return model_actions[None].copy()

    def decode(self, model_actions, transformed_state, *, bounds="clip"):
        actions = to_numpy(model_actions)
        if (
            actions.shape != (1, self.spec.horizon, self.spec.model_action_dim)
            or not np.isfinite(actions).all()
        ):
            raise ValueError("Invalid full internal policy action tensor")
        output = self.output_transform(
            {
                "actions": actions[0].copy(),
                "state": to_numpy(transformed_state)[0].copy(),
            }
        )
        decoded = np.asarray(output["actions"], dtype=np.float32)
        if decoded.shape != (self.spec.horizon, 7) or not np.isfinite(decoded).all():
            raise ValueError("Invalid decoded controller actions")
        clipped = np.clip(decoded, self.spec.lower, self.spec.upper).astype(np.float32)
        clipping = {
            "count": int(np.count_nonzero(decoded != clipped)),
            "max_abs": float(np.abs(decoded - clipped).max()),
        }
        if bounds == "reject" and clipping["count"]:
            raise ValueError("Generated policy actions violate controller bounds")
        if bounds not in ("clip", "reject"):
            raise ValueError("Unknown output bounds rule")
        return clipped, clipping
