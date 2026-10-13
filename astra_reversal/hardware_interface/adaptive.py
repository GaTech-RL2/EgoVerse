"""Per-episode interface evolution above the unchanged native execution proxy."""

import copy
import re

import numpy as np

from .common import digest, strict_json, write_json
from .proxy import CommandError

CONDITION = "F_adaptive"
CONTRACT = {
    "version": 1,
    "condition": CONDITION,
    "reset_each_trial": True,
    "cross_trial_transfer": False,
    "additional_information": "robot_proprioception_and_camera_calibration",
    "modifications": "sandboxed_python_tools_and_interface_notes",
    "calibration_steps": "count_in_episode",
}
PROMPT = """You may evolve your hardware interface during this episode. Inspect interface_catalog to discover optional hardware capabilities, then enable and read useful ones. You may define and revise reusable Python interface tools: sensor processing, coordinate conversions, action mappings, feedback controllers, or other procedures you devise. Programs receive params and the documented read/act/read_capability functions in isolated scratch; set result to their JSON output. A read_only tool cannot move the robot. Control tools use the same validated native act envelope as F. You may run probe_dynamics to measure responses, fit system-identification models from the measurements, and incorporate those models into your tools. Probes move the real episode robot and consume its remaining steps; there is no free calibration reset. Measurements are local responses, not guaranteed global dynamics. revise_interface_notes can retain your current hypotheses and corrections; restore_interface restores an earlier interface definition without undoing physical actions. Every change is versioned. All interface learning starts fresh for this trial. The evaluator, task success logic, native controller limits, simulator, and audited source remain outside your editable interface. Use the capabilities and revisions you find useful; no specific procedure is required."""

# Explicit robot encoders only. Never enumerate arbitrary observation keys.
EXTRA_SENSORS = {
    "robot0_joint_pos": (
        "native joint positions in source joint order",
        "rad for hinge, m for slide",
    ),
    "robot0_joint_vel": (
        "native joint velocities in source joint order",
        "rad/s for hinge, m/s for slide",
    ),
    "robot0_joint_acc": (
        "native joint accelerations in source joint order",
        "rad/s^2 for hinge, m/s^2 for slide",
    ),
    "robot0_gripper_qpos": ("gripper joint positions", "m"),
    "robot0_gripper_qvel": ("gripper joint velocities", "m/s"),
}


def camera_calibration(sim, names, image_size):
    """Robot camera calibration only; no scene objects or task annotations."""
    result = {}
    for name in names:
        index = sim.model.camera_name2id(name)
        fovy = float(sim.model.cam_fovy[index])
        focal = image_size / (2 * np.tan(np.deg2rad(fovy) / 2))
        result[name] = {
            "image_shape": [image_size, image_size],
            "fovy_degrees": fovy,
            "focal_pixels": [float(focal), float(focal)],
            "principal_point_pixels": [image_size / 2, image_size / 2],
            "position_world_m": np.asarray(sim.data.cam_xpos[index]).tolist(),
            "rotation_camera_to_world": np.asarray(sim.data.cam_xmat[index])
            .reshape(3, 3)
            .tolist(),
            "camera_axes": "OpenGL: x right, y up, viewing direction -z; delivered RGB is upright",
            "provenance": "Simulator-provided ideal camera calibration; an additional-information treatment, not an estimated calibration",
        }
    return result


def tool_definitions(obj, vector):
    string = {"type": "string"}
    return [
        (
            "interface_catalog",
            "Inspect optional hardware capabilities and current interface revision.",
            obj({}),
        ),
        (
            "enable_capability",
            "Enable an available hardware capability for this episode; records your reason.",
            obj({"name": string, "reason": string}),
        ),
        (
            "read_capability",
            "Read an enabled capability; no arbitrary simulator queries.",
            obj({"name": string}),
        ),
        (
            "define_interface_tool",
            "Create or replace a named Python interface tool. Code runs only when called. params contains parsed arguments_json. Set result to JSON.",
            obj(
                {
                    "name": string,
                    "description": string,
                    "code": string,
                    "read_only": {"type": "boolean"},
                    "reason": string,
                }
            ),
        ),
        (
            "call_interface_tool",
            "Execute the current version of a named interface tool in isolated scratch.",
            obj({"name": string, "arguments_json": string}),
        ),
        (
            "probe_dynamics",
            "Measure numeric sensors before and after one bounded native command. All executed steps count in the episode; no reset or hidden state.",
            obj(
                {
                    "action": vector,
                    "repeat_steps": {"type": "integer"},
                    "channels": {
                        "type": "array",
                        "items": string,
                        "minItems": 1,
                    },
                    "reason": string,
                }
            ),
        ),
        (
            "revise_interface_notes",
            "Replace persistent per-episode interface notes: assumptions, measured response models, or operating conventions.",
            obj({"notes": string, "reason": string}),
        ),
        (
            "restore_interface",
            "Restore an earlier interface revision. Does not undo robot movement or episode costs.",
            obj({"revision": {"type": "integer", "minimum": 0}, "reason": string}),
        ),
    ]


class AdaptiveInterface:
    def __init__(self, proxy):
        self.proxy = proxy
        self.extra_keys = {
            k: v
            for k, v in EXTRA_SENSORS.items()
            if k in proxy.observation and k not in proxy.native_keys
        }
        self.capabilities = {}
        if self.extra_keys:
            self.capabilities["additional_proprioception"] = {
                "description": "Additional native robot encoders currently omitted from F",
                "sensors": {
                    k: {"description": v[0], "units": v[1]}
                    for k, v in self.extra_keys.items()
                },
            }
        if callable(getattr(proxy.env, "hardware_calibration", None)):
            self.capabilities["camera_calibration"] = {
                "description": "Ideal intrinsics and current robot-camera extrinsics; extra information supplied by the benchmark adapter",
            }
        self.state = {"capabilities": [], "tools": {}, "notes": ""}
        self.versions = {0: copy.deepcopy(self.state)}
        self.revision = 0
        self.calibration_steps = self.probes = self.tool_executions = 0
        self.directory = proxy.events.directory / "interface-evolution"
        self.directory.mkdir(exist_ok=False)
        write_json(self.directory / "catalog.json", self.capabilities)
        write_json(self.directory / "revision-0000.json", self.state)

    def catalog(self):
        return {"available_capabilities": self.capabilities, **self.summary()}

    def summary(self):
        return {
            "revision": self.revision,
            "enabled_capabilities": self.state["capabilities"],
            "tools": {
                k: {p: v[p] for p in ("description", "read_only", "code_sha256")}
                for k, v in self.state["tools"].items()
            },
            "notes": self.state["notes"],
            "state_sha256": digest(self.state),
        }

    def change(self, state, operation, reason):
        previous = self.revision
        self.revision += 1
        self.state = copy.deepcopy(state)
        self.versions[self.revision] = copy.deepcopy(state)
        receipt = {
            "revision": self.revision,
            "previous_revision": previous,
            "operation": operation,
            "reason": reason,
            "simulator_step": self.proxy.step_count,
            "state": self.state,
            "state_sha256": digest(state),
        }
        write_json(self.directory / ("revision-%04d.json" % self.revision), receipt)
        self.proxy.events.emit("interface_revised", **receipt)
        return self.summary()

    def enable_capability(self, name, reason):
        if name not in self.capabilities:
            raise CommandError("hardware_capability_unavailable")
        state = copy.deepcopy(self.state)
        if name not in state["capabilities"]:
            state["capabilities"].append(name)
            state["capabilities"].sort()
        return self.change(state, "enable_capability", reason)

    def read_capability(self, name):
        if name not in self.state["capabilities"]:
            raise CommandError("hardware_capability_not_enabled")
        if name == "additional_proprioception":
            observation = self.proxy.env.read_sensors()
            values = {}
            for key in self.extra_keys:
                array = np.asarray(observation[key], dtype=float)
                if not np.isfinite(array).all():
                    raise ValueError("invalid_extra_sensor")
                values[key] = array.tolist()
        elif name == "camera_calibration":
            values = self.proxy.env.hardware_calibration()
        else:
            raise CommandError("hardware_capability_unavailable")
        return {
            "capability": name,
            "revision": self.revision,
            "simulator_step": self.proxy.step_count,
            "values": values,
        }

    def define_interface_tool(self, name, description, code, read_only, reason):
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", name):
            raise CommandError("interface_tool_name")
        try:
            compile(code, "adaptive-interface.py", "exec")
        except (SyntaxError, ValueError, TypeError):
            raise CommandError("interface_tool_syntax") from None
        state = copy.deepcopy(self.state)
        state["tools"][name] = {
            "description": description,
            "code": code,
            "read_only": read_only,
            "code_sha256": digest(code),
        }
        return self.change(state, "define_interface_tool", reason)

    def call_interface_tool(self, name, arguments_json, *, scratch, dispatch):
        if name not in self.state["tools"]:
            raise CommandError("interface_tool_unknown")
        params = strict_json(arguments_json)
        if not isinstance(params, dict):
            raise CommandError("interface_tool_parameters_object_required")
        if scratch is None:
            raise RuntimeError("scratch_isolation_unavailable")
        definition = copy.deepcopy(self.state["tools"][name])

        def routed(tool, arguments):
            if definition["read_only"] and tool == "act":
                return {
                    "error": "read_only_interface_tool",
                    "simulator_step": self.proxy.step_count,
                }
            return dispatch(tool, arguments)

        self.tool_executions += 1
        self.proxy.events.emit(
            "interface_tool_executed",
            name=name,
            revision=self.revision,
            code_sha256=definition["code_sha256"],
            read_only=definition["read_only"],
            parameters_sha256=digest(params),
            simulator_step=self.proxy.step_count,
        )
        remaining = self.proxy.remaining_wall_seconds()
        return scratch.execute(
            "params = " + repr(params) + "\n" + definition["code"],
            routed,
            seconds=15 if remaining is None else min(15, remaining),
        )

    def probe_dynamics(self, action, repeat_steps, channels, reason, *, dispatch):
        if (
            not channels
            or len(set(channels)) != len(channels)
            or any(
                c not in self.proxy.sensors
                or self.proxy.sensors[c][0].endswith("_image")
                for c in channels
            )
        ):
            raise CommandError("numeric_robot_channels_required")
        before = {
            c: dispatch("read", {"channel": c, "max_age_ms": 1000}) for c in channels
        }
        if any("error" in value for value in before.values()):
            return {"error": "probe_sensor_read_failed"}
        start = self.proxy.step_count
        result = dispatch(
            "act",
            {
                "envelope": {
                    "channel": "robot.controller_command",
                    "value": action,
                    "duration_steps": repeat_steps,
                    "metadata": {
                        "mode": "configured_controller",
                        "units": "normalized_controller_input",
                        "episode_id": self.proxy.episode_id,
                        "observation_step": start,
                    },
                }
            },
        )
        executed = self.proxy.step_count - start
        self.probes += 1
        self.calibration_steps += executed
        # The proxy retains the final step's sensors even at episode termination.
        after = {c: self.proxy._sample(self.proxy.sensors[c][0])[0] for c in channels}
        delta = {
            c: (np.asarray(after[c]) - np.asarray(before[c]["value"])).tolist()
            for c in channels
        }
        receipt = {
            "revision": self.revision,
            "reason": reason,
            "action": action,
            "start_step": start,
            "end_step": self.proxy.step_count,
            "executed_steps": executed,
            "before": {c: v["value"] for c, v in before.items()},
            "after": after,
            "delta": delta,
            "command_result": result,
            "interpretation": "Observed finite response to this command and state, including drift/contact; not a global Jacobian. Quaternion delta is a component difference, not an angular velocity.",
        }
        self.proxy.events.emit("system_identification_probe", **receipt)
        return receipt

    def revise_interface_notes(self, notes, reason):
        state = copy.deepcopy(self.state)
        state["notes"] = notes
        return self.change(state, "revise_interface_notes", reason)

    def restore_interface(self, revision, reason):
        if revision not in self.versions:
            raise CommandError("interface_revision_unknown")
        return self.change(
            self.versions[revision], "restore_interface_%d" % revision, reason
        )

    def outcome(self):
        return {
            "revisions": self.revision,
            "calibration_steps": self.calibration_steps,
            "probes": self.probes,
            "tool_executions": self.tool_executions,
            "final_interface": self.summary(),
        }
