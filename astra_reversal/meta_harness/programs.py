"""Atomic policy-program transitions at native five-action replan boundaries."""

import copy
import math

import numpy as np

from astra_reversal.demo_skill_program import ProgramExecutor
from astra_reversal.records import digest


class Programs:
    def __init__(self, compiler, limits):
        self.compiler, self.limits = compiler, limits
        self.executor = self.choice = self.program_id = None
        self.started = self.expiry = None
        self.step, self.epoch = -5, 0
        self.termination = None

    @property
    def stage_id(self):
        return f"stage_{self.epoch}"

    def status(self):
        return {
            "stage_id": self.stage_id,
            "program_id": self.program_id,
            "started_action": self.started,
            "expiry_action": self.expiry,
            "donor_cursor": None if self.choice is None else self.choice["frame"],
            "termination_reason": self.termination,
        }

    def clear(self, reason):
        self.executor = self.choice = self.program_id = None
        self.started = self.expiry = None
        self.epoch += 1
        self.termination = reason

    def boundary(self, live, step):
        if (
            type(step) is not int
            or step != self.step + 5
            or step >= self.limits.action_budget
        ):
            raise ValueError("Expected the next native five-action replan boundary")
        self.step = step
        if self.executor is not None:
            self.choice = self.executor.select(live, step - self.started)
            if self.choice is None:
                self.clear(
                    "expiry" if step >= self.expiry else "sensor_guard_or_segment_end"
                )

    def apply(self, call, request, live, now):
        """Invalid calls preserve a still-valid program; they never extend expiry."""
        receipt = {
            "requested_call": copy.deepcopy(call),
            "actually_executed_call": None,
            "action": self.step,
            "validation_error": None,
            "fallback": "retain_valid_program_else_native",
        }
        try:
            if not math.isfinite(now) or now < request["captured_at"]:
                raise ValueError("Invalid observation clock")
            if now - request["captured_at"] > self.limits.max_age_seconds:
                raise ValueError("stale_wall_time")
            age = self.step - request["action"]
            if age < 0 or age > self.limits.max_age_actions:
                raise ValueError("stale_action_age")
            if request["stage_id"] != self.stage_id:
                raise ValueError("stale_stage")
            program = self.compiler.compile(call, request["retrieved_ids"])
            args, tool = call["arguments"], call["tool"]
            if args["observation_id"] != request["observation_id"]:
                raise ValueError("wrong_observation_id")
            if tool == "set_policy_program":
                if args["expected_stage_id"] != self.stage_id:
                    raise ValueError("wrong_expected_stage_id")
                executor = ProgramExecutor(
                    program, self.compiler.bank, "input_skill_library"
                )
                choice = executor.select(live, 0)
                self.executor, self.choice = executor, choice
                self.started = self.step
                self.expiry = self.step + args["max_actions"]
                self.epoch += 1
                self.program_id = (
                    "program_" + digest([call, self.step, self.epoch])[:16]
                )
                self.termination = None
            elif tool == "keep_policy_program":
                if self.program_id is None or args["program_id"] != self.program_id:
                    raise ValueError("wrong_or_expired_program_id")
            else:
                self.clear("explicit_clear")
            receipt["actually_executed_call"] = copy.deepcopy(call)
            receipt["fallback"] = None
        except (ValueError, TypeError, KeyError) as error:
            receipt["validation_error"] = str(error)
        receipt.update(self.status())
        return receipt


def public_observation(live, *, step, now, stage_id, episode_id):
    """Copy an allowlist, never environment objects or evaluator dictionaries."""
    keys = ("observation/image", "observation/wrist_image", "observation/state")
    if any(key not in live for key in keys):
        raise ValueError("Both live cameras and live state8 are required")
    raw = {key: np.array(live[key], copy=True) for key in keys}
    for camera in keys[:2]:
        if (
            raw[camera].dtype != np.uint8
            or raw[camera].ndim != 3
            or raw[camera].shape[2] != 3
        ):
            raise ValueError("Runtime cameras must be raw RGB8 images")
    if raw[keys[-1]].shape != (8,) or not np.isfinite(raw[keys[-1]]).all():
        raise ValueError("Runtime proprioception must be finite live state8")
    return raw, {
        "observation_id": "obs_" + digest([episode_id, step, raw])[:24],
        "action": step,
        "captured_at": now,
        "stage_id": stage_id,
        "state": raw[keys[-1]].tolist(),
    }
