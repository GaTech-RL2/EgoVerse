"""Small executable demonstration compositions; all guards use live sensors."""

import copy
import json
from pathlib import Path

import numpy as np

from .demo_segments import CAMERAS, write_json
from .records import digest

ARMS = ("action_composition", "input_skill_library")
STAGE_FIELDS = {
    "source_id",
    "start_frame",
    "end_frame",
    "alpha",
    "playback",
    "state_mode",
    "min_actions",
    "max_actions",
    "advance_when",
    "threshold",
    "language",
    "vision_operator",
    "occlusion_box",
}


def validate_program(program, catalog, arm):
    if (
        arm not in ARMS
        or not isinstance(program, dict)
        or set(program) != {"native", "stages"}
    ):
        raise ValueError("Unknown arm or invalid segment program")
    if type(program["native"]) is not bool or not isinstance(program["stages"], list):
        raise ValueError("Invalid native flag/stage list")
    if program["native"]:
        if program["stages"]:
            raise ValueError("Native program must have no stages")
        return copy.deepcopy(program)
    sources = {row["source_id"]: row for row in catalog}
    if not 1 <= len(program["stages"]) <= 12:
        raise ValueError("A program needs 1..12 bounded stages")
    for stage in program["stages"]:
        if not isinstance(stage, dict) or set(stage) != STAGE_FIELDS:
            raise ValueError("Unexpected segment fields")
        if type(stage["source_id"]) is not str or stage["source_id"] not in sources:
            raise ValueError("Segment must reference a supplied source")
        if any(
            type(stage[key]) is not int
            for key in ("start_frame", "end_frame", "min_actions", "max_actions")
        ):
            raise ValueError("Frame and action counts must be integers")
        if (
            not 0
            <= stage["start_frame"]
            < stage["end_frame"]
            <= sources[stage["source_id"]]["frame_count"]
        ):
            raise ValueError("Segment exceeds demonstration frame bounds")
        if (
            not 0 <= stage["min_actions"] <= stage["max_actions"] <= 100
            or stage["max_actions"] == 0
        ):
            raise ValueError("Invalid stage time limits")
        if stage["min_actions"] % 5 or stage["max_actions"] % 5:
            raise ValueError("Stage time limits must align to five-action replans")
        if (
            type(stage["alpha"]) not in (int, float)
            or not np.isfinite(stage["alpha"])
            or not 0 <= stage["alpha"] <= 1
        ):
            raise ValueError("Donor mixing fraction must lie in [0,1]")
        if stage["playback"] not in ("advance", "hold") or stage["state_mode"] not in (
            "live",
            "donor",
        ):
            raise ValueError("Unsupported playback/state mode")
        if arm == "action_composition" and stage["state_mode"] != "live":
            raise ValueError(
                "Action composition cannot change observation conditioning"
            )
        if stage["vision_operator"] not in (
            "none",
            "pixels",
            "occlusion",
            "vei",
            "vli",
        ):
            raise ValueError("Unknown visual input skill")
        language = stage["language"]
        if language is not None:
            if not isinstance(language, dict) or set(language) != {
                "operator",
                "source_a_id",
                "source_b_id",
                "alpha",
            }:
                raise ValueError("Invalid language skill fields")
            if language["operator"] not in ("tei", "tli") or any(
                language[k] not in sources for k in ("source_a_id", "source_b_id")
            ):
                raise ValueError("Invalid text intervention/source")
            if (
                type(language["alpha"]) not in (int, float)
                or not np.isfinite(language["alpha"])
                or not 0 <= language["alpha"] <= 1
            ):
                raise ValueError("Invalid language interpolation strength")
            if language["operator"] == "tei" and stage["vision_operator"] in (
                "vei",
                "vli",
            ):
                raise ValueError(
                    "Compose TEI and VEI/VLI across stages; simultaneous hooks are not supported"
                )
        if arm == "action_composition" and (
            language is not None or stage["vision_operator"] != "none"
        ):
            raise ValueError("The action-composition arm cannot use input skills")
        if stage["state_mode"] == "donor" and stage["vision_operator"] != "pixels":
            raise ValueError(
                "Donor proprioception is only supported with donor pixel inputs"
            )
        box = stage["occlusion_box"]
        if stage["vision_operator"] == "occlusion":
            if (
                not isinstance(box, list)
                or len(box) != 4
                or any(type(x) not in (int, float) or not np.isfinite(x) for x in box)
            ):
                raise ValueError("Occlusion requires a finite normalized rectangle")
            if (
                not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1)
                or (box[2] - box[0]) * (box[3] - box[1]) > 0.5
            ):
                raise ValueError("Occlusion is limited to half of each camera image")
        elif box is not None:
            raise ValueError("Only an occlusion skill may specify an occlusion box")
        if stage["advance_when"] not in (
            "timeout",
            "segment_end",
            "eef_lift",
            "gripper_width_below",
            "gripper_width_above",
        ):
            raise ValueError("Unknown sensor guard")
        if (
            type(stage["threshold"]) not in (int, float)
            or not np.isfinite(stage["threshold"])
            or not 0 <= stage["threshold"] <= 0.3
        ):
            raise ValueError("Guard threshold must be finite and within 0..0.3m")
    return copy.deepcopy(program)


class ProgramExecutor:
    def __init__(self, program, bank, arm):
        self.program = validate_program(program, bank.catalog(), arm)
        self.bank, self.arm = bank, arm
        self.index, self.started, self.initial_state, self.last_step = (
            0,
            None,
            None,
            None,
        )

    def select(self, live, step):
        if (
            type(step) is not int
            or step < 0
            or (self.last_step is not None and step != self.last_step + 5)
            or (self.last_step is None and step != 0)
        ):
            raise ValueError("Programs require consecutive five-action replans")
        self.last_step = step
        if self.program["native"] or self.index >= len(self.program["stages"]):
            return None
        state = np.asarray(live["observation/state"])
        if state.shape != (8,) or not np.isfinite(state).all():
            raise ValueError("Stage guards require finite live state8")
        if self.started is None:
            self.started, self.initial_state = step, state.copy()
        stage = self.program["stages"][self.index]
        elapsed = step - self.started
        width = float(abs(state[6] - state[7]))
        conditions = {
            "timeout": False,
            "segment_end": stage["playback"] == "advance"
            and elapsed >= stage["end_frame"] - stage["start_frame"],
            "eef_lift": float(state[2] - self.initial_state[2]) >= stage["threshold"],
            "gripper_width_below": width < stage["threshold"],
            "gripper_width_above": width > stage["threshold"],
        }
        if elapsed >= stage["max_actions"] or (
            elapsed >= stage["min_actions"]
            and elapsed > 0
            and conditions[stage["advance_when"]]
        ):
            self.index += 1
            self.started, self.initial_state = step, state.copy()
            if self.index == len(self.program["stages"]):
                return None
            stage, elapsed = self.program["stages"][self.index], 0
        offset = elapsed if stage["playback"] == "advance" else 0
        frame = min(stage["start_frame"] + offset, stage["end_frame"] - 1)
        return {
            **stage,
            "frame": frame,
            "stage_index": self.index,
            "elapsed_actions": elapsed,
        }


def composed_reference(bank, choice, native_actions):
    """Mix actual recorded deltas with native predictions; preserve source order."""
    native = np.asarray(native_actions, dtype=np.float32)
    if native.shape != (10, 7) or not np.isfinite(native).all():
        raise ValueError("Expected a finite native ten-action chunk")
    advance = (
        np.arange(10) if choice["playback"] == "advance" else np.zeros(10, dtype=int)
    )
    indexes = np.minimum(choice["frame"] + advance, choice["end_frame"] - 1)
    donor = bank.arrays(choice["source_id"])["actions"][indexes].copy()
    requested = (1 - choice["alpha"]) * native + choice["alpha"] * donor
    result = np.clip(requested, -1, 1).astype(np.float32)
    return result, {
        "source_frames": indexes.tolist(),
        "donor_actions_sha256": digest(donor),
        "terminal_frame_repetitions": int(
            np.count_nonzero(choice["frame"] + advance >= choice["end_frame"])
        ),
        "clipped_elements": int(np.count_nonzero(requested != result)),
    }


def substituted_observation(bank, choice, live):
    """Change only model inputs. Executor guards and recorded feedback stay live."""
    donor = bank.observation(choice["source_id"], choice["frame"])
    result = {key: np.array(value, copy=True) for key, value in live.items()}
    alpha = choice["alpha"]
    for camera in CAMERAS:
        if (
            result[camera].dtype != np.uint8
            or result[camera].shape != donor[camera].shape
        ):
            raise ValueError("Donor and live cameras must share native RGB geometry")
        if alpha == 1:
            result[camera] = donor[camera].copy()
        elif alpha != 0:
            result[camera] = (
                np.rint(
                    (1 - alpha) * result[camera].astype(np.float32)
                    + alpha * donor[camera]
                )
                .clip(0, 255)
                .astype(np.uint8)
            )
    if choice["state_mode"] == "donor" and alpha:
        # Axis-angle state is replaced exactly, never linearly interpolated.
        result["observation/state"] = donor["observation/state"].copy()
    return result


def has_effect(choice, arm):
    if choice is None:
        return False
    if arm == "action_composition":
        return choice["alpha"] > 0
    language = choice["language"]
    text_active = language is not None and (
        language["operator"] == "tei"
        or (
            language["alpha"] != 0.5
            and language["source_a_id"] != language["source_b_id"]
        )
    )
    return text_active or (choice["vision_operator"] != "none" and choice["alpha"] > 0)


class BehaviorLibrary:
    """Versioned observed outcomes. Hypotheses never become successes by assertion."""

    def __init__(self, directory, bank_id, arm):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.bank_id, self.arm = bank_id, arm
        self.path = self.directory / "library.json"
        if self.path.exists():
            self.data = json.loads(self.path.read_text())
            if self.data["bank_id"] != bank_id or self.data["arm"] != arm:
                raise ValueError("Cannot mix banks or experiment arms in a library")
        else:
            self.data = {
                "schema_version": "astra-observed-behaviors-1",
                "bank_id": bank_id,
                "arm": arm,
                "frozen": False,
                "entries": [],
            }
            self.save()

    def save(self):
        write_json(self.path, self.data)

    def record(self, *, task, program, hypothesis, evidence):
        if (
            self.data["frozen"]
            or not evidence
            or any(e["split"] not in ("development", "validation") for e in evidence)
        ):
            raise ValueError(
                "Frozen/evaluation evidence must not update the skill library"
            )
        if any(
            type(e["success"]) is not bool
            or not e.get("trace_sha256")
            or not e.get("reset_id")
            for e in evidence
        ):
            raise ValueError(
                "Observed behavior requires actual outcome and trace identity"
            )
        dev = {e["reset_id"] for e in evidence if e["split"] == "development"}
        val = {e["reset_id"] for e in evidence if e["split"] == "validation"}
        if dev & val:
            raise ValueError("Repair validation must use separate resets")
        successful = any(e["success"] for e in evidence if e["split"] == "development")
        validated = (
            successful
            and len(val) >= 2
            and all(e["success"] for e in evidence if e["split"] == "validation")
        )
        row = {
            "entry_id": digest([task, program, evidence]),
            "task": task,
            "program": copy.deepcopy(program),
            "hypothesis": hypothesis,
            "status": "validated_on_new_resets"
            if validated
            else "observed_development_success"
            if successful
            else "observed_failure",
            "evidence": copy.deepcopy(evidence),
            "unseen_task_transfer": "not_established",
            "skill_type": "pi05_input_settings"
            if self.arm == "input_skill_library"
            else "demonstration_action_composition",
            "validation_scope": "whole program; individual stage causal effects not established",
            "activation_and_termination": "per-stage guards use live robot state and executed-action counts",
            "observed_outcomes": {
                "successes": sum(e["success"] for e in evidence),
                "trials": len(evidence),
            },
        }
        if row["entry_id"] not in {e["entry_id"] for e in self.data["entries"]}:
            self.data["entries"].append(row)
            self.save()
        return row

    def retrieve(self, task, limit=6):
        terms = set(task.lower().split()) - {"the", "on", "in", "put", "and", "of"}

        def rank(row):
            return (
                row["status"] == "validated_on_new_resets",
                len(terms & set(row["task"].lower().split())),
            )

        return copy.deepcopy(
            sorted(self.data["entries"], key=rank, reverse=True)[:limit]
        )

    def freeze(self):
        self.data["frozen"] = True
        self.save()
        snapshot = copy.deepcopy(self.data)
        write_json(
            self.directory / "evaluation_snapshot.json",
            {"sha256": digest(snapshot), "library": snapshot},
        )
        return digest(snapshot)
