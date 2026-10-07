"""Bounded, auditable prompt scheduling; no policy or simulator implementation."""

import copy
import json
from pathlib import Path

import numpy as np

from astra_reversal.complex_manipulation import language_teacher as teacher
from astra_reversal.records import digest, file_sha256

METHOD = "astra_phase_prompt_pi05"
MAX_CALLS = 16


def prior_failure(directory):
    """Four fixed chronological frames from one completed native episode."""
    import imageio.v2 as imageio

    directory = Path(directory)
    result = json.loads((directory / "result.json").read_text())
    reset = json.loads((directory / "reset.json").read_text())
    with np.load(directory / "initial_observation.npz", allow_pickle=False) as arrays:
        initial = {k: arrays[k] for k in arrays.files if k != "simulator_state"}
    initial["prompt"] = str(initial["prompt"])
    if digest(initial) != reset["observation_sha256"]:
        raise ValueError("Baseline observation artifact differs from reset receipt")
    if (
        not result["episode_complete"]
        or result["success"]
        or result["method"] != "native_pi05"
        or result["executed_actions"] != result["action_limit"]
    ):
        raise ValueError("Guidance requires a completed, failed native comparison")
    video = directory / "rollout.mp4"
    with imageio.get_reader(video) as reader:
        count = reader.count_frames()
        if count * 2 != result["executed_actions"]:
            raise ValueError("Prior video's two-control frame cadence differs")
        indices = [int((count - 1) * f) for f in (0.25, 0.5, 0.75, 1)]
        frames = [
            {
                "origin": "prior_native_failure",
                "step": 2 * (i + 1),
                "state": None,
                "images": {
                    "left_right_wrist_panorama": teacher.png_wire(reader.get_data(i))
                },
            }
            for i in indices
        ]
    return {
        "reset": reset,
        "initial_observation": initial,
        "episode_id": result["episode_id"],
        "frames": frames,
        "video_sha256": file_sha256(video),
        "executed_controls": result["executed_actions"],
        "success": False,
        "stop_reason": result["stop_reason"],
    }


def check_reset(actual, expected, actual_observation=None, expected_observation=None):
    fields = (
        "seed",
        "instruction",
        "state_sha256",
        "model_sha256",
        "horizon",
        "execution_prefix",
        "initial_success",
    )
    mismatches = [key for key in fields if actual.get(key) != expected.get(key)]
    if mismatches:
        raise ValueError("Matched native reset differs: " + ", ".join(mismatches))
    receipt = {
        "matched": True,
        "checked_fields": list(fields),
        "observation_match": "exact",
        "pixel_differences": {},
    }
    if actual.get("observation_sha256") == expected.get("observation_sha256"):
        return receipt
    if actual_observation is None or expected_observation is None:
        raise ValueError("Matched native reset differs: observation_sha256")
    keys = {"prompt", "observation/state", *teacher.CAMERAS.values()}
    if (
        set(actual_observation) != keys
        or set(expected_observation) != keys
        or digest(actual_observation) != actual["observation_sha256"]
        or digest(expected_observation) != expected["observation_sha256"]
    ):
        raise ValueError("Reset observation evidence does not match its hashes")
    for key in ("prompt", "observation/state"):
        if not np.array_equal(actual_observation[key], expected_observation[key]):
            raise ValueError("Matched native reset differs: " + key)
    for key in teacher.CAMERAS.values():
        a, b = (
            np.asarray(actual_observation[key]),
            np.asarray(expected_observation[key]),
        )
        if (
            a.shape != (224, 224, 3)
            or b.shape != a.shape
            or a.dtype != np.uint8
            or b.dtype != np.uint8
        ):
            raise ValueError("Unexpected reset camera shape or dtype")
        delta = np.abs(a.astype(np.int16) - b.astype(np.int16))
        count, magnitude = int(np.count_nonzero(delta)), int(delta.max())
        receipt["pixel_differences"][key] = {
            "changed_color_values": count,
            "max_uint8_difference": magnitude,
        }
        if count > 16 or magnitude > 1:
            raise ValueError("Reset camera difference exceeds render-rounding bound")
    receipt["observation_match"] = "bounded_uint8_render_rounding"
    receipt["render_rounding_bound"] = {
        "max_changed_color_values_per_camera": 16,
        "max_uint8_difference": 1,
    }
    return receipt


class LanguageGuide:
    def __init__(self, *, client, baseline, output):
        self.client, self.baseline, self.output = client, baseline, Path(output)
        self.next_review = 0
        self.calls = 0
        self.active = None
        self.previous = None
        self.decisions = []
        self.exhausted = False

    def _save(self, name, value):
        path = self.output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")

    def bind_reset(self, actual, observation):
        result = check_reset(
            actual,
            self.baseline["reset"],
            observation,
            self.baseline["initial_observation"],
        )
        self._save(
            "paired_reset.json",
            {
                **result,
                "baseline_episode_id": self.baseline["episode_id"],
                "baseline_video_sha256": self.baseline["video_sha256"],
                "baseline_executed_controls": self.baseline["executed_controls"],
                "baseline_stop_reason": self.baseline["stop_reason"],
            },
        )

    def prepare(self, observation, *, step, episode_id):
        if step >= self.next_review and not self.exhausted:
            if self.calls == MAX_CALLS:
                self.active = None
                self.exhausted = True
                self._save(
                    "budget_exhausted.json",
                    {"step": step, "action": "restore_original_prompt"},
                )
            else:
                current = teacher.live_snapshot(observation, step)
                snapshots = copy.deepcopy(self.baseline["frames"])
                if self.previous is not None:
                    snapshots.append({**self.previous, "origin": "previous_live"})
                snapshots.append(current)
                context = {
                    "original_instruction": observation["prompt"],
                    "remaining_calls_including_this": MAX_CALLS - self.calls,
                    "active_subgoal": self.active["subgoal"] if self.active else None,
                    "recent_decisions": self.decisions[-2:],
                    "prior_native_episode": {
                        "episode_id": self.baseline["episode_id"],
                        "success": False,
                        "executed_controls": self.baseline["executed_controls"],
                        "stop_reason": self.baseline["stop_reason"],
                        "video_sha256": self.baseline["video_sha256"],
                    },
                }
                request = teacher.build_request(
                    episode_id=episode_id,
                    request_index=self.calls,
                    step=step,
                    snapshots=snapshots,
                    context=context,
                )
                index = self.calls
                self.calls += 1
                self._save(f"request_{index:02d}.json", request)
                # Exceptions propagate: a provider outage is not a guided failure
                # or permission to execute an unrecorded native fallback.
                proposal = self.client.propose(request)
                self._save(f"proposal_{index:02d}.json", proposal)
                self.active = proposal
                self.next_review = step + proposal["next_review_controls"]
                self.previous = current
                self.decisions.append(
                    {
                        key: proposal[key]
                        for key in (
                            "observation_step",
                            "method",
                            "subgoal",
                            "observed_evidence",
                            "completion_signal",
                            "next_review_controls",
                            "decision_id",
                        )
                    }
                )
        model_input = dict(observation)
        model_input["prompt"] = teacher.policy_prompt(
            observation["prompt"], self.active
        )
        return model_input, {
            "assisted": self.active is not None
            and self.active["method"] == "phase_prompt",
            "decision_id": self.active["decision_id"] if self.active else None,
            "next_review_step": self.next_review,
            "budget_exhausted": self.exhausted,
        }

    def summary(self):
        from astra_reversal.codex_accounting import summarize_codex_calls

        usage = summarize_codex_calls(self.client.records)
        total = usage["tokens"]["total_tokens"]
        return {
            "teacher_calls": self.calls,
            "teacher_tokens": total["sum"] if total["complete"] else None,
            "teacher_usage": usage,
            "teacher_seconds": sum(r["latency_seconds"] for r in self.client.records),
            "guidance_budget_exhausted": self.exhausted,
            "policy_updates": 0,
        }
