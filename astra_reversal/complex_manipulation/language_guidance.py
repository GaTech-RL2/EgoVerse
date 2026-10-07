"""Bounded, auditable prompt scheduling; no policy or simulator implementation."""

import copy
import gzip
import hashlib
import json
import xml.etree.ElementTree as ET
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
    model_xml = gzip.decompress((directory / "initial_model.xml.gz").read_bytes())
    if hashlib.sha256(model_xml).hexdigest() != reset["model_sha256"]:
        raise ValueError("Baseline XML artifact differs from reset receipt")
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
        "initial_model_xml": model_xml,
        "episode_id": result["episode_id"],
        "frames": frames,
        "video_sha256": file_sha256(video),
        "executed_controls": result["executed_actions"],
        "success": False,
        "stop_reason": result["stop_reason"],
    }


def model_fingerprint(xml):
    """Retain MJCF values, normalizing only redundant OBJ media-type declarations.

    MuJoCo selects OBJ from the .obj suffix when content_type is absent. Its
    exporter sometimes retains the equivalent explicit declaration. Numeric
    attributes, mesh paths, child order and all other settings remain compared.
    """
    root = ET.fromstring(xml)
    normalized = []
    for mesh in root.findall("./asset/mesh"):
        if mesh.get("content_type") == "model/obj" and mesh.get(
            "file", ""
        ).lower().endswith(".obj"):
            normalized.append(mesh.get("name"))
            del mesh.attrib["content_type"]
    return hashlib.sha256(ET.tostring(root, encoding="utf-8")).hexdigest(), normalized


def check_reset(
    actual,
    expected,
    actual_observation=None,
    expected_observation=None,
    actual_xml=None,
    expected_xml=None,
):
    fields = (
        "seed",
        "instruction",
        "state_sha256",
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
        "model_match": "exact_raw_xml",
    }
    if actual.get("model_sha256") != expected.get("model_sha256"):
        if actual_xml is None or expected_xml is None:
            raise ValueError("Matched native reset differs: model_sha256")
        if (
            hashlib.sha256(actual_xml).hexdigest() != actual["model_sha256"]
            or hashlib.sha256(expected_xml).hexdigest() != expected["model_sha256"]
        ):
            raise ValueError("Reset XML evidence does not match its hashes")
        actual_hash, actual_normalized = model_fingerprint(actual_xml)
        expected_hash, expected_normalized = model_fingerprint(expected_xml)
        if actual_hash != expected_hash:
            raise ValueError(
                "Matched native reset differs: model geometry or configuration"
            )
        receipt.update(
            model_match="equivalent_obj_content_type",
            model_comparison_sha256=actual_hash,
            actual_normalized_obj_meshes=actual_normalized,
            baseline_normalized_obj_meshes=expected_normalized,
            actual_raw_model_sha256=actual["model_sha256"],
            baseline_raw_model_sha256=expected["model_sha256"],
        )
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
    def __init__(
        self,
        *,
        client,
        baseline,
        output,
        teacher_module=teacher,
        representation_method=None,
    ):
        self.client, self.baseline, self.output = client, baseline, Path(output)
        if representation_method not in (None, "tei", "tli"):
            raise ValueError("Unknown representation method")
        self.teacher = teacher_module
        self.representation_method = representation_method
        self.identity_method = representation_method or "phase_prompt"
        self.method_name = f"astra_{self.identity_method}_pi05"
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

    def bind_reset(self, actual, observation, model_xml):
        result = check_reset(
            actual,
            self.baseline["reset"],
            observation,
            self.baseline["initial_observation"],
            model_xml,
            self.baseline["initial_model_xml"],
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
        teacher = self.teacher
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
                if self.representation_method:
                    context["allowed_intervention"] = self.representation_method
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
                if self.representation_method:
                    self.decisions[-1]["alpha"] = proposal["alpha"]
        model_input = dict(observation)
        model_input["prompt"] = teacher.policy_prompt(
            observation["prompt"], self.active
        )
        metadata = {
            "assisted": self.active is not None and self.active["method"] != "native",
            "decision_id": self.active["decision_id"] if self.active else None,
            "next_review_step": self.next_review,
            "budget_exhausted": self.exhausted,
        }
        if self.representation_method:
            metadata["intervention"] = self.active
        return model_input, metadata

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
