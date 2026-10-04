"""Load actual useful execution windows for a fixed-data learner ablation.

No simulator or teacher is called. Labels are reconstructed from the recorded
admission ledger and checked against the independently appended action ledger.
"""

import json
from pathlib import Path

import numpy as np

from astra_reversal.records import digest, file_sha256

from .evidence import training_windows

IMMUTABLE_STEP_FIELDS = (
    "episode_id",
    "step",
    "observation_id",
    "action",
    "executed",
    "policy_version",
    "batch_id",
    "event_id",
    "preference",
    "environment_success",
    "terminated",
    "after_observation_sha256",
)


def json_lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def verify_episode(directory, *, horizon, mask_loss_by_evidence=False):
    directory = Path(directory)
    admissions = json_lines(directory / "admission.jsonl")
    if len(admissions) != 1:
        raise ValueError("One finalized admission record is required")
    labels = admissions[0]["steps"]
    executed = json_lines(directory / "executed_steps.jsonl")
    result = json.loads((directory / "result.json").read_text())
    if len(labels) != len(executed) or len(executed) != result["actions_executed"]:
        raise ValueError("Replayed labels differ from actual execution length")
    for index, (label, actual) in enumerate(zip(labels, executed, strict=True)):
        if actual["step"] != index or actual["episode_id"] != result["episode_id"]:
            raise ValueError("Execution ledger is not a single contiguous episode")
        if any(label[key] != actual[key] for key in IMMUTABLE_STEP_FIELDS):
            raise ValueError("An admitted label changed actual execution evidence")
    windows = training_windows(labels, horizon=horizon)
    if (
        windows != admissions[0]["windows"]
        or len(windows) != result["admitted_windows"]
    ):
        raise ValueError("Recomputed complete windows differ from recorded admission")
    if result["total_control_steps"] != len(executed) + result["initialization_steps"]:
        raise ValueError("Source collection interaction accounting differs")
    if mask_loss_by_evidence:
        windows = training_windows(labels, horizon=horizon, mask_loss_by_evidence=True)
    return windows, result


def load(directory, expected_manifest_sha256=None):
    """Verify fixed source bytes, executed targets and their own observations."""
    directory = Path(directory)
    path = directory / "manifest.json"
    if expected_manifest_sha256 and file_sha256(path) != expected_manifest_sha256:
        raise ValueError("Replay manifest checksum differs")
    manifest = json.loads(path.read_text())
    if manifest["schema"] != "reasoning-executed-replay-1":
        raise ValueError("Unknown executed replay schema")
    if manifest["source_protocol"]["teacher"]["frs_action_steering"]:
        raise ValueError("FRS steering data is excluded from this study")
    admission_rule = manifest.get("admission_rule", "all_steps_observed_useful")
    if admission_rule not in ("all_steps_observed_useful", "mask_loss_by_evidence"):
        raise ValueError("Unknown replay evidence admission rule")
    for relative, expected in manifest["files"].items():
        part = Path(relative)
        if part.is_absolute() or ".." in part.parts:
            raise ValueError("Replay paths must stay within the data bundle")
        if file_sha256(directory / part) != expected:
            raise ValueError("Replay source file checksum differs")
    windows, observations, results = [], {}, []
    used = set()
    horizon = manifest["source_protocol"]["checkpoint"]["runtime_action_horizon"]
    for episode in manifest["episodes"]:
        folder = episode["directory"]
        if Path(folder).name != folder:
            raise ValueError("Source episode directory must be a basename")
        source = directory / folder
        for name in ("admission.jsonl", "executed_steps.jsonl", "result.json"):
            used.add(str(Path(folder) / name))
        current, result = verify_episode(
            source,
            horizon=horizon,
            mask_loss_by_evidence=admission_rule == "mask_loss_by_evidence",
        )
        if result["episode_id"] != episode["episode_id"]:
            raise ValueError("Replay episode identity differs")
        if result["episode_id"] in {r["episode_id"] for r in results}:
            raise ValueError("Repeated physical episode cannot add experience")
        for window in current:
            oid = window["observation_id"]
            if len(oid) != 64 or any(c not in "0123456789abcdef" for c in oid):
                raise ValueError("Observation identity is not a SHA256 digest")
            filename = str(Path(folder) / (oid + ".npz"))
            used.add(filename)
            with np.load(directory / filename, allow_pickle=False) as data:
                raw = {key: data[key] for key in data.files}
            raw["prompt"] = str(raw["prompt"].item())
            if digest(raw) != oid:
                raise ValueError("Replayed pre-action observation identity differs")
            if raw["prompt"] != manifest["instruction"]:
                raise ValueError("Replay training must retain the original instruction")
            observations[oid] = raw
        windows.extend(current)
        results.append(result)
    if used != set(manifest["files"]):
        raise ValueError("Replay manifest must name exactly the consumed source files")
    if not windows or len({w["window_id"] for w in windows}) != len(windows):
        raise ValueError("Nonempty unique executed training windows are required")
    controls = sum(r["total_control_steps"] for r in results)
    if controls != manifest["source_collection_control_steps"]:
        raise ValueError("Replay collection cost differs from source episodes")
    if len(windows) != manifest["admitted_windows"]:
        raise ValueError("Replay window count differs")
    return windows, observations, manifest
