"""Three auditable data-selection conditions from the same real collections."""

import base64
import copy
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from astra_reversal.records import digest, file_sha256

from . import hindsight
from .evidence import training_windows
from .replay_data import json_lines, verify_episode

VARIANTS = ("online_gate", "hindsight_gate", "successful_episode")


def hindsight_labels(request, proposal, executed, result, *, instruction, workflow):
    """Bind every reviewed pixel, state, command and outcome to the real ledger."""
    value = copy.deepcopy(proposal)
    if value.pop("request_fingerprint", None) != request["request_fingerprint"]:
        raise ValueError("Credit response belongs to another request")
    reviewed = hindsight.parse_proposal(value, request)
    if (
        request["episode_id"] != result["episode_id"]
        or request["instruction"] != instruction
        or request["source_workflow"] != workflow
        or request["environment_success"] != result["success"]
        or request["segments"][-1]["end_step_exclusive"] != len(executed)
    ):
        raise ValueError("Credit must refer to this completed collection")
    for segment in request["segments"]:
        actual = executed[segment["start_step"] : segment["end_step_exclusive"]]
        if segment["actions"] != [r["action"] for r in actual]:
            raise ValueError("Credit request changed an executed command")
    for frame in request["frames"]:
        raw = {}
        for key, wire in zip(
            ("observation/image", "observation/wrist_image"),
            frame["images"],
            strict=True,
        ):
            with Image.open(
                io.BytesIO(base64.b64decode(wire["data"], validate=True))
            ) as image:
                if image.size != (wire["width"], wire["height"]) or image.mode != "RGB":
                    raise ValueError("Recorded RGB dimensions differ")
                raw[key] = np.asarray(image).copy()
        raw["observation/state"] = np.asarray(frame["state"], dtype=np.float32)
        if frame["step"] < len(executed):
            raw["prompt"] = instruction
            expected = executed[frame["step"]]["observation_id"]
        else:
            expected = executed[-1]["after_observation_sha256"]
        if digest(raw) != expected:
            raise ValueError("Credit images/state differ from the actual trajectory")
    labels = copy.deepcopy(executed)
    for review in reviewed["segments"]:
        segment = request["segments"][review["segment_id"]]
        outcome = review["outcome"] if review["confidence"] >= 0.8 else "ambiguous"
        for index in range(segment["start_step"], segment["end_step_exclusive"]):
            row = labels[index]
            row["evidence"] = outcome
            # The execution ledger's original stage records whether the command
            # was assisted. Retrospective phase labels cannot bypass its gate.
            row["stage"] = (
                "correction"
                if executed[index]["stage"] == "correction"
                else ("setup" if review["phase"] == "setup" else "continuation")
            )
            row["hindsight_request_fingerprint"] = request["request_fingerprint"]
    return labels


def successful_windows(executed, result, *, horizon=10, stride=5):
    """Binary-success BC control: no claim that every action was locally useful."""
    if not result["success"]:
        return []
    if result["assisted_chunks"] or any(r["preference"] is not None for r in executed):
        raise ValueError("This control requires an unassisted source trajectory")
    windows = []
    for start in range(0, len(executed) - horizon + 1, stride):
        rows = executed[start : start + horizon]
        if any(
            not r["executed"]
            or r["step"] != start + i
            or r["episode_id"] != result["episode_id"]
            for i, r in enumerate(rows)
        ):
            raise ValueError("Successful-episode BC requires complete real commands")
        actions = np.asarray([r["action"] for r in rows], dtype=np.float32)
        if actions.shape != (horizon, 7) or not np.isfinite(actions).all():
            raise ValueError("Malformed actual commands")
        record = {
            "episode_id": result["episode_id"],
            "observation_id": rows[0]["observation_id"],
            "start_step": start,
            "end_step_exclusive": start + horizon,
            "event_ids": [],
            "stages": ["successful_episode"],
            "evidence": "successful_episode",
            "source": "executed_commands",
            "actions": actions.tolist(),
            "policy_versions": sorted({r["policy_version"] for r in rows}),
        }
        record["window_id"] = digest(record)
        windows.append(record)
    return windows


def load(directory, expected_manifest_sha256=None):
    directory = Path(directory)
    path = directory / "manifest.json"
    if expected_manifest_sha256 and file_sha256(path) != expected_manifest_sha256:
        raise ValueError("Credit-selection manifest checksum differs")
    manifest = json.loads(path.read_text())
    if (
        manifest["schema"] != "reasoning-credit-selection-data-1"
        or manifest["source_protocol"]["teacher"]["frs_action_steering"]
    ):
        raise ValueError("Unknown or excluded credit-selection source")
    for relative, expected in manifest["files"].items():
        part = Path(relative)
        if (
            part.is_absolute()
            or ".." in part.parts
            or file_sha256(directory / part) != expected
        ):
            raise ValueError("Credit source path or checksum differs")
    variants, observations, results, used = {k: [] for k in VARIANTS}, {}, [], set()
    for episode in manifest["episodes"]:
        folder = episode["directory"]
        if Path(folder).name != folder:
            raise ValueError("Episode directory must be a basename")
        source = directory / folder
        online, result = verify_episode(source, horizon=10)
        if result["episode_id"] != episode["episode_id"] or result["episode_id"] in {
            r["episode_id"] for r in results
        }:
            raise ValueError("Credit source episodes differ or repeat")
        results.append(result)
        executed = json_lines(source / "executed_steps.jsonl")
        request = json.loads((source / "credit_request.json").read_text())
        proposal = json.loads((source / "credit_proposal.json").read_text())
        labels = hindsight_labels(
            request,
            proposal,
            executed,
            result,
            instruction=manifest["instruction"],
            workflow=manifest["source_workflow"],
        )
        current = {
            "online_gate": online,
            "hindsight_gate": training_windows(labels, horizon=10),
            "successful_episode": successful_windows(executed, result),
        }
        used.update(
            str(Path(folder) / name)
            for name in (
                "admission.jsonl",
                "executed_steps.jsonl",
                "result.json",
                "credit_request.json",
                "credit_proposal.json",
            )
        )
        for variant, windows in current.items():
            variants[variant].extend(windows)
            for window in windows:
                oid = window["observation_id"]
                if len(oid) != 64 or any(c not in "0123456789abcdef" for c in oid):
                    raise ValueError("Pre-action observation identity must be SHA256")
                relative = str(Path(folder) / (oid + ".npz"))
                used.add(relative)
                if oid not in observations:
                    with np.load(directory / relative, allow_pickle=False) as data:
                        raw = {key: data[key] for key in data.files}
                    raw["prompt"] = str(raw["prompt"].item())
                    if digest(raw) != oid or raw["prompt"] != manifest["instruction"]:
                        raise ValueError(
                            "Student input differs from its real pre-action observation"
                        )
                    observations[oid] = raw
    if used != set(manifest["files"]):
        raise ValueError("Manifest must name exactly the consumed credit source files")
    if (
        sum(r["total_control_steps"] for r in results)
        != manifest["source_collection_control_steps"]
    ):
        raise ValueError("Failed and successful source interactions must all count")
    for variant, windows in variants.items():
        if (
            not windows
            or len({w["window_id"] for w in windows}) != len(windows)
            or len(windows) != manifest["variant_window_counts"][variant]
        ):
            raise ValueError("Credit variant window count or identity differs")
    return variants, observations, manifest
