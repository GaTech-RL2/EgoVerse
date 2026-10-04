"""Verify a saved policy and its costs before an interrupted evaluation restarts."""

import json
from pathlib import Path

from astra_reversal.records import file_sha256

from .native_reference import native_signature


def load(directory, expected_sha256):
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    if file_sha256(manifest_path) != expected_sha256:
        raise ValueError("Evaluation input manifest checksum differs")
    source = json.loads(manifest_path.read_text())
    names = {
        "adapters.pt",
        "checkpoint.json",
        "protocol.json",
        "runtime.json",
        "updates.json",
        "collection.json",
        "preflight.json",
        "retained_run.json",
        "resets.json",
    }
    if (
        source["schema"] != "reasoning-frozen-checkpoint-evaluation-1"
        or set(source["files"]) != names
    ):
        raise ValueError("Unknown frozen evaluation input inventory")
    for name, expected in source["files"].items():
        if file_sha256(directory / name) != expected:
            raise ValueError("Frozen evaluation source artifact checksum differs")
    values = {
        name: json.loads((directory / name).read_text())
        for name in names
        if name.endswith(".json")
    }
    validate(source, values)
    return source


def validate(source, values):
    """No checkpoint selection, cost erasure, or task/reset substitution."""
    retained = values["retained_run.json"]
    runtime, protocol = values["runtime.json"], values["protocol.json"]
    updates, episodes = values["updates.json"], values["collection.json"]
    if (
        source["source_workflow"] != retained["workflow"]
        or any(
            source["source_task"] != r["task"] or source["source_seed"] != r["seed"]
            for r in (retained, runtime)
        )
        or source["source_protocol"] != protocol
        or source["source_policy_signature"]
        != native_signature(values["checkpoint.json"])
        or source["episodes"] != episodes
        or source["evaluation_reset_indices"]
        != protocol["pilot"]["autonomous_evaluation_reset_indices"]
        or len(episodes)
        not in protocol["pilot"]["evaluation_after_collection_rollouts"]
    ):
        raise ValueError("Saved evaluation deployment or schedule differs")
    if (
        not updates
        or updates[-1]["policy_version"] != source["policy_version"]
        or updates[-1]["after_sha256"] != source["adapter_parameter_sha256"]
        or updates[-1]["checkpoint"]["sha256"] != source["files"]["adapters.pt"]
        or retained["latest_updated_policy_version"] != source["policy_version"]
        or retained["latest_update_has_autonomous_evaluation"]
    ):
        raise ValueError(
            "Resume only the latest saved, incompletely evaluated checkpoint"
        )
    if (
        sum(e["total_control_steps"] for e in episodes)
        != source["source_collection_control_steps"]
        or retained["collection_steps_retained"]
        != source["source_collection_control_steps"]
        or retained["teacher_usage"]["total_tokens"]
        != source["source_teacher_total_tokens"]
        or retained["evaluation_steps_retained"]
        != source["source_evaluation_control_steps_retained"]
    ):
        raise ValueError("Original collection, teacher or evaluation costs differ")
    preflight = values["preflight.json"]
    if (
        preflight["zero_adapter_max_abs"] != 0
        or preflight["native_zero_guidance_max_abs"] != 0
    ):
        raise ValueError("The original deployment did not pass native identity gates")
