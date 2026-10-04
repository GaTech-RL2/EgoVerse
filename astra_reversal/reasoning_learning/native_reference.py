"""Reuse a measured native baseline only under an exact deployment identity.

Reusing a score creates no new evaluation samples and is never another seed or
replicate. Updated checkpoints always receive fresh autonomous evaluations.
"""

import copy
import json
from pathlib import Path

from astra_reversal.records import file_sha256

SIGNATURE_FIELDS = (
    "format",
    "backend",
    "horizon",
    "model_action_dim",
    "device",
    "dtype",
    "artifact_sha256",
    "tokenizer_sha256",
    "model_source_sha256",
    "adapter_source_sha256",
    "processor_source_sha256",
    "torch_version",
    "transformers_version",
    "transformers_source_sha256",
    "normalization",
    "provenance",
    "input_profile",
    "input_profile_source_sha256",
    "runtime_overrides",
)


def native_signature(metadata):
    return {key: copy.deepcopy(metadata[key]) for key in SIGNATURE_FIELDS}


def reuse_native_evaluation(
    path, expected_sha256, *, protocol, metadata, preflight, task, seed, manifest
):
    path = Path(path)
    if file_sha256(path) != expected_sha256:
        raise ValueError("Native reference checksum differs")
    reference = json.loads(path.read_text())
    if reference["task"] != task or reference["seed"] != seed:
        raise ValueError("Native reference task/seed differs")
    if reference["policy_signature"] != native_signature(metadata):
        raise ValueError("Native reference deployment identity differs")
    if any(reference[key] != protocol[key] for key in ("checkpoint", "environment")):
        raise ValueError("Native reference control/evaluation protocol differs")
    if (
        preflight["zero_adapter_max_abs"] != 0
        or preflight["native_zero_guidance_max_abs"] != 0
    ):
        raise ValueError("Reusing native evaluation requires exact zero-adapter parity")
    point = copy.deepcopy(reference["point"])
    if (
        point["policy_version"] != 0
        or point["collection_rollouts"] != 0
        or point["collection_steps"] != 0
    ):
        raise ValueError("Only the unchanged initial policy may reuse a baseline")
    resets = protocol["pilot"]["autonomous_evaluation_reset_indices"]
    entries = {row["initial_state_id"]: row for row in manifest["episodes"]}
    if len(point["episodes"]) != len(resets) or point["rollouts"] != len(resets):
        raise ValueError("Native reference evaluation denominator differs")
    for reset_id, result in zip(resets, point["episodes"], strict=True):
        entry, audit = entries[reset_id], result["reset_audit"]
        if result["episode_id"] != entry["episode_id"]:
            raise ValueError("Native reference reset identity differs")
        if any(
            audit[key] != entry[key]
            for key in ("reset_state_sha256", "reset_model_sha256", "bddl_sha256")
        ):
            raise ValueError("Native reference reset scene differs")
    if sum(bool(row["success"]) for row in point["episodes"]) != point["successes"]:
        raise ValueError("Native reference aggregate differs")
    point["reference_evaluation_steps"] = point["evaluation_steps_cumulative"]
    point["evaluation_steps_cumulative"] = 0
    point["reused_from_workflow"] = reference["workflow"]
    point["reference_sha256"] = expected_sha256
    point["new_evaluation_interactions"] = 0
    point["independent_new_evaluation_replicate"] = False
    return point
