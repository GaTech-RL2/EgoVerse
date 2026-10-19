"""Frozen four-cell pilot: model by policy execution, all F-adaptive."""

import copy
import random
from pathlib import Path

import yaml

from .common import digest
from .proxy import Limits
from .replication import validate_model
from .robosuite_benchmark import HORIZON, MIMICGEN_COMMIT, ROBOSUITE_COMMIT

STUDY = Path(__file__).resolve().parents[2] / "experiments/threading_hardware_interface"
ARMS = ("astra_CAP", "astra_direct", "deepseek_CAP", "deepseek_direct")
RESOURCE_CAPS = (
    "wall_seconds",
    "workflow_tokens",
    "tool_calls",
    "observer_calls",
    "actor_output_tokens",
    "observer_output_tokens",
    "context_tokens",
)


def design():
    base = STUDY.parent / "libero_hardware_interface"
    astra = yaml.safe_load((base / "preregistration-adaptive.yaml").read_text())
    deepseek = yaml.safe_load(
        (base / "preregistration-deepseek-selfhost-adaptive.yaml").read_text()
    )
    limits = copy.deepcopy(astra["limits"])
    limits["steps"] = HORIZON
    return {
        "study": "threading-four-combination-f-adaptive-20261019-api-v2",
        "version": 2,
        "stage": "descriptive_pilot",
        "task": "Threading_D0",
        "robot": "Panda",
        "controller": "OSC_POSE",
        "interface": "F_adaptive",
        "conditions": list(ARMS),
        "models": {"astra": astra["model"], "deepseek": deepseek["model"]},
        "deepseek_identity": {
            key: deepseek["model_replication"][key]
            for key in (
                "requested_model",
                "requested_hf_revision",
                "pool_with_astra",
                "self_hosted",
                "serving_image",
            )
        },
        "source_pins": {"mimicgen": MIMICGEN_COMMIT, "robosuite": ROBOSUITE_COMMIT},
        "environment_seeds": list(range(10)),
        "randomization_seed": 20261019,
        "image_size": 256,
        "control_frequency": 20,
        "limits": limits,
        "additional_settling_steps": 0,
        "success": "Unmodified native Threading_D0._check_success after each native control step; private to evaluator.",
        "adaptation": "Fresh per trial; hardware sensors, calibration, bounded dynamics probes, notes and read-only observation/coordinate helpers in all four cells. CAP additionally runs generated robot-control programs through scratch_execute. Direct issues native act commands through model calls.",
        "command_api": "CAP receives the exact nested act envelope JSON Schema and read-packet freshness fields; direct receives that same schema through its act tool. Version-1 attempts hid the CAP command schema and are diagnostic only, excluded from this comparison.",
        "pairing": "Identical seeded native reset state, model and initial sensor hashes for every cell; independent fresh-process action replay.",
        "restart_policy": "No automatic trial reruns or action redispatch. Archive completed trials and halt on unresolved infrastructure failure.",
        "timing": "Actor start to first native success; excludes provisioning/reset/replay. Provider round trips include actual server queue/compute; client backoff separately recorded. Cross-model timing includes different serving systems and reasoning settings.",
        "conditions_analysis": "Pilot only, ten starts; no confirmatory performance claim.",
        "task_ids": [0],
        "pilot_indices": list(range(10)),
        "replicates": 1,
        "primary_contrast": ["astra_CAP", "astra_direct"],
        "contrasts": [
            ["astra_CAP", "astra_direct"],
            ["deepseek_CAP", "deepseek_direct"],
            ["astra_CAP", "deepseek_CAP"],
            ["astra_direct", "deepseek_direct"],
        ],
        "analysis": {"bootstrap_samples": 10000, "bootstrap_seed": 20261019},
        "speed_analysis": {
            "report_times_seconds": [
                60,
                300,
                600,
                1200,
                1800,
                3600,
                7200,
                14400,
                28800,
            ],
            "report_times_are_limits": False,
        },
    }


def load(path=None):
    manifest = yaml.safe_load(Path(path or STUDY / "preregistration.yaml").read_text())
    validate(manifest)
    return manifest


def validate(manifest):
    if manifest != design():
        raise ValueError("threading_pilot_design_changed")
    if manifest["limits"]["steps"] != 400 or manifest["limits"]["max_repeat"] != 10:
        raise ValueError("native_horizon_or_repeat_changed")
    if any(manifest["limits"][key] is not None for key in RESOURCE_CAPS):
        raise ValueError("threading_pilot_requires_no_resource_caps")
    for key, model in manifest["models"].items():
        check = {"model": model, "observer_model": model["identifier"]}
        if key == "deepseek":
            check["model_replication"] = manifest["deepseek_identity"]
        validate_model(check)
    return Limits(**manifest["limits"])


def schedule(manifest):
    validate(manifest)
    rng, rows = random.Random(manifest["randomization_seed"]), []
    # Rotate a shuffled four-cell order to balance which policy runs first.
    order = list(ARMS)
    rng.shuffle(order)
    for index, seed in enumerate(manifest["environment_seeds"]):
        block = order[index % 4 :] + order[: index % 4]
        for position, arm in enumerate(block):
            model, policy_mode = arm.split("_", 1)
            rows.append(
                {
                    "trial_id": "threading-s%03d-%s" % (seed, arm),
                    "condition": arm,
                    "model_key": model,
                    "policy_mode": policy_mode,
                    "interface_condition": "F_adaptive",
                    "split": "pilot",
                    "task_id": 0,
                    "init_state_index": index,
                    "env_seed": seed,
                    "replicate": 0,
                    "order": len(rows),
                    "block_order": position,
                    "manifest_sha256": digest(manifest),
                }
            )
    return rows
