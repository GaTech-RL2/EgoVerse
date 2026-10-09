"""Frozen RoboCasa transfer design and matched, breadth-first scheduling."""

import copy
import random
from collections import defaultdict
from statistics import mean, median

from .protocol import CONDITIONS, RESOURCE_CAPS
from .proxy import Limits
from .robocasa import ROBOCASA_COMMIT, ROBOSUITE_COMMIT


def validate(manifest):
    if manifest["benchmark"] != "robocasa365" or manifest["conditions"] != list(
        CONDITIONS
    ):
        raise ValueError("robocasa_three_arm_design_required")
    if manifest["upstream"] != {
        "robocasa": ROBOCASA_COMMIT,
        "robosuite": ROBOSUITE_COMMIT,
    }:
        raise ValueError("upstream_pin_changed")
    tasks = manifest["catalog"]["tasks"]
    if [t["task_id"] for t in tasks] != list(range(50)) or len(
        {t["name"] for t in tasks}
    ) != 50:
        raise ValueError("target50_required")
    if manifest["task_ids"] != list(range(50)) or manifest["scene_split"] != "target":
        raise ValueError("target50_split_changed")
    if any(type(t["horizon"]) is not int or t["horizon"] < 1 for t in tasks):
        raise ValueError("official_horizon_required")
    indices = manifest["pilot_indices"]
    if (
        not indices
        or len(indices) != len(set(indices))
        or any(type(i) is not int or i < 0 for i in indices)
    ):
        raise ValueError("invalid_scenario_seeds")
    if any(manifest["limits"][key] is not None for key in RESOURCE_CAPS):
        raise ValueError("resource_caps_not_authorized")
    if (
        manifest["limits"]["steps"] is not None
        or manifest["horizon_policy"] != "official_task_registry"
    ):
        raise ValueError("per_task_horizon_required")
    model = manifest["model"]
    if (
        model["identifier"] != "openai/openai/gpt-6-astra"
        or model["input_reservation"] != {"mode": "usage_only"}
        or model["context_truncation"] != "auto"
        or model["retries"] != 3
    ):
        raise ValueError("frozen_model_contract_changed")
    if manifest["smoke"]["name"] in {t["name"] for t in tasks}:
        raise ValueError("commissioning_task_must_be_disjoint")


def limits_for(manifest, horizon):
    return Limits(**{**manifest["limits"], "steps": horizon})


def schedule(manifest):
    validate(manifest)
    rng = random.Random(manifest["randomization_seed"])
    rows = []
    # Cover every task before another seed; keep three arms adjacent in time.
    for index in manifest["pilot_indices"]:
        tasks = copy.deepcopy(manifest["catalog"]["tasks"])
        rng.shuffle(tasks)
        for task in tasks:
            arms = list(CONDITIONS)
            rng.shuffle(arms)
            for arm in arms:
                rows.append(
                    {
                        **task,
                        "trial_id": f"pilot-t{task['task_id']:02d}-s{index:04d}-{arm}",
                        "condition": arm,
                        "split": "pilot",
                        "init_state_index": index,
                        "env_seed": index,
                        "replicate": 0,
                        "order": len(rows),
                    }
                )
    return rows


def paired_success_times(outcomes):
    """Conditional time-to-success, kept distinct from all-attempt duration."""
    blocks = defaultdict(dict)
    for row in outcomes:
        if row.get("actor_started") and row.get("split") == "pilot":
            key = (row["task_id"], row["init_state_index"], row["replicate"])
            if row["condition"] in blocks[key]:
                raise ValueError("duplicate_paired_trial")
            blocks[key][row["condition"]] = row
    result = {}
    for other in ("B0", "B"):
        pairs = []
        for key, block in blocks.items():
            if "F" not in block or other not in block:
                continue
            f, b = block["F"], block[other]
            if (f["init_state_hash"], f["env_seed"]) != (
                b["init_state_hash"],
                b["env_seed"],
            ):
                raise ValueError("unmatched_initial_state")
            if (
                f["success"]
                and b["success"]
                and f.get("independent_evaluation_passed")
                and b.get("independent_evaluation_passed")
            ):
                pairs.append(
                    {
                        "task_id": key[0],
                        "seed": key[1],
                        "F_wall_s": f["wall_s"],
                        "other_wall_s": b["wall_s"],
                        "F_steps": f["sim_steps"],
                        "other_steps": b["sim_steps"],
                        "wall_difference_F_minus_other": f["wall_s"] - b["wall_s"],
                    }
                )
        differences = [p["wall_difference_F_minus_other"] for p in pairs]
        result["F_minus_" + other] = {
            "joint_success_pairs": len(pairs),
            "pairs": pairs,
            "mean_wall_difference_s": mean(differences) if differences else None,
            "median_wall_difference_s": median(differences) if differences else None,
            "F_faster_pairs": sum(d < 0 for d in differences),
            "interpretation": "Descriptive among starts both arms solved; conditions on success and is not an unconditional causal efficiency estimate.",
        }
    return result
