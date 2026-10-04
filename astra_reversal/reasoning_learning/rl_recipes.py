"""Explicit development tuning choices; every run records its actual settings."""

import copy

RECIPES = {
    "standard": {
        "dsrl": {"batch_size": 64, "updates": 200},
        "ppo": {"epochs": 1, "optimizer_batch": 8, "collection_rollouts_per_update": 1},
    },
    "more_reuse": {
        "dsrl": {"batch_size": 64, "updates": 600},
        "ppo": {"epochs": 4, "optimizer_batch": 8, "collection_rollouts_per_update": 2},
    },
}


def settings(recipe, method, *, collection_rollouts, evaluation_schedule):
    if recipe not in RECIPES or method not in RECIPES[recipe]:
        raise ValueError("Unknown recorded RL tuning recipe")
    result = copy.deepcopy(RECIPES[recipe][method])
    period = result.get("collection_rollouts_per_update", 1)
    if collection_rollouts % period or any(i % period for i in evaluation_schedule):
        raise ValueError(
            "On-policy update batches must finish at every evaluation boundary"
        )
    return result


def collect_on_policy(pending, transitions, version):
    """No policy update may occur inside a PPO collection batch."""
    if not transitions or any(
        row["policy_version"] != version for row in [*pending, *transitions]
    ):
        raise ValueError(
            "PPO collection batch mixes policy versions or contains no real data"
        )
    if not transitions[-1]["terminal"]:
        raise ValueError("Each collected episode must end before batching PPO updates")
    return [*pending, *transitions]
