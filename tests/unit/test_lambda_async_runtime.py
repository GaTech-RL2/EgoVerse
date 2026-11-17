import copy
import importlib.util
from pathlib import Path

import pytest

RUNTIME = (
    Path(__file__).parents[2]
    / "experiments/fold_clothes_moving_opscale/lambda_async/runtime.py"
)


def load_runtime():
    spec = importlib.util.spec_from_file_location("lambda_async_runtime", RUNTIME)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_evaluation_settings_reject_invalid_campaign():
    runtime = load_runtime()
    with pytest.raises(RuntimeError, match="W&B-safe"):
        runtime.evaluation_settings({"evaluation_campaign": "Not W&B safe"})


def test_evaluation_settings_accept_launch_tag_and_metrics_only():
    runtime = load_runtime()
    assert runtime.evaluation_settings(
        {
            "evaluation_campaign": "fold-moving-opscale-leakagefixed-reeval-20261117",
            "metrics_only": True,
        }
    ) == (
        "fold-moving-opscale-leakagefixed-reeval-20261117",
        True,
    )


def test_evaluation_settings_rejects_non_boolean_metrics_only():
    runtime = load_runtime()
    with pytest.raises(RuntimeError, match="metrics_only"):
        runtime.evaluation_settings({"metrics_only": "true"})


def test_evaluation_wandb_id_is_stable_across_cluster_paths():
    runtime = load_runtime()
    campaign = "fold-moving-opscale-leakagefixed-reeval-20261117"
    assert runtime.evaluation_wandb_id(campaign, "b24_op4") == (
        runtime.evaluation_wandb_id(campaign, "b24_op4")
    )
    assert runtime.evaluation_wandb_id(campaign, "b24_op4") != (
        runtime.evaluation_wandb_id(campaign, "b24_op8")
    )


def campaign_memberships(runtime):
    old_unseen = [
        f"old-{operator}-{episode}"
        for operator in range(20)
        for episode in range(20)
    ]
    corrected_unseen = old_unseen[:240] + [
        f"new-{operator}-{episode}"
        for operator in range(8)
        for episode in range(20)
    ]
    episodes = {
        episode: {"user_id": episode.rsplit("-", 1)[0]}
        for episode in old_unseen + corrected_unseen
    }
    seen_matched = [f"seen-matched-{index}" for index in range(40)]
    per_level = {
        operator_count: [
            f"seen-perlevel-{operator_count}-{index}"
            for index in range(operator_count * 3)
        ]
        for operator_count in runtime.OPERATORS
    }
    corrected = {}
    original = {}
    for cell in runtime.CELLS:
        operator_count = int(cell.split("op")[1])
        split = {
            "operators": [f"train-operator-{index}" for index in range(operator_count)],
            "train": [f"train-{cell}-{index}" for index in range(operator_count)],
            "seen_matched": seen_matched,
            "seen_perlevel": per_level[operator_count],
            "video_seen_matched": [],
            "video_seen_perlevel": [],
            "unseen": corrected_unseen,
        }
        corrected[cell] = copy.deepcopy(split)
        original[cell] = copy.deepcopy(split)
        original[cell]["unseen"] = old_unseen
    return corrected, original, episodes


def test_validate_split_cohorts_accepts_exact_campaign_memberships():
    runtime = load_runtime()
    corrected, original, episodes = campaign_memberships(runtime)
    runtime.validate_split_cohorts(corrected, original, episodes)


def test_validate_split_cohorts_rejects_cross_budget_perlevel_drift():
    runtime = load_runtime()
    corrected, original, episodes = campaign_memberships(runtime)
    corrected["b32_op4"]["seen_perlevel"][0] = "drifted-episode"
    original["b32_op4"]["seen_perlevel"][0] = "drifted-episode"
    with pytest.raises(RuntimeError, match="match across budgets"):
        runtime.validate_split_cohorts(corrected, original, episodes)


def test_validate_split_cohorts_rejects_wrong_unseen_replacement_count():
    runtime = load_runtime()
    corrected, original, episodes = campaign_memberships(runtime)
    corrected["b24_op4"]["unseen"][-20:] = original["b24_op4"]["unseen"][240:260]
    with pytest.raises(RuntimeError, match="retain 240 episodes and replace 160"):
        runtime.validate_split_cohorts(corrected, original, episodes)
