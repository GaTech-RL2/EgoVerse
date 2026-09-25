import json

import pytest

from astra_reversal.interpolation_catalog import (
    DONORS,
    ORACLES,
    catalog_metadata,
    demonstration_plan,
    donor_catalog,
    donor_for,
    oracle_for,
    paper_alpha,
)


def test_provider_catalog_contains_only_nine_standard_source_ids_and_prompts():
    rows = donor_catalog()
    assert [row["source_id"] for row in rows] == [
        "10",
        "13",
        "14",
        "17",
        "18",
        "32",
        "35",
        "36",
        "38",
    ]
    assert all(set(row) == {"source_id", "prompt"} for row in rows)
    assert all("ood" not in row.suite for row in DONORS)
    assert all(len(row.episode_indices) == 20 for row in DONORS)
    assert len({episode for row in DONORS for episode in row.episode_indices}) == 180
    assert sum(row.all_frame_count for row in DONORS) == 18947
    assert sum(row.release_frame_count for row in DONORS) == 18628
    rows[0]["prompt"] = "mutated caller copy"
    assert donor_for("10").prompt == "put the bowl on the plate"


def test_all_twenty_oracles_match_released_source_pair_ids():
    expected = {
        "libero_goal_ood": [
            ("13", "17"),
            ("14", "17"),
            ("13", "10"),
            ("13", "18"),
            ("14", "17"),
            ("14", "10"),
            ("14", "13"),
            ("13", "10"),
            ("13", "17"),
            ("13", "18"),
        ],
        "libero_spatial_ood": [
            ("13", "35"),
            ("13", "35"),
            ("14", "32"),
            ("14", "32"),
            ("35", "17"),
            ("35", "18"),
            ("36", "18"),
            ("36", "17"),
            ("38", "18"),
            ("38", "17"),
        ],
    }
    assert len(ORACLES) == 20
    for suite, pairs in expected.items():
        for task_id, pair in enumerate(pairs):
            row = oracle_for(suite, task_id)
            assert (row.source_a_id, row.source_b_id) == pair
            assert donor_for(row.source_a_id) and donor_for(row.source_b_id)
    assert (
        oracle_for("libero_goal_ood", 2).task_name == "put_the_bbq_source_on_the_plate"
    )


def test_schedule_declares_paper_release_difference_and_policy_call_units():
    wine = oracle_for("libero_goal_ood", 6)
    assert (wine.lambda_calls, wine.release_lambda) == (14, 12)
    assert paper_alpha(0, wine.lambda_calls) == 0
    assert paper_alpha(7, wine.lambda_calls) == 0.5
    assert paper_alpha(14, wine.lambda_calls) == 1
    assert paper_alpha(59, wine.lambda_calls) == 1
    details = wine.metadata()
    assert details["paper_form_port"]["first_full_second_donor_action_step"] == 70
    assert (
        details["released_implementation"]["first_full_second_donor_action_step"] == 65
    )
    assert oracle_for("libero_spatial_ood", 5).lambda_calls == 14
    assert oracle_for("libero_spatial_ood", 7).lambda_calls == 30
    assert all(
        row.lambda_calls == 24
        for row in ORACLES
        if (row.suite, row.task_id)
        not in {
            ("libero_goal_ood", 6),
            ("libero_spatial_ood", 5),
            ("libero_spatial_ood", 7),
        }
    )
    assert (
        "released_per_task"
        in oracle_for("libero_goal_ood", 0).metadata()["lambda_value_source"]
    )
    json.dumps(catalog_metadata(), allow_nan=False)


@pytest.mark.parametrize(
    "index,lam", [(-1, 24), (True, 24), (0.5, 24), (0, 0), (0, True)]
)
def test_invalid_schedule_inputs_fail(index, lam):
    with pytest.raises(ValueError):
        paper_alpha(index, lam)


def test_unknown_ids_and_modified_metadata_fail_closed(tmp_path):
    with pytest.raises(ValueError):
        donor_for("libero_goal_ood:0")
    with pytest.raises(ValueError):
        oracle_for("libero_goal_ood", True)
    (tmp_path / "info.json").write_text("{}\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        demonstration_plan(tmp_path)
