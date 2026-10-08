"""Statistical gates must retain failed trials and cannot manufacture confidence."""

import copy

import pytest

from astra_reversal.hardware_interface.analysis import analyze
from astra_reversal.hardware_interface.common import digest
from astra_reversal.hardware_interface.protocol import design
from astra_reversal.hardware_interface.provider import Meter
from astra_reversal.hardware_interface.reporting import power_worksheet


def fixture():
    manifest = design("docker.io/library/python@sha256:" + "a" * 64)
    rows = []
    for task in manifest["task_ids"]:
        for index in manifest["pilot_indices"]:
            for condition in manifest["conditions"]:
                rows.append(
                    {
                        "split": "pilot",
                        "task_id": task,
                        "init_state_index": index,
                        "replicate": 0,
                        "env_seed": 137,
                        "init_state_hash": digest([task, index]),
                        "condition": condition,
                        "success": index < 2,
                        "terminal_reason": "SUCCESS" if index < 2 else "TIMEOUT_STEPS",
                        "actor_started": True,
                        "independent_evaluation_passed": True,
                        "wall_s": 300,
                        "sim_steps": 1000,
                        "known_workflow_tokens": 4500,
                        "unknown_usage_records": 0,
                        "estimated_cost_usd": None,
                        "censored_wall": False,
                        "invalid_actions": 0,
                        "safety_attempts": 0,
                        "applied_safety_violations": 0,
                    }
                )
    return manifest, rows


def test_unknown_usage_breakdowns_are_null_and_do_not_double_count_reasoning():
    meter = Meter(1000)
    meter.record(
        "actor",
        {
            "input_tokens": 100,
            "output_tokens": 20,
            "output_tokens_details": {"reasoning_tokens": 12},
        },
    )
    assert meter.total == 120
    assert meter.usage["actor"]["cached_input_tokens"] is None
    assert meter.usage["actor"]["image_tokens"] is None
    assert meter.usage["actor"]["reasoning_tokens"] == 12
    meter.record("actor", {"input_tokens": 25, "output_tokens": 10})
    assert meter.total == 155 and meter.usage["actor"]["reasoning_tokens"] is None


def test_power_requires_complete_audited_pilot_and_accounts_for_discordance():
    manifest, rows = fixture()
    low = power_worksheet(rows, manifest)
    assert low["discordant_pairs"] == 0
    assert low["recommended_initial_states_per_task"] >= 10
    assert low["confirmation_frozen"] is False
    with pytest.raises(ValueError, match="complete_paired"):
        power_worksheet(rows[:-1], manifest)
    bad = copy.deepcopy(rows)
    bad[0]["independent_evaluation_passed"] = False
    with pytest.raises(ValueError, match="audited"):
        power_worksheet(bad, manifest)
    for row in rows:
        row["success"] = row["condition"] == "F"
    high = power_worksheet(rows, manifest)
    assert high["discordant_pairs"] == 50
    assert (
        high["recommended_initial_states_per_task"]
        > low["recommended_initial_states_per_task"]
    )


def test_all_started_analysis_keeps_timeouts_and_unknown_cost():
    manifest, rows = fixture()
    result = analyze(rows, manifest, split="pilot")
    assert result["arms"]["F"]["n"] == 50
    assert result["arms"]["F"]["task_macro_success"] == pytest.approx(0.4)
    assert result["arms"]["F"]["cost"]["n_unknown"] == 50
    assert result["arms"]["F"]["failure_codes"]["TIMEOUT_STEPS"] == 30
    assert result["comparisons"]["F_minus_B0"]["risk_difference"] == 0
    assert result["comparisons"]["F_minus_B0"]["decision"] == "inconclusive"


def test_unmatched_or_duplicate_trials_cannot_enter_paired_analysis():
    manifest, rows = fixture()
    with pytest.raises(ValueError, match="duplicate"):
        analyze(rows + [rows[0]], manifest, split="pilot")
    rows[0]["init_state_hash"] = digest("different")
    with pytest.raises(ValueError, match="unmatched"):
        analyze(rows, manifest, split="pilot")
