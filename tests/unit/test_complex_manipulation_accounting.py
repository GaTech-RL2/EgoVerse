"""Prevent reset-free segments and incomplete probes from inflating SR."""

import numpy as np
import pytest

from astra_reversal.complex_manipulation.accounting import (
    EpisodeCounts,
    checked_chunk,
    summarize_episodes,
)


def test_replanning_and_candidate_search_do_not_add_physical_trials():
    counts = EpisodeCounts("task:seed0:native:attempt0", 12, 5)
    counts.generated_chunks = 3
    counts.generated_candidates = 15
    counts.execute(5)
    counts.execute(5, assisted=True)
    counts.execute(2)
    row = counts.result(success=True, stop_reason="success")
    assert (row["reset_episodes_started"], row["reset_free_segments"]) == (1, 3)
    assert (row["assisted_segments"], row["executed_actions"]) == (1, 12)
    assert row["physical_candidate_retries"] == 0


def test_interrupted_and_smoke_episodes_remain_outside_completed_sr():
    full = EpisodeCounts("task:seed0:native:attempt0", 10, 5)
    full.execute(5)
    partial = EpisodeCounts("task:seed1:native:attempt0", 10, 5)
    partial.execute(5)
    rows = [
        full.result(success=True, stop_reason="success"),
        partial.result(success=False, stop_reason="smoke_limit"),
    ]
    summary = summarize_episodes(rows)
    assert summary["reset_episodes_started"] == 2
    assert summary["completed_episodes"] == 1
    assert summary["incomplete_episodes"] == 1
    assert summary["completed_episode_sr"] == 1
    assert summary["intended_batch_sr"] is None
    with pytest.raises(ValueError, match="shorter smoke"):
        partial.result(success=False, stop_reason="horizon")


def test_failure_counts_and_repeated_episode_ids_are_not_hidden():
    failed = EpisodeCounts("task:seed0:native:attempt0", 5, 5)
    failed.execute(5)
    row = failed.result(success=False, stop_reason="horizon")
    assert summarize_episodes([row])["intended_batch_sr"] == 0
    with pytest.raises(ValueError, match="distinct physical"):
        summarize_episodes([row, row])


def test_action_contract_rejects_wrong_embodiment_and_nonfinite_commands():
    with pytest.raises(ValueError, match="expected"):
        checked_chunk(np.zeros((50, 7)), horizon=50, action_dim=12)
    with pytest.raises(ValueError, match="finite"):
        checked_chunk(np.full((50, 12), np.nan), horizon=50, action_dim=12)
    assert checked_chunk(np.zeros((50, 12)), horizon=50, action_dim=12).shape == (
        50,
        12,
    )
