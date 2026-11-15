import math

import pytest

from reproductions.leact.core import (
    distribution, forward_delta, kl, mean_policy, parse_policy,
    qualitative_summary, select_candidates,
)


def test_baseline_is_probability_mean_before_log():
    expert = {"call": 0.5, "raise": 0.5}
    baseline = [{"call": 0.99, "raise": 0.01}, {"call": 0.01, "raise": 0.99}]
    candidate = {"call": 0.6, "raise": 0.4}
    # Averaging KL would incorrectly admit this trace. The mixture is already perfect.
    assert mean_policy(baseline) == expert
    assert forward_delta(expert, candidate, baseline) < 0
    assert sum(kl(expert, p) for p in baseline) / 2 - kl(expert, candidate) > 0


def test_delta_sign_and_strict_positive_admission():
    expert = {"call": 0.8, "raise": 0.2}
    baseline = [{"call": 0.5, "raise": 0.5}]
    assert forward_delta(expert, expert, baseline) > 0
    assert forward_delta(expert, baseline[0], baseline) == 0
    records = [{"candidate_id": i, "reasoning": "a trace", "score": score}
               for i, score in enumerate([0, -1, 0.3, 0.8, 0.5])]
    assert [c["candidate_id"] for c in select_candidates(records, 2, "leact")] == [3, 4]
    assert select_candidates(records[:2], 2, "leact") == []
    assert len(select_candidates(records[:2], 2, "exit")) == 2


def test_kl_zero_support_and_cap():
    p, q = {"a": 0.0, "b": 1.0}, {"a": 0.0, "b": 1.0}
    assert kl(p, q) == 0
    q = {"a": 1.0, "b": 0.0}
    assert kl(p, q) == 10
    assert math.isinf(kl(p, q, cap=None))


@pytest.mark.parametrize("values", [
    {"a": -0.1, "b": 1.1}, {"a": float("nan"), "b": 1},
    {"a": 0.5}, {"a": True, "b": 0}, {"a": 0.2, "b": 0.2},
])
def test_invalid_expert_probabilities_fail(values):
    with pytest.raises(ValueError):
        distribution(values, ["a", "b"])


def test_parser_ignores_thought_policy_and_handles_invalid_output():
    result = parse_policy('<think>\nPolicy: {"call":0,"raise":1}\n</think>\nAction: call\n'
                          'Policy: {"call":0.7,"raise":0.3}', ["call", "raise"])
    assert result.policy == {"call": 0.7, "raise": 0.3}
    assert result.fallback is None
    result = parse_policy('Action: raise\nPolicy: {"call":NaN,"raise":1}', ["call", "raise"])
    assert result.policy == {"call": 0, "raise": 1}
    assert result.fallback == "parsed_action"
    result = parse_policy('<think>\nAction: raise\nPolicy: {"call":0,"raise":1}', ["call", "raise"])
    assert result.fallback == "first_legal_action"
    assert result.action == "call"
    result = parse_policy('Action: raise\nPolicy: {"call":1,"call":0,"raise":1}', ["call", "raise"])
    assert result.fallback == "parsed_action"


def test_expert_redaction_has_no_numeric_probabilities():
    summary = qualitative_summary({"fold": 0.731, "call": 0.269, "raise": 0.0})
    assert not any(c.isdigit() for c in summary)
    assert "fold" in summary and "call" in summary and "raise" in summary


def test_truncated_and_empty_reasoning_never_train():
    candidates = [
        {"candidate_id": 0, "reasoning": "unfinished", "score": 1.0, "complete": False},
        {"candidate_id": 1, "reasoning": "", "score": 2.0},
    ]
    assert select_candidates(candidates, 2, "leact") == []
