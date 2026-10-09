"""Retry accounting cannot inflate SR by mixing episodes and cases."""

import copy

import pytest

from astra_reversal.complex_manipulation.xiaomi_selector_report import aggregate


def trial(case, arm, attempt, success):
    return dict(case=case, arm=arm, attempt=attempt, success=success, steps=16,
                policy_queries=1, wall_seconds=4, policy_seconds=1,
                teacher_seconds=2 if arm == 'astra' else 0,
                teacher_calls=1 if arm == 'astra' else 0,
                tokens=100 if arm == 'astra' else 0)


def cohort():
    return [trial('a', 'native', 1, True), trial('b', 'native', 1, False),
            trial('b', 'native', 2, True), trial('a', 'astra', 1, False),
            trial('a', 'astra', 2, True), trial('b', 'astra', 1, False),
            trial('b', 'astra', 2, False)]


def test_first_attempt_within_two_and_episode_denominators_are_distinct():
    native, astra = aggregate(cohort(), [{'id': 'a'}, {'id': 'b'}])
    assert (native['first_successes'], native['within_two_successes'], native['cases']) == (1, 2, 2)
    assert (native['successes'], native['episodes']) == (2, 3)
    assert (astra['first_successes'], astra['within_two_successes'], astra['cases']) == (0, 1, 2)
    assert (astra['successes'], astra['episodes'], astra['tokens']) == (1, 4, 400)


@pytest.mark.parametrize('change', ['missing_retry', 'extra_after_success', 'duplicate', 'unexpected_case'])
def test_no_silent_censoring_or_extra_attempts(change):
    rows = copy.deepcopy(cohort())
    if change == 'missing_retry':
        rows.pop()
    elif change == 'extra_after_success':
        rows.append(trial('a', 'native', 2, True))
    elif change == 'duplicate':
        rows.append(rows[0])
    else:
        rows[0]['case'] = 'unregistered'
    with pytest.raises(ValueError):
        aggregate(rows, [{'id': 'a'}, {'id': 'b'}])
