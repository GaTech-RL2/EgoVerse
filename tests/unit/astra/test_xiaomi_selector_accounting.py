"""Retry accounting cannot inflate SR by mixing episodes and cases."""

import copy

import pytest

from astra_reversal.complex_manipulation.xiaomi_selector_report import aggregate, reconcile_interrupted_jobs


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


def interrupted_receipts():
    usage = dict(input_tokens=100,output_tokens=10,reasoning_tokens=3,total_tokens=110)
    records = [dict(request_fingerprint='delivered',accepted=True,token_usage=usage,latency_seconds=4),
               dict(request_fingerprint='undelivered',accepted=False,token_usage=None,latency_seconds=300)]
    jobs = [dict(request_fingerprint=r['request_fingerprint'],delivered_and_accepted_by_worker=r['accepted'],
                 token_usage=usage.copy()) for r in records]
    return records, jobs


def test_undelivered_completed_job_is_charged_once_without_changing_acceptance():
    records, jobs = interrupted_receipts()
    value = reconcile_interrupted_jobs(records,jobs)
    assert value['completed_local_jobs'] == 2
    assert (value['delivered_proposals'],value['undelivered_completed_jobs']) == (1,1)
    assert value['token_usage']['total_tokens'] == 220
    assert value['token_usage']['reasoning_tokens'] == 6
    assert value['worker_wait_seconds'] == 304
    assert records[-1]['accepted'] is False and records[-1]['token_usage'] is None


@pytest.mark.parametrize('change',['missing','duplicate','usage_mismatch','acceptance_rewrite','unknown_usage'])
def test_interruption_reconciliation_requires_exact_receipts(change):
    records, jobs = interrupted_receipts()
    if change == 'missing':
        jobs.pop()
    elif change == 'duplicate':
        jobs[-1]['request_fingerprint'] = jobs[0]['request_fingerprint']
    elif change == 'usage_mismatch':
        jobs[0]['token_usage']['total_tokens'] += 1
    elif change == 'acceptance_rewrite':
        jobs[-1]['delivered_and_accepted_by_worker'] = True
    else:
        jobs[-1]['token_usage']['total_tokens'] = None
    with pytest.raises(ValueError):
        reconcile_interrupted_jobs(records,jobs)
