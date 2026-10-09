"""Outages must not erase failures, create extra success trials, or leak feedback."""

from pathlib import PurePosixPath

import pytest

from astra_reversal.complex_manipulation.xiaomi_selector_resume import pending_trials, retained_path


def row(case, arm, attempt, success):
    return dict(case=case, arm=arm, attempt=attempt, success=success, completed=True)


def test_completed_successes_and_failures_are_kept_when_replacing_interruption():
    rows = [row('a','native',1,False),row('a','astra',1,False),
            row('a','native',2,True),row('a','astra',2,True),row('b','native',1,False)]
    assert pending_trials([{'id':'a'},{'id':'b'},{'id':'c'}],rows) == [
        ('b','astra',1),('b','native',2),('b','astra',2),
        ('c','native',1),('c','astra',1),('c','native',2),('c','astra',2)]


@pytest.mark.parametrize('rows', [
    [row('a','native',2,True)],
    [row('a','native',1,True),row('a','native',2,False)],
    [row('a','native',1,False),row('a','native',1,True)],
    [dict(row('a','astra',1,False),completed=False)],
    [row('outside','native',1,False)],
])
def test_invalid_partial_cohorts_fail_closed(rows):
    with pytest.raises(ValueError):
        pending_trials([{'id':'a'}],rows)


def test_interrupted_feedback_is_kept_outside_replacement_episode():
    protocol = dict(cases=[{'id':'a'},{'id':'b'}],continuation=dict(parent_workflow='parent',
                    interrupted_trial=dict(case='b',arm='astra',attempt=1)))
    assert retained_path('evaluation/b/astra1/guidance/proposal_00.json',protocol) == PurePosixPath(
        'interrupted/parent/b/astra1/guidance/proposal_00.json')
    assert str(retained_path('evaluation/a/native1/episode.json',protocol)) == 'evaluation/a/native1/episode.json'
    assert str(retained_path('evaluation/b/anchor.json',protocol)) == 'evaluation/b/anchor.json'
    assert str(retained_path('evaluation/summary.json',protocol)) == 'continuation/parent/evaluation/summary.json'
    assert str(retained_path('history/a/rollout.mp4',protocol)) == 'history/a/rollout.mp4'
    with pytest.raises(ValueError):
        retained_path('../outside',protocol)
