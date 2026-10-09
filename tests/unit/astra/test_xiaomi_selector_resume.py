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


def test_nested_continuation_keeps_old_interruptions_and_isolates_parent_metadata():
    protocol = dict(cases=[{'id':'a'}],continuation=dict(parent_workflow='second',
                    interrupted_trial=dict(case='a',arm='astra',attempt=1)))
    paths = ('interrupted/first/a/astra1/provider.jsonl',
             'lineage/first/continuation/parent/protocol.json')
    for path in paths:
        assert str(retained_path(path,protocol)) == path
    assert str(retained_path('continuation/parent/protocol.json',protocol)) == (
        'lineage/second/continuation/parent/protocol.json')


def test_second_continuation_preserves_episode_provenance_and_checks_evidence(tmp_path):
    import copy
    import json
    from astra_reversal.complex_manipulation.xiaomi_selector_resume import load_parent
    from astra_reversal.complex_manipulation.worker import sha256
    evaluation = tmp_path/'evaluation'
    folder = evaluation/'a/native1'
    folder.mkdir(parents=True)
    saved = row('a','native',1,False)
    (folder/'episode.json').write_text(json.dumps(saved))
    summary_row = dict(saved,origin_workflow='first',origin_source_revision='original')
    parent = tmp_path/'continuation/parent'
    (parent/'evaluation').mkdir(parents=True)
    old_protocol = dict(cases=[{'id':'a'}],continuation={'parent_workflow':'first'})
    (parent/'protocol.json').write_text(json.dumps(old_protocol))
    (parent/'evaluation/summary.json').write_text(json.dumps(dict(
        episodes=[summary_row],resets=[{'case':'a','origin_workflow':'first'}])))
    protocol = dict(cases=[{'id':'a'}],continuation=dict(parent_workflow='second',
        parent_source_revision='new',parent_protocol_sha256=sha256(parent/'protocol.json'),
        retained_completed_trials=[['a','native',1]]))
    rows, resets = load_parent(evaluation,protocol)
    assert rows == [summary_row]
    assert resets[0]['origin_workflow'] == 'first'
    corrupt = copy.deepcopy(saved)
    corrupt['success'] = True
    (folder/'episode.json').write_text(json.dumps(corrupt))
    with pytest.raises(ValueError,match='episode and parent summary'):
        load_parent(evaluation,protocol)
