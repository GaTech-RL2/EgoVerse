import json
from types import SimpleNamespace

import pytest

from reproductions.leact.cluster import launch_efe_pear_primary as primary


def test_model_retry_cancels_only_workers_owned_by_that_attempt(tmp_path,monkeypatch):
    monkeypatch.setattr(primary,'DEST',tmp_path)
    rows={
        'owned':{'workflow':'leact-rp-jhh-pear-worker-owned-1','owner_model_workflow':'model-a'},
        'other':{'workflow':'independent-study-1','owner_model_workflow':'model-b'},
        'done':{'workflow':'leact-rp-jhh-pear-worker-done-1','owner_model_workflow':'model-a','scheduler_status':'COMPLETED'}}
    (tmp_path/'primary-native-queue.json').write_text(json.dumps(rows))
    calls=[]
    def run(args,**kwargs):calls.append(args);return SimpleNamespace(returncode=0)
    monkeypatch.setattr(primary.subprocess,'run',run)
    primary.cancel_native_workers('model-a');primary.cancel_native_workers('model-a')
    assert len(calls)==1 and calls[0][3]=='leact-rp-jhh-pear-worker-owned-1'
    assert json.loads((tmp_path/'primary-native-queue.json').read_text())['owned']['cancel_requested']


def test_cancellation_rejects_an_unrelated_workflow_even_with_a_bad_owner_tag(tmp_path,monkeypatch):
    monkeypatch.setattr(primary,'DEST',tmp_path)
    (tmp_path/'primary-native-queue.json').write_text(json.dumps({
        'bad':{'workflow':'independent-study-1','owner_model_workflow':'model-a'}}))
    monkeypatch.setattr(primary.subprocess,'run',lambda *a,**k:pytest.fail('Unrelated workflow was touched'))
    with pytest.raises(ValueError,match='outside this primary'):
        primary.cancel_native_workers('model-a')
