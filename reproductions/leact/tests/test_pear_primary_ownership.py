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


def test_gpu_device_plugin_rejection_is_retried_but_cuda_oom_is_not():
    from reproductions.leact.cluster.efe_pear_submission import infrastructure_failure
    assert infrastructure_failure({'status':'FAILED','failure_message':
        'Pod was rejected: Allocate failed due to requested number of devices unavailable for nvidia.com/gpu'})
    assert not infrastructure_failure({'status':'FAILED','failure_message':'CUDA out of memory while allocating model tensors'})


def test_verified_source_repair_still_waits_for_all_prelaunch_gates(tmp_path,monkeypatch):
    monkeypatch.setattr(primary,'DEST',tmp_path)
    (tmp_path/'primary-launch.json').write_text(json.dumps({'workflow':'terminated-attempt',
        'terminal_confirmed':True,'source_repair_ready':True,'profile':0}))
    attempted=[]
    def launch(profile):
        attempted.append(profile)
        raise primary.PendingAdmission('Exact primary command remains pending')
    monkeypatch.setattr(primary,'launch',launch)
    monkeypatch.setattr(primary,'command',lambda *a,**k:pytest.fail('Old terminated job was queried instead of checking new admission'))
    value=primary.tick()
    assert attempted==[0]and value['primary_rl_updates']==0
    assert value['phase']=='awaiting_primary_qualification'


def test_trainer_cpu_reservations_fit_queue_controller_driver_and_all_gpu_ranks():
    with pytest.raises(ValueError,match='cannot fit'):
        primary.validate_trainer_cpu_capacity(56)
    assert primary.validate_trainer_cpu_capacity(64)==58


def test_primary_model_rejects_a_pool_without_the_qualified_shared_volume():
    with pytest.raises(ValueError,match='qualified AMLFS07 H100 pool'):
        primary.launch(pool='groot-l40s-01')
