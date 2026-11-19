import json

import pytest

from reproductions.leact.cluster import efe_pear_allocation_jobs as jobs
from reproductions.leact.cluster import launch_efe_pear_primary as primary


def test_no_allocation_exec_during_the_group_startup_barrier(monkeypatch):
    monkeypatch.setattr(jobs,'command',lambda *a:{'status':'RUNNING','groups':[{'tasks':[
        {'name':'trainer','status':'RUNNING'},{'name':'rollout-0','status':'INITIALIZING'}]}]})
    monkeypatch.setattr(jobs,'remote',lambda *a,**k:pytest.fail('Premature allocation exec'))
    with pytest.raises(RuntimeError,match='startup barrier'):
        jobs.activate_embedded({'workflow':'allocated'})


def test_legacy_allocation_is_preserved_when_no_keeper_is_installed(tmp_path,monkeypatch):
    (tmp_path/'spec.json').write_text('{}')
    monkeypatch.setattr(jobs,'command',lambda *a:pytest.fail('Legacy cluster was mutated'))
    with pytest.raises(RuntimeError,match='must be preserved'):
        jobs.submit({'workflow':'legacy','build':str(tmp_path)},tmp_path)


def test_driver_retry_uses_the_same_allocation_and_leaves_optimizer_progress_for_recovery(monkeypatch):
    state={'workflow':'retained-cluster','build':'pinned-job','run':'run'}
    calls=[]
    monkeypatch.setattr(jobs,'submit',lambda s,b:calls.append((s['workflow'],b))or{'job_id':'retry-1'})
    failed={'nodes':[{'phase':'idle_failed'},{'phase':'idle_peer_failed'}],'completed_updates':0}
    result=primary.recover_job_inside_allocation(dict(state),failed)
    assert result['workflow']=='retained-cluster'and result['job_id']=='retry-1'
    assert calls==[('retained-cluster','pinned-job')]
    calls.clear()
    result=primary.recover_job_inside_allocation(dict(state),{**failed,'completed_updates':1})
    assert result['phase']=='allocation_held_for_checkpoint_recovery'and calls==[]


def test_native_production_jobs_get_a_keeper_but_qualification_jobs_still_finish(tmp_path,monkeypatch):
    import yaml
    monkeypatch.setattr(primary,'BUILD_ROOT',tmp_path)
    source=tmp_path/'native_job.py';source.write_text('pass\n')
    resources={'gpu':1,'cpu':8,'memory':'32Gi','storage':'20Gi','platform':'ovx-l40'}
    build=primary.worker_build({'owner_model_workflow':'model','transport_root':'/private/run/transport'},
        {source.name:source},source.name,'pinned-native-image','python3',**resources)
    task=yaml.safe_load((build/'workflow.yaml').read_text())['workflow']['tasks'][0]
    assert task['environment']['PEAR_KEEPER_ENTRY']=='native_job.py'
    assert json.loads((build/'spec.json').read_text())['allocation_lifecycle']=='persistent_cluster'
    check=primary.worker_build({'qualification_only':True},{source.name:source},source.name,
        'pinned-native-image','python3',**resources)
    task=yaml.safe_load((check/'workflow.yaml').read_text())['workflow']['tasks'][0]
    assert 'PEAR_KEEPER_ENTRY'not in task['environment']
