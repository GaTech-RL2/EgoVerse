import hashlib
import json

import pytest
import yaml

from reproductions.leact.cluster import efe_pear_backup as backup


@pytest.mark.parametrize('status',['RUNNING','PENDING','INITIALIZING','FAILED_CANCELED'])
def test_backup_maintenance_preserves_live_requests_and_user_cancellations(tmp_path,monkeypatch,status):
    p=tmp_path/'backup.json';p.write_text(json.dumps({'workflow':'backup','standby':True}))
    monkeypatch.setattr(backup,'command',lambda *a:{'status':status,'start_time':None})
    monkeypatch.setattr(backup,'submit_reconciled',lambda *a,**k:pytest.fail('Existing backup was replaced'))
    assert backup.maintain(p)['workflow']=='backup'


def test_expired_backup_requeues_all_seven_nodes_at_low_priority(tmp_path,monkeypatch):
    build=tmp_path/'backup';build.mkdir()
    files={'spec.json':json.dumps({'rollout_nodes':6,'allocation_control_base':'/private/allocations'}),
        'workflow.yaml':yaml.safe_dump({'workflow':{'name':'backup','groups':[{'name':'cluster','tasks':[{'name':'trainer'}]}]}})}
    for name,data in files.items():(build/name).write_text(data)
    (build/'source-manifest.json').write_text(json.dumps({n:hashlib.sha256((build/n).read_bytes()).hexdigest()for n in files}))
    p=tmp_path/'backup.json';p.write_text(json.dumps({'workflow':'expired','build':str(build),
        'pool':'groot-h100-02','standby':True}))
    monkeypatch.setattr(backup,'command',lambda *a:{'status':'FAILED_QUEUE_TIMEOUT','start_time':None})
    calls=[]
    def submit(path,**kwargs):calls.append(kwargs);return{'name':'new-backup'}
    monkeypatch.setattr(backup,'submit_reconciled',submit)
    value=backup.maintain(p)
    assert calls==[{'priority':'LOW','pool':'groot-h100-02','nodes':7}]
    assert value['gpu']==56 and value['standby']and value['workflow']=='new-backup'


def test_scheduler_control_failure_requeues_an_idle_standby_without_touching_the_primary(tmp_path,monkeypatch):
    build=tmp_path/'backup';build.mkdir()
    files={'spec.json':json.dumps({'rollout_nodes':6,'allocation_control_base':'/private/allocations','standby':True}),
        'workflow.yaml':yaml.safe_dump({'workflow':{'name':'backup'}})}
    for name,value in files.items():(build/name).write_text(value)
    (build/'source-manifest.json').write_text(json.dumps({n:hashlib.sha256((build/n).read_bytes()).hexdigest()for n in files}))
    p=tmp_path/'backup.json';p.write_text(json.dumps({'workflow':'scheduler-ended','build':str(build),
        'pool':'groot-h100-02','standby':True}))
    monkeypatch.setattr(backup,'command',lambda *a:{'status':'FAILED','start_time':'started',
        'groups':[{'tasks':[{'name':'trainer','exit_code':2001}]}]})
    calls=[]
    monkeypatch.setattr(backup,'submit_reconciled',lambda b,**k:calls.append(k)or{'name':'fresh-standby'})
    value=backup.maintain(p)
    assert calls==[{'priority':'LOW','pool':'groot-h100-02','nodes':7}]
    assert value['workflow']=='fresh-standby'and value['standby']and value['gpu']==56
