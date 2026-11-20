import hashlib
import json
from pathlib import Path
import subprocess

import pytest
import yaml

from reproductions.leact.cluster import recover_efe_pear_graphics as recovery
from reproductions.leact.cluster import launch_efe_pear_primary as primary


def readiness():
    return {'passed':True,'model_file_relay_passed':True,'transport_roundtrip_passed':True,
        'relay_sha256':hashlib.sha256((recovery.BASE/'cluster/efe_pear_file_relay.py').read_bytes()).hexdigest(),
        'transport_sha256':'transport','assets_root':'/mnt/lfs/verified-assets',
        'assets':{'passed':True,'release_manifest_sha256':'release'},
        'sources':{'passed':True,'files':{'payload.tar':{'sha256':'payload'}}},
        'executed_control_steps':1,'cameras':[{'std':1.}],'native_termination_terms':{'success':False},
        'rtx_migration_smokes':{name:{'passed':True}for name in ['robodojo','robolab','yamlab']}}


@pytest.mark.parametrize('field,value',[('relay_sha256','old'),('executed_control_steps',0),
    ('model_file_relay_passed',False),('sources',{'passed':False}),('rtx_migration_smokes',{})])
def test_graphics_recovery_requires_the_current_complete_native_check(field,value):
    ready=readiness();model={'adapter_digests':{'efe_pear_object_transport.py':'transport'}}
    assert recovery.relay_ready(ready,model)
    ready[field]=value;assert not recovery.relay_ready(ready,model)


def test_recovery_waits_without_submitting_or_canceling_the_primary(tmp_path,monkeypatch):
    monkeypatch.setattr(recovery,'DEST',tmp_path)
    (tmp_path/'control.json').write_text(json.dumps({'graphics_recovery_target_pool':'groot-l40s-01',
        'paused':False,'enabled_training_arms':['warm']}))
    build=tmp_path/'model';build.mkdir()
    (build/'spec.json').write_text(json.dumps({'adapter_digests':{'efe_pear_object_transport.py':'transport'}}))
    monkeypatch.setattr(recovery,'submit_reconciled',lambda *a,**k:pytest.fail('Unqualified worker submitted'))
    monkeypatch.setattr(subprocess,'run',lambda *a,**k:pytest.fail('Primary allocation touched'))
    state={'workflow':'current','run_id':'warm','build':str(build),'scheduler_status':'RUNNING'}
    assert recovery.tick(state)['phase']=='awaiting_native_file_relay_validation'
    state['run_id']='paused-base';assert recovery.tick(state)is None


def test_all_24_graphics_gpus_move_while_the_model_and_efe_node_stay_in_place(tmp_path,monkeypatch):
    monkeypatch.setattr(recovery,'DEST',tmp_path);root='/mnt/amlfs-07/owned/current'
    model=tmp_path/'model';model.mkdir()
    (model/'spec.json').write_text(json.dumps({'run':root,'checkpoint_archive':'s3://private/checkpoint',
        'adapter_digests':{'efe_pear_object_transport.py':'transport'}}))
    queue={}
    for i in range(16):
        build=tmp_path/f'original-{i}';build.mkdir()
        spec={'output':root+'/native','transport_root':root+'/transport','finished':root+'/finished.json'}
        if i<8:
            spec.update(kind='rtx',suite=['robodojo','robolab','yamlab'][i%3],
                payload='/old/payload.tar',payload_sha256='payload');gpu=1
        else:
            spec.update(source_archive_path='/old/payload.tar',source_archive_sha256='payload',
                generated_reset_tasks={f'task-{2*i}':{},f'task-{2*i+1}':{}},
                assets_release={'release_manifest_sha256':'release'});gpu=2
        (build/'spec.json').write_text(json.dumps(spec));(build/'native-membership.json').write_text('{}')
        (build/'workflow.yaml').write_text(yaml.safe_dump({'workflow':{
            'resources':{'default':{'gpu':gpu,'cpu':8,'platform':'ovx-l40'}},
            'tasks':[{'image':'pinned-native','command':['python3']}]}}))
        (build/'source-manifest.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in build.iterdir()}))
        queue[str(i)]={'build':str(build),'workflow':f'owned-{i}','owner_model_workflow':'current','pool':'groot-l40-03'}
    queue['efe']={'owner_model_workflow':'current','pool':'groot-h100-01'}
    (tmp_path/'l40s01-cross-pool-probe.json').write_text(json.dumps({'jobs':{'probe':{'build':str(tmp_path/'original-0')}}}))
    calls=[]
    def worker(spec,files,entry,image,python,**resources):
        calls.append((spec,resources));return tmp_path/f'replacement-{len(calls)}'
    monkeypatch.setattr(primary,'worker_build',worker)
    plan=recovery.prepare({'workflow':'current','build':str(model)},readiness(),queue)
    assert plan['graphics_gpus']==24 and len(plan['builds'])==18
    assert len(plan['originals'])==16 and 'efe'not in plan['originals']
    relays=[s for s,r in calls if r['gpu']==0]
    assert next(s for s in relays if s['mode']=='model')['root']==root+'/transport'
    assert all(r['platform']=='ovx-l40s'for s,r in calls if r['gpu'])


def test_connecting_the_new_pool_does_not_cancel_an_existing_live_simulator(tmp_path,monkeypatch):
    monkeypatch.setattr(recovery,'DEST',tmp_path)
    (tmp_path/'control.json').write_text(json.dumps({'graphics_recovery_target_pool':'groot-l40s-01',
        'paused':False,'enabled_training_arms':['warm']}))
    model=tmp_path/'model';model.mkdir()
    (model/'spec.json').write_text(json.dumps({'adapter_digests':{'efe_pear_object_transport.py':'transport'}}))
    (tmp_path/'l40s01-storage-ready.json').write_text(json.dumps(readiness()))
    (tmp_path/'primary-native-queue.json').write_text(json.dumps({'live':{
        'workflow':'leact-rp-jhh-pear-worker-live-1','scheduler_status':'RUNNING'}}))
    monkeypatch.setattr(recovery,'prepare',lambda *a:{'wire':'new','workflow':'current','builds':[],'originals':['live']})
    monkeypatch.setattr(subprocess,'run',lambda *a,**k:pytest.fail('Existing simulator was canceled'))
    state={'workflow':'current','run_id':'warm','build':str(model),'scheduler_status':'RUNNING'}
    assert recovery.tick(state)['existing_graphics_retained']is True
    item=json.loads((tmp_path/'primary-native-queue.json').read_text())['live']
    assert item['scheduler_status']=='RUNNING'and not item.get('cancel_requested')and not item.get('superseded')
