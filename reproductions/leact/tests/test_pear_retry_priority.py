import hashlib
import json

from reproductions.leact.cluster import monitor_efe_pear_full_qualification as monitor


def test_opportunistic_lanes_stay_low_when_the_pool_gains_high_priority_quota(tmp_path,monkeypatch):
    build=tmp_path/'worker';build.mkdir()
    (build/'workflow.yaml').write_text('workflow:\n  name: original\n')
    (build/'source-manifest.json').write_text(json.dumps({'workflow.yaml':hashlib.sha256((build/'workflow.yaml').read_bytes()).hexdigest()}))
    (tmp_path/'control.json').write_text(json.dumps({'l40_priority_by_pool':{'groot-l40-03':'HIGH'}}))
    monkeypatch.setattr(monitor,'DEST',tmp_path);seen=[]
    def submit(build,**kwargs):seen.append(kwargs);return {'name':'replacement'}
    monkeypatch.setattr(monitor,'submit_reconciled',submit)
    item={'build':str(build),'workflow':'old','pool':'groot-l40-03','owner_model_workflow':'model',
        'allocation_priority':'LOW'}
    monitor.retry(item)
    assert seen[0]['priority']=='LOW'and item['priority']=='LOW'


def test_fresh_scheduler_retry_cannot_inherit_an_older_allocations_keeper_root(tmp_path,monkeypatch):
    build=tmp_path/'worker';build.mkdir()
    (build/'workflow.yaml').write_text('workflow:\n  name: original\n')
    (build/'spec.json').write_text(json.dumps({'allocation_lifecycle':'persistent_cluster',
        'allocation_control_base':'/current-run/allocations'}))
    (build/'source-manifest.json').write_text(json.dumps({n:hashlib.sha256((build/n).read_bytes()).hexdigest()
        for n in ['workflow.yaml','spec.json']}))
    monkeypatch.setattr(monitor,'DEST',tmp_path)
    monkeypatch.setattr(monitor,'submit_reconciled',lambda *a,**k:{'name':'new-allocation'})
    item={'build':str(build),'allocation_build':'/old-run/frozen-bundle','workflow':'old-allocation',
        'pool':'groot-l40-03','owner_model_workflow':'model','job_id':'old-child'}
    monitor.retry(item)
    assert item['workflow']=='new-allocation'
    assert item['allocation_build']==item['build']and item['build']!=str(build)
    assert 'job_id'not in item and item['history'][0]['workflow']=='old-allocation'
