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
