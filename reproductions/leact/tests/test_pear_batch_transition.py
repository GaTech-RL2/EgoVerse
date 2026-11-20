import json

import pytest

from reproductions.leact.cluster import efe_pear_batch_transition as transition
from reproductions.leact.cluster import efe_pear_additional_capacity as additional
from reproductions.leact.cluster import efe_pear_allocation_jobs as jobs
from reproductions.leact.cluster import launch_efe_pear_primary as primary
from reproductions.leact.cluster import monitor_efe_preflight as monitor


@pytest.mark.parametrize('updates',[0,1])
def test_batch_transition_requires_exact_validation_and_preserves_completed_updates(tmp_path,monkeypatch,updates):
    monkeypatch.setattr(transition,'DEST',tmp_path)
    old=tmp_path/'old';old.mkdir();new=tmp_path/'new';new.mkdir()
    spec={'model_revision':'warm496','rollout_nodes':4,'run':'old-run'}
    (old/'spec.json').write_text(json.dumps(spec));(new/'spec.json').write_text(json.dumps({**spec,'run':'new-run'}))
    (new/'curriculum.json').write_text(json.dumps({'retained_groups_per_update':32}))
    for f,v in {'control':{},'startup-batch-transition':{'phase':'validating','current_workflow':'retained','target_groups':32},
        'primary-driver-check-launch':{'name':'cpu'}}.items():(tmp_path/(f+'.json')).write_text(json.dumps(v))
    monkeypatch.setattr(transition,'command',lambda args:{'status':'COMPLETED'}if args[3]=='cpu'else{
        'status':'RUNNING','tasks':[{'status':'RUNNING'}]})
    monkeypatch.setattr(monitor,'remote',lambda *a,**k:{'check':{'passed':True,'retained_groups_per_update':32},'updates':updates})
    monkeypatch.setattr(primary,'prepare',lambda *a,**k:new)
    calls=[]
    monkeypatch.setattr(jobs,'submit',lambda s,b,**k:calls.append((s['workflow'],b,k))or{'job_id':k['job_id']})
    state={'workflow':'retained','build':str(old),'run':'old-run','job_id':'old-job','profile':15}
    value=transition.tick(state)
    if updates:
        assert value is None and calls==[]and state['job_id']=='old-job'
        assert json.loads((tmp_path/'startup-batch-transition.json').read_text())['phase']=='checkpoint_handoff_required'
    else:
        assert value['workflow']=='retained'and value['run']=='new-run'
        assert calls[0][2]['replace_running']and calls[0][2]['job_id'].startswith('startup32-')
        assert value['job_history'][0]['completed_updates']==0


def test_batch_transition_never_interrupts_the_live_job_while_validation_is_pending(tmp_path,monkeypatch):
    monkeypatch.setattr(transition,'DEST',tmp_path)
    for f,v in {'control':{},'startup-batch-transition':{'phase':'validating','current_workflow':'model'},
        'primary-driver-check-launch':{'name':'cpu'}}.items():(tmp_path/(f+'.json')).write_text(json.dumps(v))
    monkeypatch.setattr(transition,'command',lambda *a:{'status':'RUNNING'})
    monkeypatch.setattr(jobs,'submit',lambda *a,**k:pytest.fail('Unvalidated job replaced a live one'))
    assert transition.tick({'workflow':'model'})is None


def test_pending_extra_capacity_is_retargeted_later_without_another_scheduler_submission(tmp_path,monkeypatch):
    monkeypatch.setattr(additional,'DEST',tmp_path)
    plan={'model_workflow':'model','model_run':'old-run','jobs':[{'workflow':'queued-extra','build':'old-build'}]}
    (tmp_path/'additional-l40-03-20261119.json').write_text(json.dumps(plan))
    calls=[]
    monkeypatch.setattr(additional,'command',lambda args:calls.append(args)or{'status':'PENDING'})
    monkeypatch.setattr(jobs,'submit',lambda *a,**k:pytest.fail('Pending node was entered prematurely'))
    additional.tick({'workflow':'model','run':'new-run'})
    assert len(calls)==1 and calls[0][2]=='query'
    assert json.loads((tmp_path/'additional-l40-03-20261119.json').read_text())==plan
