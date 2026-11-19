import json

import pytest

from reproductions.leact.cluster import efe_pear_allocation_jobs as jobs
from reproductions.leact.cluster import efe_pear_native_reward_transition as transition
from reproductions.leact.cluster import launch_efe_pear_primary as primary
from reproductions.leact.cluster import monitor_efe_preflight as monitor


def setup_transition(tmp_path,monkeypatch):
    monkeypatch.setattr(transition,'DEST',tmp_path)
    old=tmp_path/'old';old.mkdir()
    new=tmp_path/'replacement';new.mkdir()
    spec={'model_revision':'warm496','rollout_nodes':4}
    (old/'spec.json').write_text(json.dumps(spec));(old/'curriculum.json').write_text('{}')
    (new/'spec.json').write_text(json.dumps({**spec,'run':'repaired-run'}))
    files={'control':{},'native-reward-finalization-status':{'phase':'replacement_validating','current_workflow':'retained'},
        'primary-driver-check-launch':{'name':'cpu-validation'}}
    for name,value in files.items():(tmp_path/(name+'.json')).write_text(json.dumps(value))
    state={'workflow':'retained','build':str(old),'run':'diagnostic-run','profile':15,'job_id':'old-job'}
    monkeypatch.setattr(primary,'prepare',lambda *a,**k:new)
    return state,new


def test_reward_repair_keeps_the_live_job_until_exact_configuration_validation(tmp_path,monkeypatch):
    state,_=setup_transition(tmp_path,monkeypatch)
    monkeypatch.setattr(transition,'command',lambda *a:{'status':'RUNNING'})
    monkeypatch.setattr(jobs,'submit',lambda *a,**k:pytest.fail('Unvalidated replacement was submitted'))
    monkeypatch.setattr(monitor,'remote',lambda *a,**k:pytest.fail('Validation is not ready'))
    assert transition.tick(state)is None
    assert state['job_id']=='old-job'


@pytest.mark.parametrize('updates',[0,1])
def test_reward_repair_preserves_the_allocation_and_any_completed_optimizer_progress(tmp_path,monkeypatch,updates):
    state,new=setup_transition(tmp_path,monkeypatch)
    monkeypatch.setattr(transition,'command',lambda args:{'status':'COMPLETED'}if args[3]=='cpu-validation'else{
        'status':'RUNNING','tasks':[{'status':'RUNNING'}]})
    monkeypatch.setattr(monitor,'remote',lambda *a,**k:{'check':{'passed':True},'completed_updates':updates})
    calls=[]
    monkeypatch.setattr(jobs,'submit',lambda s,b,**kw:calls.append((s['workflow'],b,kw))or{'job_id':kw['job_id']})
    result=transition.tick(state)
    progress=json.loads((tmp_path/'native-reward-finalization-status.json').read_text())
    if updates:
        assert result is None and calls==[]and state['run']=='diagnostic-run'
        assert progress['phase']=='checkpoint_handoff_required'
    else:
        assert result['workflow']=='retained'and result['build']==str(new)
        assert result['job_history'][0]['run']=='diagnostic-run'
        assert calls[0][2]['replace_running']and calls[0][2]['job_id'].startswith('native-repair-')
        assert progress['allocation_preserved']and not progress['replacement_active']


def test_retry_after_submission_acknowledgment_loss_does_not_restart_the_child_again(tmp_path,monkeypatch):
    import hashlib
    spec={'allocation_lifecycle':'persistent_cluster','rollout_nodes':4,'allocation_control_base':'/allocations'}
    raw=json.dumps(spec);(tmp_path/'spec.json').write_text(raw)
    (tmp_path/'source-manifest.json').write_text(json.dumps({'spec.json':hashlib.sha256(raw.encode()).hexdigest()}))
    monkeypatch.setattr(jobs,'command',lambda *a:{'status':'RUNNING','tasks':[{'status':'RUNNING'}]})
    monkeypatch.setattr(jobs,'remote',lambda *a,**k:{'already_submitted':True,'job_id':'repair-1'})
    monkeypatch.setattr(jobs.subprocess,'run',lambda *a,**k:pytest.fail('An acknowledged existing job was uploaded again'))
    assert jobs.submit({'workflow':'retained','build':str(tmp_path)},tmp_path,
        replace_running=True,job_id='repair-1')['already_submitted']


def test_standby_promotion_stages_the_current_recipe_instead_of_restarting_old_reward_code(tmp_path,monkeypatch):
    monkeypatch.setattr(primary,'DEST',tmp_path)
    old=tmp_path/'old';old.mkdir();new=tmp_path/'new';new.mkdir()
    spec={'profile':0,'rollout_nodes':6,'model_revision':'warm496','run':'old-run'}
    (old/'spec.json').write_text(json.dumps(spec));(new/'spec.json').write_text(json.dumps({**spec,'run':'new-run'}))
    backup={'workflow':'standby','standby':True,'build':str(old),'priority':'LOW'}
    (tmp_path/'backup-allocation.json').write_text(json.dumps(backup))
    monkeypatch.setattr(primary,'command',lambda *a:{'status':'RUNNING'})
    monkeypatch.setattr(primary,'prepare',lambda *a,**k:new)
    monkeypatch.setattr(jobs,'activate_embedded',lambda *a,**k:pytest.fail('Stale embedded code was activated'))
    calls=[]
    monkeypatch.setattr(jobs,'submit',lambda s,b,**k:calls.append((s['workflow'],b))or{'job_id':'fresh'})
    value=primary.promote_allocated_backup({})
    assert calls==[('standby',new)]and value['run']=='new-run'
    assert value['allocation_build']==str(old)and value['workflow']=='standby'
