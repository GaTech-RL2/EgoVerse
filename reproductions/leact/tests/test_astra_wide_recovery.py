import hashlib
import contextlib
import io
import json
import fcntl
from types import SimpleNamespace

import pytest

from reproductions.leact.cluster import launch_astra_wide_collection as launch
from reproductions.leact.cluster import monitor_astra_wide_collection as watch


def test_freeze_worker_is_stable_when_manifest_already_exists(tmp_path):
    (tmp_path/'worker.py').write_text('print(1)\n')
    launch.freeze_worker(tmp_path,['worker.py'])
    first=(tmp_path/'source-manifest.json').read_bytes()
    (tmp_path/'submission.json').write_text('{"scheduler":"receipt"}')
    launch.freeze_worker(tmp_path,['worker.py'])
    assert (tmp_path/'source-manifest.json').read_bytes()==first
    assert json.loads(first)=={'worker.py':hashlib.sha256(b'print(1)\n').hexdigest()}


@pytest.mark.parametrize('status',['RUNNING','PENDING','COMPLETED','FAILED_CANCELED','FAILED_USER_CANCELED'])
def test_recovery_never_replaces_live_or_user_canceled_allocation(status,monkeypatch):
    monkeypatch.setattr(watch,'refresh_build',lambda *_:pytest.fail('Must preserve this allocation'))
    assert watch.recover_allocation({'workflow':'own'}, {'status':status})is None


def test_child_recovery_requires_exact_recorded_teacher_owner(monkeypatch):
    monkeypatch.setattr(watch,'idle_verification',lambda *_:pytest.fail('A different child owns the allocation'))
    assert watch.recover_child({'job_id':'own'},
        {'node':{'phase':'idle_failed'},'desired_id':'different'})is None


def test_refresh_rejects_non_teacher_source_before_any_build(tmp_path):
    (tmp_path/'spec.json').write_text(json.dumps({'owner_model_workflow':'hero'}))
    with pytest.raises(ValueError,match='independently owned'):
        watch.refresh_build({'build':str(tmp_path),'workflow':'hero'})


def test_native_failure_feedback_stays_scoped_to_worker_tasks(tmp_path):
    import importlib.util
    import sys
    from pathlib import Path
    cluster=Path(watch.__file__).parent
    spec=importlib.util.spec_from_file_location('astra_wide_worker_under_test',cluster/'efe_astra_wide_worker.py')
    old=list(sys.path);sys.path.insert(0,str(cluster))
    try:module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    finally:sys.path[:]=old
    for uid,task,status in [('own','owned','opening'),('other','elsewhere','opening'),('done','owned','graded_success')]:
        result=tmp_path/'episodes'/uid/'infra-00/result.json';result.parent.mkdir(parents=True)
        result.write_text(json.dumps({'status':status,'case':{'task':task},'episode_id':uid}))
        (tmp_path/'transport/episodes'/uid).mkdir(parents=True)
    assert module.fail_owned_requests({'teacher_root':str(tmp_path),'teacher_task_names':['owned'],
        'transport_root':str(tmp_path/'transport')},'native bootstrap died')==1
    assert (tmp_path/'transport/episodes/own/infrastructure-error.json').exists()
    assert not (tmp_path/'transport/episodes/other/infrastructure-error.json').exists()
    assert not (tmp_path/'transport/episodes/done/infrastructure-error.json').exists()


def test_empty_failed_allocation_json_does_not_block_other_evidence(tmp_path,monkeypatch):
    from reproductions.leact.cluster import switch_astra_wide_profile as profile
    legacy=tmp_path/'legacy';current=tmp_path/'current';allocation=tmp_path/'allocations'
    (allocation/'failed/nodes').mkdir(parents=True);(allocation/'failed/nodes/native.json').write_text('')
    (allocation/'failed/desired.json').write_text('{"id":"failed-job"}')
    build=tmp_path/'build';build.mkdir();(build/'spec.json').write_text(json.dumps({'allocation_control_base':str(allocation)}))
    result=current/'episodes/case/infra-00/result.json';result.parent.mkdir(parents=True)
    result.write_text('{"status":"graded_success","case":{"suite":"procedural_reasoning"}}')
    monkeypatch.setattr(watch,'TEACHER_ROOT',str(legacy));monkeypatch.setattr(profile,'PROFILE_ROOT',str(current))
    def execute(host,script,task):
        stream=io.StringIO()
        with contextlib.redirect_stdout(stream):exec(script,{})
        return json.loads(stream.getvalue().split('EFE_CAPTURE ',1)[1])
    monkeypatch.setattr(watch,'remote',execute)
    captured=watch.snapshot('healthy-reader',{'failed':{'build':str(build)}})
    assert captured['read_errors'][0]['error_type']=='JSONDecodeError'
    assert captured['astra62_profile']['attempt_status']=={'graded_success':1}


def test_snapshot_host_failure_does_not_skip_failed_allocation_recovery(tmp_path,monkeypatch):
    states={name:{'workflow':name,'build':'unused'}for name in ['healthy','failed']}
    (tmp_path/'launches.json').write_text(json.dumps(states))
    monkeypatch.setattr(watch,'ROOT',tmp_path);monkeypatch.setattr(watch,'DEST',tmp_path)
    def run(args,**kwargs):
        q={'status':'RUNNING'if args[3]=='healthy'else'FAILED',
           'tasks':[{'status':'RUNNING'if args[3]=='healthy'else'FAILED'}]}
        return SimpleNamespace(returncode=0,stdout=json.dumps(q))
    monkeypatch.setattr(watch.subprocess,'run',run)
    monkeypatch.setattr(watch,'snapshot',lambda *_:(_ for _ in()).throw(RuntimeError('reader disconnected')))
    recovered=[]
    def recovery(state,scheduler):
        recovered.append(state['workflow'])
        return {'workflow':'replacement'}if state['workflow']=='failed'else None
    monkeypatch.setattr(watch,'recover_allocation',recovery)
    result=watch.tick()
    assert recovered==['healthy','failed']
    assert result['snapshot_errors']and result['phase']=='observed'
    assert result['recovery_actions']==[{'workflow':'replacement','recovery_of':'failed'}]
    assert json.loads((tmp_path/'watch-status.json').read_text())['phase']=='observed'


def test_explicitly_canceled_teacher_is_not_automatically_reallocated(monkeypatch):
    monkeypatch.setattr(watch,'refresh_build',lambda *_:pytest.fail('User cancellation must remain respected'))
    assert watch.recover_allocation({}, {'status':'FAILED','cancelled_by':'rpunamiya','exit_code':2143})is None


def test_yam_heartbeat_replacement_race_preserves_missing_file_semantics(tmp_path,monkeypatch):
    from pathlib import Path
    from reproductions.leact.cluster.efe_pear_yam_server import head_heartbeat_expired
    class Stamp:
        st_mtime=1000
    monkeypatch.setattr(Path,'stat',lambda self:Stamp())
    assert not head_heartbeat_expired(tmp_path/'head',now=1299)
    assert head_heartbeat_expired(tmp_path/'head',now=1301)
    calls=[]
    def disappears(self):calls.append(str(self));raise FileNotFoundError('replaced concurrently')
    monkeypatch.setattr(Path,'stat',disappears)
    assert not head_heartbeat_expired(tmp_path/'head',now=99999)
    assert len(calls)==1


@pytest.mark.parametrize('current_writer',[False,True])
def test_lost_claim_reconciliation_preserves_traces_and_never_scores_failure(tmp_path,current_writer):
    root=tmp_path/'profile';transport=tmp_path/'transport';uid='a'*32;worker='rp-jhh-astra-wide-test-old-job-task'
    path=root/'episodes/case/infra-00/result.json';path.parent.mkdir(parents=True)
    original={'status':'running','requested_model':'openai/openai/gpt-6.2-astra','requested_effort':'xhigh',
        'episode_id':uid,'case':{'case_id':'case'},'valid_actions':7,'generated_tokens':123}
    path.write_text(json.dumps(original));turns=path.with_name('turns.jsonl');turns.write_text('original trace\n')
    episode=transport/'episodes'/uid;episode.mkdir(parents=True)
    (episode/'claimed.json').write_text(json.dumps({'worker_id':worker}))
    (transport/'workers').mkdir();(transport/'workers'/(worker+'.json')).write_text('{"updated_unix":100}')
    lock=(path.parent.parent/'writer.lock').open('a')
    if current_writer:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        receipts=watch.reconcile_lost_claims(root,transport,['rp-jhh-astra-wide-test-old-job'],now=1000)
        value=json.loads(path.read_text())
        assert value['status']==('running'if current_writer else'infrastructure_failure')
        assert value['valid_actions']==7 and value['generated_tokens']==123
        assert 'native_success'not in value and turns.read_text()=='original trace\n'
        assert (episode/'infrastructure-error.json').exists()and(episode/'cancel.json').exists()
        assert receipts[0]['orphan_record_finalized']is(not current_writer)
    finally:lock.close()


def test_fresh_or_unowned_claim_is_preserved(tmp_path):
    root=tmp_path/'profile';transport=tmp_path/'transport';uid='b'*32;worker='rp-jhh-astra-wide-healthy-new-job-task'
    path=root/'episodes/case/infra-00/result.json';path.parent.mkdir(parents=True)
    original={'status':'opening','requested_model':'openai/openai/gpt-6.2-astra','requested_effort':'xhigh',
              'episode_id':uid,'case':{'case_id':'case'}}
    path.write_text(json.dumps(original));episode=transport/'episodes'/uid;episode.mkdir(parents=True)
    (episode/'claimed.json').write_text(json.dumps({'worker_id':worker}));(transport/'workers').mkdir()
    (transport/'workers'/(worker+'.json')).write_text('{"updated_unix":990}')
    assert watch.reconcile_lost_claims(root,transport,['rp-jhh-astra-wide-healthy-new-job'],now=1000)==[]
    assert watch.reconcile_lost_claims(root,transport,['rp-jhh-astra-wide-different-job'],now=99999)==[]
    assert json.loads(path.read_text())==original and not(episode/'cancel.json').exists()
