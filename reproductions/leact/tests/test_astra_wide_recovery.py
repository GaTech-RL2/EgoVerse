import hashlib
import contextlib
import io
import json
import fcntl
import copy
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


@pytest.mark.parametrize('retries,allowed', [(16, True), (63, True), (64, False)])
def test_authorized_allocation_recovery_bound_keeps_low_priority_and_durable_lineage(
        tmp_path, monkeypatch, retries, allowed):
    recorded, submissions = [], []
    monkeypatch.setattr(watch, 'refresh_build', lambda _: tmp_path/'verified-build')
    def submit(build, **kwargs):
        submissions.append((build, kwargs))
        return {'name': 'rp-jhh-astra-wide-efe-recovery-1'}
    monkeypatch.setattr(watch, 'submit_reconciled', submit)
    monkeypatch.setattr(watch, 'record_launch', recorded.append)
    state = {'workflow': 'rp-jhh-astra-wide-efe-ended-1', 'label': 'efe',
             'pool': 'groot-h100-01', 'allocation_retries': retries}
    scheduler = {'status': 'FAILED', 'cancelled_by': None,
                 'failure_message': 'Exit code2001 due to OSMO Control failure'}
    result = watch.recover_allocation(state, scheduler)
    if not allowed:
        assert result is None and not submissions and not recorded
        return
    assert submissions == [(tmp_path/'verified-build',
                            {'pool': 'groot-h100-01', 'priority': 'LOW', 'nodes': 1})]
    assert result['allocation_retries'] == retries+1
    assert result['recovery_of'] == state['workflow']
    assert result['lost_native_prefixes'] == [state['workflow']]
    assert recorded[-1]['superseded_by'] == result['workflow']


def unallocated_queue_timeout():
    return {'status': 'FAILED_QUEUE_TIMEOUT', 'cancelled_by': None,
            'start_time': None, 'duration': None, 'groups': [{'start_time': None,
                'tasks': [{'start_time': None, 'node_name': None, 'pod_ip': None,
                           'pod_name': 'precreated-while-queued'}]}]}


@pytest.mark.parametrize('retries,expected', [(18, True), (64, False)])
def test_unfinished_never_started_queue_timeout_retries_low_with_same_bound(tmp_path, monkeypatch, retries, expected):
    submissions, recorded = [], []
    monkeypatch.setattr(watch, 'refresh_build', lambda _: tmp_path/'verified-profile')
    def submit(build, **kwargs):
        submissions.append((build, kwargs))
        return {'name': 'rp-jhh-astra-wide-efe-queue-recovery-1'}
    monkeypatch.setattr(watch, 'submit_reconciled', submit)
    monkeypatch.setattr(watch, 'record_launch', recorded.append)
    state = {'workflow': 'rp-jhh-astra-wide-efe-ended-1', 'pool': 'groot-h100-01',
             'allocation_retries': retries, 'teacher_profile_id': 'astra62-xhigh-v1'}
    result = watch.recover_allocation(state, unallocated_queue_timeout())
    assert bool(submissions) is expected
    if expected:
        assert submissions == [(tmp_path/'verified-profile',
                                {'pool': 'groot-h100-01', 'priority': 'LOW', 'nodes': 1})]
        assert result['allocation_retries'] == 19
        assert result['teacher_profile_id'] == state['teacher_profile_id']
        assert result['recovery_of'] == state['workflow']
    else:
        assert result is None and not recorded


@pytest.mark.parametrize('kind', ['cancelled', 'workflow_started', 'group_started', 'task_started',
    'node_assigned', 'pod_ip_assigned', 'nonzero_duration', 'unknown_start', 'unknown_task_allocation', 'no_tasks'])
def test_allocated_cancelled_or_unproven_queue_timeout_never_retries(kind, monkeypatch):
    scheduler = unallocated_queue_timeout()
    task = scheduler['groups'][0]['tasks'][0]
    if kind == 'cancelled':
        scheduler['cancelled_by'] = 'rpunamiya'
    elif kind == 'workflow_started':
        scheduler['start_time'] = '2026-11-20T00:00:00'
    elif kind == 'group_started':
        scheduler['groups'][0]['start_time'] = '2026-11-20T00:00:00'
    elif kind == 'task_started':
        task['start_time'] = '2026-11-20T00:00:00'
    elif kind == 'node_assigned':
        task['node_name'] = 'already-allocated'
    elif kind == 'pod_ip_assigned':
        task['pod_ip'] = '10.0.0.1'
    elif kind == 'nonzero_duration':
        scheduler['duration'] = 1
    elif kind == 'unknown_start':
        del scheduler['start_time']
    elif kind == 'unknown_task_allocation':
        del task['node_name']
    elif kind == 'no_tasks':
        scheduler['groups'] = []
    monkeypatch.setattr(watch, 'refresh_build', lambda *_: pytest.fail('Not a proven unallocated queue timeout'))
    assert watch.recover_allocation({'allocation_retries': 18}, scheduler) is None


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


def completed_collection_fixture():
    from reproductions.leact.cluster.switch_astra_wide_profile import PROFILE_ROOT
    cases = [{'case_id': 'case-'+str(index), 'task': 'assigned-task', 'config_id': 'reset-'+str(index),
              'scene_sha256': str(index)*64, 'partition': 'train', 'admitted': True}
             for index in range(2)]
    spec = {'owner_model_workflow': launch.OWNER, 'teacher_root': PROFILE_ROOT,
            'teacher_profile_id': 'astra62-xhigh-v1', 'teacher_task_names': ['assigned-task'],
            'teacher_plan': {'curriculum_sha256': 'pinned-curriculum', 'cases': cases}}
    rows = {case['case_id']: [{'case': copy.deepcopy(case),
        'requested_model': 'openai/openai/gpt-6.2-astra', 'requested_effort': 'xhigh',
        'teacher_profile_id': 'astra62-xhigh-v1', 'curriculum_sha256': 'pinned-curriculum',
        'status': 'graded_failure', 'native_success': False, 'finalized_native_outcome': True,
        'native_verdict_success': False, 'result_sha256': 'a'*64, 'native_verdict_sha256': 'b'*64}]
        for case in cases}
    return spec, rows


def test_native_graded_failures_complete_the_exact_planned_attempts():
    spec, rows = completed_collection_fixture()
    proof = watch.completed_taskset(spec, rows)
    assert proof['complete'] is True and proof['matched_finalized'] == proof['planned'] == 2
    assert all(row['status'] == 'graded_failure' for row in proof['case_receipts'])


@pytest.mark.parametrize('mismatch', ['missing', 'infrastructure', 'malformed', 'model', 'effort',
    'profile', 'scene', 'config', 'curriculum', 'not_finalized', 'missing_verdict', 'disagreeing_verdict'])
def test_unfinished_or_unproven_case_never_suppresses_recovery(mismatch):
    spec, rows = completed_collection_fixture()
    row = rows['case-1'][0]
    if mismatch == 'missing':
        del rows['case-1']
    elif mismatch == 'infrastructure':
        row['status'] = 'infrastructure_failure'
    elif mismatch == 'malformed':
        rows['case-1'] = [None, {'case': 'invalid'}]
    elif mismatch == 'model':
        row['requested_model'] = 'azure/openai/gpt-6-astra'
    elif mismatch == 'effort':
        row['requested_effort'] = 'high'
    elif mismatch == 'profile':
        row['teacher_profile_id'] = 'legacy-profile'
    elif mismatch == 'scene':
        row['case']['scene_sha256'] = 'changed-scene'
    elif mismatch == 'config':
        row['case']['config_id'] = 'another-reset'
    elif mismatch == 'curriculum':
        row['curriculum_sha256'] = 'another-curriculum'
    elif mismatch == 'not_finalized':
        row['finalized_native_outcome'] = False
    elif mismatch == 'missing_verdict':
        del row['native_verdict_sha256']
    elif mismatch == 'disagreeing_verdict':
        row['native_verdict_success'] = True
    proof = watch.completed_taskset(spec, rows)
    assert proof['complete'] is False and proof['matched_finalized'] == 1


def test_same_task_other_reset_is_not_completion_and_future_wall_arm_remains_separate():
    spec, rows = completed_collection_fixture()
    # A wrapper/aggregate saying two episodes does not prove both planned resets.
    rows['case-0'].append(copy.deepcopy(rows['case-0'][0]))
    del rows['case-1']
    assert not watch.completed_taskset(spec, rows)['complete']
    spec, rows = completed_collection_fixture()
    spec['wall_seconds'] = 3600
    assert not watch.completed_taskset(spec, rows)['complete']


@pytest.mark.parametrize('failed_status', ['FAILED', 'FAILED_QUEUE_TIMEOUT'])
def test_completed_tasksets_keep_accepted_allocations_and_skip_only_future_recovery(tmp_path, monkeypatch, failed_status):
    spec, receipts = completed_collection_fixture()
    complete = tmp_path/'complete'
    unfinished = tmp_path/'unfinished'
    complete.mkdir()
    unfinished.mkdir()
    (complete/'spec.json').write_text(json.dumps(spec))
    other = copy.deepcopy(spec)
    for case in other['teacher_plan']['cases']:
        case['case_id'] = 'unfinished-'+case['case_id']
    (unfinished/'spec.json').write_text(json.dumps(other))
    states = {'failed-complete': {'workflow': 'failed-complete', 'build': str(complete)},
              'live-complete': {'workflow': 'live-complete', 'build': str(complete)},
              'unfinished-core': {'workflow': 'unfinished-core', 'build': str(unfinished)}}
    (tmp_path/'launches.json').write_text(json.dumps(states))
    (tmp_path/'active-profile.json').write_text('{"teacher_profile_id":"astra62-xhigh-v1"}')
    monkeypatch.setattr(watch, 'ROOT', tmp_path)
    monkeypatch.setattr(watch, 'DEST', tmp_path)
    def query(args, **kwargs):
        status = 'RUNNING' if args[3] == 'live-complete' else failed_status
        return SimpleNamespace(returncode=0, stdout=json.dumps({'status': status, 'tasks': [{'status': status}]}))
    monkeypatch.setattr(watch.subprocess, 'run', query)
    monkeypatch.setattr(watch, 'snapshot', lambda *_: {
        'attempt_status': {}, 'sft_admitted': 0,
        'nodes': {'live-complete': {'node': {'phase': 'idle_failed'}, 'desired_id': 'keep-existing'}},
        'astra62_profile': {'attempt_status': {}, 'workers': {}, 'completed_cases': receipts}})
    monkeypatch.setattr(watch, 'recover_child', lambda *_: pytest.fail('Completed child must stay untouched'))
    recovered = []
    monkeypatch.setattr(watch, 'recover_allocation', lambda state, query: recovered.append(state['workflow']))
    result = watch.tick()
    assert recovered == ['unfinished-core']
    assert set(result['completed_tasksets_preserved']) == {'failed-complete', 'live-complete'}
    assert result['recovery_actions'] == []
    assert json.loads((tmp_path/'launches.json').read_text()) == states


@pytest.mark.parametrize('verdict', ['not valid JSON', '{"verifier_result":"malformed"}'])
def test_malformed_native_verdict_cannot_poison_completion_snapshot(tmp_path, monkeypatch, verdict):
    from reproductions.leact.cluster import switch_astra_wide_profile as profile
    legacy, current = tmp_path/'legacy', tmp_path/'current'
    spec, receipts = completed_collection_fixture()
    row = copy.deepcopy(receipts['case-0'][0])
    row['case']['suite'] = 'native_gr00t_yam'
    row['teacher_profile'] = {'id': 'astra62-xhigh-v1'}
    path = current/'episodes/case-0/infra-00/result.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(row))
    path.with_name('native-verdict.json').write_text(verdict)
    monkeypatch.setattr(watch, 'TEACHER_ROOT', str(legacy))
    monkeypatch.setattr(profile, 'PROFILE_ROOT', str(current))
    def execute(host, script, task):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            exec(script, {})
        return json.loads(stream.getvalue().split('EFE_CAPTURE ', 1)[1])
    monkeypatch.setattr(watch, 'remote', execute)
    result = watch.snapshot('healthy-reader', {})
    assert result['astra62_profile']['attempt_status'] == {'graded_failure': 1}
    assert result['astra62_profile']['completed_cases'] == {}
