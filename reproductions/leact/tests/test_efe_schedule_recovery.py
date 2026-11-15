import json
import hashlib
from datetime import datetime,timezone,timedelta

import yaml
import pytest

from reproductions.leact.cluster import recover_efe_schedule as recovery
from reproductions.leact.embodied_first_exam.report import schedule_plan


def test_recovery_queue_stops_after_one_submission(tmp_path,monkeypatch):
    monkeypatch.setattr(recovery,'ROOT',tmp_path)
    submitted=[]
    monkeypatch.setattr(recovery,'submit',lambda build: submitted.append(str(build)) or {'name':'retry-1'})
    monkeypatch.setattr(recovery,'ensure_monitor',lambda build:None)
    monkeypatch.setattr(recovery,'command',lambda args:{'status':'PENDING','name':'retry-1'})
    arms=[]
    for i in range(3):
        build=tmp_path/f'build-{i}';build.mkdir();dest=tmp_path/f'run-{i}';dest.mkdir()
        arms.append({'id':str(i),'build':str(build),'local_results_directory':dest.name})
    state=recovery.advance({'arms':arms})
    assert len(submitted)==1 and state['workflow']=='retry-1'
    (tmp_path/'build-0/submission.json').write_text(json.dumps({'name':'retry-1'}))
    monkeypatch.setattr(recovery,'command',lambda args:{'status':'FAILED_CANCELED','name':'retry-1'})
    state=recovery.advance({'arms':arms})
    assert state['status']=='needs_review' and len(submitted)==1
    assert json.loads((tmp_path/'run-1/scheduler.json').read_text())['status']=='HELD'


def test_recovery_keeps_original_plan_and_failed_attempt(tmp_path):
    folder=tmp_path/'training-pilot';folder.mkdir()
    original={'arms':[{'id':'qwen-warm','name':'Warmstart','build':'original','local_results_directory':'old'}]}
    path=folder/'schedule-comparison.json';path.write_text(json.dumps(original));frozen=path.read_bytes()
    (folder/'schedule-recovery.json').write_text(json.dumps({'arms':[{'id':'qwen-warm','build':'retry','local_results_directory':'new'}]}))
    active=schedule_plan(tmp_path)
    assert active['arms'][0]['local_results_directory']=='new'
    assert active['prior_attempts'][0]['local_results_directory']=='old'
    assert active['prior_attempts'][0]['id']!=active['arms'][0]['id']
    assert path.read_bytes()==frozen


def test_capacity_retries_obey_backoff_and_submit_once(tmp_path,monkeypatch):
    monkeypatch.setattr(recovery,'ROOT',tmp_path)
    original=tmp_path/'original';original.mkdir();destination=tmp_path/'run';destination.mkdir()
    (original/'submission.json').write_text(json.dumps({'name':'old'}))
    arm={'id':'arm','build':str(original),'local_results_directory':'run','attempt':1}
    plan={'arms':[arm],'priority':'LOW','retry_policy':{'enabled':True,'backoff_seconds':300,'max_backoff_seconds':900}}
    state={'status':'FAILED_CANCELED','name':'old','end_time':datetime.now(timezone.utc).isoformat(),
           'groups':[{'failure_message':'GPU quota enforcement: capacity reclaimed'}]}
    submitted=[];prepared=[]
    monkeypatch.setattr(recovery,'command',lambda args:state if args[3]=='old' else {'status':'PENDING','name':'new'})
    monkeypatch.setattr(recovery,'submit',lambda build: submitted.append(str(build)) or {'name':'new'})
    monkeypatch.setattr(recovery,'ensure_monitor',lambda build:None)
    def prepare_next(plan,arm,workflow):
        prepared.append(workflow);new=tmp_path/'new';new.mkdir();out=tmp_path/'next';out.mkdir()
        arm.update(build=str(new),local_results_directory='next',attempt=2)
    monkeypatch.setattr(recovery,'next_attempt',prepare_next)
    assert recovery.advance(plan)['status']=='retry_wait'
    assert submitted==prepared==[]
    state['end_time']=(datetime.now(timezone.utc)-timedelta(seconds=301)).isoformat()
    assert recovery.advance(plan)['workflow']=='new'
    assert prepared==['old'] and len(submitted)==1
    (tmp_path/'new/submission.json').write_text(json.dumps({'name':'new'}))
    assert recovery.advance(plan)['status']=='active'
    assert len(submitted)==1
    assert recovery.retry_delay(plan,{'attempt':99})==900


def test_manual_cancellation_is_not_a_capacity_retry():
    assert not recovery.capacity_interruption({'status':'FAILED_CANCELED','groups':[{'failure_message':'Canceled by user'}]})
    assert not recovery.capacity_interruption({'status':'FAILED','groups':[{'failure_message':'CUDA out of memory'}]})
    assert recovery.capacity_interruption({'status':'FAILED_PREEMPTED','groups':[]})
    assert recovery.capacity_interruption({'status':'FAILED_QUEUE_TIMEOUT','groups':[]})


def test_verified_osmo_control_failure_is_retryable_but_sigterm_alone_is_not():
    state={'status':'FAILED','groups':[{'failure_message':
        'Exit code 2001 due to OSMO Control failure. Exit code 143 due to Task preflight failure.',
        'tasks':[{'exit_code':2001},{'exit_code':143}]}]}
    assert recovery.capacity_interruption(state)
    state['groups'][0]['tasks']=[{'exit_code':143}]
    assert not recovery.capacity_interruption(state)
    state['groups'][0]['failure_message']='Training process exited with SIGTERM'
    assert not recovery.capacity_interruption(state)


def test_control_failure_preserves_preemption_evidence_once(tmp_path,monkeypatch):
    from types import SimpleNamespace
    event='2026-10-28 03:25:19+00:00 [preflight] Evict: Pod example was preempted by higher priority workload another'
    calls=[]
    monkeypatch.setattr(recovery.subprocess,'run',lambda args,**kwargs:
        calls.append(args) or SimpleNamespace(returncode=0,stdout=event+'\n'))
    state={'name':'fixture','groups':[{'tasks':[{'exit_code':2001}]}]}
    recovery.capture_control_failure(state,tmp_path)
    evidence=json.loads((tmp_path/'interruption.json').read_text())
    assert evidence['kind']=='priority_preemption' and evidence['evidence']==[event]
    recovery.capture_control_failure(state,tmp_path)
    assert len(calls)==1


def test_completed_training_is_not_restarted_after_control_failure(tmp_path,monkeypatch):
    monkeypatch.setattr(recovery,'ROOT',tmp_path)
    build=tmp_path/'build';build.mkdir();dest=tmp_path/'run';dest.mkdir()
    (build/'submission.json').write_text(json.dumps({'name':'finished'}))
    (dest/'complete.json').write_text(json.dumps({'success':True}))
    monkeypatch.setattr(recovery,'command',lambda args:{'status':'FAILED','name':'finished',
        'groups':[{'failure_message':'OSMO Control failure','tasks':[{'exit_code':2001}]}]})
    monkeypatch.setattr(recovery,'ensure_monitor',lambda build:None)
    monkeypatch.setattr(recovery,'submit',lambda build:pytest.fail('Completed training must not restart'))
    assert recovery.advance({'arms':[{'id':'arm','build':str(build),'local_results_directory':'run'}],
        'retry_policy':{'enabled':True}})['status']=='training_complete'


def test_only_verified_pre_model_nvlink_fault_is_retried(tmp_path):
    state={'status':'FAILED','name':'failed-node','groups':[{'tasks':[{'node_name':'node-a'}]}]}
    log=tmp_path/'smoke.log'
    log.write_text('[rank3]: torch.cuda.OutOfMemoryError: CUDA out of memory\n')
    assert not recovery.gpu_interconnect_interruption(state,tmp_path)
    log.write_text('[rank3]: ncclUnhandledCudaError: Call to CUDA function failed.\n')
    assert not recovery.gpu_interconnect_interruption(state,tmp_path)
    log.write_text('[rank3]: probe = torch.tensor(float(rank+1), device=device); dist.all_reduce(probe)\n'
        '[rank3]: ncclUnhandledCudaError: Call to CUDA function failed.\n'
        '[rank3]: P2P is disabled between NVLINK connected GPUs 1 and 0.\n')
    assert recovery.gpu_interconnect_interruption(state,tmp_path)
    diagnosis=json.loads((tmp_path/'interruption.json').read_text())
    assert diagnosis['kind']=='gpu_interconnect_failure' and diagnosis['nodes']==['node-a']
    assert diagnosis['source_sha256']==hashlib.sha256(log.read_bytes()).hexdigest()
    assert not recovery.gpu_interconnect_interruption({**state,'status':'FAILED_CANCELED'},tmp_path)
    (tmp_path/'train').mkdir();(tmp_path/'train/metrics.jsonl').write_text('{"step":1}\n')
    assert not recovery.gpu_interconnect_interruption(state,tmp_path)


def test_parallel_queue_starts_an_independent_arm_and_respects_two_nodes(tmp_path,monkeypatch):
    arms=[]
    for i in range(3):
        folder=tmp_path/str(i);folder.mkdir()
        if i==0:(folder/'submission.json').write_text(json.dumps({'name':'active-0'}))
        arms.append({'id':str(i),'build':str(folder)})
    plan={'arms':arms,'max_concurrent_training_nodes':2};calls=[]
    monkeypatch.setattr(recovery,'command',lambda args:{'workflows':[{'name':'active-0'}]})
    def advance(plan_arg,selected,allow_submit):
        assert plan_arg is plan
        arm=selected[0];calls.append((arm['id'],allow_submit))
        if arm['id']=='0':return {'status':'active','workflow':'active-0'}
        return {'status':'active' if allow_submit else 'queued','workflow':'new-'+arm['id']}
    monkeypatch.setattr(recovery,'advance',advance)
    result=recovery.advance_queue(plan)
    assert calls==[('0',True),('1',True),('2',False)]
    assert [row['status'] for row in result['arms']]==['active','active','queued']


def test_parallel_queue_uses_preempted_slot_while_retry_backs_off(tmp_path,monkeypatch):
    arms=[]
    for i in range(3):
        folder=tmp_path/str(i);folder.mkdir()
        if i<2:(folder/'submission.json').write_text(json.dumps({'name':'active-'+str(i)}))
        arms.append({'id':str(i),'build':str(folder)})
    plan={'arms':arms,'max_concurrent_training_nodes':2};calls=[]
    monkeypatch.setattr(recovery,'command',lambda args:{'workflows':[{'name':'active-0'},{'name':'active-1'}]})
    def advance(plan_arg,selected,allow_submit):
        arm=selected[0];calls.append((arm['id'],allow_submit))
        if arm['id']=='0':return {'status':'retry_wait','workflow':'active-0'}
        return {'status':'active','workflow':'active-'+arm['id']}
    monkeypatch.setattr(recovery,'advance',advance)
    result=recovery.advance_queue(plan)
    assert calls==[('0',True),('1',True),('2',True)]
    assert sum(row['status']=='active' for row in result['arms'])==2


def test_new_attempt_copies_frozen_training_recipe(tmp_path,monkeypatch):
    monkeypatch.setattr(recovery,'ROOT',tmp_path);monkeypatch.setattr(recovery,'PLAN',tmp_path/'plan.json')
    source=tmp_path/'source';source.mkdir()
    spec={'lr':1e-4,'effective_batch':32,'expected_optimizer_steps':48,'seed':17,
          'expected_schedule_sha256':'frozen','archive_root':'/shared/original','local_results_directory':'run'}
    (source/'spec.json').write_text(json.dumps(spec));(source/'trainer.py').write_text('frozen trainer\n')
    (source/'workflow.yaml').write_text(yaml.safe_dump({'workflow':{'name':'job','resources':{'default':{'gpu':8}}}}))
    (source/'source-manifest.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}))
    arm={'id':'arm','build':str(tmp_path/'first'),'local_results_directory':'first-run',
         'original_build':str(source),'original_local_results_directory':'run','attempt':1}
    plan={'arms':[arm],'priority':'LOW'};recovery.next_attempt(plan,arm,'old-workflow')
    updated=json.loads((recovery.Path(arm['build'])/'spec.json').read_text())
    assert {k for k in spec.keys()|updated.keys() if spec.get(k)!=updated.get(k)}=={
        'archive_root','local_results_directory','recovery_of','recovery_reason'}
    assert (recovery.Path(arm['build'])/'trainer.py').read_bytes()==(source/'trainer.py').read_bytes()
    assert arm['previous_attempts'][0]['local_results_directory']=='first-run'
    assert arm['attempt']==2


def test_osmo_init_retry_requires_verified_failure_before_training(tmp_path):
    task={'exit_code':392,'start_time':None,'node_name':'node-a',
          'failure_message':'Exit code 392 due to Task osmo-init failure.'}
    state={'name':'init-failure','status':'FAILED','start_time':None,'groups':[{'tasks':[task]}]}
    assert recovery.initialization_interruption(state,tmp_path)
    diagnosis=json.loads((tmp_path/'interruption.json').read_text())
    assert diagnosis['kind']=='osmo_initialization_failure'
    assert diagnosis['before_user_container_started'] and diagnosis['nodes']==['node-a']
    assert not recovery.initialization_interruption({**state,'status':'FAILED_CANCELED'},tmp_path)
    assert not recovery.initialization_interruption({**state,'start_time':'2026-10-28T17:00:00'},tmp_path)
    unknown=dict(state);unknown.pop('start_time')
    assert not recovery.initialization_interruption(unknown,tmp_path)
    task['start_time']='2026-10-28T17:00:00'
    assert not recovery.initialization_interruption(state,tmp_path)
    task['start_time']=None;task['exit_code']=137
    assert not recovery.initialization_interruption(state,tmp_path)
    task['exit_code']=392;task['failure_message']='Training process failed'
    assert not recovery.initialization_interruption(state,tmp_path)
    task['failure_message']='Exit code 392 due to Task osmo-init failure.'
    (tmp_path/'train').mkdir();(tmp_path/'train/metrics.jsonl').write_text('{"step":1}\n')
    assert not recovery.initialization_interruption(state,tmp_path)


def test_parallel_unknown_failure_does_not_hold_independent_arms(tmp_path,monkeypatch):
    monkeypatch.setattr(recovery,'ROOT',tmp_path)
    arms=[];submitted=[]
    for i in range(3):
        build=tmp_path/f'build-{i}';build.mkdir()
        (tmp_path/f'run-{i}').mkdir()
        if i==0:(build/'submission.json').write_text(json.dumps({'name':'failed'}))
        arms.append({'id':str(i),'build':str(build),'local_results_directory':f'run-{i}'})
    plan={'arms':arms,'max_concurrent_training_nodes':2}
    def command(args):
        if args[2]=='list':return {'workflows':[]}
        if args[3]=='failed':return {'name':'failed','status':'FAILED','groups':[{'failure_message':'CUDA out of memory'}]}
        return {'name':args[3],'status':'PENDING'}
    def submit(build):
        submitted.append(build);receipt={'name':build.name}
        (build/'submission.json').write_text(json.dumps(receipt));return receipt
    monkeypatch.setattr(recovery,'command',command)
    monkeypatch.setattr(recovery,'submit',submit)
    monkeypatch.setattr(recovery,'ensure_monitor',lambda build:None)
    result=recovery.advance_queue(plan)
    assert result['status']=='active' and len(submitted)==2
    assert [row['status'] for row in result['arms']]==['needs_review','active','active']
    assert result['review_required'][0]['id']=='0'
    assert json.loads((tmp_path/'run-1/scheduler.json').read_text())['status']=='PENDING'
    result=recovery.advance_queue(plan)
    assert result['status']=='active' and len(submitted)==2


def test_parallel_queue_reports_review_when_no_arm_can_progress(tmp_path,monkeypatch):
    arms=[]
    for i in range(3):
        build=tmp_path/str(i);build.mkdir();arms.append({'id':str(i),'build':str(build)})
    monkeypatch.setattr(recovery,'command',lambda args:{'workflows':[]})
    monkeypatch.setattr(recovery,'advance',lambda plan,selected,allow_submit:
        {'status':'training_complete' if selected[0]['id']=='0' else 'needs_review'})
    result=recovery.advance_queue({'arms':arms,'max_concurrent_training_nodes':2})
    assert result['status']=='needs_review' and len(result['review_required'])==2
