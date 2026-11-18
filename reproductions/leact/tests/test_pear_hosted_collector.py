import json

from reproductions.leact.tracking.pear_collect import discovered_workflows,fold,fold_launch_check,parse_records,needs_qualification_poll


def test_hosted_collector_handles_prefixes_repeated_steps_and_separate_measurements():
    step={'kind':'step','run_id':'arm','step':1,'updated_utc':'now','headline':{'policy_loss':.2,'reference_policy_kl':None},
          'metrics':{'actor/pg_loss':.2,'reward_unfiltered_cumulative/task/sim/pick':.5,
                     'reward_unfiltered_cumulative/all':.5}}
    logs='2026 [driver] PEAR_EVENT '+json.dumps(step)+'\n'
    logs+='2026 [driver] PEAR_HEARTBEAT '+json.dumps({'run_id':'arm','step':1,'updated_utc':'now','rollout_diagnostics':{'completed_episodes':8}})+'\n'
    state={}
    for marker,row in parse_records(logs+logs):fold(state,marker,row)
    assert len(state['history'])==1
    assert state['latest_task_rewards']=={'sim/pick':.5}
    assert state['rollout_diagnostics']['completed_episodes']==8
    assert state['history'][0]['reference_policy_kl']is None
    assert list(parse_records('not a telemetry record\nPEAR_EVENT {partial'))==[]


def test_qualification_never_creates_primary_steps_and_old_attempts_cannot_replace_new():
    workflow={'labels':{'pear_run_id':'arm','pear_phase':'router_optimizer_qualification'},
              'submit_time':'2026-11-18T06:00:00','status':'RUNNING'}
    state={'history':[],'optimizer_steps_observed':0}
    assert fold_launch_check(state,workflow,[('PEAR_MODEL_CHECK',{'phase':'engine_update','primary_rl_updates':0})])
    assert state['history']==[]and state['optimizer_steps_observed']==0
    assert state['metadata']['qualification_only']is True
    assert not fold_launch_check(state,{**workflow,'submit_time':'2026-11-17T06:00:00','status':'FAILED'},[])
    fold_launch_check(state,workflow,[])
    assert state['metadata']['launch_checks']['router_optimizer_qualification']['phase']=='engine_update'
    fold(state,'PEAR_RUN',{'run_id':'arm','model_revision':'pinned','retained_groups':128})
    fold_launch_check(state,workflow,[])
    assert not state['metadata'].get('qualification_only',False)


def test_finished_qualification_does_not_delay_polling_the_live_run():
    workflow={'labels':{'pear_run_id':'arm','pear_phase':'all_tasks_sim'},
              'submit_time':'2026-11-18T06:00:00','status':'COMPLETED'}
    state={}
    assert needs_qualification_poll(state,workflow)
    fold_launch_check(state,workflow,[])
    assert needs_qualification_poll(state,workflow)
    fold_launch_check(state,workflow,[('PEAR_ADMISSION_CHECK',{'phase':'control_admission_completed','checked':3,'passed':3})])
    assert not needs_qualification_poll(state,workflow)
    assert needs_qualification_poll(state,{**workflow,'status':'RUNNING'})
    assert needs_qualification_poll(state,{**workflow,'submit_time':'2026-11-18T07:00:00'})


def test_native_retries_cannot_hide_the_model_behind_the_recent_workflow_limit():
    labels={'pear_run_id':'arm','pear_phase':'primary_training'}
    older={'name':'model-old','labels':labels,'submit_time':'2026-11-18T06:00:00'}
    current={**older,'name':'model-current','submit_time':'2026-11-18T07:00:00'}
    retries=[{'name':f'native-{i}','labels':{**labels,'pear_phase':'primary_native_workers'}}for i in range(100)]
    found=discovered_workflows(retries,[older,current],{'arm'})
    assert 'model-current'in found and'model-old'not in found and len(found)==101


def test_old_attempt_heartbeat_cannot_replace_a_newer_live_heartbeat():
    state={}
    fold(state,'PEAR_HEARTBEAT',{'updated_utc':'2026-11-18T07:00:00','step':2})
    fold(state,'PEAR_HEARTBEAT',{'updated_utc':'2026-11-18T06:00:00','step':0})
    assert state['heartbeat']['step']==2
