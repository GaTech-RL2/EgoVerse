import json

from reproductions.leact.tracking.pear_collect import fold,fold_launch_check,parse_records


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
