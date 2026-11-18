import json

from reproductions.leact.tracking.pear_collect import fold,parse_records


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
