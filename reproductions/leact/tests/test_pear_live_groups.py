from reproductions.leact.tracking.pear_collect import fold


def test_task_rewards_are_live_before_first_optimizer_update_and_idempotent():
    state={'optimizer_steps_observed':0}
    first={'kind':'group','event_id':'group/one','partition':'train','infrastructure_ok':True,
        'family':'general_manipulation','suite':'sim','task':'pick','rewards':[0,0,0,0,0,0,0,1]}
    second={**first,'event_id':'group/two','rewards':[1]*8}
    fold(state,'PEAR_EVENT',first);fold(state,'PEAR_EVENT',second);fold(state,'PEAR_EVENT',first)
    assert state['optimizer_steps_observed']==0
    assert state['latest_task_rewards']['sim/pick']==9/16
    assert state['latest_metrics']['episodes_cumulative/all']==16
    assert state['latest_metrics']['groups_cumulative/judged']==2


def test_infrastructure_failure_never_becomes_a_zero_reward_episode():
    state={}
    fold(state,'PEAR_EVENT',{'kind':'group','event_id':'group/lost','partition':'train',
        'infrastructure_ok':False,'family':'general_manipulation','suite':'sim','task':'pick','rewards':[0]*8})
    assert not state.get('latest_task_rewards')
