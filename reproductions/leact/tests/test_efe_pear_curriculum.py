import json
import random

import pytest

from reproductions.leact.cluster.efe_pear_curriculum import (
    FLOORS, INITIAL_QUOTAS, REASONING_FAMILIES, SuccessHistory,
    allocate_bounded_quotas, choose_reset, retained_selection, sgs_probabilities,
    token_mean_factors, usable_group_probability,
)


def test_sgs_prefers_learnable_resets_and_keeps_a_uniform_floor():
    q=sgs_probabilities({'impossible':0.,'learnable':.5,'solved':1.})
    assert q['learnable']>q['impossible']
    assert q['impossible']==pytest.approx(q['solved'])
    assert sum(q.values())==pytest.approx(1.)
    assert min(q.values())>=.2/3
    assert usable_group_probability(.01)==pytest.approx(0.0772553055720799)
    assert usable_group_probability(.5)==pytest.approx(0.9921875)
    assert sgs_probabilities({'a':None,'b':None})=={'a':.5,'b':.5}


def test_filtered_groups_are_recorded_in_history_but_infrastructure_is_not():
    h=SuccessHistory()
    assert h.observe('fail','reset',[0]*8)
    assert h.observe('success','reset',[1]*8)
    assert h.rate('reset')==.5
    assert not h.observe('broken','reset',[0]*8,infrastructure_ok=False)
    restored=SuccessHistory.from_state_dict(json.loads(json.dumps(h.state_dict())))
    assert not restored.observe('success','reset',[1]*8)
    assert restored.rate('reset')==.5
    for i in range(20):restored.observe('more'+str(i),'reset',[1]*8)
    assert len(restored.history['reset'])==100 and restored.rate('reset')==1


def test_sgs_cannot_change_task_stratum_or_evaluation_sampling():
    configs=[{'config_id':k,'task':task,'stratum':stratum,'partition':'train',
              'admitted':True,'already_successful':False,'observation_complete':True}
             for k,task,stratum in [('a','pick','full'),('b','pick','full'),
                                    ('c','pick','near_goal'),('d','place','full')]]
    rng=random.Random(11);history=SuccessHistory()
    for _ in range(30):
        chosen,receipt=choose_reset(configs,history,rng,task='pick',stratum='full')
        assert chosen in {'a','b'} and set(receipt['probabilities'])=={'a','b'}
    with pytest.raises(ValueError,match='evaluation'):
        choose_reset(configs,history,rng,task='pick',stratum='full',partition='test')
    configs[0]['already_successful']=True
    with pytest.raises(ValueError,match='terminal'):
        choose_reset(configs,history,rng,task='pick',stratum='full')


def test_exact_retained_quotas_do_not_backfill_with_easy_families():
    groups=[{'uid':f'{family}-{i}','family':family,'dispatch_step':i,
             'outcomes':[0,1]*4,'infrastructure_ok':True}
            for family,n in INITIAL_QUOTAS.items() for i in range(n)]
    assert len(retained_selection(groups,INITIAL_QUOTAS))==128
    groups[-1]['outcomes']=[1]*8
    assert retained_selection(groups,INITIAL_QUOTAS) is None
    groups.extend({'uid':f'easy-{i}','family':'general_manipulation','dispatch_step':0,
                   'outcomes':[0,1]*4,'infrastructure_ok':True} for i in range(100))
    assert retained_selection(groups,INITIAL_QUOTAS) is None
    groups[-101]['outcomes']=[0,1]*4
    selected=retained_selection(groups,INITIAL_QUOTAS)
    assert len(selected)==128 and 'general_manipulation-23' not in selected


def test_family_loss_is_independent_of_response_length_and_dp_partition():
    # Rank 0 and rank 1 jointly contribute 10 control tokens and 100 reasoning
    # tokens. Apply factors once to the advantage, preserving the actor's global
    # generated-token denominator. Equal per-token values keep equal family loss.
    counts={'control':10,'reasoning':100};weights={'control':.75,'reasoning':.25}
    factors=token_mean_factors(counts,weights)
    rank_tokens=[{'control':[2.]*3,'reasoning':[6.]*90},
                 {'control':[2.]*7,'reasoning':[6.]*10}]
    loss=sum(sum(values)*factors[k] for rank in rank_tokens for k,values in rank.items())/110
    assert loss==pytest.approx(.75*2+.25*6)
    for missing in [{'control':0,'reasoning':100},{'control':10}]:
        with pytest.raises(ValueError):token_mean_factors(missing,weights)


def test_adaptive_allocation_preserves_floors_and_control_reasoning_budgets():
    rng=random.Random(19)
    for _ in range(100):
        scores={key:rng.random()**10 for key in INITIAL_QUOTAS}
        q=allocate_bounded_quotas(scores)
        assert sum(q.values())==128
        assert sum(q[k] for k in REASONING_FAMILIES)==32
        assert all(q[k]>=FLOORS[k] for k in q)
    with pytest.raises(ValueError):
        allocate_bounded_quotas({key:float('nan') for key in INITIAL_QUOTAS})
