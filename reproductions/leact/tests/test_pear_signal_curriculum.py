from collections import Counter
import copy
import json

import pytest

from reproductions.leact.cluster.efe_pear_curriculum import INITIAL_QUOTAS,retained_selection
from reproductions.leact.cluster.efe_pear_signal_curriculum import (
    POLICY,active_weights,dispatch_weights,select_available)
from reproductions.leact.cluster.efe_pear_curriculum_state import CurriculumState
from test_efe_pear_curriculum_state import manifest


def groups(family,n,*,prefix='g',rewards=None):
    return[{'uid':f'{prefix}-{i}','family':family,'dispatch_step':i,
        'infrastructure_ok':True,'outcomes':rewards or[0,1]*4}for i in range(n)]


def test_zero_signal_family_does_not_block_or_contribute_fake_training_data():
    available=groups('general_manipulation',128)
    available+=groups('native_gr00t_yam',50,prefix='zero',rewards=[0]*8)
    assert retained_selection(available,INITIAL_QUOTAS)is None
    selected=select_available(available,INITIAL_QUOTAS,[])
    assert len(selected)==128 and all(uid.startswith('g-')for uid in selected)
    assert select_available(available[1:],INITIAL_QUOTAS,[])is None


def test_newly_available_family_receives_priority_for_rolling_coverage_debt():
    history=[{'family_counts':{'general_manipulation':128}}for _ in range(7)]
    candidates=groups('general_manipulation',128)
    candidates+=groups('object_pointing',4,prefix='new')
    selected=select_available(candidates,INITIAL_QUOTAS,history)
    assert selected[0].startswith('new-')
    assert all('new-'+str(i)in selected for i in range(4))
    weights=active_weights(INITIAL_QUOTAS,{'general_manipulation','object_pointing'})
    assert sum(weights.values())==pytest.approx(1.)
    assert weights['general_manipulation']/weights['object_pointing']==6


def test_signal_dispatch_has_an_exploration_floor_for_every_family():
    outcomes={'native_gr00t_yam':[{'retained':False}]*30,
              'general_manipulation':[{'retained':True}]*30}
    q=dispatch_weights(INITIAL_QUOTAS,[],outcomes,{})
    assert sum(q.values())==pytest.approx(1.)
    assert all(q[f]>=.2*n/128 for f,n in INITIAL_QUOTAS.items())
    assert q['general_manipulation']/24>q['native_gr00t_yam']/32


def test_signal_selection_and_next_dispatches_survive_a_checkpoint_restart():
    m=manifest();m['sampling_policy']=POLICY;a=CurriculumState(m)
    for i in range(128):
        uid=f'uid-{i}';a.register(uid,'general_manipulation/0',1);a.observe(uid,[0,1]*4)
    selected=a.select_groups(list(a.groups.values()));a.record_selection(1,selected)
    a.record_selection(1,list(reversed(selected)))
    b=CurriculumState(m);b.restore(json.loads(json.dumps(a.state_dict())))
    assert b.selection_history==a.selection_history
    assert[a.draw(step=2)for _ in range(100)]==[b.draw(step=2)for _ in range(100)]
    assert b.batch_loss_weights({'general_manipulation'})=={'general_manipulation':1.}


def test_task_sgs_prefers_mixed_success_over_observed_all_failure_tasks():
    m=manifest();m['sampling_policy']=POLICY
    extra=[]
    for c in m['configs']:
        if c['family']!='general_manipulation':continue
        row=copy.deepcopy(c);row['config_id']='learnable/'+c['config_id'].rsplit('/',1)[1]
        row['task']='learnable';row['env_args']['task']='learnable';extra.append(row)
    m['configs']+=extra;a=CurriculumState(m)
    for i in range(20):
        for key,values in [('general_manipulation/0',[0]*8),('learnable/0',[0,1]*4)]:
            uid=f'{key}-{i}';a.register(uid,key,1);a.observe(uid,values)
    counts=Counter()
    for _ in range(3000):
        key=a.draw(step=1)
        if a.configs[key]['family']=='general_manipulation':
            counts[a.configs[key]['task']]+=1
            assert a.last_draw['sgs_scope']=='task_and_reset'
    assert counts['learnable']/sum(counts.values())>.55
