import json

import pytest

from reproductions.leact.cluster.efe_pear_curriculum import initial_quotas,validate_quotas,REASONING_FAMILIES
from reproductions.leact.cluster.efe_pear_curriculum_state import CurriculumState
from reproductions.leact.cluster.efe_pear_multitask_config import configuration
from reproductions.leact.cluster.efe_pear_signal_curriculum import POLICY,select_available
from test_efe_pear_curriculum_state import manifest
from test_pear_multitask_config import inputs


@pytest.mark.parametrize('size',[32,64,128])
def test_startup_batch_preserves_target_ratios_and_complete_groups_of_eight(size):
    config,m=inputs(None)
    config.data['train_batch_size']=size
    m.update(sampling_policy=POLICY,retained_groups_per_update=size)
    state,metadata=configuration(config,m)
    assert metadata['retained_groups']==size and metadata['rollouts_per_group']==8
    assert sum(state.quotas.values())==size
    assert sum(state.quotas[f]for f in REASONING_FAMILIES)==size//4
    assert sum(state.loss_weights.values())==pytest.approx(1.)
    validate_quotas(state.quotas,batch_size=size)


@pytest.mark.parametrize('size',[0,16,True,33])
def test_unvalidated_batch_sizes_are_rejected(size):
    with pytest.raises(ValueError,match='32, 64 or 128'):initial_quotas(size)


def test_32_group_update_still_needs_32_mixed_groups_and_no_single_family_can_block_it():
    good=[{'uid':str(i),'family':'general_manipulation','dispatch_step':1,
        'infrastructure_ok':True,'outcomes':[0,1]*4}for i in range(32)]
    zero=[{'uid':'zero','family':'native_gr00t_yam','dispatch_step':1,
        'infrastructure_ok':True,'outcomes':[0]*8}]
    assert select_available(good[:-1]+zero,initial_quotas(32),[])is None
    assert set(select_available(good+zero,initial_quotas(32),[]))=={str(i)for i in range(32)}


def test_32_group_checkpoint_keeps_sampling_history_and_rejects_silent_recipe_change():
    m=manifest();m.update(sampling_policy=POLICY,retained_groups_per_update=32)
    state=CurriculumState(m)
    for i in range(32):
        state.register(str(i),'general_manipulation/0',1);state.observe(str(i),[0,1]*4)
    selected=state.select_groups(list(state.groups.values()));state.record_selection(1,selected)
    saved=json.loads(json.dumps(state.state_dict()));restored=CurriculumState(m);restored.restore(saved)
    assert len(restored.selection_history[0]['uids'])==32
    assert[state.draw(step=2)for _ in range(20)]==[restored.draw(step=2)for _ in range(20)]
    with pytest.raises(ValueError,match='changed'):CurriculumState({**m,'retained_groups_per_update':128}).restore(saved)
