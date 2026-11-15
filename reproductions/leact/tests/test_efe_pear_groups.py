import copy
import pytest
from reproductions.leact.cluster.efe_pear_groups import admit_group


def fixture():
    membership={'training_tasks':{'pick':[{'scene_sha256':'scene','initial_state_sha256':'reset'}]}}
    episodes=[dict(episode_id=str(i),task='pick',scene_sha256='scene',initial_state_sha256='reset',
        partition='train_candidates',status='graded',reward=int(i==7),sampler_revision='policy',
        sampling_capture_complete=True,generated_tokens=100,native_events_verified=True) for i in range(8)]
    return episodes,membership


def test_failures_and_truncation_remain_in_native_reward_group():
    rows,membership=fixture();rows[0]['output_truncated']=True
    result=admit_group(rows,membership,'policy')
    assert result['admitted'] and result['rewards']==[0]*7+[1]
    assert result['native_rollouts_spent']==8 and result['generated_tokens_spent']==800
    assert result['truncated_episodes']==1 and result['advantages'][0]<0<result['advantages'][-1]


@pytest.mark.parametrize('reward',[0,1])
def test_constant_group_is_filtered_but_cost_not_erased(reward):
    rows,membership=fixture()
    for row in rows:row['reward']=reward
    result=admit_group(rows,membership,'policy')
    assert not result['admitted'] and result['native_rollouts_spent']==8
    assert result['successes']==8*reward


@pytest.mark.parametrize('change',[{'partition':'protected'},{'status':'infrastructure_failure'},
    {'sampler_revision':'stale'},{'initial_state_sha256':'different'},
    {'sampling_capture_complete':False},{'native_events_verified':False},{'episode_id':'1'}])
def test_group_cannot_hide_invalid_provenance(change):
    rows,membership=fixture();rows[0].update(change)
    with pytest.raises(ValueError):admit_group(rows,membership,'policy')


def test_whole_group_from_unseen_reset_is_rejected():
    rows,membership=fixture()
    for row in rows:row['initial_state_sha256']='heldout_reset'
    with pytest.raises(ValueError):admit_group(rows,membership,'policy')
