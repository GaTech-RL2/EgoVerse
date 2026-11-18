import random
from types import SimpleNamespace as Namespace

import pytest

from reproductions.leact.cluster.efe_pear_curriculum import INITIAL_QUOTAS
from reproductions.leact.cluster.efe_pear_multitask_config import configuration


class Data(dict):
    def __getattr__(self,name):return self[name]


def inputs(seed):
    rows=[{'config_id':f,'family':f,'suite':'suite','task':f,'env_class':'Env','env_args':{'task':f},
        'partition':'train','stratum':'full','admitted':True,'already_successful':False,
        'observation_complete':True,'scene_sha256':'scene','admission_receipt_sha256':'receipt'}for f in INITIAL_QUOTAS]
    manifest={'partition':'train','qualification':'passed','curriculum_stage':'expanded_full','configs':rows}
    actor=Namespace(ppo_epochs=1,loss_agg_mode='token-mean',entropy_coeff=0,use_kl_loss=False,
        optim=Namespace(lr=1e-6),fsdp_config=Namespace(kernels='module.FrozenRouterTrainingKernels'),
        policy_loss=Namespace(rollout_correction=Namespace(score_centering=True)))
    config=Namespace(data=Data(seed=seed,train_batch_size=128),
        actor_rollout_ref=Namespace(actor=actor,rollout=Namespace(n=8)),algorithm=Namespace(use_kl_in_reward=False))
    return config,manifest


@pytest.mark.parametrize(('value','expected'),[(None,1701),(0,0),(1701,1701)])
def test_nullable_upstream_seed_uses_default_and_preserves_explicit_zero(value,expected):
    cfg,manifest=inputs(value);state,metadata=configuration(cfg,manifest,model_revision='warm496')
    assert state.rng.getstate()==random.Random(expected).getstate()
    assert metadata['retained_groups']==128 and metadata['rollouts_per_group']==8
    assert metadata['model_revision']=='warm496' and metadata['router_policy']=='frozen_expert_selection_router'
    assert metadata['score_centering'] and sum(metadata['family_quotas'].values())==128


def test_wrong_training_batch_is_rejected_before_any_model_is_initialized():
    cfg,manifest=inputs(None);cfg.data['train_batch_size']=64
    with pytest.raises(ValueError,match='128x8'):configuration(cfg,manifest)
