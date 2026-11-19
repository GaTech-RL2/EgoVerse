"""Validate custom trainer configuration before allocating model workers."""
try:
    from .efe_pear_curriculum_state import CurriculumState
except ImportError:
    from efe_pear_curriculum_state import CurriculumState


def configuration(config,manifest,*,model_revision=None,workflow=None):
    seed=config.data.get('seed')
    curriculum=CurriculumState(manifest,seed=1701 if seed is None else int(seed))
    if curriculum.stage!='expanded_full':
        raise ValueError('Primary RL requires every authorized control and reasoning family')
    actor=config.actor_rollout_ref.actor
    if(int(config.data.train_batch_size)!=128 or int(config.actor_rollout_ref.rollout.n)!=8 or
        int(actor.ppo_epochs)!=1 or actor.loss_agg_mode!='token-mean'):
        raise ValueError('Curriculum reduction requires one update of the full 128x8 token-mean batch')
    metadata={'model_revision':model_revision,
        'router_policy':'frozen_expert_selection_router'if str(actor.fsdp_config.kernels).endswith('FrozenRouterTrainingKernels')else'not_applicable',
        'retained_groups':int(config.data.train_batch_size),'rollouts_per_group':int(config.actor_rollout_ref.rollout.n),
        'learning_rate':float(actor.optim.lr),'parameter_update':'full_finetuning',
        'ppo_epochs':int(actor.ppo_epochs),
        'score_centering':bool(actor.policy_loss.rollout_correction.score_centering),
        'entropy_coefficient':float(actor.entropy_coeff),
        'kl_penalty':bool(actor.use_kl_loss or config.algorithm.use_kl_in_reward),
        'family_quotas':curriculum.quotas,'curriculum_sha256':curriculum.manifest_sha256,
        'sampling_policy':curriculum.sampling_policy,
        'family_targets_scope':'rolling_window'if curriculum.signal_balancing else'per_optimizer_batch',
        'family_window_updates':8 if curriculum.signal_balancing else 1,
        'sgs_scope':'task_and_reset'if curriculum.signal_balancing else'reset_only',
        'curriculum_stage':curriculum.stage,'workflow':workflow,'sgs_uniform_fraction':.2}
    return curriculum,metadata
