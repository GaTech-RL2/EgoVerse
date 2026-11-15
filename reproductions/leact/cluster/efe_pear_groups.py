"""Admission contract for native EFE RL groups, before any policy update."""
try:
    from .efe_pear_loss import group_advantages
except ImportError:
    from efe_pear_loss import group_advantages


def admit_group(episodes, membership, policy_revision, expected_size=8):
    if len(episodes)!=expected_size or len({e['episode_id'] for e in episodes})!=expected_size:
        raise ValueError('Missing or duplicate native episode in group')
    identity={(e['task'],e['scene_sha256'],e['initial_state_sha256']) for e in episodes}
    if len(identity)!=1:
        raise ValueError('Group must share one native task, physical scene and reset')
    task,scene,initial=next(iter(identity))
    expected={'scene_sha256':scene,'initial_state_sha256':initial}
    if expected not in membership['training_tasks'].get(task,[]):
        raise ValueError('Episode is outside frozen warmstart training membership')
    for episode in episodes:
        if episode.get('partition')!='train_candidates' or episode.get('evaluation_only'):
            raise ValueError('Held-out or evaluation episode cannot enter RL')
        if episode.get('status')!='graded' or type(episode.get('reward')) not in (int,float) or episode['reward'] not in (0,1):
            raise ValueError('Native infrastructure failures cannot be converted to zero reward')
        if episode.get('sampler_revision')!=policy_revision:
            raise ValueError('The synchronous pilot requires the exact sampling policy revision')
        if not episode.get('sampling_capture_complete') or episode.get('generated_tokens',0)<1:
            raise ValueError('Missing sampler probabilities or generated tokens')
        if not episode.get('native_events_verified'):
            raise ValueError('The native reset and grader evidence must be verified')
    rewards=[e['reward'] for e in episodes]
    advantages=group_advantages(rewards)
    return {'admitted':advantages is not None,'filter_reason':None if advantages is not None else 'zero_reward_variance',
            'task':task,'episode_ids':[e['episode_id'] for e in episodes],'rewards':rewards,'advantages':advantages,
            'native_rollouts_spent':len(episodes),'successes':sum(rewards),
            'generated_tokens_spent':sum(e['generated_tokens'] for e in episodes),
            'truncated_episodes':sum(e.get('output_truncated',False) for e in episodes)}
