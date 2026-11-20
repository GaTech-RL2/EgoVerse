"""Collect the native terminal verdict when the RL loop ends an open episode."""
import time

VERSION='native-terminal-verdict-v2'


class MissingNativeVerdict(RuntimeError):pass


async def finalize_episode(client,episode):
    if episode.done:return None
    before=time.monotonic()
    result=await client.step(episode.uid,{'stop':True,'summary':'Finalize the native grade after the RL episode ended.'})
    rewards=result.get('rewards')
    if(result.get('done')is not True or not isinstance(rewards,list)or not rewards
        or any(type(v)not in(int,float)or v not in(0,1)for v in rewards)
        or not result.get('native_trial_path')):
        raise MissingNativeVerdict('Native episode did not return a finalized official verdict')
    if result.get('actions')or result.get('executed',0):
        raise MissingNativeVerdict('Harness finalization unexpectedly executed a policy action')
    # This is environment cleanup, not another sampled model action. Keep
    # actions, chunk count and generated-token masks exactly as recorded.
    episode.done=True;episode.success=episode.is_success(result)
    receipt={'version':VERSION,'uid':episode.uid,'env':episode.env_args,'success':episode.success,
        'native_trial_path':result['native_trial_path'],'native_termination_terms':result.get('native_termination_terms'),
        'harness_finalization':True,'sampled_action_added':False,'seconds':time.monotonic()-before}
    episode.native_finalization=receipt
    return receipt
