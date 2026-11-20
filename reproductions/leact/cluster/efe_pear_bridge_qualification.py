"""One native transport/action/verifier check, with no model inference or RL."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import traceback
from uuid import uuid4

from efe_pear_bridge import HarborEpisode, observation_payload
from efe_action_loop import parse_action


def main():
    import efe_native_pilot as native
    from scene_identity import scene_identity
    lane=Path(__file__).resolve().parent
    spec=json.loads((lane/'spec.json').read_text())
    native.ROOT=Path('/tmp/efe-native')
    native.ARCHIVE=Path(spec['archive_root'])/os.environ['EFE_WORKFLOW_ID']
    native.ARCHIVE.mkdir(parents=True,exist_ok=True)
    row=spec['tasks'][0];episode=None;service=None;log=None
    result={'primary_rl_updates':0,'model_inference_calls':0,'policy_success_evaluation':False,
            'scope':'Native bridge qualification only; scripted transport probe is never training data.'}
    try:
        reference,env,service,log=native.setup(spec)
        scene=scene_identity(reference,row)
        if scene['scene_sha256']!=spec['scene_sha256']:
            raise ValueError('Previously audited training scene changed')
        config={'reference':str(reference),'episode_root':str(native.ARCHIVE/'episodes'),
                'agent_source':str(lane),'harbor':shutil.which('harbor',path=env['PATH']),
                'action_config':spec['fixed_action_config'],'container_environment':env}
        episode=HarborEpisode(row,config,uuid4().hex)
        first=episode.start()
        if first['done']:raise ValueError('Native episode ended before observation')
        visible,_=observation_payload(first['body']);contract=visible['action_contract']
        kinds=sorted(contract['action_types'],key=lambda k:('delta' not in k,k))
        chosen=None
        for kind in kinds:
            a=contract['action_types'][kind]
            if all(lo<=0<=hi for lo,hi in zip(a['lower'],a['upper'])):
                chosen={'action_type':kind,'action':[0]*a['dim'],'repeat':1,
                        'summary':'One native API transport check, not a policy rollout.'}
                break
        if chosen is None:raise ValueError('No zero-valued valid action available for the transport probe')
        parse_action(json.dumps(chosen),contract=contract,allow_stop=True)
        next_result=episode.step(chosen)
        if not next_result['done']:
            visible2,_=observation_payload(next_result['body'])
            feedback=visible2['recent_actions_and_feedback'][-1]
            if not feedback.get('execution_completed'):raise ValueError('Native action was not executed')
            if feedback['action']!=chosen:raise ValueError('Bridge changed the policy action')
            final=episode.step({'stop':True,'summary':'End native API transport qualification.'})
        else:final=next_result
        if not final['done']:raise ValueError('Native trial did not finalize')
        result.update(success=True,native_reward=final['reward'],native_trial_path=final['native_trial_path'],
            scene=scene,action_contract_sha256=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest(),
            action_types=list(contract['action_types']),native_action=chosen,public_observations=episode.calls)
    except Exception as exc:
        result.update(success=False,error=repr(exc),traceback=traceback.format_exc())
        raise
    finally:
        (native.ARCHIVE/'bridge-qualification.json').write_text(json.dumps(result,indent=2)+'\n')
        print('PEAR_BRIDGE_QUALIFICATION '+json.dumps(result),flush=True)
        if episode:episode.close()
        if service is not None:
            service.terminate()
            try:service.wait(timeout=10)
            except Exception:service.kill();service.wait()
        if log:log.close()


if __name__=='__main__':main()
