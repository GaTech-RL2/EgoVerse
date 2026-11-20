"""Inspect native control contracts in parallel campaign shards; no policy or RL."""
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import traceback
from uuid import uuid4

from efe_action_loop import parse_action
from efe_pear_bridge import HarborEpisode, observation_payload


def probe_action(contract):
    """Choose an in-bounds transport test; this is never a training target."""
    for kind in sorted(contract['action_types'],key=lambda k:('delta' not in k,k)):
        spec=contract['action_types'][kind]
        values=[]
        for lower,upper in zip(spec['lower'],spec['upper']):
            if lower>upper:raise ValueError('Invalid action bounds')
            value=min(max(0.,lower),upper)
            if not math.isfinite(value):break
            values.append(value)
        if len(values)==spec['dim']:
            action={'action_type':kind,'action':values,'repeat':1,
                    'summary':'One native contract test; no model inference or training data.'}
            parse_action(json.dumps(action),contract=contract,allow_stop=True)
            return action
    raise ValueError('No finite valid native transport action')


def main():
    import efe_native_pilot as native
    lane=Path(__file__).resolve().parent
    spec=json.loads((lane/'spec.json').read_text())
    native.ROOT=Path('/tmp/efe-native')
    native.ARCHIVE=Path(spec['archive_root'])/os.environ['EFE_WORKFLOW_ID']
    native.ARCHIVE.mkdir(parents=True,exist_ok=True)
    result={'workflow':os.environ['EFE_WORKFLOW_ID'],'primary_rl_updates':0,
            'model_inference_calls':0,'training_admission':False,'tasks':[],
            'scope':'Native action/observation/verifier integration checks only. Scene and model qualification are separate.'}
    service=log=None
    def record():
        result['updated_utc']=datetime.now(timezone.utc).isoformat()
        target=native.ARCHIVE/'catalog-probe.json';temp=target.with_suffix('.tmp')
        temp.write_text(json.dumps(result,indent=2)+'\n');temp.replace(target)
        print('PEAR_CATALOG_PROBE '+json.dumps(result),flush=True)
    try:
        reference,env,service,log=native.setup(spec)
        if spec.get('enable_gpu_cdi'):
            from efe_pear_gpu_cdi import ensure_gpu_cdi
            result['gpu_cdi']=ensure_gpu_cdi(native.run)
            record()
        config={'reference':str(reference),'episode_root':str(native.ARCHIVE/'episodes'),
                'agent_source':str(lane),'harbor':shutil.which('harbor',path=env['PATH']),
                'action_config':spec['fixed_action_config'],'container_environment':env}
        for row in spec['tasks']:
            episode=None
            evidence={'task':row['task'],'suite':row['suite'],'success':False}
            try:
                episode=HarborEpisode(row,config,uuid4().hex)
                first=episode.start()
                if first['done']:raise ValueError('Episode ended before its first observation')
                visible,_=observation_payload(first['body'])
                contract=visible['action_contract'];action=probe_action(contract)
                after=episode.step(action)
                if not after['done']:
                    visible2,_=observation_payload(after['body'])
                    feedback=visible2['recent_actions_and_feedback'][-1]
                    if not feedback.get('execution_completed') or feedback['action']!=action:
                        raise ValueError('Native feedback does not confirm the unchanged requested action')
                    after=episode.step({'stop':True,'summary':'Finish the native contract check.'})
                if not after['done']:raise ValueError('Native grade not finalized')
                evidence.update(success=True,action_contract=contract,
                    action_contract_sha256=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest(),
                    native_reward=after['reward'],native_trial_path=after['native_trial_path'],
                    observations=episode.calls,policy_success_evaluation=False)
            except Exception as exc:
                evidence.update(error=repr(exc),traceback=traceback.format_exc())
            finally:
                if episode:episode.close()
            result['tasks'].append(evidence);record()
        result['success']=len(result['tasks'])==len(spec['tasks']) and all(x['success'] for x in result['tasks'])
    except Exception as exc:
        result.update(success=False,error=repr(exc),traceback=traceback.format_exc())
    finally:
        record()
        if service is not None:
            service.terminate()
            try:service.wait(timeout=10)
            except Exception:service.kill();service.wait()
        if log:log.close()
    if not result.get('success'):raise SystemExit(1)


if __name__=='__main__':main()
