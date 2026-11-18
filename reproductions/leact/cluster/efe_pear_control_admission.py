"""Qualify existing training resets; scripted checks never become policy data."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import traceback
from uuid import uuid4

import efe_native_pilot as native
from efe_action_loop import parse_action
from efe_pear_bridge import HarborEpisode,observation_payload,image_blocks
from efe_pear_observation_check import validate_frames
from scene_identity import scene_identity
from efe_pear_catalog_probe import probe_action


def control(reference,env,output,row,index,agent):
    """Retry only a known daemon-connect failure, with separate native logs."""
    attempts=[]
    for attempt in range(3):
        job=f'control-{index:03d}-{agent}-attempt-{attempt}'
        receipt=native.run(job,['harbor','run','-d','embodied-first-exam@all','-p',row['path'],
            '-a',agent,'-e','podman','-y','-k','1','-n','1','--max-retries','0',
            '--job-name',job,'-o',str(output/'controls')],cwd=reference,env=env,timeout=7200,required=False)
        trials=[json.loads(p.read_text())for p in(output/'controls'/job).glob('*/result.json')]
        trial=native.find_trial_result(trials,row['task'])
        verdict,score=native.control_verdict(agent,trial)if trial else('missing',None)
        attempts.append({'attempt':attempt,'verdict':verdict,'score':score,**receipt})
        if receipt['returncode']==0 and verdict=='passed':return attempts
        if'Podman is not responding'not in receipt['log_tail']or attempt==2:
            raise ValueError(f'{agent} control failed: {verdict}, {score}; '+receipt['log_tail'][-1200:])
        time.sleep(3*(attempt+1))
    raise AssertionError('Unreachable control retry state')


def main():
    lane=Path(__file__).resolve().parent;spec=json.loads((lane/'spec.json').read_text())
    output=Path(spec['archive_root'])/os.environ['EFE_WORKFLOW_ID']
    native.ROOT=Path('/tmp/efe-native');native.ARCHIVE=output;output.mkdir(parents=True,exist_ok=True)
    results={};service=log=None
    if spec.get('resume_checks'):
        prior=Path(spec['resume_checks']['path']);raw=prior.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=spec['resume_checks']['sha256']:
            raise ValueError('Previous native validation receipt changed')
        rows={r['task']:r for r in spec['tasks']}
        for task,value in json.loads(raw).items():
            if value.get('success')or value.get('qualification')=='protected_scene_check_failed':
                if (task not in rows or value['scene_sha256']!=spec['expected_scenes'][task]
                        or value['row']['task_toml_sha256']!=rows[task]['task_toml_sha256']):
                    raise ValueError('Previous validation belongs to a different native task or scene')
                results[task]=value
    def record(phase):
        value={'phase':phase,'updated_utc':datetime.now(timezone.utc).isoformat(),
               'run_id':spec['tracking_run_id'],'primary_rl_updates':0,'checked':len(results),
               'total':len(spec['tasks']),'passed':sum(v['success']for v in results.values()),
               'scope':'Native reset, oracle/nop, public observation and action/verifier qualification only'}
        (output/'status.json').write_text(json.dumps(value)+'\n')
        print('PEAR_ADMISSION_CHECK '+json.dumps(value),flush=True)
    try:
        record('native_setup')
        reference,env,service,log=native.setup(spec)
        if spec.get('enable_gpu_cdi'):
            from efe_pear_gpu_cdi import ensure_gpu_cdi
            ensure_gpu_cdi(native.run)
        if spec.get('prebuild_native_base'):
            from efe_pear_native_image import prepare_base
            image=prepare_base(reference,spec['tasks'][0],spec['prebuild_native_base'],env,native.run)
            (output/'native-base-image.json').write_text(json.dumps(image,indent=2)+'\n')
            record('native_simulator_base_ready')
        shared=output/'reference';shutil.copytree(reference,shared,ignore=shutil.ignore_patterns('.git','__pycache__'))
        config={'reference':str(reference),'episode_root':str(output/'episodes'),'agent_source':str(lane),
                'harbor':shutil.which('harbor',path=env['PATH']),'action_config':spec['fixed_action_config'],
                'container_environment':env}
        def check(index,row):
            result={'task':row['task'],'success':False};episode=None
            try:
                scene=scene_identity(reference,row)
                if scene['scene_sha256']!=spec['expected_scenes'][row['task']]:
                    raise ValueError('Previously admitted warmstart scene changed')
                control_attempts={agent:control(reference,env,output,row,index,agent)for agent in ['nop','oracle']}
                episode=HarborEpisode(row,config,uuid4().hex)
                first=episode.start()
                if first['done']:raise ValueError('Reset ended before an observation')
                visible,content=observation_payload(first['body']);contract=visible['action_contract']
                first_frames=validate_frames([b['image']for b in image_blocks(content)if b['type']=='image'],contract['cameras'])
                chosen=probe_action(contract)
                parse_action(json.dumps(chosen),contract=contract,allow_stop=True)
                observed=episode.step(chosen)
                if observed['done']:raise ValueError('API probe ended before the second observation')
                visible2,content2=observation_payload(observed['body'])
                second_frames=validate_frames([b['image']for b in image_blocks(content2)if b['type']=='image'],contract['cameras'])
                if not visible2['recent_actions_and_feedback'][-1].get('execution_completed'):
                    raise ValueError('Native simulator did not execute the action')
                final=episode.step({'stop':True,'summary':'End discarded API qualification.'})
                if not final['done']or final['reward']!=0:raise ValueError('Reset/API probe lacks the expected nonterminal outcome')
                source_ok=spec.get('source_scene_checks',{}).get(row['task'],True)
                result.update(success=source_ok,native_interface_passed=True,
                    qualification='passed'if source_ok else'protected_scene_check_failed',scene_sha256=scene['scene_sha256'],
                    row=row,action_contract=contract,oracle_success=1,nop_success=0,native_probe_success=0,
                    native_trial_path=final['native_trial_path'],control_attempts=control_attempts,
                    camera_evidence={'reset':first_frames,'after_action':second_frames})
            except Exception as exc:result.update(error=repr(exc),traceback=traceback.format_exc())
            finally:
                if episode:episode.close()
            return result
        with ThreadPoolExecutor(max_workers=spec.get('max_workers',4))as pool:
            for future in as_completed([pool.submit(check,i,row)for i,row in enumerate(spec['tasks'])if row['task']not in results]):
                value=future.result();results[value['task']]=value
                (output/'checks.json').write_text(json.dumps(results,indent=2)+'\n');record('checking_training_resets')
        admitted={key:value for key,value in results.items()if value['success']}
        membership={'scope':'Existing warmstart control-task admission; expanded YAM and reasoning admission remains separate.',
                    'source_revision':spec['revision'],'training_tasks':admitted,
                    'scripted_checks_are_training_data':False,'full_curriculum_qualified':False,
                    'reference':str(shared)}
        (output/'training-membership.json').write_text(json.dumps(membership,indent=2)+'\n')
        record('control_admission_completed')
        if not admitted:raise RuntimeError('No native control task passed admission')
    finally:
        if service:
            service.terminate()
            try:service.wait(timeout=10)
            except Exception:service.kill();service.wait()
        if log:log.close()


if __name__=='__main__':main()
