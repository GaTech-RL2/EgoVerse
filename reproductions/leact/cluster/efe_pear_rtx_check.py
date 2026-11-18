"""Check pinned native RTX tasks through the actual production file transport."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

from efe_pear_catalog_probe import probe_action
from efe_pear_file_rpc import FileEpisodeClient,claim_episode,serve_episode
from efe_pear_rtx_env import NativeRtxEpisode


def main():
    lane=Path(__file__).resolve().parent;spec=json.loads((lane/'spec.json').read_text())
    root=Path(spec['output'])/os.environ['EFE_WORKFLOW_ID'];root.mkdir(parents=True,exist_ok=True)
    archive=Path(spec['payload']);digest=hashlib.sha256()
    with archive.open('rb')as stream:
        while block:=stream.read(8<<20):digest.update(block)
    if digest.hexdigest()!=spec['payload_sha256']:
        raise ValueError('Native RTX task payload changed')
    source=root/'native-inputs';source.mkdir()
    with tarfile.open(archive)as stream:stream.extractall(source,filter='data')
    for name,expected in spec['files'].items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('Native RTX service input changed')
    names=subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],text=True).splitlines()
    if len(names)!=1 or not any(k in names[0]for k in ['L40','RTX','L4']):
        raise ValueError('One allocated RTX GPU is required for native camera qualification')
    sys.path.insert(0,str(source/'standard'));results={}
    for row in spec['tasks']:
        task=root/row['task'];task.mkdir();transport=task/'transport';transport.mkdir(mode=0o700)
        config={'source':str(source),'suite':spec['suite'],'native_tool_relative':spec['native_tool_relative'],
            'episode_root':str(task/'episodes'),'qualification_only':True,
            'training_tasks':{row['task']:{'qualification':'checking','row':row,
                'scene_sha256':spec['expected_scenes'][row['task']]}}}
        value={'task':row['task'],'row':row,'scene_sha256':spec['expected_scenes'][row['task']],
            'native_rtx_api_passed':False,'training_admitted':False,'primary_rl_updates':0,
            'positive_solver_control':False,'benchmark_resource_enforced':False}
        client=FileEpisodeClient(transport,'efe/'+spec['suite'],{'task':row['task'],'seed':row['seeds']['default']},timeout=1800)
        def server():
            deadline=time.monotonic()+1800
            while time.monotonic()<deadline:
                claimed=claim_episode(transport,'native-rtx-qualification',kinds={'efe/'+spec['suite']})
                if claimed:
                    folder,request=claimed
                    return serve_episode(folder,lambda **args:NativeRtxEpisode(config=config,**args),request['env_args'])
                time.sleep(.1)
            raise TimeoutError('No native qualification request arrived')
        try:
            with ThreadPoolExecutor(max_workers=1)as executor:
                future=executor.submit(server)
                try:
                    initial=client.open()
                    if initial['done']or not initial['images']:
                        raise ValueError('Native reset lacks usable public RGB')
                    for index,image in enumerate(initial['images']):image.save(task/f'reset-camera-{index}.png')
                    action=probe_action(initial['action_contract'])
                    after=client.step(action)
                    if after['done']or after['executed_control_steps']!=action['repeat']:
                        raise ValueError('Native action execution accounting failed')
                    for index,image in enumerate(after['images']):image.save(task/f'action-camera-{index}.png')
                    final=client.step({'stop':True,'summary':'End discarded simulator API check.'})
                    if not final['done']or final['rewards']!=[0.]:
                        raise ValueError('Native negative control has no finalized failure verdict')
                    value.update(native_rtx_api_passed=True,native_file_transport_passed=True,
                        action_contract=initial['action_contract'],native_action=action,
                        native_final=json.loads(Path(final['native_trial_path']).read_text()),
                        interface_context_version=initial['interface_context_version'],
                        interface_context_sha256=initial['interface_context_sha256'])
                finally:client.close()
                future.result(timeout=60)
            # Where upstream supplies a recorded oracle, exercise it on a
            # separate fresh scene. These controls never become policy data.
            if row.get('oracle')=='full'and spec.get('positive_oracles',True):
                oracle=NativeRtxEpisode(task=row['task'],seed=row['seeds']['default'],config=config)
                try:
                    oracle.open();path=source/row['path']/'solution/oracle_replay.py'
                    if not path.is_file():raise ValueError('Advertised recorded oracle is missing')
                    env={**oracle.env,'PYTHONPATH':str(source/'standard')+':'+oracle.env['PYTHONPATH']}
                    with(task/'positive-oracle.log').open('w')as log:
                        run=subprocess.run([sys.executable,'-u',str(path)],env=env,cwd=path.parent,
                            stdout=log,stderr=subprocess.STDOUT,timeout=1800)
                    positive=oracle._finalize()
                    if run.returncode or positive['rewards']!=[1.]:
                        raise ValueError('The upstream recorded oracle did not receive native success')
                    value.update(positive_solver_control=True,
                        positive_native_final=json.loads(Path(positive['native_trial_path']).read_text()))
                finally:oracle.close()
        except Exception as exc:
            value.update(native_rtx_api_passed=False,error=repr(exc))
        finally:client.close()
        results[row['task']]=value
        (root/'checks.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
        print('PEAR_NATIVE_RTX_CHECK '+json.dumps({'checked':len(results),'total':len(spec['tasks']),
            'passed':sum(v['native_rtx_api_passed']for v in results.values()),'primary_rl_updates':0,
            'training_admitted':False}),flush=True)
    if not all(v['native_rtx_api_passed']for v in results.values()):
        raise RuntimeError('One or more native RTX production transport checks failed')


if __name__=='__main__':main()
