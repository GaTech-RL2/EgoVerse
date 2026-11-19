"""Compose the exact primary training command on CPU before reserving GPUs."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    lane=Path(__file__).resolve().parent;spec=json.loads((lane/'spec.json').read_text())
    original=spec['primary'];source=Path(original['pear_source']);runtime=Path(original['runtime'])
    previous=Path(original['run']);root=Path(spec['output'])/os.environ['EFE_WORKFLOW_ID']
    root.mkdir(parents=True,exist_ok=True)
    digest=hashlib.sha256((lane/'efe_pear_scaled_train.sh').read_bytes()).hexdigest()
    if digest!=spec['script_sha256']:raise ValueError('Frozen driver command changed')
    print('PEAR_DRIVER_CHECK '+json.dumps({'phase':'installing_system_dependencies','gpu':0}),flush=True)
    with(root/'system-packages.log').open('w')as log:
        subprocess.run(['apt-get','update','-qq'],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
        subprocess.run(['apt-get','install','-y','-qq','git','perl','libosmesa6','libgl1','libglib2.0-0'],
            stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
    checked=subprocess.run(['bash','3rdparty/verl/scripts/checkout_verl.sh','--check'],
        cwd=source,capture_output=True,text=True,check=True,timeout=120)
    (root/'source-check.log').write_text(checked.stdout+checked.stderr)
    print('PEAR_DRIVER_CHECK '+json.dumps({'phase':'source_checked','source_check_passed':True,'gpu':0}),flush=True)
    files=sorted(str(p)for p in previous.glob('train-*.json'))
    if len(files)!=3:raise ValueError('Expected the complete native, YAM and reasoning datasets')
    env={**os.environ,'WORK_DIR':str(runtime),'PEAR_SOURCE':str(source),'MODEL':original['model'],
        'MODEL_REVISION':original['model_revision'],'MODEL_QUALIFICATION':str(previous/'model-qualification.json'),
        'EPISODES':json.dumps(files),'VALIDATION_EPISODES':str(previous/'development.json'),
        'EFE_PEAR_NATIVE_CONFIG':str(previous/'native-config.json'),
        'EFE_PEAR_CURRICULUM':str(previous/'curriculum.json'),'TOOL_CONFIG':str(previous/'tools.yaml'),
        'EFE_PEAR_REASONING_MEMBERSHIP':str(previous/'reasoning-membership.json'),
        'EFE_PEAR_REMOTE_ROOT':str(previous/'transport'),'TRAINING_STEPS':'32','TRAINER_NNODES':'1',
        'PEAR_CHECKPOINT_ARCHIVE':original['checkpoint_archive']+'/cpu-composition-only',
        'PEAR_OBJECTSTORE_CREDENTIAL_FILE':'/dev/null','LOCAL_DIR':str(root/'local'),
        'PYTHONPATH':':'.join(map(str,[lane,source/'pear/utils/sitecustomize',source,source/'3rdparty/verl/src'])),
        'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1','TOKENIZERS_PARALLELISM':'false',
        'PEAR_DRIVER_PREFLIGHT_ONLY':'1'}
    started=time.monotonic()
    policies=['strict-family-v1','task-signal-window-v2']
    for policy in policies:
        curriculum=json.loads((previous/'curriculum.json').read_text());curriculum['sampling_policy']=policy
        curriculum.update(spec.get('curriculum_overrides',{}))
        path=root/(policy+'-curriculum.json');path.write_text(json.dumps(curriculum)+'\n')
        env['EFE_PEAR_CURRICULUM']=str(path)
        for count in [6,4]:
            print('PEAR_DRIVER_CHECK '+json.dumps({'phase':'composing_configuration','rollout_nodes':count,'sampling_policy':policy,'gpu':0}),flush=True)
            env.update(RUN=str(root/f'{policy}-rollout-nodes-{count}'),ROLLOUT_NNODES=str(count))
            logfile=root/f'{policy}-rollout-nodes-{count}.log'
            with logfile.open('w')as log:
                result=subprocess.run(['bash',str(lane/'efe_pear_scaled_train.sh')],
                    cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=300)
            if result.returncode:
                print(logfile.read_text(errors='replace')[-5000:],flush=True)
                raise RuntimeError('Primary training command failed CPU configuration composition')
            content=logfile.read_text(errors='replace')
            records=[json.loads(line.split('PEAR_DRIVER_VALIDATED ',1)[1])for line in content.splitlines()if 'PEAR_DRIVER_VALIDATED 'in line]
            if(len(records)!=1 or not records[0]['custom_trainer_config_passed']
                or records[0]['metadata']['sampling_policy']!=policy):
                raise ValueError('The exact custom trainer configuration was not validated')
    value={'passed':True,'source_check_passed':True,'rollout_nodes':[6,4],
        'script_sha256':digest,'gpu':0,'primary_rl_updates':0,'custom_trainer_config_passed':True,
        'driver_entry_sha256':spec['files']['efe_pear_multitask_train.py'],
        'multitask_config_sha256':spec['files']['efe_pear_multitask_config.py'],
        'curriculum_state_sha256':spec['files']['efe_pear_curriculum_state.py'],
        'signal_curriculum_sha256':spec['files']['efe_pear_signal_curriculum.py'],
        'focus_curriculum_sha256':spec['files']['efe_pear_curriculum_focus.py'],
        'warmup_sampling_checked':bool(spec.get('curriculum_overrides',{}).get('warmup_sampling')),
        'sampling_policies':policies,
        'wall_seconds':time.monotonic()-started,
        'scope':'Exact primary shell, resolved Hydra overrides, worker file paths, full curriculum and custom trainer metadata. No model load, Ray cluster or optimizer update.'}
    (root/'checks.json').write_text(json.dumps(value,indent=2)+'\n')
    print('PEAR_DRIVER_CHECK '+json.dumps(value),flush=True)


if __name__=='__main__':main()
