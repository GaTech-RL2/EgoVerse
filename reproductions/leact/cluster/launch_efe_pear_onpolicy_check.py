"""Submit the discarded, two-node MoE score-centering qualification."""
import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

from .launch_efe_pear_scaled import DEST,REMOTE,TEMPLATE,PEAR_REVISION
from .launch_efe_training import freeze_sources
from .scale_efe import BASE,BUILD_ROOT,read,write,submit


def prepare():
    control=read(DEST/'control.json',{})
    if control.get('paused')or'moe-warmstart496-pearplusplus'not in control.get('enabled_training_arms',[]):
        raise RuntimeError('Warmstart preparation is paused')
    model=next(m for m in read(DEST/'model-files-ready.json')['models']if m['kind']=='moe_warmstart496')
    receipt=read(DEST/'warmstart-reasoning-check.json')
    if receipt['worker'].get('passed')is not True:raise RuntimeError('Reasoning transport is not qualified')
    worker=receipt['worker'];membership=worker['membership_file']
    # The remote file was hash-verified when the bank's qualification completed.
    membership_sha=read(DEST/'reasoning-qualified-membership.json',{}).get('sha256')
    if not membership_sha:raise RuntimeError('Missing exact qualified remote membership checksum')
    names=['efe_pear_onpolicy_check.py','efe_pear_onpolicy_check.sh','efe_pear_optimizer_check.py',
        'efe_pear_moe_kernels.py','efe_pear_router.py','efe_pear_reasoning_env.py','efe_pear_reasoning_scenes.py',
        'efe_pear_checkpoint_archive.py','efe_pear_agent_metrics.py','efe_pear_agent_metrics.yaml',
        'efe_pear_telemetry.py','efe_pear_curriculum.py']
    digests={n:hashlib.sha256((BASE/'cluster'/n).read_bytes()).hexdigest()for n in names}
    worker_name='efe_pear_ray_check_worker.py'
    spec={'remote_root':REMOTE,'output':REMOTE+'/onpolicy-checks','pear_source':REMOTE+'/sources/pear-'+PEAR_REVISION,
        'model':model['path'],'model_revision':model['revision'],'reasoning_membership':membership,
        'reasoning_membership_sha256':membership_sha,'adapter_digests':digests,
        'qualification_only':True,'primary_rl_updates':0,'discarded_retained_groups':8,'rollouts_per_group':8,
        'maximum_judged_groups':128,'response_limit':8192,'primary_recipe_response_limit':160000,
        'worker_sha256':hashlib.sha256((BASE/'cluster'/worker_name).read_bytes()).hexdigest()}
    key=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()[:12]
    build=BUILD_ROOT/('rp-jhh-pear-onpolicy-check-'+key)
    if(build/'source-manifest.json').exists():return build
    build.mkdir(exist_ok=True);write(build/'spec.json',spec)
    for name in names:shutil.copyfile(BASE/'cluster'/name,build/name)
    shutil.copyfile(BASE/'cluster'/worker_name,build/'worker.py')
    original=yaml.safe_load(TEMPLATE.read_text())['workflow'];task_source=original['tasks'][0]
    tasks=[]
    for role in ['trainer','rollout']:
        task={'name':role,'resource':role,'image':task_source['image'],'command':['python3'],
            'args':['-u','/tmp/rp-jhh-onpolicy-check/worker.py'],
            'environment':{'EFE_ROLE':role,'EFE_WORKFLOW_ID':'{{workflow_id}}','EFE_HEAD_HOST':'{{host:trainer}}',
                'PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1','NVIDIA_DRIVER_CAPABILITIES':'all'},
            'files':[{'localpath':p.name,'path':'/tmp/rp-jhh-onpolicy-check/'+p.name}
                for p in sorted(build.iterdir())if p.is_file()]}
        if role=='trainer':task['lead']=True
        tasks.append(task)
    workflow={'workflow':{'name':'leact-'+build.name,'timeout':{'queue_timeout':'20m','exec_timeout':'4h'},
        'labels':{'pear_run_id':'moe-warmstart496-pearplusplus','pear_phase':'on_policy_score_centering_qualification'},
        'resources':{role:{'gpu':8,'cpu':64,'memory':'1536Gi'if role=='trainer'else'512Gi',
            'storage':'250Gi','platform':'dgx-h100'}for role in ['trainer','rollout']},
        'groups':[{'name':'qualification','tasks':tasks}]}}
    (build/'workflow.yaml').write_text(yaml.safe_dump(workflow,sort_keys=False));freeze_sources(build)
    return build


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--submit',action='store_true')
    parser.add_argument('--watch',action='store_true')
    parser.add_argument('--priority',choices=['HIGH','LOW'],default='HIGH');args=parser.parse_args()
    if args.watch:
        watch();return
    build=prepare()
    if args.submit:
        receipt=submit(build,priority=args.priority,pool='groot-h100-02',nodes=2)
        launch={'workflow':receipt['name'],'build':str(build),'primary_rl_updates':0,
                'qualification_only':True,'gpu':16,'priority':args.priority}
        write(DEST/'onpolicy-check-launch.json',launch);print(json.dumps(launch),flush=True)
        with(DEST/'onpolicy-check-observer.log').open('ab',buffering=0)as log:
            child=subprocess.Popen([sys.executable,'-u','-m',__package__+'.launch_efe_pear_onpolicy_check','--watch'],
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(DEST/'onpolicy-check-observer.json',{'pid':child.pid,'scope':'Qualification monitoring; allocation failures retry at LOW priority'})
    else:print(build)


def watch():
    from .scale_efe import command
    from .launch_efe_coverage import infra_retryable
    with(DEST/'onpolicy-check-watch.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        deadline=time.monotonic()+6*3600
        while time.monotonic()<deadline:
            launch=read(DEST/'onpolicy-check-launch.json')
            try:workflow=command(['osmo','workflow','query',launch['workflow'],'--format-type','json'])
            except Exception as exc:
                write(DEST/'onpolicy-check-observer-error.json',{'error':str(exc)[-500:],
                    'updated_utc':datetime.now(timezone.utc).isoformat(),'primary_rl_updates':0})
                time.sleep(30);continue
            status={**launch,'scheduler_status':workflow['status'],'updated_utc':datetime.now(timezone.utc).isoformat()}
            previous=read(DEST/'onpolicy-check-status.json',{})
            if previous.get('workflow')==launch['workflow']and previous.get('worker'):status['worker']=previous['worker']
            if workflow.get('start_time'):
                try:
                    logs=subprocess.run(['osmo','workflow','logs',launch['workflow'],'--task','trainer','-n','80'],
                        capture_output=True,text=True,timeout=6).stdout
                except subprocess.TimeoutExpired as exc:
                    logs=exc.stdout or'';logs=logs.decode(errors='replace')if isinstance(logs,bytes)else logs
                for line in logs.splitlines():
                    if'PEAR_ONPOLICY_CHECK 'in line:
                        try:status['worker']=json.JSONDecoder().raw_decode(line.split('PEAR_ONPOLICY_CHECK ',1)[1])[0]
                        except ValueError:pass
            write(DEST/'onpolicy-check-status.json',status)
            if workflow['status']in {'COMPLETED','CANCELED','CANCELLED'}:return
            if workflow['status'].startswith('FAILED'):
                attempt=launch.get('attempt',0)
                if not infra_retryable(workflow)or attempt>=6:return
                original=Path(launch['build']);target=BUILD_ROOT/(original.name.split('-allocation-retry-')[0]+f'-allocation-retry-{attempt+1}')
                target.mkdir(exist_ok=True)
                for name,digest in read(original/'source-manifest.json').items():
                    if hashlib.sha256((original/name).read_bytes()).hexdigest()!=digest:raise ValueError('Frozen qualification source changed')
                    shutil.copyfile(original/name,target/name)
                frozen=yaml.safe_load((target/'workflow.yaml').read_text());frozen['workflow']['name']='leact-'+target.name
                (target/'workflow.yaml').write_text(yaml.safe_dump(frozen,sort_keys=False));freeze_sources(target)
                receipt=submit(target,priority='LOW',pool='groot-h100-02',nodes=2)
                write(DEST/'onpolicy-check-launch.json',{**launch,'workflow':receipt['name'],'build':str(target),
                    'priority':'LOW','attempt':attempt+1})
            time.sleep(30)


if __name__=='__main__':main()
