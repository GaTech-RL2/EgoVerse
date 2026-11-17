"""Queue two independent full-model engine checks after shared runtime readiness."""
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import time

import yaml

from .launch_efe_pear_scaled import DEST, REMOTE, PEAR_REVISION, TEMPLATE, now, require_unpaused
from .launch_efe_training import freeze_sources
from .scale_efe import BASE, BUILD_ROOT, command, read, submit, write


ARMS=('dense_base','moe_warmstart496')
VERSION=4


def prepare(arm,index=0):
    require_unpaused()
    build=BUILD_ROOT/f'efe-pear-model-check-20261117-{arm}-v{VERSION}-r{index}'
    if (build/'source-manifest.json').exists():return build
    models=read(DEST/'model-files-ready.json')['models']
    model=next(m for m in models if m['kind']==arm)
    observation=read(DEST/'model-check-observation.json')
    if not observation.get('sha256'):raise ValueError('Frozen public observation is not staged')
    build.mkdir(parents=True,exist_ok=True)
    spec={'model':model['path'],'model_revision':model['revision'],'model_type':model['model_type'],
          'model_repository':'Qwen/Qwen3.8-27B' if arm=='dense_base' else None,
          'observation':observation,'remote_root':REMOTE,'pear_source':REMOTE+'/sources/pear-'+PEAR_REVISION,
          'output':REMOTE+'/model-checks/'+arm,'primary_rl_updates':0,
          'parameter_update':'full_finetuning','lora_rank':0,
          'scope':'One discarded native-image NLL diagnostic; no on-policy PEAR or curriculum qualification.'}
    write(build/'spec.json',spec)
    shutil.copyfile(BASE/'cluster/efe_pear_model_check.py',build/'worker.py')
    shutil.copyfile(BASE/'cluster/efe_pear_bridge.py',build/'efe_pear_bridge.py')
    workflow=yaml.safe_load(TEMPLATE.read_text())
    slug=arm.replace('_','-')
    workflow['workflow']['name']=f'leact-efe-pear-model-check-{slug}-v{VERSION}-r{index}'
    workflow['workflow']['timeout']={'queue_timeout':'20m','exec_timeout':'12h'}
    workflow['workflow']['resources']['default'].update(gpu=8,cpu=48,memory='1024Gi',storage='200Gi')
    task=workflow['workflow']['tasks'][0];task.pop('credentials',None);task.pop('privileged',None)
    task['args']=['-u','/tmp/efe-pear-model-check/worker.py']
    task['files']=[{'localpath':p.name,'path':'/tmp/efe-pear-model-check/'+p.name} for p in build.iterdir() if p.is_file()]
    (build/'workflow.yaml').write_text(yaml.safe_dump(workflow,sort_keys=False));freeze_sources(build)
    return build


def tick():
    if read(DEST/'control.json',{}).get('paused'):
        return True,{'phase':'paused_by_user','primary_rl_updates':0}
    queues=read(DEST/'model-check-queues.json',{})
    if read(DEST/'status.json',{}).get('phase')!='runtime_ready' and not queues:
        result={'phase':'waiting_for_shared_runtime','updated_utc':now(),
                'primary_rl_updates':0,'automatic_next_action':'Submit one 8-H100 full-model engine check for each arm'}
        write(DEST/'model-check-status.json',result)
        return False,result
    from .launch_efe_coverage import infra_retryable
    from .efe_coverage_retry import retry_due
    statuses=[];done=True
    for arm in ARMS:
        queue=queues.setdefault(arm,{'attempts':[]})
        if not queue['attempts']:
            build=prepare(arm);require_unpaused()
            receipt=submit(build,priority='HIGH',pool='groot-h100-02')
            queue['attempts'].append({'workflow':receipt['name'],'build':str(build),
                'pool':'groot-h100-02','priority':'HIGH','submitted_utc':now()})
            write(DEST/'model-check-queues.json',queues)
        attempt=queue['attempts'][-1];name=attempt['workflow']
        scheduler=command(['osmo','workflow','query',name,'--format-type','json'])
        write(Path(attempt['build'])/'scheduler.json',scheduler)
        state={'arm':arm,'workflow':name,'scheduler_status':scheduler['status']}
        terminal=scheduler['status']=='COMPLETED' or scheduler['status'].startswith('FAILED')
        if terminal or scheduler['status']=='RUNNING':
            try:
                logs=subprocess.run(['osmo','workflow','logs',name,'-n','35'],capture_output=True,text=True,timeout=12)
                for line in logs.stdout.splitlines():
                    if 'PEAR_MODEL_CHECK ' in line:
                        try:state['worker']=json.loads(line.split('PEAR_MODEL_CHECK ',1)[1])
                        except json.JSONDecodeError:pass
            except subprocess.TimeoutExpired:state['log_error']='OSMO log request timed out'
        if scheduler['status'].startswith('FAILED') and infra_retryable(scheduler):
            terminal=False
            if retry_due(queue,name):
                build=prepare(arm,len(queue['attempts']));require_unpaused()
                receipt=submit(build,priority='LOW',pool=attempt['pool'])
                queue['attempts'].append({'workflow':receipt['name'],'build':str(build),
                    'pool':attempt['pool'],'priority':'LOW','submitted_utc':now()})
                state.update(workflow=receipt['name'],scheduler_status='PENDING')
        statuses.append(state);done=done and terminal
        write(DEST/'model-check-queues.json',queues)
    result={'phase':'model_checks_finished' if done else 'model_engine_checks',
            'updated_utc':now(),'arms':statuses,'primary_rl_updates':0}
    write(DEST/'model-check-status.json',result)
    return done,result


def monitor():
    from ..embodied_first_exam.report import publish
    with (DEST/'model-check-monitor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            try:
                done,state=tick();print(json.dumps(state),flush=True)
                publish()
                if done:return
            except Exception as exc:print(json.dumps({'utc':now(),'error':repr(exc)}),flush=True)
            time.sleep(45)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['status','monitor']);args=parser.parse_args()
    if args.action=='monitor':monitor()
    else:print(json.dumps(tick()[1],indent=2))
