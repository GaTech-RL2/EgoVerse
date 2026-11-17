"""Pin, verify and evaluate R1 update 96 while its full fixed-budget run continues."""
import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import yaml

from .scale_efe import BASE,BUILD_ROOT,command,read,write,submit
from .launch_efe_training import freeze_sources
from .launch_efe_coverage import DEST as COVERAGE,infra_retryable
from .efe_coverage_retry import retry_due,submission_priority

STEP=96
MODEL='leact-from-step496-r1-step96'
DEST=COVERAGE/f'intermediate-{MODEL}'


def now():return datetime.now(timezone.utc).isoformat()


def prepare_pin(index=0):
    branch=next(r for r in read(COVERAGE/'branch-runs.json')['runs'] if r['id']=='leact-from-step496-r1')
    if not branch.get('authorized'):raise ValueError('R1 branch is not authorized')
    folder=BUILD_ROOT/f'efe-leact-r1-step96-pin-20261117-v1-r{index}'
    if (folder/'source-manifest.json').exists():return folder
    folder.mkdir(parents=True,exist_ok=True)
    destination=str(Path(branch['archive_root']).parent/'intermediate/rp_jhh_leact_from_step496_r1_step96')
    spec={'training_root':branch['archive_root'],'destination':destination,'step':STEP,
        'selection':'Declared before R1 optimization: fixed update 96 early readout on the frozen panel. Full R1 training continues; no checkpoint selection by evaluation performance.',
        'timeout_seconds':24*3600,'optimizer_updates_by_watcher':0}
    write(folder/'spec.json',spec)
    for source,target in [('efe_leact_milestone_pin.py','worker.py'),('efe_intermediate_checkpoint.py','efe_intermediate_checkpoint.py')]:
        shutil.copyfile(BASE/'cluster'/source,folder/target)
    source=Path(branch['build'])/'workflow.yaml';workflow=yaml.safe_load(source.read_text())
    workflow['workflow']['name']=f'leact-efe-r1-step96-pin-v1-r{index}'
    workflow['workflow']['timeout']={'queue_timeout':'20m','exec_timeout':'25h'}
    workflow['workflow']['resources']['default'].update(gpu=0,cpu=2,memory='8Gi',storage='40Gi',platform='ovx-l40')
    task=workflow['workflow']['tasks'][0];task.pop('credentials',None);task.pop('privileged',None)
    task.update(command=['python3'],args=['-u','/tmp/efe-r1-pin/worker.py'])
    task['files']=[{'localpath':n,'path':'/tmp/efe-r1-pin/'+n} for n in ['spec.json','worker.py','efe_intermediate_checkpoint.py']]
    (folder/'workflow.yaml').write_text(yaml.safe_dump(workflow,sort_keys=False));freeze_sources(folder)
    return folder


def pin_receipt(logs):
    result=None
    for line in logs.splitlines():
        if 'EFE_MILESTONE_PIN ' not in line:continue
        try:result=json.loads(line.split('EFE_MILESTONE_PIN ',1)[1])
        except json.JSONDecodeError:continue
    if not result:return None
    if (result.get('success') is not True or result.get('step')!=STEP
        or result.get('optimizer_updates_by_watcher')!=0):raise ValueError('Unexpected pin result')
    raw=result['pin_json'];pin=json.loads(raw)
    if hashlib.sha256(raw.encode()).hexdigest()!=result['pin_sha256']:
        raise ValueError('Pin receipt bytes changed')
    if pin.get('step')!=STEP or pin.get('final') is not False or pin.get('additional_optimizer_updates')!=0:
        raise ValueError('Unexpected intermediate checkpoint')
    return result


def tick():
    DEST.mkdir(parents=True,exist_ok=True)
    if not (DEST/'checkpoint-pin.json').exists():
        queue=read(DEST/'pin-queue.json',{'attempts':[]})
        if not queue['attempts']:
            folder=prepare_pin();receipt=submit(folder,priority='HIGH',pool='groot-l40-03')
            queue['attempts'].append({'workflow':receipt['name'],'build':str(folder),'priority':'HIGH','submitted_utc':now()})
            write(DEST/'pin-queue.json',queue)
        attempt=queue['attempts'][-1]
        scheduler=command(['osmo','workflow','query',attempt['workflow'],'--format-type','json'])
        write(Path(attempt['build'])/'scheduler.json',scheduler)
        state={'phase':'waiting_for_R1_update_96','step':STEP,'workflow':attempt['workflow'],
            'scheduler_status':scheduler['status'],'updated_utc':now(),'optimizer_updates_by_watcher':0}
        from .launch_efe_intermediate_eval import worker_logs
        result=pin_receipt(worker_logs(attempt['workflow']))
        if result:
            (DEST/'checkpoint-pin.json').write_text(result['pin_json'])
            write(DEST/'pin-result.json',result)
        elif scheduler['status'].startswith('FAILED'):
            if infra_retryable(scheduler):
                if retry_due(queue,attempt['workflow']):
                    folder=prepare_pin(len(queue['attempts']));priority=submission_priority(queue)
                    receipt=submit(folder,priority=priority,pool='groot-l40-03')
                    queue['attempts'].append({'workflow':receipt['name'],'build':str(folder),'priority':priority,'submitted_utc':now()})
                    state.update(workflow=receipt['name'],scheduler_status='PENDING')
                write(DEST/'pin-queue.json',queue)
            else:state['phase']='pin_needs_review'
        write(DEST/'milestone-status.json',state)
        if not result:return False
    from . import launch_efe_intermediate_eval as exporter
    exporter.configure_leact_milestone(STEP)
    return exporter.tick()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','tick','monitor']);args=parser.parse_args()
    DEST.mkdir(parents=True,exist_ok=True)
    if args.action=='prepare':print(prepare_pin());return
    with (DEST/'milestone-monitor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            try:
                done=tick()
                if done or args.action=='tick':return
            except Exception as error:
                write(DEST/'milestone-error.json',{'updated_utc':now(),'error':repr(error)})
                if args.action=='tick':raise
            time.sleep(45)


if __name__=='__main__':main()
