"""Export the user-selected update 496 and enqueue its frozen native panel."""
import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import yaml

from .scale_efe import BASE,BUILD_ROOT,read,write,submit,command
from .launch_efe_training import freeze_sources
from .launch_efe_coverage import DEST as COVERAGE,DURABLE,infra_retryable
from .efe_coverage_retry import retry_due

STEP=496
MODEL='warmstart-step496'
DEST=COVERAGE/'intermediate-496'
BUILD=BUILD_ROOT/'efe-intermediate-export-step496-20261116-v1'
LANE='/tmp/efe-intermediate-export'


def now():return datetime.now(timezone.utc).isoformat()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def worker_logs(workflow):
    # OSMO follows live logs even with -n. Keep a bounded snapshot, including
    # bytes already received when the still-running stream times out.
    try:
        result=subprocess.run(['osmo','workflow','logs',workflow,'-n','100'],
            capture_output=True,text=True,timeout=8)
        return result.stdout if result.returncode==0 else ''
    except subprocess.TimeoutExpired as error:
        value=error.stdout or ''
        return value.decode(errors='replace') if isinstance(value,bytes) else value


def final_receipt(logs,step,pin_sha256):
    values=[]
    for line in logs.splitlines():
        marker='EFE_INTERMEDIATE_RESULT '
        if marker not in line:continue
        try:row=json.loads(line.split(marker,1)[1])
        except json.JSONDecodeError:continue
        if isinstance(row,dict):values.append(row)
    if not values:return None
    row=values[-1];ready=row.get('ready',{})
    replay=ready.get('loss_replay',{})
    if (row.get('success') is True and row.get('step')==step and row.get('additional_optimizer_updates')==0
        and ready.get('step')==step and ready.get('final') is False and ready.get('pin_sha256')==pin_sha256
        and ready.get('additional_optimizer_updates')==0 and replay.get('passed') is True
        and replay.get('optimizer_updates')==0 and isinstance(replay.get('absolute_error'),(int,float))
        and 0<=replay['absolute_error']<=.0005 and ready.get('checkpoint') and ready.get('manifest') and ready.get('revision')):
        return row
    return None


def publish(state,ready=None):
    write(DEST/'status.json',state)
    path=COVERAGE/'intermediate-evaluations.json'
    with (COVERAGE/'intermediate-evaluations.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        registry=read(path,{'runs':[]})
        entry=next((r for r in registry['runs'] if r['id']==MODEL),None)
        if entry is None:
            entry={'id':MODEL,'label':'Expanded warmstart · update 496','step':STEP,'baseline':'base',
                'selection':'User-requested intermediate diagnostic; final fixed-budget comparison remains separate.',
                'panel_sha256':sha(COVERAGE/'evaluation-panel.json'),'planned_tasks':36,
                'pin_sha256':sha(DEST/'checkpoint-pin.json')}
            registry['runs'].append(entry)
        entry['export']=state
        if ready is not None:entry['ready']=ready
        write(path,registry)
    from ..embodied_first_exam import report
    report.publish()


def prepare():
    pin=read(DEST/'checkpoint-pin.json')
    if pin['step']!=STEP or pin['final'] is not False:raise ValueError('Requested intermediate update must be pinned first')
    if (BUILD/'source-manifest.json').exists():return BUILD
    BUILD.mkdir(parents=True,exist_ok=True)
    mapping={'efe_intermediate_export_worker.py':'worker.py','efe_qwen36_export_recovery.py':'verify.py',
        'efe_train_qwen36_lora.py':'trainer.py','efe_merge_qwen36_warm.py':'merge.py'}
    names=['efe_training_data.py','efe_budget_gateway.py','efe_visual_tensor_store.py','efe_model_cache.py',
        'efe_qwen36_adapter_io.py','efe_training_state.py','efe_qwen36_restart.py','efe_qwen36_health.py']
    mapping.update({n:n for n in names})
    for source,target in mapping.items():shutil.copyfile(BASE/'cluster'/source,BUILD/target)
    spec={'step':STEP,'archive_root':str(Path(pin['checkpoint']).parent),
        'pin_sha256':sha(DEST/'checkpoint-pin.json'),'base_manifest':DURABLE+'/base-checkpoint-manifest.json',
        'additional_optimizer_updates':0,'evaluation_model':MODEL,'scope':'User-requested intermediate native evaluation on the unchanged 36-task panel.'}
    write(BUILD/'spec.json',spec)
    training=read(COVERAGE/'training-queue.json')['stages']['warmstart']['attempts'][-1]
    workflow=yaml.safe_load((Path(training['build'])/'workflow.yaml').read_text())
    workflow['workflow']['name']='leact-efe-intermediate-export-step496-v1'
    workflow['workflow']['timeout']={'queue_timeout':'20m','exec_timeout':'3h'}
    task=workflow['workflow']['tasks'][0];task.pop('credentials',None)
    task.update(command=['python3'],args=['-u',LANE+'/worker.py'])
    task['files']=[{'localpath':n,'path':LANE+'/'+n} for n in [*mapping.values(),'spec.json']]
    (BUILD/'workflow.yaml').write_text(yaml.safe_dump(workflow,sort_keys=False))
    for p in BUILD.glob('*.py'):compile(p.read_text(),str(p),'exec')
    freeze_sources(BUILD)
    write(DEST/'plan.json',{'step':STEP,'build':str(BUILD),'spec':spec,'created_utc':now(),
        'tasks':36,'cohorts':{'seen_training_scene':24,'same_simulator_new_task':12},
        'sequence':['Pinned committed adapter','Exact full-model loss replay','Merge native-serving model','Matched native robot evaluations'],
        'selection':'Intermediate diagnostic. Zero additional optimizer updates; final warmstart continues independently.'})
    publish({'phase':'prepared','step':STEP,'updated_utc':now(),'summary':'Update 496 is pinned; native evaluation waits for exact replay and export.'})
    return BUILD


def tick():
    prepare();q=read(DEST/'queue.json',{'attempts':[]})
    if not q['attempts']:
        receipt=submit(BUILD,priority='HIGH')
        q['attempts'].append({'workflow':receipt['name'],'build':str(BUILD),'submitted_utc':now(),'priority':'HIGH'})
        write(DEST/'queue.json',q)
    a=q['attempts'][-1];scheduler=command(['osmo','workflow','query',a['workflow'],'--format-type','json'])
    write(Path(a['build'])/'scheduler.json',scheduler)
    state={'workflow':a['workflow'],'scheduler_status':scheduler['status'],'phase':scheduler['status'].lower(),
        'step':STEP,'updated_utc':now(),'additional_optimizer_updates':0}
    logs=worker_logs(a['workflow'])
    result=None
    if logs:
        (DEST/'latest-worker.log').write_text(logs)
        result=final_receipt(logs,STEP,sha(DEST/'checkpoint-pin.json'))
        for line in logs.splitlines():
            marker='EFE_INTERMEDIATE_PROGRESS '
            if marker not in line:continue
            try:progress=json.loads(line.split(marker,1)[1])
            except json.JSONDecodeError:continue
            if progress.get('workflow')==a['workflow']:state['progress']=progress;state['phase']=progress['phase']
    if result:
        write(DEST/'result.json',result)
        state.update(phase='ready_for_native_evaluation',result=result,
            summary='Checkpoint 496 passed exact loss replay and native-model export; the frozen 36-task panel is released to the evaluator.')
        publish(state,result['ready']);return True
    if scheduler['status'].startswith('FAILED'):
        cuda_start=('torch.cuda.set_device(local)' in logs and
            'CUDA error: CUDA-capable device(s) is/are busy or unavailable' in logs)
        if infra_retryable(scheduler) or cuda_start:
            state['phase']='retry_backoff'
            if retry_due(q,a['workflow']):
                folder=BUILD.parent/(BUILD.name+'-retry-'+str(len(q['attempts'])))
                if not (folder/'source-manifest.json').exists():
                    folder.mkdir(exist_ok=True)
                    for name in read(BUILD/'source-manifest.json'):shutil.copyfile(BUILD/name,folder/name)
                    spec=yaml.safe_load((folder/'workflow.yaml').read_text())
                    spec['workflow']['name']='leact-efe-intermediate-export-step496-v1-r'+str(len(q['attempts']))
                    (folder/'workflow.yaml').write_text(yaml.safe_dump(spec,sort_keys=False));freeze_sources(folder)
                receipt=submit(folder,priority='LOW')
                q['attempts'].append({'workflow':receipt['name'],'build':str(folder),'submitted_utc':now(),'priority':'LOW'})
                state.update(workflow=receipt['name'],scheduler_status='PENDING',phase='pending')
            write(DEST/'queue.json',q)
        else:state['phase']='needs_review'
    elif scheduler['status']=='COMPLETED':state['phase']='awaiting_terminal_receipt'
    state['summary']='Checkpoint 496 intermediate evaluation: '+state['phase'].replace('_',' ')+'. The original 36-task panel and baseline budgets are unchanged.'
    publish(state);return False


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','submit','tick','monitor']);a=p.parse_args()
    if a.action=='prepare':print(prepare());return
    if a.action=='submit':
        tick()
        with (DEST/'observer.log').open('ab',buffering=0) as out:
            child=subprocess.Popen([sys.executable,'-S','-u','-m',__package__+'.launch_efe_intermediate_eval','monitor'],
                cwd=BASE.parents[1],stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
        write(DEST/'observer-launch.json',{'pid':child.pid,'utc':now()})
        print(json.dumps(read(DEST/'queue.json')['attempts'][-1]));return
    with (DEST/'monitor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            try:
                complete=tick()
                if complete or a.action=='tick':return
            except Exception as error:
                write(DEST/'observer-error.json',{'utc':now(),'error':repr(error)})
                if a.action=='tick':raise
            time.sleep(45)


if __name__=='__main__':main()
