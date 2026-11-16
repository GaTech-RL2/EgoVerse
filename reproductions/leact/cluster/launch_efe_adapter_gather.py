"""Submit and observe the isolated checkpoint-gather qualification."""
import argparse
from datetime import datetime,timezone
import fcntl
import json
from pathlib import Path
import shutil
import time

import yaml

from .scale_efe import BASE,BUILD_ROOT,read,write,submit,command
from .launch_efe_coverage import DEST,infra_retryable
from .launch_efe_training import freeze_sources
from .efe_coverage_retry import retry_due
from .recover_efe_coverage_storage import remote

STATUS=DEST/'checkpoint-gather-status.json'


def prepare(source,receipt):
    source=Path(source);folder=BUILD_ROOT/'efe-checkpoint-gather-20261116-v1'
    if (folder/'source-manifest.json').exists():return folder
    folder.mkdir(exist_ok=True)
    for name in read(source/'source-manifest.json'):
        shutil.copyfile(source/name,folder/name)
    for name,target in [('efe_qwen36_restart.py','efe_qwen36_restart.py'),
                        ('efe_qwen36_restart_probe.py','probe.py'),
                        ('efe_qwen36_export_recovery.py','verify.py'),
                        ('efe_adapter_gather_qualification.py','qualification.py')]:
        shutil.copyfile(BASE/'cluster'/name,folder/target)
    spec=read(source/'spec.json')
    spec.update(qualification_root=spec['coverage_root']+'/performance/checkpoint-gather-20261116-v1',
        qualification_step=receipt['step'],qualification_receipt=receipt,
        qualification_dataset_sha256=read(DEST/'evidence.json')['training']['warmstart']['dataset']['sha256'])
    write(folder/'spec.json',spec)
    bootstrap=(folder/'bootstrap.py').read_text()
    old="str(lane/'pipeline.py'),'--spec',str(lane/'spec.json')"
    if bootstrap.count(old)!=1:raise ValueError('Unexpected bootstrap entrypoint')
    (folder/'bootstrap.py').write_text(bootstrap.replace(old,"str(lane/'qualification.py')"))
    workflow=yaml.safe_load((folder/'workflow.yaml').read_text())
    workflow['workflow']['name']='leact-efe-checkpoint-gather-v1'
    workflow['workflow']['timeout']['exec_timeout']='3h'
    workflow['workflow']['tasks'][0]['files'].append({'localpath':'qualification.py','path':'/tmp/efe-coverage-training/qualification.py'})
    (folder/'workflow.yaml').write_text(yaml.safe_dump(workflow,sort_keys=False))
    for path in folder.glob('*.py'):compile(path.read_text(),str(path),'exec')
    freeze_sources(folder)
    return folder


def tick():
    status=read(STATUS)
    if status.get('status') in {'QUALIFIED','NEEDS_REVIEW'}:return status
    active=command(['osmo','workflow','list','--name','leact','--status','RUNNING','PENDING','WAITING','--format-type','json'])['workflows']
    previous=status.get('attempts',[])[-1] if status.get('attempts') else None
    retry=previous is None
    if previous:
        scheduler=command(['osmo','workflow','query',previous['workflow'],'--format-type','json'])
        status['status']=scheduler['status']
        write(Path(previous['build'])/'scheduler.json',scheduler)
        hosts=[w['name'] for w in active if w['status']=='RUNNING' and not w['name'].startswith('leact-efe-pear-')]
        host=previous['workflow'] if scheduler['status']=='RUNNING' else next(iter(hosts),None)
        if host:
            root=read(Path(previous['build'])/'spec.json')['qualification_root']+'/'+previous['workflow']
            script="from pathlib import Path\nimport json\np=Path("+repr(root)+")\nr={n:json.loads((p/n).read_text()) for n in ['progress.json','complete.json'] if (p/n).exists()}\nprint('EFE_CAPTURE '+json.dumps(r),flush=True)\n"
            try:status['evidence']=remote(host,'preflight',script)
            except RuntimeError as error:status['capture_error']=str(error)
        complete=status.get('evidence',{}).get('complete.json',{})
        if complete.get('workflow')==previous['workflow'] and complete.get('success'):
            status['status']='QUALIFIED'
        elif scheduler['status'].startswith('FAILED'):
            if infra_retryable(scheduler):retry=retry_due(status,previous['workflow'])
            else:status['status']='NEEDS_REVIEW'
    if retry:
        from .compute_budget import reserved_nodes,node_limit
        if reserved_nodes(active)<node_limit():
            original=Path(status['build']);folder=original
            if previous:
                count=len(status['attempts']);folder=original.parent/(original.name+'-retry-'+str(count))
                if not (folder/'source-manifest.json').exists():
                    folder.mkdir(exist_ok=True)
                    for name in read(original/'source-manifest.json'):shutil.copyfile(original/name,folder/name)
                    workflow=yaml.safe_load((folder/'workflow.yaml').read_text())
                    workflow['workflow']['name']+='-r'+str(count)
                    (folder/'workflow.yaml').write_text(yaml.safe_dump(workflow,sort_keys=False));freeze_sources(folder)
            receipt=submit(folder,priority='LOW')
            status.setdefault('attempts',[]).append({'workflow':receipt['name'],'build':str(folder),'priority':'LOW'})
            status['status']='PENDING'
    status['updated_utc']=datetime.now(timezone.utc).isoformat();write(STATUS,status)
    performance=read(DEST/'performance-analysis.json',{})
    detail=performance.setdefault('checkpoint_optimization',{})
    detail.update(status=status['status'],updated_utc=status['updated_utc'],production_enabled=False,
        workflow=status['attempts'][-1]['workflow'] if status.get('attempts') else None,
        progress=status.get('evidence',{}).get('progress.json',{}))
    complete=status.get('evidence',{}).get('complete.json',{})
    if complete:detail['qualification_success']=complete.get('success')
    write(DEST/'performance-analysis.json',performance)
    return status


def monitor():
    with (DEST/'checkpoint-gather-monitor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            try:
                status=tick()
                print(json.dumps({'utc':status.get('updated_utc'),'status':status['status']}),flush=True)
                if status['status'] in {'QUALIFIED','NEEDS_REVIEW'}:return
            except Exception as error:
                write(DEST/'checkpoint-gather-monitor-error.json',{'utc':datetime.now(timezone.utc).isoformat(),'error':str(error)})
            time.sleep(45)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['tick','monitor'])
    if parser.parse_args().action=='monitor':monitor()
    else:print(json.dumps(tick()))
