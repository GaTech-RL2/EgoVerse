"""Keep the declared HF rescore and CPU assembly jobs alive through infrastructure failures."""
import argparse
from datetime import datetime,timezone
import fcntl
from pathlib import Path
import shutil
import time

import yaml

from .scale_efe import read,write,command,submit
from .launch_efe_coverage import DEST,infra_retryable
from .launch_efe_training import freeze_sources
from .efe_coverage_retry import retry_due,submission_priority


def now():return datetime.now(timezone.utc).isoformat()


def observe(job,pool):
    state=command(['osmo','workflow','query',job['workflow'],'--format-type','json'])
    write(Path(job['build'])/'scheduler.json',state)
    job['status']=state['status']
    if not state['status'].startswith('FAILED'):return
    if not infra_retryable(state):job['status']='NEEDS_REVIEW';return
    if not retry_due(job,job['workflow']):job['status']='RETRY_BACKOFF';return
    attempts=job.setdefault('attempts',[]);source=Path(job['build'])
    folder=source.parent/(source.name+'-infra-retry-'+str(len(attempts)))
    if not(folder/'source-manifest.json').exists():
        folder.mkdir(exist_ok=True)
        for name in read(source/'source-manifest.json'):shutil.copyfile(source/name,folder/name)
        workflow=yaml.safe_load((folder/'workflow.yaml').read_text())
        workflow['workflow']['name']+='-r'+str(len(attempts))
        (folder/'workflow.yaml').write_text(yaml.safe_dump(workflow,sort_keys=False));freeze_sources(folder)
    priority=submission_priority({'attempts':attempts+[{'submitted_utc':job.get('submitted_utc','2026-11-17T17:00:00+00:00')}]})
    receipt=submit(folder,priority=priority,pool=pool)
    attempts.append({k:job[k]for k in ['workflow','build','priority']if k in job})
    job.update(workflow=receipt['name'],build=str(folder),priority=priority,status='PENDING',submitted_utc=now())


def tick():
    queue=read(DEST/'hf-rescore-queue.json',{'jobs':[]})
    for job in queue['jobs']:
        observe(job,'groot-h100-02');write(DEST/'hf-rescore-queue.json',queue)
    assembly=read(DEST/'hf-assembly-launch.json',{})
    if assembly:
        observe(assembly,'groot-l40-03');write(DEST/'hf-assembly-launch.json',assembly)
    state={'updated_utc':now(),'rescore_jobs':queue['jobs'],'assembly':assembly,
        'automatic_training_release':'The coverage controller releases the reviewed primary payload after verified admission promotion.',
        'production_optimizer_updates_by_this_monitor':0}
    write(DEST/'hf-recovery-status.json',state)
    return bool(queue['jobs'])and all(j.get('status')=='COMPLETED'for j in queue['jobs'])and assembly.get('status')=='COMPLETED'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['tick','monitor']);args=parser.parse_args()
    with(DEST/'hf-recovery-monitor.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            try:
                done=tick()
                if done or args.action=='tick':return
            except Exception as error:
                write(DEST/'hf-recovery-monitor-error.json',{'updated_utc':now(),'error':repr(error)})
                if args.action=='tick':raise
            time.sleep(45)


if __name__=='__main__':main()
