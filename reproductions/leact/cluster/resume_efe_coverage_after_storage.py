"""Resume the authorized frozen job once the inactive storage copy is verified."""
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import time

from .recover_efe_coverage_storage import remote,DEST
from .scale_efe import command,read,write,submit
from .launch_efe_coverage_train import retry_build
from .supervise_efe_coverage import install

REMOTE_STATE='/mnt/amlfs-07/shared/rpunamiya/leact/studies/efe-coverage-20261105-v1/storage-recovery-20261113'
EXPECTED_PRIOR='leact-efe-coverage-warmstart-v1-p7-r20-1'


def now():return datetime.now(timezone.utc).isoformat()


def main():
    with (DEST/'storage-resume.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        deadline=time.monotonic()+7200
        while time.monotonic()<deadline:
            active=command(['osmo','workflow','list','--name','leact','--status','RUNNING','PENDING','WAITING','--format-type','json'])['workflows']
            hosts=[r['name'] for r in active if r['status']=='RUNNING']
            if not hosts:raise RuntimeError('No live reader remains for storage verification')
            host=next((h for h in hosts if 'astra-b09-v1-r9' in h),hosts[0])
            script="""import json
from pathlib import Path
p=Path(STATE)
v={name:json.loads((p/name).read_text()) for name in ['progress.json','complete.json','error.json'] if (p/name).exists()}
print('EFE_CAPTURE '+json.dumps(v),flush=True)
""".replace('STATE',repr(REMOTE_STATE))
            value=remote(host,'preflight',script)
            write(DEST/'warmstart-storage-progress-20261113.json',value)
            write(DEST/'storage-resume-health.json',{'utc':now(),'pid':os.getpid(),'phase':value.get('progress.json',{}).get('phase')})
            if value.get('error.json'):raise RuntimeError('Storage copy failed: '+str(value['error.json']))
            complete=value.get('complete.json',{})
            if complete.get('success'):
                if complete.get('checkpoint_step')!=176 or complete.get('target_step')!=737 or complete.get('recipe_changed'):
                    raise ValueError('Storage recovery receipt does not match the frozen run')
                write(DEST/'warmstart-storage-complete-20261113.json',complete)
                with (DEST/'monitor.lock').open('a') as monitor_lock:
                    fcntl.flock(monitor_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    queue=read(DEST/'training-queue.json');record=queue['stages']['warmstart']
                    if record['attempts'][-1]['workflow']!=EXPECTED_PRIOR:
                        raise ValueError('Training queue advanced; inspect before another submission')
                    if any(r['name'].startswith('leact-efe-coverage-warmstart') for r in active):
                        raise ValueError('Warmstart allocation already active')
                    folder=retry_build(record['attempts'][-1]['build'],len(record['attempts']))
                    receipt=submit(folder,priority='HIGH')
                    record['attempts'].append({'workflow':receipt['name'],'build':str(folder),
                        'submitted_utc':now(),'priority':'HIGH','reason':'Verified storage-only recovery; exact continuation from committed checkpoint 176.'})
                    record.update(status='PENDING',retry=False)
                    write(DEST/'training-queue.json',queue)
                    recovery=read(DEST/'training-recovery-status.json')
                    recovery.update(utc=now(),status='PENDING',workflow=receipt['name'],
                        automatic_progress_updates=True,supervisor_temporarily_stopped_for_storage_move=False,
                        storage_relocation=complete,committed_restart_step=176,
                        summary='Verified warmstart storage relocation is complete. The unchanged HIGH-priority job is submitted to resume checkpoint 176/737. LeAct remains dependency queued; native baseline evaluation workers continue independently.')
                    write(DEST/'training-recovery-status.json',recovery)
                supervisor=install()
                result={'utc':now(),'workflow':receipt['name'],'supervisor':supervisor,'storage':complete}
                write(DEST/'storage-resume-complete.json',result)
                print(json.dumps(result),flush=True)
                return
            time.sleep(30)
        raise TimeoutError('Storage copy did not finish within the bounded two-hour recovery window')


if __name__=='__main__':
    try:main()
    except Exception as error:
        write(DEST/'storage-resume-error.json',{'utc':now(),'error':str(error),'pid':os.getpid()})
        # Training remains NEEDS_REVIEW; restore evaluation monitoring/retries.
        install()
        raise
