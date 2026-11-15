"""Watch the bounded PEAR qualification; persist results and publish honest state."""
from datetime import datetime, timezone
from pathlib import Path
import fcntl
import json
import shutil
import subprocess
import time
import yaml
from .launch_efe_pear import DEST,prepare,PAYLOAD_VERSION
from .scale_efe import command, read, write, submit
from .launch_efe_training import freeze_sources
from .launch_efe_coverage import infra_retryable
from .efe_coverage_retry import retry_due


def tick():
    base=prepare()
    queue=read(DEST/'qualification-queue.json',{})
    if queue.get('payload_version')!=PAYLOAD_VERSION:
        queue={'payload_version':PAYLOAD_VERSION,'attempts':[],
               'previous_payloads':queue.get('previous_payloads',[])+([queue] if queue else [])}
    build=Path(queue['attempts'][-1]['build']) if queue['attempts'] else base
    receipt=read(build/'submission.json',{})
    if not receipt:return False
    if not queue['attempts']:
        queue['attempts'].append({'build':str(build),'workflow':receipt['name'],'priority':'HIGH'})
        write(DEST/'qualification-queue.json',queue)
    name=receipt['name'];job=command(['osmo','workflow','query',name,'--format-type','json'])
    status=read(DEST/'status.json',{})
    status.update(workflow=name,scheduler_status=job['status'],updated_utc=datetime.now(timezone.utc).isoformat())
    done=job['status'] not in {'RUNNING','PENDING','WAITING','INITIALIZING','SCHEDULING','SUBMITTED','QUEUED'}
    if done:
        result=subprocess.run(['osmo','workflow','logs',name],capture_output=True,text=True,timeout=50)
        (DEST/'qualification.log').write_text(result.stdout+result.stderr)
        rows=[]
        for line in result.stdout.splitlines():
            if 'PEAR_RESULT ' in line:
                try:rows.append(json.loads(line.split('PEAR_RESULT ',1)[1]))
                except json.JSONDecodeError:pass
        if rows:
            write(DEST/'qualification-result.json',rows[-1])
            status.update(phase='qualification_passed' if rows[-1].get('success') else 'qualification_failed',
                summary=('GPU loss checks and exact Qwen sampler capture passed. Native grouped rollout integration and the common final warmstart are still required before RL training.'
                    if rows[-1].get('success') else 'Qualification failed; no RL optimizer update or native success result. '+rows[-1].get('error','')))
        else:status.update(phase='qualification_ended_without_result',summary='OSMO qualification ended without a validated sampler/loss receipt; inspect logs before continuing.')
        # Retry capacity/startup interruptions, never unexplained model errors.
        if not rows and infra_retryable(job) and len(queue['attempts'])<8:
            done=False
            due=retry_due(queue,name)
            status.update(phase='qualification_retry_backoff',summary='Verified allocation/startup interruption; the unchanged qualification will retry at LOW priority within the shared node limit.')
            write(DEST/'qualification-queue.json',queue)
            if due:
                number=len(queue['attempts'])
                retry=base.parent/(base.name+f'-allocation-retry-{number}')
                if not (retry/'source-manifest.json').exists():
                    retry.mkdir(exist_ok=True)
                    for filename in read(base/'source-manifest.json'):
                        shutil.copyfile(base/filename,retry/filename)
                    workflow=yaml.safe_load((retry/'workflow.yaml').read_text())
                    workflow['workflow']['name']=f'leact-efe-pear-qualification-v{PAYLOAD_VERSION}-r{number}'
                    (retry/'workflow.yaml').write_text(yaml.safe_dump(workflow,sort_keys=False))
                    freeze_sources(retry)
                receipt=submit(retry,priority='LOW')
                queue['attempts'].append({'build':str(retry),'workflow':receipt['name'],'priority':'LOW'})
                write(DEST/'qualification-queue.json',queue)
                status.update(workflow=receipt['name'],scheduler_status='PENDING',phase='qualification_pending')
    else:
        status.update(phase='qualification_'+job['status'].lower(),summary='PEAR sampler/loss qualification '+job['status'].lower()+' on one 8-H100 node. This is a runtime check, not an RL training or robot-success result.')
    write(DEST/'status.json',status)
    from ..embodied_first_exam import report
    report.publish()
    return done


if __name__=='__main__':
    with (DEST/'monitor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        deadline=time.monotonic()+4*3600
        while time.monotonic()<deadline:
            try:
                if tick():break
            except Exception as error:
                write(DEST/'monitor-error.json',{'utc':datetime.now(timezone.utc).isoformat(),'error':repr(error)})
            time.sleep(45)
