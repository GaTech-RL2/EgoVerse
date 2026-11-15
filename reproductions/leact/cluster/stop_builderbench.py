"""Finish the user's BuilderBench stop request when the OSMO API recovers."""
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

BASE=Path(__file__).resolve().parents[1]
STOP=BASE/'artifacts/builderbench-stop.json'


def write(path,value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');temp.replace(path)


def query(args):
    result=subprocess.run(['osmo',*args,'--format-type','json'],capture_output=True,text=True,timeout=40)
    if result.returncode:raise RuntimeError((result.stdout+result.stderr)[-1800:])
    return json.loads(result.stdout[result.stdout.index('{'):])


def main():
    with (BASE/'cluster/.build/builderbench-stop.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for _ in range(2880):
            record=json.loads(STOP.read_text())
            if not record.get('stop_requested'):return
            record.update(cancellation_worker_pid=os.getpid(),updated_utc=datetime.now(timezone.utc).isoformat())
            try:
                active=query(['workflow','list','--name','leact-builderbench','--status','RUNNING','PENDING','WAITING'])['workflows']
                record['known_workflows']=sorted(set(record.get('known_workflows',[]))|{w['name'] for w in active})
                if not active:
                    record.update(remote_cancellation_status='confirmed_no_active_builderbench_jobs',
                                  confirmed_utc=datetime.now(timezone.utc).isoformat(),active_workflows=[])
                    record.pop('last_service_error',None)
                    # Collect terminal scheduler receipts while the API is healthy.
                    for name in record['known_workflows']:
                        try:
                            state=query(['workflow','query',name])
                            for b in (BASE/'cluster/.build').glob('builderbench-gpt6-*-20261024-v1'):
                                p=b/'submission.json'
                                if p.exists() and json.loads(p.read_text()).get('name')==name:write(b/'status-current.json',state)
                        except Exception:pass
                    write(STOP,record)
                    from ..embodied_first_exam.report import publish
                    publish()
                    return
                record['active_workflows']=[w['name'] for w in active]
            except Exception as error:
                record['last_service_error']=str(error)
            attempts=[]
            for name in record.get('known_workflows',[]):
                try:
                    result=subprocess.run(['osmo','workflow','cancel',name,'--message',
                       'User requested stopping BuilderBench to focus exclusively on EFE','--format-type','json'],
                       capture_output=True,text=True,timeout=40)
                    # The CLI can return exit zero even when cancellation fails.
                    # Only a later successful active-workflow listing confirms it.
                    attempts.append({'workflow':name,'returncode':result.returncode,
                                     'output':(result.stdout+result.stderr)[-1800:]})
                except Exception as error:attempts.append({'workflow':name,'error':str(error)})
            record.update(remote_cancellation_status='awaiting_scheduler_confirmation',last_cancel_attempts=attempts)
            write(STOP,record)
            time.sleep(30)
        record=json.loads(STOP.read_text());record['remote_cancellation_status']='retry_deadline_reached';write(STOP,record)


if __name__=='__main__':main()
