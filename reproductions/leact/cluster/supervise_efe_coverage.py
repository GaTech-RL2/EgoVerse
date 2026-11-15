"""Supervise the EFE controller without depending on status-file writes."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[3]
DEST=REPO/'reproductions/leact/artifacts/embodied-first-exam/fixed-action-v1/coverage-scale-v1'


def write_health(root, value):
    value={'utc':datetime.now(timezone.utc).isoformat(),'pid':os.getpid(),
           'kind':'detached-watchdog',**value}
    path=root/'monitor-supervisor.json'
    try:
        temporary=path.with_suffix('.tmp')
        temporary.write_text(json.dumps(value,indent=2)+'\n')
        temporary.replace(path)
    except OSError:
        # A full disk can prevent reporting but must not disable supervision.
        pass


def watch(command=None, root=DEST, retry_seconds=60, poll_seconds=15):
    root.mkdir(parents=True,exist_ok=True)
    command=command or [str(REPO/'emimic/bin/python'),'-S','-u','-m',
                        'reproductions.leact.cluster.launch_efe_coverage','monitor']
    site=REPO/f'emimic/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages'
    env={**os.environ,'PYTHONPATH':str(site),'PYTHONUNBUFFERED':'1',
         'EFE_MONITOR_SUPERVISOR':'detached-watchdog:'+str(os.getpid())}
    with (root/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        starts=0
        while True:
            child=None
            try:
                with (root/'monitor.log').open('ab',buffering=0) as log:
                    child=subprocess.Popen(command,cwd=REPO,env=env,stdin=subprocess.DEVNULL,
                        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    starts+=1
                    while child.poll() is None:
                        write_health(root,{'phase':'running','child_pid':child.pid,'starts':starts})
                        time.sleep(poll_seconds)
                    code=child.returncode
            except OSError as error:
                write_health(root,{'phase':'retrying','error':str(error),'starts':starts})
                time.sleep(retry_seconds)
                continue
            finally:
                if child is not None and child.poll() is None:
                    child.terminate()
                    try:child.wait(timeout=20)
                    except subprocess.TimeoutExpired:child.kill();child.wait()
            write_health(root,{'phase':'completed' if code==0 else 'retrying',
                'child_pid':child.pid,'last_exit_code':code,'starts':starts})
            if code==0:return
            time.sleep(retry_seconds)


def state():
    path=DEST/'monitor-supervisor.json'
    record=json.loads(path.read_text()) if path.exists() else {}
    pid=record.get('pid',0)
    process=subprocess.run(['ps','-p',str(pid),'-o','command='],capture_output=True,text=True)
    record['alive']='reproductions.leact.cluster.supervise_efe_coverage watch' in process.stdout
    return record


def install():
    if state().get('alive'):return state()
    launch=DEST/'monitor-launch.json'
    if launch.exists():
        prior=json.loads(launch.read_text())
        process=subprocess.run(['ps','-p',str(prior.get('pid',0)),'-o','command='],capture_output=True,text=True)
        if 'reproductions.leact.cluster.launch_efe_coverage monitor' in process.stdout:
            raise RuntimeError('Existing manual monitor must exit before starting its supervisor')
    with (DEST/'supervisor.log').open('ab',buffering=0) as log:
        child=subprocess.Popen([sys.executable,'-S','-u','-m',
            'reproductions.leact.cluster.supervise_efe_coverage','watch'],cwd=REPO,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    return {'started_supervisor_pid':child.pid,'kind':'detached-watchdog'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['install','status','watch'])
    args=parser.parse_args()
    if args.action=='watch':watch()
    else:print(json.dumps(install() if args.action=='install' else state()))
