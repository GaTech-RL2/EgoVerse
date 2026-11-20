"""Keep an OSMO allocation alive independently of its training/simulator job.

Only an explicit release request ends this process. A child failure is recorded
and leaves the node idle; another versioned job can use the same allocation.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


def write(path,value):
    path=Path(path);path.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value,allow_nan=False)+'\n');temporary.replace(path)


def read(path,default=None):
    try:return json.loads(Path(path).read_text())
    except FileNotFoundError:return default


def validate_request(request,roles):
    if not re.fullmatch(r'[A-Za-z0-9_.-]{1,100}',request['id']):raise ValueError('Invalid allocation job id')
    if request['roles']!=roles:raise ValueError('A job cannot change the allocated node roles')
    bundle=Path(request['bundle'])
    if not bundle.is_absolute():raise ValueError('Job bundle must be absolute')
    entry=request['entry']
    if Path(entry).name!=entry or not entry.endswith('.py'):raise ValueError('Invalid allocation job entry')
    if entry not in request['files']or 'spec.json'not in request['files']:raise ValueError('Incomplete allocation job manifest')
    for name,expected in request['files'].items():
        if Path(name).name!=name:raise ValueError('Allocation job inputs must be flat')
        if hashlib.sha256((bundle/name).read_bytes()).hexdigest()!=expected:raise ValueError('Allocation job source changed: '+name)
    return bundle/entry


def stop_child(child):
    if child is None or child.poll()is not None:return
    # All child entries install a Python SIGTERM handler, so their finally
    # blocks close Ray/simulators before this process remains idle.
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid,signal.SIGKILL);child.wait()


class Keeper:
    def __init__(self,root,role,roles,*,environment=None,popen=subprocess.Popen):
        self.root=Path(root);self.root.mkdir(mode=0o700,parents=True,exist_ok=True)
        if self.root.stat().st_mode&0o077:raise ValueError('Allocation control directory must be private')
        self.role,self.roles=role,roles
        if role not in roles:raise ValueError('Unknown allocated role')
        self.environment=dict(os.environ if environment is None else environment)
        self.popen=popen;self.child=None;self.job=None;self.phase='waiting_for_job';self.error=None

    def status(self):
        value={'role':self.role,'job_id':self.job,'phase':self.phase,'updated_unix':time.time(),
            'keeper_pid':os.getpid(),'child_pid':self.child.pid if self.child else None,'error':self.error}
        write(self.root/'nodes'/(self.role+'.json'),value)
        return value

    def tick(self):
        release=read(self.root/'release.json',{})
        if release.get('release')is True and release.get('authorized_by')=='user':
            stop_child(self.child);self.phase='released_by_user';self.status();return False
        request=read(self.root/'desired.json')
        if request and request['id']!=self.job:
            # Validate everything before stopping an existing job.
            try:entry=validate_request(request,self.roles)
            except Exception as exc:
                write(self.root/'rejected'/(request.get('id','invalid')+'.json'),{'error':str(exc)})
                self.status();return True
            stop_child(self.child);self.child=None;self.job=request['id'];self.error=None
            try:
                env={**self.environment,'EFE_PEAR_ATTEMPT_ID':self.job,'PEAR_ALLOCATION_CONTROL':str(self.root)}
                if self.role=='native':
                    env['EFE_ALLOCATION_WORKFLOW_ID']=env['EFE_WORKFLOW_ID']
                    env['EFE_WORKFLOW_ID']=env['EFE_WORKFLOW_ID']+'-'+self.job
                    # Native bootstraps use node-local credential filenames.
                    # An earlier child is gone before another job starts here.
                    for name in ['rp_jhh_native_archive_credentials.json','rp_jhh_rtx_archive_credentials.json']:
                        (Path('/dev/shm')/name).unlink(missing_ok=True)
                # Preserve each entry's normal finally-based cleanup when a
                # job is replaced. Exiting this child never exits the keeper.
                script='import runpy,signal,sys;'+(
                    '\ndef stop(*args): raise SystemExit(143)\n')+(
                    'signal.signal(signal.SIGTERM,stop)\n')+(
                    'sys.path.insert(0,sys.argv[1]);sys.argv=sys.argv[2:];runpy.run_path(sys.argv[0],run_name="__main__")')
                logs=self.root/'jobs'/self.job;logs.mkdir(mode=0o700,parents=True,exist_ok=True)
                with(logs/(self.role+'.log')).open('a')as log:
                    self.child=self.popen([sys.executable,'-u','-c',script,str(entry.parent),str(entry)],
                        env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                self.phase='running'
            except Exception as exc:self.phase='idle_failed';self.error=repr(exc)
        if self.child is not None:
            code=self.child.poll()
            if code is not None:
                self.phase='idle_completed'if code==0 else'idle_failed'
                if code:self.error=f'Child process exited with return code {code}'
            elif any(v.get('job_id')==self.job and v.get('phase')=='idle_failed'
                for v in(read(p,{})for p in(self.root/'nodes').glob('*.json'))):
                stop_child(self.child);self.phase='idle_peer_failed';self.error='A job peer failed; allocation retained'
        self.status();return True


def main():
    lane=Path(__file__).resolve().parent;workflow=os.environ['EFE_WORKFLOW_ID']
    role=os.environ.get('EFE_ROLE','native');roles=json.loads(os.environ.get('PEAR_ALLOCATION_ROLES','["native"]'))
    root=Path(os.environ['PEAR_ALLOCATION_BASE'])/workflow
    keeper=Keeper(root,role,roles)
    if role==roles[0]and os.environ.get('PEAR_KEEPER_AUTOSTART','1')!='0'and not(root/'desired.json').exists():
        spec=read(lane/'spec.json');names=[p.name for p in lane.iterdir()if p.is_file()and p.suffix in {'.py','.json','.yaml','.sh'}and p.name!='bootstrap.py']
        initial={'id':'initial-'+hashlib.sha256((lane/'spec.json').read_bytes()).hexdigest()[:12],
            'bundle':str(lane),'entry':os.environ['PEAR_KEEPER_ENTRY'],'roles':roles,
            'files':{n:hashlib.sha256((lane/n).read_bytes()).hexdigest()for n in names}}
        write(root/'desired.json',initial)
    print('PEAR_ALLOCATION_READY '+json.dumps({'workflow':workflow,'role':role,'control':str(root)}),flush=True)
    try:
        while True:
            try:
                if not keeper.tick():break
            except Exception as exc:
                # A malformed request or transient shared-filesystem error
                # must not terminate an otherwise allocated node.
                print('PEAR_ALLOCATION_CONTROL_ERROR '+json.dumps({'error':repr(exc)}),flush=True)
            time.sleep(2)
    finally:stop_child(keeper.child)


if __name__=='__main__':main()
