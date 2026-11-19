"""Reap processes owned by an earlier job without ending its OSMO allocation."""
import os
from pathlib import Path
import signal
import time


def process_state(path):
    try:
        fields=(path/'stat').read_text().rsplit(')',1)[1].split()
        return {'state':fields[0],'parent':int(fields[1]),'started':int(fields[19])}
    except(FileNotFoundError,ProcessLookupError):return None


def owned_processes(workflow,control,*,keep_attempt=None,only_attempt=None,proc=Path('/proc')):
    if not workflow or not Path(control).is_absolute()or Path(control).name!=workflow:
        raise ValueError('Cleanup requires the exact allocation control directory')
    if keep_attempt is not None and only_attempt is not None:
        raise ValueError('Choose previous-job or current-job cleanup')
    # Keep the calling entry, the keeper and the scheduler's exec ancestry.
    protected={1};pid=os.getpid()
    while pid>1 and pid not in protected:
        protected.add(pid);row=process_state(proc/str(pid))
        if row is None:break
        pid=row['parent']
    result={}
    for path in proc.iterdir():
        if not path.name.isdigit()or int(path.name)in protected:continue
        row=process_state(path)
        if row is None or row['state']=='Z':continue
        try:env=dict(v.split(b'=',1)for v in(path/'environ').read_bytes().split(b'\0')if b'='in v)
        except(FileNotFoundError,ProcessLookupError,PermissionError):continue
        attempt=env.get(b'EFE_PEAR_ATTEMPT_ID',b'').decode()
        if(env.get(b'EFE_WORKFLOW_ID')!=workflow.encode()or
            env.get(b'PEAR_ALLOCATION_CONTROL')!=str(control).encode()or not attempt):continue
        if keep_attempt is not None and attempt==keep_attempt:continue
        if only_attempt is not None and attempt!=only_attempt:continue
        result[int(path.name)]={**row,'attempt':attempt}
    return result


def signal_process(pid,identity,sig,*,proc=Path('/proc')):
    current=process_state(proc/str(pid))
    # A PID can be reused between the ownership scan and signaling.
    if current is None or current['state']=='Z'or current['started']!=identity['started']:return False
    try:os.kill(pid,sig)
    except ProcessLookupError:return False
    return True


def cleanup(workflow,control,*,keep_attempt=None,only_attempt=None,proc=Path('/proc'),grace_seconds=8.):
    kwargs={'keep_attempt':keep_attempt,'only_attempt':only_attempt,'proc':proc}
    found=owned_processes(workflow,control,**kwargs)
    terminated=[pid for pid,row in found.items()if signal_process(pid,row,signal.SIGTERM,proc=proc)]
    deadline=time.monotonic()+grace_seconds
    while time.monotonic()<deadline:
        if not owned_processes(workflow,control,**kwargs):break
        time.sleep(.1)
    # Rescan also catches descendants spawned while a parent was terminating.
    remaining=owned_processes(workflow,control,**kwargs)
    killed=[pid for pid,row in remaining.items()if signal_process(pid,row,signal.SIGKILL,proc=proc)]
    deadline=time.monotonic()+5.
    while time.monotonic()<deadline:
        remaining=owned_processes(workflow,control,**kwargs)
        if not remaining:break
        time.sleep(.1)
    if remaining:raise RuntimeError('Owned job processes did not stop: '+str(sorted(remaining)))
    return {'terminated_pids':terminated,'killed_pids':killed,'remaining':0,'allocation_preserved':True}
