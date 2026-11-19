"""Owned shared-filesystem transport between model and native simulator pools.

Only trusted simulator results use pickle, as with PEAR's local process pipe.
Model commands stay JSON. Episode directories and all files are private; no
simulator result is reconstructed from a missing or failed native verdict.
"""
import json
import os
from pathlib import Path
import pickle
import re
import time
from uuid import uuid4


class RemoteEpisodeLost(RuntimeError):pass


def atomic_bytes(path,raw):
    path=Path(path);temporary=path.with_name(path.name+'.'+uuid4().hex+'.tmp')
    descriptor=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    try:
        with os.fdopen(descriptor,'wb')as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
        temporary.replace(path)
    finally:temporary.unlink(missing_ok=True)


def atomic_json(path,value):atomic_bytes(path,json.dumps(value,allow_nan=False).encode())


def publish_result(folder,index,result):
    raw=pickle.dumps(result,protocol=5)
    if len(raw)>64<<20:raise ValueError('Simulator result exceeds the bounded transport payload')
    atomic_bytes(Path(folder)/f'response-{index:06d}.pkl',raw)


def wait_command(folder,index,timeout=1800):
    folder=Path(folder);path=folder/f'request-{index:06d}.json';deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        if(folder/'cancel.json').exists():return None
        if path.exists():
            if path.stat().st_size>128<<10:raise ValueError('Model action exceeds transport size limit')
            value=json.loads(path.read_text())
            if not isinstance(value,dict):raise ValueError('Model action is not a JSON object')
            return value
        time.sleep(.1)
    raise TimeoutError('No policy action arrived before the native transport deadline')


class FileEpisodeClient:
    def __init__(self,root,kind,env_args,timeout=1800):
        self.root=Path(root);self.kind=kind;self.args=env_args;self.timeout=timeout;self.index=0
        self.root.mkdir(mode=0o700,parents=True,exist_ok=True)
        if self.root.stat().st_mode&0o077:raise ValueError('Native transport root must be private')
        self.uid=uuid4().hex;self.folder=self.root/'episodes'/self.uid
        self.folder.mkdir(mode=0o700,parents=True,exist_ok=False)
        (self.root/'pending').mkdir(mode=0o700,exist_ok=True)

    def _receive(self):
        path=self.folder/f'response-{self.index:06d}.pkl';deadline=time.monotonic()+self.timeout
        while time.monotonic()<deadline:
            error=self.folder/'infrastructure-error.json'
            if error.exists():raise RemoteEpisodeLost(json.loads(error.read_text())['error'])
            if path.exists():
                if path.stat().st_size>64<<20:raise RemoteEpisodeLost('Native transport result exceeds size limit')
                with path.open('rb')as stream:result=pickle.load(stream)
                if not isinstance(result,dict)or type(result.get('done'))is not bool:
                    raise RemoteEpisodeLost('Native simulator returned an invalid episode result')
                rewards=result.get('rewards')
                if not isinstance(rewards,list)or not rewards or any(type(v)not in(int,float)or v not in(0,1)for v in rewards):
                    raise RemoteEpisodeLost('Native simulator returned an invalid binary reward')
                # Observations are already held by the model-side episode and
                # its rollout logger. Avoid retaining uncompressed RGB twice
                # in the transport directory throughout a long training run.
                path.unlink()
                return result
            claimed=self.folder/'claimed.json'
            if claimed.exists():
                worker=json.loads(claimed.read_text())['worker_id'];heartbeat=self.root/'workers'/(worker+'.json')
                if heartbeat.exists()and time.time()-json.loads(heartbeat.read_text())['updated_unix']>120:
                    raise RemoteEpisodeLost('Native simulator worker heartbeat expired')
            time.sleep(.1)
        raise RemoteEpisodeLost('Native simulator did not respond before its infrastructure deadline')

    def open(self):
        atomic_json(self.root/'pending'/(self.uid+'.json'),{'kind':self.kind,'env_args':self.args,
            'open_deadline_unix':time.time()+self.timeout})
        return self._receive()

    def step(self,action):
        self.index+=1;atomic_json(self.folder/f'request-{self.index:06d}.json',action)
        return self._receive()

    def close(self):atomic_json(self.folder/'cancel.json',{'closed_unix':time.time()})


def retire_abandoned_request(root,pending,request,*,now=None,grace_seconds=300):
    """Preserve an unclaimed request after cancellation or its open deadline.

    Older EFE clients may die at the outer simulator timeout before closing.
    Their open deadline is 1800s; native YAM has a distinct 14400s deadline.
    A grace period protects against small inter-node wall-clock differences.
    """
    root,pending=Path(root),Path(pending)
    if not re.fullmatch('[0-9a-f]{32}',pending.stem):raise ValueError('Unexpected native episode identifier')
    folder=root/'episodes'/pending.stem;now=time.time()if now is None else now
    target=folder/'retired-pending-request.json'
    if target.exists():
        # A stale reader or transport replay can expose a retired UID again.
        # Preserve its original evidence and never launch a duplicate episode.
        if json.loads(target.read_text())!=request:
            raise ValueError('A retired request was replayed with different contents')
        pending.unlink(missing_ok=True)
        return None
    reason='cancelled'if(folder/'cancel.json').exists()else None
    deadline=request.get('open_deadline_unix')
    if deadline is None:
        timeout=1800 if request['kind'].startswith('efe/')else 14400 if request['kind'].startswith('native_yam/')else None
        if timeout is not None:
            try:deadline=pending.stat().st_mtime+timeout
            except FileNotFoundError:return None
    if reason is None and deadline is not None and now>deadline+grace_seconds:reason='open_deadline_expired'
    if reason is None:return None
    try:pending.rename(target)
    except FileNotFoundError:return None
    receipt={'uid':pending.stem,'kind':request['kind'],'reason':reason,'retired_unix':now,
        'open_deadline_unix':deadline,'grace_seconds':grace_seconds}
    atomic_json(folder/'retired-pending-receipt.json',receipt)
    return receipt


def retire_abandoned_pending(root,*,now=None):
    root=Path(root);retired=[]
    for pending in sorted((root/'pending').glob('*.json')):
        try:request=json.loads(pending.read_text())
        except FileNotFoundError:continue
        receipt=retire_abandoned_request(root,pending,request,now=now)
        if receipt is not None:retired.append(receipt)
    return retired


def claim_episode(root,worker_id,kinds=None):
    root=Path(root)
    for pending in sorted((root/'pending').glob('*.json')):
        if not re.fullmatch('[0-9a-f]{32}',pending.stem):raise ValueError('Unexpected native episode identifier')
        try:request=json.loads(pending.read_text())
        except FileNotFoundError:continue
        if retire_abandoned_request(root,pending,request)is not None:continue
        if kinds is not None and request['kind']not in kinds:continue
        # Atomic rename grants one owner across every native simulator worker.
        folder=root/'episodes'/pending.stem
        try:pending.rename(folder/'claimed-request.json')
        except FileNotFoundError:continue
        atomic_json(folder/'claimed.json',{'worker_id':worker_id,'claimed_unix':time.time()})
        return folder,json.loads((folder/'claimed-request.json').read_text())
    return None


def serve_episode(folder,factory,env_args):
    environment=None
    try:
        if(Path(folder)/'cancel.json').exists():return
        environment=factory(**env_args);value=environment.open();index=0
        publish_result(folder,index,value)
        while not value['done']:
            index+=1;action=wait_command(folder,index)
            if action is None:return
            value=environment.step(action);publish_result(folder,index,value)
    except Exception as exc:
        # Infrastructure errors propagate and cause a fresh native episode;
        # a missing outcome never becomes a training failure reward.
        atomic_json(Path(folder)/'infrastructure-error.json',{'error':repr(exc)})
        raise
    finally:
        if environment is not None:environment.close()
