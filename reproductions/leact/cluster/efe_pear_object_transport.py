"""Private object-store transport between pools with different filesystems.

The native side keeps the qualified file protocol. One bridge owns a private
local directory; model clients exchange exactly the same JSON actions and
trusted native result bytes through the study's private object-store prefix.
"""
import json
from pathlib import Path
import pickle
import re
import time
from uuid import uuid4


UID=re.compile(r'^[0-9a-f]{32}$')
MAX_RESULT=64<<20


class LostEpisode(RuntimeError):pass


class Store:
    def __init__(self,uri,client):
        from urllib.parse import urlsplit
        value=urlsplit(uri)
        if value.scheme!='s3'or not value.netloc or not value.path.strip('/'):
            raise ValueError('A private object-store prefix is required')
        self.client=client;self.bucket=value.netloc;self.prefix=value.path.strip('/')+'/'
        if any(p in {'.','..'}for p in self.prefix.split('/')):raise ValueError('Unsafe object prefix')

    def put(self,key,raw):
        self.client.put_object(Bucket=self.bucket,Key=self.prefix+key,Body=raw)

    def put_json(self,key,value):self.put(key,json.dumps(value,allow_nan=False).encode())

    def get(self,key,limit=MAX_RESULT):
        try:value=self.client.get_object(Bucket=self.bucket,Key=self.prefix+key)
        except Exception as exc:
            code=getattr(exc,'response',{}).get('Error',{}).get('Code')
            if code in {'NoSuchKey','404','NotFound'}:return None
            raise
        body=value['Body']
        try:
            raw=body.read(limit+1)
            if len(raw)>limit:raise ValueError('Transport payload exceeds its bound')
            return raw
        finally:body.close()

    def remove(self,key):self.client.delete_object(Bucket=self.bucket,Key=self.prefix+key)

    def keys(self,prefix):
        token=None
        while True:
            args={'Bucket':self.bucket,'Prefix':self.prefix+prefix}
            if token:args['ContinuationToken']=token
            value=self.client.list_objects_v2(**args)
            for row in value.get('Contents',[]):yield row['Key'][len(self.prefix):]
            if not value.get('IsTruncated'):return
            token=value['NextContinuationToken']


def checked_result(raw):
    value=pickle.loads(raw)
    if not isinstance(value,dict)or type(value.get('done'))is not bool:
        raise LostEpisode('Native simulator returned an invalid episode result')
    rewards=value.get('rewards')
    if not isinstance(rewards,list)or not rewards or any(type(v)not in(int,float)or v not in(0,1)for v in rewards):
        raise LostEpisode('Native simulator returned an invalid binary reward')
    return value


class ObjectEpisodeClient:
    def __init__(self,store,kind,env_args,timeout=1800,poll_seconds=.2):
        self.store=store;self.kind=kind;self.args=env_args;self.timeout=timeout
        self.poll_seconds=poll_seconds;self.uid=uuid4().hex;self.index=0

    def _receive(self):
        deadline=time.monotonic()+self.timeout;key=f'responses/{self.uid}/{self.index:06d}.pkl'
        while time.monotonic()<deadline:
            error=self.store.get(f'errors/{self.uid}.json',128<<10)
            if error is not None:raise LostEpisode(json.loads(error)['error'])
            raw=self.store.get(key)
            if raw is not None:
                result=checked_result(raw);self.store.remove(key);return result
            time.sleep(self.poll_seconds)
        raise LostEpisode('Native simulator did not respond before its infrastructure deadline')

    def open(self):
        self.store.put_json(f'pending/{self.uid}.json',{'kind':self.kind,'env_args':self.args})
        return self._receive()

    def step(self,action):
        raw=json.dumps(action,allow_nan=False).encode()
        if not isinstance(action,dict)or len(raw)>128<<10:raise ValueError('Invalid bounded JSON action')
        self.index+=1;self.store.put(f'requests/{self.uid}/{self.index:06d}.json',raw)
        return self._receive()

    def close(self):self.store.put_json(f'cancels/{self.uid}.json',{'closed_unix':time.time()})


class NativeBridge:
    """Single-owner mirror. Native claim/termination semantics stay unchanged."""
    def __init__(self,root,store):
        from efe_pear_file_rpc import atomic_json
        self.atomic_json=atomic_json;self.root=Path(root);self.store=store
        self.root.mkdir(mode=0o700,parents=True,exist_ok=True)
        if self.root.stat().st_mode&0o077:raise ValueError('Bridge filesystem must be private')
        for name in ['pending','episodes','workers']:(self.root/name).mkdir(mode=0o700,exist_ok=True)
        self.state=self.root/'object-bridge-state.json'
        self.active=json.loads(self.state.read_text())if self.state.exists()else{}

    @staticmethod
    def episode_id(key,prefix,suffix):
        uid=key.removeprefix(prefix).removesuffix(suffix)
        if not key.startswith(prefix)or not key.endswith(suffix)or not UID.fullmatch(uid):
            raise ValueError('Unexpected episode object key')
        return uid

    def tick(self):
        from efe_pear_file_rpc import atomic_bytes
        # Persist local ownership before deleting any input object. A retry
        # cannot publish a second local pending request for a claimed episode.
        for key in list(self.store.keys('pending/')):
            uid=self.episode_id(key,'pending/','.json');folder=self.root/'episodes'/uid
            raw=self.store.get(key,128<<10)
            if raw is None:continue
            request=json.loads(raw)
            if not isinstance(request,dict)or not isinstance(request.get('kind'),str)or not isinstance(request.get('env_args'),dict):
                raise ValueError('Invalid native episode request')
            if uid not in self.active:
                folder.mkdir(mode=0o700,exist_ok=True);self.active[uid]={'index':0,'cancelled':False}
                self.atomic_json(self.state,self.active)
            if not(folder/'claimed-request.json').exists()and not(folder/'claimed.json').exists():
                atomic_bytes(self.root/'pending'/(uid+'.json'),raw)
            self.store.remove(key)
        for key in list(self.store.keys('requests/')):
            parts=key.split('/')
            if len(parts)!=3 or not UID.fullmatch(parts[1])or not re.fullmatch(r'[0-9]{6}\.json',parts[2]):
                raise ValueError('Invalid action object key')
            uid=parts[1];raw=self.store.get(key,128<<10)
            if raw is None:continue
            if uid not in self.active:
                if(self.root/'episodes'/uid/'cancel.json').exists():
                    self.store.remove(key);continue
                raise ValueError('Action has no owned episode')
            if not isinstance(json.loads(raw),dict):raise ValueError('Action must be a JSON object')
            atomic_bytes(self.root/'episodes'/uid/('request-'+parts[2]),raw);self.store.remove(key)
        for uid,value in list(self.active.items()):
            folder=self.root/'episodes'/uid
            error=folder/'infrastructure-error.json'
            if error.exists():self.store.put(f'errors/{uid}.json',error.read_bytes())
            while True:
                index=value['index'];path=folder/f'response-{index:06d}.pkl'
                if not path.exists():break
                raw=path.read_bytes()
                if len(raw)>MAX_RESULT:raise ValueError('Native response exceeds its bound')
                checked_result(raw)
                self.store.put(f'responses/{uid}/{index:06d}.pkl',raw)
                value['index']+=1;self.atomic_json(self.state,self.active);path.unlink(missing_ok=True)
        for key in list(self.store.keys('cancels/')):
            uid=self.episode_id(key,'cancels/','.json');raw=self.store.get(key,128<<10)
            if raw is None:continue
            if uid in self.active:
                atomic_bytes(self.root/'episodes'/uid/'cancel.json',raw)
                del self.active[uid];self.atomic_json(self.state,self.active)
            self.store.remove(key)
        for name in ['head-heartbeat.json','finished.json']:
            raw=self.store.get('control/'+name,128<<10)
            path=self.root.parent/name
            if raw is not None and(not path.exists()or path.read_bytes()!=raw):atomic_bytes(path,raw)


def main():
    import argparse,fcntl,os,sys,subprocess
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,default=Path(__file__).with_name('spec.json'));a=p.parse_args()
    spec=json.loads(a.spec.read_text());root=Path(spec['root']);root.mkdir(mode=0o700,parents=True,exist_ok=True)
    credential=Path('/dev/shm/rp_jhh_transport_credentials.json')
    fd=os.open(credential,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w')as stream:stream.write(os.environ.pop('RP_JHH_PRIVATE_ARCHIVE_CONFIGURATION'))
    os.environ['PEAR_OBJECTSTORE_CREDENTIAL_FILE']=str(credential)
    dependencies=Path('/tmp/rp_jhh_transport_packages')
    subprocess.run([sys.executable,'-m','pip','install','--quiet','--target',str(dependencies),'boto3==1.44.5'],check=True,timeout=300)
    sys.path.insert(0,str(dependencies))
    from efe_pear_object_archive import configured_client
    store=Store(spec['uri'],configured_client());bridge=NativeBridge(root,store)
    with(root/'object-bridge.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            while not(root.parent/'finished.json').exists():
                bridge.tick();time.sleep(.2)
        finally:credential.unlink(missing_ok=True)


if __name__=='__main__':main()
