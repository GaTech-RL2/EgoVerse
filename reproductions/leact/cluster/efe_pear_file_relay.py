"""Move native file RPC across pools without restarting the model cluster."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re

try:
    from efe_pear_file_rpc import atomic_bytes,atomic_json
    from efe_pear_object_transport import NativeBridge,UID,MAX_RESULT,checked_result
except ImportError:
    from .efe_pear_file_rpc import atomic_bytes,atomic_json
    from .efe_pear_object_transport import NativeBridge,UID,MAX_RESULT,checked_result


def worker_name(value):
    if not isinstance(value,str)or not re.fullmatch(r'[A-Za-z0-9_.-]{1,255}',value)or value in {'.','..'}:
        raise ValueError('Invalid native worker identifier')
    return value


class FileModelRelay:
    def __init__(self,root,store,kinds,*,workers=16):
        self.root=Path(root);self.store=store;self.kinds=set(kinds)
        if not self.kinds:raise ValueError('A relay requires explicit admitted episode kinds')
        self.root.mkdir(mode=0o700,parents=True,exist_ok=True)
        if self.root.stat().st_mode&0o077:raise ValueError('Relay filesystem must be private')
        for name in ['pending','episodes','workers','object-relay-owned']:
            (self.root/name).mkdir(mode=0o700,exist_ok=True)
        identity={'bucket':store.bucket,'prefix':store.prefix,'kinds':sorted(self.kinds)}
        binding=self.root/'object-model-relay-identity.json'
        if binding.exists()and json.loads(binding.read_text())!=identity:
            raise ValueError('Relay ownership differs from the recorded object prefix or admitted kinds')
        if not binding.exists():atomic_json(binding,identity)
        self.owned=self.root/'object-relay-owned';self.state=self.root/'object-model-relay-state.json'
        saved=json.loads(self.state.read_text())if self.state.exists()else{}
        self.active={}
        for path in self.owned.glob('*.json'):
            if not UID.fullmatch(path.stem):raise ValueError('Invalid relay episode identifier')
            self.active[path.stem]=saved.get(path.stem,{'index':0,'request':1,'sent':False})
        self.pool=ThreadPoolExecutor(max_workers=workers)

    def close(self):self.pool.shutdown(wait=True)

    def _poll(self,item):
        uid,old=item;value=dict(old);folder=self.root/'episodes'/uid
        if(folder/'cancel.json').exists():
            self.store.put(f'cancels/{uid}.json',(folder/'cancel.json').read_bytes())
            return uid,None
        if not value['sent']:
            self.store.put(f'pending/{uid}.json',(self.owned/(uid+'.json')).read_bytes())
            value['sent']=True
        error=self.store.get(f'errors/{uid}.json',128<<10)
        if error is not None:
            if not isinstance(json.loads(error).get('error'),str):raise ValueError('Invalid native infrastructure error')
            atomic_bytes(folder/'infrastructure-error.json',error)
        if not value.get('worker'):
            claimed=self.store.get(f'claims/{uid}.json',128<<10)
            if claimed is not None:
                value['worker']=worker_name(json.loads(claimed)['worker_id'])
                atomic_bytes(folder/'claimed.json',claimed)
        action=folder/f"request-{value['request']:06d}.json"
        if action.exists():
            raw=action.read_bytes()
            if len(raw)>128<<10 or not isinstance(json.loads(raw),dict):raise ValueError('Invalid bounded JSON action')
            self.store.put(f"requests/{uid}/{value['request']:06d}.json",raw)
            value['request']+=1
        response=f"responses/{uid}/{value['index']:06d}.pkl"
        raw=self.store.get(response)
        if raw is not None:
            if len(raw)>MAX_RESULT:raise ValueError('Native response exceeds its bound')
            checked_result(raw)
            atomic_bytes(folder/f"response-{value['index']:06d}.pkl",raw)
            value['index']+=1
        return uid,value

    def _worker(self,name):
        raw=self.store.get(f'workers/{name}.json',128<<10)
        if raw is not None:
            value=json.loads(raw)
            if value.get('worker_id')!=name or not isinstance(value.get('updated_unix'),(int,float)):
                raise ValueError('Invalid native worker heartbeat')
            target=self.root/'workers'/(name+'.json')
            if not target.exists()or target.read_bytes()!=raw:atomic_bytes(target,raw)

    def tick(self):
        for pending in sorted((self.root/'pending').glob('*.json')):
            if not UID.fullmatch(pending.stem):raise ValueError('Invalid pending episode identifier')
            try:raw=pending.read_bytes()
            except FileNotFoundError:continue
            if len(raw)>128<<10:raise ValueError('Native request exceeds its bound')
            request=json.loads(raw)
            if request.get('kind')not in self.kinds:continue
            if not isinstance(request.get('env_args'),dict):raise ValueError('Invalid native environment arguments')
            try:pending.rename(self.owned/pending.name)
            except FileNotFoundError:continue
            self.active.setdefault(pending.stem,{'index':0,'request':1,'sent':False})
        changed=list(self.pool.map(self._poll,list(self.active.items())))
        acknowledgements=[]
        for uid,value in changed:
            old=self.active[uid]
            if value is None:
                (self.owned/(uid+'.json')).unlink(missing_ok=True);del self.active[uid]
            else:
                if value['index']>old['index']:
                    acknowledgements.append(f"responses/{uid}/{old['index']:06d}.pkl")
                self.active[uid]=value
        # Commit local receipt progress before removing remote result bytes.
        atomic_json(self.state,self.active)
        for key in acknowledgements:self.store.remove(key)
        list(self.pool.map(self._worker,{v['worker']for v in self.active.values()if v.get('worker')}))


class RelayNativeBridge(NativeBridge):
    def __init__(self,root,store):
        super().__init__(root,store);self.last_claims={};self.last_workers={}

    def tick(self):
        super().tick();workers=set()
        for uid in self.active:
            path=self.root/'episodes'/uid/'claimed.json'
            if not path.exists():continue
            raw=path.read_bytes();workers.add(worker_name(json.loads(raw)['worker_id']))
            if self.last_claims.get(uid)!=raw:
                self.store.put(f'claims/{uid}.json',raw);self.last_claims[uid]=raw
        self.last_claims={k:v for k,v in self.last_claims.items()if k in self.active}
        for name in workers:
            path=self.root/'workers'/(name+'.json')
            if not path.exists():continue
            raw=path.read_bytes()
            if self.last_workers.get(name)!=raw:
                self.store.put(f'workers/{name}.json',raw);self.last_workers[name]=raw


def main():
    import argparse,fcntl,os,subprocess,sys,time
    parser=argparse.ArgumentParser();parser.add_argument('--spec',type=Path,default=Path(__file__).with_name('spec.json'))
    spec=json.loads(parser.parse_args().spec.read_text());root=Path(spec['root'])
    if spec.get('mode')not in {'model','native'}:raise ValueError('Invalid file relay mode')
    credential=Path('/dev/shm/rp_jhh_file_relay_credentials.json')
    descriptor=os.open(credential,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(descriptor,'w')as stream:stream.write(os.environ.pop('RP_JHH_PRIVATE_ARCHIVE_CONFIGURATION'))
    os.environ['PEAR_OBJECTSTORE_CREDENTIAL_FILE']=str(credential)
    deps=Path('/tmp/rp_jhh_file_relay_packages')
    subprocess.run([sys.executable,'-m','pip','install','--quiet','--target',str(deps),'boto3==1.44.5'],check=True,timeout=300)
    sys.path.insert(0,str(deps))
    from efe_pear_object_archive import configured_client
    from efe_pear_object_transport import Store
    store=Store(spec['uri'],configured_client());model=spec['mode']=='model'
    bridge=FileModelRelay(root,store,spec['kinds'],workers=32)if model else RelayNativeBridge(root,store)
    with(root/('object-model-relay.lock'if model else'object-bridge.lock')).open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        last={};reported=0
        try:
            while True:
                if model:
                    for name in ['head-heartbeat.json','finished.json']:
                        path=root.parent/name
                        if not path.exists():continue
                        raw=path.read_bytes();value=json.loads(raw)
                        if name=='head-heartbeat.json'and(value.get('workflow')!=spec['owner_model_workflow']
                            or time.time()-value['updated_unix']>300):
                            raise RuntimeError('Model owner heartbeat is absent or stale')
                        if last.get(name)!=raw:store.put('control/'+name,raw);last[name]=raw
                bridge.tick()
                if time.monotonic()-reported>60:
                    print('PEAR_FILE_RELAY '+json.dumps({'mode':spec['mode'],'active_episodes':len(bridge.active),
                        'updated_unix':time.time()}),flush=True);reported=time.monotonic()
                if(root.parent/'finished.json').exists():break
                time.sleep(.2)
        finally:
            if model:bridge.close()
            credential.unlink(missing_ok=True)


if __name__=='__main__':main()
