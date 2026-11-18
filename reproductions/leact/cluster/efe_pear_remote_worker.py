"""Keep an allocated native simulator lane serving constrained PEAR episodes."""
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time

from efe_pear_file_rpc import atomic_json,claim_episode,serve_episode


def serve(root,worker_id,kinds,factory,*,finished):
    root=Path(root);root.mkdir(mode=0o700,parents=True,exist_ok=True)
    if root.stat().st_mode&0o077:raise ValueError('Native transport root must be private')
    (root/'pending').mkdir(mode=0o700,exist_ok=True)
    workers=root/'workers';workers.mkdir(mode=0o700,exist_ok=True)
    stopped=threading.Event();head_lost=threading.Event();active=[None]
    state={'phase':'ready','episodes':0,'errors':0}
    def heartbeat():
        while not stopped.is_set():
            heartbeat=finished.with_name('head-heartbeat.json')
            if heartbeat.exists()and time.time()-heartbeat.stat().st_mtime>300:head_lost.set()
            if(finished.exists()or head_lost.is_set())and active[0]is not None:
                atomic_json(active[0]/'cancel.json',{'native_worker_stopping':True})
            atomic_json(workers/(worker_id+'.json'),{'worker_id':worker_id,'updated_unix':time.time(),
                'kinds':sorted(kinds),**state})
            stopped.wait(10)
    thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
    try:
        while not finished.exists()and not head_lost.is_set():
            claimed=claim_episode(root,worker_id,kinds=kinds)
            if claimed is None:time.sleep(.1);continue
            folder,request=claimed;active[0]=folder;state['phase']='native_episode'
            try:
                serve_episode(folder,factory,request['env_args']);state['episodes']+=1;state['consecutive_errors']=0
            except Exception as exc:
                state['errors']+=1
                state['consecutive_errors']=state.get('consecutive_errors',0)+1
                print('PEAR_NATIVE_EPISODE_INFRASTRUCTURE_ERROR '+json.dumps({'error_type':type(exc).__name__,
                    'worker_id':worker_id,'episodes':state['episodes'],'errors':state['errors']}),flush=True)
                if state['consecutive_errors']>=5:
                    raise RuntimeError('Native simulator repeatedly failed to complete an episode')from exc
            finally:state['phase']='ready';active[0]=None
    finally:
        stopped.set();thread.join(timeout=2)
        atomic_json(workers/(worker_id+'.json'),{'worker_id':worker_id,'updated_unix':time.time(),
            'kinds':sorted(kinds),**state,'phase':'stopped'})


def main():
    lane=Path(__file__).resolve().parent
    path=Path(sys.argv[1])if len(sys.argv)>1 else lane/'spec.json'
    spec=json.loads(path.read_text())
    config=json.loads(Path(spec['config']).read_text())
    if spec['kind']=='rtx':
        sys.path.insert(0,str(Path(config['source'])/'standard'))
        os.environ['EFE_PEAR_RTX_CONFIG']=spec['config']
        from efe_pear_rtx_env import NativeRtxEpisode
        factory=NativeRtxEpisode
    elif spec['kind']=='efe':
        os.environ['EFE_PEAR_NATIVE_CONFIG']=spec['config']
        from efe_pear_native_episode import NativeEfeEpisode
        factory=NativeEfeEpisode
    else:raise ValueError('Unknown native simulator lane kind')
    serve(spec['transport_root'],spec['worker_id'],set(spec['kinds']),factory,
        finished=Path(spec['finished']))


if __name__=='__main__':main()
