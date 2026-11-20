"""Run a separate teacher-native child in an otherwise idle retained allocation."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from uuid import uuid4

from efe_astra_wide_collect import collect
from efe_pear_file_rpc import atomic_json


def fail_owned_requests(spec, error):
    """Unblock this worker's collector only when its own native child died."""
    root=Path(spec['teacher_root']);names=set(spec['teacher_task_names']);count=0
    for path in root.glob('episodes/*/infra-*/result.json'):
        result=json.loads(path.read_text())
        if result.get('status')not in {'opening','running'}or result['case']['task']not in names:continue
        uid=result.get('episode_id')
        if not uid:continue
        episode=Path(spec['transport_root'])/'episodes'/uid
        if episode.is_dir():
            atomic_json(episode/'infrastructure-error.json',{'error':error,'native_child_exit':True,
                'updated_unix':time.time()});count+=1
    return count


def main():
    lane=Path(__file__).resolve().parent;spec=json.loads((lane/'spec.json').read_text())
    root=Path(spec['teacher_root']);root.mkdir(parents=True,exist_ok=True,mode=0o700)
    provider_key=os.environ.pop('NVIDIA_INFERENCE_API_KEY',None)
    if provider_key:
        credential=Path(spec['teacher_credential_file'])
        temporary=credential.with_name(credential.name+'.'+uuid4().hex+'.tmp')
        descriptor=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(descriptor,'w')as stream:stream.write(provider_key)
        temporary.replace(credential)
        del provider_key
    stopped=threading.Event()
    def heartbeat():
        while not stopped.is_set():
            atomic_json(root/'head-heartbeat.json',{'workflow':spec['owner_model_workflow'],
                'updated_unix':time.time(),'phase':'independent_astra_teacher_collection'})
            stopped.wait(10)
    thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
    log=root/(os.environ['EFE_WORKFLOW_ID']+'-native.log')
    child=None;watcher=None
    try:
        if spec['teacher_native_entry']!='reasoning':
            with log.open('a')as stream:
                child=subprocess.Popen([sys.executable,'-u',str(lane/spec['teacher_native_entry'])],
                    stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            def watch_child():
                while not stopped.wait(2):
                    if child.poll()is not None:
                        fail_owned_requests(spec,'Owned teacher native bootstrap exited with code '+str(child.returncode))
            watcher=threading.Thread(target=watch_child,daemon=True);watcher.start()
        if not spec.get('teacher_native_only'):collect(spec)
        atomic_json(root/(os.environ['EFE_WORKFLOW_ID']+'-collection-complete.json'),
                    {'complete':True,'updated_unix':time.time()})
        # Keep native lanes and the allocation available for an explicit next
        # pass; a completed collection never tears down the retained cluster.
        while child is None or child.poll()is None:time.sleep(10)
        raise RuntimeError('Teacher native child exited; retained allocation is available for recovery')
    finally:
        stopped.set();thread.join(timeout=2)
        if watcher is not None:watcher.join(timeout=3)
        if child is not None and child.poll()is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=60)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()


if __name__=='__main__':main()
