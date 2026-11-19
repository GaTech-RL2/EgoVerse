"""Run a separate teacher-native child in an otherwise idle retained allocation."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from efe_astra_wide_collect import collect
from efe_pear_file_rpc import atomic_json


def main():
    lane=Path(__file__).resolve().parent;spec=json.loads((lane/'spec.json').read_text())
    root=Path(spec['teacher_root']);root.mkdir(parents=True,exist_ok=True,mode=0o700)
    provider_key=os.environ.pop('NVIDIA_INFERENCE_API_KEY',None)
    if provider_key:
        credential=Path(spec['teacher_credential_file'])
        descriptor=os.open(credential,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(descriptor,'w')as stream:stream.write(provider_key)
        del provider_key
    stopped=threading.Event()
    def heartbeat():
        while not stopped.is_set():
            atomic_json(root/'head-heartbeat.json',{'workflow':spec['owner_model_workflow'],
                'updated_unix':time.time(),'phase':'independent_astra_teacher_collection'})
            stopped.wait(10)
    thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
    log=root/(os.environ['EFE_WORKFLOW_ID']+'-native.log')
    child=None
    try:
        with log.open('a')as stream:
            child=subprocess.Popen([sys.executable,'-u',str(lane/spec['teacher_native_entry'])],
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        collect(spec)
        atomic_json(root/(os.environ['EFE_WORKFLOW_ID']+'-collection-complete.json'),
                    {'complete':True,'updated_unix':time.time()})
        # Keep native lanes and the allocation available for an explicit next
        # pass; a completed collection never tears down the retained cluster.
        while child.poll()is None:time.sleep(10)
        raise RuntimeError('Teacher native child exited; retained allocation is available for recovery')
    finally:
        stopped.set();thread.join(timeout=2)
        if child is not None and child.poll()is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=60)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()


if __name__=='__main__':main()
