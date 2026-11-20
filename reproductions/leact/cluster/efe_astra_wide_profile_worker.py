"""Collect a new teacher profile while retaining the existing native servers."""
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time

from efe_astra_wide_collect import LegacyCaseGuards, collect, model_profile
from efe_pear_file_rpc import atomic_json


def main():
    lane=Path(__file__).resolve().parent;spec=json.loads((lane/'spec.json').read_text())
    profile=model_profile(spec)
    if profile['id']=='astra6-high-v1':raise ValueError('Sidecar must use the explicitly selected new teacher profile')
    if spec['teacher_native_entry']=='reasoning':
        packages=Path('/tmp/rp_jhh_astra_reasoning_dependencies')
        if not(packages/'ready').exists():raise RuntimeError('Retained reasoning dependencies are not ready')
        sys.path.insert(0,str(packages));os.environ.update(MUJOCO_GL='osmesa',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    root=Path(spec['teacher_root']);root.mkdir(mode=0o700,parents=True,exist_ok=True)
    worker=os.environ['EFE_WORKFLOW_ID'];status=root/'profile-workers'/(worker+'.json')
    status.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
    stopped=threading.Event();phase={'value':'draining_legacy_episode_boundaries'}
    def heartbeat():
        while not stopped.is_set():
            atomic_json(status,{'pid':os.getpid(),'profile':profile,'phase':phase['value'],
                'updated_unix':time.time(),'native_workers_restarted':False})
            stopped.wait(10)
    thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
    cases=[r for r in spec['teacher_plan']['cases']if r['task']in spec['teacher_task_names']]
    guards=LegacyCaseGuards(spec,cases)
    try:
        phase['value']='collecting_at_legacy_episode_boundaries'
        collect(spec,legacy_guards=guards)
        phase['value']='complete_native_servers_retained'
        atomic_json(root/(worker+'-collection-complete.json'),{'complete':True,'profile':profile,'updated_unix':time.time()})
        # Retain the legacy guards so old controller retries cannot restart the
        # superseded teacher profile. The underlying native servers stay live.
        while True:time.sleep(10)
    finally:
        stopped.set();thread.join(timeout=2);guards.close()


if __name__=='__main__':main()
