"""Bootstrap isolated native EFE lanes without changing the model runtime."""
import json
import hashlib
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from efe_pear_file_rpc import atomic_json


def runtime_root(workflow):
    return Path('/tmp/efe-native-jobs')/hashlib.sha256(workflow.encode()).hexdigest()[:16]


def wait_for_model_head(spec,timeout=1800):
    """A prior attempt's completion marker must not stop a new allocation."""
    finished=Path(spec['finished']);heartbeat=Path(spec.get('head_heartbeat',finished.with_name('head-heartbeat.json')))
    deadline=time.monotonic()+timeout
    print('PEAR_PRIMARY_NATIVE_WAITING_FOR_OWNER '+spec['owner_model_workflow'],flush=True)
    while True:
        try:value=json.loads(heartbeat.read_text())
        except(FileNotFoundError,ValueError):value={}
        if(value.get('workflow')==spec['owner_model_workflow']
            and time.time()-value.get('updated_unix',0)<120 and not finished.exists()):return
        if time.monotonic()>=deadline:raise TimeoutError('Current model head has not initialized its run')
        time.sleep(2)


def main():
    import efe_native_pilot as native
    lane=Path(__file__).resolve().parent;spec=json.loads((lane/'spec.json').read_text())
    expected_gpus=spec.get('native_gpu_count',8)
    if type(expected_gpus)is not int or expected_gpus not in {2,4,8}:
        raise ValueError('Native EFE GPU count must be explicitly two, four, or the default eight')
    selected_devices=ownership=None
    if expected_gpus!=8:
        from efe_pear_native_gpu_shape import assigned_devices
        # Prove the scheduler assignment before Podman setup or native startup.
        selected_devices,ownership=assigned_devices(expected_gpus)
    wait_for_model_head(spec)
    output=Path(spec['output'])/os.environ['EFE_WORKFLOW_ID'];output.mkdir(parents=True,exist_ok=True)
    (output/'logs').mkdir()
    if ownership:atomic_json(output/'native-gpu-ownership.json',ownership)
    # A retained node can still have conmon processes mapping the prior job's
    # executable. Stage this job's tools, source and socket in a fresh root,
    # while reusing the pinned image-layer cache rather than downloading it.
    native.ROOT=runtime_root(os.environ['EFE_WORKFLOW_ID'])
    native.ARCHIVE=output
    service=None;log=None;children=[]
    try:
        reference,env,service,log=native.setup({**spec['native_setup'],'container_storage_root':'/tmp/efe-native'})
        if spec.get('control_interface'):
            from efe_pear_eef import enabled
            enabled(spec['control_interface'])
            from efe_pear_vlabench_eef import stage
            overlays=stage(reference,spec['native_setup']['tasks'],lane/'efe_pear_vlabench_eef.py')
            atomic_json(output/'eef-native-source-overlays.json',overlays)
        from efe_pear_gpu_cdi import ensure_gpu_cdi
        cdi_receipt=ensure_gpu_cdi(native.run)
        if ownership:
            from efe_pear_native_gpu_shape import verify_cdi
            ownership['verified_cdi_devices']=verify_cdi(selected_devices,cdi_receipt)
            atomic_json(output/'native-gpu-ownership.json',ownership)
        raw=os.environ.pop('RP_JHH_PRIVATE_ARCHIVE_CONFIGURATION')
        credential=Path('/dev/shm/rp_jhh_native_archive_credentials.json')
        descriptor=os.open(credential,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(descriptor,'w')as stream:stream.write(raw)
        # Native bootstrap installs its packages in this task's own Python;
        # it never installs Harbor into the qualified shared training venv.
        subprocess.run([sys.executable,'-m','pip','install','--quiet','boto3==1.44.5'],check=True,timeout=300)
        env.pop('RP_JHH_PRIVATE_ARCHIVE_CONFIGURATION',None)
        env.update(PEAR_OBJECTSTORE_CREDENTIAL_FILE=str(credential),
            PEAR_EPISODE_ARCHIVE=spec['episode_archive'],PEAR_PRIMARY_TRAINING='1',
            PYTHONPATH=str(lane)+':'+env.get('PYTHONPATH',''),OMP_NUM_THREADS='2')
        devices=selected_devices if selected_devices is not None else subprocess.check_output(
            ['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()
        if len(devices)!=expected_gpus:raise ValueError('The native EFE allocation must own its requested GPU count')
        lanes_per_gpu=spec.get('lanes_per_gpu',2)
        stagger=spec.get('startup_stagger_seconds',0)
        if type(lanes_per_gpu)is not int or lanes_per_gpu not in {1,2}:
            raise ValueError('Native EFE lanes per GPU must be one or two')
        if type(stagger)not in {int,float}or not 0<=stagger<=30:
            raise ValueError('Native EFE startup stagger must be between zero and thirty seconds')
        workers=len(devices)*lanes_per_gpu
        kinds={'efe/'+suite for suite in spec['suites']}
        for index in range(workers):
            directory=output/f'lane-{index:02d}';directory.mkdir()
            config={**spec['native_config'],'reference':str(reference),'agent_source':str(lane),
                'episode_root':str(directory/'episodes'),'harbor':__import__('shutil').which('harbor',path=env['PATH']),
                'retain_transport_observations':False,
                'container_environment':{**env,'EFE_PODMAN_GPUS':'nvidia.com/gpu='+devices[index%len(devices)]}}
            if spec.get('native_membership_local'):
                config['membership_file']=str(lane/'native-membership.json')
            if spec.get('control_interface'):config['control_interface']=spec['control_interface']
            config_path=directory/'native-config.json'
            # No credential bytes are retained in configs: the inherited
            # archive environment contains only a node-local filename.
            atomic_json(config_path,config)
            worker={'kind':'efe','worker_id':os.environ['EFE_WORKFLOW_ID']+f'-lane-{index:02d}',
                'config':str(config_path),'kinds':sorted(kinds),'transport_root':spec['transport_root'],
                'finished':spec['finished'],'retry_native_infrastructure':spec.get('retry_native_infrastructure',False)}
            atomic_json(directory/'spec.json',worker)
            with(directory/'worker.log').open('w')as worker_log:
                command=[sys.executable,'-u',str(lane/'efe_pear_remote_worker.py'),str(directory/'spec.json')]
                child=subprocess.Popen(command,env={**env,'EFE_PODMAN_GPUS':'nvidia.com/gpu='+devices[index%len(devices)]},
                    stdout=worker_log,stderr=subprocess.STDOUT,start_new_session=True)
            children.append(child)
            if stagger and index+1<workers:time.sleep(stagger)
        print('PEAR_PRIMARY_NATIVE_READY '+json.dumps({'workers':workers,'gpu':len(devices),'primary_rl_updates':0,
            'lanes_per_gpu':lanes_per_gpu,'startup_stagger_seconds':stagger}),flush=True)
        while not Path(spec['finished']).exists():
            if any(p.poll()is not None for p in children):
                raise RuntimeError('A native EFE lane exited prematurely')
            heartbeat=Path(spec['head_heartbeat'])
            if heartbeat.exists()and time.time()-heartbeat.stat().st_mtime>300:
                raise RuntimeError('Primary model head heartbeat expired')
            time.sleep(5)
    finally:
        for child in children:
            if child.poll()is None:os.killpg(child.pid,signal.SIGTERM)
        for child in children:
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        if service:
            service.terminate()
            try:service.wait(timeout=10)
            except subprocess.TimeoutExpired:service.kill();service.wait()
        if log:log.close()


if __name__=='__main__':main()
