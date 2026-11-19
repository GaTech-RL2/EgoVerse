"""A real full-mixture PEAR model cluster with one trainer and rollout nodes."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time

import yaml

from efe_pear_file_rpc import atomic_json


def prepare_system_packages(sources=Path('/etc/apt/sources.list.d'),
    retries=Path('/etc/apt/apt.conf.d/99rp_jhh_retries')):
    # The optional Python PPA returned HTTP 503 during a real model-node
    # startup. None of the required Ubuntu Git/GL packages comes from it.
    for path in sources.iterdir():
        if'deadsnakes'in path.name and path.suffix in {'.list','.sources'}:
            path.rename(path.with_name(path.name+'.disabled'))
    retries.write_text('Acquire::Retries "5";\n')
    subprocess.run(['apt-get','update','-qq'],check=True,timeout=300)
    subprocess.run(['apt-get','install','-y','-qq','git','perl','libosmesa6','libgl1','libglib2.0-0'],check=True,timeout=300)


def main():
    lane=Path(__file__).resolve().parent;spec=json.loads((lane/'spec.json').read_text())
    role=os.environ['EFE_ROLE'];head=role=='trainer';run=Path(spec['run'])
    run.mkdir(mode=0o700,parents=True,exist_ok=True)
    attempt=run/'attempts'/os.environ['EFE_WORKFLOW_ID']
    execution=os.environ.get('EFE_PEAR_ATTEMPT_ID')
    if execution:attempt=attempt/execution
    attempt.mkdir(parents=True,exist_ok=True)
    bundle=run/'adapters';source=Path(spec['pear_source']);runtime=Path(spec['runtime']);python=runtime/'envs/verl/bin/python'
    deadline=time.monotonic()+48*3600;stopped=threading.Event();phase={'phase':'initializing'}
    object_transport={'store':None}
    def record(stage,**fields):
        phase['phase']=stage
        value={'phase':stage,'role':role,'updated_unix':time.time(),**fields}
        atomic_json(attempt/(role+'-status.json'),value)
        print('PEAR_PRIMARY_STATUS '+json.dumps(value),flush=True)
    def heartbeat():
        while not stopped.is_set():
            value={'updated_unix':time.time(),'workflow':os.environ['EFE_WORKFLOW_ID'],**phase}
            atomic_json(run/'head-heartbeat.json',value)
            if object_transport['store']is not None:
                try:object_transport['store'].put_json('control/head-heartbeat.json',value)
                except Exception as exc:
                    print('PEAR_OBJECT_HEARTBEAT_RETRY '+json.dumps({'error_type':type(exc).__name__}),flush=True)
            stopped.wait(10)
    credential=None;shared_credential=None
    if head:
        threading.Thread(target=heartbeat,daemon=True).start()
        (run/'finished.json').unlink(missing_ok=True)
        bundle.mkdir(exist_ok=True)
        for name,expected in spec['adapter_digests'].items():
            if hashlib.sha256((lane/name).read_bytes()).hexdigest()!=expected:raise ValueError('Frozen primary adapter changed')
            shutil.copyfile(lane/name,bundle/name)
        for name,expected in spec['data_digests'].items():
            if hashlib.sha256((lane/name).read_bytes()).hexdigest()!=expected:raise ValueError('Frozen primary admission file changed')
            shutil.copyfile(lane/name,run/name)
        transport=run/'transport';transport.mkdir(mode=0o700,exist_ok=True)
        if transport.stat().st_mode&0o077:raise ValueError('The native transport directory must be private')
        if any(p.is_dir()and p!=attempt for p in(attempt.parent).iterdir()):
            # The scheduler confirmed the prior model attempt terminated before
            # a replacement was submitted. Release its orphaned native waits;
            # restored PEAR prompts create fresh simulator episode identities.
            for folder in(transport/'episodes').glob('*'):
                if folder.is_dir():atomic_json(folder/'cancel.json',{'cancelled_after_model_restart':True})
            for pending in(transport/'pending').glob('*.json'):pending.unlink(missing_ok=True)
            # Only disposable staging copies from the terminated attempt live
            # here. Committed checkpoints remain in private object storage.
            shutil.rmtree(run/'queue-checkpoint-staging',ignore_errors=True)
        atomic_json(run/'native-config.json',{'qualification_only':False,'membership_file':str(run/'native-membership.json')})
        curriculum=json.loads((run/'curriculum.json').read_text())
        datasets=[]
        for index,env_class in enumerate(sorted({v['env_class']for v in curriculum['configs']})):
            rows=[v for v in curriculum['configs']if v['env_class']==env_class]
            env=[{k:[v]for k,v in row['env_args'].items()}for row in rows]
            path=run/f'train-{index}.json'
            atomic_json(path,{'env_class':env_class,'data_source':'family'if 'ReasoningEnv'in env_class else'task','env':env})
            datasets.append(str(path))
        bank=json.loads((run/'reasoning-membership.json').read_text())
        dev=[v for v in bank['scenes'].values()if v['partition']=='development']
        atomic_json(run/'development.json',{'env_class':'efe_pear_reasoning_env.ReasoningDevelopmentEnv','data_source':'family',
            'env':[{k:[v[k]]for k in ['family','seed','partition']}for v in dev]})
        tools=[]
        for cls,name,count in [('efe_pear_remote_env.RemoteNativeEnv','efe_pear_native_tool.NativeEfeTool',32),
            ('efe_pear_remote_env.NativeYamEnv','efe_pear_native_tool.NativeEfeTool',128),
            ('efe_pear_reasoning_env.ReasoningEnv','efe_pear_reasoning_tool.ReasoningTool',6),
            ('efe_pear_reasoning_env.ReasoningDevelopmentEnv','efe_pear_reasoning_tool.ReasoningTool',2)]:
            tools.append({'class_name':name,'config':{'type':'native','env_class':cls,'server_n':count,'log_dir':str(run/'sim')}})
        (run/'tools.yaml').write_text(yaml.safe_dump({'tools':tools},sort_keys=False))
        credential=Path('/dev/shm/rp_jhh_pear_archive_credentials'+('_'+execution if execution else'')+'.json')
        descriptor=os.open(credential,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(descriptor,'w')as stream:stream.write(os.environ.pop('RP_JHH_PRIVATE_ARCHIVE_CONFIGURATION'))
        if spec.get('object_transport_uri'):
            shared_credential=attempt/'objectstore-credentials.json'
            descriptor=os.open(shared_credential,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
            with os.fdopen(descriptor,'w')as stream:stream.write(credential.read_text())
            atomic_json(run/'l40-transport.json',{'uri':spec['object_transport_uri'],
                'credential_file':str(shared_credential)})
        auth=attempt/'ray-auth';auth.write_text(os.urandom(32).hex());auth.chmod(0o600)
        atomic_json(attempt/'launch.json',{'head_host':os.environ['EFE_HEAD_HOST'],'train_files':datasets,'auth':str(auth)})
    else:
        while not(attempt/'launch.json').exists():
            if time.monotonic()>deadline:raise TimeoutError('Primary model head did not stage its launch')
            time.sleep(2)
    launch=json.loads((attempt/'launch.json').read_text());address=launch['head_host']+':6379'
    env={**os.environ,'WORK_DIR':str(runtime),'PEAR_SOURCE':str(source),'RUN':str(run),
        'PEAR_RUN_ID':spec['run_id'],'MODEL':spec['model'],'MODEL_REVISION':spec['model_revision'],
        'MODEL_QUALIFICATION':str(run/'model-qualification.json'),'RAY_ADDRESS':address,
        'EPISODES':json.dumps(launch['train_files']),'VALIDATION_EPISODES':str(run/'development.json'),
        'TRAINING_STEPS':str(spec['training_steps']),'TRAINER_NNODES':'1','ROLLOUT_NNODES':str(spec['rollout_nodes']),
        'EFE_PEAR_NATIVE_CONFIG':str(run/'native-config.json'),'EFE_PEAR_CURRICULUM':str(run/'curriculum.json'),
        'EFE_PEAR_SAMPLING_CONTROL':str(run/'curriculum-sampling.json'),
        'EFE_PEAR_REASONING_MEMBERSHIP':str(run/'reasoning-membership.json'),'EFE_PEAR_REMOTE_ROOT':str(run/'transport'),
        'TOOL_CONFIG':str(run/'tools.yaml'),'EFE_PEAR_METRICS_DIR':str(run/'telemetry'),
        'PEAR_LOG_DIR':str(run/'sim'),'PEAR_TRACE_FRAMES':'1','PEAR_PRIMARY_TRAINING':'1',
        'PEAR_CHECKPOINT_ARCHIVE':spec['checkpoint_archive']+'/attempts/rp_jhh_'+hashlib.sha256((os.environ['EFE_WORKFLOW_ID']+'/'+(execution or'')).encode()).hexdigest()[:12],
        'PEAR_CHECKPOINT_SCRATCH':'/dev/shm/rp_jhh_pear_checkpoints',
        'PEAR_OBJECTSTORE_CREDENTIAL_FILE':'/dev/shm/rp_jhh_pear_archive_credentials'+('_'+execution if execution else'')+'.json',
        'LOCAL_DIR':'/tmp/rp-jhh-pear-primary','RAY_AUTH_MODE':'token','RAY_AUTH_TOKEN_PATH':launch['auth'],
        'PYTHONPATH':':'.join(map(str,[bundle,source/'pear/utils/sitecustomize',source,source/'3rdparty/verl/src'])),
        'NCCL_ALGO':'Ring','NCCL_PROTO':'Simple','NCCL_P2P_DISABLE':'1','RAY_DEDUP_LOGS':'0',
        'TOKENIZERS_PARALLELISM':'false','MUJOCO_GL':'osmesa','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
    if spec.get('object_transport_uri'):
        env['EFE_PEAR_OBJECT_TRANSPORT_CONFIG']=str(run/'l40-transport.json')
        env['PYTHONPATH']=str(attempt/'object-store-packages')+':'+env['PYTHONPATH']
    env.pop('PEAR_DRIVER_PREFLIGHT_ONLY',None)
    prepare_system_packages()
    # The storage client is absent from some pinned training environments. Use
    # an isolated target overlay instead of changing the qualified shared venv.
    if head:
        dependencies=Path('/tmp/rp_jhh_object_store_packages')
        subprocess.run([sys.executable,'-m','pip','install','--quiet','--target',str(dependencies),'boto3==1.44.5'],check=True,timeout=300)
        sys.path.insert(0,str(dependencies))
        env['PYTHONPATH']=str(dependencies)+':'+env['PYTHONPATH']
        if spec.get('object_transport_uri'):
            shutil.copytree(dependencies,attempt/'object-store-packages')
            from efe_pear_object_archive import configured_client
            from efe_pear_object_transport import Store
            object_transport['store']=Store(spec['object_transport_uri'],configured_client(credential))
    if not head:
        while True:
            try:
                with socket.create_connection((launch['head_host'],6379),timeout=3):break
            except OSError:
                if time.monotonic()>deadline:raise TimeoutError('Primary head network connection timed out')
                time.sleep(2)
    command='cd "$PEAR_SOURCE"\nsource "$PEAR_SOURCE/scripts/gpu_process_variables.sh" "$WORK_DIR/envs/verl"\nexec "$WORK_DIR/envs/verl/bin/ray" "$@"'
    resources=({'trainer':8,'pear_driver':1}if head else
        {'rollout':8,'reasoning':8,'native_proxy':16,'native_yam_proxy':spec['rollout_ray_cpus']})
    ray_cpus=spec['trainer_ray_cpus']if head else spec['rollout_ray_cpus']
    args=['start','--block','--disable-usage-stats','--num-cpus='+str(ray_cpus),'--num-gpus=8',
        '--resources='+json.dumps(resources),'--object-store-memory='+str(16<<30),
        '--min-worker-port=3000','--max-worker-port=5999','--temp-dir=/tmp/rp-jhh-pear-primary/ray']
    args+=['--head','--port=6379','--include-dashboard=false']if head else['--address='+address]
    record('starting_ray_node',gpus=8)
    ray_env=dict(env);ray_env.pop('RAY_ADDRESS',None)
    ray_process=subprocess.Popen(['bash','-c',command,'ray-start',*args],env=ray_env,start_new_session=True)
    try:
        if head:
            code='''import ray,time
ray.init(address="auto")
deadline=time.monotonic()+1200
while time.monotonic()<deadline:
 if ray.cluster_resources().get("rollout",0)>='''+str(8*spec['rollout_nodes'])+''':break
 time.sleep(2)
else:raise TimeoutError("Not all model rollout nodes joined")
ray.shutdown()
'''
            subprocess.run([str(python),'-u','-c',code],env=env,check=True,timeout=1500)
            resume=[];pointer=run/'checkpoint-latest.json'
            if pointer.exists():
                latest=json.loads(pointer.read_text())
                if not latest['readback_verified']:raise ValueError('Latest checkpoint is not committed')
                target=Path(env['PEAR_CHECKPOINT_SCRATCH'])/f"global_step_{latest['step']}"
                from efe_pear_object_archive import restore
                os.environ['PEAR_OBJECTSTORE_CREDENTIAL_FILE']=str(credential)
                restore(latest['archive'],target)
                resume=['trainer.resume_mode=resume_path','trainer.resume_from_path='+str(target)]
            else:resume=['trainer.resume_mode=disable']
            record('running_full_mixture_pear',model_nodes=1+spec['rollout_nodes'])
            result=subprocess.run(['bash',str(bundle/'efe_pear_scaled_train.sh'),*resume,
                'trainer.rollout_data_dir='+str(run/'rollout-records')],env=env,timeout=47*3600)
            atomic_json(run/'finished.json',{'returncode':result.returncode,'completed':result.returncode==0})
            if object_transport['store']is not None:
                object_transport['store'].put_json('control/finished.json',
                    {'returncode':result.returncode,'completed':result.returncode==0})
            if result.returncode:raise RuntimeError('Primary PEAR driver failed')
        else:
            while not(run/'finished.json').exists():
                if ray_process.poll()is not None:raise RuntimeError('Primary rollout Ray node exited prematurely')
                if time.monotonic()>deadline:raise TimeoutError('Primary cluster exceeded its two-day execution bound')
                time.sleep(5)
    finally:
        stopped.set()
        if ray_process.poll()is None:
            os.killpg(ray_process.pid,signal.SIGTERM)
            try:ray_process.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(ray_process.pid,signal.SIGKILL);ray_process.wait()
        if credential:credential.unlink(missing_ok=True)
        if shared_credential:shared_credential.unlink(missing_ok=True)


if __name__=='__main__':main()
