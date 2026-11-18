"""Two-node OSMO/Ray launcher for an isolated discarded PEAR pipeline check."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

import yaml


def atomic(path,value):
    temporary=path.with_suffix('.'+os.environ['EFE_ROLE']+'.tmp')
    temporary.write_text(json.dumps(value)+'\n');temporary.replace(path)


def main():
    lane=Path(__file__).resolve().parent;spec=json.loads((lane/'spec.json').read_text())
    role=os.environ['EFE_ROLE'];root=Path(spec['output'])/os.environ['EFE_WORKFLOW_ID']
    root.mkdir(mode=0o700,parents=True,exist_ok=True)
    def record(phase,**extra):
        value={'phase':phase,'role':role,'primary_rl_updates':0,**extra}
        atomic(root/(role+'-status.json'),value);print('PEAR_ONPOLICY_CHECK '+json.dumps(value),flush=True)
    runtime=Path(spec['remote_root'])/'runtime';source=Path(spec['pear_source'])
    python=runtime/'envs/verl/bin/python';bundle=root/'adapters';deadline=time.monotonic()+4*3600
    subprocess.run(['apt-get','update','-qq'],check=True,timeout=300)
    subprocess.run(['apt-get','install','-y','-qq','libosmesa6','libgl1','libglib2.0-0'],check=True,timeout=300)
    if role=='trainer':
        membership=Path(spec['reasoning_membership'])
        if hashlib.sha256(membership.read_bytes()).hexdigest()!=spec['reasoning_membership_sha256']:
            raise ValueError('Qualified reasoning membership changed')
        bundle.mkdir(exist_ok=True)
        for name,digest in spec['adapter_digests'].items():
            if hashlib.sha256((lane/name).read_bytes()).hexdigest()!=digest:raise ValueError('Adapter bytes changed')
            shutil.copyfile(lane/name,bundle/name)
        bank=json.loads(membership.read_text())
        for partition,env_name in [('train','ReasoningEnv'),('development','ReasoningDevelopmentEnv')]:
            rows=[v for v in bank['scenes'].values()if v['partition']==partition]
            atomic(root/(partition+'.json'),{'env_class':'efe_pear_reasoning_env.'+env_name,'data_source':'family',
                'env':[{k:[v[k]]for k in ['family','seed','partition']}for v in rows]})
        tools={'tools':[{'class_name':'efe_pear_reasoning_env.ReasoningTool','config':{
            'type':'native','env_class':'efe_pear_reasoning_env.'+name,'server_n':count,
            'log_dir':str(root/'sim')}}for name,count in [('ReasoningEnv',6),('ReasoningDevelopmentEnv',2)]]}
        (root/'tools.yaml').write_text(yaml.safe_dump(tools,sort_keys=False))
        token=root/'ray-auth';token.write_text(os.urandom(32).hex());token.chmod(0o600)
        atomic(root/'launch.json',{'head_host':os.environ['EFE_HEAD_HOST'],'port':6379})
    else:
        while not(root/'launch.json').exists():
            if time.monotonic()>deadline:raise TimeoutError('Head staging timed out')
            time.sleep(2)
    head=json.loads((root/'launch.json').read_text());address=head['head_host']+':6379'
    env={**os.environ,'WORK_DIR':str(runtime),'PEAR_SOURCE':str(source),'RUN':str(root),'MODEL':spec['model'],
        'LOCAL_DIR':'/tmp/rp-jhh-pear-check','RAY_ADDRESS':address,
        'ADAPTER_ROOT':str(bundle),'RAY_AUTH_MODE':'token','RAY_AUTH_TOKEN_PATH':str(root/'ray-auth'),
        'PYTHONPATH':':'.join(map(str,[bundle,source/'pear/utils/sitecustomize',source,source/'3rdparty/verl/src'])),
        'EFE_PEAR_REASONING_MEMBERSHIP':spec['reasoning_membership'],
        'EFE_PEAR_QUALIFICATION_DIR':str(root),'EFE_PEAR_METRICS_DIR':str(root/'telemetry'),
        'EFE_PEAR_OPTIMIZER_CHECK_DIR':str(root/'optimizer-checks'),
        'PEAR_PRIMARY_TRAINING':'0','PEAR_LOG_DIR':str(root/'sim'),
        'NCCL_ALGO':'Ring','NCCL_PROTO':'Simple','NCCL_P2P_DISABLE':'1',
        'RAY_DEDUP_LOGS':'0','TOKENIZERS_PARALLELISM':'false','MUJOCO_GL':'osmesa',
        'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
    # Configure the very same runtime libraries/caches that the training driver
    # uses before Ray starts its native children.
    command='source "$PEAR_SOURCE/scripts/gpu_process_variables.sh" "$WORK_DIR/envs/verl"\nexec "$WORK_DIR/envs/verl/bin/ray" "$@"'
    ray_args=['start','--block','--disable-usage-stats','--num-cpus=56','--num-gpus=8',
        '--resources='+json.dumps({role:8,**({'reasoning':8}if role=='rollout'else{})}),
        '--object-store-memory='+str(16<<30),'--min-worker-port=3000','--max-worker-port=5999',
        '--temp-dir=/tmp/rp-jhh-pear-check/ray']
    if role=='trainer':ray_args.extend(['--head','--port=6379','--include-dashboard=false'])
    else:
        while True:
            try:
                with socket.create_connection((head['head_host'],6379),timeout=3):break
            except OSError:
                if time.monotonic()>deadline:raise TimeoutError('Head network connection timed out')
                time.sleep(2)
        record('head_network_connection_passed',head_host=head['head_host'])
        ray_args.append('--address='+address)
    record('starting_'+role+'_ray_node')
    ray_process=subprocess.Popen(['bash','-c',command,'ray-start',*ray_args],env=env,start_new_session=True)
    try:
        if role=='trainer':
            check='''import time,ray,json
ray.init(address="auto")
deadline=time.monotonic()+900
while time.monotonic()<deadline:
    resources=ray.cluster_resources()
    if resources.get("trainer",0)>=8 and resources.get("rollout",0)>=8:
        print("PEAR_CLUSTER_READY "+json.dumps(resources),flush=True);break
    time.sleep(2)
else:raise TimeoutError("Both Ray GPU roles did not join")
ray.shutdown()
'''
            subprocess.run([str(python),'-u','-c',check],env=env,check=True,timeout=1200)
            record('two_node_gpu_cluster_ready',gpus=16)
            result=subprocess.run(['bash',str(bundle/'efe_pear_onpolicy_check.sh')],env=env,timeout=3*3600)
            atomic(root/'finished.json',{'returncode':result.returncode,'primary_rl_updates':0})
            if result.returncode:raise RuntimeError('On-policy qualification process failed')
        else:
            while not(root/'finished.json').exists():
                if ray_process.poll()is not None:raise RuntimeError('Rollout Ray node exited prematurely')
                if time.monotonic()>deadline:raise TimeoutError('Qualification exceeded four hours')
                time.sleep(5)
    finally:
        if ray_process.poll()is None:
            os.killpg(ray_process.pid,signal.SIGTERM)
            try:ray_process.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(ray_process.pid,signal.SIGKILL);ray_process.wait()


if __name__=='__main__':main()
