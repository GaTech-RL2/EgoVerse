"""Two CPU Ray nodes: save and fresh-process restore of the real rollout queue."""
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time


def exercise(phase,root):
    import ray
    import torch
    from tensordict import TensorDict
    from omegaconf import OmegaConf
    import transfer_queue as tq
    from transfer_queue.storage.managers.simple_storage_manager import _SU_SUBDIR,_SU_INFO_FILE
    from efe_pear_queue_checkpoint import shared_queue_checkpoint
    ray.init(address='auto',namespace='rp_jhh_queue_qualification')
    deadline=time.monotonic()+300
    while ray.cluster_resources().get('queue_check_storage',0)<2:
        if time.monotonic()>deadline:raise TimeoutError('Both CPU queue nodes did not join')
        time.sleep(1)
    nodes=[v['NodeID']for v in ray.nodes()if v['Alive']]
    assert len(nodes)==2
    tq.init(OmegaConf.create({'backend':{'storage_backend':'SimpleStorage','SimpleStorage':{
        'num_data_storage_units':2,'required_node_resource':'queue_check_storage'}},'metrics':{'enabled':False}}))
    keys=[f'rp_jhh_group_{i}'for i in range(16)]
    fields=TensorDict({'tokens':torch.arange(16*19).reshape(16,19),
        'pixels':torch.arange(16*3*8*8,dtype=torch.int64).remainder(256).to(torch.uint8).reshape(16,3,8,8),
        'responses':torch.nested.as_nested_tensor([torch.arange(i+4)for i in range(16)],layout=torch.jagged)},batch_size=16)
    local=Path('/tmp/rp_jhh_queue_head_only')/root.name/'transfer_queue'
    try:
        with shared_queue_checkpoint(tq,root/'shared-staging'):
            if phase=='save':
                tq.kv_batch_put(keys=keys,partition_id='train',fields=fields,
                    tags=[{'is_prompt':False,'status':'finished'}for _ in keys])
                tq.save_checkpoint(str(local),metadata={'step':4,'qualification_only':True})
                units=json.loads((local/_SU_SUBDIR/_SU_INFO_FILE).read_text())
                assert len(units)==2
                head=ray.get_runtime_context().get_node_id()
                if hasattr(head,'hex'):head=head.hex()
                other=next(v for v in nodes if v!=head)
                from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
                @ray.remote
                def visible(path):return Path(path).exists()
                assert ray.get(visible.options(num_cpus=.01,scheduling_strategy=
                    NodeAffinitySchedulingStrategy(other,soft=False)).remote(str(local)))is False
                (root/'queue-save.json').write_text(json.dumps({'nodes':nodes,'storage_units':len(units),
                    'head_local_path_invisible_on_other_node':True})+'\n')
            else:
                before=tq.kv_list('train');assert not before or not before.get('train')
                tq.load_checkpoint(str(local))
                restored=tq.kv_batch_get(keys=keys,partition_id='train')
                # TransferQueue returns jagged batches even for equal-length
                # rows. Compare native row tensors, independent of batching.
                for name in fields.keys():
                    for left,right in zip(restored[name].unbind(),fields[name].unbind(),strict=True):
                        assert left.dtype==right.dtype and torch.equal(left,right)
                assert not list((root/'shared-staging').iterdir())
    finally:tq.close();ray.shutdown()


def main():
    lane=Path(__file__).resolve().parent;spec=json.loads((lane/'spec.json').read_text())
    if os.environ.get('RP_JHH_QUEUE_RUNTIME')!='1':
        env={**os.environ,'RP_JHH_QUEUE_RUNTIME':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
            'MKL_NUM_THREADS':'1','PYTHONUNBUFFERED':'1','PYTHONPATH':str(lane)}
        os.execve(spec['python'],[spec['python'],'-u',str(Path(__file__).resolve()),*sys.argv[1:]],env)
    root=Path(spec['output'])/os.environ['EFE_WORKFLOW_ID'];root.mkdir(mode=0o700,parents=True,exist_ok=True)
    if len(sys.argv)>1:return exercise(sys.argv[1],root)
    role=os.environ['EFE_ROLE'];auth=root/'ray-auth';finished=root/'finished.json'
    if role=='head':auth.write_text(os.urandom(32).hex());auth.chmod(0o600)
    deadline=time.monotonic()+900
    while not auth.exists():
        if time.monotonic()>deadline:raise TimeoutError('Queue-check head did not supply its Ray token')
        time.sleep(1)
    address=os.environ['EFE_HEAD_HOST']+':6379'
    env={**os.environ,'RAY_AUTH_MODE':'token','RAY_AUTH_TOKEN_PATH':str(auth),'RAY_USAGE_STATS_ENABLED':'0'}
    args=[str(Path(spec['python']).with_name('ray')),'start','--block','--disable-usage-stats',
        '--num-cpus=5','--num-gpus=0','--resources='+json.dumps({'queue_check_storage':1}),
        '--object-store-memory='+str(256<<20),'--temp-dir=/tmp/rp_jhh_queue_ray']
    args+=['--head','--port=6379','--include-dashboard=false']if role=='head'else['--address='+address]
    if role!='head':
        while True:
            try:
                with socket.create_connection((os.environ['EFE_HEAD_HOST'],6379),timeout=2):break
            except OSError:
                if time.monotonic()>deadline:raise TimeoutError('Queue-check head did not listen')
                time.sleep(1)
    ray_process=subprocess.Popen(args,env=env,start_new_session=True)
    try:
        if role=='head':
            while True:
                try:
                    with socket.create_connection((os.environ['EFE_HEAD_HOST'],6379),timeout=2):break
                except OSError:
                    if time.monotonic()>deadline:raise TimeoutError('Queue Ray head did not start')
                    time.sleep(1)
            for phase in ['save','load']:
                subprocess.run([spec['python'],'-u',str(Path(__file__).resolve()),phase],
                    env={**env,'RAY_ADDRESS':address},check=True,timeout=500)
                time.sleep(1)
            saved=json.loads((root/'queue-save.json').read_text())
            result={'passed':True,'ray_nodes':len(saved['nodes']),'storage_units':saved['storage_units'],
                'fresh_process_restore':True,'tensors_equal':True,
                'head_local_path_invisible_on_other_node':saved['head_local_path_invisible_on_other_node'],
                'gpu':0,'primary_rl_updates':0}
            (root/'checks.json').write_text(json.dumps(result)+'\n');finished.write_text('{}\n')
            print('PEAR_QUEUE_CHECK '+json.dumps(result),flush=True)
        else:
            while not finished.exists():
                if time.monotonic()>deadline:raise TimeoutError('Queue-check head did not finish')
                if ray_process.poll()is not None:raise RuntimeError('Queue-check Ray worker exited')
                time.sleep(2)
    finally:
        if ray_process.poll()is None:
            os.killpg(ray_process.pid,signal.SIGTERM)
            try:ray_process.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(ray_process.pid,signal.SIGKILL);ray_process.wait()


if __name__=='__main__':main()
