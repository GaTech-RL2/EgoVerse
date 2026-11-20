"""Exercise reasoning children and Ray pools before loading model replicas."""
import asyncio
import json
import os
from pathlib import Path
import sys

from efe_pear_reasoning_env import ReasoningEnv
from efe_pear_sim_process import SpawnedSimulatorProcess


def check_ray(root):
    import ray
    from pear.sim.client import StatefulClient
    from pear.sim.server import ServerPool
    from efe_pear_reasoning_env import ReasoningDevelopmentEnv
    from efe_pear_reasoning_scenes import SceneEpisode,REASONING_FAMILIES
    bank=json.loads(Path(os.environ['EFE_PEAR_REASONING_MEMBERSHIP']).read_text())
    ray.init(num_cpus=16,num_gpus=0,include_dashboard=False,resources={'reasoning':8},
        _temp_dir='/tmp/rp-jhh-pear-server-check-ray')
    pools=[];results={}
    try:
        for cls,count,partition in [(ReasoningEnv,6,'train'),(ReasoningDevelopmentEnv,2,'development')]:
            pools.append(ServerPool.start(cls,{'server_n':count,'log_dir':str(root/'sim')}))
            client=StatefulClient(cls)
            async def run():
                for family in REASONING_FAMILIES:
                    row=next(v for v in bank['scenes'].values()if v['family']==family and v['partition']==partition)
                    args={k:row[k]for k in ['family','seed','partition']};uid=partition+'-'+family
                    oracle=SceneEpisode(**args)
                    try:
                        oracle.observation();observed=await client.open(uid,args)
                        if observed['done']or not observed['images']:raise ValueError('Ray pool lost reset RGB')
                        answer=oracle.oracle_answer()
                        if answer.get('action_type')=='view':
                            oracle.step(answer);await client.step(uid,answer);answer=oracle.oracle_answer()
                        final=await client.step(uid,answer)
                        if not final['done']or final['rewards']!=[1.]:raise ValueError('Ray pool changed native reward')
                        results[uid]={'passed':True,'final_reward':1.,'scene_sha256':row['scene_sha256']}
                    finally:
                        oracle.close();await client.close(uid)
            asyncio.run(run())
        value={'phase':'reasoning_ray_server_preflight_passed','passed':True,'processes':8,
            'sessions':results,'policy_model_inference_calls':0,'primary_rl_updates':0}
        (root/'ray-result.json').write_text(json.dumps(value,indent=2)+'\n')
        print('PEAR_SERVER_CHECK '+json.dumps({k:v for k,v in value.items()if k!='sessions'}),flush=True)
    finally:ray.shutdown()


def main():
    root=Path(os.environ['EFE_PEAR_QUALIFICATION_DIR'])/'process-preflight';root.mkdir(exist_ok=True)
    if '--ray' in sys.argv:return check_ray(root)
    processes=[]
    try:
        for index in range(8):
            process=SpawnedSimulatorProcess('reasoning-'+str(index),ReasoningEnv,sys.executable,[],
                {**os.environ,'MUJOCO_GL':'osmesa','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'},
                Path(__file__).resolve().parent,root)
            process.start();processes.append(process)
        if not all(p.wait_ready(120)for p in processes):raise RuntimeError('A reasoning simulator exited before ready')
        value={'phase':'reasoning_process_preflight_passed','processes':len(processes),'primary_rl_updates':0}
        (root/'result.json').write_text(json.dumps(value)+'\n');print('PEAR_ONPOLICY_CHECK '+json.dumps(value),flush=True)
    finally:
        for process in processes:process.close()
        for process in processes:process.wait_or_kill(10)


if __name__=='__main__':main()
