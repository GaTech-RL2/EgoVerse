"""Qualify all reasoning scenes and the actual PEAR stateful transport on Linux."""
import asyncio
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback


def transport(root):
    import ray
    from pear.sim.client import StatefulClient
    from pear.sim.server import ServerPool
    from efe_pear_reasoning_env import ReasoningEnv
    from efe_pear_reasoning_scenes import SceneEpisode,REASONING_FAMILIES
    manifest=json.loads((root/'membership.json').read_text())
    os.environ['EFE_PEAR_REASONING_MEMBERSHIP']=str(root/'membership.json')
    ray.init(num_cpus=8,num_gpus=0,include_dashboard=False,resources={'reasoning':2},
        _temp_dir='/tmp/rp-jhh-reasoning-ray',runtime_env={'env_vars':{
            'EFE_PEAR_REASONING_MEMBERSHIP':str(root/'membership.json'),'MUJOCO_GL':'osmesa',
            'PYTHONPATH':os.environ['PYTHONPATH'],'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}})
    results={}
    try:
        pool=ServerPool.start(ReasoningEnv,{'server_n':2,'log_dir':str(root/'stateful-transport')})
        client=StatefulClient(ReasoningEnv)
        async def run():
            for family in REASONING_FAMILIES:
                row=next(v for v in manifest['scenes'].values()if v['family']==family and v['partition']=='train')
                args={k:row[k]for k in ['family','seed','partition']};uid='qualification-'+family
                oracle=SceneEpisode(**args)
                try:
                    oracle.observation();observed=await client.open(uid,args)
                    if observed['done']or not observed['images']:raise ValueError('Transport lost initial observation')
                    answer=oracle.oracle_answer()
                    if answer.get('action_type')=='view':
                        oracle.step(answer);observed=await client.step(uid,answer);answer=oracle.oracle_answer()
                        if observed['done']or not observed['images']:raise ValueError('Transport lost inspected view')
                    final=await client.step(uid,answer)
                    if not final['done']or final['rewards']!=[1.]:raise ValueError('PEAR stateful transport changed the verified outcome')
                    results[family]={'passed':True,'final_reward':1.,'scene_sha256':row['scene_sha256']}
                finally:
                    oracle.close();await client.close(uid)
        asyncio.run(run());return results
    finally:ray.shutdown()


def main():
    lane=Path(__file__).resolve().parent;spec=json.loads((lane/'spec.json').read_text())
    root=Path(spec['output'])/os.environ['EFE_WORKFLOW_ID'];root.mkdir(parents=True,exist_ok=True)
    def record(phase,**extra):
        v={'phase':phase,'primary_rl_updates':0,**extra}
        (root/'status.json').write_text(json.dumps(v)+'\n');print('PEAR_REASONING_CHECK '+json.dumps(v),flush=True)
    if '--generate'in sys.argv:
        from efe_pear_reasoning_qualify import qualify
        qualify(root,train_scenes=64,development_scenes=8,cluster=True)
        return
    if '--transport'in sys.argv:
        checks=transport(root)
        receipt={'phase':'reasoning_cluster_validation_passed','passed':True,
            'families':checks,'policy_model_inference_calls':0,'primary_rl_updates':0,
            'membership_file':str(root/'membership.json'),'qualification_scope':'Generated scene/reward controls and PEAR stateful transport; model sampler and optimizer remain separate.'}
        (root/'result.json').write_text(json.dumps(receipt,indent=2)+'\n')
        record(receipt.pop('phase'),**receipt);return
    try:
        record('reasoning_dependencies')
        subprocess.run(['apt-get','update','-qq'],check=True,timeout=300)
        subprocess.run(['apt-get','install','-y','-qq','libosmesa6','libgl1','libglib2.0-0'],check=True,timeout=300)
        runtime=Path(spec['remote_root'])/'runtime';python=runtime/'envs/verl/bin/python'
        with (runtime/'reasoning-package-install.lock').open('a')as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            uv=shutil.which('uv')
            if uv is None:
                subprocess.run([sys.executable,'-m','pip','install','--disable-pip-version-check','uv==0.12.24'],check=True,timeout=180)
                uv=shutil.which('uv')
            if uv is None:raise RuntimeError('Pinned package installer is missing')
            subprocess.run([uv,'pip','install','--python',str(python),'mujoco==3.3.7'],check=True,timeout=600)
        source=Path(spec['pear_source'])
        env={**os.environ,'MUJOCO_GL':'osmesa','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1',
            'PYTHONPATH':str(lane)+':'+str(source)+':'+str(source/'3rdparty/verl/src'),
            'RAY_DEDUP_LOGS':'0','TOKENIZERS_PARALLELISM':'false'}
        auth=Path('/tmp/rp-jhh-reasoning-ray-auth');auth.mkdir(mode=0o700,exist_ok=True)
        token=auth/'token';token.write_text(os.urandom(32).hex());token.chmod(0o600)
        # Ray reads its auth mode at import time; set it before either child
        # imports Ray, not just immediately before ray.init().
        env.update(RAY_AUTH_MODE='token',RAY_AUTH_TOKEN_PATH=str(token))
        # Mesa and Torch load native runtimes. Keep bulk rendering and the Ray/
        # Torch transport in fresh processes, matching the production workers.
        env['PYTHONFAULTHANDLER']='1'
        for phase in ['generate','transport']:
            record('reasoning_'+phase+'_checks')
            subprocess.run([str(python),'-u',str(Path(__file__).resolve()),'--'+phase],env=env,check=True,timeout=3600)
    except Exception as exc:
        record('reasoning_cluster_validation_failed',error=repr(exc),traceback=traceback.format_exc());raise


if __name__=='__main__':main()
