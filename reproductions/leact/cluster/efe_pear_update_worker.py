"""Isolated base-policy PEAR optimizer qualification; never writes warmstart."""
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import traceback


def main(lane):
    from efe_training_data import write,sha
    from efe_model_cache import stage
    from efe_qwen36_restart import choose_resume
    spec=json.loads((lane/'spec.json').read_text());root=Path(spec['qualification_root'])
    root.mkdir(parents=True,exist_ok=True)
    lock=(root/'writer.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    # save_exact creates the pending checkpoint itself; its state parent must
    # exist before we spend a complete native-group update on the allocation.
    states=Path(spec['durable_restart']['state_directory'])
    if not states.resolve().is_relative_to(root.resolve()):
        raise ValueError('Restart states must stay inside this qualification')
    states.mkdir(parents=True,exist_ok=True)
    workflow=os.environ['EFE_WORKFLOW_ID']
    attempt=root/'attempts'/workflow;attempt.mkdir(parents=True,exist_ok=True)
    def progress(phase,**extra):
        value={'phase':phase,'workflow':workflow,'primary_rl_updates':0,
            'utc':datetime.now(timezone.utc).isoformat(),**extra}
        write(root/'progress.json',value);print('PEAR_OPTIMIZER_PROGRESS '+json.dumps(value),flush=True)
    def logged(name,cmd,timeout):
        with (attempt/(name+'.log')).open('w') as out:
            result=subprocess.run(cmd,stdout=out,stderr=subprocess.STDOUT,timeout=timeout,cwd=lane)
        if result.returncode:raise RuntimeError(name+' failed; see '+str(attempt/(name+'.log')))
    try:
        if (root/'result.json').exists():
            result=json.loads((root/'result.json').read_text())
            if result['success']:
                print('PEAR_OPTIMIZER_RESULT '+json.dumps(result),flush=True);return
        write(attempt/'worker-start.json',{'workflow':workflow,'utc':datetime.now(timezone.utc).isoformat()})
        progress('staging_verified_base',optimizer_updates=0)
        cache=stage(spec['base_model'],spec['base_manifest'],'/tmp/efe-pear-update-model',spec['base_revision'],
            progress=lambda value:write(root/'cache-progress.json',value))
        spec['local_base_model']=cache['local_model']
        spec_path=attempt/'run-spec.json';write(spec_path,spec)
        preprocess=lane/'preprocessing-env'
        subprocess.run([sys.executable,'-m','venv','--system-site-packages',str(preprocess)],check=True)
        pp=str(preprocess/'bin/python')
        logged('preprocessing-environment',[pp,'-m','pip','install','--disable-pip-version-check','transformers==4.57.6'],900)
        progress('preparing_exact_native_targets',optimizer_updates=0)
        logged('prepare',[pp,str(lane/'efe_pear_update_data.py'),'--spec',str(spec_path),'--root',str(root)],7200)
        spec['prepared_manifest_sha256']=sha(root/'dataset-manifest.json');write(spec_path,spec)
        training=lane/'training-env'
        subprocess.run([sys.executable,'-m','venv','--system-site-packages',str(training)],check=True)
        python=str(training/'bin/python')
        logged('training-environment',[python,'-m','pip','install','--disable-pip-version-check',
            'transformers==5.11.0','peft==0.21.0','accelerate==1.11.0'],900)
        train=[python,'-m','torch.distributed.run','--standalone','--nproc_per_node=8',str(lane/'rl_trainer.py'),
            '--spec',str(spec_path),'--root',str(root)]
        if choose_resume(spec)['step']==0:
            progress('real_policy_gradient_update',optimizer_updates=0,
                generated_tokens=spec['input']['generated_tokens'],requests=spec['input']['requests'])
            logged('update',train+['--mode','update'],8*3600)
        progress('fresh_process_restart_verification',optimizer_updates=1)
        logged('verify',train+['--mode','verify'],3600)
        result=json.loads((root/'result.json').read_text())
        progress('optimizer_qualification_passed',optimizer_updates=1)
        print('PEAR_OPTIMIZER_RESULT '+json.dumps(result),flush=True)
    except Exception as error:
        result={'success':False,'workflow':workflow,'error':repr(error),'traceback':traceback.format_exc(),
            'optimizer_updates':choose_resume(spec)['step'],'primary_rl_updates':0,
            'utc':datetime.now(timezone.utc).isoformat()}
        write(attempt/'error.json',result);progress('optimizer_qualification_needs_review',**result)
        print('PEAR_OPTIMIZER_RESULT '+json.dumps(result),flush=True);raise


if __name__=='__main__':
    if not os.environ.get('EFE_PEAR_UPDATE_DRIVER_READY'):
        version=re.search(r'([0-9]{3}[.][0-9]+(?:[.][0-9]+)?)',Path('/proc/driver/nvidia/version').read_text()).group(1)
        libs=[Path('/usr/lib/x86_64-linux-gnu')/(name+'.so.'+version) for name in ['libcuda','libnvidia-ptxjitcompiler','libnvidia-nvvm']]
        if not all(p.is_file() for p in libs):raise RuntimeError('Matching host CUDA libraries required')
        env={**os.environ,'EFE_PEAR_UPDATE_DRIVER_READY':'1','LD_PRELOAD':':'.join(map(str,libs)),
            'LD_LIBRARY_PATH':'/usr/lib/x86_64-linux-gnu:'+os.environ.get('LD_LIBRARY_PATH',''),
            'NCCL_P2P_DISABLE':'1','NCCL_ALGO':'Ring','NCCL_PROTO':'Simple','OMP_NUM_THREADS':'2',
            'TOKENIZERS_PARALLELISM':'false','TORCH_NCCL_ASYNC_ERROR_HANDLING':'1'}
        os.execve(sys.executable,[sys.executable,*sys.argv],env)
    main(Path(__file__).parent)
