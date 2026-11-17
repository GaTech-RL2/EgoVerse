"""Verify and export a pinned intermediate adapter; never update its weights."""
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import traceback


def main(lane):
    from efe_training_data import sha,write
    from efe_model_cache import stage
    request=json.loads((lane/'spec.json').read_text());root=Path(request['archive_root'])
    lock=(root/'export.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    workflow=os.environ['EFE_WORKFLOW_ID'];attempt=root/'attempts'/workflow
    attempt.mkdir(parents=True,exist_ok=True)
    def progress(phase,**extra):
        value={'phase':phase,'workflow':workflow,'step':request['step'],'additional_optimizer_updates':0,
            'updated_utc':datetime.now(timezone.utc).isoformat(),**extra}
        write(root/'progress.json',value);print('EFE_INTERMEDIATE_PROGRESS '+json.dumps(value),flush=True)
    def logged(name,command,timeout):
        with (attempt/(name+'.log')).open('w') as output:
            result=subprocess.run(command,stdout=output,stderr=subprocess.STDOUT,cwd=lane,timeout=timeout)
        if result.returncode:
            tail=(attempt/(name+'.log')).read_text(errors='replace')[-12000:]
            raise RuntimeError(name+' failed: '+tail)
    try:
        pin_path=root/'checkpoint-pin.json'
        if sha(pin_path)!=request['pin_sha256']:raise ValueError('Pinned checkpoint receipt changed')
        pin=json.loads(pin_path.read_text())
        if pin['step']!=request['step'] or pin['final'] is not False:raise ValueError('Wrong intermediate checkpoint')
        for name,row in pin['files'].items():
            if sha(Path(pin['checkpoint'])/name)!=row['sha256']:raise ValueError('Pinned adapter changed: '+name)
        if sha(Path(pin['training_root'])/'dataset-manifest.json')!=pin['dataset_sha256']:
            raise ValueError('Pinned training inputs changed')
        if sha(Path(pin['training_root'])/'spec.json')!=pin['source_spec_sha256']:
            raise ValueError('Pinned training schedule configuration changed')
        if (root/'complete.json').exists():
            done=json.loads((root/'complete.json').read_text())
            if done.get('success'):
                manifest=json.loads(Path(done['ready']['manifest']).read_text())
                for name,row in manifest['files'].items():
                    if sha(Path(done['ready']['checkpoint'])/name)!=row['sha256']:raise ValueError('Export changed')
                print('EFE_INTERMEDIATE_RESULT '+json.dumps(done),flush=True);return
        progress('staging_verified_base')
        cache=stage(pin['base_model'],request['base_manifest'],str(lane/'model-cache'),pin['base_revision'],
            progress=lambda value:write(root/'cache-progress.json',value))
        environment=lane/'environment'
        subprocess.run([sys.executable,'-m','venv','--system-site-packages',str(environment)],check=True)
        python=str(environment/'bin/python')
        logged('environment',[python,'-m','pip','install','--disable-pip-version-check',
            'transformers==5.11.0','peft==0.21.0','accelerate==1.11.0'],900)
        runtime=json.loads((Path(pin['training_root'])/'spec.json').read_text())
        spec={**runtime,**pin,'archive_root':str(root),'local_base_model':cache['local_model'],
            'checkpoint_revision_prefix':request.get('checkpoint_revision_prefix','efe-coverage-warmstart-intermediate'),
            'export_final':False,'export_selection':pin['selection'],'ready_name':'adapter-ready.json'}
        spec_path=attempt/'verify-spec.json';write(spec_path,spec)
        progress('verifying_checkpoint',expected_post_update_ce=pin['expected_post_update_ce'])
        logged('verify',[python,'-m','torch.distributed.run','--standalone','--nproc_per_node=8',
            str(lane/'verify.py'),'--spec',str(spec_path),'--mode','verify'],3600)
        replay=json.loads((root/'loss-replay.json').read_text())
        if not replay['passed'] or replay['optimizer_updates']!=0:raise ValueError('Intermediate replay failed')
        progress('merging_native_model',loss_replay=replay)
        work=lane/'merge-output'
        if work.exists():shutil.rmtree(work)
        work.mkdir()
        shutil.copyfile(Path(pin['training_root'])/'dataset-manifest.json',work/'dataset-manifest.json')
        write(work/'train-loss-replay.json',replay)
        write(work/'adapter-ready.json',{'checkpoint':pin['checkpoint'],'manifest':pin['manifest'],
            'step':pin['step'],'final':False})
        merge_spec={**spec,'archive_root':str(work)};write(work/'spec.json',merge_spec)
        logged('merge',[python,str(lane/'merge.py'),'--spec',str(work/'spec.json')],7200)
        source=work/'model';manifest=json.loads((source/'checkpoint-manifest.json').read_text())
        destination=root/'model';destination.mkdir(exist_ok=True)
        members={**manifest['files'],'checkpoint-manifest.json':{
            'sha256':sha(source/'checkpoint-manifest.json'),'bytes':(source/'checkpoint-manifest.json').stat().st_size}}
        for name,row in members.items():
            if Path(name).name!=name:raise ValueError('Nonlocal model member')
            target=destination/name
            if target.exists() and target.stat().st_size==row['bytes'] and sha(target)==row['sha256']:continue
            progress('persisting_native_model',file=name,bytes=row['bytes'])
            temporary=target.with_suffix(target.suffix+'.partial')
            shutil.copyfile(source/name,temporary)
            if temporary.stat().st_size!=row['bytes'] or sha(temporary)!=row['sha256']:
                raise ValueError('Persisted model differs: '+name)
            with temporary.open('rb') as stream:os.fsync(stream.fileno())
            temporary.replace(target)
        ready={'checkpoint':str(destination),'manifest':str(destination/'checkpoint-manifest.json'),
            'revision':manifest['revision'],'step':pin['step'],'final':False,'selection':pin['selection'],
            'pin_sha256':request['pin_sha256'],'loss_replay':replay,'additional_optimizer_updates':0}
        result={'success':True,'step':pin['step'],'ready':ready,'additional_optimizer_updates':0,
            'completed_utc':datetime.now(timezone.utc).isoformat(),'workflow':workflow}
        write(root/'intermediate-ready.json',ready);write(root/'complete.json',result)
        progress('ready_for_native_evaluation',loss_replay=replay)
        print('EFE_INTERMEDIATE_RESULT '+json.dumps(result),flush=True)
    except Exception as error:
        value={'success':False,'workflow':workflow,'step':request['step'],'error':str(error),
            'traceback':traceback.format_exc(),'additional_optimizer_updates':0,
            'updated_utc':datetime.now(timezone.utc).isoformat()}
        write(attempt/'error.json',value);progress('needs_review',error=str(error)[-3000:])
        print('EFE_INTERMEDIATE_RESULT '+json.dumps(value),flush=True);raise


if __name__=='__main__':
    if not os.environ.get('EFE_INTERMEDIATE_CUDA_READY'):
        version=re.search(r'([0-9]{3}[.][0-9]+(?:[.][0-9]+)?)',Path('/proc/driver/nvidia/version').read_text()).group(1)
        libraries=[Path('/usr/lib/x86_64-linux-gnu')/(name+'.so.'+version) for name in ['libcuda','libnvidia-ptxjitcompiler','libnvidia-nvvm']]
        if not all(p.is_file() for p in libraries):raise RuntimeError('Matching host CUDA libraries required')
        env={**os.environ,'EFE_INTERMEDIATE_CUDA_READY':'1','LD_PRELOAD':':'.join(map(str,libraries)),
            'LD_LIBRARY_PATH':'/usr/lib/x86_64-linux-gnu:'+os.environ.get('LD_LIBRARY_PATH',''),
            'NCCL_P2P_DISABLE':'1','NCCL_ALGO':'Ring','NCCL_PROTO':'Simple','OMP_NUM_THREADS':'2',
            'TOKENIZERS_PARALLELISM':'false','TORCH_NCCL_ASYNC_ERROR_HANDLING':'1'}
        os.execve(sys.executable,[sys.executable,*sys.argv],env)
    main(Path(__file__).parent)
