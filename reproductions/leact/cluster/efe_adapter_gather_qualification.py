"""Isolated checkpoint performance qualification; never update the study model."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from efe_training_data import sha, write


def run(spec, lane):
    root=Path(spec['qualification_root'])/os.environ['EFE_WORKFLOW_ID']
    root.mkdir(parents=True,exist_ok=True)
    def progress(phase, **extra):
        value={'phase':phase,'workflow':os.environ['EFE_WORKFLOW_ID'],
            'updated_utc':datetime.now(timezone.utc).isoformat(),'production_enabled':False,**extra}
        write(root/'progress.json',value);write(root.parent/'progress.json',value)
        print(json.dumps(value),flush=True)
    def logged(name, command, timeout):
        progress(name)
        with (root/(name+'.log')).open('w') as log:
            subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=timeout)
    env={**os.environ,**spec['nccl_collective_workaround'],'NCCL_P2P_DISABLE':'1',
        'OMP_NUM_THREADS':'2','TOKENIZERS_PARALLELISM':'false'}
    try:
        source=Path(spec['coverage_root'])/'warmstart'
        checkpoint=source/'restart-states'/('checkpoint-'+str(spec['qualification_step']))
        receipt=json.loads((checkpoint/'training-state.json').read_text())
        if receipt!=spec['qualification_receipt'] or not receipt.get('portable'):
            raise ValueError('Qualification requires the unchanged retained portable checkpoint')
        from efe_qwen36_restart import verify_files
        verify_files(checkpoint,receipt['files'])
        if sha(source/'dataset-manifest.json')!=spec['qualification_dataset_sha256']:
            raise ValueError('Frozen qualification dataset changed')
        write(root/'parent-receipt.json',receipt)
        environment=lane/'qualification-env'
        subprocess.run([sys.executable,'-m','venv','--system-site-packages',str(environment)],check=True)
        python=str(environment/'bin/python')
        logged('environment',[python,'-m','pip','install','--disable-pip-version-check',
            'transformers==5.11.0','peft==0.21.0','accelerate==1.11.0'],900)
        results={}
        for name,flags in [('normal',[]),('empty-shards',['--empty-shards']),('nested',['--nested'])]:
            probe=root/name
            logged(name,[python,'-m','torch.distributed.run','--standalone','--nproc_per_node=8',
                str(lane/'probe.py'),'--root',str(probe),'--adapter-gather',*flags],900)
            results[name]=json.loads((probe/'result.json').read_text())
            if not results[name]['success'] or not results[name]['gather_qualification']['passed']:
                raise ValueError('Small-model exact gather/resume qualification failed')
        progress('model_cache')
        from efe_model_cache import stage
        cache=stage(spec['base_model'],Path(spec['coverage_root'])/'base-checkpoint-manifest.json',
            lane/'qualification-model-cache',spec['base_revision'],lambda value:write(root/'model-cache-progress.json',value))
        verify={**spec,'archive_root':str(root/'real-model'),'training_root':str(source),
            'checkpoint':str(checkpoint),'local_base_model':cache['local_model'],'verification_mode':'train',
            'expected_post_update_ce':receipt['post_update_full_target_ce'],'adapter_gather_qualification':True}
        Path(verify['archive_root']).mkdir()
        write(root/'verify-spec.json',verify)
        logged('real-model',[python,'-m','torch.distributed.run','--standalone','--nproc_per_node=8',
            str(lane/'verify.py'),'--spec',str(root/'verify-spec.json'),'--mode','verify'],3600)
        results['real_model']=json.loads((Path(verify['archive_root'])/'adapter-gather-qualification.json').read_text())
        # A fresh process must reproduce the reference loss from the newly
        # serialized adapter, without carrying over any old model tensors.
        reload={**verify,'archive_root':str(root/'reload'),'adapter_gather_qualification':False,
            'checkpoint':str(Path(verify['archive_root'])/'gathered-adapter')}
        Path(reload['archive_root']).mkdir();write(root/'reload-spec.json',reload)
        logged('reload',[python,'-m','torch.distributed.run','--standalone','--nproc_per_node=8',
            str(lane/'verify.py'),'--spec',str(root/'reload-spec.json'),'--mode','verify'],3600)
        results['reload']=json.loads((Path(reload['archive_root'])/'train-loss-replay.json').read_text())
        result={'success':results['real_model']['passed'] and results['reload']['passed'],
            'production_enabled':False,'workflow':os.environ['EFE_WORKFLOW_ID'],'parent_step':receipt['step'],
            'study_optimizer_updates':0,'results':results,'completed_utc':datetime.now(timezone.utc).isoformat()}
        write(root/'complete.json',result);write(root.parent/'complete.json',result)
        progress('qualified',success=result['success'])
    except Exception as error:
        result={'success':False,'workflow':os.environ['EFE_WORKFLOW_ID'],'error':str(error),
            'production_enabled':False,'completed_utc':datetime.now(timezone.utc).isoformat()}
        write(root/'complete.json',result);write(root.parent/'complete.json',result)
        progress('qualification_failed',error=str(error));raise


if __name__=='__main__':
    lane=Path(__file__).parent
    run(json.loads((lane/'spec.json').read_text()),lane)
