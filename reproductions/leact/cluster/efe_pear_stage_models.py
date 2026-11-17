"""Stage the pinned dense base weights; verify the existing MoE parent is present."""
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import traceback

ROOT=Path('/mnt/amlfs-07/shared/rpunamiya/leact/studies/efe-pear-scaled-20261116-v1')
BASE_REVISION='37561816e66b5b9153b1b64c10968c18c1e6f4a0'
MOE=Path('/mnt/amlfs-07/shared/rpunamiya/leact/studies/efe-coverage-20261105-v1/intermediate/warmstart-step496/model')


def record(phase,**kwargs):
    result={'phase':phase,'updated_utc':datetime.now(timezone.utc).isoformat(),'primary_rl_updates':0,**kwargs}
    p=ROOT/'model-stage-status.json';tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(p)
    print('PEAR_MODEL_STAGE '+json.dumps(result),flush=True)
    return result


def main():
    from huggingface_hub import snapshot_download
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'model-stage.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        record('downloading_dense_base',repository='Qwen/Qwen3.8-27B',revision=BASE_REVISION)
        target=ROOT/'models/Qwen3.8-27B'
        snapshot_download('Qwen/Qwen3.8-27B',revision=BASE_REVISION,local_dir=target,max_workers=8,
                          allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.tiktoken'])
        models=[]
        for kind,path,expected in [('dense_base',target,'qwen3_5'),('moe_warmstart496',MOE,'qwen3_5_moe')]:
            config=json.loads((path/'config.json').read_text())
            if config['model_type']!=expected:raise ValueError('Unexpected model type: '+kind)
            index=json.loads((path/'model.safetensors.index.json').read_text())
            filenames=set(index['weight_map'].values())
            if not filenames or any(not (path/f).is_file() for f in filenames):
                raise ValueError('Missing weight shard: '+kind)
            models.append({'kind':kind,'path':str(path),'model_type':expected,'weight_shards':len(filenames),
                'weight_bytes':sum((path/f).stat().st_size for f in filenames),
                'config_sha256':hashlib.sha256((path/'config.json').read_bytes()).hexdigest(),
                'revision':BASE_REVISION if kind=='dense_base' else 'efe-coverage-warmstart-intermediate-step496'})
        receipt=record('model_files_staged',success=True,models=models,
            full_model_numerical_qualification=False,scope='Pinned model files present; no inference, optimizer or replay pass claimed.')
        (ROOT/'model-files-ready.json').write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__':
    try:main()
    except Exception as exc:
        record('model_staging_failed',success=False,error=repr(exc),traceback=traceback.format_exc())
        raise
