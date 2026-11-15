"""Rebuild an evicted model from retained tensors/candidates without resampling."""
from pathlib import Path
from datetime import datetime, timezone
import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time

from efe_training_data import sha, write, encode_example, forward_target, backward_messages, balanced_order

SPEC = Path('/tmp/efe-rebuild-spec.json')


def main():
    spec = json.loads(SPEC.read_text())
    workflow = os.environ['EFE_WORKFLOW_ID']
    root = Path(spec['archive_root']) / workflow
    root.mkdir(parents=True, exist_ok=True)
    def progress(phase, detail, **extra):
        record = {'phase':phase,'detail':detail,'updated_utc':datetime.now(timezone.utc).isoformat(), **extra}
        write(root/'progress.json',record);print(json.dumps(record),flush=True)
    try:
        version=re.search(r'([0-9]{3}[.][0-9]+(?:[.][0-9]+)?)',Path('/proc/driver/nvidia/version').read_text()).group(1)
        libraries=[Path('/usr/lib/x86_64-linux-gnu')/(name+'.so.'+version) for name in ['libcuda','libnvidia-ptxjitcompiler','libnvidia-nvvm']]
        if not all(p.is_file() for p in libraries):raise RuntimeError('Matching CUDA driver/compiler libraries missing')
        env={**os.environ,'LD_PRELOAD':':'.join(map(str,libraries)),
             'LD_LIBRARY_PATH':'/usr/lib/x86_64-linux-gnu:'+os.environ.get('LD_LIBRARY_PATH',''),
             'NCCL_P2P_DISABLE':'1','OMP_NUM_THREADS':'2','TOKENIZERS_PARALLELISM':'false'}
        bundle=root/'retained.tar.gz'
        bundle.write_bytes(base64.b64decode(Path('/tmp/efe-retained-leact.b64').read_text()))
        if sha(bundle)!=spec['retained_bundle_sha256']:raise ValueError('Retained input bundle changed')
        with tarfile.open(bundle) as archive:archive.extractall(root/'retained',filter='data')
        retained=root/'retained'
        original=json.loads((retained/'dataset-manifest.json').read_text())
        membership=json.loads((retained/'membership.json').read_text())
        original_weights=json.loads((retained/'checkpoint-manifest.json').read_text())
        base=Path(spec['base_model']);base_manifest=json.loads(Path(spec['base_manifest']).read_text())
        progress('verify_inputs','Verifying the warmstart checkpoint and all retained LeAct training inputs.')
        for name,record in base_manifest['files'].items():
            if sha(base/name)!=record['sha256']:raise ValueError('Warmstart checkpoint changed: '+name)
        parent=Path(spec['parent_corpus']);parent_manifest=json.loads((parent/'dataset-manifest.json').read_text())
        if sha(parent/'warmstart-states.jsonl')!=parent_manifest['states_sha256']:raise ValueError('Parent corpus changed')
        states=[json.loads(line) for line in (parent/'warmstart-states.jsonl').read_text().splitlines()]
        selected=balanced_order(states,spec['seed'])[:spec['leact_decisions']]
        if [s['id'] for s in selected]!=membership['states']:raise ValueError('LeAct membership changed')
        lookup={s['id']:s for s in selected}
        (root/'states.jsonl').write_text(''.join(json.dumps(s)+'\n' for s in selected))
        if sha(root/'states.jsonl')!=original['states_sha256']:raise ValueError('LeAct state ordering changed')
        import torch
        from transformers import AutoProcessor
        processor=AutoProcessor.from_pretrained(spec['tokenizer_base'],local_files_only=True)
        processor.image_processor.max_pixels=1048576
        if isinstance(getattr(processor.image_processor,'size',None),dict):processor.image_processor.size['longest_edge']=1048576
        (root/'tensors').mkdir(exist_ok=True);(root/'probes').mkdir(exist_ok=True)
        progress('reconstruct_tensors','Re-encoding saved admitted explanations; every tensor must match its original checksum. No generation or reselection.')
        for index,row in enumerate(original['examples']):
            target=root/row['tensor_file'];target.parent.mkdir(parents=True,exist_ok=True)
            if row['source']=='warmstart':
                source=parent/row['tensor_file']
                if sha(source)!=row['tensor_sha256']:raise ValueError('Warmstart replay tensor changed')
                try:os.link(source,target)
                except OSError:shutil.copyfile(source,target)
            else:
                state=lookup[row['decision_id']]
                record=json.loads((retained/'candidates'/(state['id']+'.json')).read_text())
                candidate=int(row['state_id'].rsplit('-leact-',1)[1])
                if candidate not in record['selected']:raise ValueError('Attempted to restore an unadmitted candidate')
                rationale=next(c['rationale'] for c in record['candidates'] if c['candidate']==candidate)
                if row['direction']=='forward':messages,target_message=state['messages'],forward_target(state['action'],rationale)
                else:messages,target_message=backward_messages(state['messages'],state['action']),{'role':'assistant','content':'<think>\n'+rationale+'\n</think>\n\nDONE'}
                encoded,receipt=encode_example(processor,messages,state['tools'],target_message,spec['context'])
                if encoded is None:raise ValueError('Previously admitted complete pair no longer fits')
                torch.save(encoded,target)
            if sha(target)!=row['tensor_sha256']:raise ValueError('Reconstructed tensor checksum differs: '+row['id'])
            if index%50==0:progress('reconstruct_tensors','Original training tensor checksums verified.',completed=index+1,total=len(original['examples']))
        shutil.copyfile(retained/'dataset-manifest.json',root/'dataset-manifest.json')
        probes=[]
        for shard,state in enumerate(selected[:2]):
            record=json.loads((retained/'candidates'/(state['id']+'.json')).read_text())
            encoded,token_receipt=encode_example(processor,state['messages'],state['tools'],forward_target(state['action'],''),spec['context'],action_only=True)
            ids=encoded['input_ids'][0].tolist()[token_receipt['target_start_token']:]
            target_sha=hashlib.sha256(json.dumps(ids).encode()).hexdigest()
            token_check={'state':state['id'],'target_tokens':len(ids),'target_ids_sha256':target_sha,'complete_prompt_tokens':encoded['input_ids'].shape[1],
                         'expected':record['base']}
            write(root/'probes'/('base-'+str(shard)+'-tokens.json'),token_check)
            if any(token_check[k]!=record['base'][k] for k in ['target_tokens','target_ids_sha256','complete_prompt_tokens']):
                raise ValueError('Rebuilt likelihood probe token identities differ from original scoring')
            path=root/'probes'/('base-'+str(shard)+'.pt');torch.save(encoded,path)
            probes.append({'condition':'base','tensor_file':str(path.relative_to(root)),'tensor_sha256':sha(path),'mean_target_logp':record['base']['mean_target_logp']})
        write(root/'scoring-probes.json',probes)
        write(root/'reconstruction.json',{'all_original_tensor_hashes_match':True,'examples':len(original['examples']),
            'dataset_sha256':sha(root/'dataset-manifest.json'),'candidate_generation_performed':False,
            'original_schedule_sha256':spec['expected_schedule_sha256'],'original_optimizer_steps':12})
        import efe_training_pilot as pilot
        pilot.ROOT=root
        train=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=8','/tmp/efe_train_vl.py','--spec',str(SPEC),'--root',str(root)]
        progress('optimizer_preflight','Checking original scoring parity and memory before rebuilding the 12 updates.')
        pilot.run_logged('smoke',train+['--mode','smoke'],env,3600)
        progress('rebuilding_12_updates','Replaying the original fixed 12-update schedule from warmstart 96.',training_started=True)
        pilot.run_logged('train',train+['--mode','train'],env,8*3600)
        ready=json.loads((root/'leact-ready.json').read_text());manifest=json.loads(Path(ready['manifest']).read_text())
        matches={name:record==original_weights['files'].get(name) for name,record in manifest['files'].items()}
        write(root/'weight-comparison.json',{'all_original_files_identical':all(matches.values()),'files':matches,
            'original_revision':original_weights['revision'],'rebuilt_revision':ready['revision'],
            'rule':'Rebuilt evaluation stays separately labeled even if all serialized weight files match.'})
        from efe_leact_pipeline import export_portable
        portable=export_portable(root,ready)
        progress('rebuild_complete','Checkpoint rebuilt; starting 21 separately labeled training-task evaluations.',checkpoint=portable['checkpoint'])
        native_path=Path('/tmp/efe-training-evaluation-v1/warm96/spec.json')
        shutil.copyfile('/tmp/efe_budget_gateway.py',native_path.parent/'efe_budget_gateway.py')
        native=json.loads(native_path.read_text());native.update(student_checkpoint=portable['checkpoint'],student_manifest=portable['manifest'],student_revision=ready['revision'])
        write(native_path,native)
        with (root/'native.log').open('w') as log:
            subprocess.run([sys.executable,'-u','/tmp/efe_eval_bootstrap.py'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        write(root/'complete.json',{'success':True,'rebuild_and_evaluation_complete':True})
    except Exception as error:
        progress('needs_review',str(error));write(root/'complete.json',{'success':False,'error':str(error)})
        raise
    finally:
        for _ in range(720):
            if (root/'collected').exists():break
            time.sleep(30)


if __name__=='__main__':main()
