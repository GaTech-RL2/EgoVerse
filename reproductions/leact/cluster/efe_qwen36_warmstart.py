"""Prepare public backward explanations, train LoRA, and export a native checkpoint."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import copy
import gzip
import importlib.metadata
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import time

from efe_training_data import encode_example, sha, write
import efe_training_pilot as pilot

LANE=Path('/tmp/efe-qwen36-warmstart')


def main():
    spec=json.loads((LANE/'spec.json').read_text());root=Path(spec['archive_root'])
    root.mkdir(parents=True,exist_ok=True);pilot.ROOT=root
    def progress(phase,detail,**extra):pilot.progress(phase,detail,**extra)
    write(root/'spec.json',spec)
    write(root/'worker-start.json',{'workflow':os.environ['EFE_WORKFLOW_ID'],'started_utc':datetime.now(timezone.utc).isoformat()})
    try:
        versions={name:importlib.metadata.version(name) for name in ['torch','transformers','peft','safetensors','pillow']}
        write(root/'versions.json',versions)
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText
        from transformers import AutoConfig
        from huggingface_hub import snapshot_download
        import efe_cuda_helpers as helpers
        helpers.ROOT=root;helpers.ARCHIVE=root
        env=helpers.driver_environment()
        if os.environ.get('LD_PRELOAD'):env['LD_PRELOAD']=os.environ['LD_PRELOAD']
        env.update(CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',NCCL_P2P_DISABLE='1',OMP_NUM_THREADS='2',
            TOKENIZERS_PARALLELISM='false',TORCH_NCCL_ASYNC_ERROR_HANDLING='1')
        if spec.get('prior_checkpoint_audit'):
            import struct
            audit=[]
            for step in (8,16,32):
                p=Path(spec['prior_checkpoint_audit'])/'train'/f'checkpoint-{step}'/'adapter_model.safetensors'
                with p.open('rb') as f:
                    header=json.loads(f.read(struct.unpack('<Q',f.read(8))[0]))
                keys=[k for k in header if k!='__metadata__']
                audit.append({'step':step,'path':str(p),'bytes':p.stat().st_size,'tensors':len(keys),'sha256':sha(p)})
            write(root/'prior-checkpoint-audit.json',{'checkpoints':audit,'original_files_untouched':True})
            if any(r['tensors'] for r in audit):raise ValueError('An earlier checkpoint has weights; recover it before repeating training')
        progress('serialization_probe','Testing the checkpoint fix on a tiny distributed model before loading Qwen.',training_started=False)
        pilot.run_logged('serialization-probe',[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
            str(LANE/'efe_qwen36_adapter_io.py'),'--root',str(root)],env,300)
        if not json.loads((root/'serialization-probe.json').read_text())['success']:raise ValueError('Adapter serialization probe failed')
        progress('preparing_base','Fetching the pinned Qwen3.6 base weights for the warmstart.',training_started=False)
        snapshot_download(spec['model'],revision=spec['base_revision'],local_dir=spec['base_model'],max_workers=4,
            allow_patterns=['*.json','*.safetensors','*.txt','*.jinja','LICENSE*'])
        hf_config=AutoConfig.from_pretrained(spec['base_model'],local_files_only=True)
        architecture=AutoModelForImageTextToText._model_mapping[type(hf_config)]
        write(root/'architecture-preflight.json',{'model_type':hf_config.model_type,'class':architecture.__name__,'transformers':versions['transformers']})
        config=json.loads((Path(spec['base_model'])/'config.json').read_text())
        if config['architectures']!=['Qwen3_5MoeForConditionalGeneration']:raise ValueError('Unexpected student architecture')
        if spec.get('prepared_dataset'):
            prepared=spec['prepared_dataset'];previous=Path(prepared['root'])
            manifest=previous/'dataset-manifest.json'
            if sha(manifest)!=prepared['manifest_sha256']:raise ValueError('Prepared manifest changed')
            dataset=json.loads(manifest.read_text())
            if dataset['source_sha256']!=spec['decisions_sha256'] or len(dataset['examples'])!=86:
                raise ValueError('Prepared corpus membership changed')
            (root/'tensors').mkdir(exist_ok=True)
            for row in dataset['examples']:
                source=previous/row['tensor_file'];destination=root/row['tensor_file']
                if sha(source)!=row['tensor_sha256']:raise ValueError('Prepared tensor changed')
                if not destination.exists():os.link(source,destination)
            for filename in ['dataset-manifest.json','processor.json','admission.json']:
                shutil.copyfile(previous/filename,root/filename)
            write(root/'prepared-dataset-receipt.json',{'source':str(previous),'manifest_sha256':sha(manifest),
                'verified_examples':86,'teacher_regenerated':False,'retokenized':False,
                'note':'Reuse exact serving-compatible inputs and complete targets from the prior data preparation; only the model training library changes.'})
            progress('dataset_reused','Verified and reused all 86 complete examples; no new teacher calls.',training_started=False)
        else:
            source=LANE/'decisions.jsonl.gz'
            if sha(source)!=spec['decisions_sha256']:raise ValueError('Admitted corpus changed')
            states=[json.loads(line) for line in gzip.decompress(source.read_bytes()).splitlines() if line]
            if len(states)!=43:raise ValueError('Pilot membership changed')
            write(root/'admission.json',json.loads((LANE/'admission.json').read_text()))
            progress('backward_explanations','Generating brief public Astra explanations for all 43 admitted actions.',training_started=False)
            (root/'rationales').mkdir(exist_ok=True)
            summaries={}
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures={pool.submit(pilot.rationale,state,spec):state['id'] for state in states}
                for future in as_completed(futures):
                    record=future.result()
                    if not record['accepted']:raise ValueError('Backward explanation failed: '+str(record.get('error')))
                    summaries[futures[future]]=record['rationale']
                    progress('backward_explanations',f'{len(summaries)}/43 public action explanations saved.',training_started=False)
            processor=AutoProcessor.from_pretrained(spec['base_model'],local_files_only=True,
                min_pixels=65536,max_pixels=1048576)
            write(root/'processor.json',{'image_processor':processor.image_processor.to_dict(),
                'chat_template_sha256':sha(Path(spec['base_model'])/'chat_template.jinja'),
                'min_pixels':65536,'max_pixels':1048576,'prefix_validation':'Exact processor token-prefix equality on every example.'})
            rows=[];(root/'tensors').mkdir(exist_ok=True)
            progress('tokenizing','Building 43 forward/backward pairs with complete image inputs and targets.',training_started=False)
            for state in states:
                backward=copy.deepcopy(state['messages'])
                backward[0]['content']+='\nFor the following recorded-action explanation request, explain the supplied command without executing it, then finish with DONE.'
                backward.append({'role':'user','content':'The expert chose this exact action for the preceding observation. Explain the visible evidence, intended effect and relevant constraints. Do not invent future observations. Finish with DONE.\nExpert action:\n'+json.dumps(state['action'])})
                for direction,messages,answer in [('forward',state['messages'],json.dumps(state['action'],ensure_ascii=False)),('backward',backward,'DONE')]:
                    target={'role':'assistant','reasoning_content':summaries[state['id']],'content':answer}
                    batch,meta=encode_example(processor,messages,[],target,spec['context'])
                    if batch is None:raise ValueError('Pilot example exceeds the frozen context: '+state['id'])
                    name='tensors/'+state['id']+'-'+direction+'.pt';torch.save(batch,root/name)
                    rows.append({'id':state['id']+'-'+direction,'state_id':state['id'],'decision_id':state['id'],
                        'suite':state['suite'],'task':state['task'],'episode':state['episode'],'source':'warmstart',
                        'direction':direction,'tensor_file':name,'tensor_sha256':sha(root/name),**meta})
            write(root/'dataset-manifest.json',{'examples':rows,'states':43,'pairs':43,'examples_count':86,
                'source_sha256':spec['decisions_sha256'],'rationale':'Brief public backward explanations from Astra; no provider-private reasoning.',
                'context':spec['context'],'target_truncation':False})
        # HF5 requires an explicit modality mask. Validate it against the official
        # processor for every frozen example without retokenizing or changing images.
        from trainer import load_tensor, modality_inputs
        processor=AutoProcessor.from_pretrained(spec['base_model'],local_files_only=True)
        dataset=json.loads((root/'dataset-manifest.json').read_text())
        for row in dataset['examples']:
            batch=load_tensor(root/row['tensor_file'])
            adapted=modality_inputs(batch,hf_config)
            expected=torch.tensor(processor.create_mm_token_type_ids(batch['input_ids'].tolist()),dtype=batch['input_ids'].dtype)
            if not torch.equal(adapted['mm_token_type_ids'],expected):raise ValueError('Modality adapter differs from HF5 processor')
            if any(not torch.equal(batch[k],adapted[k]) for k in batch):raise ValueError('Frozen model input changed')
        write(root/'modality-parity.json',{'examples_checked':len(dataset['examples']),'matches_official_processor':True,
            'frozen_inputs_and_targets_unchanged':True,'additional_runtime_field':'mm_token_type_ids'})
        command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=8',str(LANE/'trainer.py'),
                 '--spec',str(root/'spec.json'),'--root',str(root)]
        progress('optimizer_smoke','Testing two optimizer updates on the longest complete examples; smoke weights are discarded.',training_started=False)
        pilot.run_logged('smoke',command+['--mode','smoke'],env,7200)
        if not json.loads((root/'smoke/complete.json').read_text())['success']:raise ValueError('Optimizer smoke failed')
        def verify_saved(mode):
            ready=json.loads((root/('smoke-adapter-ready.json' if mode=='smoke' else 'adapter-ready.json')).read_text())
            completion=json.loads((root/mode/'complete.json').read_text())
            verification={**spec,'checkpoint':ready['checkpoint'],'training_root':str(root),'verification_mode':mode,
                'expected_post_update_ce':completion['post_update_full_target_ce']}
            path=root/(mode+'-verification-spec.json');write(path,verification)
            progress('checkpoint_loss_replay',f'Reloading all 700 {mode} adapter tensors and replaying the recorded loss.',training_started=mode=='train')
            pilot.run_logged(mode+'-reload',[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=8',
                str(LANE/'verify.py'),'--spec',str(path),'--mode','verify'],env,1800)
            if not json.loads((root/(mode+'-loss-replay.json')).read_text())['passed']:raise ValueError('Fresh model reload did not reproduce loss')
        verify_saved('smoke')
        progress('warmstart_training','Training 32 optimizer updates on all 43 forward/backward action pairs.',training_started=True)
        pilot.run_logged('train',command+['--mode','train'],env,14400)
        ready=json.loads((root/'adapter-ready.json').read_text())
        if not ready['final']:raise ValueError('Missing final adapter checkpoint')
        verify_saved('train')
        progress('merging_checkpoint','Exporting the trained adapter into a complete native-serving model.',training_started=True)
        # Merge in a fresh process after all distributed training workers exit.
        pilot.run_logged('merge',[sys.executable,str(LANE/'merge.py'),'--spec',str(root/'spec.json')],env,3600)
        ready=json.loads((root/'warmstart-ready.json').read_text())
        write(root/'complete.json',{'success':True,'optimizer_steps':32,'warmstart':ready,
            'completed_utc':datetime.now(timezone.utc).isoformat()})
        progress('warmstart_ready','Warmstart checkpoint is durable; matched native robot evaluation is next.',training_started=True)
    except Exception as error:
        write(root/'complete.json',{'success':False,'error':str(error),'completed_utc':datetime.now(timezone.utc).isoformat()})
        progress('warmstart_failed',str(error),training_started=(root/'train/metrics.jsonl').exists())
        raise
    finally:
        for _ in range(120):
            if (root/'collected').exists():break
            time.sleep(5)


if __name__=='__main__':main()
