"""Recover the completed Qwen warmstart without additional optimizer updates."""
import argparse
from datetime import datetime, timedelta, timezone
from functools import partial
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from efe_training_data import sha, write


def stamp():return datetime.now(timezone.utc).isoformat()


def attach_exact(model, checkpoint, report=None):
    """Restore canonical PEFT tensors directly, bypassing HF5 key conversion."""
    import torch
    from peft import LoraConfig, get_peft_model
    from safetensors.torch import load_file
    config=LoraConfig.from_pretrained(checkpoint);config.inference_mode=False
    model=get_peft_model(model,config)
    parameters={name.replace('.lora_A.default.','.lora_A.').replace('.lora_B.default.','.lora_B.'):p
        for name,p in model.named_parameters() if '.lora_A.default.' in name or '.lora_B.default.' in name}
    state=load_file(str(Path(checkpoint)/'adapter_model.safetensors'),device='cpu')
    inspection={'saved_tensors':len(state),'expected_tensors':len(parameters),
        'saved_keys':list(state),'missing_keys':sorted(set(parameters)-set(state)),
        'unexpected_keys':sorted(set(state)-set(parameters)),
        'nonzero_B_tensors':sum(bool(v.count_nonzero()) for k,v in state.items() if '.lora_B.' in k)}
    if report:write(report,inspection)
    if set(state)!=set(parameters) or len(state)!=700:raise ValueError('Canonical adapter key coverage is not exact')
    with torch.no_grad():
        for name,value in state.items():
            parameter=parameters[name]
            if parameter.shape!=value.shape or parameter.dtype!=value.dtype:raise ValueError('Adapter shape/dtype mismatch: '+name)
            if not torch.isfinite(value).all():raise ValueError('Nonfinite saved adapter: '+name)
            parameter.copy_(value.to(parameter.device))
            if not torch.equal(parameter.detach().cpu(),value):raise ValueError('Adapter tensor reload differs: '+name)
    if inspection['nonzero_B_tensors']!=350:raise ValueError('Saved adapter lacks learned updates')
    inspection.update(all_tensors_restored_exactly=True,loading='Exact named-parameter copies; no heuristic renaming or Transformers weight conversion.')
    if report:write(report,inspection)
    return model


def verify(spec,root):
    import torch
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from transformers import AutoConfig, AutoModelForImageTextToText
    from peft import LoraConfig,get_peft_model
    from trainer import load_tensor,target_loss,training_plan
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(local)
    dist.init_process_group('nccl',timeout=timedelta(seconds=spec.get('collective_timeout_seconds',900)));device=torch.device('cuda',local)
    if dist.get_world_size()!=8:raise ValueError('Verification must match the original eight-rank layout')
    model_source=spec.get('local_base_model',spec['base_model'])
    if rank==0:
        model=AutoModelForImageTextToText.from_pretrained(model_source,dtype=torch.float32,
            attn_implementation='sdpa',local_files_only=True)
        model.to(dtype=torch.float32)
        model=attach_exact(model,spec['checkpoint'],root/'adapter-load-verification.json')
    else:
        with torch.device('meta'):
            model=AutoModelForImageTextToText.from_config(AutoConfig.from_pretrained(model_source,local_files_only=True))
        model.config._attn_implementation='sdpa';model.to(dtype=torch.float32)
        config=LoraConfig.from_pretrained(spec['checkpoint']);config.inference_mode=False
        model=get_peft_model(model,config)
    model.config.use_cache=False
    if spec.get('adapter_gather_qualification'):
        from efe_qwen36_adapter_io import parameter_shapes
        expected_adapter=parameter_shapes(model)
    decoder_types={type(m) for m in model.modules() if type(m).__name__.endswith('DecoderLayer')}
    wrapped=FSDP(model,auto_wrap_policy=partial(transformer_auto_wrap_policy,transformer_layer_cls=decoder_types),
        sharding_strategy=ShardingStrategy.FULL_SHARD,use_orig_params=True,
        mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,reduce_dtype=torch.float32,buffer_dtype=torch.float32),
        param_init_fn=lambda m:m.to_empty(device=device,recurse=False),device_id=device,sync_module_states=True,limit_all_gathers=True)
    source=Path(spec['training_root']);rows=json.loads((source/'dataset-manifest.json').read_text())['examples']
    source_spec=json.loads((source/'spec.json').read_text());_,schedule,_,digest=training_plan(rows,source_spec)
    mode=spec.get('verification_mode','train')
    saved_schedule=json.loads((source/mode/'schedule.json').read_text())
    if mode=='train':
        if digest!=saved_schedule['schedule_sha256']:raise ValueError('Original training schedule differs')
    elif mode=='smoke':
        lookup={r['id']:r for r in rows};schedule=[lookup[k] for k in saved_schedule['exposures']]
    else:raise ValueError('Unknown verification mode')
    row=schedule[rank];path=source/row['tensor_file']
    if sha(path)!=row['tensor_sha256']:raise ValueError('Original score input changed')
    batch={k:v.to(device) for k,v in load_tensor(path).items()};wrapped.eval()
    with torch.no_grad():loss=target_loss(wrapped,batch)
    dist.all_reduce(loss);loss/=8
    value=loss.item();expected=spec['expected_post_update_ce'];error=abs(value-expected)
    if rank==0:write(root/(mode+'-loss-replay.json' if 'verification_mode' in spec else 'loss-replay.json'),{'expected_ce':expected,'reloaded_ce':value,'absolute_error':error,
        'tolerance':0.0005,'passed':error<=0.0005,'optimizer_updates':0,'examples':8,'completed_utc':stamp()})
    if not torch.isfinite(loss) or error>0.0005:raise ValueError('Reloaded adapter does not reproduce the final training score')
    if spec.get('adapter_gather_qualification'):
        from efe_qwen36_restart import qualify_adapter_gather
        from efe_qwen36_adapter_io import save_exact
        state,qualification=qualify_adapter_gather(wrapped,expected_adapter)
        with torch.no_grad():after=target_loss(wrapped,batch)
        dist.all_reduce(after);after/=8
        change=abs(after.item()-value)
        if not torch.isfinite(after) or change>0.0005:
            raise ValueError('Full-model loss changed during adapter-gather qualification')
        if rank==0:
            audit=save_exact(model,state,expected_adapter,root/'gathered-adapter')
            qualification.update(before_ce=value,after_ce=after.item(),loss_change=change,
                loss_tolerance=0.0005,serialization=audit,optimizer_updates=0,completed_utc=stamp())
            write(root/'adapter-gather-qualification.json',qualification)
        del state
    dist.barrier();dist.destroy_process_group()


def export(spec,root):
    import torch
    from transformers import AutoModelForImageTextToText
    if not json.loads((root/'loss-replay.json').read_text())['passed']:raise ValueError('Loss replay must pass before export')
    model=AutoModelForImageTextToText.from_pretrained(spec['base_model'],dtype=torch.bfloat16,
        device_map='cpu',attn_implementation='sdpa',local_files_only=True)
    adapted=attach_exact(model,spec['checkpoint'],root/'export-adapter-verification.json')
    witness=None
    for name,module in adapted.named_modules():
        if not hasattr(module,'get_delta_weight'):continue
        delta=module.get_delta_weight('default').detach();index=delta.abs().flatten().argmax().item()
        layer=module.get_base_layer();before=layer.weight.detach().flatten()[index].float().item()
        if (layer.weight.detach().float()+delta.float()).to(layer.weight.dtype).flatten()[index].float().item()!=before:
            witness=(name,layer,index,before);break
    if witness is None:raise ValueError('No representable merged-weight update')
    merged=adapted.merge_and_unload(safe_merge=True)
    if witness[1].weight.detach().flatten()[witness[2]].float().item()==witness[3]:raise ValueError('Merge did not change witnessed weight')
    destination=root/'model';destination.mkdir(exist_ok=False);merged.config.use_cache=True
    merged.save_pretrained(destination,safe_serialization=True,max_shard_size='4GB')
    for asset in Path(spec['base_model']).iterdir():
        if asset.is_file() and asset.suffix in {'.json','.txt','.jinja','.model'} and asset.name!='model.safetensors.index.json':
            shutil.copyfile(asset,destination/asset.name)
    base_index=json.loads((Path(spec['base_model'])/'model.safetensors.index.json').read_text())['weight_map']
    index=json.loads((destination/'model.safetensors.index.json').read_text())['weight_map']
    expected={key for key in base_index if not key.startswith('mtp.')}
    if set(index)!=expected:
        write(root/'export-key-mismatch.json',{'missing':sorted(expected-set(index)),'unexpected':sorted(set(index)-expected)})
        raise ValueError('Exported model keys differ from the original native checkpoint')
    files={p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in destination.iterdir() if p.is_file()}
    if not set(index.values())<=set(files):raise ValueError('Missing exported weight shard')
    revision='efe-qwen36-constrained-warm-lora-v1-step32-recovered'
    write(destination/'checkpoint-manifest.json',{'revision':revision,'files':files,'step':32,
        'source_base':spec['model'],'source_revision':spec['base_revision'],'adapter_checkpoint':spec['checkpoint'],
        'full_parameters':False,'merged_adapter':True,'adapter_reload_verified':True,'merged_weight_update_verified':witness[0],
        'loss_replay':json.loads((root/'loss-replay.json').read_text()),'additional_optimizer_updates':0,
        'dataset_sha256':spec['dataset_sha256']})
    write(root/'warmstart-ready.json',{'checkpoint':str(destination),'manifest':str(destination/'checkpoint-manifest.json'),
        'revision':revision,'step':32,'final':True,'selection':'Original fixed step-32 checkpoint; export-only recovery, no additional training.'})


def run(spec,root,lane):
    root.mkdir(parents=True,exist_ok=True)
    def progress(phase,detail):
        write(root/'progress.json',{'phase':phase,'detail':detail,'updated_utc':stamp(),'training_started':True,
            'optimizer_steps_completed':32,'additional_optimizer_updates':0,'workflow':os.environ['EFE_WORKFLOW_ID']})
    def logged(name,command,timeout):
        with (root/(name+'.log')).open('w') as out:
            result=subprocess.run(command,stdout=out,stderr=subprocess.STDOUT,env=os.environ,timeout=timeout)
        if result.returncode:raise RuntimeError(name+' failed; see '+name+'.log')
    write(root/'worker-start.json',{'workflow':os.environ['EFE_WORKFLOW_ID'],'started_utc':stamp()})
    try:
        progress('checkpoint_inspection','Verifying the retained step-32 adapter and original training receipts.')
        source=Path(spec['training_root'])
        for name,wanted in [('train/complete.json',spec['training_complete_sha256']),('dataset-manifest.json',spec['dataset_sha256'])]:
            if sha(source/name)!=wanted:raise ValueError('Original training receipt changed: '+name)
        checkpoint=Path(spec['checkpoint']);manifest=json.loads((checkpoint/'checkpoint-manifest.json').read_text())
        for name,row in manifest['files'].items():
            if sha(checkpoint/name)!=row['sha256']:raise ValueError('Checkpoint checksum mismatch: '+name)
        from safetensors import safe_open
        with safe_open(str(checkpoint/'adapter_model.safetensors'),framework='pt',device='cpu') as f:
            keys=list(f.keys());shapes={k:list(f.get_slice(k).get_shape()) for k in keys}
            summary={'tensors':len(keys),'bytes':(checkpoint/'adapter_model.safetensors').stat().st_size,
                'keys':keys,'shapes':shapes,'manifest':manifest,'original_checkpoint_untouched':True}
        write(root/'checkpoint-inspection.json',summary)
        if len(keys)!=700 or any(len(shape)!=2 or min(shape)<=0 for shape in shapes.values()):raise ValueError('Saved adapter tensors are missing or incomplete')
        write(root/'versions.json',{n:importlib.metadata.version(n) for n in ['torch','transformers','peft','safetensors']})
        from huggingface_hub import snapshot_download
        progress('preparing_base','Fetching the original pinned base; no training will be repeated.')
        snapshot_download(spec['model'],revision=spec['base_revision'],local_dir=spec['base_model'],max_workers=4,
            allow_patterns=['*.json','*.safetensors','*.txt','*.jinja','LICENSE*'])
        progress('checkpoint_loss_replay','Reloading every adapter tensor exactly and replaying the original final-loss check.')
        logged('verify',[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=8',
            str(lane/'recovery.py'),'--spec',str(lane/'spec.json'),'--mode','verify'],1800)
        progress('exporting_checkpoint','Verified the trained adapter; merging a complete native-serving checkpoint.')
        logged('export',[sys.executable,str(lane/'recovery.py'),'--spec',str(lane/'spec.json'),'--mode','export'],3600)
        write(root/'complete.json',{'success':True,'optimizer_steps':32,'additional_optimizer_updates':0,
            'warmstart':json.loads((root/'warmstart-ready.json').read_text()),'completed_utc':stamp()})
        progress('warmstart_ready','Step-32 checkpoint recovered and verified; ready for the matched robot evaluation.')
    except Exception as error:
        write(root/'complete.json',{'success':False,'error':str(error),'completed_utc':stamp(),'additional_optimizer_updates':0})
        progress('export_recovery_failed',str(error));raise
    finally:
        for _ in range(120):
            if (root/'collected').exists():break
            time.sleep(5)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--mode',choices=['run','verify','export'],default='run')
    a=p.parse_args();spec=json.loads(a.spec.read_text());root=Path(spec['archive_root'])
    if a.mode=='run':run(spec,root,a.spec.parent)
    elif a.mode=='verify':verify(spec,root)
    else:export(spec,root)
