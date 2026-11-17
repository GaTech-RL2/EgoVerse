"""Qwen3.6 language LoRA SFT with full-target likelihood on allocated H100s.

One example per GPU, FSDP full sharding, no packing or target truncation.
The smoke run is discarded. Named continuations restore explicit training state.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

try:
    from .efe_training_data import sha, write
except ImportError:
    from efe_training_data import sha, write


def load_tensor(path):
    import torch
    from transformers.feature_extraction_utils import BatchFeature
    # Earlier frozen processor outputs use this specific Transformers wrapper.
    # Keep restricted deserialization and validate the tensor-only payload.
    with torch.serialization.safe_globals([BatchFeature]):
        batch=dict(torch.load(path,map_location='cpu',weights_only=True))
    if 'tensor_storage' in batch:
        try:from .efe_visual_tensor_store import expand
        except ImportError:from efe_visual_tensor_store import expand
        batch=expand(batch,path)
    if not {'input_ids','attention_mask','labels'}<=set(batch) or not all(isinstance(v,torch.Tensor) for v in batch.values()):
        raise ValueError('Training example must contain only tensors and required masks')
    return batch


def modality_inputs(inputs, config):
    """Supply HF5 modality labels without changing the frozen HF4 token/image inputs."""
    import torch
    inputs=dict(inputs)
    if 'image_grid_thw' not in inputs and 'video_grid_thw' not in inputs:return inputs
    ids=inputs['input_ids'];types=torch.zeros_like(ids)
    types[ids==config.image_token_id]=1
    types[ids==config.video_token_id]=2
    merge=config.vision_config.spatial_merge_size
    for key,kind in [('image_grid_thw',1),('video_grid_thw',2)]:
        if key in inputs and int((types==kind).sum())!=int(inputs[key].prod(-1).sum())//(merge*merge):
            raise ValueError('Image/video tokens do not match the frozen vision grid')
    if 'mm_token_type_ids' in inputs and not torch.equal(inputs['mm_token_type_ids'],types):
        raise ValueError('Existing modality labels disagree with model token IDs')
    inputs['mm_token_type_ids']=types
    return inputs


def target_loss(model, batch):
    import torch
    import torch.nn.functional as F
    labels = batch['labels'][0]
    selected = torch.nonzero(labels != -100, as_tuple=False).flatten()
    if selected.numel() == 0 or selected.min() <= 0:
        raise ValueError('Empty target or missing causal context')
    inputs = modality_inputs({k: v for k, v in batch.items() if k != 'labels'},model.config)
    output = model(**inputs, logits_to_keep=selected-1, use_cache=False)
    logits = output.logits[0]
    if logits.shape[0] != selected.numel(): raise ValueError('Full-target logits misaligned')
    # Chunk the float32 softmax without changing the target-token mean.
    loss = sum(F.cross_entropy(logits[i:i+128].float(), labels[selected[i:i+128]], reduction='sum')
               for i in range(0, selected.numel(), 128)) / selected.numel()
    return loss


def selection_reference_required(mode,start_step,require_full_model_replay):
    """Compare selection scores to the untouched parent; replay resumed weights."""
    if mode=='train' and start_step:
        if not require_full_model_replay:
            raise ValueError('Resumed LeAct must verify its committed full-model loss')
        return False
    return True


def presample(rows, steps, seed):
    """Uniform suite/task/episode/decision and exactly equal forward/backward mass."""
    groups = {}
    for row in rows:
        if row['direction'] != 'forward': continue
        groups.setdefault(row.get('source','warmstart'), {}).setdefault(row['suite'], {}).setdefault(row['task'], {}).setdefault(row['episode'], {}).setdefault(row.get('decision_id',row['state_id']), []).append(row['state_id'])
    lookup = {(r['state_id'], r['direction']): r for r in rows}
    rng = random.Random(seed); result = []
    sources=sorted(groups)
    for index in range(steps * 16):
        # Equal replay/new mass when both sources exist; candidate count never
        # multiplies a decision's contribution.
        source=sources[index%len(sources)];suites=groups[source]
        suite = rng.choice(sorted(suites)); task = rng.choice(sorted(suites[suite])); episode = rng.choice(sorted(suites[suite][task]))
        decision=rng.choice(sorted(suites[suite][task][episode]));state=rng.choice(suites[suite][task][episode][decision])
        result.extend([lookup[state, 'forward'], lookup[state, 'backward']])
    return result


def training_plan(rows, spec):
    decision_units=len({(r.get('source','warmstart'),r.get('decision_id',r['state_id'])) for r in rows})
    steps=max(1,math.ceil(2*decision_units*spec.get('balanced_passes',1.0)/spec['effective_batch']))
    if 'fixed_optimizer_steps' in spec:
        steps=spec['fixed_optimizer_steps']
        if type(steps) is not int or steps<1:raise ValueError('Fixed optimizer budget must be a positive integer')
    if spec.get('expected_optimizer_steps',steps)!=steps:
        raise ValueError('Optimizer budget differs from the frozen comparison')
    policy=spec.get('sampling_policy','balanced-replacement-v1')
    if policy=='coverage-first-balanced-v1':
        try:from .efe_training_data import balanced_order
        except ImportError:from efe_training_data import balanced_order
        lookup={(r['state_id'],r['direction']):r for r in rows}
        forward=[r for r in rows if r['direction']=='forward']
        if len(forward)!=decision_units:raise ValueError('Coverage-first warmstart requires one pair per decision')
        coverage=[]
        for row in balanced_order(forward,spec['seed']):
            coverage.extend([lookup[row['state_id'],'forward'],lookup[row['state_id'],'backward']])
        if len(coverage)>steps*spec['effective_batch']:raise ValueError('Budget cannot cover all complete pairs')
        schedule=coverage+presample(rows,steps,spec['seed'])[len(coverage):]
    elif policy=='balanced-replacement-v1':schedule=presample(rows,steps,spec['seed'])
    else:raise ValueError('Unknown sampling policy')
    digest=hashlib.sha256(json.dumps([r['id'] for r in schedule],separators=(',',':')).encode()).hexdigest()
    if spec.get('expected_schedule_sha256',digest)!=digest:
        raise ValueError('Sampled exposures differ from the frozen comparison')
    fractions=spec.get('checkpoint_fractions',[.25,.5,1.])
    if not fractions or any(not 0<f<=1 for f in fractions) or 1. not in fractions:
        raise ValueError('Checkpoint fractions must include the final budget')
    return steps,schedule,sorted({max(1,math.ceil(steps*f)) for f in fractions}),digest


def main():
    import torch
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy, StateDictType, FullStateDictConfig, FullOptimStateDictConfig
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from transformers import AutoConfig, AutoProcessor, AutoModelForImageTextToText
    from peft import LoraConfig, get_peft_model, PeftModel
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--mode', choices=['smoke', 'train', 'scoring-check', 'hf-rescore'], required=True)
    args = parser.parse_args(); spec = json.loads(args.spec.read_text()); root = args.root
    continuation=spec.get('continuation')
    durable=spec.get('durable_restart') if args.mode=='train' else None
    restart_policy=continuation or durable
    if restart_policy:
        if spec.get('adapter_restart_only'):
            try:from . import efe_qwen36_restart as restart
            except ImportError:import efe_qwen36_restart as restart
        else:
            try:from . import efe_training_state as restart
            except ImportError:import efe_training_state as restart
        if args.mode!='train':raise ValueError('Continuation uses the previously validated memory recipe and verifies restored optimizer updates directly')
    rank = int(os.environ['RANK']); local = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local); device = torch.device('cuda', local)
    dist.init_process_group('nccl', timeout=timedelta(seconds=spec.get('collective_timeout_seconds',240)))
    world = dist.get_world_size()
    if world != 8 or spec['effective_batch'] != 32: raise ValueError('Unvalidated distributed layout')
    # All ranks run a real collective before loading model/optimizer state.
    probe = torch.tensor(float(rank+1), device=device); dist.all_reduce(probe)
    if probe.item() != world*(world+1)/2: raise ValueError('NCCL all-reduce failed')
    torch.manual_seed(spec['seed']); random.seed(spec['seed'])
    rows = json.loads((root/'dataset-manifest.json').read_text())['examples']
    if not rows: raise ValueError('Empty frozen dataset')
    planned_steps,planned_schedule,planned_saves,schedule_digest=training_plan(rows,spec)
    if continuation:restart.validate_continuation(spec,planned_schedule)
    start_step=spec['active_restore']['step'] if restart_policy else 0
    if start_step>=planned_steps:raise ValueError('Already-complete continuation must be finalized without another optimizer launch')
    model_path=spec['active_restore']['path'] if restart_policy and not spec.get('adapter_restart_only') else spec.get('local_base_model',spec['base_model'])
    directory = root / args.mode; directory.mkdir(exist_ok=True)
    if (directory/'complete.json').exists(): raise ValueError('Refusing to overwrite completed training')
    started = time.monotonic()
    # Rank zero alone loads FP32 master weights. Other ranks materialize each
    # sharded module on GPU, then receive rank-zero initialization.
    if rank == 0:
        model = AutoModelForImageTextToText.from_pretrained(model_path, dtype=torch.float32,
                    attn_implementation='sdpa', local_files_only=True)
    else:
        with torch.device('meta'):
            model = AutoModelForImageTextToText.from_config(AutoConfig.from_pretrained(model_path, local_files_only=True))
        model.config._attn_implementation = 'sdpa'
    # Nested vision configs can retain bf16 on meta-created workers even when
    # the language modules use FP32. FSDP master parameters must agree across
    # every module and rank before sharding; forward casting remains bf16.
    model.to(dtype=torch.float32)
    if any(p.dtype!=torch.float32 for p in model.parameters()):
        raise ValueError('FSDP master parameter dtypes are not uniformly FP32')
    if model.config.model_type != 'qwen3_5_moe':raise ValueError('Unexpected warmstart architecture')
    targets=[name for name,module in model.named_modules() if '.language_model.' in name and isinstance(module,torch.nn.Linear)]
    if len(targets)<40:raise ValueError('Could not identify language attention/shared-expert linear layers')
    model=get_peft_model(model,LoraConfig(r=spec['lora_rank'],lora_alpha=spec['lora_alpha'],
        lora_dropout=0.0,bias='none',target_modules=targets,task_type='CAUSAL_LM'))
    from efe_qwen36_adapter_io import parameter_shapes,save_exact
    expected_adapter=parameter_shapes(model)
    if len(expected_adapter)!=700 or sum(math.prod(s) for s in expected_adapter.values())!=44954880:
        raise ValueError('The frozen 350-module LoRA parameter contract changed')
    model._efe_adapter_shapes=expected_adapter
    if spec.get('adapter_restart_only') and start_step and rank==0:
        from efe_qwen36_adapter_io import load_into_exact
        load_into_exact(model,spec['active_restore']['path'],root/'adapter-resume-verification.json')
    decoder_types={type(module) for module in model.modules() if type(module).__name__.endswith('DecoderLayer')}
    if not decoder_types:raise ValueError('No decoder layers for sharding')
    if rank==0:write(root/'adapter-configuration.json',{'target_modules':targets,'r':spec['lora_rank'],
        'alpha':spec['lora_alpha'],'vision_encoder_frozen':True,'routed_expert_banks_frozen':True,
        'note':'Train language dense attention/shared-expert LoRA adapters; this is a separate recipe from full 8B fine-tuning.'})
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    wrapped = FSDP(model, auto_wrap_policy=partial(transformer_auto_wrap_policy, transformer_layer_cls=decoder_types),
                   sharding_strategy=ShardingStrategy.FULL_SHARD, use_orig_params=True,
                   mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32),
                   param_init_fn=lambda module: module.to_empty(device=device, recurse=False),
                   device_id=device, sync_module_states=True, limit_all_gathers=True)
    if args.mode=='hf-rescore':
        from efe_leact_hf_rescore import run
        run(wrapped,spec,root)
        dist.barrier();dist.destroy_process_group();return
    optimizer = torch.optim.AdamW([p for p in wrapped.parameters() if p.requires_grad], lr=spec['lr'],
                                 betas=(0.9, 0.999), weight_decay=.01, foreach=False)
    if restart_policy and start_step:restart.restore_optimizer(wrapped,optimizer,spec,root)
    accumulation = spec['effective_batch'] // world
    steps = planned_steps if args.mode == 'train' else 2
    if args.mode == 'smoke':
        # Every rank sees real p95/max-length images and complete targets; this
        # is an optimizer-state memory test, not merely a model loading test.
        ordered = sorted(rows, key=lambda r: r['tokens'])
        stress = [ordered[min(len(ordered)-1, int(.95*len(ordered)))], ordered[-1]]
        schedule = [stress[(i//32) % 2] for i in range(steps*32)]
    else:
        schedule = planned_schedule
    if args.mode=='train' and spec.get('require_full_model_replay'):
        from efe_qwen36_health import validate
        validate(wrapped,spec,root,schedule,start_step)
    reference_metrics={}
    if args.mode=='train' and spec.get('reference_training_metrics'):
        from efe_qwen36_health import training_reference
        reference_metrics=training_reference(spec)
    if rank == 0:
        write(directory/'schedule.json', {'steps': steps, 'effective_batch':32, 'accumulation':4,
              'exposures': [r['id'] for r in schedule], 'unique_states':len({r['state_id'] for r in schedule}),
              'balanced_passes':spec.get('balanced_passes',1.0),
              'schedule_sha256':schedule_digest if args.mode=='train' else None,
              'resumed_from_step':start_step,
              'sampling':('One complete pass over all decision pairs, then balanced hierarchical sampling with replacement; every admitted state is exposed.'
                  if spec.get('sampling_policy')=='coverage-first-balanced-v1' else 'Balanced hierarchical sampling with replacement; not a coverage-complete epoch'),
              'suite_exposures': {s:sum(r['suite']==s for r in schedule) for s in sorted({r['suite'] for r in rows})}})
        write(directory/'configuration.json', {'trainable_parameters': trainable, 'frozen_parameters': frozen,
              'optimizer': 'AdamW FP32 LoRA parameters/states', 'lr': spec['lr'], 'effective_batch':32,
              'sharding':'FSDP FULL_SHARD', 'gradient_checkpointing': True, 'forward_backward_equal_mass':True,
              'loss_reduction':'mean of complete target-token CE per example, then mean over 32 examples',
              'initial_checkpoint':model_path, 'context':spec['context'], 'mode':args.mode,
              'continuation':continuation,'durable_restart':durable,'resumed_from_step':start_step})
    save_steps = planned_saves
    warmup = max(1, math.ceil(steps*.05)); checkpoints=[]; metrics=[]
    if (root/'scoring-probes.json').exists() and selection_reference_required(
            args.mode,start_step,spec.get('require_full_model_replay',False)):
        wrapped.eval();checks=[]
        with torch.no_grad():
            for probe in json.loads((root/'scoring-probes.json').read_text()):
                path=root/probe['tensor_file']
                if sha(path)!=probe['tensor_sha256']:raise ValueError('Scoring probe changed')
                batch={k:v.to(device) for k,v in load_tensor(path).items()}
                actual=-target_loss(wrapped,batch).item();error=abs(actual-probe['mean_target_logp'])
                limit=.0005 if probe.get('score_backend')=='training_hf_fsdp' else .05
                checks.append({'condition':probe['condition'],'vllm_logp':probe['mean_target_logp'],
                    'hf_logp':actual,'absolute_error':error,'rank':rank,
                    'target_tokens':int((batch['labels']!=-100).sum()),
                    'finite':math.isfinite(actual),'tolerance':limit,'passed':math.isfinite(actual) and error<=limit})
        # Persist the failure before raising; otherwise an overnight stopped
        # allocation leaves no numerical evidence to diagnose the mismatch.
        gathered=[None]*world;dist.all_gather_object(gathered,checks)
        result={'checks_by_rank':gathered,'tolerance':max(c['tolerance']for values in gathered for c in values),
            'passed':all(c['passed'] for values in gathered for c in values),
            'maximum_absolute_error':max(c['absolute_error'] for values in gathered for c in values),
            'optimizer_updates_before_check':0,'model':model_path,
            'workflow':os.environ.get('EFE_WORKFLOW_ID')}
        if rank==0:
            write(directory/'scoring-parity.json',result)
            print('EFE_SCORING_PARITY '+json.dumps(result),flush=True)
        if args.mode=='scoring-check':
            dist.barrier();dist.destroy_process_group();return
        if not result['passed']:
            raise ValueError('Training/selection complete-target likelihood differs by over 0.05 nats/token; see '+str(directory/'scoring-parity.json'))
    elif args.mode=='scoring-check':
        raise ValueError('No frozen scoring probes for the numerical diagnostic')
    for step in range(start_step,steps):
        tick = time.monotonic(); optimizer.zero_grad(set_to_none=True)
        factor = (step+1)/warmup if step<warmup else .5*(1+math.cos(math.pi*(step-warmup)/max(1,steps-warmup)))
        for group in optimizer.param_groups:
            group['lr'] = restart.continuation_lr(spec,step) if continuation else spec['lr']*factor
        stats = torch.zeros(6, device=device); wrapped.train()
        torch.cuda.reset_peak_memory_stats()
        for micro in range(accumulation):
            row = schedule[step*32+micro*world+rank]
            path=root/row['tensor_file']
            if sha(path)!=row['tensor_sha256']: raise ValueError('Frozen tensor changed')
            batch={k:v.to(device) for k,v in load_tensor(path).items()}
            # Synchronize every microbatch to keep full gradients sharded;
            # no_sync would materialize full gradient copies at long context.
            loss=target_loss(wrapped,batch)
            if not torch.isfinite(loss): raise ValueError('Non-finite training loss')
            (loss/accumulation).backward()
            direction=0 if row['direction']=='forward' else 2
            stats[direction]+=loss.detach();stats[direction+1]+=1
            stats[4]+=row['tokens'];stats[5]+=row['target_tokens']
            del batch,loss
        grad_norm=wrapped.clip_grad_norm_(1.0)
        if not torch.isfinite(grad_norm) or grad_norm<=0: raise ValueError('Non-finite or zero gradients')
        # At the cosine tail a chosen scalar can round to the same FP32 value
        # while other adapter parameters update. Compare every optimizer shard.
        witnesses=[(parameter,parameter.detach().clone())for parameter in wrapped.parameters()
            if parameter.requires_grad and parameter.grad is not None and parameter.numel()]
        dist.all_reduce(stats)
        if spec.get('loss_jump_guard'):
            from efe_qwen36_health import suspicious_loss,check_training_reference
            previous=metrics[-1] if metrics else spec.get('active_restore',{}).get('receipt',{}).get('metric')
            current={'forward_ce':stats[0].item()/max(1,stats[1].item()),
                     'backward_ce':stats[2].item()/max(1,stats[3].item())}
            check_training_reference(current,reference_metrics.get(step+1),step+1)
            if suspicious_loss(current,previous):
                raise ValueError('Loss jump guard stopped before optimizer update: '+json.dumps({'step':step+1,'current':current,'previous':previous}))
        optimizer.step();torch.cuda.synchronize()
        changed_shards=sum(not parameter.detach().equal(before)for parameter,before in witnesses)
        changed=torch.tensor(int(changed_shards>0),device=device)
        del witnesses
        dist.all_reduce(changed)
        if not changed.item():raise ValueError('Optimizer did not change any witnessed parameter')
        peak=torch.tensor([torch.cuda.max_memory_allocated(),torch.cuda.max_memory_reserved()],device=device,dtype=torch.float64)
        dist.all_reduce(peak,op=dist.ReduceOp.MAX)
        record={'step':step+1,'steps':steps,'seconds':time.monotonic()-tick,'lr':optimizer.param_groups[0]['lr'],
                'forward_ce':stats[0].item()/max(1,stats[1].item()),'backward_ce':stats[2].item()/max(1,stats[3].item()),
                'forward_examples':int(stats[1].item()),'backward_examples':int(stats[3].item()),
                'processed_tokens':int(stats[4].item()),'supervised_tokens':int(stats[5].item()),'gradient_norm':grad_norm.item(),
                'max_allocated_bytes_per_gpu':int(peak[0].item()),'max_reserved_bytes_per_gpu':int(peak[1].item())}
        record['ranks_with_verified_parameter_update']=int(changed.item())
        if continuation:record.update(continuation_origin_step=continuation['origin_step'],resumed_from_step=start_step)
        metrics.append(record)
        if rank==0:
            with (directory/'metrics.jsonl').open('a') as out:out.write(json.dumps(record)+'\n')
            write(root/'progress.json',{'phase':args.mode,'detail':f'{args.mode}: optimizer step {step+1}/{steps}',
                  'training_started':args.mode=='train','updated_utc':datetime.now(timezone.utc).isoformat(),**record})
            print(json.dumps(record),flush=True)
        # Require 10% headroom at the real longest-example optimizer step.
        if args.mode=='smoke' and peak[0].item()>.90*torch.cuda.get_device_properties(device).total_memory:
            raise ValueError('Longest-example optimizer smoke lacks 10% memory headroom')
        if restart_policy and restart.restart_due(restart_policy,step+1,start_step,save_steps):
            save_started=time.monotonic()
            if rank==0:write(root/'progress.json',{'phase':'checkpoint_saving','detail':f'Saving full-precision restart state at update {step+1}/{steps}',
                'training_started':True,'updated_utc':datetime.now(timezone.utc).isoformat(),**record})
            ready=restart.save_state(wrapped,optimizer,model,spec,root,step+1,steps,record,step+1 in save_steps)
            if rank==0:
                if ready:checkpoints.append(ready['checkpoint'])
                with (directory/'checkpoint-timings.jsonl').open('a') as out:
                    out.write(json.dumps({'step':step+1,'seconds':time.monotonic()-save_started,'portable':bool(ready)})+'\n')
        elif (args.mode=='train' and step+1 in save_steps) or (args.mode=='smoke' and step+1==steps):
            checkpoint=directory/f'checkpoint-{step+1}'
            with FSDP.state_dict_type(wrapped,StateDictType.FULL_STATE_DICT,FullStateDictConfig(offload_to_cpu=True,rank0_only=True),
                                     FullOptimStateDictConfig(offload_to_cpu=True,rank0_only=True)):
                state=wrapped.state_dict();optim_state=FSDP.optim_state_dict(wrapped,optimizer)
            if rank==0:
                audit=save_exact(model,state,expected_adapter,checkpoint)
                AutoProcessor.from_pretrained(spec['base_model'],local_files_only=True).save_pretrained(checkpoint)
                torch.save({'optimizer':optim_state,'step':step+1,'torch_rng':torch.get_rng_state()},checkpoint/'optimizer.pt')
                files={p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in checkpoint.iterdir() if p.is_file() and p.suffix in {'.json','.safetensors','.jinja','.txt'}}
                revision=spec.get('checkpoint_revision_prefix','efe-qwen36-constrained-warm-lora-v1')+'-'+args.mode+'-step-'+str(step+1)
                write(checkpoint/'checkpoint-manifest.json',{'files':files,'revision':revision,
                      'step':step+1,'balanced_pass_fraction':(step+1)/steps,
                      'source_base':spec['base_model'],'full_parameters':False,'adapter_only':True,
                      'serialization_verified':audit['all_tensors_round_trip_exactly'],'saved_tensors':audit['saved_tensors'],
                      'dataset_sha256':sha(root/'dataset-manifest.json')})
                checkpoints.append(str(checkpoint));write(root/(spec.get('ready_name','warmstart-ready.json') if args.mode=='train' else 'smoke-adapter-ready.json'),{'checkpoint':str(checkpoint),'step':step+1,'final':step+1==steps,
                       'manifest':str(checkpoint/'checkpoint-manifest.json'),'revision':revision,
                       'selection':'fixed development-pilot snapshots, no core/capability selection'})
            del state,optim_state;dist.barrier()
    # Evaluate the same complete-target objective after the update. A change in
    # loss plus nonzero finite gradients verifies an actual update took place.
    wrapped.eval();row=schedule[rank]
    batch={k:v.to(device) for k,v in load_tensor(root/row['tensor_file']).items()}
    with torch.no_grad():post=target_loss(wrapped,batch)
    if not torch.isfinite(post):raise ValueError('Full-target post-update scorer failed')
    dist.all_reduce(post);post/=world
    if rank==0:
        write(directory/'complete.json',{'success':True,'optimizer_steps':steps,'checkpoints':checkpoints,
              'optimizer_updates_this_attempt':steps-start_step,'resumed_from_step':start_step,
              'post_update_full_target_ce':post.item(),'seconds':time.monotonic()-started,'metrics':metrics,
              'smoke_checkpoint_discarded':args.mode=='smoke','updated_utc':datetime.now(timezone.utc).isoformat()})
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
