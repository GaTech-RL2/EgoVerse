"""One real PEAR update and fresh-process exact restart check on eight H100s.

All targets from one admitted native group enter the generated-token mean.
The resulting base-policy checkpoint is discarded from primary comparisons.
"""
import argparse
from datetime import datetime, timedelta, timezone
from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

from efe_training_data import sha, write
from efe_pear_update_data import schedule_rows


def full_loss(wrapped, batch, centered):
    import torch
    from torch.utils.checkpoint import checkpoint
    from trainer import modality_inputs
    from efe_pear_loss import token_losses
    labels=batch['labels'][0]
    positions=torch.nonzero(labels!=-100,as_tuple=False).flatten()
    if not positions.numel() or positions.min()<=0:
        raise ValueError('Empty target or missing causal context')
    extras={k:v for k,v in batch.items() if k.startswith('pear_')}
    if positions.numel()!=extras['pear_logq'].numel():
        raise ValueError('Target positions and original sampler distribution differ')
    inputs=modality_inputs({k:v for k,v in batch.items() if k!='labels' and not k.startswith('pear_')},wrapped.config)
    logits=wrapped(**inputs,logits_to_keep=positions-1,use_cache=False).logits[0]
    if logits.shape[0]!=positions.numel():
        raise ValueError('Full-target causal logits misaligned')
    def chunk(raw, ids, head_ids, head_q, sampled_q, advantage):
        logp=raw.float().log_softmax(-1)
        loss=token_losses(logp,ids,advantage,score_centering=centered,topk_ids=head_ids,topk_logq=head_q).sum()
        selected=logp.gather(-1,ids[:,None])[:,0]
        # Diagnostic tensors are detached and do not change the policy gradient.
        stats=torch.stack((-selected.sum(),-sampled_q.sum(),(selected-sampled_q).abs().sum())).detach()
        return loss,stats
    loss=logits.new_zeros((),dtype=torch.float32)
    stats=logits.new_zeros(3,dtype=torch.float32)
    for i in range(0,positions.numel(),128):
        args=(logits[i:i+128],labels[positions[i:i+128]],extras['pear_topk_ids'][i:i+128],
              extras['pear_topk_logq'][i:i+128],extras['pear_logq'][i:i+128],extras['pear_advantage'])
        # Do not retain a full FP32 vocabulary softmax for every target token.
        value,diagnostics=checkpoint(chunk,*args,use_reentrant=False) if torch.is_grad_enabled() else chunk(*args)
        loss=loss+value;stats=stats+diagnostics
    return loss,stats


def load_example(root,row,device):
    from trainer import load_tensor
    path=root/row['tensor_file']
    if sha(path)!=row['tensor_sha256']:
        raise ValueError('Frozen training tensor checksum differs')
    return {k:v.to(device) for k,v in load_tensor(path).items()}


def model_setup(spec,root):
    import torch
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from transformers import AutoConfig, AutoModelForImageTextToText
    from peft import LoraConfig,get_peft_model
    from efe_qwen36_adapter_io import parameter_shapes,load_into_exact
    rank,local=int(os.environ['RANK']),int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local);device=torch.device('cuda',local)
    dist.init_process_group('nccl',timeout=timedelta(seconds=600))
    if dist.get_world_size()!=8:
        raise ValueError('Qualification requires the validated eight-GPU layout')
    probe=torch.tensor(float(rank+1),device=device);dist.all_reduce(probe)
    if probe.item()!=36:
        raise ValueError('NCCL all-reduce failed')
    torch.manual_seed(spec['seed']);random.seed(spec['seed'])
    path=spec['local_base_model']
    if rank==0:
        model=AutoModelForImageTextToText.from_pretrained(path,dtype=torch.float32,attn_implementation='sdpa',local_files_only=True)
    else:
        with torch.device('meta'):
            model=AutoModelForImageTextToText.from_config(AutoConfig.from_pretrained(path,local_files_only=True))
        model.config._attn_implementation='sdpa'
    model.to(dtype=torch.float32)
    if model.config.model_type!='qwen3_5_moe' or any(p.dtype!=torch.float32 for p in model.parameters()):
        raise ValueError('Architecture or FP32 master-weight contract changed')
    targets=[n for n,m in model.named_modules() if '.language_model.' in n and isinstance(m,torch.nn.Linear)]
    model=get_peft_model(model,LoraConfig(r=32,lora_alpha=64,lora_dropout=0.,bias='none',target_modules=targets,task_type='CAUSAL_LM'))
    shapes=parameter_shapes(model)
    if len(shapes)!=700 or sum(math.prod(s) for s in shapes.values())!=44954880:
        raise ValueError('Language LoRA parameter contract changed')
    model._efe_adapter_shapes=shapes
    if spec['active_restore']['step'] and rank==0:
        load_into_exact(model,spec['active_restore']['path'],root/'adapter-resume-verification.json')
    decoder_types={type(m) for m in model.modules() if type(m).__name__.endswith('DecoderLayer')}
    if not decoder_types:raise ValueError('No transformer layers available for sharding')
    model.config.use_cache=False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    wrapped=FSDP(model,auto_wrap_policy=partial(transformer_auto_wrap_policy,transformer_layer_cls=decoder_types),
        sharding_strategy=ShardingStrategy.FULL_SHARD,use_orig_params=True,
        mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,reduce_dtype=torch.float32,buffer_dtype=torch.float32),
        param_init_fn=lambda module:module.to_empty(device=device,recurse=False),device_id=device,
        sync_module_states=True,limit_all_gathers=True)
    optimizer=torch.optim.AdamW([p for p in wrapped.parameters() if p.requires_grad],
        lr=spec['lr'],betas=tuple(spec['betas']),weight_decay=spec['weight_decay'],foreach=False)
    return model,wrapped,optimizer,device


def fingerprint(optimizer):
    import torch
    def digest(values):
        h=hashlib.sha256()
        for key,value in values:
            h.update(key.encode()+b'\0')
            if isinstance(value,torch.Tensor):
                h.update(str(value.dtype).encode()+str(tuple(value.shape)).encode())
                h.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
            else:h.update(repr(value).encode())
        return h.hexdigest()
    states=[]
    for gi,group in enumerate(optimizer.param_groups):
        states.extend((f'group/{gi}/{k}',v) for k,v in sorted(group.items()) if k!='params')
        for pi,p in enumerate(group['params']):
            states.extend((f'{gi}/{pi}/{k}',v) for k,v in sorted(optimizer.state.get(p,{}).items()))
    return {'adam':digest(states),'rng':digest([('cpu',torch.get_rng_state()),
        ('cuda',torch.cuda.get_rng_state()),('python',random.getstate())])}


def gather(value):
    import torch.distributed as dist
    from efe_training_state import object_group
    values=[None]*dist.get_world_size();dist.all_gather_object(values,value,group=object_group());return values


def native_probe(wrapped,root,rows,device):
    import torch
    import torch.distributed as dist
    saved=(torch.get_rng_state(),torch.cuda.get_rng_state(),random.getstate(),wrapped.training)
    try:
        # Short native calls exercise the real image/action path in every rank.
        row=sorted(rows,key=lambda r:(r['tokens'],r['id']))[dist.get_rank()]
        batch=load_example(root,row,device);wrapped.eval()
        with torch.no_grad():_,stats=full_loss(wrapped,batch,True)
        return gather({'id':row['id'],'ce':float(stats[0]/row['generated_tokens'])})
    finally:
        torch.set_rng_state(saved[0]);torch.cuda.set_rng_state(saved[1]);random.setstate(saved[2]);wrapped.train(saved[3])


def check_base(wrapped,spec,root):
    import torch.distributed as dist
    from efe_qwen36_health import measure
    reference=Path(spec['base_health_reference'])
    metric=json.loads((reference/'smoke/metrics.jsonl').read_text().splitlines()[0])
    expected=metric['forward_ce'] if metric['forward_examples'] else metric['backward_ce']
    lookup={r['id']:r for r in json.loads((reference/'dataset-manifest.json').read_text())['examples']}
    rows=[lookup[k] for k in json.loads((reference/'smoke/schedule.json').read_text())['exposures'][:8]]
    actual,values=measure(wrapped,reference,rows)
    result={'expected_ce':expected,'actual_ce':actual,'per_rank_ce':values,'absolute_error':abs(actual-expected),
        'passed':abs(actual-expected)<=.0005,'reference_directory':str(reference),'reference_is_read_only':True}
    if dist.get_rank()==0:write(root/'untouched-base-replay.json',result)
    if not result['passed']:raise ValueError('Untouched full-model likelihood replay failed')


def run(spec,root,mode):
    import torch
    import torch.distributed as dist
    import efe_qwen36_restart as restart
    spec['active_restore']=restart.choose_resume(spec)
    if (mode=='verify') != (spec['active_restore']['step']==1):
        raise ValueError('Only one qualification update is allowed; restore state and mode differ')
    model,wrapped,optimizer,device=model_setup(spec,root)
    rank=dist.get_rank()
    if sha(root/'dataset-manifest.json')!=spec['prepared_manifest_sha256']:
        raise ValueError('Prepared training manifest changed')
    manifest=json.loads((root/'dataset-manifest.json').read_text())
    rows=manifest['examples'];schedule,total=schedule_rows(rows)
    if total!=spec['input']['generated_tokens']:raise ValueError('Generated-token denominator changed')
    if mode=='verify':
        restart.restore_optimizer(wrapped,optimizer,spec,root)
        restored=gather(fingerprint(optimizer))
        expected=json.loads((root/'post-update-reference.json').read_text())
        if restored!=expected['state_fingerprints']:
            raise ValueError('Fresh-process Adam or per-rank RNG state differs')
        actual=native_probe(wrapped,root,rows,device)
        errors=[abs(x['ce']-y['ce']) if x['id']==y['id'] else float('inf') for x,y in zip(actual,expected['native_probe'])]
        if not all(math.isfinite(x) and x<=.0005 for x in errors):
            raise ValueError('Fresh-process native likelihood replay failed')
        if rank==0:
            write(root/'result.json',{'success':True,'optimizer_updates':1,'generated_tokens':total,
                'native_episodes':8,'original_successes':1,'fresh_process_replay_max_error':max(errors),
                'exact_adam_and_rng_restored':True,'adapter_checkpoint':spec['active_restore']['path'],
                'scope':spec['scope'],'primary_rl_updates':0,'completed_utc':datetime.now(timezone.utc).isoformat()})
    else:
        check_base(wrapped,spec,root)
        wrapped.train();optimizer.zero_grad(set_to_none=True)
        stats_total=torch.zeros(5,device=device,dtype=torch.float64);start=time.monotonic()
        for index in range(0,len(schedule),8):
            row=schedule[index+rank];batch=load_example(root,row,device)
            loss,stats=full_loss(wrapped,batch,spec['score_centering'])
            # FSDP averages rank gradients. Undo that average before dividing
            # by ALL actual generated tokens, never by episodes or call length.
            scaled=loss*(row['weight']*8/total)
            if not torch.isfinite(scaled):raise ValueError('Nonfinite policy-gradient objective')
            scaled.backward()
            stats_total+=torch.cat([loss.detach().reshape(1).double(),stats.double(),
                torch.tensor([row['generated_tokens']],device=device,dtype=torch.float64)])*row['weight']
            del batch,loss,stats,scaled
            seen=index//8+1
            if rank==0:
                value={'phase':'accumulating_policy_gradient','microbatches':seen,'total_microbatches':len(schedule)//8,
                    'optimizer_updates':0,'seconds':time.monotonic()-start,'peak_memory_gib':torch.cuda.max_memory_allocated()/2**30,
                    'utc':datetime.now(timezone.utc).isoformat()}
                write(root/'update-progress.json',value);print('PEAR_UPDATE_PROGRESS '+json.dumps(value),flush=True)
        dist.all_reduce(stats_total)
        if int(stats_total[-1])!=total:raise ValueError('Generated-token reduction lost or duplicated targets')
        norm=wrapped.clip_grad_norm_(spec['grad_clip'])
        if not torch.isfinite(norm) or norm.item()<=0:raise ValueError('Nonfinite or empty policy gradient')
        before=[p.detach().clone() for g in optimizer.param_groups for p in g['params']]
        optimizer.step()
        changed=sum(int((p.detach()!=old).sum()) for p,old in zip((p for g in optimizer.param_groups for p in g['params']),before))
        del before
        change=torch.tensor(changed,device=device,dtype=torch.long);dist.all_reduce(change)
        if change.item()==0:raise ValueError('Optimizer failed to change any adapter parameter')
        optimizer.zero_grad(set_to_none=True)
        metrics={'optimizer_updates':1,'generated_tokens':total,'requests':len(rows),
            'loss':float(stats_total[0]/total),'trainer_nll':float(stats_total[1]/total),
            'sampler_nll':float(stats_total[2]/total),'mean_abs_logprob_gap':float(stats_total[3]/total),
            'gradient_norm':float(norm),'changed_adapter_values':change.item(),
            'update_seconds':time.monotonic()-start,'score_centering':spec['score_centering']}
        probes=native_probe(wrapped,root,rows,device)
        fingerprints=gather(fingerprint(optimizer))
        if rank==0:
            write(root/'post-update-reference.json',{'native_probe':probes,'state_fingerprints':fingerprints,'metrics':metrics})
        restart.save_state(wrapped,optimizer,model,spec,root,1,1,metrics,True)
        after=native_probe(wrapped,root,rows,device)
        if any(abs(a['ce']-b['ce'])>.0005 for a,b in zip(probes,after)):
            raise ValueError('Full-model likelihood changed during checkpoint gathering')
        if rank==0:
            write(root/'update-complete.json',metrics);print('PEAR_UPDATE_COMPLETE '+json.dumps(metrics),flush=True)
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--mode',choices=['update','verify'],required=True);a=p.parse_args()
    run(json.loads(a.spec.read_text()),a.root,a.mode)
