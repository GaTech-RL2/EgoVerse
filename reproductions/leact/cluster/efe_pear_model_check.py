"""Discarded full-model PEAR engine check; never a primary RL checkpoint.

Exercises the actual FSDP2 model, vision input, one diagnostic NLL update and
fresh-process checkpoint reload. On-policy rollout/score-centering and the full
multitask curriculum are separate required checks, explicitly left unqualified.
"""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback


def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def record(root, phase, **extra):
    value={'phase':phase,'updated_utc':datetime.now(timezone.utc).isoformat(),
           'primary_rl_updates':0,**extra}
    write(root/'status.json',value)
    print('PEAR_MODEL_CHECK '+json.dumps(value),flush=True)


def rank_error_context(text):
    """Keep the original rank traceback instead of only torchrun's wrapper."""
    by_rank={}
    for line in text.splitlines():
        match=re.match(r'\[rank(\d+)\]:\s?(.*)',line)
        if match:by_rank.setdefault(match[1],[]).append(match[2])
    result={}
    for rank,lines in by_rank.items():
        starts=[i for i,line in enumerate(lines) if line.startswith('Traceback (')]
        if starts:
            excerpt='\n'.join(lines[starts[-1]:])[-5000:]
            result.setdefault(excerpt,[]).append(int(rank))
    return [{'ranks':ranks,'traceback':trace} for trace,ranks in list(result.items())[:4]]


def unused_checkpoint_auxiliaries(checkpoint_names, loaded_names, ignored_patterns):
    """Allow only MTP auxiliaries explicitly omitted by the installed model class.

    Qwen's autoregressive Transformers model declares ^mtp.* as unexpected
    checkpoint keys. Vision, language and expert weights must still all load.
    """
    missing=set(checkpoint_names)-set(loaded_names)
    allowed={name for name in missing if name.startswith('mtp.')
             and any(re.match(pattern,name) for pattern in ignored_patterns)}
    if missing-allowed:
        raise ValueError('Parent weights omitted: '+str(sorted(missing-allowed)[:10]))
    return sorted(allowed)


def make_engine(spec):
    import torch
    import torch.distributed as dist
    from pear.trainer.config import PearFSDPEngineConfig
    from pear.trainer.engine import PearFSDPEngine
    from verl.workers.config import HFModelConfig, FSDPOptimizerConfig
    from verl.trainer.config import CheckpointConfig
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl',timeout=timedelta(minutes=20))
    assert dist.get_world_size()==8
    # The dense model's GPU backend is not implemented for the MoE architecture.
    dense=spec['model_type']=='qwen3_5'
    model=HFModelConfig(path=spec['model'],trust_remote_code=True,lora_rank=0,
        use_remove_padding=True,use_fused_kernels=True,fused_kernel_options={'impl_backend':'torch'},
        use_liger=True,liger_kwargs={'rms_norm':False},enable_activation_offload=True,
        override_config={'attn_implementation':{'text_config':'flash_attention_3',
                                               'vision_config':'flash_attention_2'}})
    if model.hf_config.model_type!=spec['model_type']:raise ValueError('Architecture changed')
    config=PearFSDPEngineConfig(strategy='fsdp2',fsdp_size=8,ulysses_sequence_parallel_size=8,
        model_dtype='fp32',param_offload=False,optimizer_offload=True,optimizer_offload_step=True,
        mixed_precision={'param_dtype':'bf16','reduce_dtype':'fp32','buffer_dtype':'fp32',
            'fp32_modules':['model.visual'],'fp32_modules_tf32':True,'fp32_residual':True},
        kernels=('efe_pear_moe_kernels.FrozenRouterTrainingKernels'if spec.get('router_policy')=='frozen_expert_selection_router'
                 else 'pear.models.qwen3_5.trainer.training_kernels.Qwen3_5TrainingKernels'),
        use_gpu_backend=dense,use_torch_compile=False)
    optimizer=FSDPOptimizerConfig(lr=1e-6,total_training_steps=1,lr_warmup_steps=0,
        weight_decay=.01,betas=(.9,.999),clip_grad=1.,
        # The portable MoE path has no streamed AdamW backend. Avoid the
        # parameter-sized foreach temporary while retaining ordinary FP32 AdamW.
        override_optimizer_config=None if dense else {'foreach':False})
    engine=PearFSDPEngine(model,config,optimizer,CheckpointConfig())
    engine.initialize()
    if spec.get('router_policy')=='frozen_expert_selection_router':
        from efe_pear_router import ROUTER,assert_expert_routers_frozen
        assert_expert_routers_frozen(engine.module)
        if any(not p.requires_grad for name,p in engine.module.named_parameters()if not ROUTER.search(name)):
            raise ValueError('Unexpected frozen non-router parameters')
    elif any(not p.requires_grad for p in engine.module.parameters()):
        raise ValueError('Unexpected frozen parameters in full-finetuning model')
    return engine


def loaded_weight_check(engine, model):
    import torch
    import torch.distributed as dist
    from safetensors import safe_open
    from torch.distributed.tensor import DTensor
    index=json.loads((Path(model)/'model.safetensors.index.json').read_text())['weight_map']
    # get_per_tensor_param is the *rollout export*: it intentionally unfuses
    # packed MoE weights into individual experts. Audit the native FSDP state
    # against the native checkpoint, before any such export transformation.
    weights=engine.module.state_dict()
    seen=set();maximum=0.;count=0
    for name,shard in weights.items():
        value=shard.full_tensor() if isinstance(shard,DTensor) else shard
        if dist.get_rank()!=0:continue
        if name not in index:raise ValueError('Engine weight is absent from parent: '+name)
        with safe_open(str(Path(model)/index[name]),framework='pt',device='cpu') as stream:
            expected=stream.get_tensor(name)
        actual=value.detach().float().cpu()
        if actual.shape!=expected.shape:raise ValueError('Weight shape changed: '+name)
        error=float((actual-expected.float()).abs().max())
        if not math.isfinite(error) or error>5e-4:raise ValueError(f'Parent replay differs for {name}: {error}')
        maximum=max(maximum,error);count+=actual.numel();seen.add(name)
    ignored=[]
    if dist.get_rank()==0:
        ignored=unused_checkpoint_auxiliaries(index,seen,
            getattr(engine.module,'_keys_to_ignore_on_load_unexpected',()) or ())
    dist.barrier()
    return {'loaded_weight_max_error':maximum,'checked_parameters':count,'checked_tensors':len(seen),
            'unused_checkpoint_auxiliary_tensors':ignored,
            'audit_scope':'Every active model weight; only model-declared unused MTP auxiliaries are excluded.'}


def probe_batch(engine, spec, *, processor=None):
    import torch
    from tensordict import TensorDict
    from verl.utils import tensordict_utils as tu
    from efe_pear_bridge import observation_payload, image_blocks
    path=Path(spec['observation']['path'])
    if hashlib.sha256(path.read_bytes()).hexdigest()!=spec['observation']['sha256']:
        raise ValueError('Frozen public probe observation changed')
    _,content=observation_payload(json.loads(path.read_text()))
    blocks=image_blocks(content)
    response='```json\n{"action_type":"ee_delta","action":[0.0,0.0,0.0,0.0],"repeat":1,"summary":"Discarded numerical probe, never a policy demonstration."}\n```'
    messages=[{'role':'user','content':blocks},{'role':'assistant','content':response}]
    processor=processor or engine.model_config.get_processor()
    encoded=processor.apply_chat_template(messages,tokenize=True,return_dict=True,return_tensors='pt',
                                          add_generation_prompt=False,enable_thinking=False)
    ids=encoded.pop('input_ids');attention=encoded.pop('attention_mask')
    mm=dict(encoded)
    kwargs={'image_grid_thw':mm.get('image_grid_thw'),'video_grid_thw':mm.get('video_grid_thw')}
    types=mm.pop('mm_token_type_ids',None)
    if types is not None:kwargs['mm_token_type_ids']=types
    if hasattr(processor,'get_rope_index_kwargs'):kwargs.update(processor.get_rope_index_kwargs(mm))
    vision,_=processor.get_rope_index(input_ids=ids,attention_mask=attention,**kwargs)
    positions=torch.cat([torch.arange(ids.shape[-1])[None,None],vision.transpose(0,1)],dim=1)[0]
    length=ids.shape[-1]
    if length>8192:raise ValueError('Numerical probe unexpectedly exceeds 8192 tokens')
    offsets=torch.tensor([0,length],dtype=torch.int64)
    mask=torch.zeros(length);mask[-33:-1]=1
    def jagged(value,dim=1):return torch.nested.nested_tensor_from_jagged(value,offsets,jagged_dim=dim)
    batch=TensorDict({'input_ids':jagged(ids[0]),'position_ids':jagged(positions,2),
                     'loss_mask':jagged(mask)},batch_size=[1])
    tu.assign_non_tensor(batch,temperature=1.,use_remove_padding=True,use_fused_kernels=True,
                         score_centering=False,calculate_entropy=False)
    tu.assign_non_tensor_stack(batch,'multi_modal_inputs',[mm])
    return batch


def nll_loss(*, model_output, data, dp_group=None):
    mask=data['loss_mask'].values()
    loss=-(model_output['log_probs'].values()*mask).sum()/mask.sum()
    return loss,{}


def forward(engine, batch):
    import torch
    with engine.eval_mode(),torch.no_grad():
        run=lambda:engine.forward_step(batch,nll_loss,forward_only=True)
        _,out=engine.kernels.run_checked(run,lambda:None) if engine.use_gpu_backend else run()
    return out['model_output']['log_probs'].values().float().cpu()


def fingerprints(engine):
    import torch
    from torch.distributed.tensor import DTensor
    def add(h,name,value,other=None):
        def update(raw):
            h.update(raw)
            if other is not None:other.update(raw)
        update(name.encode())
        if isinstance(value,DTensor):value=value.to_local()
        if isinstance(value,torch.Tensor):
            value=value.detach();update(str(value.shape).encode()+str(value.dtype).encode())
            flat=value.reshape(-1)
            for i in range(0,flat.numel(),1<<20):
                update(flat[i:i+(1<<20)].contiguous().cpu().view(torch.uint8).numpy().tobytes())
        else:update(repr(value).encode())
    model=hashlib.sha256();optim=hashlib.sha256()
    groups={key:hashlib.sha256()for key in ['routers','experts','backbone']}
    for name,value in sorted(engine.module.state_dict().items()):
        group='routers'if re.search(r'(?:^|\.)mlp\.gate\.(?:weight|bias)$',name)else'experts'if'.experts.'in name else'backbone'
        add(model,name,value,groups[group])
    for i,group in enumerate(engine.optimizer.param_groups):
        for key,val in sorted(group.items()):
            if key!='params':add(optim,f'{i}/{key}',val)
        for j,param in enumerate(group['params']):
            for key,val in sorted(engine.optimizer.state.get(param,{}).items()):add(optim,f'{i}/{j}/{key}',val)
    return {'model':model.hexdigest(),'optimizer':optim.hexdigest(),**{key:value.hexdigest()for key,value in groups.items()}}


def checkpoint_directory(spec,root):
    return Path(spec.get('checkpoint_local_root',root))/os.environ['EFE_WORKFLOW_ID']/'rp_jhh_discarded_engine_step1'


def distributed(spec, phase, root):
    import torch
    import torch.distributed as dist
    rank=int(os.environ['RANK'])
    engine=make_engine(spec)
    batch=probe_batch(engine,spec)
    checkpoint=checkpoint_directory(spec,root)
    if phase=='update':
        if rank==0:record(root,'checking_full_parent_weights')
        replay=loaded_weight_check(engine,spec['model'])
        before=forward(engine,batch);repeat=forward(engine,batch)
        repeat_error=float((before-repeat).abs().max())
        if not torch.isfinite(before).all() or repeat_error>5e-4:
            raise ValueError(f'Unstable full-model native-image forward: {repeat_error}')
        initial=fingerprints(engine)
        if rank==0:record(root,'discarded_nll_optimizer_check',replay=replay)
        with engine.train_mode():
            engine.optimizer_zero_grad()
            def backward():
                loss,_=engine.forward_step(batch,nll_loss,forward_only=False)
                loss.backward()
            if engine.use_gpu_backend:engine.kernels.run_checked(backward,engine.optimizer_zero_grad)
            else:backward()
            if spec.get('router_policy')=='frozen_expert_selection_router':
                from efe_pear_router import ROUTER
                if any(p.grad is not None for name,p in engine.module.named_parameters()if ROUTER.search(name)):
                    raise ValueError('A frozen router received gradients')
            grad_norm=engine.optimizer_step()
            engine.lr_scheduler_step()
        if not math.isfinite(grad_norm) or grad_norm<=0:raise ValueError('Invalid full-FT gradient')
        after=forward(engine,batch);state=fingerprints(engine)
        if initial['model']==state['model']:raise ValueError('Diagnostic update did not change the model')
        if spec.get('router_policy')=='frozen_expert_selection_router':
            if initial['routers']!=state['routers']:raise ValueError('Router parameters changed')
            if any(initial[key]==state[key]for key in ['experts','backbone']):
                raise ValueError('Expert and backbone parameters must both learn')
        engine.save_checkpoint(str(checkpoint),global_step=1)
        torch.save(after,root/f'expected-logprobs-rank{rank}.pt')
        write(root/f'expected-state-rank{rank}.json',state)
        if rank==0:write(root/'update.json',{'parameter_update':'full_finetuning','lora_rank':0,
            'diagnostic_optimizer_updates':1,'primary_rl_updates':0,'gradient_norm':grad_norm,
            'repeated_forward_max_error':repeat_error,'model_revision':spec['model_revision'],
            'router_policy':spec.get('router_policy'),'router_gradients_absent':spec.get('router_policy')=='frozen_expert_selection_router',
            'router_unchanged_after_update':initial['routers']==state['routers'],
            'expert_parameters_changed':initial['experts']!=state['experts'],
            'backbone_parameters_changed':initial['backbone']!=state['backbone'],
            'loss':'fixed-target NLL numerical test; not on-policy PEAR training',**replay})
    else:
        engine.load_checkpoint(str(checkpoint),del_local_after_load=False)
        expected=json.loads((root/f'expected-state-rank{rank}.json').read_text())
        restored=fingerprints(engine)
        if restored!=expected:raise ValueError('Model or optimizer differs after fresh-process reload')
        expected_logp=torch.load(root/f'expected-logprobs-rank{rank}.pt',weights_only=True)
        actual=forward(engine,batch)
        error=float((actual-expected_logp).abs().max())
        if not torch.isfinite(actual).all() or error>5e-4:raise ValueError(f'Reloaded forward differs: {error}')
        all_errors=[None]*8;dist.all_gather_object(all_errors,error)
        if rank==0:
            value={**json.loads((root/'update.json').read_text()),'engine_check_passed':True,
                   'checkpoint_reload_max_error':max(all_errors),'checkpoint_reload_passed':True,
                   'primary_training_qualified':False,'on_policy_score_centering_validated':False,
                   'router_unchanged_after_update_and_reload':expected['routers']==restored['routers'],
                   'scope':'Full-model weights, native-image forward, discarded NLL update and fresh-process restart only.'}
            write(root/'result.json',value);record(root,'model_engine_check_passed',**value)
    dist.barrier();dist.destroy_process_group()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['preflight','update','verify'])
    args=parser.parse_args();lane=Path(__file__).resolve().parent
    spec=json.loads((lane/'spec.json').read_text());root=Path(spec['output'])/os.environ['EFE_WORKFLOW_ID']
    root.mkdir(parents=True,exist_ok=True)
    if args.phase=='preflight':
        from verl.workers.config import HFModelConfig
        if spec.get('model_repository'):
            from huggingface_hub import hf_hub_download
            # The original snapshot glob omitted the official Jinja template.
            # Fetch only that pinned auxiliary file; parent weights stay intact.
            hf_hub_download(spec['model_repository'],'chat_template.jinja',
                            revision=spec['model_revision'],local_dir=spec['model'])
        processor=HFModelConfig(path=spec['model'],trust_remote_code=True).get_processor()
        batch=probe_batch(None,spec,processor=processor)
        record(root,'processor_preflight_passed',probe_tokens=int(batch['input_ids'].values().numel()))
        return
    if args.phase:
        distributed(spec,args.phase,root);return
    try:
        if spec.get('checkpoint_local_root'):
            local=Path(spec['checkpoint_local_root']);local.mkdir(parents=True,exist_ok=True)
            if shutil.disk_usage(local).free<spec.get('minimum_checkpoint_free_bytes',500_000_000_000):
                raise RuntimeError('Insufficient local scratch for a full-model optimizer checkpoint')
        if not Path(spec['remote_root'],'runtime-ready.json').exists():
            raise RuntimeError('Pinned shared runtime is not ready')
        python=Path(spec['remote_root'],'runtime/envs/verl/bin/python')
        source=Path(spec['pear_source'])
        env={**os.environ,'NCCL_ALGO':'Ring','NCCL_PROTO':'Simple','NCCL_P2P_DISABLE':'1',
             'TOKENIZERS_PARALLELISM':'false','PYTHONUNBUFFERED':'1',
             'PYTHONPATH':str(lane)+':'+str(source)+':'+str(source/'3rdparty/verl/src'),
             'OMP_NUM_THREADS':'4'}
        for phase in ['preflight','update','verify']:
            record(root,'engine_'+phase,model=spec['model'],model_revision=spec['model_revision'])
            with (root/(phase+'.log')).open('a') as log:
                command=([str(python)] if phase=='preflight' else
                    [str(python),'-m','torch.distributed.run','--standalone','--nproc_per_node=8'])
                subprocess.run([*command,str(Path(__file__).resolve()),'--phase',phase],env=env,cwd=source,
                    stdout=log,stderr=subprocess.STDOUT,check=True,timeout=14400)
        # These are explicitly discarded diagnostic updates, never run parents.
        shutil.rmtree(checkpoint_directory(spec,root))
        record(root,'model_engine_check_passed',result=json.loads((root/'result.json').read_text()))
    except Exception as exc:
        logs={p.name:p.read_text(errors='replace') for p in root.glob('*.log')}
        record(root,'model_engine_check_failed',error=repr(exc),traceback=traceback.format_exc(),
               log_tails={k:v[-3500:] for k,v in logs.items()},
               rank_errors={k:rank_error_context(v) for k,v in logs.items()})
        raise


if __name__=='__main__':main()
