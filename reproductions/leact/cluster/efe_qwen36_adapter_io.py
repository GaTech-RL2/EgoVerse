"""Save canonical full LoRA tensors independently of FSDP module wrappers."""
import argparse
import json
from pathlib import Path

from efe_training_data import sha, write


def canonical_name(name):
    for component in ('A', 'B'):
        suffix=f'.lora_{component}.default.weight'
        if name.endswith(suffix):return name[:-len(suffix)]+f'.lora_{component}.weight'
    raise ValueError('Unexpected trainable adapter parameter: '+name)


def parameter_shapes(model):
    return {name:list(p.shape) for name,p in model.named_parameters() if p.requires_grad}


def save_exact(model, state, expected, checkpoint):
    """Select by the pre-FSDP parameter contract, then verify serialized values."""
    import torch
    from safetensors.torch import save_file,load_file
    checkpoint=Path(checkpoint);checkpoint.mkdir(exist_ok=False)
    actual={k for k in state if '.lora_' in k}
    audit={'expected_tensors':len(expected),'gathered_adapter_tensors':len(actual),
        'missing_keys':sorted(set(expected)-actual),'unexpected_keys':sorted(actual-set(expected)),
        'pre_fsdp_keys':list(expected),'full_state_keys':sorted(actual)}
    write(checkpoint/'serialization-audit.json',audit)
    if not expected or set(expected)!=actual:raise ValueError('Full FSDP adapter state does not match the pre-wrap contract')
    saved={}
    for name,shape in expected.items():
        value=state[name].detach().cpu().contiguous()
        if list(value.shape)!=shape or value.dtype!=torch.float32 or not torch.isfinite(value).all():
            raise ValueError('Incomplete, nonfinite, or non-FP32 full adapter tensor: '+name)
        key=canonical_name(name)
        if key in saved:raise ValueError('Duplicate canonical adapter key')
        saved[key]=value
    nonzero_b=sum(bool(t.count_nonzero()) for k,t in saved.items() if '.lora_B.' in k)
    if nonzero_b==0:raise ValueError('Adapter has no learned update')
    # PEFT 0.21 derives keys from live module paths. Nested FSDP adds wrapper
    # components to those paths, while FULL_STATE_DICT removes them. Saving
    # the canonical gathered tensors directly avoids that incompatible filter.
    destination=checkpoint/'adapter_model.safetensors'
    temporary=checkpoint/'adapter_model.safetensors.tmp'
    save_file(saved,str(temporary),metadata={'format':'pt'})
    restored=load_file(str(temporary),device='cpu')
    if set(restored)!=set(saved) or any(not torch.equal(restored[k],v) for k,v in saved.items()):
        raise ValueError('Serialized adapter failed the exact tensor round trip')
    temporary.replace(destination)
    model.peft_config['default'].save_pretrained(checkpoint)
    # Independently serialized full-precision adapter backup, not optimizer
    # moments, permits recovery even if a later format conversion fails.
    torch.save(saved,checkpoint/'adapter-backup.pt')
    audit.update(success=True,saved_tensors=len(saved),saved_parameters=sum(v.numel() for v in saved.values()),
        nonzero_B_tensors=nonzero_b,all_tensors_round_trip_exactly=True,
        safetensors_bytes=destination.stat().st_size,safetensors_sha256=sha(destination),
        backup_sha256=sha(checkpoint/'adapter-backup.pt'))
    write(checkpoint/'serialization-audit.json',audit)
    return audit


def load_into_exact(model,checkpoint,report=None):
    import torch
    from safetensors.torch import load_file
    parameters={canonical_name(n):p for n,p in model.named_parameters()
        if n.endswith(('.lora_A.default.weight','.lora_B.default.weight'))}
    state=load_file(str(Path(checkpoint)/'adapter_model.safetensors'),device='cpu')
    audit={'saved_tensors':len(state),'expected_tensors':len(parameters),
        'missing_keys':sorted(set(parameters)-set(state)),'unexpected_keys':sorted(set(state)-set(parameters))}
    if report:write(report,audit)
    if not parameters or set(parameters)!=set(state):raise ValueError('Adapter key coverage is not exact')
    with torch.no_grad():
        for name,value in state.items():
            parameter=parameters[name]
            if parameter.shape!=value.shape or parameter.dtype!=value.dtype:raise ValueError('Adapter shape/dtype mismatch: '+name)
            if not torch.isfinite(value).all():raise ValueError('Nonfinite adapter: '+name)
            parameter.copy_(value.to(parameter.device))
            if not torch.equal(parameter.detach().cpu(),value):raise ValueError('Adapter reload differs: '+name)
    audit.update(all_tensors_restored_exactly=True,nonzero_B_tensors=sum(bool(v.count_nonzero()) for k,v in state.items() if '.lora_B.' in k))
    if not audit['nonzero_B_tensors']:raise ValueError('Loaded adapter has no learned update')
    if report:write(report,audit)
    return model


def attach_exact(model,checkpoint,report=None):
    from peft import LoraConfig,get_peft_model
    config=LoraConfig.from_pretrained(checkpoint);config.inference_mode=False
    return load_into_exact(get_peft_model(model,config),checkpoint,report)


def probe(root):
    """Reproduce the wrapper mismatch and verify a fresh model before training."""
    import os
    from functools import partial
    import torch
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP,StateDictType,FullStateDictConfig
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from peft import LoraConfig,get_peft_model,get_peft_model_state_dict
    class Block(torch.nn.Module):
        def __init__(self):super().__init__();self.proj=torch.nn.Linear(8,8)
        def forward(self,x):return self.proj(x).tanh()
    class Tiny(torch.nn.Module):
        def __init__(self):super().__init__();self.first=Block();self.second=Block()
        def forward(self,x):return self.second(self.first(x))
    rank=int(os.environ['RANK']);torch.cuda.set_device(rank);dist.init_process_group('nccl')
    torch.manual_seed(341);base=Tiny();base_state={k:v.clone() for k,v in base.state_dict().items()}
    model=get_peft_model(base,LoraConfig(r=2,lora_alpha=4,target_modules=['proj']))
    expected=parameter_shapes(model)
    wrapped=FSDP(model,auto_wrap_policy=partial(transformer_auto_wrap_policy,transformer_layer_cls={Block}),
        use_orig_params=True,device_id=rank,sync_module_states=True)
    optimizer=torch.optim.AdamW([p for p in wrapped.parameters() if p.requires_grad],lr=.01)
    x=torch.ones(3,8,device=torch.device('cuda',rank))
    for _ in range(2):
        optimizer.zero_grad();wrapped(x).square().mean().backward();optimizer.step()
    wrapped.eval()
    with torch.no_grad():original=wrapped(x).cpu()
    with FSDP.state_dict_type(wrapped,StateDictType.FULL_STATE_DICT,FullStateDictConfig(offload_to_cpu=True,rank0_only=True)):
        state=wrapped.state_dict()
    if rank==0:
        old=get_peft_model_state_dict(model,state_dict=state)
        audit=save_exact(model,state,expected,root/'probe-adapter')
        fresh=Tiny();fresh.load_state_dict(base_state);fresh=attach_exact(fresh,root/'probe-adapter')
        fresh.eval()
        with torch.no_grad():restored=fresh(torch.ones(3,8))
        error=(original-restored).abs().max().item()
        if error>1e-6:raise ValueError('Fresh-model probe prediction mismatch')
        write(root/'serialization-probe.json',{'success':True,'legacy_saved_tensors':len(old),
            'new_saved_tensors':audit['saved_tensors'],'fresh_model_max_error':error,
            'nested_fsdp_wrappers':True,'optimizer_updates':2,'probe_model':'two 8x8 linear layers; discarded'})
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    probe(parser.parse_args().root)
