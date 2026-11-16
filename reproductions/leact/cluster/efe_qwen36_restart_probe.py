"""Compare uninterrupted versus save/reload Adam updates on allocated GPUs."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path


def main(root, empty_shards=False, adapter_gather=False, nested=False):
    import torch
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
    from peft import LoraConfig, get_peft_model
    import efe_qwen36_restart as restart
    from efe_qwen36_adapter_io import parameter_shapes, load_into_exact
    from efe_training_data import write
    from efe_training_state import object_group
    rank = int(os.environ['RANK'])
    local = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local)
    dist.init_process_group('nccl')
    root.mkdir(parents=True,exist_ok=True)
    (root/'restart-states').mkdir(exist_ok=True)
    class Block(torch.nn.Module):
        def __init__(self):super().__init__();self.proj=torch.nn.Linear(8,8)
        def forward(self,x):return self.proj(x)
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            if empty_shards:
                # This frozen bank dominates the flattened FSDP parameter.
                # The tiny trainable adapters then live only on the last rank.
                self.frozen = torch.nn.Linear(128,128,bias=False)
            self.first = Block() if nested else torch.nn.Linear(8,8)
            self.second = Block() if nested else torch.nn.Linear(8,8)
        def forward(self,x):
            if empty_shards:x=self.frozen(x)[:,:8]
            return self.second(self.first(x).tanh())
    def make(restore=None):
        torch.manual_seed(749)
        model = get_peft_model(Tiny(),LoraConfig(r=2,lora_alpha=4,target_modules=['proj'] if nested else ['first','second']))
        model._efe_adapter_shapes = parameter_shapes(model)
        if restore and rank==0:
            load_into_exact(model,restore)
        policy = None
        if nested:
            from functools import partial
            from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
            policy = partial(transformer_auto_wrap_policy,transformer_layer_cls={Block})
        wrapped = FSDP(model,use_orig_params=True,device_id=local,sync_module_states=True,auto_wrap_policy=policy)
        optimizer = torch.optim.AdamW([p for p in wrapped.parameters() if p.requires_grad],lr=.01,foreach=False)
        return model, wrapped, optimizer
    def update(wrapped,optimizer):
        optimizer.zero_grad()
        x = torch.randn(3,128 if empty_shards else 8,device='cuda')
        loss = wrapped(x).square().mean()
        loss.backward()
        optimizer.step()
        return loss.detach().cpu()
    def weights(wrapped):
        with FSDP.state_dict_type(wrapped,StateDictType.FULL_STATE_DICT,FullStateDictConfig(offload_to_cpu=True,rank0_only=True)):
            state = wrapped.state_dict()
        return {k:v.clone() for k,v in state.items() if '.lora_' in k} if rank==0 else {}
    spec = {'base_model':'immutable-tiny-probe','checkpoint_revision_prefix':'qwen36-restart-probe',
        'checkpoint_gather':'adapter-shards-v1' if adapter_gather else 'full-state-v1',
        'validate_final_probe':False,'seed':749,'durable_restart':{
            'state_directory':str(root/'restart-states'),'lineage_sha256':hashlib.sha256(b'probe-empty-shards-v1' if empty_shards else b'probe-v1').hexdigest(),
            'target_step':4,'restart_interval':2}}
    model, wrapped, optimizer = make()
    update(wrapped,optimizer)
    update(wrapped,optimizer)
    qualification = None
    if adapter_gather:
        state,qualification = restart.qualify_adapter_gather(wrapped,model._efe_adapter_shapes)
        del state
    restart.save_state(wrapped,optimizer,model,spec,root,2,4,{},False)
    expected_losses = [update(wrapped,optimizer), update(wrapped,optimizer)]
    expected = weights(wrapped)
    del wrapped,model,optimizer
    gc.collect()
    torch.cuda.empty_cache()
    spec['active_restore'] = restart.choose_resume(spec)
    model,wrapped,optimizer = make(spec['active_restore']['path'])
    restart.restore_optimizer(wrapped,optimizer,spec,root)
    state_counts=[None]*dist.get_world_size()
    dist.all_gather_object(state_counts,len(optimizer.state),group=object_group())
    if empty_shards and not (0 in state_counts and max(state_counts)>0):
        raise ValueError('Probe did not exercise both empty and nonempty optimizer shards')
    actual_losses = [update(wrapped,optimizer), update(wrapped,optimizer)]
    actual = weights(wrapped)
    loss_error = max((x-y).abs().item() for x,y in zip(actual_losses,expected_losses))
    error = torch.tensor(loss_error,device='cuda')
    dist.all_reduce(error,op=dist.ReduceOp.MAX)
    passed = error.item()<=1e-7
    if rank==0:
        weight_error = max((expected[k]-actual[k]).abs().max().item() for k in expected)
        passed = passed and weight_error<=1e-7
        write(root/'result.json',{'success':passed,'adapter_tensors':len(expected),
            'checkpoint_gather':spec['checkpoint_gather'],'nested_fsdp':nested,'gather_qualification':qualification,
            'maximum_weight_error':weight_error,'maximum_loss_error':error.item(),
            'empty_shards_tested':empty_shards,'optimizer_states_per_rank':state_counts,
            'metadata_backend':'gloo',
            'restored_step':2,'final_step':4,'world_size':dist.get_world_size(),
            'test':'Two additional random-input Adam updates match uninterrupted execution after exact adapter, Adam and per-rank RNG restoration.'})
    decision = [passed if rank==0 else None]
    dist.broadcast_object_list(decision,src=0,group=object_group())
    if not decision[0]:
        raise ValueError('Restart continuation differs from uninterrupted training')
    dist.barrier()
    dist.destroy_process_group()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--empty-shards',action='store_true')
    parser.add_argument('--adapter-gather',action='store_true')
    parser.add_argument('--nested',action='store_true')
    args=parser.parse_args()
    main(args.root,args.empty_shards,args.adapter_gather,args.nested)
