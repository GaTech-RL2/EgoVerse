"""Atomic FP32 LoRA + Adam + per-rank RNG snapshots over an immutable base."""
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

from efe_training_data import sha, write
from efe_qwen36_adapter_io import save_exact
from efe_training_state import object_group

FORMAT = 'efe-fp32-lora-adam-rng-v1'


def adapter_shard_layout(expected, metadata):
    """Validate complete, nonreplicated coverage before gathering any tensors."""
    names = sorted(expected)
    contract = hashlib.sha256(json.dumps([(n, expected[n]) for n in names]).encode()).hexdigest()
    if not names or any(not shape or any(type(s) is not int or s <= 0 for s in shape)
                        for shape in expected.values()):
        raise ValueError('Invalid pre-FSDP adapter shapes')
    for rank, row in enumerate(metadata):
        if row.get('error') or row.get('contract') != contract:
            raise ValueError(f'Adapter shard contract differs on rank {rank}: {row.get("error")}')
        lengths = row['lengths']
        if len(lengths) != len(names) or any(type(n) is not int or n < 0 for n in lengths):
            raise ValueError('Invalid adapter shard lengths')
    if not metadata:
        raise ValueError('No adapter shards')
    for i, name in enumerate(names):
        if sum(row['lengths'][i] for row in metadata) != math.prod(expected[name]):
            raise ValueError('Incomplete or replicated adapter shards: ' + name)
    return names, [sum(row['lengths']) for row in metadata]


def gather_adapter_state(wrapped, expected):
    """Gather only FP32 original-parameter shards; never unshard frozen weights.

    FULL_SHARD/use_orig_params exposes each original parameter's contiguous
    slice (possibly empty), without FlatParameter alignment padding. In the
    default full-world process group, rank order is original element order.
    Reject other layouts instead of guessing how to reconstruct them.
    """
    import torch
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
    rank, world = dist.get_rank(), dist.get_world_size()
    names = sorted(expected)
    local = {'contract': hashlib.sha256(json.dumps([(n, expected[n]) for n in names]).encode()).hexdigest()}
    parameters = {}
    try:
        modules = FSDP.fsdp_modules(wrapped)
        if not modules:
            raise ValueError('FSDP wrapper required')
        for module in modules:
            if not module._use_orig_params or module.sharding_strategy != ShardingStrategy.FULL_SHARD:
                raise ValueError('Adapter gather requires FULL_SHARD with original parameters')
            group = module.process_group
            if dist.get_world_size(group) != world or [dist.get_global_rank(group, r) for r in range(world)] != list(range(world)):
                raise ValueError('Adapter gather requires the default full-world rank order')
        for name, parameter in wrapped.named_parameters():
            if not parameter.requires_grad:
                continue
            name = '.'.join(part for part in name.split('.') if part != '_fsdp_wrapped_module')
            if name in parameters or '.lora_' not in name or parameter.dtype != torch.float32:
                raise ValueError('Unexpected, duplicate, or non-FP32 trainable parameter: ' + name)
            parameters[name] = parameter
        if set(parameters) != set(expected):
            raise ValueError('Local adapter keys differ from the pre-FSDP contract')
        local['lengths'] = [parameters[n].numel() for n in names]
    except Exception as error:
        local['error'] = str(error)
    metadata = [None] * world
    group = object_group()
    dist.all_gather_object(metadata, local, group=group)
    names, lengths = adapter_shard_layout(expected, metadata)
    width = max(lengths)
    packed = torch.zeros(width, dtype=torch.float32, device='cpu')
    offset = 0
    for name in names:
        value = parameters[name].detach().reshape(-1)
        packed[offset:offset + value.numel()].copy_(value)
        offset += value.numel()
    received = [torch.empty_like(packed) for _ in range(world)] if rank == 0 else None
    dist.gather(packed, gather_list=received, dst=0, group=group)
    state = {}
    if rank == 0:
        offsets = [0] * world
        for i, name in enumerate(names):
            pieces = []
            for r in range(world):
                size = metadata[r]['lengths'][i]
                pieces.append(received[r][offsets[r]:offsets[r] + size])
                offsets[r] += size
            state[name] = torch.cat(pieces).view(expected[name])
    return state


def qualify_adapter_gather(wrapped, expected):
    """Compare all adapter values against FULL_STATE_DICT on the same model."""
    import torch
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
    dist.barrier()
    started = time.monotonic()
    adapter = gather_adapter_state(wrapped, expected)
    dist.barrier()
    adapter_seconds = time.monotonic() - started
    started = time.monotonic()
    with FSDP.state_dict_type(wrapped, StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
        full = wrapped.state_dict()
    dist.barrier()
    full_seconds = time.monotonic() - started
    result = None
    if dist.get_rank() == 0:
        keys = {k for k in full if '.lora_' in k}
        mismatches = [k for k in expected if k not in full or not torch.equal(adapter[k], full[k])]
        result = {'passed': keys == set(expected) and not mismatches,
            'adapter_tensors': len(adapter), 'adapter_parameters': sum(v.numel() for v in adapter.values()),
            'mismatches': mismatches, 'unexpected_keys': sorted(keys - set(expected)),
            'adapter_gather_seconds': adapter_seconds, 'full_state_gather_seconds': full_seconds,
            'comparison': 'Every FP32 adapter tensor must be exactly equal to FULL_STATE_DICT.'}
    del full
    decision = [result]
    dist.broadcast_object_list(decision, src=0, group=object_group())
    if not decision[0]['passed']:
        raise ValueError('Adapter-only gather differs from FULL_STATE_DICT: ' + str(decision[0]))
    return adapter, decision[0]


def restart_due(plan, step, start, portable):
    return step==start+1 or step%plan['restart_interval']==0 or step in portable


def verify_files(directory, files):
    for name, record in files.items():
        if Path(name).name!=name:
            raise ValueError('Nonlocal restart member')
        path = Path(directory)/name
        if path.stat().st_size!=record['bytes'] or sha(path)!=record['sha256']:
            raise ValueError('Restart checksum mismatch: '+name)


def choose_resume(spec):
    plan = spec['durable_restart']
    root = Path(plan['state_directory'])
    candidates = []
    for path in root.glob('checkpoint-*'):
        receipt = path/'training-state.json'
        if not receipt.is_file():
            continue
        data = json.loads(receipt.read_text())
        if data['format']!=FORMAT or data['lineage_sha256']!=plan['lineage_sha256']:
            raise ValueError('Restart lineage differs')
        if not 0<data['step']<=plan['target_step']:
            raise ValueError('Restart step outside frozen schedule')
        candidates.append((data['step'], path, data))
    if not candidates:
        return {'step':0,'path':spec['base_model'],'format':'untouched-base','rng_policy':'Frozen initial seed'}
    step, path, data = max(candidates,key=lambda row:row[0])
    verify_files(path, data['files'])
    return {'step':step,'path':str(path),'optimizer_path':str(path/'optimizer.pt'),
            'format':FORMAT,'receipt':data,'rng_policy':'Exact saved CPU, CUDA and Python RNG on each rank'}


def restore_optimizer(wrapped, optimizer, spec, root):
    import efe_training_state as full
    # The validated optimizer scatter and per-rank RNG format are shared; model
    # restoration is deliberately separate and uses the exact 700 LoRA tensors.
    copied = {**spec, 'active_restore':{**spec['active_restore'],'format':full.FORMAT}}
    full.restore_optimizer(wrapped, optimizer, copied, root)


def save_state(wrapped, optimizer, model, spec, root, step, total, record, portable):
    import torch
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig, FullOptimStateDictConfig
    rank = dist.get_rank()
    plan = spec['durable_restart']
    directory = Path(plan['state_directory'])
    rng = {'torch_cpu':torch.get_rng_state(),'torch_cuda':torch.cuda.get_rng_state(),'python':random.getstate()}
    rngs = [None]*dist.get_world_size() if rank==0 else None
    dist.gather_object(rng, rngs, dst=0, group=object_group())
    final_probe = None
    if spec.get('validate_each_restart_probe') or (step==total and spec.get('validate_final_probe',True)):
        from trainer import training_plan
        from efe_qwen36_health import measure
        rows = json.loads((Path(root)/'dataset-manifest.json').read_text())['examples']
        _, schedule, _, _ = training_plan(rows, spec)
        final_probe,probe_values = measure(wrapped,root,schedule)
    gather_mode = spec.get('checkpoint_gather', 'full-state-v1')
    if gather_mode not in {'full-state-v1', 'adapter-shards-v1'}:
        raise ValueError('Unknown checkpoint gathering mode')
    state = gather_adapter_state(wrapped, model._efe_adapter_shapes) if gather_mode == 'adapter-shards-v1' else None
    with FSDP.state_dict_type(wrapped, StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True,rank0_only=True),
            FullOptimStateDictConfig(offload_to_cpu=True,rank0_only=True)):
        if state is None:
            state = wrapped.state_dict()
        optim = FSDP.optim_state_dict(wrapped,optimizer)
    if spec.get('validate_each_restart_probe'):
        after,_=measure(wrapped,root,schedule)
        if abs(after-final_probe)>.0005:
            raise ValueError('Full-model loss changed during checkpoint gathering')
    ready = None
    if rank==0:
        pending = directory/f'.pending-{os.environ.get("EFE_WORKFLOW_ID","worker")}-{step}'
        if pending.exists():
            shutil.rmtree(pending)
        audit = save_exact(model, state, model._efe_adapter_shapes, pending)
        torch.save({'optimizer':optim,'step':step,'rng_states':rngs},pending/'optimizer.pt')
        files = {p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in pending.iterdir() if p.is_file()}
        revision = spec['checkpoint_revision_prefix']+'-step-'+str(step)
        manifest = {'files':{k:v for k,v in files.items() if k!='optimizer.pt'},'revision':revision,
            'step':step,'source_base':spec['base_model'],'adapter_only':True,'full_parameters':False,
            'serialization_verified':True,'saved_tensors':audit['saved_tensors'],
            'dataset_sha256':sha(Path(root)/'dataset-manifest.json') if (Path(root)/'dataset-manifest.json').exists() else None}
        write(pending/'checkpoint-manifest.json',manifest)
        files['checkpoint-manifest.json']={'bytes':(pending/'checkpoint-manifest.json').stat().st_size,
                                           'sha256':sha(pending/'checkpoint-manifest.json')}
        destination = directory/f'checkpoint-{step}'
        if portable:
            ready = {'checkpoint':str(destination),'manifest':str(destination/'checkpoint-manifest.json'),
                'revision':revision,'step':step,'final':step==total,'selection':'Frozen data-scaled budget; no evaluation selection'}
        receipt = {'format':FORMAT,'step':step,'target_step':total,'lineage_sha256':plan['lineage_sha256'],
            'checkpoint_gather':gather_mode,
            'files':files,'metric':record,'portable':ready,'post_update_full_target_ce':final_probe,
            'world_size':dist.get_world_size(),'immutable_base':spec['base_model']}
        write(pending/'training-state.json',receipt)
        for path in pending.iterdir():
            if path.is_file():
                with path.open('rb') as stream:
                    os.fsync(stream.fileno())
        if destination.exists():
            raise ValueError('Refusing to replace an existing committed restart')
        pending.rename(destination)
        if ready:
            write(Path(root)/spec.get('ready_name','adapter-ready.json'),ready)
        recent=sorted(directory.glob('checkpoint-*'),key=lambda p:int(p.name.split('-')[-1]),reverse=True)
        keep=set(recent[:spec.get('restart_keep_recent',3)])
        for old in recent:
            if old in keep:continue
            info = json.loads((old/'training-state.json').read_text())
            if not info.get('portable'):
                shutil.rmtree(old)
    del state, optim
    dist.barrier()
    return ready
