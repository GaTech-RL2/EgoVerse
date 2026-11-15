"""Atomic FP32 LoRA + Adam + per-rank RNG snapshots over an immutable base."""
import hashlib
import json
import os
from pathlib import Path
import random
import shutil

from efe_training_data import sha, write
from efe_qwen36_adapter_io import save_exact
from efe_training_state import object_group

FORMAT = 'efe-fp32-lora-adam-rng-v1'


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
    with FSDP.state_dict_type(wrapped, StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True,rank0_only=True),
            FullOptimStateDictConfig(offload_to_cpu=True,rank0_only=True)):
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
