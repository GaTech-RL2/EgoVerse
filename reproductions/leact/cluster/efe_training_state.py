"""Explicit EFE continuation and atomic full-precision restart checkpoints."""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil

try:
    from .efe_training_data import sha, write
except ImportError:
    from efe_training_data import sha, write


FORMAT='efe-fp32-adam-rng-v1'
_object_group = None


def object_group():
    """CPU transport for checkpoint metadata; model tensors still use NCCL."""
    global _object_group
    import torch.distributed as dist
    if _object_group is None:
        # Every rank reaches metadata collectives in the same order.
        _object_group = dist.new_group(backend='gloo',timeout=timedelta(seconds=300))
    return _object_group


def restart_due(plan, completed_steps, start_step, portable_steps):
    """Permit an explicit storage-only save schedule without changing updates."""
    if not plan:
        return False
    if completed_steps in portable_steps:
        return True
    if 'save_restart_steps' in plan:
        return completed_steps in plan['save_restart_steps']
    return (completed_steps == start_step + 1 or
            (completed_steps-plan['origin_step']) % plan['restart_interval'] == 0)


def continuation_lr(spec, completed_steps):
    plan=spec['continuation'];origin=plan['origin_step'];end=spec['expected_optimizer_steps']
    if not origin<=completed_steps<end:raise ValueError('Continuation update outside its frozen budget')
    return plan['initial_learning_rate']*.5*(1+math.cos(math.pi*(completed_steps-origin)/(end-origin)))


def validate_continuation(spec, schedule):
    plan=spec['continuation'];origin=plan['origin_step'];end=spec['expected_optimizer_steps']
    if not 0<origin<end or plan['target_step']!=end:raise ValueError('Invalid continuation budget')
    if not math.isfinite(plan['initial_learning_rate']) or plan['initial_learning_rate']<=0:
        raise ValueError('Invalid continuation learning rate')
    if spec['lr']!=plan['initial_learning_rate']:raise ValueError('Continuation LR must match its saved-optimizer origin')
    prefix=[r['id'] for r in schedule[:origin*spec['effective_batch']]]
    digest=hashlib.sha256(json.dumps(prefix,separators=(',',':')).encode()).hexdigest()
    if digest!=plan['source_prefix_sha256']:raise ValueError('Continuation changed already-trained example order')
    if plan['restart_interval']<1 or plan['keep_restart_states']!=1:
        raise ValueError('Unvalidated restart retention policy')


def verify_files(directory, files, *, hashes=True):
    directory=Path(directory)
    for name,record in files.items():
        if Path(name).name!=name:raise ValueError('Checkpoint contains a non-local file reference')
        path=directory/name
        if path.stat().st_size!=record['bytes']:raise ValueError('Checkpoint size mismatch: '+name)
        if hashes and sha(path)!=record['sha256']:raise ValueError('Checkpoint checksum mismatch: '+name)


def choose_resume(spec):
    plan=spec.get('continuation') or spec['durable_restart'];root=Path(plan['state_directory']);candidates=[]
    for directory in root.glob('checkpoint-*'):
        receipt=directory/'training-state.json'
        if not receipt.is_file():continue
        state=json.loads(receipt.read_text())
        if state.get('format')!=FORMAT or state.get('lineage_sha256')!=plan['lineage_sha256']:
            raise ValueError('Incompatible state in continuation namespace')
        if not plan['origin_step']<state['step']<=plan['target_step']:
            raise ValueError('Restart checkpoint step outside continuation budget')
        verify_files(directory,state['files'],hashes=False)
        candidates.append((state['step'],directory,state))
    if candidates:
        step,directory,state=max(candidates,key=lambda row:row[0])
        return {'path':str(directory),'optimizer_path':str(directory/'optimizer.pt'),'step':step,
                'format':FORMAT,'receipt':state,'rng_policy':'Restore saved CPU, CUDA and Python RNG independently on each rank.'}
    if not spec.get('continuation'):
        return {'path':spec['base_model'],'step':0,'format':'untouched-base',
                'rng_policy':'Initialize all ranks from the frozen seed; no optimizer updates have been committed.'}
    return {'path':plan['initial_checkpoint'],'optimizer_path':plan['initial_checkpoint']+'/optimizer.pt',
            'step':plan['origin_step'],'format':'legacy-bf16-model-fp32-adam',
            'rng_policy':'Initial continuation resets CPU, CUDA and Python RNG to the frozen seed; original checkpoint lacks per-rank CUDA RNG.'}


def discard_uncommitted(spec):
    """Called only while the worker holds the continuation namespace lock."""
    plan=spec.get('continuation') or spec['durable_restart'];root=Path(plan['state_directory'])
    for path in root.glob('.pending-*'):
        receipt=path/'pending.json'
        if path.is_symlink() or not receipt.is_file():continue
        if json.loads(receipt.read_text()).get('lineage_sha256')!=plan['lineage_sha256']:
            raise ValueError('Unexpected pending state in continuation namespace')
        shutil.rmtree(path)


class HashingWriter:
    """Hash the sequential torch serialization stream without rereading 67 GB."""
    def __init__(self,stream):self.stream=stream;self.digest=hashlib.sha256();self.bytes=0
    def write(self,value):
        count=self.stream.write(value)
        if count!=len(value):raise OSError('Incomplete checkpoint write')
        self.digest.update(value);self.bytes+=count;return count
    def flush(self):return self.stream.flush()
    def tell(self):return self.stream.tell()


def validate_optimizer_resume_records(records, expected_step):
    """Validate Adam counters collectively, including empty FSDP shards."""
    errors=[]
    for record in records:
        rank=record['rank']
        errors.extend(f'rank {rank}: {error}' for error in record.get('errors',[]))
        low,high=record['min_step'],record['max_step']
        if low is None or high is None:
            # use_orig_params can place every trainable parameter on other
            # ranks. Such a rank must have neither parameters nor Adam moments.
            if low is not None or high is not None or record['trainable_values'] or record['moment_values']:
                errors.append(f'rank {rank}: missing optimizer state for a nonempty shard')
        elif low<1 or high!=expected_step:
            errors.append(f'rank {rank}: optimizer counter range {low}..{high}, expected {expected_step}')
    if not any(record['max_step']==expected_step and record['moment_values']>0 for record in records):
        errors.append('No rank restored nonempty optimizer state at the checkpoint step')
    if errors:
        raise ValueError('Restored optimizer validation failed: '+'; '.join(errors))


def restore_optimizer(wrapped,optimizer,spec,root):
    import torch
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    rank=dist.get_rank();restore=spec['active_restore'];payload=None
    if rank==0:
        payload=torch.load(restore['optimizer_path'],map_location='cpu',weights_only=True,mmap=True)
        if payload['step']!=restore['step']:raise ValueError('Optimizer and model checkpoint steps differ')
    sharded=FSDP.scatter_full_optim_state_dict(payload['optimizer'] if rank==0 else None,wrapped,optim=optimizer)
    optimizer.load_state_dict(sharded);del sharded
    rng=[payload.get('rng_states') if rank==0 else None]
    dist.broadcast_object_list(rng,src=0,group=object_group())
    if restore['format']==FORMAT:
        if not rng[0] or len(rng[0])!=dist.get_world_size():raise ValueError('Missing per-rank restart RNG')
        state=rng[0][rank]
        torch.set_rng_state(state['torch_cpu']);torch.cuda.set_rng_state(state['torch_cuda'])
        random.setstate(state['python'])
    else:
        torch.manual_seed(spec['seed']);random.seed(spec['seed'])
    local_steps=[];moments=0;errors=[]
    for state in optimizer.state.values():
        if 'step' in state:local_steps.append(int(state['step'].item()))
        for key in ('exp_avg','exp_avg_sq'):
            if key in state:
                if state[key].dtype!=torch.float32:errors.append('Adam moments were not restored in FP32')
                moments+=state[key].numel()
    records=[None]*dist.get_world_size()
    dist.all_gather_object(records,{'rank':rank,'moment_values':moments,
        'trainable_values':sum(p.numel() for group in optimizer.param_groups for p in group['params']),
        'min_step':min(local_steps) if local_steps else None,
        'max_step':max(local_steps) if local_steps else None,'errors':errors},group=object_group())
    # All ranks validate the same evidence, so one bad shard cannot leave its
    # peers waiting in the next collective. Empty shards do not reset Adam.
    validate_optimizer_resume_records(records,restore['step'])
    if rank==0:
        write(Path(root)/'optimizer-resume.json',{'source':restore['path'],'step':restore['step'],
            'format':restore['format'],'rng_policy':restore['rng_policy'],'ranks':records,
            'initial_bf16_rounding':('The original step-32 export did not preserve FP32 master weights; this is a new documented continuation.'
                if spec.get('continuation') else 'No restart rounding: full FP32 master parameters and Adam moments restored.'),
            'updated_utc':datetime.now(timezone.utc).isoformat()})
    del payload,rng


def save_state(wrapped,optimizer,model,spec,root,step,total,record,portable):
    import torch
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP,StateDictType,FullStateDictConfig,FullOptimStateDictConfig
    from transformers import AutoProcessor
    rank=dist.get_rank();plan=spec.get('continuation') or spec['durable_restart'];state_root=Path(plan['state_directory'])
    rng={'torch_cpu':torch.get_rng_state(),'torch_cuda':torch.cuda.get_rng_state(),'python':random.getstate()}
    rng_states=[None]*dist.get_world_size() if rank==0 else None
    dist.gather_object(rng,rng_states,dst=0,group=object_group())
    with FSDP.state_dict_type(wrapped,StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True,rank0_only=True),
            FullOptimStateDictConfig(offload_to_cpu=True,rank0_only=True)):
        state=wrapped.state_dict();optim_state=FSDP.optim_state_dict(wrapped,optimizer)
    ready=None
    if rank==0:
        if any(v.is_floating_point() and v.dtype!=torch.float32 for v in state.values()):
            raise ValueError('Restart master parameters must be FP32')
        if portable:
            checkpoint=Path(root)/'train'/f'checkpoint-{step}'
            temporary=checkpoint.with_name('.'+checkpoint.name+'.pending')
            temporary.mkdir(exist_ok=False)
            serving={k:(v.to(torch.bfloat16) if v.is_floating_point() else v) for k,v in state.items()}
            model.config.use_cache=True
            model.save_pretrained(temporary,state_dict=serving,safe_serialization=True,max_shard_size='4GB')
            model.config.use_cache=False;del serving
            AutoProcessor.from_pretrained(spec['base_model'],local_files_only=True).save_pretrained(temporary)
            files={p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in temporary.iterdir() if p.is_file()}
            revision='efe-'+spec['stage']+'-step-'+str(step)
            write(temporary/'checkpoint-manifest.json',{'files':files,'revision':revision,'step':step,
                'full_parameters':True,'source_base':plan['initial_checkpoint'],
                'dataset_sha256':sha(Path(root)/'dataset-manifest.json'),'continuation_origin_step':plan['origin_step']})
            temporary.rename(checkpoint)
            ready={'checkpoint':str(checkpoint),'manifest':str(checkpoint/'checkpoint-manifest.json'),
                'revision':revision,'step':step,'final':step==total,
                'selection':'Predeclared final snapshot; native evaluation partition is fixed by the parent study. Capability screens paused.'}
        temporary=state_root/f'.pending-{os.environ.get("EFE_WORKFLOW_ID","worker")}-{step}'
        temporary.mkdir(exist_ok=False)
        write(temporary/'pending.json',{'lineage_sha256':plan['lineage_sha256'],'step':step})
        model.config.use_cache=True
        model.save_pretrained(temporary,state_dict=state,safe_serialization=True,max_shard_size='4GB')
        model.config.use_cache=False
        AutoProcessor.from_pretrained(spec['base_model'],local_files_only=True).save_pretrained(temporary)
        with (temporary/'optimizer.pt').open('wb') as raw:
            writer=HashingWriter(raw)
            torch.save({'optimizer':optim_state,'step':step,'rng_states':rng_states},writer)
            writer.flush();os.fsync(raw.fileno())
            optimizer_file={'bytes':writer.bytes,'sha256':writer.digest.hexdigest()}
        files={}
        for path in temporary.iterdir():
            if not path.is_file() or path.name=='pending.json':continue
            files[path.name]=optimizer_file if path.name=='optimizer.pt' else {'bytes':path.stat().st_size,'sha256':sha(path)}
            with path.open('rb') as stream:os.fsync(stream.fileno())
        receipt={'format':FORMAT,'lineage_sha256':plan['lineage_sha256'],'step':step,'target_step':total,
            'files':files,'metric':record,'portable':ready,'world_size':dist.get_world_size(),
            'updated_utc':datetime.now(timezone.utc).isoformat()}
        write(temporary/'training-state.json',receipt)
        with (temporary/'training-state.json').open('rb') as stream:os.fsync(stream.fileno())
        committed=state_root/f'checkpoint-{step}';temporary.rename(committed)
        write(state_root/'latest.json',{'checkpoint':str(committed),'step':step,'lineage_sha256':plan['lineage_sha256']})
        write(Path(root)/'restart-ready.json',{'checkpoint':str(committed),'step':step,'target_step':total,'format':FORMAT})
        if ready:write(Path(root)/spec.get('ready_name','warmstart-ready.json'),ready)
        # One committed snapshot is retained while its successor is written.
        # Deletion happens only after the successor's complete atomic commit.
        for old in state_root.glob('checkpoint-*'):
            if old==committed or old.is_symlink():continue
            old_receipt=old/'training-state.json'
            if old_receipt.is_file() and json.loads(old_receipt.read_text()).get('lineage_sha256')==plan['lineage_sha256']:
                shutil.rmtree(old)
    del state,optim_state
    dist.barrier()
    return ready
