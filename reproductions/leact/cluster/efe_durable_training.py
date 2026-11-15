"""Resume the same curated SFT schedule from atomic shared-storage snapshots."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

try:
    from . import efe_training_state as restart
    from .efe_training_data import sha, write
    from .efe_train_vl import training_plan
except ImportError:
    import efe_training_state as restart
    from efe_training_data import sha, write
    from efe_train_vl import training_plan


def dataset_receipt(root):
    root=Path(root)
    data=json.loads((root/'dataset-manifest.json').read_text())
    if sha(root/data.get('states_file','warmstart-states.jsonl'))!=data['states_sha256']:
        raise ValueError('Frozen public training states changed')
    for row in data['examples']:
        if sha(root/row['tensor_file'])!=row['tensor_sha256']:
            raise ValueError('Frozen tensor changed: '+row['id'])
    files={str(p.relative_to(root)):{'bytes':p.stat().st_size,'sha256':sha(p)}
           for p in sorted((root/'tensor-blobs').glob('*.pt'))}
    return data, {'dataset_sha256':sha(root/'dataset-manifest.json'),'visual_blobs':files}


def make_plan(spec, root, data, corpus):
    steps,schedule,_,digest=training_plan(data['examples'],spec)
    keys=['base_model','base_revision','context','seed','lr','effective_batch',
          'balanced_passes','sampling_policy','tensor_storage','checkpoint_fractions']
    lineage={'recipe':{k:spec.get(k) for k in keys},'corpus':corpus,
             'steps':steps,'schedule_sha256':digest,'world_size':8}
    identity=hashlib.sha256(json.dumps(lineage,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    interval=spec['checkpoint_durability']['restart_interval']
    if not isinstance(interval,int) or not 1<=interval<=8:raise ValueError('Invalid restart interval')
    policy = {'origin_step':0,'target_step':steps,'initial_checkpoint':spec['base_model'],
        'state_directory':str(Path(root)/'restart-states'),'restart_interval':interval,
        'keep_restart_states':1,'lineage_sha256':identity}
    explicit = spec['checkpoint_durability'].get('save_restart_steps')
    if explicit is not None:
        if not isinstance(explicit,list) or not explicit or steps not in explicit or any(
                type(s) is not int or not 0 < s <= steps for s in explicit):
            raise ValueError('Explicit restart schedule must include the final update')
        policy['save_restart_steps'] = explicit
    return lineage, policy, steps


def run(pilot, spec, env=None):
    root=pilot.ROOT
    if not str(root.resolve()).startswith(('/mnt/amlfs-03/','/mnt/amlfs-01/home/rpunamiya/leact/')):
        raise ValueError('Durable training requires the shared project filesystem')
    lock=(root/'training-writer.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        data,corpus=dataset_receipt(root)
        lineage,policy,steps=make_plan(spec,root,data,corpus)
        frozen=root/'training-lineage.json'
        if frozen.exists() and json.loads(frozen.read_text())!=lineage:
            raise ValueError('Restart changed the frozen corpus, model, or training schedule')
        write(frozen,lineage)
        Path(policy['state_directory']).mkdir(exist_ok=True)
        runtime={**spec,'durable_restart':policy,'expected_optimizer_steps':steps}
        restart.discard_uncommitted(runtime)
        restore=restart.choose_resume(runtime)
        if restore['step']:
            restart.verify_files(restore['path'],restore['receipt']['files'])
        runtime['active_restore']=restore
        write(root/'runtime-spec.json',runtime)
        if restore['step']==steps:
            ready=restore['receipt'].get('portable')
            if not ready or not ready.get('final'):raise ValueError('Final restart lacks its serving export')
            manifest=json.loads(Path(ready['manifest']).read_text())
            restart.verify_files(ready['checkpoint'],manifest['files'])
            write(root/spec.get('ready_name','warmstart-ready.json'),ready)
            write(root/'train/complete.json',{'success':True,'optimizer_steps':steps,
                'recovered_completed_checkpoint':True,'optimizer_updates_this_attempt':0})
            return
        # Preserve superseded receipts, including updates newer than the last
        # committed snapshot, before restarting that suffix of the schedule.
        attempt=root/'attempts'/os.environ['EFE_WORKFLOW_ID']
        attempt.mkdir(parents=True,exist_ok=True)
        for name in ['train/metrics.jsonl','train/configuration.json','train/schedule.json',
                     'train/complete.json','train.log','progress.json']:
            src=root/name;dst=attempt/name
            if src.exists() and not dst.exists():
                dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
        metrics=root/'train/metrics.jsonl'
        if metrics.exists():
            rows=[json.loads(line) for line in metrics.read_text().splitlines() if line.strip()]
            metrics.write_text(''.join(json.dumps(r)+'\n' for r in rows if r['step']<=restore['step']))
        if (root/'train/complete.json').exists():
            raise ValueError('Completed training lacks a final durable restart; inspect before overwriting')
        write(attempt/'restore.json',{'step':restore['step'],'path':restore['path'],
            'lineage_sha256':policy['lineage_sha256'],'learning_rate_schedule':'Original warmup/cosine schedule; global step is preserved.'})
        if env is None:
            import efe_cuda_helpers as helpers
            helpers.ROOT=root;helpers.ARCHIVE=root
            env=helpers.driver_environment()
            env.update(CUDA_VISIBLE_DEVICES=','.join(map(str,range(8))),NCCL_P2P_DISABLE='1',
                OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',TORCH_NCCL_ASYNC_ERROR_HANDLING='1')
        pilot.progress('warmstart_training',
            f"Curated SFT from update {restore['step']} to {steps}, LR={spec['lr']}; corpus and restart state are on shared storage.",
            training_started=True,resumed_from_step=restore['step'],steps=steps)
        command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=8',
                 '/tmp/efe_train_vl.py','--spec',str(root/'runtime-spec.json'),'--root',str(root),'--mode','train']
        pilot.run_logged('train',command,env,spec.get('training_timeout_seconds',24*3600))
    finally:
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
