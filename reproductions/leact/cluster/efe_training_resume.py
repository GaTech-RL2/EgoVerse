"""Reuse an immutable EFE corpus for recovery or an explicitly named schedule comparison."""
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

try:
    from . import efe_training_pilot as pilot
    from .efe_training_data import sha, write
except ImportError:
    import efe_training_pilot as pilot
    from efe_training_data import sha, write


def validate_reuse(spec, previous):
    """A named schedule experiment may change LR/budget, never frozen inputs."""
    for key in ('base_model','base_revision','context','seed','effective_batch','scene_audit_sha256'):
        if spec[key]!=previous[key]:raise ValueError('Corpus reuse changed frozen condition: '+key)
    changes={key:{'previous':previous.get(key,1.0),'current':spec.get(key,1.0)}
             for key in ('lr','balanced_passes') if spec.get(key,1.0)!=previous.get(key,1.0)}
    comparison=spec.get('schedule_comparison')
    if comparison:
        if not comparison.get('study_id') or comparison.get('declared_changes')!=changes:
            raise ValueError('Schedule changes must be declared exactly in the named comparison')
        limit=6 if spec.get('continuation',{}).get('target_step')==96 else 3
        if not math.isfinite(spec['lr']) or spec['lr']<=0 or not 0<spec.get('balanced_passes',1.0)<=limit:
            raise ValueError('Invalid schedule comparison budget or learning rate')
    elif changes:
        raise ValueError('Recovery changed frozen training condition: '+','.join(changes))
    return changes


def main():
    spec=json.loads(pilot.SPEC.read_text())
    cleanup_receipt=None
    if spec.get('approved_storage_cleanup'):
        try:from . import efe_approved_optimizer_cleanup as cleanup
        except ImportError:import efe_approved_optimizer_cleanup as cleanup
        cleanup_receipt=cleanup.execute(spec['approved_storage_cleanup'])
        print(json.dumps({'approved_storage_cleanup':{key:cleanup_receipt[key] for key in
            ('status','removed_bytes','protected_file_count','protected_metadata_unchanged','available_after_bytes')}}),flush=True)
        # This Lustre mount reports zero in statvfs even when real allocations
        # succeed. Pin the actual full-budget allocation receipt instead.
        proof_spec=spec['storage_capacity_verification']
        proof=json.loads(Path(proof_spec['path']).read_text())
        if hashlib.sha256(json.dumps(proof,sort_keys=True).encode()).hexdigest()!=proof_spec['canonical_sha256']:
            raise ValueError('Storage allocation verification changed')
        if not proof.get('success') or not proof.get('probe_removed') or min(proof['bytes'],proof['allocated_bytes'])<280_000_000_000:
            raise ValueError('Full checkpoint-budget storage allocation was not verified')
    root=Path(spec['archive_root'])/os.environ['EFE_WORKFLOW_ID'];root.mkdir(parents=True,exist_ok=True)
    pilot.ROOT=root;write(root/'spec.json',spec)
    if cleanup_receipt:write(root/'storage-cleanup.json',cleanup_receipt)
    if cleanup_receipt:write(root/'storage-capacity-verification.json',proof)
    continuation=spec.get('continuation');state_lock=None
    try:
        pilot.progress('verifying_saved_corpus','Verifying the saved 512-example corpus and pinned initial weights; no teacher regeneration.')
        source=Path(spec['prepared_dataset']['archive'])
        if sha(source/'dataset-manifest.json')!=spec['prepared_dataset']['manifest_sha256']:
            raise ValueError('Saved dataset manifest changed')
        data=json.loads((source/'dataset-manifest.json').read_text())
        previous=json.loads((source/'spec.json').read_text())
        changes=validate_reuse(spec,previous)
        checkpoint=json.loads(Path(spec['base_manifest']).read_text())
        if checkpoint['revision']!=spec['base_revision']:raise ValueError('Base revision changed')
        for name,record in checkpoint['files'].items():
            path=Path(spec['base_model'])/name
            if path.stat().st_size!=record['bytes'] or sha(path)!=record['sha256']:
                raise ValueError('Base checkpoint changed: '+name)
        if sha(source/'warmstart-states.jsonl')!=data['states_sha256']:
            raise ValueError('Admitted public states changed')
        for row in data['examples']:
            path=source/row['tensor_file'];target=root/row['tensor_file']
            if sha(path)!=row['tensor_sha256']:raise ValueError('Encoded example changed: '+row['id'])
            target.parent.mkdir(parents=True,exist_ok=True);os.link(path,target)
        for name in ['dataset-manifest.json','corpus-manifest.json','warmstart-states.jsonl','pilot-membership.json',
                     'trace-admission.json','processor.json','student-interface.json','versions.json']:
            os.link(source/name,root/name)
        write(root/'corpus-recovery.json',{'source':str(source),'manifest_sha256':spec['prepared_dataset']['manifest_sha256'],
              'examples':len(data['examples']),'decisions':data['decisions'],'unchanged_tensors':True,
              'teacher_explanations_reused':True,'training_restarts_from_untouched_base':not bool(continuation),
              'reason':spec.get('recovery_reason',spec.get('scope')),
              'declared_schedule_changes':changes})
        runtime_spec=pilot.SPEC
        if continuation:
            try:from . import efe_training_state as restart
            except ImportError:import efe_training_state as restart
            state_root=Path(continuation['state_directory']);state_root.mkdir(parents=True,exist_ok=True)
            state_lock=(state_root/'writer.lock').open('a');fcntl.flock(state_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            restart.discard_uncommitted(spec)
            restore=restart.choose_resume(spec)
            pilot.progress('verifying_resume_state',f"Verifying saved model and Adam state at update {restore['step']}; continuing to {continuation['target_step']}.")
            if restore['format']==restart.FORMAT:
                restart.verify_files(restore['path'],restore['receipt']['files'])
            else:
                initial=Path(continuation['initial_manifest'])
                if sha(initial)!=continuation['initial_manifest_sha256']:raise ValueError('Initial continuation checkpoint manifest changed')
                receipt=json.loads(initial.read_text())
                if receipt['step']!=continuation['origin_step']:raise ValueError('Initial continuation step changed')
                restart.verify_files(restore['path'],receipt['files'])
                if sha(Path(restore['optimizer_path']))!=continuation['initial_optimizer_sha256']:
                    raise ValueError('Initial Adam optimizer state changed')
            spec={**spec,'active_restore':restore}
            runtime_spec=root/'runtime-spec.json';write(runtime_spec,spec)
            if restore['step']==continuation['target_step']:
                ready=restore['receipt'].get('portable')
                if not ready or not ready.get('final'):raise ValueError('Final restart lacks its native-evaluation export')
                receipt=json.loads(Path(ready['manifest']).read_text());restart.verify_files(ready['checkpoint'],receipt['files'])
                (root/'train').mkdir(exist_ok=True)
                write(root/'train/complete.json',{'success':True,'optimizer_steps':restore['step'],
                    'optimizer_updates_this_attempt':0,'recovered_completed_checkpoint':True})
                write(root/spec.get('ready_name','warmstart-ready.json'),ready)
                write(root/'complete.json',{'success':True,'warmstart':ready,'completed_utc':datetime.now(timezone.utc).isoformat()})
                return
        import efe_cuda_helpers as helpers
        helpers.ROOT=root;helpers.ARCHIVE=root;env=helpers.driver_environment()
        env.update(CUDA_VISIBLE_DEVICES=','.join(map(str,range(8))),OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',
                   TORCH_NCCL_ASYNC_ERROR_HANDLING='1')
        command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=8',
                 '/tmp/efe_train_vl.py','--spec',str(runtime_spec),'--root',str(root)]
        if continuation:
            memory=Path(continuation['validated_memory_preflight'])
            if sha(memory)!=continuation['validated_memory_preflight_sha256'] or not json.loads(memory.read_text()).get('success'):
                raise ValueError('Prior full-length optimizer memory validation changed')
            write(root/'memory-preflight-reuse.json',{'source':str(memory),
                'reason':'Same frozen corpus, model, FSDP layout, FP32 Adam states, microbatch and context. Restore checks and the first real update are mandatory; no discarded training steps.',
                'continuation_step':restore['step']})
            pilot.progress('warmstart_training',f"Continuing saved model and Adam state from update {restore['step']} to {continuation['target_step']}; frequent FP32 restart checkpoints enabled.",training_started=True)
        else:
            pilot.progress('optimizer_memory_preflight','Saved corpus verified; testing uniform FP32 master parameters, bf16 forward passes and full optimizer memory.')
            pilot.run_logged('smoke',command+['--mode','smoke'],env,3600)
            if not json.loads((root/'smoke/complete.json').read_text())['success']:raise ValueError('Optimizer smoke did not pass')
            pilot.progress('warmstart_training',f"Optimizer preflight passed; training from base at LR={spec['lr']:g}, {spec.get('balanced_passes',1.0):g} balanced passes.",training_started=True)
        pilot.run_logged('train',command+['--mode','train'],env,8*3600)
        write(root/'complete.json',{'success':True,'warmstart':json.loads((root/'warmstart-ready.json').read_text()),
              'completed_utc':datetime.now(timezone.utc).isoformat()})
        pilot.progress('warmstart_complete','Warmstart checkpoints saved; native development evaluation will start automatically.',training_started=True)
    except Exception as exc:
        pilot.progress('needs_review',str(exc)[-1600:]);write(root/'complete.json',{'success':False,'error':str(exc)[-1600:]});raise
    finally:
        for _ in range(30):
            if (root/'collected').exists():break
            time.sleep(5)


if __name__=='__main__':main()
