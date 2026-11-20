"""Pin inference weights from one committed training update without changing it."""
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def pin(training_root, destination, step, selection=None):
    training_root=Path(training_root);destination=Path(destination)
    source=training_root/'restart-states'/f'checkpoint-{step}'
    destination.mkdir(parents=True,exist_ok=True)
    with (destination/'pin.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        done=destination/'checkpoint-pin.json'
        if done.exists():
            value=json.loads(done.read_text())
            if value['step']!=step:raise ValueError('Pinned update differs')
            for name,record in value['files'].items():
                path=destination/'adapter'/name
                if path.stat().st_size!=record['bytes'] or sha(path)!=record['sha256']:
                    raise ValueError('Pinned adapter bytes differ: '+name)
            return value
        receipt_bytes=(source/'training-state.json').read_bytes()
        receipt=json.loads(receipt_bytes)
        runtime_bytes=(training_root/'spec.json').read_bytes();runtime=json.loads(runtime_bytes)
        if receipt['step']!=step or receipt['format']!='efe-fp32-lora-adam-rng-v1':
            raise ValueError('Expected the requested committed FP32 adapter update')
        if receipt['lineage_sha256']!=runtime['durable_restart']['lineage_sha256']:
            raise ValueError('Source training lineage differs')
        if receipt.get('post_update_full_target_ce') is None:
            raise ValueError('Committed update has no full-model replay reference')
        names=['adapter_model.safetensors','adapter_config.json','serialization-audit.json']
        pending=destination/'.pending-adapter'
        if pending.exists():shutil.rmtree(pending)
        pending.mkdir()
        # Open all source members first. A later retention unlink cannot replace
        # or invalidate the bytes held by these descriptors on the shared store.
        with ExitStack() as stack:
            streams={name:stack.enter_context((source/name).open('rb')) for name in names}
            for name,stream in streams.items():
                path=pending/name
                with path.open('wb') as output:shutil.copyfileobj(stream,output,1024*1024)
                record=receipt['files'][name]
                if path.stat().st_size!=record['bytes'] or sha(path)!=record['sha256']:
                    raise ValueError('Source checkpoint checksum mismatch: '+name)
        (pending/'training-state.json').write_bytes(receipt_bytes)
        files={name:receipt['files'][name] for name in names}
        manifest={'step':step,'files':files,'source_checkpoint':str(source),
            'source_training_receipt_sha256':hashlib.sha256(receipt_bytes).hexdigest(),
            'lineage_sha256':receipt['lineage_sha256'],'inference_only':True}
        (pending/'checkpoint-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        pending.rename(destination/'adapter')
        reference=destination/'reference';(reference/'train').mkdir(parents=True,exist_ok=True)
        (reference/'spec.json').write_bytes(runtime_bytes)
        for name in ['dataset-manifest.json','train/schedule.json']:
            shutil.copyfile(training_root/name,reference/name)
        for name in ['tensors','tensor-blobs']:
            original=training_root/name
            if original.exists() and not (reference/name).exists():
                (reference/name).symlink_to(original.resolve(),target_is_directory=True)
        value={'step':step,'checkpoint':str(destination/'adapter'),
            'manifest':str(destination/'adapter/checkpoint-manifest.json'),
            'files':files,'source_checkpoint':str(source),'source_training_root':str(training_root),
            'training_root':str(reference),'source_training_receipt_sha256':manifest['source_training_receipt_sha256'],
            'lineage_sha256':receipt['lineage_sha256'],
            'expected_post_update_ce':receipt['post_update_full_target_ce'],
            'dataset_sha256':sha(reference/'dataset-manifest.json'),
            'source_spec_sha256':sha(reference/'spec.json'),
            'base_model':runtime['base_model'],'base_revision':runtime['base_revision'],
            'model':runtime['model'],'additional_optimizer_updates':0,'final':False,
            'selection':selection or 'User-requested fixed update; intermediate diagnostic, not selected by evaluation performance.',
            'pinned_utc':datetime.now(timezone.utc).isoformat()}
        temporary=done.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(done)
        return value
