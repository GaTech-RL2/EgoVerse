"""Copy the unchanged LeAct corpus and committed restart to available storage."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import shutil

SOURCE = Path('/mnt/amlfs-03/home/rpunamiya/leact/studies/efe-curated-leact-20261101-v1/curated-leact')
PARENT = Path('/mnt/amlfs-01/home/rpunamiya/leact/studies/efe-curated-leact-20261103-v5')
LOCAL = Path('/tmp/efe-leact-relocation-20261103-v5')


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def write(name, value):
    value = {**value, 'updated_utc': datetime.now(timezone.utc).isoformat()}
    path = LOCAL / name
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--spec-sha256', required=True)
    args = parser.parse_args()
    LOCAL.mkdir(exist_ok=True)
    own_lock = (LOCAL / 'relocation.lock').open('a')
    fcntl.flock(own_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if sha(args.spec) != args.spec_sha256:
        raise ValueError('Relocated specification differs from frozen launch')
    spec = json.loads(args.spec.read_text())
    prior = json.loads((SOURCE / 'spec.json').read_text())
    assert spec['archive_root'] == str(PARENT)
    assert {k: v for k, v in spec.items() if k not in ('archive_root','checkpoint_durability')} == {
        k: v for k, v in prior.items() if k not in ('archive_root','checkpoint_durability')}
    assert spec['checkpoint_durability'] == {**prior['checkpoint_durability'], 'save_restart_steps':[18]}
    destination = PARENT / 'curated-leact'
    staging = PARENT / '.curated-leact.copying'
    PARENT.mkdir(parents=True, exist_ok=True, mode=0o700)
    if destination.exists():
        raise ValueError('Destination exists; inspect its receipt before reusing')
    if shutil.disk_usage(PARENT).free < 150_000_000_000:
        raise RuntimeError('Destination lacks room for the corpus and final atomic save/export')
    source_lock = (SOURCE / 'training-writer.lock').open('a')
    fcntl.flock(source_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        write('progress.json', {'phase': 'copying', 'source': str(SOURCE), 'destination': str(destination)})
        shutil.copytree(SOURCE, staging, dirs_exist_ok=True, symlinks=True,
            ignore=shutil.ignore_patterns('.pending-*', 'pipeline-writer.lock', 'training-writer.lock', 'native-artifacts','restart-states'))
        states = staging / 'restart-states'
        states.mkdir(exist_ok=True)
        checkpoint = states / 'checkpoint-12'
        expected_source = SOURCE / 'restart-states/checkpoint-12'
        if not checkpoint.exists():
            checkpoint.symlink_to(expected_source, target_is_directory=True)
        if not checkpoint.is_symlink() or checkpoint.resolve() != expected_source.resolve():
            raise ValueError('Restart reference must point to the retained verified source')
        write('progress.json', {'phase': 'verifying_destination', 'destination': str(destination)})
        checkpoint = staging / 'restart-states/checkpoint-12'
        receipt = json.loads((checkpoint / 'training-state.json').read_text())
        assert receipt['step'] == 12 and receipt['target_step'] == 18
        for name, item in receipt['files'].items():
            assert Path(name).name == name
            p = checkpoint / name
            if p.stat().st_size != item['bytes'] or sha(p) != item['sha256']:
                raise ValueError('Relocated restart file differs: ' + name)
        for name in ['dataset-manifest.json', 'training-lineage.json', 'selection.json', 'membership.json']:
            if sha(staging / name) != sha(SOURCE / name):
                raise ValueError('Frozen study metadata changed during relocation: ' + name)
        (staging / 'relocation-source-spec.json').write_text(json.dumps(prior, indent=2) + '\n')
        shutil.copyfile(args.spec, staging / 'spec.json')
        if sha(staging / 'spec.json') != args.spec_sha256:
            raise ValueError('Relocated launch spec differs')
        staging.rename(destination)
        record = {'success': True, 'source': str(SOURCE), 'destination': str(destination),
                  'checkpoint': str(destination / 'restart-states/checkpoint-12'),
                  'step': 12, 'target_step': 18, 'verified_restart_files': len(receipt['files']),
                  'spec_sha256': args.spec_sha256, 'recipe_changed': False,
                  'source_retained': True, 'checkpoint_source_reference':str(expected_source),
                  'storage_only_change':'Retain committed update 12 on the original filesystem; write the next full restart and portable export at the predeclared final update 18. Interruption before final commit resumes from 12.',
                  'shared_free_bytes': shutil.disk_usage(PARENT).free}
        (destination / 'relocation-receipt.json').write_text(json.dumps(record, indent=2) + '\n')
        write('complete.json', record)
        write('progress.json', {**record, 'phase': 'ready_to_resume'})
    finally:
        fcntl.flock(source_lock, fcntl.LOCK_UN)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        LOCAL.mkdir(exist_ok=True)
        write('progress.json', {'phase': 'failed', 'error': repr(error), 'success': False})
        raise
