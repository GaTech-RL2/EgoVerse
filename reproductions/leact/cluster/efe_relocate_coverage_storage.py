"""Preserve the coverage campaign while relocating it off an exhausted quota.

Preparation may run during collection. Activation requires fresh scheduler
evidence that all campaign workers have ended and the controller is stopped.
The canonical path remains valid; the complete original tree is retained.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat


SOURCE = Path('/mnt/amlfs-01/home/rpunamiya/leact/studies/efe-coverage-20261105-v1')
DESTINATION = Path('/mnt/amlfs-02/shared/egoverse/rpunamiya/leact/studies/efe-coverage-20261105-v1')
LOCAL = Path('/tmp/efe-coverage-storage-relocation-20261107')


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def inventory(root):
    result = {}
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in sorted(dirs + files):
            path = Path(directory) / name
            relative = str(path.relative_to(root))
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                result[relative] = {'kind': 'symlink', 'target': os.readlink(path)}
            elif stat.S_ISREG(mode):
                result[relative] = {'kind': 'file', 'bytes': path.stat().st_size, 'sha256': sha(path)}
            elif stat.S_ISDIR(mode):
                result[relative] = {'kind': 'directory'}
            else:
                raise ValueError('Unsupported special file: ' + str(path))
    return result


def capacity_probe(parent, required_bytes):
    probe = parent / '.efe-capacity-probe-20261107'
    fd = os.open(probe, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
    try:
        os.posix_fallocate(fd, 0, required_bytes)
        os.fsync(fd)
        allocated = os.fstat(fd).st_blocks * 512
        if allocated < required_bytes:
            raise RuntimeError('Capacity proof is sparse rather than allocated')
        return {'bytes': required_bytes, 'allocated_bytes': allocated,
                'probe_removed': True, 'utc': now()}
    finally:
        os.close(fd)
        probe.unlink()


def prepare(source, destination, local, required_bytes=350_000_000_000):
    if source.is_symlink() or not source.is_dir():
        raise ValueError('Expected the original, unredirected campaign directory')
    if destination.exists() or destination.is_symlink():
        raise ValueError('Destination already exists; inspect its receipt')
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    staging = destination.with_name(destination.name + '.copying')
    if staging.is_symlink():
        raise ValueError('Staging must not be a symlink')
    intent = {'source': str(source), 'destination': str(destination), 'staging': str(staging)}
    if staging.exists():
        if not (local / 'intent.json').exists() or json.loads((local / 'intent.json').read_text()) != intent:
            raise ValueError('Refusing to replace an unrecognized staging directory')
        # This is only our regenerable copy. The original tree still exists.
        shutil.rmtree(staging)
    write(local / 'intent.json', intent)
    capacity = capacity_probe(destination.parent, required_bytes)
    before = inventory(source)
    write(local / 'progress.json', {'phase': 'copying', 'utc': now(), 'entries': len(before)})
    shutil.copytree(source, staging, symlinks=True, dirs_exist_ok=True)
    after = inventory(source)
    copied = inventory(staging)
    stable = before == after == copied
    receipt = {'phase': 'prepared' if stable else 'source_changed_wait_for_workers',
               'utc': now(), 'source': str(source), 'destination': str(destination),
               'staging': str(staging), 'stable': stable, 'capacity': capacity,
               'files': after, 'source_retained': True}
    write(local / 'prepared.json', receipt)
    write(local / 'progress.json', {k: v for k, v in receipt.items() if k != 'files'})
    return receipt


def activate(source, destination, local, quiescent):
    checked = datetime.fromisoformat(quiescent['utc'])
    age = (datetime.now(timezone.utc) - checked).total_seconds()
    if not 0 <= age <= 120 or not quiescent.get('controller_stopped') or quiescent.get('active_campaign_workflows'):
        raise ValueError('Activation requires a fresh, quiescent scheduler receipt')
    receipt = json.loads((local / 'prepared.json').read_text())
    if not receipt['stable'] or receipt['source'] != str(source) or receipt['destination'] != str(destination):
        raise ValueError('No matching verified preparation')
    staging = Path(receipt['staging'])
    if inventory(source) != receipt['files'] or inventory(staging) != receipt['files']:
        raise ValueError('Source or copied evidence changed; prepare again after workers finish')
    backup = source.with_name(source.name + '.before-storage-relocation-20261107')
    redirect = source.with_name(source.name + '.storage-redirect-20261107')
    if backup.exists() or redirect.exists() or redirect.is_symlink() or destination.exists():
        raise ValueError('A relocation path already exists; inspect before resuming')
    redirect.symlink_to(destination, target_is_directory=True)
    try:
        staging.rename(destination)
        source.rename(backup)
        try:
            redirect.replace(source)
        except BaseException:
            backup.rename(source)
            raise
    except BaseException:
        if destination.exists() and not source.is_symlink() and not backup.exists():
            destination.rename(staging)
        redirect.unlink(missing_ok=True)
        raise
    if not source.is_symlink() or source.resolve() != destination.resolve():
        raise ValueError('Canonical campaign path does not resolve to the verified destination')
    result = {k: v for k, v in receipt.items() if k != 'files'}
    result.update(success=True, phase='storage_relocated', utc=now(),
                  backup=str(backup), canonical_path=str(source),
                  verified_files=sum(r['kind'] == 'file' for r in receipt['files'].values()),
                  verified_bytes=sum(r.get('bytes', 0) for r in receipt['files'].values()),
                  inventory_sha256=hashlib.sha256(json.dumps(receipt['files'], sort_keys=True).encode()).hexdigest(),
                  quiescence=quiescent, training_recipe_changed=False,
                  original_tree_deleted=False)
    write(destination / 'storage-relocation.json', result)
    write(local / 'complete.json', result)
    write(local / 'progress.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'activate'])
    parser.add_argument('--quiescent-receipt', type=Path)
    args = parser.parse_args()
    LOCAL.mkdir(exist_ok=True)
    lock = (LOCAL / 'relocation.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.action == 'prepare':
        if not os.path.ismount('/mnt/amlfs-02/shared'):
            raise ValueError('Expected the verified second shared filesystem mount')
        result = prepare(SOURCE, DESTINATION, LOCAL)
    else:
        if not args.quiescent_receipt:
            raise ValueError('A scheduler receipt is required')
        result = activate(SOURCE, DESTINATION, LOCAL, json.loads(args.quiescent_receipt.read_text()))
    print(json.dumps({k: v for k, v in result.items() if k != 'files'}), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        write(LOCAL / 'error.json', {'utc': now(), 'error': repr(error)})
        raise
