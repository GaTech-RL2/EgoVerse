"""Preserve node-local model weights across bounded OSMO rsync connections."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from .monitor_efe_preflight import remote, write

CHUNK_BYTES = 8 * 1024**2


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def valid(path, record):
    return path.is_file() and path.stat().st_size == record['bytes'] and sha(path) == record['sha256']


def assemble(directory, output, record):
    """Never replace a checkpoint member until its complete digest agrees."""
    partial = output.with_suffix(output.suffix + '.assembling')
    with partial.open('wb') as stream:
        for part in sorted(directory.glob('part-*.bin')):
            with part.open('rb') as source:
                shutil.copyfileobj(source, stream, 8 * 1024**2)
        stream.flush()
        os.fsync(stream.fileno())
    if not valid(partial, record):
        partial.unlink()
        raise ValueError('Assembled checkpoint member failed verification: ' + output.name)
    partial.replace(output)


def rsync(workflow, source, dest, log, timeout=120):
    with log.open('a') as output:
        return subprocess.run(['osmo', 'workflow', 'rsync', 'download', workflow, 'preflight',
            source + ':' + str(dest), '--no-progress', '--timeout', '20'],
            stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
            timeout=timeout, check=False).returncode


def download_part(workflow, source, directory, record, log):
    output = directory / Path(source).name
    for attempt in range(12):
        if valid(output, record):
            return
        try:
            rsync(workflow, source, directory, log)
        except subprocess.TimeoutExpired:
            pass
        time.sleep(2)
    if not valid(output, record):
        raise RuntimeError('Chunk transfer failed: ' + output.name)


def collect(workflow, destination):
    destination = destination.resolve()
    with (destination / 'model-copy.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ready = json.loads((destination / 'portable-ready.json').read_text())
        storage = destination / 'collected'
        model = storage / 'portable-model'
        model.mkdir(parents=True, exist_ok=True)
        log = storage / 'chunk-transfer.log'
        manifest = model / 'checkpoint-manifest.json'
        if not manifest.is_file() or sha(manifest) != ready['manifest_sha256']:
            if rsync(workflow, ready['manifest'], model, log):
                raise RuntimeError('Could not download checkpoint manifest')
        if sha(manifest) != ready['manifest_sha256']:
            raise ValueError('Checkpoint manifest hash mismatch')
        members = json.loads(manifest.read_text())['files']
        if any(Path(name).name != name for name in members):
            raise ValueError('Unsafe checkpoint member name')
        done = 0
        for name, record in members.items():
            output = model / name
            if valid(output, record):
                done += record['bytes']
                continue
            # One file's chunks and assembly coexist until verified. No existing
            # experiment artifacts are removed to make space.
            if shutil.disk_usage(storage).free < 2 * record['bytes'] + 5 * 1024**3:
                raise RuntimeError('Insufficient free space for verified model assembly')
            write(destination / 'model-copy-progress.json', {
                'phase': 'copying', 'member': name, 'verified_bytes': done,
                'total_bytes': ready['bytes'], 'updated_utc': datetime.now(timezone.utc).isoformat()})
            source = ready['checkpoint'] + '/' + name
            if record['bytes'] <= CHUNK_BYTES:
                for attempt in range(8):
                    rsync(workflow, source, model, log)
                    if valid(output, record):
                        break
                    time.sleep(3)
                else:
                    raise RuntimeError('Checkpoint member download failed: ' + name)
            else:
                remote_parts = ready['checkpoint'] + '-transfer8/' + name
                # Splitting is read-only with respect to the original model.
                script = '''from pathlib import Path
import hashlib,json,os
source=Path(SOURCE);dest=Path(DEST);dest.mkdir(parents=True,exist_ok=True)
records={}
with source.open('rb') as stream:
 for index in range((source.stat().st_size+SIZE-1)//SIZE):
  data=stream.read(SIZE);part=dest/f'part-{index:05d}.bin'
  if not part.exists() or part.stat().st_size!=len(data):
   temp=part.with_suffix('.tmp');temp.write_bytes(data);temp.replace(part)
  records[part.name]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
(dest/'chunks.json').write_text(json.dumps(records))
print('EFE_CAPTURE '+json.dumps({'parts':len(list(dest.glob('part-*.bin')))}),flush=True)
'''.replace('SOURCE', repr(source)).replace('DEST', repr(remote_parts)).replace('SIZE', str(CHUNK_BYTES))
                count = remote(workflow, script)['parts']
                directory = storage / name
                directory.mkdir(exist_ok=True)
                if rsync(workflow, remote_parts + '/chunks.json', directory, log):
                    raise RuntimeError('Could not download chunk manifest')
                chunks = json.loads((directory / 'chunks.json').read_text())
                if len(chunks) != count or sum(c['bytes'] for c in chunks.values()) != record['bytes']:
                    raise ValueError('Unexpected chunk manifest size')
                def fetch(item):
                    chunk, chunk_record = item
                    if Path(chunk).name != chunk:
                        raise ValueError('Unsafe chunk name')
                    download_part(workflow, remote_parts + '/' + chunk, directory, chunk_record, log)
                with ThreadPoolExecutor(max_workers=8) as pool:
                    for _ in pool.map(fetch, chunks.items()):
                        received=sum(p.stat().st_size for p in directory.glob('part-*.bin'))
                        write(destination / 'model-copy-progress.json', {
                            'phase': 'copying', 'member': name, 'verified_bytes': done,
                            'received_member_bytes': received, 'total_bytes': ready['bytes'],
                            'updated_utc': datetime.now(timezone.utc).isoformat()})
                assemble(directory, output, record)
                shutil.rmtree(directory)  # Only this collector's verified temporary chunks.
            if not valid(output, record):
                raise ValueError('Checkpoint member verification failed: ' + name)
            done += record['bytes']
        receipt = {**ready, 'local_checkpoint': str(model), 'local_manifest': str(manifest),
                   'verified_utc': datetime.now(timezone.utc).isoformat(), 'all_files_verified': True,
                   'transfer_method': 'Eight concurrent 8 MiB chunks; chunk and complete-member SHA256 verification'}
        write(destination / 'portable-collected.json', receipt)
        write(destination / 'model-copy-progress.json', {
            'phase': 'complete', 'verified_bytes': done, 'total_bytes': ready['bytes'],
            'updated_utc': receipt['verified_utc']})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workflow')
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    try:
        collect(args.workflow, args.destination)
    except Exception as error:
        write(args.destination / 'model-copy-error.json', {
            'error': str(error), 'updated_utc': datetime.now(timezone.utc).isoformat()})
        raise
