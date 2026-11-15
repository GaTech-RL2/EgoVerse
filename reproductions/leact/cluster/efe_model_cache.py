"""Stage a pinned model on node-local disk without changing its logical source."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def stage(source, manifest_path, cache_root, revision, progress=None):
    source, manifest_path, cache_root = map(Path, (source, manifest_path, cache_root))
    raw = manifest_path.read_bytes()
    manifest_digest = hashlib.sha256(raw).hexdigest()
    manifest = json.loads(raw)
    if manifest['revision'] != revision:
        raise ValueError('Model cache revision differs from the pinned source')
    files = manifest['files']
    if not files or not any(name.endswith('.safetensors') for name in files):
        raise ValueError('Model cache requires a complete weight manifest')
    for name in files:
        if Path(name).name != name or name in {'.', '..'}:
            raise ValueError('Nonlocal model cache member: ' + name)
    cache_root.mkdir(parents=True, exist_ok=True)
    lock = (cache_root / 'writer.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    destination = cache_root / manifest_digest
    destination.mkdir(exist_ok=True)
    try:
        needed = sum(row['bytes'] for name, row in files.items() if not (destination / name).exists())
        if shutil.disk_usage(cache_root).free < needed + 5_000_000_000:
            raise RuntimeError('Insufficient node-local disk for the verified model cache')

        def copy(name, expected):
            src, dst = source / name, destination / name
            if dst.exists():
                if dst.is_symlink() or dst.stat().st_size != expected['bytes'] or sha(dst) != expected['sha256']:
                    raise ValueError('Existing local model file differs: ' + name)
                return name, expected['bytes']
            before = src.stat()
            if before.st_size != expected['bytes']:
                raise ValueError('Source model size differs: ' + name)
            temporary = destination / (name + '.copying')
            h = hashlib.sha256()
            with src.open('rb') as stream, temporary.open('wb') as out:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    out.write(block)
                    h.update(block)
                out.flush()
                os.fsync(out.fileno())
            after = src.stat()
            if ((before.st_ino, before.st_mtime_ns, before.st_size)
                    != (after.st_ino, after.st_mtime_ns, after.st_size)):
                raise ValueError('Source model changed during staging: ' + name)
            if h.hexdigest() != expected['sha256'] or sha(temporary) != expected['sha256']:
                raise ValueError('Staged model checksum differs: ' + name)
            temporary.replace(dst)
            return name, expected['bytes']

        completed = total_bytes = 0
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(copy, name, row) for name, row in files.items()]
            for future in as_completed(futures):
                name, size = future.result()
                completed += 1
                total_bytes += size
                if progress:
                    progress({'completed_files': completed, 'total_files': len(files),
                              'verified_bytes': total_bytes, 'last_file': name})
        if sha(manifest_path) != manifest_digest:
            raise ValueError('Source model manifest changed during staging')
        receipt = {'source': str(source), 'source_manifest': str(manifest_path),
                   'source_manifest_sha256': manifest_digest, 'revision': revision,
                   'local_model': str(destination), 'verified_files': len(files),
                   'verified_bytes': total_bytes, 'files': files,
                   'completed_utc': datetime.now(timezone.utc).isoformat(),
                   'source_retained': True, 'all_files_readback_verified': True}
        temporary = destination / 'local-cache-receipt.json.tmp'
        temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
        temporary.replace(destination / 'local-cache-receipt.json')
        return receipt
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
