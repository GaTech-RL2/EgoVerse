"""Recover this experiment's failed save and relocate a reproducible model cache."""
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import urllib.request

TRAIN = Path('/mnt/amlfs-03/home/rpunamiya/leact/studies/efe-curated-leact-20261101-v1/curated-leact')
CACHE = Path('/mnt/amlfs-03/home/rpunamiya/leact/studies/efe-student-qwen36-20261103-v1')
LOCAL = Path('/tmp/efe-storage-recovery-20261103-v1')
REVISION = '995ad96eacd98c81ed38be0c5b274b04031597b0'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(value):
    value = {**value, 'updated_utc': datetime.now(timezone.utc).isoformat(),
             'shared_free_bytes': shutil.disk_usage(TRAIN).free}
    path = LOCAL / 'progress.json'
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)
    print(json.dumps({k: v for k, v in value.items() if k != 'cache_files'}), flush=True)


def main():
    LOCAL.mkdir(exist_ok=True)
    own_lock = (LOCAL / 'recovery.lock').open('a')
    fcntl.flock(own_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_root = TRAIN / 'restart-states'
    good = state_root / 'checkpoint-12'
    pending = state_root / '.pending-leact-efe-curated-leact-r1-1-16'
    receipt = json.loads((good / 'training-state.json').read_text())
    assert receipt['step'] == 12 and receipt['target_step'] == 18
    assert receipt['format'] == 'efe-fp32-adam-rng-v1'
    training_lock = (TRAIN / 'training-writer.lock').open('a')
    fcntl.flock(training_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        write({'phase': 'verifying_committed_restart', 'checkpoint': str(good)})
        for name, item in receipt['files'].items():
            assert Path(name).name == name
            path = good / name
            if path.stat().st_size != item['bytes'] or digest(path) != item['sha256']:
                raise ValueError('Committed restart failed verification: ' + name)
        if pending.exists():
            if pending.is_symlink() or (pending / 'training-state.json').exists():
                raise ValueError('Refusing to discard a committed or redirected checkpoint')
            marker = json.loads((pending / 'pending.json').read_text())
            assert marker['step'] == 16 and marker['lineage_sha256'] == receipt['lineage_sha256']
            assert (pending / 'optimizer.pt').stat().st_size < receipt['files']['optimizer.pt']['bytes']
            evidence = {'path': str(pending), 'marker': marker,
                        'reason': 'Confirmed ENOSPC during optimizer serialization; no committed restart receipt.',
                        'files': {p.name: p.stat().st_size for p in pending.iterdir() if p.is_file()}}
            (LOCAL / 'discarded-uncommitted-save.json').write_text(json.dumps(evidence, indent=2) + '\n')
            shutil.rmtree(pending)
        write({'phase': 'committed_restart_verified_partial_save_removed', 'checkpoint': str(good), 'step': 12})
    finally:
        fcntl.flock(training_lock, fcntl.LOCK_UN)

    # This cache contains untouched public weights, not trained experiment outputs.
    # Wait until the live server has completely loaded them before relocating them.
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen('http://127.0.0.1:19880/health', timeout=5) as reply:
                if reply.status == 200:
                    break
        except Exception:
            pass
        time.sleep(5)
    else:
        raise RuntimeError('Model server did not become healthy; public cache retained')
    manifest = json.loads((CACHE / 'checkpoint-manifest.json').read_text())
    assert manifest['revision'] == REVISION and manifest['model'] == 'Qwen/Qwen3.6-35B-A3B'
    destination = LOCAL / 'qwen36-public-model-cache'
    destination.mkdir(exist_ok=True)
    copied = {}
    weights = {n: r for n, r in manifest['files'].items() if n.endswith('.safetensors')}
    assert len(weights) > 0
    if shutil.disk_usage(LOCAL).free < sum(r['bytes'] for r in weights.values()) + 20_000_000_000:
        raise RuntimeError('Insufficient node-local space to retain the public cache')
    write({'phase': 'relocating_public_weight_cache', 'checkpoint': str(good), 'cache_files': copied})
    for name, item in weights.items():
        assert Path(name).name == name
        src, dst = CACHE / 'base-model' / name, destination / name
        if dst.exists() and dst.stat().st_size == item['bytes'] and digest(dst) == item['sha256']:
            pass
        else:
            before = src.stat()
            if before.st_size != item['bytes'] or src.is_symlink():
                raise ValueError('Unexpected public cache file: ' + name)
            h = hashlib.sha256()
            tmp = dst.with_suffix('.copying')
            with src.open('rb') as stream, tmp.open('wb') as out:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    out.write(block)
                    h.update(block)
                out.flush()
                os.fsync(out.fileno())
            if tmp.stat().st_size != item['bytes'] or h.hexdigest() != item['sha256']:
                raise ValueError('Public model relocation checksum differs: ' + name)
            after = src.stat()
            if (before.st_ino, before.st_mtime_ns, before.st_size) != (after.st_ino, after.st_mtime_ns, after.st_size):
                raise ValueError('Source cache changed during relocation')
            tmp.replace(dst)
        copied[name] = {**item, 'node_local_path': str(dst)}
        # Validated local copy plus the immutable public revision makes this a
        # reproducible cache eviction. Shared manifests and tokenizer stay intact.
        if src.exists():
            src.unlink()
        write({'phase': 'relocating_public_weight_cache', 'checkpoint': str(good),
               'copied_files': len(copied), 'total_files': len(weights), 'cache_files': copied})
    record = {'success': True, 'phase': 'storage_recovered', 'checkpoint': str(good), 'step': 12,
              'verified_restart_files': len(receipt['files']), 'cache_files': copied,
              'model': manifest['model'], 'revision': REVISION,
              'cache_policy': 'Only untouched public weight shards moved to this active node. Public revision and per-file checksums retained; snapshot_download can restore missing shards. Trained checkpoints and results retained.'}
    (CACHE / 'cache-relocation-20261103.json').write_text(json.dumps(record, indent=2) + '\n')
    write(record)
    if shutil.disk_usage(TRAIN).free < 130_000_000_000:
        raise RuntimeError('Storage reclaimed but below the restart/export reserve; do not start training')
    (LOCAL / 'complete.json').write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        LOCAL.mkdir(exist_ok=True)
        write({'phase': 'failed', 'error': repr(error), 'success': False})
        raise
