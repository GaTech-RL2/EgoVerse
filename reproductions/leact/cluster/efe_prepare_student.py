"""Download and checksum the frozen EFE student checkpoint; no inference score."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download

MODEL = 'Qwen/Qwen3-VL-8B-Thinking'
REVISION = '92f3c4b4feadd3a016ef468d103bb5f58b2a2c6b'
ROOT = Path('/mnt/amlfs-03/home/rpunamiya/leact/studies/efe-student-v1')


def write(name, value):
    ROOT.mkdir(parents=True, exist_ok=True)
    value['updated_utc'] = datetime.now(timezone.utc).isoformat()
    path = ROOT / name
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temp.replace(path)


def main():
    import fcntl
    ROOT.mkdir(parents=True, exist_ok=True)
    lock = (ROOT / 'checkpoint.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    spec = {'model': MODEL, 'revision': REVISION, 'baseline_evaluation_started': False,
            'training_started': False}
    write('preparation.json', {**spec, 'phase': 'downloading_checkpoint'})
    try:
        path = Path(snapshot_download(MODEL, revision=REVISION, local_dir=str(ROOT / 'base-model'),
                                      allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja', 'LICENSE*'],
                                      max_workers=2))
        write('preparation.json', {**spec, 'phase': 'checksumming_checkpoint'})
        files = {}
        for p in sorted(path.iterdir()):
            if not p.is_file():
                continue
            digest = hashlib.sha256()
            with p.open('rb') as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    digest.update(chunk)
            files[p.name] = {'bytes': p.stat().st_size, 'sha256': digest.hexdigest()}
        index = json.loads((path / 'model.safetensors.index.json').read_text())
        if not set(index['weight_map'].values()).issubset(files):
            raise ValueError('Student checkpoint is missing weight shards')
        write('checkpoint-manifest.json', {**spec, 'path': str(path), 'files': files})
        write('preparation.json', {**spec, 'phase': 'checkpoint_ready', 'path': str(path),
                                   'files': len(files), 'bytes': sum(r['bytes'] for r in files.values())})
    except Exception as error:
        write('preparation.json', {**spec, 'phase': 'preparation_failed', 'error': repr(error)})
        raise


if __name__ == '__main__':
    main()
