"""Fetch a pinned open-weight student before its native robot qualification."""
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download


def main():
    lane = Path(__file__).resolve().parent
    spec = json.loads((lane / 'spec.json').read_text())
    model = Path(spec['student_checkpoint'])
    root = model.parent
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'checkpoint.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def write(name, value):
        value = {**value, 'updated_utc': datetime.now(timezone.utc).isoformat()}
        path = root / name
        tmp = path.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(value, indent=2) + '\n')
        tmp.replace(path)

    provenance = {'model': spec['request_model'], 'revision': spec['student_revision'],
                  'training_started': False, 'baseline_evaluation_started': False}
    try:
        write('preparation.json', {**provenance, 'phase': 'downloading_checkpoint'})
        snapshot_download(spec['request_model'], revision=spec['student_revision'],
                          local_dir=str(model), max_workers=4,
                          allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja', 'LICENSE*'])
        write('preparation.json', {**provenance, 'phase': 'checksumming_checkpoint'})
        files = {}
        for path in sorted(model.iterdir()):
            if not path.is_file():
                continue
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    digest.update(block)
            files[path.name] = {'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}
        index = json.loads((model / 'model.safetensors.index.json').read_text())
        if not set(index['weight_map'].values()).issubset(files):
            raise ValueError('Downloaded checkpoint is missing indexed shards')
        config = json.loads((model / 'config.json').read_text())
        if config['architectures'] != ['Qwen3_5MoeForConditionalGeneration']:
            raise ValueError('Unexpected student model architecture')
        write('checkpoint-manifest.json', {**provenance, 'path': str(model), 'files': files})
        write('preparation.json', {**provenance, 'phase': 'checkpoint_ready',
              'files': len(files), 'bytes': sum(x['bytes'] for x in files.values())})
    except Exception as error:
        write('preparation.json', {**provenance, 'phase': 'preparation_failed', 'error': repr(error)})
        raise


if __name__ == '__main__':
    main()
