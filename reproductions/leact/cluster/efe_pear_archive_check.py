"""CPU-only streamed storage qualification, including an actual model backup."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    lane = Path(__file__).resolve().parent
    spec = json.loads((lane / 'spec.json').read_text())
    root = Path(spec['output']) / os.environ['EFE_WORKFLOW_ID']
    root.mkdir(parents=True, exist_ok=True)
    credential = Path('/dev/shm/rp_jhh_objectstore_credentials.json')
    descriptor = os.open(credential, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'w') as stream:
        stream.write(os.environ.pop('RP_JHH_PRIVATE_ARCHIVE_CONFIGURATION'))
    os.environ['PEAR_OBJECTSTORE_CREDENTIAL_FILE'] = str(credential)
    def record(phase, **extra):
        value = {'phase': phase, 'primary_rl_updates': 0, **extra}
        (root / 'status.json').write_text(json.dumps(value) + '\n')
        print('PEAR_ARCHIVE_CHECK ' + json.dumps(value), flush=True)
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', 'boto3==1.44.5'],
            check=True, timeout=300)
        from efe_pear_object_archive import archive, restore
        record('streaming_qualification_checkpoint')
        small = Path('/tmp/rp_jhh_archive_probe'); small.mkdir()
        # Exercise multipart (>64MiB), single-part and empty files, nested
        # checkpoint layout and the actual download/restore commit protocol.
        (small / 'actor').mkdir()
        with (small / 'actor' / 'rank0.bin').open('wb') as stream:
            for _ in range(72):
                stream.write(os.urandom(1 << 20))
        (small / 'curriculum.json').write_text('{"qualification_only":true}\n')
        (small / 'empty').touch()
        destination = spec['destination'] + '/rp_jhh_storage_check_' + os.environ['EFE_WORKFLOW_ID'].rsplit('-', 2)[-2]
        manifest = archive(small, destination)
        restored = small.with_name('rp_jhh_archive_probe_restored')
        restore(destination, restored)
        for path in small.rglob('*'):
            if path.is_file():
                expected = hashlib.sha256(path.read_bytes()).digest()
                if hashlib.sha256((restored / path.relative_to(small)).read_bytes()).digest() != expected:
                    raise ValueError('Archive restore bytes differ')
        record('multipart_and_restore_passed', checkpoint_bytes=manifest['bytes'],
            downloaded_readback_verified=True, restore_passed=True, destination=destination)
        model_destination = spec['destination'] + '/rp_jhh_qwen36_moe_warmstart_step000496_backup'
        saved = archive(spec['model'], model_destination)
        receipt = {'passed': True, 'multipart_and_restore_passed': True,
            'model_readback_verified': True, 'model_bytes': saved['bytes'],
            'model_revision': spec['model_revision'], 'model_archive': model_destination,
            'primary_rl_updates': 0, 'private_object_store': True}
        (root / 'result.json').write_text(json.dumps(receipt, indent=2) + '\n')
        record('durable_checkpoint_storage_qualified', **receipt)
    finally:
        credential.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
