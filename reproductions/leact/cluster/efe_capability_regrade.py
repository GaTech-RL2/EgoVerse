"""Recover completed capability generations with node-local scorer scratch.

Reuse the exact archived scorer image, inputs and generations. The unprivileged
scorer must not write to shared storage under the shared nobody-UID quota.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tarfile

import efe_capability_helpers as helpers


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', action='append', required=True)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    helpers.ROOT.mkdir(parents=True, exist_ok=True)
    environment = None
    for source_name in args.source:
        source = Path(source_name)
        spec = json.loads((source / 'spec.json').read_text())
        identity = spec['checkpoint']['id']
        destination = args.root / identity
        destination.mkdir(parents=True, exist_ok=True)
        helpers.ARCHIVE = destination
        try:
            if environment is None:
                environment = helpers.container_runtime(spec)
            bundle_archive = Path(spec['bundle_path'])
            if helpers.sha(bundle_archive) != spec['bundle_sha256']:
                raise ValueError('Frozen input archive changed')
            bundle = helpers.ROOT / ('regrade-inputs-' + identity)
            with tarfile.open(bundle_archive) as archive:
                archive.extractall(bundle, filter='data')
            plan = json.loads((bundle / 'suite.json').read_text())
            for name, digest in plan['file_sha256'].items():
                if helpers.sha(bundle / name) != digest:
                    raise ValueError('Frozen input changed: ' + name)
            cache = Path(spec['shared_root']) / 'scorer-cache' / spec['protocol_sha256']
            if not (cache / 'image.json').exists():
                raise ValueError('Original frozen scorer image is unavailable')
            image = helpers.scorer_image(spec, bundle, environment)
            generations = source / 'generations.jsonl'
            generation_sha = helpers.sha(generations)
            scratch = helpers.ROOT / ('regrade-output-' + identity)
            scratch.mkdir(exist_ok=False)
            scratch.chmod(0o777)
            command = helpers.sandbox(image, mounts=[
                (bundle / 'inputs', '/inputs', 'ro'),
                (generations, '/generations.jsonl', 'ro'),
                (scratch, '/output', 'rw')])
            command += ['python', '/opt/score.py', '--inputs', '/inputs',
                        '--generations', '/generations.jsonl', '--output', '/output',
                        '--suite-sha256', spec['protocol_sha256']]
            helpers.progress('regrading_saved_answers',
                             'Scoring the original saved answers with the original frozen image and node-local scratch.')
            helpers.run('grading', command, env=environment, timeout=2400)
            if helpers.sha(generations) != generation_sha:
                raise ValueError('Generations changed while grading')
            shutil.copytree(scratch, destination / 'grading', dirs_exist_ok=True)
            result = json.loads((scratch / 'results.json').read_text())
            checkpoint = json.loads((source / 'checkpoint.json').read_text())
            result.update(checkpoint=identity, weights_sha256=checkpoint['weights_sha256'],
                          workflow=source.name, generation=plan['generation'],
                          scorer_image_sha256=json.loads((destination / 'scorer-image.json').read_text())['archive_sha256'],
                          recovery={'original_archive': str(source), 'generations_sha256': generation_sha,
                                    'regenerated_answers': 0, 'change': 'Only grader output storage moved to node-local scratch; scoring code, image and saved answers unchanged.'})
            write(destination / 'results.json', result)
            write(destination / 'complete.json', {'success': True, 'updated_utc': datetime.now(timezone.utc).isoformat()})
            helpers.progress('completed', 'Saved capability answers successfully graded.', completed=985, total=985)
        except Exception as exc:
            write(destination / 'complete.json', {'success': False, 'error': str(exc)[-1500:]})
            helpers.progress('needs_review', str(exc)[-1500:])
    write(args.root / 'complete.json', {'finished': True, 'updated_utc': datetime.now(timezone.utc).isoformat()})


if __name__ == '__main__':
    main()
