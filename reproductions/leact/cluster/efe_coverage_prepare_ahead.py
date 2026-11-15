"""Finish corpus admission and cache pinned weights beside a live prefetch job.

Uses the running worker's frozen admission code and plan. Its open partial
corpus is never rewritten. This helper performs no optimizer/model inference.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def safely_prefetching(root):
    progress = json.loads((root / 'warmstart/progress.json').read_text())
    return (progress.get('phase') == 'warmstart_explanations_prefetch'
            and not progress.get('training_started')
            and progress.get('total_states', 0) - progress.get('completed_states', 0) >= 128)


def publish_copy(source, destination):
    temporary = destination.with_name(destination.name + '.prepare-ahead.tmp')
    with temporary.open('xb') as out:
        try:
            with source.open('rb') as stream:
                shutil.copyfileobj(stream, out, 8 * 1024 * 1024)
            out.flush()
            os.fsync(out.fileno())
            if sha(temporary) != sha(source):
                raise ValueError('Preparation copy checksum differs')
            os.link(temporary, destination)
        finally:
            temporary.unlink()


def promote(root, audited, plan):
    corpus = root / 'corpus'
    receipt = json.loads((audited / 'admission.json').read_text())
    if not (receipt.get('gate_passed') and receipt.get('native_execution_verified')
            and receipt.get('training_partition_only')
            and receipt['successful_episodes'] >= plan['target_successful_episodes']
            and receipt['unique_tasks'] >= plan['minimum_distinct_successful_tasks']):
        raise ValueError('Complete native admission and the frozen corpus gate are required')
    if sha(audited / 'decisions.jsonl.gz') != receipt['decisions_sha256']:
        raise ValueError('Admitted corpus checksum changed')
    if (corpus / 'admission.json').exists():
        prior = json.loads((corpus / 'admission.json').read_text())
        if prior != receipt or sha(corpus / 'decisions.jsonl.gz') != receipt['decisions_sha256']:
            raise ValueError('Another corpus was already frozen; do not replace it')
        return receipt
    if not safely_prefetching(root):
        raise ValueError('Live worker is near the admission boundary; leave its finalization intact')
    partial = corpus / 'decisions.jsonl.gz.tmp'
    before = sha(partial)
    final = corpus / 'decisions.jsonl.gz'
    if final.exists():
        if sha(final) != receipt['decisions_sha256']:
            raise ValueError('An unexpected final corpus already exists')
    else:
        publish_copy(audited / 'decisions.jsonl.gz', final)
    # The receipt is the commit point. Never replace an existing admission.
    publish_copy(audited / 'admission.json', corpus / 'admission.json')
    if sha(partial) != before:
        raise ValueError('The live partial corpus changed unexpectedly')
    return receipt


def run(stage, lane, expected):
    for name, digest in expected.items():
        if Path(name).name != name or sha(lane / name) != digest:
            raise ValueError('Frozen worker source differs: ' + name)
    sys.path.insert(0, str(lane))
    spec = json.loads((lane / 'spec.json').read_text())
    plan = json.loads((lane / 'coverage-plan.json').read_text())
    root = Path(spec['coverage_root'])
    output = root / 'prepare-ahead-20261107'
    output.mkdir(exist_ok=True)
    lock = (output / (stage + '.lock')).open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    record = {'stage': stage, 'started_utc': now(), 'frozen_source_hashes': expected,
              'training_recipe_changed': False, 'optimizer_updates': 0}
    write(output / (stage + '-start.json'), record)
    try:
        if stage == 'admission':
            from efe_coverage_admission import prepare
            audit = output / 'audit-root'
            audit.mkdir(exist_ok=True)
            for name in ['scene-audit', 'collection']:
                link = audit / name
                if not link.is_symlink():
                    link.symlink_to(root / name, target_is_directory=True)
                if link.resolve() != (root / name).resolve():
                    raise ValueError('Auditor input link differs')
            admitted = prepare(plan, audit)
            if not admitted['gate_passed']:
                raise ValueError('Frozen target is still short after full audit: ' + str(admitted['successful_episodes']))
            admitted = promote(root, audit / 'corpus', plan)
            result = {k: admitted[k] for k in ['successful_episodes', 'unique_tasks', 'decisions', 'decisions_sha256']}
            result['live_partial_corpus_preserved'] = True
        else:
            from huggingface_hub import snapshot_download
            snapshot_download(spec['model'], revision=spec['base_revision'], local_dir=spec['base_model'],
                              max_workers=4, allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja', 'LICENSE*'])
            files = {p.name: {'bytes': p.stat().st_size, 'sha256': sha(p)}
                     for p in Path(spec['base_model']).iterdir() if p.is_file()}
            result = {'model': spec['model'], 'revision': spec['base_revision'],
                      'files': len(files), 'bytes': sum(r['bytes'] for r in files.values())}
            ready = root / 'base-ready.json'
            # If the worker already reached its own download stage, let it
            # produce the normal manifests and use the now-cached shards.
            if not ready.exists() and safely_prefetching(root):
                manifest = root / 'base-checkpoint-manifest.json'
                write(manifest, {'files': files, 'revision': spec['base_revision'],
                                 'model': spec['model'], 'untuned_base': True})
                write(ready, {'checkpoint': spec['base_model'], 'manifest': str(manifest),
                              'revision': spec['base_revision']})
            result['base_ready_available'] = ready.exists()
        write(output / (stage + '-complete.json'), {**record, **result, 'success': True, 'completed_utc': now()})
        print(json.dumps(result), flush=True)
    except Exception as error:
        write(output / (stage + '-error.json'), {**record, 'error': str(error), 'utc': now()})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['admission', 'weights'])
    parser.add_argument('--lane', type=Path, required=True)
    parser.add_argument('--expected', type=Path, required=True)
    args = parser.parse_args()
    run(args.stage, args.lane, json.loads(args.expected.read_text()))
