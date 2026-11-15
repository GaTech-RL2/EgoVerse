"""Collect small experiment records through an active OSMO task, with checksums.

Requires pexpect. Model weights and caches are never transferred. Metrics and
generation rows are checksum-collected; generation rows use bounded incremental
chunks. The gate-only mode captures the small diagnostic shards and exits after
the gate result. Changed metadata is retained by content hash. The tracking sync
can release acknowledged rollout buffers after storing their records on ITSS.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import re
import shlex
import time
import zlib

REMOTE = r'''
from pathlib import Path
import base64, hashlib, json, time
root = Path(ROOT)
deadline = time.monotonic() + DURATION
seen = {}
offsets = {}
selection_stamps = {}
role = globals().get('REPRODUCTION_ROLE')
astra = globals().get('ASTRA', False)
coldstart = globals().get('COLDSTART', False)
builderbench = globals().get('BUILDERBENCH_ROLE')
max_rollout_bytes = globals().get('MAX_ROLLOUT_BYTES', 0)
patterns = [
    'full_protocol.json', 'base_model.json', 'reasoning_anchor.json', 'results.json', 'comparison.json',
    'comparison.csv', 'coldstart/manifest.json', 'coldstart/train.yaml',
    'coldstart/training_complete.json', 'coldstart/checkpoints/trainer_state.json',
    'coldstart/checkpoints/trainer_log.jsonl',
    'coldstart/checkpoints/checkpoint-*/trainer_state.json',
    'coldstart/checkpoints/weight_retention.json', 'coldstart_check/result.json',
    'eval/*/summary.json', 'eval/*/seed-*.json',
    'eval/bc_epochs/*/summary.json', 'eval/bc_epochs/*/seed-*.json',
    'leact/round-*/sft/manifest.json', 'exit/round-*/sft/manifest.json',
    'leact/round-*/sft/training_complete.json', 'exit/round-*/sft/training_complete.json',
    'leact/round-*/sft/checkpoints/trainer_state.json',
    'exit/round-*/sft/checkpoints/trainer_state.json',
    'leact/round-*/sft/checkpoints/trainer_log.jsonl',
    'exit/round-*/sft/checkpoints/trainer_log.jsonl',
    'leact/round-*/sft/train.yaml', 'exit/round-*/sft/train.yaml',
    'leact/round-*/eval/*/summary.json', 'exit/round-*/eval/*/summary.json',
]
if GATE_ONLY:
    patterns = ['coldstart_check/result.json', 'coldstart_check/generation/shard-*.jsonl',
                'coldstart_check/generation/shard-*.meta.json']
elif INCLUDE_ROLLOUTS:
    patterns += ['coldstart_check/generation/shard-*.meta.json',
                 'eval/**/shard-*.meta.json', 'leact/round-*/generation/shard-*.meta.json',
                 'exit/round-*/generation/shard-*.meta.json']
if PROMPT_DIAGNOSTIC:
    patterns = ['protocol.json', 'runtime.json', 'result.json', 'complete.json', '*/*/result.json',
                '*/*/sft-*/manifest.json', '*/*/sft-*/data.json', '*/*/sft-*/selection.jsonl',
                '*/*/generation/shard-*.meta.json']
if role:
    patterns = [role+'/'+p for p in patterns]
    patterns += [role+'/progress.json', role+'/runtime.json', role+'/complete.json',
                 role+'/bc/**/comparison.json', role+'/bc/results.json',
                 role+'/bc/**/summary.json', role+'/bc/**/seed-*.json',
                 role+'/bc/**/shard-*.meta.json',
                 role+'/*/round-*/sft/checkpoints/weight_retention.json',
                 role+'/*/round-*/sft/training_verification.json',
                 role+'/*/round-*/sft/checkpoints/checkpoint-*/trainer_state.json',
                 role+'/*/round-*/eval/*/seed-*.json',
                 'results.json', 'comparison.json', 'comparison.csv']
    if role == 'leact':
        patterns += ['anchor/progress.json', 'anchor/ready.json', 'anchor/training_verification.json',
                     'anchor/sft/manifest.json', 'anchor/sft/train.yaml',
                     'anchor/sft/training_complete.json', 'anchor/sft/selection.jsonl',
                     'anchor/sft/data.json', 'anchor/sft/checkpoints/trainer_state.json',
                     'anchor/sft/checkpoints/trainer_log.jsonl',
                     'anchor/sft/checkpoints/weight_retention.json']
if astra:
    patterns = ['protocol.json', 'progress.json', 'results.json', 'complete.json',
                'astra_*/round-*/sft/manifest.json', 'astra_*/round-*/sft/train.yaml',
                'astra_*/round-*/sft/training_complete.json', 'astra_*/round-*/sft/training_verification.json',
                'astra_*/round-*/sft/checkpoints/trainer_state.json',
                'astra_*/round-*/sft/checkpoints/trainer_log.jsonl',
                'astra_*/round-*/sft/checkpoints/weight_retention.json',
                'astra_*/round-*/eval/*/summary.json', 'astra_*/round-*/eval/*/seed-*.json',
                'astra_*/round-*/assisted_eval/summary.json', 'astra_*/round-*/assisted_eval/seed-*/result.json',
                'astra_*/round-*/generation/shard-*.meta.json', 'astra_*/round-*/eval/**/shard-*.meta.json',
                'astra_*/round-*/assisted_eval/**/shard-*.meta.json']
if coldstart:
    patterns = ['protocol.json', 'progress.json', 'results.json', 'comparison.json', 'comparison.csv',
                'complete.json', 'anchor-ready.json', '**/sft/manifest.json', '**/sft/train.yaml',
                '**/sft/training_complete.json', '**/sft/training_verification.json',
                '**/sft/checkpoints/trainer_state.json', '**/sft/checkpoints/trainer_log.jsonl',
                '**/sft/checkpoints/checkpoint-*/trainer_state.json', '**/sft/checkpoints/weight_retention.json',
                '**/eval/**/summary.json', '**/eval/**/seed-*.json', '**/shard-*.meta.json']
if builderbench:
    patterns = ['*-protocol.json', '*-progress.json', '*-complete.json', 'anchor-ready.json',
                'ready/*.json', 'training/**/sft/manifest.json', 'training/**/sft/train.yaml',
                'training/**/sft/training_complete.json', 'training/**/sft/training_verification.json',
                'training/**/sft/no_admissions.json', 'training/**/sft/checkpoints/trainer_state.json',
                'training/**/sft/checkpoints/trainer_log.jsonl',
                'evaluations/*/manifest.json', 'evaluations/*/metrics.json', 'evaluations/*/speed.json',
                'evaluations/*/preview.json', 'iteration-corpus/manifest.json', 'iteration-corpus/ready.json',
                'runtime-*.json', 'simulator-replay-*.json', 'fixed-report.json']
def emit(relative, data):
    digest = hashlib.sha256(data).hexdigest()
    record = {'relative': relative, 'sha256': digest,
              'bytes': len(data), 'base64': base64.b64encode(data).decode()}
    print('LEACT_FILE ' + json.dumps(record), flush=True)
    seen[relative] = digest
while True:
    for path in sorted({p for pattern in patterns for p in root.glob(pattern)}):
        if not path.is_file() or path.is_symlink() or path.stat().st_size > 2_000_000:
            continue
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        relative = str(path.relative_to(root))
        if seen.get(relative) == digest:
            continue
        # Skip incomplete JSON writes; the next poll will collect them.
        if path.suffix == '.json':
            try:
                json.loads(data)
            except (ValueError, UnicodeDecodeError):
                continue
        emit(relative, data)
    if role or astra or coldstart:
        selection_pattern = ('**/sft/selection.jsonl' if coldstart else
                             'astra_*/round-*/sft/selection.jsonl' if astra else role+'/*/round-*/sft/selection.jsonl')
        for path in root.glob(selection_pattern):
            if not (path.parent/'manifest.json').exists():
                continue
            stamp = (path.stat().st_mtime_ns, path.stat().st_size)
            if selection_stamps.get(str(path)) == stamp:
                continue
            admission = {}
            for line in path.read_text().splitlines():
                row = json.loads(line)
                admission[row['state_id']] = {
                    'baseline_valid': row.get('baseline_valid'),
                    'selected': [{'candidate_id': c['candidate_id'],
                                  'admitted_to_sft': c.get('admitted_to_sft', False)}
                                 for c in row['selected']],
                }
            emit(str(path.with_name('admission.json').relative_to(root)),
                 json.dumps(admission, sort_keys=True).encode())
            selection_stamps[str(path)] = stamp
    if coldstart:
        for path in root.glob('**/sft/coldstart-admission.jsonl'):
            if not (path.parent/'manifest.json').exists():
                continue
            stamp = (path.stat().st_mtime_ns, path.stat().st_size)
            if selection_stamps.get(str(path)) == stamp:
                continue
            admission = {row['state_id']: {'selected': [c['candidate_id'] for c in row['candidates'] if c['admitted_to_sft']],
                                          'retained': row['retained']}
                         for row in map(json.loads, path.read_text().splitlines())}
            emit(str(path.with_name('admission.json').relative_to(root)), json.dumps(admission, sort_keys=True).encode())
            selection_stamps[str(path)] = stamp
    if INCLUDE_ROLLOUTS and not GATE_ONLY:
        raw_patterns = ['coldstart_check/generation/shard-*.jsonl', 'eval/**/shard-*.jsonl',
                        'leact/round-*/generation/shard-*.jsonl', 'exit/round-*/generation/shard-*.jsonl']
        if PROMPT_DIAGNOSTIC:
            raw_patterns = ['*/*/generation/shard-*.jsonl']
        if role:
            raw_patterns = [role+'/'+p for p in raw_patterns]
            raw_patterns += [role+'/bc/**/shard-*.jsonl']
        if astra:
            raw_patterns = ['astra_*/round-*/generation/shard-*.jsonl',
                            'astra_*/round-*/eval/**/shard-*.jsonl',
                            'astra_*/round-*/assisted_eval/**/shard-*.jsonl']
        if coldstart:
            raw_patterns = ['self_generation/shard-*.jsonl', '**/generation/shard-*.jsonl', '**/eval/**/shard-*.jsonl']
        if builderbench:
            raw_patterns = ['training/*/generation/shard-*.jsonl']
        for path in sorted({p for pattern in raw_patterns for p in root.glob(pattern)}):
            if path.is_symlink() or not path.is_file():
                continue
            relative = str(path.relative_to(root))
            offset = offsets.get(relative, 0)
            if path.stat().st_size < offset:
                raise RuntimeError('An append-only generation shard shrank: '+relative)
            # Transfer completed JSONL rows in bounded chunks, including large
            # shards. Subsequent polls send only newly appended rows.
            with path.open('rb') as stream:
                stream.seek(offset)
                while True:
                    start = stream.tell()
                    if max_rollout_bytes and start >= max_rollout_bytes:
                        break
                    lines, size = [], 0
                    while size < 500_000:
                        position = stream.tell()
                        line = stream.readline(2_000_001)
                        if not line or not line.endswith(b'\n'):
                            if len(line) > 2_000_000:
                                raise RuntimeError('A generation record exceeds the transfer limit')
                            stream.seek(position)
                            break
                        json.loads(line)
                        lines.append(line)
                        size += len(line)
                    if not lines:
                        break
                    emit(relative+'.parts/'+format(start, '012d')+'.jsonl', b''.join(lines))
                    offsets[relative] = stream.tell()
    if PROMPT_DIAGNOSTIC and 'complete.json' in seen:
        (root/'collection-exported.json').write_text(json.dumps({'files_sent':len(seen)}))
        break
    if role and role+'/complete.json' in seen:
        (root/role/'collection-exported.json').write_text(json.dumps({'files_sent':len(seen)}))
        break
    if (astra or coldstart) and 'complete.json' in seen:
        (root/'collection-exported.json').write_text(json.dumps({'files_sent':len(seen)}))
        break
    if builderbench and builderbench+'-complete.json' in seen:
        (root/('collection-exported-'+builderbench+'.json')).write_text(json.dumps({'files_sent':len(seen)}))
        break
    if time.monotonic() >= deadline or (GATE_ONLY and 'coldstart_check/result.json' in seen):
        break
    time.sleep(INTERVAL)
print('LEACT_COLLECT_DONE', flush=True)
time.sleep(2)
'''


def main() -> None:
    # The transport runs on the submitting machine. GPU tests can validate the
    # portable remote collector without installing a local PTY client there.
    import pexpect

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workflow')
    parser.add_argument('--task', default='leact')
    parser.add_argument('--experiment-root', required=True)
    parser.add_argument('--source-root', required=True, help='Source directory containing emimic/bin/activate')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--watch-seconds', type=int, default=0)
    parser.add_argument('--interval', type=int, default=10)
    parser.add_argument('--gate-only', action='store_true',
                        help='Collect the small coldstart gate shards and stop after its result')
    parser.add_argument('--prompt-diagnostic', action='store_true',
                        help='Collect all matched prompt arms and stop after completion')
    parser.add_argument('--no-rollouts', action='store_true',
                        help='Collect only metrics/manifests; by default complete generation rows are also streamed')
    parser.add_argument('--reproduction-role', choices=['leact', 'exit'])
    parser.add_argument('--astra', action='store_true', help='Collect the queued two-arm Astra extension')
    parser.add_argument('--coldstart', action='store_true', help='Collect the frontier or self-scaling coldstart study')
    parser.add_argument('--builderbench-role', choices=['train', 'evaluate', 'fixed'])
    parser.add_argument('--max-rollout-bytes', type=int, default=0,
                        help='Bound the live preview bytes per shard; complete raw shards remain on shared storage')
    args = parser.parse_args()
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', args.workflow):
        parser.error('Invalid workflow name')
    if args.watch_seconds < 0 or not 1 <= args.interval <= 60:
        parser.error('Invalid watch duration or poll interval')
    if sum([args.prompt_diagnostic, args.gate_only, bool(args.reproduction_role), args.astra, args.coldstart, bool(args.builderbench_role)]) > 1:
        parser.error('Choose one diagnostic collection mode')
    if args.max_rollout_bytes < 0:
        parser.error('Invalid rollout preview limit')
    for value in [args.experiment_root, args.source_root]:
        if not any(value.startswith(prefix) for prefix in (
                '/mnt/amlfs-07/shared/egoverse/rpunamiya/leact/',
                '/mnt/amlfs-03/home/rpunamiya/leact/')):
            parser.error('Expected this reproduction\'s shared experiment/source path')
    prefix = (f'ROOT={args.experiment_root!r}\nDURATION={args.watch_seconds}\n'
              f'INTERVAL={args.interval}\nGATE_ONLY={args.gate_only!r}\n'
              f'PROMPT_DIAGNOSTIC={args.prompt_diagnostic!r}\n'
              f'REPRODUCTION_ROLE={args.reproduction_role!r}\n'
              f'ASTRA={args.astra!r}\n'
              f'COLDSTART={args.coldstart!r}\n'
              f'BUILDERBENCH_ROLE={args.builderbench_role!r}\n'
              f'MAX_ROLLOUT_BYTES={args.max_rollout_bytes}\n'
              f'INCLUDE_ROLLOUTS={not args.no_rollouts!r}\n')
    # OSMO encodes --entry in its exec URL. Compress the read-only collector to
    # keep the URL below the gateway's limit as artifact patterns grow.
    packed = base64.b64encode(zlib.compress((prefix + REMOTE).encode(), level=9)).decode()
    script = "import base64,zlib;exec(zlib.decompress(base64.b64decode(" + repr(packed) + ")))"
    remote = ('set -e\ncd ' + shlex.quote(args.source_root)
              + "\nsource emimic/bin/activate\npython -u -c " + shlex.quote(script))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / 'collection-policy.json').write_text(json.dumps({
        'workflow': args.workflow, 'complete_raw_artifacts': args.experiment_root,
        'max_preview_bytes_per_shard': args.max_rollout_bytes,
        'coverage': 'bounded live preview' if args.max_rollout_bytes else 'complete available rows',
    }, indent=2)+'\n')
    child = pexpect.spawn('osmo', ['workflow', 'exec', args.workflow, args.task,
                                  '--entry', 'bash -c ' + shlex.quote(remote)],
                          encoding='utf-8', timeout=30, maxread=65536, echo=False,
                          dimensions=(40, 200))
    count = 0
    completed = False
    deadline = time.monotonic() + args.watch_seconds + 120
    try:
        with (output / 'collection.jsonl').open('a') as audit:
            while True:
                if time.monotonic() > deadline:
                    raise TimeoutError('Result collector exceeded its bounded local deadline')
                event = child.expect([r'LEACT_FILE (\{[^\r\n]+\})\r?\n',
                                      'LEACT_COLLECT_DONE', pexpect.EOF, pexpect.TIMEOUT])
                if event == 0:
                    record = json.loads(child.match.group(1))
                    data = base64.b64decode(record.pop('base64'), validate=True)
                    if hashlib.sha256(data).hexdigest() != record['sha256'] or len(data) != record['bytes']:
                        raise ValueError('Remote record checksum/length mismatch')
                    relative = Path(record['relative'])
                    destination = (output / 'latest' / relative).resolve()
                    if relative.is_absolute() or not destination.is_relative_to(output / 'latest'):
                        raise ValueError('Invalid remote record path')
                    history = output / 'history' / record['sha256'] / relative
                    for path in [history, destination]:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        temporary = path.with_suffix(path.suffix + '.partial')
                        temporary.write_bytes(data)
                        temporary.replace(path)
                    record.update(workflow=args.workflow, collected_unix=time.time())
                    audit.write(json.dumps(record) + '\n')
                    audit.flush()
                    count += 1
                    print(json.dumps(record), flush=True)
                elif event == 1:
                    completed = True
                    child.expect(pexpect.EOF, timeout=20)
                    break
                elif event == 2:
                    print(str(child.before)[-1500:], flush=True)
                    break
        child.close()
        summary = {'workflow': args.workflow, 'records_collected': count,
                   'watch_duration_completed': completed, 'osmo_exit_status': child.exitstatus,
                   'note': 'Task termination may end a watch before its duration; records already received remain verified.'}
        (output / 'collection-status.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps(summary), flush=True)
        if count == 0:
            raise RuntimeError('No experiment records received; confirm the task and its environment are running')
    finally:
        if child.isalive():
            child.close(force=True)


if __name__ == '__main__':
    main()
