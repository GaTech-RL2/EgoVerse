"""Publish changed experiment snapshots over authenticated SSH; no public writes."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

from .export import snapshot
from .traces import records, sha


def ssh_command(args, mode):
    connection = ['ssh', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=10']
    if args.identity.is_file():
        return connection+['-S', 'none', '-o', 'IdentitiesOnly=yes', '-i', str(args.identity), args.host, 'leact-import-'+mode]
    command = ('cd "$HOME/leact-tracker" && . emimic/bin/activate && '
               'python current/server.py '+('import-records' if mode == 'records' else 'import')+
               ' --db data/experiments.sqlite3 --snapshot -')
    return connection+['-S', str(args.control), args.host, command]


def publish_records(args):
    journal_path = args.state.with_suffix('.records.json')
    journal = json.loads(journal_path.read_text()) if journal_path.exists() else {}
    destination = args.host
    if journal.get('host') != destination:
        journal = {'host': destination, 'records': {}}
    changed, payloads, size = {}, [], 0

    def flush():
        if not payloads:
            return
        result = subprocess.run(
            ssh_command(args, 'records'),
            input=b''.join(payloads), capture_output=True, timeout=45, check=True)
        receipt = json.loads(result.stdout)
        expected = hashlib.sha256(''.join(d+'\n' for d in changed.values()).encode()).hexdigest()
        if receipt['records'] != len(payloads) or receipt['batch_sha256'] != expected:
            raise ValueError('Record import count/checksum did not match')
        journal['records'].update(changed)
        temporary = journal_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(journal)+'\n')
        temporary.replace(journal_path)
        print(json.dumps({'trace_import': receipt}), flush=True)
        changed.clear()
        payloads.clear()

    observed_chunks = []
    for value in records(observed_chunks=observed_chunks):
        key = '/'.join([value['kind'], value.get('run', value['oracle_sha256']), value['state_id'], str(value.get('seed', ''))])
        digest = sha(value)
        if journal['records'].get(key) == digest or changed.get(key) == digest:
            continue
        if key in changed:
            flush()
            size = 0
        line = (json.dumps(value, allow_nan=False)+'\n').encode()
        if size+len(line) > 1_000_000 and payloads:
            flush()
            size = 0
        changed[key] = digest
        payloads.append(line)
        size += len(line)
    flush()
    if args.prune_uploaded_chunks:
        pruned = 0
        for path, digest in observed_chunks:
            # SQLite has acknowledged all records above. Release only unchanged
            # transfer buffers, keeping original generations on Lustre and ITSS.
            if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == digest:
                root = next(p for p in path.parents if p.name == 'latest')
                history = root.parent/'history'/digest/path.relative_to(root)
                path.unlink()
                if history.exists() and hashlib.sha256(history.read_bytes()).hexdigest() == digest:
                    history.unlink()
                pruned += 1
        if pruned:
            print(json.dumps({'acknowledged_transfer_chunks_removed': pruned}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='rpunamiya@yam-coverage-rpunamiya.nvidia.com')
    parser.add_argument('--control', type=Path, default=Path('/tmp/leact-itss-ssh-rpunamiya/control'))
    parser.add_argument('--identity', type=Path, default=Path.home()/'.ssh/leact_tracker_itss')
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--interval', type=int, default=30)
    parser.add_argument('--prune-uploaded-chunks', action='store_true',
                        help='Release acknowledged rollout transfer buffers; originals remain on Lustre and ITSS')
    args = parser.parse_args()
    if args.interval < 5:
        parser.error('Polling interval must be at least five seconds')
    args.state.parent.mkdir(parents=True, exist_ok=True)
    destination = {'host': args.host, 'control': str(args.control)}
    last = None
    if args.state.exists():
        state = json.loads(args.state.read_text())
        if state.get('destination') == destination:
            last = state.get('sha256')
    while True:
        try:
            value = snapshot()
            payload = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
            digest = hashlib.sha256(payload).hexdigest()
            if digest != last:
                result = subprocess.run(
                    ssh_command(args, 'snapshot'),
                    input=payload, capture_output=True, timeout=45, check=True)
                imported = json.loads(result.stdout)
                if imported['sha256'] != digest:
                    raise ValueError('Tracking host returned a different snapshot checksum')
                state = {'destination': destination, **imported,
                         'synced_utc': datetime.now(timezone.utc).isoformat(),
                         'runs': len(value['runs']), 'live_training_series': len(value['live_training'])}
                temporary = args.state.with_suffix('.tmp')
                temporary.write_text(json.dumps(state, indent=2)+'\n')
                temporary.replace(args.state)
                last = digest
                print(json.dumps(state), flush=True)
            publish_records(args)
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            print(json.dumps({'time': datetime.now(timezone.utc).isoformat(),
                              'error': str(exc), 'uploaded': False}), flush=True)
            if not args.watch:
                raise
        if not args.watch:
            return
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
