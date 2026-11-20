"""Build and submit disjoint EFE teacher shards under the authorized node cap."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import yaml

from .compute_budget import require_capacity
from ..embodied_first_exam.report import BASE, ROOT, read, write

BUILD_ROOT = BASE / 'cluster/.build'
PILOT = BUILD_ROOT / 'efe-astra-pilot-20261026-v2'


def command(args, **kwargs):
    result = subprocess.run(args, capture_output=True, text=True, timeout=150, **kwargs)
    if result.returncode:
        if kwargs.get('cwd'):
            (Path(kwargs['cwd'])/'last-command-error.log').write_text(result.stdout+result.stderr)
        raise RuntimeError(f'{args[1:3]} failed ({result.returncode}): '+(result.stderr+result.stdout)[-1600:])
    return json.loads(result.stdout[result.stdout.index('{'):])


def prepare():
    original = read(PILOT / 'spec.json')
    splits = read(ROOT / 'generalization-v2/splits.json')
    inventory = read(ROOT / 'generalization-v2/inventory.json')
    excluded = {r['task'] for r in original['tasks']}
    shards = [[], []]
    counts = {'robocasa': 6, 'dexhand': 4, 'embodied-safety-exam': 6, 'metaworldplus': 6}
    for suite, count in counts.items():
        rows = [dict(r, partition='train_candidates') for r in inventory
                if r['task'] in splits['train_candidates'] and r['task'] not in excluded
                and r['suite'] == suite and not r['resources'].get('gpu_required')]
        rows.sort(key=lambda r: hashlib.sha256(('efe-scale-20261026-v1/' + r['task']).encode()).hexdigest())
        if len(rows) < count:
            raise ValueError(f'Insufficient disjoint training tasks in {suite}')
        for i, row in enumerate(rows[:count]):
            shards[i % 2].append(row)
    chosen = [r['task'] for shard in shards for r in shard]
    if len(chosen) != len(set(chosen)) or set(chosen) & excluded:
        raise ValueError('Teacher shards overlap another task owner')
    builds = []
    for index, rows in enumerate(shards, 1):
        # Cycle suites so early results cover multiple environments.
        groups = {suite: [r for r in rows if r['suite'] == suite] for suite in counts}
        rows = [group[i] for i in range(max(map(len, groups.values())))
                for group in groups.values() if i < len(group)]
        build = BUILD_ROOT / f'efe-astra-scale-20261026-{index}'
        build.mkdir(exist_ok=True)
        if (build / 'source-manifest.json').exists():
            builds.append(build)
            continue
        spec = {**original, 'tasks': rows,
                'archive_root': '/mnt/amlfs-03/home/rpunamiya/leact/studies/efe-astra-scale-20261026-v1',
                'local_results_directory': f'astra-workers/scale-{index}',
                'purpose': 'Disjoint expansion of frozen training candidates for Astra warmstart; no development/core targets.',
                'task_ownership': {'shard': index, 'total_shards': 2, 'selection_seed': 'efe-scale-20261026-v1',
                                   'existing_pilot_excluded': sorted(excluded)}}
        write(build / 'spec.json', spec)
        for source, target in [('efe_native_pilot.py', 'runtime.py'), ('efe_astra_pilot.py', 'pilot.py'),
                               ('efe_budget_gateway.py', 'gateway.py')]:
            shutil.copyfile(BASE / 'cluster' / source, build / target)
        shutil.copyfile(PILOT / 'bootstrap.sh', build / 'bootstrap.sh')
        workflow = yaml.safe_load((PILOT / 'workflow.yaml').read_text())
        workflow['workflow']['name'] = f'leact-efe-astra-scale-{index}'
        (build / 'workflow.yaml').write_text(yaml.safe_dump(workflow, sort_keys=False))
        write(build / 'source-manifest.json', {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in build.iterdir() if p.name in {'runtime.py','pilot.py','gateway.py','bootstrap.sh','spec.json','workflow.yaml'}})
        builds.append(build)
    write(ROOT / 'parallel-collection-plan.json', {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'additional_train_tasks': len(chosen),
        'existing_pilot_tasks': len(original['tasks']), 'teacher_workers': 3,
        'train_partition_only': True, 'no_duplicate_task_ownership': True,
        'suite_counts': counts, 'builds': [str(b) for b in builds],
        'admission': 'Successful train traces still require scene/provenance checks before SFT.'})
    return builds


def submit(build, *, priority=None, pool='groot-h100-01', nodes=1):
    priority=priority or read(ROOT/'osmo-priority-policy.json',{}).get('current_priority','HIGH')
    if priority not in {'HIGH','NORMAL','LOW'}:raise ValueError('Unsupported OSMO priority: '+str(priority))
    authorization=read(ROOT/'compute-authorization.json',{})
    if type(nodes) is not int or not 1<=nodes<=authorization.get('max_nodes_per_job',2):
        raise ValueError('Requested node count exceeds the authorized per-job bound')
    with (BUILD_ROOT / 'builderbench-submission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (build / 'submission.json').exists():
            return read(build / 'submission.json')
        if (build / 'submission-intent.json').exists():
            raise RuntimeError('Reconcile prior submission intent before retrying')
        for name, digest in read(build / 'source-manifest.json').items():
            if hashlib.sha256((build/name).read_bytes()).hexdigest()!=digest:
                raise ValueError('Frozen launch file checksum mismatch: '+name)
        active = command(['osmo', 'workflow', 'list', '--name', 'leact', '--status',
                          'RUNNING', 'PENDING', 'WAITING', '--count', '1000', '--format-type', 'json'])['workflows']
        require_capacity(active, additional_nodes=nodes)
        validation = subprocess.run(['osmo', 'workflow', 'validate', 'workflow.yaml', '--pool', pool],
                                    cwd=build, capture_output=True, text=True, timeout=90)
        (build / 'validation.log').write_text(validation.stdout + validation.stderr)
        if validation.returncode:
            raise RuntimeError('OSMO validation failed: '+(validation.stdout+validation.stderr)[-1800:])
        write(build / 'submission-intent.json', {'utc': datetime.now(timezone.utc).isoformat(), 'nodes': nodes, 'priority':priority, 'pool':pool})
        receipt = command(['osmo', 'workflow', 'submit', 'workflow.yaml', '--pool', pool,
                           '--priority', priority, '--format-type', 'json'], cwd=build)
        write(build / 'submission.json', receipt)
        reservations=read(ROOT/'node-reservations.json',{})
        reservations[receipt['name']]=nodes
        write(ROOT/'node-reservations.json',reservations)
        return receipt


def monitor(build):
    with (build / 'monitor.log').open('a') as log:
        p = subprocess.Popen([sys.executable, '-u', '-m', 'reproductions.leact.cluster.monitor_efe_astra',
                              '--build', str(build)], cwd=BASE.parents[1],
                             stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write(build / 'monitor-launch.json', {'pid': p.pid, 'utc': datetime.now(timezone.utc).isoformat()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args()
    for build in prepare():
        if args.submit:
            receipt = submit(build)
            monitor(build)
            print(json.dumps({'workflow': receipt['name'], 'build': str(build)}), flush=True)
        else:
            print(build)


if __name__ == '__main__':
    main()
