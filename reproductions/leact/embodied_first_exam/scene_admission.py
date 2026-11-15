"""Audit initial scene identities before admitting a teacher development pilot.

Only immutable scene metadata is inspected here; oracle programs/trajectories
are never read or exported. This audit does not claim coverage for other suites.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess

from .report import ROOT, collection, read, write

SUITES = {'metaworldplus', 'robocasa', 'embodied-safety-exam'}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def scene_identity(reference, row):
    import numpy as np
    directory = reference / row['path']
    if hashlib.sha256((directory / 'task.toml').read_bytes()).hexdigest() != row['task_toml_sha256']:
        raise ValueError('Task source changed: ' + row['task'])
    if (directory / 'DIGEST').read_text().strip().removeprefix('sha256:') != row['upstream_digest']:
        raise ValueError('Task digest changed: ' + row['task'])
    sim = directory / 'environment/sim'
    evidence = {}
    if row['suite'] in {'metaworldplus', 'robocasa'}:
        path = sim / 'instance.npz'
        with np.load(path, allow_pickle=False) as data:
            # Hash physical arrays, not ZIP timestamps, task names or language.
            keys = sorted(k for k in data.files if k == 'integration' or k == 'xml' or k.startswith('model_'))
            if 'integration' not in keys or not any(k.startswith('model_') for k in keys):
                raise ValueError('No frozen physical state: ' + row['task'])
            physical = {k: {'shape': data[k].shape, 'dtype': str(data[k].dtype),
                            'sha256': hashlib.sha256(data[k].tobytes()).hexdigest()} for k in keys}
            published = str(data['instance_sha'].item())
        identity = digest(physical)
        evidence[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        method = 'canonical initial MuJoCo integration, physical model arrays and XML'
    elif row['suite'] == 'embodied-safety-exam':
        path = sim / 'instance.json'; data = read(path)
        published = data['state_sha']
        if len(published) != 64 or any(c not in '0123456789abcdef' for c in published):
            raise ValueError('Invalid ESE frozen initial-state hash')
        identity = published
        evidence[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        method = 'upstream frozen initial state_sha; native instance_mismatch verifier must pass'
    elif row['suite'] == 'dexhand':
        path = sim / 'instance.json'; data = read(path)
        descriptive = {'id', 'title', 'description', 'instruction', 'difficulty', 'category', 'checker', 'modeling', 'mechanism'}
        physical = {k: v for k, v in data.items() if k not in descriptive}
        if not any(k.startswith('initial_') or k == 'reset_state' for k in physical):
            raise ValueError('DexHand lacks an initial-state specification: ' + row['task'])
        source = sim / 'scene.py'
        identity = digest({'physical_configuration': physical, 'scene_builder': hashlib.sha256(source.read_bytes()).hexdigest()})
        evidence = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (path, source)}
        published = None
        method = 'frozen scene builder and physical configuration including initial robot targets/state'
    else:
        raise ValueError('Unaudited suite: ' + row['suite'])
    return {'task': row['task'], 'suite': row['suite'], 'scene_sha256': identity,
            'upstream_instance_sha256': published, 'method': method, 'source_files': evidence,
            'task_toml_sha256': row['task_toml_sha256'], 'upstream_digest': row['upstream_digest']}


def audit(reference, output):
    manifest = read(ROOT / 'generalization-v2/manifest.json')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=reference, text=True).strip()
    if revision != manifest['upstream_revision']:
        raise ValueError('Wrong EFE source revision')
    rows = read(ROOT / 'generalization-v2/inventory.json')
    split = read(ROOT / 'generalization-v2/splits.json')
    by_name = {r['task']: r for r in rows}
    protected = set(split['development'] + split['official_core_evaluation'] + split['simulator_holdout_reserve'])
    # All canonical modes are checked; split validation already excludes their
    # alternate-mode twins at the base-task level.
    reference_scenes = defaultdict(list)
    protected_receipts = []
    for name in sorted(protected):
        row = by_name[name]
        if row['suite'] not in SUITES:
            continue
        receipt = scene_identity(reference, row)
        reference_scenes[row['suite'], receipt['scene_sha256']].append(name)
        protected_receipts.append(receipt)
    _, collection_data, _ = collection()
    accepted, excluded = [], []
    for trial in collection_data['trials']:
        if trial['partition'] != 'train_candidates' or trial['status'] != 'graded' or trial['score'] != 1:
            continue
        row = by_name[trial['task']]
        if trial['task'] not in split['train_candidates'] or row['simulator_runtime'] in split['held_out_runtimes']:
            raise ValueError('Protected task entered the training candidate list')
        if row['suite'] not in SUITES:
            excluded.append({'task': row['task'], 'suite': row['suite'], 'trial': trial,
                             'reason': 'Suite initial-state canonicalization not validated for this pilot; retain episode for later admission.'})
            continue
        receipt = scene_identity(reference, row)
        collisions = reference_scenes[row['suite'], receipt['scene_sha256']]
        record = {**receipt, 'trial': trial, 'partition': 'train_candidates', 'collisions': collisions,
                  'semantic_task': row['semantic_task'], 'simulator_runtime': row['simulator_runtime']}
        (excluded if collisions else accepted).append(record)
    result = {'schema_version': 1, 'upstream_revision': revision,
              'split_manifest_sha256': hashlib.sha256((ROOT/'generalization-v2/manifest.json').read_bytes()).hexdigest(),
              'checked_suites': sorted(SUITES), 'protected_scene_count': len(protected_receipts),
              'protected_scene_receipts': protected_receipts, 'eligible_episodes': accepted, 'excluded_episodes': excluded,
              'teacher_episodes_eligible_for_trace_review': len(accepted),
              'scope': 'Initial scene audit for a three-suite development pilot. DexHand is retained but deferred until its heterogeneous initial-state formats are validated. Native verifier instance checks and trace integrity remain required. No oracle solutions read.'}
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and read(output) != result:
        raise ValueError('Frozen scene audit changed; use a new version')
    write(output, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.reference, args.output)
    print(json.dumps({k: result[k] for k in ['checked_suites', 'protected_scene_count', 'teacher_episodes_eligible_for_trace_review']}))
