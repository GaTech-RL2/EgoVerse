"""Prepare task-, variant-, and simulator-held-out evaluation cohorts.

All official core instances remain evaluation-only. These labels describe
planned training exposure and must be recomputed from admitted training data.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import json
from pathlib import Path
import re
import subprocess

from .prepare import REVISION, encode, sha

SEED = 'efe-generalization-20261025-v2'
HELD_OUT_RUNTIMES = frozenset({'sapien', 'genesis'})
RUNTIMES = {
    'mujoco': ('mujoco', 'mujoco'), 'robosuite': ('robosuite', 'mujoco'),
    'autobio': ('mujoco', 'mujoco'), 'bigym2': ('mujoco', 'mujoco'),
    'dexjoco': ('mujoco', 'mujoco'), 'humanoidbench': ('mujoco', 'mujoco'),
    'kinder': ('mujoco', 'mujoco'), 'molmospaces': ('mujoco', 'mujoco'),
    'myosuite': ('mujoco', 'mujoco'), 'roboarc': ('mujoco', 'mujoco'),
    'vlabench': ('mujoco', 'mujoco'),
    'ambench': ('isaac_sim', 'physx'), 'behavior': ('isaac_sim', 'physx'),
    'ispatialgym': ('isaac_sim', 'physx'), 'labutopia': ('isaac_sim', 'physx'),
    'maniguard': ('isaac_sim', 'physx'), 'robodojo': ('isaac_sim', 'physx'),
    'robolab': ('isaac_sim', 'physx'), 'yamlab': ('isaac_sim', 'physx'),
    'maniskill': ('sapien', 'physx'), 'robotwin': ('sapien', 'physx'),
    'robowits': ('genesis', 'genesis'),
}
# Checked against these suites' task metadata and README descriptions. Other
# upstream family fields are not assumed to identify the same semantic task.
VARIANT_SUITES = {'roboarc', 'maniguard', 'yamlab', 'roboengineer'}


def task_identity(row):
    suite, base = row['suite'], row['base_task']
    if suite in VARIANT_SUITES:
        return suite + '/' + row['family']
    if suite == 'libero-max':
        # Its upstream "family" groups different goals by perturbation type.
        perturbation = row['family'].rsplit('/', 1)[-1].replace('_', '-')
        stem = re.sub(r'-i\d+$', '', base)
        if not stem.endswith('-' + perturbation):
            raise ValueError(f'Cannot identify LIBERO-MAX semantic task: {base}')
        return suite + '/' + stem[: -len(perturbation) - 1]
    return suite + '/' + base


def enrich(rows):
    result = copy.deepcopy(rows)
    for row in result:
        try:
            runtime, engine = RUNTIMES[row['simulator']]
        except KeyError as error:
            raise ValueError('Unaudited simulator taxonomy: ' + str(row['simulator'])) from error
        row['simulator_runtime'] = runtime
        row['physics_engine'] = engine
        row['semantic_task'] = task_identity(row)
        row['task_group'] = row['semantic_task'] + '|' + row['mode']
        row['variant_identity_status'] = ('Documented task family; scene identity audit still required'
            if row['suite'] in VARIANT_SUITES or row['suite'] == 'libero-max'
            else 'Conservative: distinct base task is a distinct semantic task')
    return result


def canonical_rows(rows):
    variants = defaultdict(list)
    suite_modes = defaultdict(Counter)
    group_modes = defaultdict(Counter)
    for row in rows:
        variants[row['base_task']].append(row)
        if row['official_default']:
            suite_modes[row['suite']][row['mode']] += 1
            group_modes[row['semantic_task']][row['mode']] += 1
    chosen = []
    for base, group in sorted(variants.items()):
        official = [r for r in group if r['official_default']]
        if official:
            if len(official) != 1:
                raise ValueError('Core task has several default modes: ' + base)
            chosen.append(official[0])
            continue
        available = [r for r in group if r['mode'] in ('standard', 'standard-realtime')]
        preferences = group_modes[group[0]['semantic_task']] or suite_modes[group[0]['suite']]
        chosen.append(min(available, key=lambda r: (-preferences[r['mode']], r['mode'] != 'standard', r['task'])))
    return chosen


def rank(row):
    return sha(f"{SEED}/{row['task']}".encode())


def classify(row, training):
    runtime_seen = any(t['simulator_runtime'] == row['simulator_runtime'] for t in training)
    engine_seen = any(t['physics_engine'] == row['physics_engine'] for t in training)
    same_suite = [t for t in training if t['suite'] == row['suite']]
    same_task = [t for t in same_suite if t['task_group'] == row['task_group']]
    if same_task:
        cohort, witnesses = 'same_sim_same_task_new_variant', same_task
    elif same_suite:
        cohort, witnesses = 'same_sim_new_task', same_suite
    elif runtime_seen:
        # A failed teacher collection can leave a suite with no admitted
        # examples. That does not make its shared runtime unseen.
        cohort, witnesses = 'unseen_suite_seen_simulator', []
    else:
        cohort, witnesses = 'held_out_simulator', []
    return dict(task=row['task'], base_task=row['base_task'], suite=row['suite'],
                semantic_task=row['semantic_task'], simulator_runtime=row['simulator_runtime'],
                physics_engine=row['physics_engine'], mode=row['mode'], cohort=cohort,
                runtime_seen_in_training=runtime_seen, physics_engine_seen_in_training=engine_seen,
                training_witnesses=sorted(t['task'] for t in witnesses)[:3],
                variant_scene_verified=False if same_task else None)


def build_split(rows):
    chosen = canonical_rows(rows)
    core = [r for r in chosen if r['official_default']]
    core_groups = {r['task_group'] for r in core}
    candidates = [r for r in chosen if not r['official_core'] and r['simulator_runtime'] not in HELD_OUT_RUNTIMES]
    groups = defaultdict(list)
    for row in candidates:
        groups[row['suite'], row['task_group']].append(row)
    train, dev = [], []
    novel_by_suite = defaultdict(list)
    for (suite, group), members in sorted(groups.items()):
        ordered = sorted(members, key=rank)
        if group in core_groups:
            # Keep at least one different, non-core instance as the train-side
            # witness. A separate variant can be used for development.
            count = max(1, round(len(ordered)*.2)) if len(ordered) >= 3 else 0
            dev.extend(ordered[:count]); train.extend(ordered[count:])
        else:
            novel_by_suite[suite].append((group, ordered))
    for suite, family_groups in sorted(novel_by_suite.items()):
        ordered = sorted(family_groups, key=lambda item: sha(f'{SEED}/{suite}/{item[0]}'.encode()))
        count = max(1, round(len(ordered)*.2)) if len(ordered) > 1 else 0
        for index, (_, members) in enumerate(ordered):
            (dev if index < count else train).extend(members)
    assignments = {r['base_task']: 'train_candidate' for r in train}
    assignments.update({r['base_task']: 'development' for r in dev})
    for row in chosen:
        if row['official_core']:
            assignments[row['base_task']] = 'core_evaluation'
        elif row['simulator_runtime'] in HELD_OUT_RUNTIMES:
            assignments[row['base_task']] = 'simulator_holdout_reserve'
    chosen_names = {r['task'] for r in chosen}
    for row in rows:
        row['partition'] = assignments[row['base_task']]
        row['selected_mode'] = row['task'] in chosen_names
    split = dict(
        schema_version=2, split_seed=SEED,
        official_core_evaluation=sorted(r['task'] for r in core),
        protected_core_all_modes=sorted(r['task'] for r in rows if r['official_core']),
        train_candidates=sorted(r['task'] for r in train), development=sorted(r['task'] for r in dev),
        simulator_holdout_reserve=sorted(r['task'] for r in chosen if r['partition']=='simulator_holdout_reserve'),
        held_out_runtimes=sorted(HELD_OUT_RUNTIMES),
        held_out_suites=sorted({r['suite'] for r in chosen if r['simulator_runtime'] in HELD_OUT_RUNTIMES}),
        core_cohorts=[classify(r, train) for r in sorted(core, key=lambda x:x['task'])],
        development_cohorts=[classify(r, train) for r in sorted(dev, key=lambda x:x['task'])],
        cohort_status='Planned from candidate membership; recompute using actual admitted task exposure before reporting results',
        scene_audit_status='Pending: distinct task names/digests are insufficient to prove new scene identities',
        engine_note='SAPIEN and Isaac Sim both use PhysX: SAPIEN is an unseen simulator runtime here, not an unseen physics engine. Genesis is unseen at both levels.',
    )
    validate(rows, split)
    return split


def validate(rows, split):
    by_name = {r['task']:r for r in rows}
    partitions = [split[k] for k in ('train_candidates','development','official_core_evaluation','simulator_holdout_reserve')]
    if any(len(names) != len(set(names)) for names in partitions):
        raise ValueError('Duplicate task in a partition')
    expected_core = {r['task'] for r in rows if r['official_default']}
    if set(split['official_core_evaluation']) != expected_core:
        raise ValueError('Official evaluation task membership changed')
    if set(split['protected_core_all_modes']) != {r['task'] for r in rows if r['official_core']}:
        raise ValueError('Official core alternate-mode protection changed')
    bases = [{by_name[t]['base_task'] for t in names} for names in partitions]
    if any(bases[i] & bases[j] for i in range(len(bases)) for j in range(i)):
        raise ValueError('A task/variant or mode twin crosses a partition')
    train = [by_name[t] for t in split['train_candidates']]
    for name in split['train_candidates'] + split['development']:
        if by_name[name]['simulator_runtime'] in split['held_out_runtimes']:
            raise ValueError('Held-out simulator entered training or development')
    declared = {r['task']:r for r in split['core_cohorts']}
    for name in split['official_core_evaluation']:
        actual = classify(by_name[name], train)
        if actual != declared[name]:
            raise ValueError('Cohort is inconsistent with actual training membership')
        if actual['cohort'] == 'held_out_simulator' and actual['runtime_seen_in_training']:
            raise ValueError('An unseen suite was mislabeled as an unseen simulator runtime')
    for row in rows:
        if row['base_task'] in bases[2] and row['partition'] != 'core_evaluation':
            raise ValueError('A protected core alternate mode is not evaluation-only')
    if len(set(r['cohort'] for r in split['core_cohorts'])) != 3:
        raise ValueError('All three requested generalization groups must be represented')


def freeze(preparation, reference, output):
    old = json.loads((preparation/'manifest.json').read_text())
    raw = (preparation/'inventory.json').read_bytes()
    if sha(raw) != old['inventory_sha256']:
        raise ValueError('Preparation inventory checksum mismatch')
    pin = subprocess.check_output(['git','rev-parse','HEAD'],cwd=reference,text=True).strip()
    if pin != REVISION:
        raise ValueError('Reference revision changed')
    rows = enrich(json.loads(raw))
    for row in rows:
        if sha((reference/row['path']/'task.toml').read_bytes()) != row['task_toml_sha256']:
            raise ValueError('Source metadata differs from preparation')
    split = build_split(rows)
    train_names = set(split['train_candidates'])
    dev_names = set(split['development'])
    pilot = {'train_candidates': [], 'development': []}
    for suite in ('dexhand','metaworldplus','robocasa','embodied-safety-exam'):
        for role, names in [('train_candidates',train_names),('development',dev_names)]:
            candidates=sorted([r for r in rows if r['task'] in names and r['suite']==suite
                               and r['declared_gpu_class']=='none' and r['oracle']=='full'],key=rank)
            pilot[role].extend(r['task'] for r in candidates[:2])
    split['initial_multitask_pilot'] = pilot
    cohort_counts = Counter(r['cohort'] for r in split['core_cohorts'])
    runtime_counts = Counter(r['simulator_runtime'] for r in split['core_cohorts'] if r['cohort']=='held_out_simulator')
    artifacts = {'inventory.json': encode(rows), 'splits.json': encode(split)}
    manifest = dict(schema_version=2, upstream_revision=REVISION,
        source_preparation_manifest_sha256=sha((preparation/'manifest.json').read_bytes()),
        inventory_sha256=sha(artifacts['inventory.json']), splits_sha256=sha(artifacts['splits.json']),
        official_core_tasks=len(split['official_core_evaluation']),
        train_candidate_tasks=len(split['train_candidates']), development_tasks=len(split['development']),
        simulator_holdout_reserve_tasks=len(split['simulator_holdout_reserve']),
        planned_core_cohorts=dict(cohort_counts), held_out_runtimes=split['held_out_runtimes'],
        held_out_suites=split['held_out_suites'], held_out_core_by_runtime=dict(runtime_counts),
        strict_unseen_engine_core_tasks=sum(not r['physics_engine_seen_in_training'] for r in split['core_cohorts']),
        taxonomy_evidence=['docs/TAXONOMY.md','tasks/kinder/README.md','tasks/roboarc/README.md'],
        semantic_group_evidence=['tasks/roboarc/README.md','tasks/maniguard/README.md','tasks/yamlab/README.md','tasks/roboengineer/README.md','tasks/libero-max/README.md'],
        status='Task membership frozen; scene collision audit and successful training exposure remain required',
        initial_multitask_pilot=pilot)
    artifacts['manifest.json'] = encode(manifest)
    output.mkdir(parents=True, exist_ok=True)
    for name, data in artifacts.items():
        path=output/name
        if path.exists() and path.read_bytes()!=data:
            raise ValueError('Refusing to replace frozen artifact: '+str(path))
        path.write_bytes(data)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('preparation','reference','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args=parser.parse_args()
    print(json.dumps(freeze(args.preparation,args.reference,args.output),indent=2))


if __name__=='__main__':
    main()
