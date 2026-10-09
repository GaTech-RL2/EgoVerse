"""Measure reset observation differences without executing policy actions."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from astra_reversal.records import digest
from .worker import write_json
from .xiaomi_selector_eval import restore_anchor, fingerprint
from .xiaomi_selector_resume import load_anchor


def compare_observations(actual, expected):
    if set(actual) != set(expected):
        raise ValueError('Observation keys changed')
    rows = {}
    for key in expected:
        a, b = np.asarray(actual[key]), np.asarray(expected[key])
        row = dict(shape_matches=a.shape == b.shape, dtype_matches=a.dtype == b.dtype,
                   exact=np.array_equal(a,b), actual_sha256=digest(a), expected_sha256=digest(b))
        if a.shape == b.shape and np.issubdtype(a.dtype,np.number) and np.issubdtype(b.dtype,np.number):
            delta = np.abs(a.astype(np.float64)-b.astype(np.float64))
            row.update(max_abs=float(delta.max(initial=0)),mean_abs=float(delta.mean()),
                       changed_elements=int(np.count_nonzero(delta)),elements=int(delta.size))
            if a.ndim == 3 and a.shape[-1] == 3:
                row['changed_pixels'] = int(np.count_nonzero(np.any(delta,axis=-1)))
                row['pixels'] = int(a.shape[0]*a.shape[1])
        rows[key] = row
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    protocol = json.loads(args.protocol.read_text())
    parent = args.output.parent/'replay_parent'
    anchor, expected_fingerprint = load_anchor(parent)
    with np.load(parent/f"native1/seed{protocol['replay_seed']}/initial_observation.npz",allow_pickle=False) as archive:
        expected = {k:archive[k] for k in archive.files if k != 'simulator_state'}
    sys.path.insert(0,'/opt/astra-xiaomi/sources/xiaomi/eval_robocasa365')
    import entry
    import gymnasium as gym
    import robocasa  # noqa:F401
    counts = dict(constructor_setup_resets=0,anchor_selection_resets=0,diagnostic_restore_resets=0,
                  controls=0,policy_queries=0,teacher_jobs=0,policy_success_trial_denominator_contribution=0)
    rows = []
    for constructor_index in range(3):
        np.random.seed(protocol['constructor_seed'])
        env = gym.make('robocasa/'+protocol['replay_task'],split='pretrain',
                       seed=protocol['constructor_seed'],disable_env_checker=True)
        counts['constructor_setup_resets'] += 1
        try:
            env.reset(seed=protocol['replay_seed'])
            counts['anchor_selection_resets'] += 1
            for _ in range(2):
                restore_anchor(env,anchor)
                counts['diagnostic_restore_resets'] += 1
            for repeat in range(3):
                obs = restore_anchor(env,anchor)
                counts['diagnostic_restore_resets'] += 1
                actual_fingerprint = fingerprint(env,obs)
                name = f'constructor{constructor_index}_restore{repeat}'
                np.savez_compressed(args.output/(name+'.npz'),**obs)
                row = dict(name=name,actual_fingerprint=actual_fingerprint,
                    expected_fingerprint=expected_fingerprint,
                    fingerprint_exact=actual_fingerprint == expected_fingerprint,
                    observations=compare_observations(obs,expected),
                    policy_state_exact=np.array_equal(entry.observation_to_state(obs),entry.observation_to_state(expected)))
                rows.append(row)
                write_json(args.output/'summary.json',dict(counts=counts,runs=rows))
        finally:
            env.close()
    write_json(args.output/'completed.json',dict(complete=True,counts=counts,comparisons=len(rows),
               exact_fingerprint_matches=sum(r['fingerprint_exact'] for r in rows),
               exact_policy_state_matches=sum(r['policy_state_exact'] for r in rows)))


if __name__ == '__main__':
    main()
