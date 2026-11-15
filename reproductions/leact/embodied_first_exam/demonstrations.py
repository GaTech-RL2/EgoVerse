"""Fail-closed admission of native simulator demonstrations.

Replay and scene-audit receipts are produced by the trusted runner, never by
the teacher/student. This module validates the receipts; it does not itself
replay a simulator or certify an external dataset's compatibility.
"""
from __future__ import annotations

import re

from .prepare import encode, sha


def _digest(value, label):
    if not isinstance(value, str) or not re.fullmatch(r'[a-f0-9]{64}', value):
        raise ValueError('Missing or malformed checksum: ' + label)
    return value


def admit(episode, inventory, split, *, binding, scene_audit, replay):
    """Return a provenance record only after identity and fidelity gates pass.

    Binding maps an upstream dataset task to an eligible EFE train task. Both
    scene-audit and replay records must be independently reviewed runner
    outputs. The caller is responsible for file integrity and trusted origin.
    """
    split_sha = sha(encode(split))
    by_name = {r['task']: r for r in inventory}
    name = binding.get('efe_task')
    if name not in split['train_candidates']:
        raise ValueError('Demonstration binding must be an eligible training task')
    row = by_name[name]
    if row['official_core'] or row['simulator_runtime'] in split['held_out_runtimes']:
        raise ValueError('Held-out simulator or evaluation instance entered demonstration training')
    if row['mode'] not in {'standard', 'standard-realtime'}:
        raise ValueError('Privileged observations cannot be a training input')
    for key in ('dataset_url', 'dataset_revision', 'dataset_task', 'license_id'):
        if not episode.get(key) or episode[key] != binding.get(key):
            raise ValueError('Unverified dataset binding: ' + key)
    if binding.get('split_sha256') != split_sha or binding.get('reviewed') is not True:
        raise ValueError('Dataset binding has not been reviewed against this split')
    if binding.get('semantic_task') != row['semantic_task']:
        raise ValueError('Dataset goal differs from the admitted training task')
    if episode.get('simulator_runtime') != row['simulator_runtime']:
        raise ValueError('Demonstration simulator runtime differs from its binding')
    if binding.get('mode') != row['mode']:
        raise ValueError('Demonstration observation/action protocol mode differs')
    for key in ('episode_sha256', 'native_actions_sha256', 'public_observations_sha256',
                'scene_identity_sha256', 'adapter_sha256'):
        _digest(episode.get(key), key)
    if scene_audit.get('complete') is not True or scene_audit.get('split_sha256') != split_sha:
        raise ValueError('Complete cross-partition scene audit is required')
    protected = set(split['official_core_evaluation'] + split['development'] + split['simulator_holdout_reserve'])
    if not protected <= set(scene_audit.get('audited_protected_tasks', [])):
        raise ValueError('Scene audit omits protected evaluation/development tasks')
    if scene_audit.get('episode_sha256') != episode['episode_sha256']:
        raise ValueError('Scene audit does not cover this demonstration episode')
    if scene_audit.get('scene_identity_sha256') != episode['scene_identity_sha256']:
        raise ValueError('Scene identity does not match the audited demonstration')
    if scene_audit.get('collisions') != []:
        raise ValueError('Demonstration scene or episode collides with a held-out instance')
    if scene_audit.get('method') not in {'canonical_scene_and_initial_state', 'verified_native_episode_identity'}:
        raise ValueError('Task names or file digests alone do not establish scene independence')
    for key in ('episode_sha256', 'native_actions_sha256', 'public_observations_sha256',
                'adapter_sha256'):
        if replay.get(key) != episode[key]:
            raise ValueError('Replay receipt does not match ' + key)
    if replay.get('efe_task') != name or replay.get('upstream_digest') != row['upstream_digest']:
        raise ValueError('Replay was not verified against the pinned EFE task')
    for key in ('success', 'public_observation_boundary_passed', 'action_fidelity_passed'):
        if replay.get(key) is not True:
            raise ValueError('Replay gate failed: ' + key)
    if replay.get('input_format') not in {'rgb_proprioception', 'rgbd_proprioception'}:
        raise ValueError('Hidden simulator state cannot be an observation input')
    if replay.get('target_format') != 'native_action_sequence':
        raise ValueError('Native trajectories cannot be relabeled as invented expert code')
    return {
        'efe_task': name, 'semantic_task': row['semantic_task'], 'suite': row['suite'],
        'simulator_runtime': row['simulator_runtime'], 'mode': row['mode'],
        'episode_sha256': episode['episode_sha256'], 'native_actions_sha256': episode['native_actions_sha256'],
        'split_sha256': split_sha, 'binding_sha256': sha(encode(binding)),
        'scene_audit_sha256': sha(encode(scene_audit)), 'replay_receipt_sha256': sha(encode(replay)),
        'eligibility': 'Native action data only; reasoning must pass a separate positive-delta admission',
    }
