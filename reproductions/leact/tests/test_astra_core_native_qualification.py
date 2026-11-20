import json
from pathlib import Path

import pytest
import yaml

from reproductions.leact.cluster import stage_astra_core_native_qualification as stage
from reproductions.leact.cluster.launch_astra_wide_collection import CREDENTIAL, freeze_worker
from reproductions.leact.tests.test_pear_native_gpu_shape import frozen_teacher


def source_build(tmp_path):
    source, spec = frozen_teacher(tmp_path)
    spec.update(suites=sorted(stage.SUITES), lanes_per_gpu=1,
        native_setup={'pinned_repository': 'unchanged'}, episode_archive='s3://private/native',
        token_budget=160000)
    spec['teacher_plan']['cases'] = [{'suite': suite, 'task': 'train-' + suite,
        'config_id': suite + '-seed0', 'case_id': 'case-' + suite, 'scene_sha256': 'a' * 64,
        'admitted': True, 'partition': 'train', 'env_args': {'task': 'train-' + suite}}
        for suite in sorted(stage.SUITES)]
    spec['teacher_task_names'] = [c['task'] for c in spec['teacher_plan']['cases']]
    (source / 'spec.json').write_text(json.dumps(spec))
    workflow = yaml.safe_load((source / 'workflow.yaml').read_text())
    workflow['workflow']['labels'] = {'existing': 'preserved'}
    workflow['workflow']['tasks'][0]['credentials'] = {
        stage.ARCHIVE_CREDENTIAL: {'RP_JHH_PRIVATE_ARCHIVE_CONFIGURATION': 'configuration'},
        CREDENTIAL: {'NVIDIA_INFERENCE_API_KEY': 'api_key'}}
    (source / 'workflow.yaml').write_text(yaml.safe_dump(workflow))
    freeze_worker(source, [p.name for p in source.iterdir() if p.name != 'source-manifest.json'])
    return source, spec


def test_service_only_qualification_keeps_native_recipe_and_uses_private_roots(tmp_path):
    source, spec = source_build(tmp_path)
    original = {p.name: p.read_bytes() for p in source.iterdir()}
    build = stage.prepare(source, build_root=tmp_path / 'builds')
    actual = json.loads((build / 'spec.json').read_text())
    for key in ['native_setup', 'native_config', 'teacher_plan', 'teacher_task_names',
                'max_calls', 'wall_seconds', 'token_budget', 'lanes_per_gpu']:
        assert actual[key] == spec[key]
    assert actual['native_gpu_count'] == 4
    assert actual['teacher_qualification_only'] is True
    assert actual['teacher_native_only'] is True
    assert actual['teacher_root'] != spec['teacher_root']
    assert actual['transport_root'].startswith(actual['teacher_root'] + '/')
    assert actual['output'].startswith(actual['teacher_root'] + '/')
    workflow = yaml.safe_load((build / 'workflow.yaml').read_text())['workflow']
    task = workflow['tasks'][0]
    assert workflow['resources']['default']['gpu'] == 4
    assert set(task['credentials']) == {stage.ARCHIVE_CREDENTIAL}
    assert task['environment']['PEAR_KEEPER_AUTOSTART'] == '0'
    assert task['environment']['PEAR_ALLOCATION_BASE'] == actual['allocation_control_base']
    assert 'NVIDIA_INFERENCE_API_KEY' not in task['environment']
    assert not {'NVIDIA_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES'} & task['environment'].keys()
    cases = json.loads((build / 'qualification-cases.json').read_text())
    assert len(cases) == 6 and {c['suite'] for c in cases} == stage.SUITES
    assert all(c in spec['teacher_plan']['cases'] for c in cases)
    assert stage.prepare(source, build_root=tmp_path / 'builds') == build
    assert {p.name: p.read_bytes() for p in source.iterdir()} == original
    assert not (build / 'submission.json').exists()


@pytest.mark.parametrize('field,value', [('control_interface', 'pear-eef-native-v1'),
    ('native_membership_local', True), ('suites', ['libero-max'])])
def test_reject_interface_or_suite_scope_drift(tmp_path, field, value):
    source, spec = source_build(tmp_path)
    spec[field] = value
    (source / 'spec.json').write_text(json.dumps(spec))
    with pytest.raises(ValueError, match='original six-suite'):
        stage.prepare(source, build_root=tmp_path / 'must-not-write')
    assert not (tmp_path / 'must-not-write').exists()


def test_never_select_protected_or_unadmitted_scene(tmp_path):
    _, spec = source_build(tmp_path)
    spec['teacher_plan']['cases'][0]['partition'] = 'final_test'
    with pytest.raises(ValueError, match='Missing admitted training'):
        stage.cases_for(spec)


def test_qualification_uses_observed_interface_scene_when_admitted(tmp_path):
    _, spec = source_build(tmp_path)
    preferred = {**next(c for c in spec['teacher_plan']['cases'] if c['suite'] == 'libero-max'),
                 'task': 'libero-flat-stove-on-illumination-switch-train'}
    spec['teacher_plan']['cases'].append(preferred)
    spec['teacher_task_names'].append(preferred['task'])
    actual = next(c for c in stage.cases_for(spec) if c['suite'] == 'libero-max')
    assert actual == preferred
