import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import yaml

from reproductions.leact.cluster import efe_pear_native_gpu_shape as shape
from reproductions.leact.cluster import stage_astra_native_gpu_shape as stage


GPUS = ['GPU-'+f'{index:08x}'+'-1111-2222-3333-444444444444' for index in range(8)]


@pytest.mark.parametrize('count', [2, 4])
def test_privileged_host_visibility_never_selects_first_n_unassigned_gpus(count):
    assigned = list(reversed(GPUS[-count:]))
    devices, proof = shape.assigned_devices(count,
        environ={'NVIDIA_VISIBLE_DEVICES': ','.join(assigned)}, query=lambda: GPUS)
    assert devices == assigned and set(devices).isdisjoint(GPUS[:count])
    assert proof['assigned_gpu_uuids'] == assigned
    assert proof['enumerated_gpu_uuids'] == GPUS


@pytest.mark.parametrize('assignment', ['', 'all', 'none', '0,1', 'GPU-abcd,GPU-efgh',
    GPUS[0]+','+GPUS[0], GPUS[0], ','.join(GPUS[:4]), 'MIG-'+GPUS[0]+','+GPUS[1]])
def test_ambiguous_assignment_fails_before_enumeration(assignment):
    with pytest.raises(ValueError, match='Cannot prove allocated GPU UUIDs'):
        shape.assigned_devices(2, environ={'NVIDIA_VISIBLE_DEVICES': assignment},
                               query=lambda: pytest.fail('Never infer ownership from host enumeration'))


@pytest.mark.parametrize('observed', [GPUS[:2], [GPUS[6], GPUS[6], GPUS[7]], ['ambiguous'], []])
def test_missing_or_ambiguous_enumerated_assignment_fails_closed(observed):
    with pytest.raises(ValueError, match='do not match'):
        shape.assigned_devices(2, environ={'NVIDIA_VISIBLE_DEVICES': ','.join(GPUS[-2:])},
                               query=lambda: observed)


def test_cdi_all_does_not_substitute_for_exact_assigned_uuid_devices():
    with pytest.raises(ValueError, match='exact assigned GPU UUID'):
        shape.verify_cdi(GPUS[-2:], {'devices': ['nvidia.com/gpu=all', 'nvidia.com/gpu=0']})
    expected = ['nvidia.com/gpu='+gpu for gpu in GPUS[-2:]]
    assert shape.verify_cdi(GPUS[-2:], {'devices': ['nvidia.com/gpu=all', *expected]}) == expected


def load_native(monkeypatch, tmp_path, filename='efe_astra_native_gpu_bootstrap.py'):
    source = Path(shape.__file__).with_name(filename)
    monkeypatch.syspath_prepend(str(source.parent))
    descriptor = importlib.util.spec_from_file_location('native_shape_bootstrap_under_test', source)
    module = importlib.util.module_from_spec(descriptor)
    descriptor.loader.exec_module(module)
    module.__file__ = str(tmp_path/'efe_pear_primary_native.py')
    return module


@pytest.mark.parametrize('count,filename', [
    (8, 'efe_pear_primary_native.py'),
    (8, 'efe_astra_native_gpu_bootstrap.py'),
    (2, 'efe_astra_native_gpu_bootstrap.py'),
    (4, 'efe_astra_native_gpu_bootstrap.py'),
])
def test_bootstrap_default_eight_and_opt_in_shapes_bind_only_their_devices(tmp_path, monkeypatch, count, filename):
    module = load_native(monkeypatch, tmp_path, filename)
    assigned = GPUS if count == 8 else list(reversed(GPUS[-count:]))
    finished = tmp_path/'finished.json'
    spec = {'owner_model_workflow': 'test-teacher', 'output': str(tmp_path/'output'),
            'native_setup': {}, 'native_config': {'action_config': {'preserved': True}},
            'episode_archive': 'unused', 'finished': str(finished),
            'transport_root': str(tmp_path/'transport'), 'head_heartbeat': str(tmp_path/'head'),
            'lanes_per_gpu': 1, 'suites': ['libero-max']}
    if count != 8:
        spec['native_gpu_count'] = count
    (tmp_path/'spec.json').write_text(json.dumps(spec))
    calls, children = [], []
    native = SimpleNamespace(run=lambda *_args, **_kwargs: None)
    def setup(config):
        calls.append('setup')
        return tmp_path, {'PATH': os.environ.get('PATH', '')}, None, None
    native.setup = setup
    monkeypatch.setitem(sys.modules, 'efe_native_pilot', native)
    monkeypatch.setitem(sys.modules, 'efe_pear_gpu_cdi', SimpleNamespace(
        ensure_gpu_cdi=lambda _: {'devices': ['nvidia.com/gpu='+gpu for gpu in GPUS]}))
    monkeypatch.setattr(module, 'wait_for_model_head', lambda _: None)
    def enumerate_gpus(*args, **kwargs):
        calls.append('enumerate')
        return '\n'.join(GPUS)
    monkeypatch.setattr(module.subprocess, 'check_output', enumerate_gpus)
    monkeypatch.setattr(module.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=0))
    def popen(command, **kwargs):
        children.append(kwargs['env']['EFE_PODMAN_GPUS'])
        if len(children) == count:
            finished.touch()
        return SimpleNamespace(poll=lambda: 0, wait=lambda **_: 0)
    monkeypatch.setattr(module.subprocess, 'Popen', popen)
    original_open = os.open
    def private_open(path, *args, **kwargs):
        if str(path) == '/dev/shm/rp_jhh_native_archive_credentials.json':
            path = tmp_path/'protected-archive-credential'
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(module.os, 'open', private_open)
    monkeypatch.setenv('EFE_WORKFLOW_ID', 'test-only')
    monkeypatch.setenv('RP_JHH_PRIVATE_ARCHIVE_CONFIGURATION', '{}')
    monkeypatch.setenv('NVIDIA_VISIBLE_DEVICES', 'all' if count == 8 else ','.join(assigned))
    module.main()
    assert children == ['nvidia.com/gpu='+gpu for gpu in assigned]
    assert calls == (['setup', 'enumerate'] if count == 8 else ['enumerate', 'setup'])
    receipt = tmp_path/'output/test-only/native-gpu-ownership.json'
    assert receipt.exists() is (count != 8)
    if count != 8:
        assert json.loads(receipt.read_text())['assigned_gpu_uuids'] == assigned


def test_ambiguous_small_bootstrap_never_starts_native_setup(tmp_path, monkeypatch):
    module = load_native(monkeypatch, tmp_path)
    (tmp_path/'spec.json').write_text('{"native_gpu_count":2}')
    monkeypatch.setenv('NVIDIA_VISIBLE_DEVICES', 'all')
    monkeypatch.setitem(sys.modules, 'efe_native_pilot', SimpleNamespace(
        setup=lambda *_: pytest.fail('Ambiguous assignment must stop before Podman or simulator setup')))
    with pytest.raises(ValueError, match='Cannot prove allocated GPU UUIDs'):
        module.main()


def frozen_teacher(tmp_path, environment=None):
    source = tmp_path/'original'
    source.mkdir()
    spec = {'owner_model_workflow': stage.OWNER, 'teacher_profile_id': 'astra62-xhigh-v1',
            'teacher_native_entry': 'efe_pear_primary_native.py', 'teacher_native_only': True,
            'allocation_job_entry': 'efe_astra_wide_worker.py', 'teacher_root': 'existing-teacher',
            'teacher_concurrency': 8, 'max_calls': 96, 'wall_seconds': 1500,
            'teacher_plan': {'cases': [{'task': 'train', 'scene': 'pinned'}]},
            'native_config': {'action_config': {'native': 'unchanged'}}, 'suites': ['libero-max']}
    for name in ['efe_pear_primary_native.py', 'efe_astra_wide_collect.py', 'efe_pear_bridge.py',
                 'efe_action_loop.py', 'efe_pear_allocation_keeper.py']:
        (source/name).write_text('# pinned '+name+'\n')
    (source/'spec.json').write_text(json.dumps(spec))
    workflow = {'workflow': {'name': 'original', 'tasks': [{'name': 'native', 'privileged': True,
        'environment': environment or {}, 'credentials': {'private-reference': {'ENV': 'field'}}}],
        'resources': {'default': {'gpu': 8, 'cpu': 64, 'memory': '512Gi', 'storage': '400Gi'}}}}
    (source/'workflow.yaml').write_text(yaml.safe_dump(workflow))
    stage.freeze_worker(source, [p.name for p in source.iterdir()])
    return source, spec


@pytest.mark.parametrize('count', [2, 4])
def test_staged_shape_preserves_recipe_and_sources_without_submission(tmp_path, count):
    source, spec = frozen_teacher(tmp_path)
    original = {p.name: p.read_bytes() for p in source.iterdir()}
    qualified_bootstrap = Path(shape.__file__).with_name('efe_pear_primary_native.py')
    qualified_bytes = qualified_bootstrap.read_bytes()
    build = stage.prepare(source, count, build_root=tmp_path/'builds')
    actual = json.loads((build/'spec.json').read_text())
    assert {k: v for k, v in actual.items() if k not in {'native_gpu_count', 'native_gpu_ownership'}} == spec
    for name in ['efe_astra_wide_collect.py', 'efe_pear_bridge.py', 'efe_action_loop.py', 'efe_pear_allocation_keeper.py']:
        assert (build/name).read_bytes() == original[name]
    workflow = yaml.safe_load((build/'workflow.yaml').read_text())['workflow']
    assert workflow['resources']['default']['gpu'] == count
    assert workflow['resources']['default']['cpu'] == 8*count
    assert workflow['resources']['default']['memory'] == f'{64*count}Gi'
    assert workflow['resources']['default']['storage'] == '400Gi'
    assert workflow['tasks'] == yaml.safe_load(original['workflow.yaml'])['workflow']['tasks']
    assert stage.prepare(source, count, build_root=tmp_path/'builds') == build
    assert not (build/'submission.json').exists()
    assert not (build/'submission-intent.json').exists()
    assert {p.name: p.read_bytes() for p in source.iterdir()} == original
    assert qualified_bootstrap.read_bytes() == qualified_bytes
    assert (build/'efe_pear_primary_native.py').read_bytes() == Path(shape.__file__).with_name(
        'efe_astra_native_gpu_bootstrap.py').read_bytes()


def test_staging_rejects_manual_gpu_assignment_override(tmp_path):
    source, _ = frozen_teacher(tmp_path, {'NVIDIA_VISIBLE_DEVICES': '0,1'})
    with pytest.raises(ValueError, match='must not override scheduler'):
        stage.prepare(source, 2, build_root=tmp_path/'must-not-write')
    assert not (tmp_path/'must-not-write').exists()
