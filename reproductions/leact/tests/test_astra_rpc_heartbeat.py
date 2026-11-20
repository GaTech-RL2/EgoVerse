import errno
import hashlib
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
import threading
import time

import pytest
import yaml

from reproductions.leact.cluster import efe_astra_heartbeat_io as heartbeat_io
from reproductions.leact.cluster import stage_astra_rpc_heartbeat as stage


WORKER = Path(stage.__file__).with_name('efe_pear_remote_worker.py')


@pytest.mark.parametrize('age,expected', [(0, False), (300, False), (300.001, True)])
def test_expiry_retains_the_existing_strict_300_second_boundary(tmp_path, age, expected):
    path = tmp_path/'head-heartbeat.json'
    path.touch()
    stamp = path.stat().st_mtime
    assert heartbeat_io.head_heartbeat_expired(path, now=stamp+age) is expected


def test_missing_heartbeat_remains_non_expired(tmp_path):
    assert heartbeat_io.head_heartbeat_expired(tmp_path/'missing.json') is False


def test_unrelated_filesystem_error_is_not_hidden(tmp_path, monkeypatch):
    failure = PermissionError(errno.EACCES, 'denied')
    def denied(path, **kwargs):
        raise failure
    monkeypatch.setattr(Path, 'stat', denied)
    with pytest.raises(PermissionError) as caught:
        heartbeat_io.head_heartbeat_expired(tmp_path/'head-heartbeat.json')
    assert caught.value is failure


def test_patch_is_exactly_reversible_and_refuses_unknown_source():
    original = WORKER.read_text()
    patched = stage.patch_worker(original)
    assert patched.replace(stage.HELPER_IMPORT, '').replace(stage.FIXED_CHECK, stage.ORIGINAL_CHECK) == original
    assert patched.count(stage.FIXED_CHECK) == 1
    for invalid in [original.replace(stage.ORIGINAL_CHECK, 'different_predicate()'),
                    original+'\n# '+stage.ORIGINAL_CHECK, patched]:
        with pytest.raises(ValueError):
            stage.patch_worker(invalid)


def test_real_rpc_heartbeat_survives_disappearing_head_file(tmp_path, monkeypatch):
    """Reproduce exists=True/stat=ENOENT while the actual serve loop is active."""
    monkeypatch.syspath_prepend(str(WORKER.parent))
    module = ModuleType('isolated_teacher_heartbeat_worker')
    exec(compile(stage.patch_worker(WORKER.read_text()), str(WORKER), 'exec'), module.__dict__)
    finished = tmp_path/'finished.json'
    head = finished.with_name('head-heartbeat.json')
    head.touch()
    original_stat, original_exists = Path.stat, Path.exists
    observations = {'head_stats': 0, 'heartbeats': 0, 'claims': 0}

    def disappearing_stat(path, *args, **kwargs):
        if path == head:
            observations['head_stats'] += 1
            raise FileNotFoundError(errno.ENOENT, 'heartbeat replaced between filesystem operations')
        return original_stat(path, *args, **kwargs)

    def raced_exists(path):
        return True if path == head else original_exists(path)

    monkeypatch.setattr(Path, 'stat', disappearing_stat)
    monkeypatch.setattr(Path, 'exists', raced_exists)
    # This is the precise failure mode in the original predicate.
    with pytest.raises(FileNotFoundError):
        if head.exists() and time.time()-head.stat().st_mtime > 300:
            pytest.fail('Unreachable')

    class FastEvent(threading.Event):
        def wait(self, timeout=None):
            return super().wait(.005 if timeout == 10 else timeout)

    module.threading = SimpleNamespace(Event=FastEvent, Thread=threading.Thread)
    original_write = module.atomic_json
    def write(path, payload):
        if path.parent.name == 'workers' and payload['phase'] != 'stopped':
            observations['heartbeats'] += 1
        original_write(path, payload)
    module.atomic_json = write
    folder = tmp_path/'episode'
    folder.mkdir()
    def claim(*args, **kwargs):
        observations['claims'] += 1
        return folder, {'env_args': {}}
    module.claim_episode = claim
    def native_episode(*args):
        try:
            deadline = time.monotonic()+2
            while observations['heartbeats'] < 3 and time.monotonic() < deadline:
                time.sleep(.005)
            assert observations['heartbeats'] >= 3, 'Heartbeat thread died in the replacement race'
        finally:
            finished.touch()
    module.serve_episode = native_episode
    module.serve(tmp_path/'transport', 'teacher-worker', {'efe/robodojo'}, object, finished=finished)
    result = json.loads((tmp_path/'transport/workers/teacher-worker.json').read_text())
    assert observations['head_stats'] >= 4
    assert observations['claims'] == 1
    assert result['episodes'] == 1 and result['errors'] == 0 and result['phase'] == 'stopped'


def original_build(tmp_path):
    source = tmp_path/'original'
    source.mkdir()
    spec = {'owner_model_workflow': stage.OWNER, 'teacher_profile_id': 'astra62-xhigh-v1',
            'teacher_native_entry': 'efe_pear_primary_rtx.py', 'teacher_native_only': True,
            'teacher_root': stage.PROFILE_ROOT, 'suite': 'robodojo', 'max_calls': 96,
            'wall_seconds': 1500, 'token_budget': 160000, 'native_config': {'joint_only': True},
            'transport_root': 'unchanged', 'teacher_plan': {'cases': ['train-only']}}
    (source/'spec.json').write_text(json.dumps(spec, indent=3)+'\n')
    (source/'efe_pear_remote_worker.py').write_bytes(WORKER.read_bytes())
    for name in ['efe_pear_bridge.py', 'efe_pear_file_rpc.py', 'efe_astra_wide_worker.py',
                 'efe_pear_allocation_keeper.py']:
        (source/name).write_text('# pinned '+name+'\n')
    workflow = {'workflow': {'name': 'original', 'tasks': [{'name': 'native',
        'environment': {'UNCHANGED': 'yes'}, 'credentials': {'secret-reference': {'ENV': 'field'}}}],
        'resources': {'default': {'gpu': 2}}}}
    (source/'workflow.yaml').write_text(yaml.safe_dump(workflow))
    stage.freeze_worker(source, [p.name for p in source.iterdir()])
    return source, spec


def test_frozen_overlay_changes_no_recipe_sources_or_shared_runtime(tmp_path):
    source, spec = original_build(tmp_path)
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    shared_before = hashlib.sha256(WORKER.read_bytes()).hexdigest()
    build = stage.prepare(source, build_root=tmp_path/'builds')
    assert (build/'spec.json').read_bytes() == before['spec.json']
    assert json.loads((build/'spec.json').read_text()) == spec
    assert (build/'efe_astra_original_rpc_worker.py').read_bytes() == before[WORKER.name]
    assert (build/WORKER.name).read_text() == stage.patch_worker(before[WORKER.name].decode())
    for name in ['efe_pear_bridge.py', 'efe_pear_file_rpc.py', 'efe_astra_wide_worker.py',
                 'efe_pear_allocation_keeper.py']:
        assert (build/name).read_bytes() == before[name]
    old_workflow = yaml.safe_load(before['workflow.yaml'])['workflow']
    new_workflow = yaml.safe_load((build/'workflow.yaml').read_text())['workflow']
    assert new_workflow['tasks'] == old_workflow['tasks']
    assert new_workflow['resources'] == old_workflow['resources']
    provenance = json.loads((build/stage.PROVENANCE).read_text())
    assert provenance['submitted'] is False and provenance['processes_changed'] is False
    assert provenance['watcher_adoption'] is False and provenance['exhausted_case_retry_pass'] is False
    assert stage.prepare(source, build_root=tmp_path/'builds') == build
    assert not (build/'submission.json').exists() and not (build/'submission-intent.json').exists()
    assert {p.name: p.read_bytes() for p in source.iterdir()} == before
    assert hashlib.sha256(WORKER.read_bytes()).hexdigest() == shared_before


@pytest.mark.parametrize('change', [
    {'owner_model_workflow': 'hero'}, {'teacher_profile_id': 'legacy-high'},
    {'teacher_native_entry': 'efe_pear_primary_native.py'}, {'teacher_native_only': False},
    {'teacher_root': 'other-root'}, {'suite': 'robolab'}, {'teacher_transport_diagnostics': True}])
def test_overlay_scope_fails_closed_before_writing(tmp_path, change):
    source, spec = original_build(tmp_path)
    (source/'spec.json').write_text(json.dumps({**spec, **change}))
    with pytest.raises(ValueError, match='independently owned'):
        stage.prepare(source, build_root=tmp_path/'must-not-exist')
    assert not (tmp_path/'must-not-exist').exists()
