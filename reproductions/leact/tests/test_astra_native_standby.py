import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from reproductions.leact.cluster import stage_astra_native_standby as standby
from reproductions.leact.cluster import stage_astra_rpc_heartbeat as heartbeat
from reproductions.leact.cluster import monitor_astra_wide_collection as watch
from reproductions.leact.tests.test_astra_rpc_heartbeat import original_build, WORKER


def staged_standby(tmp_path):
    source, _ = original_build(tmp_path)
    workflow = yaml.safe_load((source/'workflow.yaml').read_text())
    task = workflow['workflow']['tasks'][0]
    task.update(image='pinned-native-image', command=['python3'],
                files=[{'localpath': 'bootstrap.py', 'path': '/tmp/pear-native/bootstrap.py'}])
    task['credentials'][standby.CREDENTIAL] = {'NVIDIA_INFERENCE_API_KEY': 'api_key'}
    (source/'workflow.yaml').write_text(yaml.safe_dump(workflow))
    heartbeat.freeze_worker(source, [p.name for p in source.iterdir() if p.name != 'source-manifest.json'])
    fixed = heartbeat.prepare(source, build_root=tmp_path/'fixes')
    return fixed, standby.prepare(fixed, build_root=tmp_path/'standbys')


def test_standby_preserves_native_payload_and_recipe_without_model_credential(tmp_path):
    fixed, build = staged_standby(tmp_path)
    old_spec = json.loads((fixed/'spec.json').read_text())
    new_spec = json.loads((build/'spec.json').read_text())
    assert {k: v for k, v in new_spec.items() if k not in
            ['teacher_native_service_only', 'teacher_activation_required']} == old_spec
    assert new_spec['teacher_native_only'] is True and new_spec['teacher_native_service_only'] is True
    for name in heartbeat.frozen_files(fixed):
        if name != heartbeat.PROVENANCE:
            assert (build/name).read_bytes() == (fixed/name).read_bytes()
    before = yaml.safe_load((fixed/'workflow.yaml').read_text())['workflow']
    after = yaml.safe_load((build/'workflow.yaml').read_text())['workflow']
    assert after['resources'] == before['resources']
    assert after['tasks'][0]['image'] == before['tasks'][0]['image']
    assert after['tasks'][0]['environment']['PEAR_KEEPER_AUTOSTART'] == '0'
    assert standby.CREDENTIAL not in after['tasks'][0]['credentials']
    assert 'secret-reference' in after['tasks'][0]['credentials']
    assert {'localpath': 'source-manifest.json', 'path': '/tmp/pear-native/source-manifest.json'} in after['tasks'][0]['files']
    assert after['name'].startswith('rp-jhh-astra-wide-robodojo-native-only-')
    assert (build/WORKER.name).read_text() == heartbeat.patch_worker(WORKER.read_text())
    assert standby.prepare(fixed, build_root=tmp_path/'standbys') == build
    assert not (build/'submission.json').exists() and not (build/'submission-intent.json').exists()


def test_native_only_task_is_idempotent_and_keeps_archive_credentials():
    task = {'credentials': {standby.CREDENTIAL: {'API_KEY': 'field'}, 'private-archive': {'CONFIG': 'field'}}}
    standby.native_only_task(task)
    first = json.dumps(task, sort_keys=True)
    standby.native_only_task(task)
    assert json.dumps(task, sort_keys=True) == first
    assert set(task['credentials']) == {'private-archive'}


def test_native_service_never_triggers_a_profile_collector(tmp_path, monkeypatch):
    from reproductions.leact.cluster import switch_astra_wide_profile as profile
    _, build = staged_standby(tmp_path)
    state = {'workflow': 'rp-jhh-astra-wide-robodojo-native-only-test', 'build': str(build),
             'label': 'robodojo', 'job_id': 'native-only-job', 'native_service_only': True}
    (tmp_path/'launches.json').write_text(json.dumps({state['workflow']: state}))
    (tmp_path/'active-profile.json').write_text('{"teacher_profile_id":"astra62-xhigh-v1"}')
    monkeypatch.setattr(watch, 'ROOT', tmp_path)
    monkeypatch.setattr(watch, 'DEST', tmp_path)
    monkeypatch.setattr(watch.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(
        returncode=0, stdout=json.dumps({'status': 'RUNNING', 'tasks': [{'status': 'RUNNING'}]})))
    monkeypatch.setattr(watch, 'snapshot', lambda *_: {'attempt_status': {}, 'sft_admitted': 0,
        'nodes': {state['workflow']: {'node': {'phase': 'running', 'child_pid': 123}, 'desired_id': 'native-only-job'}},
        'astra62_profile': {'attempt_status': {}, 'completed_cases': {}, 'workers': {}}})
    monkeypatch.setattr(profile, 'deploy', lambda *_: pytest.fail('A native-only service must not create a model collector'))
    result = watch.tick()
    assert result['native_only_services'] == {state['workflow']: 123}
    assert result['recovery_actions'] == []


def test_native_service_guard_survives_refreshed_allocation_build(tmp_path, monkeypatch):
    _, build = staged_standby(tmp_path)
    monkeypatch.setattr(watch, 'ROOT', tmp_path/'control')
    monkeypatch.setattr(heartbeat, 'BUILD_ROOT', tmp_path/'refresh-overlays')
    workflow = yaml.safe_load((build/'workflow.yaml').read_text())
    def worker_build(spec, files, *args, **kwargs):
        candidate = tmp_path/'refreshed'
        candidate.mkdir()
        for name, path in files.items():(candidate/name).write_bytes(path.read_bytes())
        (candidate/'spec.json').write_text(json.dumps(spec))
        (candidate/'bootstrap.py').write_text('# keeper bootstrap\n')
        (candidate/'workflow.yaml').write_text(yaml.safe_dump(workflow))
        return candidate
    monkeypatch.setattr(watch, 'worker_build', worker_build)
    refreshed = watch.refresh_build({'workflow': 'rp-jhh-astra-wide-robodojo-native-only-test',
        'build': str(build), 'label': 'robodojo', 'native_service_only': True})
    task = yaml.safe_load((refreshed/'workflow.yaml').read_text())['workflow']['tasks'][0]
    assert task['environment']['PEAR_KEEPER_AUTOSTART'] == '0'
    assert standby.CREDENTIAL not in task['credentials']
    assert json.loads((refreshed/'spec.json').read_text())['teacher_native_service_only'] is True
    assert (refreshed/WORKER.name).read_text() == heartbeat.patch_worker(WORKER.read_text())
