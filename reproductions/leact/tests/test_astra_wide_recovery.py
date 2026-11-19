import hashlib
import json

import pytest

from reproductions.leact.cluster import launch_astra_wide_collection as launch
from reproductions.leact.cluster import monitor_astra_wide_collection as watch


def test_freeze_worker_is_stable_when_manifest_already_exists(tmp_path):
    (tmp_path/'worker.py').write_text('print(1)\n')
    launch.freeze_worker(tmp_path,['worker.py'])
    first=(tmp_path/'source-manifest.json').read_bytes()
    (tmp_path/'submission.json').write_text('{"scheduler":"receipt"}')
    launch.freeze_worker(tmp_path,['worker.py'])
    assert (tmp_path/'source-manifest.json').read_bytes()==first
    assert json.loads(first)=={'worker.py':hashlib.sha256(b'print(1)\n').hexdigest()}


@pytest.mark.parametrize('status',['RUNNING','PENDING','COMPLETED','FAILED_CANCELED','FAILED_USER_CANCELED'])
def test_recovery_never_replaces_live_or_user_canceled_allocation(status,monkeypatch):
    monkeypatch.setattr(watch,'refresh_build',lambda *_:pytest.fail('Must preserve this allocation'))
    assert watch.recover_allocation({'workflow':'own'}, {'status':status})is None


def test_child_recovery_requires_exact_recorded_teacher_owner(monkeypatch):
    monkeypatch.setattr(watch,'idle_verification',lambda *_:pytest.fail('A different child owns the allocation'))
    assert watch.recover_child({'job_id':'own'},
        {'node':{'phase':'idle_failed'},'desired_id':'different'})is None


def test_refresh_rejects_non_teacher_source_before_any_build(tmp_path):
    (tmp_path/'spec.json').write_text(json.dumps({'owner_model_workflow':'hero'}))
    with pytest.raises(ValueError,match='independently owned'):
        watch.refresh_build({'build':str(tmp_path),'workflow':'hero'})


def test_native_failure_feedback_stays_scoped_to_worker_tasks(tmp_path):
    import importlib.util
    import sys
    from pathlib import Path
    cluster=Path(watch.__file__).parent
    spec=importlib.util.spec_from_file_location('astra_wide_worker_under_test',cluster/'efe_astra_wide_worker.py')
    old=list(sys.path);sys.path.insert(0,str(cluster))
    try:module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    finally:sys.path[:]=old
    for uid,task,status in [('own','owned','opening'),('other','elsewhere','opening'),('done','owned','graded_success')]:
        result=tmp_path/'episodes'/uid/'infra-00/result.json';result.parent.mkdir(parents=True)
        result.write_text(json.dumps({'status':status,'case':{'task':task},'episode_id':uid}))
        (tmp_path/'transport/episodes'/uid).mkdir(parents=True)
    assert module.fail_owned_requests({'teacher_root':str(tmp_path),'teacher_task_names':['owned'],
        'transport_root':str(tmp_path/'transport')},'native bootstrap died')==1
    assert (tmp_path/'transport/episodes/own/infrastructure-error.json').exists()
    assert not (tmp_path/'transport/episodes/other/infrastructure-error.json').exists()
    assert not (tmp_path/'transport/episodes/done/infrastructure-error.json').exists()
