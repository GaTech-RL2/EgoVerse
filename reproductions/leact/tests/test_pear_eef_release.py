import copy
from pathlib import Path

import pytest


@pytest.fixture
def release(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'cluster'))
    import efe_pear_primary_worker as worker
    from efe_pear_eef import VERSION,POSE_SUITES,DELTA_SUITES
    spec={'control_interface':VERSION,'run':'new-eef-run','adapter_digests':{'adapter.py':'pinned'}}
    proof={'passed':True,'run':spec['run'],'control_interface':VERSION,'scope':'pear_rl_only',
        'teacher_collection_changed':False,'qualification_sha256':'native-receipt',
        'adapter_sha256':{'adapter.py':'pinned'},
        'simulators':{s:{'passed':True}for s in POSE_SUITES|DELTA_SUITES|{'native_gr00t_yam'}}}
    return worker,spec,proof


def test_model_runtime_can_release_only_the_complete_qualified_eef_branch(release):
    worker,spec,proof=release
    worker.validate_eef_release(spec,proof)


@pytest.mark.parametrize('change', ['missing_suite','wrong_run','different_adapter','changed_teacher','no_receipt'])
def test_runtime_preparation_cannot_bypass_native_motion_or_source_checks(release,change):
    worker,spec,original=release;proof=copy.deepcopy(original)
    if change=='missing_suite':proof['simulators'].pop('vlabench')
    elif change=='wrong_run':proof['run']='old-joint-run'
    elif change=='different_adapter':proof['adapter_sha256']['adapter.py']='changed'
    elif change=='changed_teacher':proof['teacher_collection_changed']=True
    elif change=='no_receipt':proof.pop('qualification_sha256')
    with pytest.raises(ValueError,match='EEF'):worker.validate_eef_release(spec,proof)


def test_native_episode_returns_the_scene_identity_checked_at_initialization(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'cluster'))
    from efe_pear_native_episode import NativeEfeEpisode
    episode=NativeEfeEpisode.__new__(NativeEfeEpisode)
    episode.expected_scene='source-checked-reset'
    episode.last_action=None;episode.contract={};episode.contract_sha='contract';episode.instruction='task'
    assert episode._result({'done':True,'reward':0.})['scene_sha256']=='source-checked-reset'
