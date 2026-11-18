import copy
import json

import pytest

from reproductions.leact.cluster.efe_pear_validation_resume import passed_checks


def sample(oracle='none'):
    row={'task':'task-a','oracle':oracle,'task_toml_sha256':'metadata'}
    spec={'tasks':[row],'expected_scenes':{'task-a':'scene'}}
    value={'row':row,'scene_sha256':'scene','action_contract':{'bounded':True},
        'native_rtx_api_passed':True,'native_file_transport_passed':True,
        'native_final':{'finalized':True,'success':False,'graded_by':'live','replays':0}}
    return spec,{'task-a':value}


def test_completed_native_control_survives_allocation_retry_without_mutating_its_receipt():
    spec,checks=sample();original=copy.deepcopy(checks)
    saved=passed_checks(spec,checks,'a'*64)
    assert set(saved)=={'task-a'}and saved['task-a']['resumed_from_receipt_sha256']=='a'*64
    assert checks==original


def test_advertised_oracle_must_finish_before_a_task_can_be_reused():
    spec,checks=sample('full')
    assert passed_checks(spec,checks)=={}
    checks['task-a'].update(positive_solver_control=True,positive_native_final={
        'finalized':True,'success':True,'graded_by':'live','replays':0})
    assert set(passed_checks(spec,checks))=={'task-a'}


def test_another_scene_or_another_shard_cannot_supply_a_cached_pass():
    spec,checks=sample();checks['task-a']['scene_sha256']='another-scene'
    assert passed_checks(spec,checks)=={}
    with pytest.raises(ValueError,match='another shard'):
        passed_checks(spec,{'task-from-another-shard':{}})


def test_updated_orchestrator_does_not_duplicate_an_already_owned_task(tmp_path,monkeypatch):
    from reproductions.leact.cluster import launch_efe_pear_rtx as launcher
    monkeypatch.setattr(launcher,'DEST',tmp_path)
    build=tmp_path/'new-code-hash';build.mkdir()
    (build/'spec.json').write_text(json.dumps({'tasks':[{'task':'task-a'}]}))
    (tmp_path/'rtx-transport-queue.json').write_text(json.dumps({
        'old-code-hash':{'suite':'robodojo','tasks':['task-a'],'workflow':'already-submitted'}}))
    monkeypatch.setattr(launcher,'prepare',lambda *a,**k:[build])
    monkeypatch.setattr(launcher,'submit_reconciled',lambda *a,**k:pytest.fail('Duplicate GPU job submitted'))
    result=launcher.launch('robodojo',lanes=1)
    assert set(result)=={'old-code-hash'}
