import hashlib
import json

import pytest

from reproductions.leact.cluster.efe_pear_eef_admission import (
    SUITES,VERSION,admission,protected_teacher_allocations,qualify)
from reproductions.leact.cluster.efe_pear_full_admission import PendingAdmission


def test_incomplete_eef_mixture_cannot_activate_training(tmp_path):
    with pytest.raises(PendingAdmission,match='still running'):qualify(tmp_path,tmp_path)
    root=tmp_path/'eef-interface-v1';root.mkdir()
    (root/'qualification.json').write_text(json.dumps({'passed':True,'control_interface':VERSION,
        'scope':'pear_rl_only','teacher_collection_changed':False,'simulators':{'native_gr00t_yam':{'passed':True}}}))
    with pytest.raises(PendingAdmission,match='complete PEAR-only'):qualify(tmp_path,tmp_path)


def test_source_and_native_receipts_must_match_before_eef_activation(tmp_path):
    root=tmp_path/'eef-interface-v1';root.mkdir();source=tmp_path/'controller.py';source.write_text('version=1\n')
    receipt=root/'native-motion.json';receipt.write_text('{"passed":true}')
    proof={'passed':True,'control_interface':VERSION,'scope':'pear_rl_only','teacher_collection_changed':False,
        'simulators':{suite:{'passed':True,'receipt':str(receipt),
            'receipt_sha256':hashlib.sha256(receipt.read_bytes()).hexdigest()}for suite in SUITES},
        'adapter_sha256':{'controller.py':hashlib.sha256(source.read_bytes()).hexdigest()}}
    (root/'qualification.json').write_text(json.dumps(proof))
    assert qualify(tmp_path,tmp_path)['initialization']=='moe_warmstart496'
    source.write_text('version=2\n')
    with pytest.raises(PendingAdmission,match='changed after native'):qualify(tmp_path,tmp_path)


def test_eef_fork_preserves_task_membership_and_original_action_evidence():
    original={'native_membership':{'training_tasks':{suite:{'row':{'suite':suite},
        'action_contract':{'old':suite}}for suite in SUITES-{'native_gr00t_yam'}}},
        'curriculum':{'retained_groups_per_update':32,'configs':['same-scenes']}}
    result=admission(original,{'version':VERSION})
    assert result['curriculum']['retained_groups_per_update']==32
    assert result['curriculum']['configs']==['same-scenes']
    for suite,item in result['native_membership']['training_tasks'].items():
        assert item['prior_native_action_contract']=={'old':suite}
        assert item['control_interface']==VERSION
        assert original['native_membership']['training_tasks'][suite]['action_contract']=={'old':suite}


def test_live_teacher_allocations_are_protected_from_rl_worker_replacement(tmp_path):
    root=tmp_path/'astra-wide-warmstart-v1';root.mkdir()
    (root/'launches.json').write_text(json.dumps({'current':{'workflow':'keep-teacher'},
        'old':{'workflow':'old-teacher','superseded_by':'keep-teacher'}}))
    assert protected_teacher_allocations(tmp_path)=={'keep-teacher'}


def test_controller_proof_requires_complete_motion_and_actual_native_finalization():
    from reproductions.leact.cluster.release_efe_pear_eef import measured
    checks=[{'suite':'abc-sim','task':'training-scene','passed':True}for _ in range(5)]
    status={'phase':'waiting_for_native_motion','checks':checks}
    assert not measured(status,suite='abc-sim',count=5)
    status.update(phase='finished',finalized_tasks=['training-scene'])
    assert measured(status,suite='abc-sim',count=5)
    status['checks'][0]['passed']=False
    assert not measured(status,suite='abc-sim',count=5)


def test_source_pinning_allows_only_non_executable_changes(tmp_path,monkeypatch):
    from reproductions.leact.cluster import release_efe_pear_eef as release
    source=tmp_path/'current';(source/'cluster').mkdir(parents=True)
    build=tmp_path/'frozen';build.mkdir()
    (build/'controller.py').write_text('"""Old description."""\nvalue=1\n')
    current=source/'cluster/controller.py';current.write_text('"""New description."""\n# comment\nvalue = 1\n')
    monkeypatch.setattr(release,'BASE',source)
    pins,equivalent=release.source_pins(build,['controller.py'])
    assert pins['controller.py']==release.sha(current)and len(equivalent)==1
    current.write_text('value=2\n')
    with pytest.raises(PendingAdmission,match='executable adapter change'):
        release.source_pins(build,['controller.py'])


def test_recorded_contract_equivalence_rejects_changed_bounds_even_when_text_changes_are_allowed(tmp_path):
    from reproductions.leact.cluster.release_efe_pear_eef import contract_metadata_equivalent
    before=tmp_path/'before.py';after=tmp_path/'after.py'
    template='''import copy
def contract(spec,suite,max_repeat):
 result={'control_dt':spec['control_dt'],'max_repeat':max_repeat,'cameras':list(spec['cameras']),
  'action_types':copy.deepcopy(spec['action_types'])}
 result['action_types']['ee_pose']['units']=UNITS
 return result
'''
    before.write_text(template.replace('UNITS',repr('old native unit description')))
    after.write_text(template.replace('UNITS',repr('correct normalized unit description')))
    records=[{'action_contract':{'control_dt':.1,'max_repeat':25,'cameras':['rgb'],
        'action_types':{'ee_pose':{'units':'old native unit description','lower':[-3.],'upper':[3.]}}}}]
    assert contract_metadata_equivalent(before,after,'robolab',records)
    after.write_text(after.read_text().replace(' return result'," result['action_types']['ee_pose']['lower']=[-2.]\n return result"))
    assert not contract_metadata_equivalent(before,after,'robolab',records)
