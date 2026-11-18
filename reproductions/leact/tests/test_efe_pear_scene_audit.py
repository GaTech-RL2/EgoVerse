import hashlib
import json

import numpy as np
import pytest

from reproductions.leact.cluster.efe_pear_scene_audit import audit,scene_identity


def task(root,name,suite='robolab',*,privileged=False,state=0,instruction='a'):
    directory=root/name;sim=directory/'environment'
    if not privileged:sim=sim/'sim'
    sim.mkdir(parents=True)
    (directory/'task.toml').write_text('task metadata')
    (directory/'DIGEST').write_text('sha256:'+'a'*64)
    if suite=='metaworldplus':
        np.savez(sim/'instance.npz',integration=np.array([state],dtype=np.float64),
                 model_qpos0=np.array([0.]),instance_sha=np.array('b'*64))
    else:
        (sim/'instance.json').write_text(json.dumps({'state_sha':'b'*16,
            'initial_state':{'qpos':[state]},'instruction':instruction}))
    return {'task':name,'path':name,'suite':suite,
            'task_toml_sha256':hashlib.sha256((directory/'task.toml').read_bytes()).hexdigest(),
            'upstream_digest':'a'*64}


def test_physical_identity_ignores_instruction_and_matches_privileged_mode(tmp_path):
    a=task(tmp_path,'train',state=1,instruction='train instruction')
    b=task(tmp_path,'reserved',state=1,instruction='another goal',privileged=True)
    assert scene_identity(tmp_path,a)['scene_sha256']==scene_identity(tmp_path,b)['scene_sha256']
    result=audit(tmp_path,[a],[b])
    assert result['tasks']['train']['protected_reset_overlap']==['reserved']
    assert not result['tasks']['train']['source_scene_check_passed']


def test_new_physical_state_is_separate_and_source_changes_fail_closed(tmp_path):
    a=task(tmp_path,'train',state=1);b=task(tmp_path,'reserved',state=2)
    assert audit(tmp_path,[a],[b])['tasks']['train']['source_scene_check_passed']
    (tmp_path/'train/task.toml').write_text('changed')
    with pytest.raises(ValueError,match='metadata changed'):scene_identity(tmp_path,a)


def test_missing_protected_state_blocks_admission(tmp_path):
    a=task(tmp_path,'train');b=task(tmp_path,'reserved')
    (tmp_path/'reserved/environment/sim/instance.json').unlink()
    result=audit(tmp_path,[a],[b])
    assert result['protected_state_errors']
    assert not result['tasks']['train']['source_scene_check_passed']


def test_mujoco_fingerprint_includes_model_arrays_and_initial_integration(tmp_path):
    a=task(tmp_path,'a','metaworldplus',state=1)
    b=task(tmp_path,'b','metaworldplus',state=2)
    assert scene_identity(tmp_path,a)['scene_sha256']!=scene_identity(tmp_path,b)['scene_sha256']
