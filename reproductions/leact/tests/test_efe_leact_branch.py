from copy import deepcopy
import errno
import hashlib
import json

import pytest

from reproductions.leact.cluster.launch_efe_leact_branch import parent_for_branch
from reproductions.leact.cluster.launch_efe_coverage_eval import register_branches
from reproductions.leact.cluster.efe_qwen36_leact import link_immutable
from reproductions.leact.cluster.launch_efe_coverage_train import retry_build


def parent():
    result={'success':True,'step':496,'ready':{'step':496,'final':False,'additional_optimizer_updates':0,
        'checkpoint':'model','manifest':'manifest','revision':'step496',
        'loss_replay':{'passed':True,'optimizer_updates':0,'absolute_error':1.48e-8}}}
    pin={'step':496,'dataset_sha256':'data'}
    dataset={'sha256':'data','source_sha256':'decisions'}
    admission={'gate_passed':True,'decisions_sha256':'decisions'}
    return result,pin,dataset,admission


def test_leact_can_branch_from_verified_intermediate_without_changing_parent():
    values=parent();before=deepcopy(values)
    assert parent_for_branch(*values)['ready']['step']==496
    assert values==before


@pytest.mark.parametrize('change',[
    lambda v:v[0]['ready'].update(step=500),
    lambda v:v[0]['ready']['loss_replay'].update(passed=False),
    lambda v:v[0]['ready']['loss_replay'].update(absolute_error=float('nan')),
    lambda v:v[0]['ready'].update(additional_optimizer_updates=1),
    lambda v:v[2].update(sha256='other'),lambda v:v[3].update(gate_passed=False),
])
def test_leact_parent_requires_matching_weights_and_corpus(change):
    values=parent();change(values)
    with pytest.raises(ValueError):parent_for_branch(*values)


def test_branch_evaluation_waits_for_its_own_export_and_preserves_panel():
    panel={'tasks':[{'task':str(i)} for i in range(5)]};before=deepcopy(panel)
    run={'id':'leact-from-step496-r1','archive_root':'branch','panel_sha256':'panel','planned_tasks':5}
    queue={'batches':[]}
    assert register_branches(panel,queue,[run],{},'panel')=={}
    assert len(queue['batches'])==2 and panel==before
    stage={'complete':{'success':True},'ready':{'final':True,'checkpoint':'branch/model'}}
    assert register_branches(panel,queue,[run],{run['id']:stage},'panel')=={run['id']:stage['ready']}
    assert len(queue['batches'])==2
    with pytest.raises(ValueError):register_branches(panel,queue,[run],{},'different-panel')
    stage['ready']['checkpoint']='final737/model'
    with pytest.raises(ValueError):register_branches(panel,queue,[run],{run['id']:stage},'panel')
    stage['complete']['no_admissions']=True
    assert register_branches(panel,queue,[run],{run['id']:stage},'panel')=={}


def test_cross_mount_replay_retains_immutable_source(tmp_path,monkeypatch):
    source=tmp_path/'source.pt';source.write_bytes(b'frozen visual tensor')
    expected=hashlib.sha256(source.read_bytes()).hexdigest()
    def cross_device(*args):raise OSError(errno.EXDEV,'different filesystem')
    monkeypatch.setattr('os.link',cross_device)
    destination=tmp_path/'branch/blob.pt';link_immutable(source,destination,expected)
    assert destination.is_symlink() and destination.read_bytes()==source.read_bytes()
    link_immutable(source,destination,expected)
    source.write_bytes(b'changed')
    with pytest.raises(ValueError):link_immutable(source,destination,expected)
    def no_space(*args):raise OSError(errno.ENOSPC,'full')
    monkeypatch.setattr('os.link',no_space)
    with pytest.raises(OSError):link_immutable(source,tmp_path/'other.pt')


def test_allocation_retry_keeps_branch_separate_from_final_warmstart(tmp_path):
    source=tmp_path/'source';source.mkdir()
    spec={'stage':'leact-r1','run_id':'leact-from-step496-r1','base_revision':'step496'}
    (source/'spec.json').write_text(json.dumps(spec))
    (source/'workflow.yaml').write_text('workflow:\n  name: original\n')
    (source/'source-manifest.json').write_text(json.dumps({name:'unused' for name in ['spec.json','workflow.yaml']}))
    retry=retry_build(source,1)
    assert 'leact-from-step496-r1' in retry.name
    assert json.loads((retry/'spec.json').read_text())==spec
    assert 'leact-from-step496-r1-v1-p7-r1' in (retry/'workflow.yaml').read_text()
