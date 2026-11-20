from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from reproductions.leact.cluster import efe_training_data
from reproductions.leact.cluster import efe_leact_hf_assemble as assembly
from reproductions.leact.cluster.efe_leact_hf_rescore import corrected_record


def score(value):
    return {'mean_target_logp':value,'target_tokens':4,'target_ids_sha256':'same-command'}


def test_hf_readmission_keeps_samples_and_applies_positive_top_two():
    original={'base':score(-1),'selected':[0,1],'candidates':[
        {'candidate':i,'rationale':f'cached {i}','score':score(-.5),'delta':.5,'eligible':True,'reason':None}
        for i in range(3)]+[{'candidate':3,'eligible':False,'reason':'generation_did_not_finish'}]}
    before=deepcopy(original)
    result=corrected_record(original,{-1:score(-2),0:score(-2.1),1:score(-1.2),2:score(-1)},'original-sha')
    assert original==before and result['selected']==[2,1]
    assert not result['candidates'][0]['eligible']
    assert [c.get('rationale')for c in result['candidates']]==[c.get('rationale')for c in original['candidates']]
    assert result['original_vllm_selected']==[0,1]
    assert result['original_vllm_record_sha256']=='original-sha'


@pytest.mark.parametrize('mutation',['missing','tokens','nonfinite'])
def test_hf_readmission_rejects_incomplete_or_different_target_scores(mutation):
    original={'base':score(-2),'selected':[0],'candidates':[
        {'candidate':0,'rationale':'cached','score':score(-1),'delta':1,'eligible':True}]}
    scores={-1:score(-2),0:score(-1)}
    if mutation=='missing':scores.pop(0)
    elif mutation=='tokens':scores[0]['target_ids_sha256']='different-command'
    else:scores[0]['mean_target_logp']=float('nan')
    with pytest.raises(ValueError):corrected_record(original,scores,'sha')


@pytest.fixture
def promotion(tmp_path,monkeypatch):
    monkeypatch.setitem(sys.modules,'efe_training_data',efe_training_data)
    root=tmp_path/'training';prepared=root/'hf-rescore/prepared';scores=root/'hf-rescore'
    prepared.mkdir(parents=True);(scores/'candidates').mkdir()
    efe_training_data.write(root/'dataset-manifest.json',{'original':True})
    efe_training_data.write(root/'selection.json',{'original':True})
    (root/'candidates').mkdir();(root/'candidates/original.json').write_text('preserved')
    (root/'training-lineage.json').write_text('old lineage before first update')
    original=efe_training_data.sha(root/'dataset-manifest.json')
    for name in assembly.INSTALL_NAMES:
        if '.'not in name:(prepared/name).mkdir()
        else:efe_training_data.write(prepared/name,{'corrected':True,'selection':{'admitted_states':2}})
    return root,prepared,scores,original


@pytest.mark.parametrize('phase',['archive','install'])
def test_interrupted_admission_promotion_resumes_without_losing_originals(promotion,monkeypatch,phase):
    root,prepared,scores,original=promotion;move=assembly.shutil.move;count=0
    def interrupted(source,target):
        nonlocal count
        value=move(source,target)
        if ('admission-before' in str(target))==(phase=='archive'):
            count+=1
            if count==2:raise OSError('simulated node interruption after a move')
        return value
    monkeypatch.setattr(assembly.shutil,'move',interrupted)
    with pytest.raises(OSError):assembly.activate_prepared(root,prepared,scores,original,'scores-sha')
    monkeypatch.setattr(assembly.shutil,'move',move)
    result=assembly.activate_prepared(root,prepared,scores,original,'scores-sha')
    assert result['success'] and (root/'candidates').resolve()==scores/'candidates'
    backup=Path(result['backup'])
    assert efe_training_data.sha(backup/'dataset-manifest.json')==original
    assert (backup/'candidates/original.json').read_text()=='preserved'
    assert (backup/'training-lineage.json').read_text()=='old lineage before first update'
    assert assembly.activate_prepared(root,prepared,scores,original,'scores-sha')==result


def test_admission_promotion_refuses_to_replace_production_updates(promotion):
    root,prepared,scores,original=promotion;(root/'train').mkdir()
    (root/'train/metrics.jsonl').write_text('{"step":1}\n')
    with pytest.raises(ValueError,match='production optimizer update'):
        assembly.activate_prepared(root,prepared,scores,original,'scores-sha')
    assert efe_training_data.sha(root/'dataset-manifest.json')==original
