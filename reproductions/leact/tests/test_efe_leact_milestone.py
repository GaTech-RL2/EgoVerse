from copy import deepcopy
import hashlib
import json

import pytest

from reproductions.leact.cluster.launch_efe_coverage_eval import register_intermediates
from reproductions.leact.cluster.launch_efe_leact_milestone import pin_receipt


def receipt():
    raw=json.dumps({'step':96,'final':False,'additional_optimizer_updates':0})+'\n'
    return {'success':True,'step':96,'optimizer_updates_by_watcher':0,'pin_json':raw,
            'pin_sha256':hashlib.sha256(raw.encode()).hexdigest()}


def test_pin_receipt_preserves_exact_remote_bytes():
    value=receipt()
    assert pin_receipt('prefix EFE_MILESTONE_PIN '+json.dumps(value))==value
    assert pin_receipt('EFE_MILESTONE_WAIT {}') is None


@pytest.mark.parametrize('field,value',[('step',100),('optimizer_updates_by_watcher',1),('pin_sha256','wrong')])
def test_pin_receipt_rejects_wrong_or_changed_checkpoint(field,value):
    result=receipt();result[field]=value
    with pytest.raises(ValueError):pin_receipt('EFE_MILESTONE_PIN '+json.dumps(result))


def test_early_r1_registration_keeps_full_panel_and_requires_replay():
    panel={'tasks':[{'task':str(i)}for i in range(36)]};before=deepcopy(panel)
    run={'id':'leact-from-step496-r1-step96','training_stage':'leact-from-step496-r1',
         'step':96,'panel_sha256':'fixed','planned_tasks':36,'pin_sha256':'a'*64}
    queue={'batches':[]}
    assert register_intermediates(panel,queue,{'runs':[run]},'fixed')=={}
    assert len(queue['batches'])==9 and panel==before
    run['ready']={'step':96,'final':False,'pin_sha256':'a'*64,'additional_optimizer_updates':0,
                  'checkpoint':'model','manifest':'manifest','revision':'rp_jhh_early_r1',
                  'loss_replay':{'passed':True,'optimizer_updates':0,'absolute_error':0.00001}}
    assert register_intermediates(panel,queue,{'runs':[run]},'fixed')=={run['id']:run['ready']}
    run['ready']['loss_replay']['passed']=False
    with pytest.raises(ValueError):register_intermediates(panel,queue,{'runs':[run]},'fixed')
