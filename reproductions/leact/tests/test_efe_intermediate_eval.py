"""A fixed-update diagnostic must preserve weights and the matched panel."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from reproductions.leact.cluster.efe_intermediate_checkpoint import pin
from reproductions.leact.cluster.launch_efe_intermediate_eval import final_receipt,worker_logs
from reproductions.leact.cluster.launch_efe_coverage_eval import register_intermediates


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def committed(tmp_path):
    training=tmp_path/'training';source=training/'restart-states/checkpoint-496'
    source.mkdir(parents=True)
    files={}
    for name in ['adapter_model.safetensors','adapter_config.json','serialization-audit.json']:
        data=('committed '+name).encode();(source/name).write_bytes(data)
        files[name]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
    write(source/'training-state.json',{'step':496,'format':'efe-fp32-lora-adam-rng-v1',
        'lineage_sha256':'lineage','post_update_full_target_ce':.22,'files':files})
    write(training/'spec.json',{'durable_restart':{'lineage_sha256':'lineage'},
        'base_model':'base','base_revision':'revision','model':'qwen'})
    write(training/'dataset-manifest.json',{'examples':[]})
    write(training/'train/schedule.json',{'schedule_sha256':'schedule'})
    (training/'tensors').mkdir();(training/'tensor-blobs').mkdir()
    return training,tmp_path/'pinned'


def test_pin_survives_source_retention_and_is_idempotent(committed):
    training,destination=committed
    before={p.name:p.read_bytes() for p in (training/'restart-states/checkpoint-496').iterdir()}
    receipt=pin(training,destination,496)
    assert receipt['step']==496 and receipt['final'] is False
    assert receipt['additional_optimizer_updates']==0
    assert receipt['expected_post_update_ce']==.22
    assert (destination/'reference/tensors').resolve()==training/'tensors'
    for name,data in before.items():
        original=training/'restart-states/checkpoint-496'/name
        assert original.read_bytes()==data
        original.unlink()
    assert pin(training,destination,496)==receipt


def test_pin_rejects_changed_source(committed):
    training,destination=committed
    (training/'restart-states/checkpoint-496/adapter_model.safetensors').write_bytes(b'wrong')
    with pytest.raises(ValueError,match='checksum mismatch'):pin(training,destination,496)
    assert not (destination/'checkpoint-pin.json').exists()


def test_pin_rejects_changed_copy_or_wrong_update(committed):
    training,destination=committed;pin(training,destination,496)
    with pytest.raises(ValueError,match='update differs'):pin(training,destination,500)
    (destination/'adapter/adapter_model.safetensors').write_bytes(b'wrong')
    with pytest.raises(ValueError,match='bytes differ'):pin(training,destination,496)


def receipt():
    return {'success':True,'step':496,'additional_optimizer_updates':0,'ready':{
        'checkpoint':'model','manifest':'manifest','revision':'step496','step':496,'final':False,
        'pin_sha256':'a'*64,'additional_optimizer_updates':0,
        'loss_replay':{'passed':True,'optimizer_updates':0,'absolute_error':0.0}}}


def test_final_receipt_accepts_verified_fixed_update():
    result=receipt()
    assert final_receipt('prefix EFE_INTERMEDIATE_RESULT '+json.dumps(result),496,'a'*64)==result


@pytest.mark.parametrize('change',[
    lambda r:r.update(step=500),lambda r:r.update(additional_optimizer_updates=1),
    lambda r:r['ready'].update(final=True),lambda r:r['ready'].update(pin_sha256='b'*64),
    lambda r:r['ready']['loss_replay'].update(passed=False),
    lambda r:r['ready']['loss_replay'].update(absolute_error=.001),
    lambda r:r['ready']['loss_replay'].update(absolute_error=float('nan')),
    lambda r:r['ready']['loss_replay'].update(optimizer_updates=1),
])
def test_final_receipt_rejects_wrong_checkpoint_or_failed_replay(change):
    result=receipt();change(result)
    assert final_receipt('EFE_INTERMEDIATE_RESULT '+json.dumps(result),496,'a'*64) is None


def test_latest_failed_receipt_overrides_earlier_success():
    logs='EFE_INTERMEDIATE_RESULT '+json.dumps(receipt())+'\nEFE_INTERMEDIATE_RESULT {"success":false}'
    assert final_receipt(logs,496,'a'*64) is None
    assert final_receipt('EFE_INTERMEDIATE_RESULT {"success":',496,'a'*64) is None


def test_live_log_timeout_preserves_received_evidence(monkeypatch):
    def streaming(*args,**kwargs):
        raise subprocess.TimeoutExpired(args[0],8,output=b'EFE_INTERMEDIATE_PROGRESS {"phase":"verifying_checkpoint"}\n')
    monkeypatch.setattr(subprocess,'run',streaming)
    assert 'verifying_checkpoint' in worker_logs('running-workflow')


def test_intermediate_registration_preserves_panel_and_existing_attempts():
    frozen={'tasks':[{'task':f't{i}'} for i in range(6)],'conditions':['astra','base','warmstart']}
    original=deepcopy(frozen);queue={'batches':[]}
    run={'id':'warmstart-step496','step':496,'panel_sha256':'panel','planned_tasks':6,'pin_sha256':'a'*64}
    registry={'runs':[run]}
    assert register_intermediates(frozen,queue,registry,'panel')=={}
    assert len(queue['batches'])==2 and frozen==original
    queue['batches'][0]['attempts']=[{'workflow':'already-launched'}]
    run['ready']=receipt()['ready']
    assert register_intermediates(frozen,queue,registry,'panel')=={'warmstart-step496':run['ready']}
    assert len(queue['batches'])==2 and queue['batches'][0]['attempts']==[{'workflow':'already-launched'}]
    with pytest.raises(ValueError,match='panel differs'):register_intermediates(frozen,queue,registry,'changed')
    run['ready']['final']=True
    with pytest.raises(ValueError,match='export verification'):register_intermediates(frozen,queue,registry,'panel')
