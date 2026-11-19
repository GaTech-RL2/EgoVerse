import json

import pytest

from reproductions.leact.cluster import recover_efe_pear_numeric as numeric


def test_numeric_recovery_needs_real_physics_and_current_source_receipts():
    receipt={'passed':True,'model_file_relay_passed':True,'native_float32_json_passed':True,
        'numeric_representation':numeric.VERSION,'native_numeric_sources':numeric.source_pins(),
        'executed_control_steps':1,'cameras':[{'std':1.}],'native_termination_terms':{'success':False}}
    assert numeric.qualified(receipt)
    assert not numeric.qualified({**receipt,'executed_control_steps':0})
    assert not numeric.qualified({**receipt,'native_numeric_sources':{}})
    assert not numeric.qualified({**receipt,'native_float32_json_passed':False})


@pytest.mark.parametrize('failed_role',['native','bridge','probe'])
def test_numeric_recovery_retries_each_transport_dependency_before_admission(tmp_path,monkeypatch,failed_role):
    from reproductions.leact.cluster import monitor_efe_pear_full_qualification as monitor
    monkeypatch.setattr(numeric,'DEST',tmp_path)
    (tmp_path/'control.json').write_text(json.dumps({'native_float32_recovery':True,
        'enabled_training_arms':['moe']}))
    (tmp_path/'native-numeric-probe.json').write_text(json.dumps({'jobs':{
        role:{'workflow':role}for role in ['native','bridge','probe']}}))
    monkeypatch.setattr(numeric,'command',lambda args:{'status':'FAILED_QUEUE_TIMEOUT'if args[3]==failed_role else'RUNNING'})
    monkeypatch.setattr(numeric,'infrastructure_failure',lambda query:query['status']=='FAILED_QUEUE_TIMEOUT')
    retries=[]
    def retry(row):retries.append(row['workflow']);row['workflow']+='-retry'
    monkeypatch.setattr(monitor,'retry',retry)
    result=numeric.tick({'workflow':'current-model','scheduler_status':'RUNNING','run_id':'moe'})
    assert retries==[failed_role]
    assert result['phase']=='validating_native_float32_json'
    assert not(tmp_path/'native-float32-proof.json').exists()
    saved=json.loads((tmp_path/'native-numeric-probe.json').read_text())
    assert saved['jobs'][failed_role]['workflow']==failed_role+'-retry'
