import gzip
import hashlib
import json

import pytest

from reproductions.leact.cluster.efe_training_data import extract_decisions, balanced_order, sha


def fixture(tmp_path):
    root=tmp_path/'episode';job=root/'jobs/00-astra';job.mkdir(parents=True)
    trial_dir=job/'trial';trial_dir.mkdir()
    actual={'task_name':'train-task','verifier_result':{'rewards':{'success':1,'resource_enforced':1}},'exception_info':None}
    (trial_dir/'result.json').write_text(json.dumps(actual))
    req=root/'requests/00-astra';req.mkdir(parents=True)
    tools=[{'type':'namespace','name':'functions','tools':[{'type':'function','name':'exec_command','parameters':{'type':'object'}}]}]
    request={'instructions':'Use apply_patch.','input':[{'role':'user','content':'Observe the red cube.'}], 'tools':tools}
    (req/'a.request.json.gz').write_bytes(gzip.compress(json.dumps(request).encode()))
    record={'id':'a','started_utc':'2026-10-27','status':'completed','http_status':200,'public_output':[
        {'type':'function_call','call_id':'call-a','name':'exec_command','namespace':'functions','arguments':'{"cmd":"python controller.py"}'}]}
    (req/'a.json').write_text(json.dumps(record))
    trial={'archive':str(job),'task':'train-task','job':'00-astra','workflow':'frozen-workflow','model_requests':1,
           'trial_result_sha256':hashlib.sha256(json.dumps(actual,sort_keys=True).encode()).hexdigest()}
    audit=tmp_path/'audit.json';audit.write_text(json.dumps({'eligible_episodes':[{'trial':trial,'suite':'suite','scene_sha256':'scene'}]}))
    output=tmp_path/'output';output.mkdir()
    return {'scene_audit':str(audit),'scene_audit_sha256':sha(audit),'student_prompt_requests':str(req)},output,req,trial_dir


def test_admission_preserves_exact_action_and_only_predecision_observation(tmp_path):
    spec,output,_,_=fixture(tmp_path)
    rows=extract_decisions(spec,output)
    assert len(rows)==1
    assert rows[0]['action'][0]['function']=={'name':'functions__exec_command','arguments':{'cmd':'python controller.py'}}
    assert rows[0]['messages'][1:]==[{'role':'user','content':'Observe the red cube.'}]
    assert 'verifier_result' not in json.dumps(rows[0]['messages'])
    assert 'apply_patch' not in rows[0]['messages'][0]['content']


def test_changed_native_verdict_is_rejected_even_if_the_cached_receipt_said_success(tmp_path):
    spec,output,_,trial_dir=fixture(tmp_path)
    (trial_dir/'result.json').write_text(json.dumps({'task_name':'train-task','verifier_result':{'rewards':{'success':0,'resource_enforced':1}}}))
    with pytest.raises(ValueError,match='grade receipt changed'):extract_decisions(spec,output)


def test_private_reasoning_in_archived_input_is_never_admitted(tmp_path):
    spec,output,requests,_=fixture(tmp_path)
    path=requests/'a.request.json.gz';body=json.loads(gzip.decompress(path.read_bytes()))
    body['input'].append({'type':'reasoning','encrypted_content':'private'})
    path.write_bytes(gzip.compress(json.dumps(body).encode()))
    with pytest.raises(ValueError,match='No public executable'):extract_decisions(spec,output)


def test_suite_and_task_balancing_does_not_flatten_the_largest_episode(tmp_path):
    rows=[{'id':str(i),'suite':'large','task':'one'} for i in range(100)]+[
         {'id':'b','suite':'small','task':'two'},{'id':'c','suite':'third','task':'three'}]
    result=balanced_order(rows)
    assert {r['suite'] for r in result[:3]}=={'large','small','third'}
    assert len({r['id'] for r in result})==len(rows)
