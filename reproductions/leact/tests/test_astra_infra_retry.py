import copy
import fcntl
import json
from pathlib import Path
import time

import pytest

from reproductions.leact.cluster import efe_astra_infra_retry as retry


def fixture(tmp_path):
    root=tmp_path/(retry.OWNER+'-astra62-xhigh-v1')
    root.mkdir()
    case={'case_id':'rp_jhh_case','task':'robodojo-task','suite':'robodojo','partition':'train',
          'admitted':True,'scene_sha256':'a'*64,'config_id':'training-reset','env_args':{'seed':1}}
    spec={'owner_model_workflow':retry.OWNER,'teacher_root':str(root),
          'transport_root':str(tmp_path/retry.OWNER/'transport'),'teacher_profile_id':'astra62-xhigh-v1',
          'suite':'robodojo','teacher_task_names':[case['task']],
          'teacher_plan':{'partition':'train','curriculum_sha256':'c'*64,'cases':[case]}}
    folder=root/'episodes'/case['case_id'];folder.mkdir(parents=True)
    return spec,case,folder


def attempt(spec,case,folder,number,**updates):
    path=folder/f'infra-{number:02d}';path.mkdir(exist_ok=True)
    row={'case':copy.deepcopy(case),'status':'infrastructure_failure','infrastructure_attempt':number,
         'requested_model':'openai/openai/gpt-6.2-astra','requested_effort':'xhigh',
         'curriculum_sha256':spec['teacher_plan']['curriculum_sha256'],
         'error':'Native simulator worker heartbeat expired',**updates}
    (path/'result.json').write_text(json.dumps(row))
    return path,row


def exhausted(spec,case,folder):
    for i in range(3):attempt(spec,case,folder,i)


def test_plan_keeps_cases_recipe_and_originals_while_reserving_only_three_new_ids(tmp_path):
    spec,case,folder=fixture(tmp_path);exhausted(spec,case,folder)
    before={str(p):p.read_bytes()for p in folder.rglob('*.json')}
    manifest=retry.prepare(spec)
    assert manifest['identity']['new_attempts_per_case']==3
    assert manifest['selected'][0]['case']==case
    assert manifest['selected'][0]['new_infrastructure_attempts']==[3,4,5]
    assert manifest['training_recipe_changed'] is False and manifest['model_calls_started']==0
    assert {str(p):p.read_bytes()for p in folder.rglob('*.json')}==before
    assert retry.prepare(spec)['id']==manifest['id']


@pytest.mark.parametrize('kind',['graded_success','graded_failure','finalized_flag','native_verdict',
    'malformed_verdict','inflight_original','missing_original','mismatched_scene','mismatched_model','already_retried'])
def test_unqualified_or_any_previously_graded_case_never_enters_retry_cohort(tmp_path,kind):
    spec,case,folder=fixture(tmp_path);exhausted(spec,case,folder)
    path=folder/'infra-01/result.json';row=json.loads(path.read_text())
    if kind in{'graded_success','graded_failure'}:row.update(status=kind,finalized_native_outcome=True)
    elif kind=='finalized_flag':row['finalized_native_outcome']=True
    elif kind in{'native_verdict','malformed_verdict'}:
        (path.parent/'native-verdict.json').write_text('{"finalized":true,"success":false}'if kind=='native_verdict'else'broken')
    elif kind=='inflight_original':row['status']='opening'
    elif kind=='missing_original':path.unlink()
    elif kind=='mismatched_scene':row['case']['scene_sha256']='b'*64
    elif kind=='mismatched_model':row['requested_model']='azure/openai/gpt-6-astra'
    elif kind=='already_retried':attempt(spec,case,folder,3)
    if kind!='missing_original':path.write_text(json.dumps(row))
    manifest=retry.prepare(spec)
    assert manifest['selected']==[] and len(manifest['excluded'])==1


def test_newer_native_failure_is_not_hidden_behind_an_older_interrupted_retry(tmp_path):
    spec,case,folder=fixture(tmp_path);exhausted(spec,case,folder)
    attempt(spec,case,folder,3,status='opening')
    attempt(spec,case,folder,4,status='graded_failure',native_success=False,finalized_native_outcome=True)
    assert retry.history(folder)[1]=='prior_native_outcome'
    assert retry.prepare(spec)['selected']==[]


@pytest.mark.parametrize('field,value',[('owner_model_workflow','hero'),('wall_seconds',3600),
    ('max_calls',192),('token_budget',320000),('max_infrastructure_attempts',6),
    ('teacher_profile_id','astra6-high-v1'),('control_interface','pear-eef-native-v1')])
def test_recipe_or_owner_change_is_rejected_before_creating_attempts(tmp_path,field,value):
    spec,case,folder=fixture(tmp_path);spec[field]=value
    with pytest.raises(ValueError):retry.prepare(spec)
    assert list(folder.iterdir())==[]


def test_runner_stops_on_native_failure_and_never_retries_it_on_resume(tmp_path,monkeypatch):
    spec,case,folder=fixture(tmp_path);exhausted(spec,case,folder)
    manifest=retry.prepare(spec);before={str(p):p.read_bytes()for p in folder.rglob('*.json')};calls=[]
    monkeypatch.setattr(retry,'verify_health',lambda *args:None)
    def execute(spec,case,*,infrastructure_attempt):
        calls.append(infrastructure_attempt)
        status='graded_failure'if infrastructure_attempt==4 else'infrastructure_failure'
        _,row=attempt(spec,case,folder,infrastructure_attempt,status=status,
            finalized_native_outcome=status=='graded_failure',native_success=False)
        return row
    monkeypatch.setattr(retry,'run_case',execute)
    result=retry.run(spec,manifest,{},tmp_path/'cohort')
    assert calls==[3,4] and result['events'][-1]['status']=='graded_failure'
    retry.run(spec,manifest,{},tmp_path/'cohort')
    assert calls==[3,4] and not(folder/'infra-05').exists()
    assert all(Path(p).read_bytes()==raw for p,raw in before.items())


def test_all_infrastructure_failures_are_bounded_and_replay_uses_existing_ids(tmp_path,monkeypatch):
    spec,case,folder=fixture(tmp_path);exhausted(spec,case,folder);manifest=retry.prepare(spec);calls=[]
    monkeypatch.setattr(retry,'verify_health',lambda *args:None)
    def execute(spec,case,*,infrastructure_attempt):
        existing=folder/f'infra-{infrastructure_attempt:02d}/result.json'
        if existing.exists():return json.loads(existing.read_text())
        calls.append(infrastructure_attempt)
        return attempt(spec,case,folder,infrastructure_attempt)[1]
    monkeypatch.setattr(retry,'run_case',execute)
    retry.run(spec,manifest,{},tmp_path/'cohort');retry.run(spec,manifest,{},tmp_path/'cohort')
    assert calls==[3,4,5] and not(folder/'infra-06').exists()
    assert retry.prepare(spec)['selected']==[]


def test_live_case_writer_is_preserved_without_model_call(tmp_path,monkeypatch):
    spec,case,folder=fixture(tmp_path);exhausted(spec,case,folder);manifest=retry.prepare(spec)
    monkeypatch.setattr(retry,'run_case',lambda *args,**kwargs:pytest.fail('Existing writer must be preserved'))
    with(folder/'writer.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        result=retry.run(spec,manifest,{},tmp_path/'cohort')
    assert result['events']==[{'case_id':case['case_id'],'status':'skipped_active_case_writer'}]


def test_modified_original_or_manifest_never_starts_a_model_call(tmp_path,monkeypatch):
    spec,case,folder=fixture(tmp_path);exhausted(spec,case,folder);manifest=retry.prepare(spec)
    monkeypatch.setattr(retry,'run_case',lambda *args,**kwargs:pytest.fail('No model call'))
    bad=copy.deepcopy(manifest);bad['identity']['cases'].append('other')
    with pytest.raises(ValueError,match='identity changed'):retry.run(spec,bad,{},tmp_path/'bad-manifest')
    (folder/'infra-00/result.json').write_text('{}')
    with pytest.raises(ValueError,match='Original trace changed'):retry.run(spec,manifest,{},tmp_path/'changed')


def test_actual_owned_fresh_native_health_is_required(tmp_path):
    spec,_,_=fixture(tmp_path)
    health={'teacher_profile_id':'astra62-xhigh-v1','suite':'robodojo','source_payload_verified':True,
            'native_reset_open_verified':True,'same_native_action_contract':True,
            'action_contract_sha256':'d'*64,'rpc_worker_id':'rp-jhh-astra-wide-native-fixed-job'}
    path=Path(spec['transport_root'])/'workers'/(health['rpc_worker_id']+'.json');path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'worker_id':health['rpc_worker_id'],'updated_unix':time.time(),'phase':'ready'}))
    retry.verify_health(spec,health)
    path.write_text(json.dumps({'worker_id':health['rpc_worker_id'],'updated_unix':time.time()-121,'phase':'ready'}))
    with pytest.raises(RuntimeError,match='heartbeat'):retry.verify_health(spec,health)
    with pytest.raises(RuntimeError,match='contract/reset'):retry.verify_health(spec,{**health,'same_native_action_contract':False})
