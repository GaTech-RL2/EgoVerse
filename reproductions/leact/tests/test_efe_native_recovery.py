import hashlib
import json

import pytest
import yaml

from reproductions.leact.cluster.recover_efe_native import nccl_startup_interruption,retry_spec
from reproductions.leact.cluster import recover_efe_native as recovery


SIGNATURE=('WorkerProc failed to start.\ninit_worker_distributed_environment\n'
           'ncclCommInitRank\nRuntimeError: NCCL error: unhandled cuda error')


def diagnostic(tmp_path,text=SIGNATURE):
    (tmp_path/'student-serving-failure.json').write_text(json.dumps({
        'log_tail':text,'log_sha256':hashlib.sha256(text.encode()).hexdigest()}))


def test_recognize_pre_serving_communicator_failure(tmp_path):
    diagnostic(tmp_path)
    assert nccl_startup_interruption({'status':'FAILED','name':'fixture'},tmp_path)
    receipt=json.loads((tmp_path/'interruption.json').read_text())
    assert receipt['kind']=='nccl_initialization_failure' and receipt['before_model_serving']


@pytest.mark.parametrize('suffix',['CUDA out of memory','torch.OutOfMemoryError',
                                  'Model loading took','Application startup complete'])
def test_never_treat_model_or_memory_failure_as_startup_transient(tmp_path,suffix):
    diagnostic(tmp_path,SIGNATURE+'\n'+suffix)
    assert not nccl_startup_interruption({'status':'FAILED','name':'fixture'},tmp_path)


def test_completed_requests_or_manual_cancel_are_not_startup_retry(tmp_path):
    diagnostic(tmp_path)
    assert not nccl_startup_interruption({'status':'FAILED_CANCELED','name':'fixture'},tmp_path)
    (tmp_path/'transport.json').write_text(json.dumps({'completed':1}))
    assert not nccl_startup_interruption({'status':'FAILED','name':'fixture'},tmp_path)


def test_generic_cuda_error_requires_review(tmp_path):
    diagnostic(tmp_path,'RuntimeError: NCCL error: unhandled cuda error')
    assert not nccl_startup_interruption({'status':'FAILED','name':'fixture'},tmp_path)


def test_explicit_nvlink_fault_is_distinguished_from_generic_initialization(tmp_path):
    diagnostic(tmp_path,SIGNATURE+'\nP2P is disabled between NVLINK connected GPUs 1 and 0')
    assert nccl_startup_interruption({'status':'FAILED','name':'fixture'},tmp_path)
    assert json.loads((tmp_path/'interruption.json').read_text())['kind']=='gpu_interconnect_failure'


def test_transport_retry_preserves_protocol_and_pins_prior_grade_sources(tmp_path):
    previous={'archive_root':'/archive/old','tasks':[{'task':'one'}],
              'student_checkpoint':'step32','sampling':{'temperature':0.7},
              'max_output_tokens':32768,'resume_completed_trials':{'archive':'older','workflow':'earlier'}}
    frozen=json.loads(json.dumps(previous))
    transport={'mode':'nccl_no_p2p_v1','speed_comparison_eligible':False}
    spec=retry_spec(previous,'parent-1',tmp_path,transport)
    assert previous==frozen
    assert all(spec[k]==previous[k] for k in ('tasks','student_checkpoint','sampling','max_output_tokens'))
    resume=spec['resume_completed_trials']
    assert resume['source_spec_sha256']==hashlib.sha256(json.dumps(previous,sort_keys=True).encode()).hexdigest()
    assert resume['fallback_sources']==[previous['resume_completed_trials']]
    assert spec['serving_transport']==transport


def test_final_evaluation_retry_keeps_its_own_paths_and_step32_baseline(tmp_path,monkeypatch):
    root=tmp_path/'artifacts';(root/'training-pilot').mkdir(parents=True)
    base=tmp_path/'base';(base/'cluster').mkdir(parents=True)
    builds=tmp_path/'builds';source=builds/'final';source.mkdir(parents=True)
    for name in ['efe_astra_pilot.py','efe_qwen_baseline.py']:(base/'cluster'/name).write_text('# fixture\n')
    original={'archive_root':'/old','tasks':[{'task':'one'}],'student_checkpoint':'step96'}
    files={'spec.json':json.dumps(original),'workflow.yaml':yaml.safe_dump({'workflow':{'name':'old'}}),
           'runtime.py':'# runtime\n','pilot.py':'# pilot\n','student.py':'# student\n'}
    for name,data in files.items():(source/name).write_text(data)
    (source/'source-manifest.json').write_text(json.dumps({k:hashlib.sha256(v.encode()).hexdigest() for k,v in files.items()}))
    continuation={'arms':[{'id':'final','baseline':'step32','evaluation_directory':'old'}]}
    (root/'training-pilot/continuation-recovery.json').write_text(json.dumps(continuation))
    for key,value in [('ROOT',root),('BASE',base),('BUILD_ROOT',builds),('PLAN',root/'plan.json')]:monkeypatch.setattr(recovery,key,value)
    plan={'build':str(source),'attempt':1,'build_prefix':'final-retry-',
          'workflow_prefix':'leact-final-r','local_results_prefix':'training-pilot/final-retry-',
          'evaluation_stage':'final','continuation_arm_id':'final'}
    target=recovery.next_build(plan,'parent-1')
    assert target.name=='final-retry-2'
    assert yaml.safe_load((target/'workflow.yaml').read_text())['workflow']['name']=='leact-final-r2'
    spec=json.loads((target/'spec.json').read_text())
    assert spec['student_checkpoint']=='step96' and spec['local_results_directory']=='training-pilot/final-retry-2'
    arm=json.loads((root/'training-pilot/continuation-recovery.json').read_text())['arms'][0]
    assert arm['baseline']=='step32' and arm['evaluation_directory']==spec['local_results_directory']


def test_completed_worker_recovers_receipts_before_any_policy_retry(tmp_path,monkeypatch):
    root=tmp_path/'artifacts';root.mkdir();build=tmp_path/'build';build.mkdir()
    (build/'submission.json').write_text(json.dumps({'name':'finished'}))
    (build/'spec.json').write_text(json.dumps({'local_results_directory':'evaluation'}))
    monkeypatch.setattr(recovery,'ROOT',root)
    monkeypatch.setattr(recovery,'command',lambda args:{'name':'finished','status':'COMPLETED'})
    monitors=[];monkeypatch.setattr(recovery,'monitor',monitors.append)
    result=recovery.advance({'build':str(build),'attempt':1})
    assert result['status']=='collecting' and monitors==[build]
