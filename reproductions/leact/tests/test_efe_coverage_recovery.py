from copy import deepcopy
import hashlib
import json
import pytest

from reproductions.leact.cluster.efe_coverage_retry import init_interruption_proven,submission_priority,native_disconnect_retry_tasks,cuda_unavailable_before_training,reviewed_preflight_retry
from reproductions.leact.cluster.efe_coverage_retry import backend_node_interruption_proven
from reproductions.leact.cluster.efe_coverage_retry import reviewed_hf_scoring_recovery


def startup_failure():
    return {'status':'FAILED','start_time':None,'groups':[{'tasks':[
        {'exit_code':4000,'start_time':None,'input_download_start_time':None}]}]}


EVENTS='Pulling image "nvcr.io/nvstaging/osmo/init-container:6.4.0"\nKilling: Stopping container osmo-init\n'


def test_review_requires_scheduler_and_event_evidence():
    assert init_interruption_proven(startup_failure(),EVENTS)
    assert not init_interruption_proven(startup_failure(),'Failed to start')
    assert not init_interruption_proven(startup_failure(),EVENTS+'Started container osmo-main')
    readiness='Pulling osmo/init-container:6.4.0\nPodReadyToStartContainers: True\nPodReadyToStartContainers: False'
    assert init_interruption_proven(startup_failure(),readiness)
    assert not init_interruption_proven(startup_failure(),readiness+'\nStarted container osmo-main')


def test_training_or_input_download_start_prevents_startup_retry():
    for key in ['start_time','input_download_start_time']:
        state=deepcopy(startup_failure());state['groups'][0]['tasks'][0][key]='2026-11-08T09:00:16'
        assert not init_interruption_proven(state,EVENTS)
    state=startup_failure();state['start_time']='2026-11-08T09:00:16'
    assert not init_interruption_proven(state,EVENTS)


def test_unknown_exit_code_does_not_become_retryable():
    state=startup_failure();state['groups'][0]['tasks'][0]['exit_code']=1
    assert not init_interruption_proven(state,EVENTS)


def test_pod_deleted_during_binding_is_retryable_only_before_execution():
    events='BindingError: pod example is being deleted, cannot be assigned to a host'
    assert init_interruption_proven(startup_failure(),events)
    assert not init_interruption_proven(startup_failure(),'BindingError: unknown scheduling failure')
    assert not init_interruption_proven(startup_failure(),events+'\nStarted container osmo-main')
    for key in ['start_time','input_download_start_time']:
        state=deepcopy(startup_failure());state['groups'][0]['tasks'][0][key]='2026-11-16T23:02:00'
        assert not init_interruption_proven(state,events)


def test_new_quota_restores_high_priority_then_low_fallback():
    policy={'current_priority':'HIGH','fallback_priority':'LOW','updated_utc':'2026-11-14T01:40:00+00:00'}
    assert submission_priority({},policy)=='HIGH'
    assert submission_priority({'attempts':[{'submitted_utc':'2026-11-08T00:00:00+00:00'}]},policy)=='HIGH'
    assert submission_priority({'attempts':[{'submitted_utc':'2026-11-14T01:41:00+00:00'}]},policy)=='LOW'


def test_backend_node_retry_requires_both_scheduler_code_and_unhealthy_node_eviction():
    scheduler={'status':'FAILED','groups':[{'tasks':[{'status':'FAILED_BACKEND_ERROR','exit_code':3001}]}]}
    events=('NodeNotReady: Node is not ready\nTaintManagerEviction: Marking for deletion Pod example\n'
            'DisruptionTarget: True, Message: Taint manager: deleting due to NoExecute taint')
    assert backend_node_interruption_proven(scheduler,events)
    for key,value in [('status','FAILED'),('exit_code',1),('exit_code',4000)]:
        changed=deepcopy(scheduler);changed['groups'][0]['tasks'][0][key]=value
        assert not backend_node_interruption_proven(changed,events)
    assert not backend_node_interruption_proven(scheduler,'NodeNotReady: Node is not ready')
    assert not backend_node_interruption_proven({'status':'FAILED','groups':[]},events)
    changed=deepcopy(scheduler);changed['groups'][0]['tasks'].append({'status':'FAILED','exit_code':1})
    assert not backend_node_interruption_proven(changed,events)


def test_hf_scoring_repair_requires_verified_promotion_and_no_production_updates():
    scheduler={'name':'stopped-r1','status':'FAILED'}
    review={'workflow':'stopped-r1','action':'resume_with_training_backend_admissions',
        'source_spec_sha256':'frozen-spec','original_dataset_sha256':'original-admission'}
    evidence={'error':{'workflow':'stopped-r1','error':
        'smoke exited 1: Training/selection complete-target likelihood differs'},
        'scoring_recovery':{'success':True,'production_optimizer_updates_before_recovery':0,
            'original_dataset_sha256':'original-admission','dataset_sha256':'hf-admission',
            'selection':{'score_backend':'exact_training_fsdp_fp32_master_bf16_forward'}},
        'dataset':{'sha256':'hf-admission'},'committed_restart_step':0,'train_complete':{}}
    assert reviewed_hf_scoring_recovery(scheduler,evidence,review,'frozen-spec')
    assert not reviewed_hf_scoring_recovery(scheduler,evidence,review,'changed-spec')
    assert not reviewed_hf_scoring_recovery({'name':'later-unrelated-failure','status':'FAILED'},evidence,review,'frozen-spec')
    for key,value in [('committed_restart_step',1),('train_complete',{'success':True}),
                      ('dataset',{'sha256':'different-admission'}),('scoring_recovery',{})]:
        changed=deepcopy(evidence);changed[key]=value
        assert not reviewed_hf_scoring_recovery(scheduler,changed,review,'frozen-spec')


def test_explicit_preemption_before_start_retries_without_guessing_other_exit_4000_causes():
    events='Pipelined: Pod example\nEvict: Pod example was preempted by higher priority workload other'
    assert init_interruption_proven(startup_failure(),events)
    assert not init_interruption_proven(startup_failure(),'Pipelined: Pod example')
    assert not init_interruption_proven(startup_failure(),'Evict: Pod example unknown reason')
    assert not init_interruption_proven(startup_failure(),events+'\nStarted container osmo-main')
    for key in ['start_time','input_download_start_time']:
        state=deepcopy(startup_failure());state['groups'][0]['tasks'][0][key]='2026-11-17T04:00:00'
        assert not init_interruption_proven(state,events)


def native_interruption():
    marker='Native episode infrastructure/resource verification failed; inspect the recorded trial before expansion'
    return {'results':{'setup_error':marker,'trials':[
        {'task':'graded_failure','status':'graded','score':0},
        {'task':'interrupted','status':'infrastructure_failure','score':None,'resource_metrics':{},
         'action_metrics':{'error':'RemoteDisconnected: Remote end closed connection without response','valid_actions':6}}]},
        'complete':{'error':marker,'completed_trials':2,'planned_trials':3}}


SPEC={'tasks':[{'task':name} for name in ['graded_failure','interrupted','untouched']]}


def test_native_disconnect_preserves_graded_failure():
    assert native_disconnect_retry_tasks(SPEC,native_interruption())==['interrupted','untouched']


def test_native_unknown_missing_or_scored_outcome_cannot_retry():
    for key,value in [('status','unknown'),('score',0),('resource_metrics',{'success':0})]:
        evidence=native_interruption();evidence['results']['trials'][-1][key]=value
        assert native_disconnect_retry_tasks(SPEC,evidence)==[]
    evidence=native_interruption();evidence['results']['trials'].pop()
    assert native_disconnect_retry_tasks(SPEC,evidence)==[]
    evidence=native_interruption();evidence['results']['trials'].reverse()
    assert native_disconnect_retry_tasks(SPEC,evidence)==[]


def test_native_model_error_cannot_retry():
    evidence=native_interruption()
    evidence['results']['trials'][-1]['action_metrics']['error']='Invalid JSON action'
    assert native_disconnect_retry_tasks(SPEC,evidence)==[]


def cuda_startup_evidence():
    return {'worker_start':{'workflow':'allocation-1'},
        'progress':{'workflow':'allocation-1','phase':'restart_validation','training_started':False},
        'error':{'workflow':'allocation-1','error':
            'restart-empty-shard-probe exited 1: Traceback\n'
            'torch.cuda.set_device(local)\ntorch._C._cuda_setDevice(device)\n'
            'CUDA error: CUDA-capable device(s) is/are busy or unavailable'}}


def test_initial_device_unavailability_can_retry_without_bypassing_training_guards():
    scheduler={'name':'allocation-1','status':'FAILED'}
    assert cuda_unavailable_before_training(scheduler,cuda_startup_evidence())
    for key,value in [('phase','training'),('training_started',True),('workflow','older-allocation')]:
        evidence=cuda_startup_evidence();evidence['progress'][key]=value
        assert not cuda_unavailable_before_training(scheduler,evidence)
    for message in ['CUDA error: device-side assert triggered','Full-model replay mismatch',
                    'CUDA-capable device(s) is/are busy or unavailable']:
        evidence=cuda_startup_evidence();evidence['error']['error']=message
        assert not cuda_unavailable_before_training(scheduler,evidence)
    assert not cuda_unavailable_before_training({'name':'allocation-2','status':'FAILED'},cuda_startup_evidence())
    assert not cuda_unavailable_before_training({'name':'allocation-1','status':'RUNNING'},cuda_startup_evidence())


@pytest.mark.parametrize('action',['retry_original_preflight_once','retry_with_matched_constrained_preflight'])
def test_reviewed_preflight_retry_never_repeats_a_benchmark_attempt(action):
    name='failed-preflight';marker='Student vision/function-call preflight failed; do not run benchmark tasks'
    scheduler={'name':name,'status':'FAILED'}
    evidence={'progress':{'workflow':name,'phase':'qwen_baseline_needs_review','benchmark_rollouts_started':False},
        'preflight':{'status':'failed','error':marker},'worker_complete':{'workflow':name,'success':False,'error':marker},
        'complete':{'completed_trials':0,'error':marker},'results':{},'public_manifest':{}}
    review={'workflow':name,'action':action,'source_spec_sha256':'source',
        'target_preflight_protocol':'constrained-image-action-preflight-v1',
        'evidence_sha256':hashlib.sha256(json.dumps(evidence,sort_keys=True).encode()).hexdigest()}
    assert reviewed_preflight_retry(scheduler,evidence,review,'source')
    assert not reviewed_preflight_retry(scheduler,evidence,{},'source')
    assert not reviewed_preflight_retry(scheduler,evidence,review,'changed')
    assert not reviewed_preflight_retry({'name':'retry-failed','status':'FAILED'},evidence,review,'source')
    for field,changed in [('results',{'trials':[{'status':'graded','score':0}]}),
                           ('public_manifest',{'episodes':[{'score':0}]}),
                           ('progress',{'workflow':name,'phase':'qwen_baseline_needs_review','benchmark_rollouts_started':True})]:
        current=deepcopy(evidence);current[field]=changed
        current_review={**review,'evidence_sha256':hashlib.sha256(json.dumps(current,sort_keys=True).encode()).hexdigest()}
        assert not reviewed_preflight_retry(scheduler,current,current_review,'source')
