from copy import deepcopy

from reproductions.leact.cluster.efe_coverage_retry import init_interruption_proven,submission_priority,native_disconnect_retry_tasks


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


def test_new_quota_restores_high_priority_then_low_fallback():
    policy={'current_priority':'HIGH','fallback_priority':'LOW','updated_utc':'2026-11-14T01:40:00+00:00'}
    assert submission_priority({},policy)=='HIGH'
    assert submission_priority({'attempts':[{'submitted_utc':'2026-11-08T00:00:00+00:00'}]},policy)=='HIGH'
    assert submission_priority({'attempts':[{'submitted_utc':'2026-11-14T01:41:00+00:00'}]},policy)=='LOW'


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
