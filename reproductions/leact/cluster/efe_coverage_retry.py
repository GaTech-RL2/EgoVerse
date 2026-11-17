"""Backoff for verified infrastructure failures under the existing node cap."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import subprocess


ROOT=Path(__file__).resolve().parents[1]/'artifacts/embodied-first-exam'


def submission_priority(record,policy=None):
    """Use the newly authorized priority once, then the user's LOW fallback."""
    if policy is None:
        path=ROOT/'osmo-priority-policy.json'
        policy=json.loads(path.read_text()) if path.exists() else {}
    preferred=policy.get('current_priority','HIGH')
    attempts=record.get('attempts',[])
    updated=policy.get('updated_utc')
    if attempts and updated:
        submitted=attempts[-1].get('submitted_utc')
        if submitted and datetime.fromisoformat(submitted)>=datetime.fromisoformat(updated):
            return policy.get('fallback_priority','LOW')
    return preferred


def never_started_init_failure(scheduler):
    if scheduler.get('status')!='FAILED' or scheduler.get('start_time'):
        return False
    tasks=[t for g in scheduler.get('groups',[]) for t in g.get('tasks',[])]
    return bool(tasks) and all(t.get('exit_code')==4000 and not t.get('start_time')
        and not t.get('input_download_start_time') for t in tasks)


def init_interruption_proven(scheduler,events):
    """Exit 4000 alone is insufficient; require the observed startup interruption."""
    readiness_lost=(events.rfind('PodReadyToStartContainers: False')>
                    events.rfind('PodReadyToStartContainers: True')>
                    events.find('osmo/init-container:')>=0)
    deleted_during_binding=('BindingError' in events
        and 'is being deleted, cannot be assigned to a host' in events)
    preempted_before_start=('Evict: Pod ' in events
        and ' was preempted by higher priority workload ' in events)
    return (never_started_init_failure(scheduler)
        and (deleted_during_binding or preempted_before_start or ('osmo/init-container:' in events
            and ('Stopping container osmo-init' in events or readiness_lost)))
        and 'Started container osmo-main' not in events)


def reviewed_init_interruption(scheduler):
    if not never_started_init_failure(scheduler):return False
    name=scheduler['name']
    directory=ROOT/'fixed-action-v1/coverage-scale-v1/initialization-reviews'
    directory.mkdir(parents=True,exist_ok=True)
    key=hashlib.sha256(name.encode()).hexdigest()[:16]
    record=directory/(key+'.json')
    if record.exists():
        value=json.loads(record.read_text())
        if value.get('rule_version')==4:
            return value['workflow']==name and value.get('retryable',False)
    result=subprocess.run(['osmo','workflow','events',name],capture_output=True,text=True,timeout=30)
    if result.returncode:return False
    events=result.stdout
    (directory/(key+'.events.txt')).write_text(events)
    value={'workflow':name,'utc':datetime.now(timezone.utc).isoformat(),'rule_version':4,
        'retryable':init_interruption_proven(scheduler,events),
        'events_sha256':hashlib.sha256(events.encode()).hexdigest(),
        'rule':'No workflow/task/input download start; exit 4000; events confirm interrupted OSMO init, pod deletion during binding, or explicit higher-priority preemption before task execution.'}
    temporary=record.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(record)
    return value['retryable']


def backend_node_interruption_proven(scheduler,events):
    """Require the scheduler's backend error and explicit unhealthy-node eviction."""
    tasks=[t for g in scheduler.get('groups',[]) for t in g.get('tasks',[])]
    return (scheduler.get('status')=='FAILED' and len(tasks)==1
        and tasks[0].get('status')=='FAILED_BACKEND_ERROR' and tasks[0].get('exit_code')==3001
        and 'NodeNotReady: Node is not ready' in events
        and 'TaintManagerEviction: Marking for deletion Pod ' in events
        and 'deleting due to NoExecute taint' in events)


def reviewed_backend_node_interruption(scheduler):
    tasks=[t for g in scheduler.get('groups',[]) for t in g.get('tasks',[])]
    if not (scheduler.get('status')=='FAILED' and len(tasks)==1
            and tasks[0].get('status')=='FAILED_BACKEND_ERROR' and tasks[0].get('exit_code')==3001):
        return False
    name=scheduler['name'];directory=ROOT/'fixed-action-v1/coverage-scale-v1/backend-node-reviews'
    directory.mkdir(parents=True,exist_ok=True)
    key=hashlib.sha256(name.encode()).hexdigest()[:16];path=directory/(key+'.json')
    if path.exists():
        value=json.loads(path.read_text())
        if value.get('rule_version')==1 and value.get('workflow')==name:return value.get('retryable',False)
    try:
        result=subprocess.run(['osmo','workflow','events',name],capture_output=True,text=True,timeout=12)
        if result.returncode:return False
        events=result.stdout
    except subprocess.TimeoutExpired as error:
        events=error.stdout or '';events=events.decode(errors='replace') if isinstance(events,bytes) else events
        if not backend_node_interruption_proven(scheduler,events):return False
    (directory/(key+'.events.txt')).write_text(events)
    value={'workflow':name,'rule_version':1,'utc':datetime.now(timezone.utc).isoformat(),
        'retryable':backend_node_interruption_proven(scheduler,events),
        'events_sha256':hashlib.sha256(events.encode()).hexdigest(),
        'rule':'Single task FAILED_BACKEND_ERROR/3001 plus NodeNotReady and explicit NoExecute-taint eviction.'}
    pending=path.with_suffix('.tmp');pending.write_text(json.dumps(value,indent=2)+'\n');pending.replace(path)
    return value['retryable']


def native_disconnect_retry_tasks(spec,evidence):
    """Only a recorded disconnect and the untouched suffix may be retried.

    The native pilot records each trial and stops immediately on this exact
    infrastructure-verification error. A missing/unknown result is not proof.
    """
    marker='Native episode infrastructure/resource verification failed; inspect the recorded trial before expansion'
    results=evidence.get('results',{})
    complete=evidence.get('complete',{})
    if results.get('setup_error')!=marker or complete.get('error')!=marker:return []
    planned=[row['task'] for row in spec.get('tasks',[])]
    trials=results.get('trials',[])
    if not trials or len(set(planned))!=len(planned):return []
    if complete.get('completed_trials')!=len(trials) or complete.get('planned_trials')!=len(planned):return []
    if [row.get('task') for row in trials]!=planned[:len(trials)]:return []
    if any(row.get('status')!='graded' or row.get('score') not in (0,1) for row in trials[:-1]):return []
    interrupted=trials[-1]
    metrics=interrupted.get('action_metrics',{})
    if (interrupted.get('status')!='infrastructure_failure' or interrupted.get('score') is not None
        or interrupted.get('resource_metrics',{}).get('success') is not None
        or metrics.get('error')!='RemoteDisconnected: Remote end closed connection without response'
        or metrics.get('valid_actions',0)<1):return []
    return planned[len(trials)-1:]


def cuda_unavailable_before_training(scheduler, evidence):
    """Retry only a matching worker's initial device acquisition failure.

    A generic CUDA exception, a failed optimizer/replay assertion, or evidence
    from an earlier allocation must remain under review.
    """
    name=scheduler.get('name')
    if not name or scheduler.get('status')!='FAILED':return False
    progress=evidence.get('progress',{})
    error=evidence.get('error',{})
    start=evidence.get('worker_start',{})
    if any(row.get('workflow')!=name for row in (progress,error,start)):return False
    if progress.get('phase')!='restart_validation' or progress.get('training_started') is not False:return False
    message=error.get('error','')
    return (message.startswith('restart-empty-shard-probe exited 1:')
        and 'torch.cuda.set_device(local)' in message
        and 'torch._C._cuda_setDevice(device)' in message
        and 'CUDA error: CUDA-capable device(s) is/are busy or unavailable' in message)


def reviewed_hf_scoring_recovery(scheduler,evidence,review,source_spec_sha256):
    """Release one reviewed R1 repair only after exact-score data promotion."""
    error=evidence.get('error',{});activation=evidence.get('scoring_recovery',{})
    if not(scheduler.get('status')=='FAILED' and scheduler.get('name')==review.get('workflow')
           and review.get('action')=='resume_with_training_backend_admissions'
           and review.get('source_spec_sha256')==source_spec_sha256
           and error.get('workflow')==scheduler['name']
           and error.get('error','').startswith('smoke exited 1:')
           and 'Training/selection complete-target likelihood differs' in error['error']):return False
    return (activation.get('success')is True and activation.get('production_optimizer_updates_before_recovery')==0
        and activation.get('original_dataset_sha256')==review.get('original_dataset_sha256')
        and activation.get('dataset_sha256')==evidence.get('dataset',{}).get('sha256')
        and activation.get('selection',{}).get('score_backend')=='exact_training_fsdp_fp32_master_bf16_forward'
        and evidence.get('committed_restart_step',0)==0 and not evidence.get('train_complete'))


def reviewed_preflight_retry(scheduler,evidence,review,spec_sha256):
    """One explicit diagnostic retry before any benchmark episode; no inferred cause."""
    name=scheduler.get('name')
    marker='Student vision/function-call preflight failed; do not run benchmark tasks'
    approved=(review.get('action')=='retry_original_preflight_once' or (
        review.get('action')=='retry_with_matched_constrained_preflight'
        and review.get('target_preflight_protocol')=='constrained-image-action-preflight-v1'))
    if (scheduler.get('status')!='FAILED' or not name or review.get('workflow')!=name
        or not approved
        or review.get('source_spec_sha256')!=spec_sha256):return False
    if review.get('evidence_sha256')!=hashlib.sha256(json.dumps(evidence,sort_keys=True).encode()).hexdigest():return False
    progress=evidence.get('progress',{});worker=evidence.get('worker_complete',{})
    return (progress.get('workflow')==name and progress.get('benchmark_rollouts_started') is False
        and progress.get('phase')=='qwen_baseline_needs_review'
        and evidence.get('preflight',{}).get('status')=='failed'
        and evidence['preflight'].get('error')==marker
        and evidence.get('complete',{}).get('completed_trials')==0
        and evidence['complete'].get('error')==marker
        and worker.get('workflow')==name and worker.get('success') is False and worker.get('error')==marker
        and not evidence.get('results',{}).get('trials')
        and not evidence.get('public_manifest',{}).get('episodes'))


def retry_due(record, workflow, at=None):
    """Call only after the scheduler failure has been classified as infrastructure."""
    at=at or datetime.now(timezone.utc)
    if record.get('infrastructure_retry_workflow')!=workflow:
        count=record.get('infrastructure_failure_count',0)+1
        delay=min(300,30*(2**min(count-1,4)))
        record.update(infrastructure_failure_count=count,
                      infrastructure_retry_workflow=workflow,
                      retry_not_before=(at+timedelta(seconds=delay)).isoformat())
    return at>=datetime.fromisoformat(record['retry_not_before'])
