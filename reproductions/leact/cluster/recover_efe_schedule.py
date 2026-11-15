"""Retry interrupted EFE comparisons within bounded training-node reservations."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import fcntl
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import time

import yaml

from .scale_efe import BASE, ROOT, read, write, submit, command
from .launch_efe_training import freeze_sources
from .launch_efe_schedule import ensure_monitor
from ..embodied_first_exam.report import publish

PLAN=ROOT/'training-pilot/schedule-recovery.json'
STATE=ROOT/'training-pilot/schedule-recovery-status.json'
LOCK=STATE.parent/'schedule-recovery.lock'


def capacity_interruption(state):
    messages=' '.join(g.get('failure_message','') or '' for g in state.get('groups',[])).lower()
    tasks=[t for g in state.get('groups',[]) for t in g.get('tasks',[])]
    control_failure=(state['status']=='FAILED' and 'osmo control failure' in messages
                     and any(t.get('exit_code') in (2001,2143) for t in tasks))
    return (control_failure or state['status'] in {'FAILED_PREEMPTED','FAILED_QUEUE_TIMEOUT',
                                                 'FAILED_SERVER_ERROR','FAILED_BACKEND_ERROR','FAILED_EVICTED'}
            or state['status']=='FAILED_CANCELED' and 'gpu quota enforcement' in messages
            or state['status'].startswith('FAILED') and ('preempt' in messages or 'queue timeout' in messages))


def capture_control_failure(state,dest):
    """OSMO can report priority preemption only in events, as generic exit 2001."""
    tasks=[t for g in state.get('groups',[]) for t in g.get('tasks',[])]
    if not any(t.get('exit_code') in (2001,2143) for t in tasks):return
    path=dest/'interruption.json';previous=read(path,{})
    if previous.get('workflow')==state['name']:return
    try:
        result=subprocess.run(['osmo','workflow','events',state['name']],capture_output=True,text=True,timeout=30)
    except (OSError,subprocess.TimeoutExpired):return
    if result.returncode:return
    evidence=[line for line in result.stdout.splitlines()
              if 'Evict:' in line and 'was preempted by higher priority workload' in line]
    write(path,{'workflow':state['name'],'kind':'priority_preemption' if evidence else 'osmo_control_failure',
                'evidence':evidence,'updated_utc':datetime.now(timezone.utc).isoformat()})


def gpu_interconnect_interruption(state,dest):
    """Retry only the verified pre-model NCCL peer-access fault, not arbitrary CUDA errors."""
    if state['status']!='FAILED':return False
    metrics=dest/'train/metrics.jsonl'
    if metrics.exists() and metrics.stat().st_size:return False
    signature='P2P is disabled between NVLINK connected GPUs'
    matches=[p for p in (dest/'smoke.log',dest/'train.log') if p.exists()
             and all(s in p.read_text() for s in (signature,'ncclUnhandledCudaError','dist.all_reduce(probe)'))]
    if not matches:return False
    log=matches[0];text=log.read_text()
    evidence=list(dict.fromkeys(line.split(']: ',1)[-1] for line in text.splitlines() if signature in line))
    write(dest/'interruption.json',{'workflow':state['name'],'kind':'gpu_interconnect_failure',
        'evidence':evidence,'source':log.name,'source_sha256':hashlib.sha256(log.read_bytes()).hexdigest(),
        'before_model_loading':True,
        'nodes':sorted({t['node_name'] for g in state.get('groups',[]) for t in g.get('tasks',[]) if t.get('node_name')}),
        'updated_utc':datetime.now(timezone.utc).isoformat()})
    return True


def initialization_interruption(state,dest):
    """Retry the observed OSMO init exit only when user containers never started."""
    tasks=[t for g in state.get('groups',[]) for t in g.get('tasks',[])]
    metrics=dest/'train/metrics.jsonl'
    if (state['status']!='FAILED' or state.get('start_time','unknown') is not None
            or not tasks or any(t.get('start_time','unknown') is not None for t in tasks)
            or metrics.exists() and metrics.stat().st_size):return False
    failed=[t for t in tasks if t.get('exit_code')==392
            and 'Task osmo-init failure' in (t.get('failure_message') or '')]
    if not failed:return False
    write(dest/'interruption.json',{'workflow':state['name'],'kind':'osmo_initialization_failure',
        'evidence':[t['failure_message'] for t in failed],'source':'scheduler.json',
        'before_user_container_started':True,
        'nodes':sorted({t['node_name'] for t in failed if t.get('node_name')}),
        'updated_utc':datetime.now(timezone.utc).isoformat()})
    return True


def retry_delay(plan,arm):
    policy=plan['retry_policy']
    return min(policy['backoff_seconds']*2**min(arm.get('attempt',1)-1,8),policy['max_backoff_seconds'])


def next_attempt(plan,arm,previous_workflow):
    """Copy the original immutable recipe, assigning fresh receipt directories."""
    source=Path(arm['original_build']);attempt=arm.get('attempt',1)+1
    build=source.with_name(source.name+f'-quota-retry{attempt}');build.mkdir(exist_ok=True)
    spec=read(source/'spec.json')
    reason=('Authorized infrastructure retry; resume the latest committed continuation state.'
            if spec.get('continuation') else 'Authorized capacity retry; unchanged recipe, restarting from base.')
    spec.update(archive_root=spec['archive_root']+f'-quota-retry{attempt}',
                local_results_directory=arm['original_local_results_directory']+f'-quota-retry{attempt}',
                recovery_of=previous_workflow,
                recovery_reason=reason)
    if (build/'source-manifest.json').exists():
        if read(build/'spec.json')!=spec:raise ValueError('Frozen retry specification changed')
    else:
        for name,digest in read(source/'source-manifest.json').items():
            if hashlib.sha256((source/name).read_bytes()).hexdigest()!=digest:
                raise ValueError('Original frozen launch changed: '+name)
            if name not in ('spec.json','workflow.yaml'):shutil.copyfile(source/name,build/name)
        write(build/'spec.json',spec)
        workflow=yaml.safe_load((source/'workflow.yaml').read_text());workflow['workflow']['name']+=f'-retry{attempt}'
        (build/'workflow.yaml').write_text(yaml.safe_dump(workflow,sort_keys=False));freeze_sources(build)
    prior={k:arm[k] for k in ('build','local_results_directory')}
    prior.update(attempt=attempt-1,workflow=previous_workflow)
    arm.setdefault('previous_attempts',[]).append(prior)
    arm.update(attempt=attempt,build=str(build),local_results_directory=spec['local_results_directory'])
    dest=ROOT/arm['local_results_directory'];dest.mkdir(parents=True,exist_ok=True)
    write(dest/'scheduler.json',{'status':'WAITING','priority':plan.get('priority','LOW')})
    write(dest/'progress.json',{'phase':'waiting','detail':'Capacity retry prepared; awaiting scheduler submission.'})
    write(PLAN,plan)


def enable_retries(training_nodes=None):
    plan=prepare();now=datetime.now(timezone.utc).isoformat()
    if training_nodes is not None:
        if training_nodes not in (1,2):raise ValueError('Training concurrency must stay within two nodes')
        plan['max_concurrent_training_nodes']=training_nodes
    plan.update(attempts_per_arm=None,priority='LOW',
        reason='User requested continued retries and LOW priority whenever HIGH cannot obtain GPUs.',
        retry_policy={'enabled':True,'backoff_seconds':300,'max_backoff_seconds':900,
                      'authorized_utc':now,'retry_on':'Quota reclamation, preemption, queue timeout, eviction, server/backend failure, verified OSMO Control exit 2001/2143, pre-start OSMO init exit 392, or verified pre-model NVLink peer-access failure; review other training failures independently.'})
    for arm in plan['arms']:
        arm.setdefault('attempt',1);arm.setdefault('previous_attempts',[])
        dest=ROOT/arm['local_results_directory']
        if not (Path(arm['build'])/'submission.json').exists():
            write(dest/'scheduler.json',{'status':'WAITING','priority':'LOW'})
            write(dest/'progress.json',{'phase':'waiting','detail':'Waiting behind the preceding EFE training run; LOW-priority fallback enabled.'})
    write(PLAN,plan)
    write(ROOT/'osmo-priority-policy.json',{'preferred_priority':'HIGH','fallback_priority':'LOW','current_priority':'LOW',
         'updated_utc':now,'scope':'New authorized EFE workflow submissions, including training and downstream evaluations.',
         'reason':'HIGH-priority attempts were reclaimed by GPU quota enforcement; user explicitly requested LOW fallback.',
         'user_instruction':"whenever u can't get gpus try low priority"})
    return plan


def prepare():
    if PLAN.exists():return read(PLAN)
    original=read(ROOT/'training-pilot/schedule-comparison.json');arms=[]
    # The paper-recipe reference goes first; the data, seed and budget stay fixed.
    for lr in (1e-4,1e-5,1e-6):
        arm=next(a for a in original['arms'] if a['lr']==lr)
        source=Path(arm['build']);prior=read(source/'submission.json')['name']
        state=command(['osmo','workflow','query',prior,'--format-type','json'])
        if state['status']!='FAILED_CANCELED' or not any('GPU quota enforcement' in g.get('failure_message','') for g in state.get('groups',[])):
            raise ValueError('Automatic recovery is restricted to the verified quota cancellations: '+prior)
        if read(ROOT/arm['local_results_directory']/'train/complete.json',{}).get('success'):
            raise ValueError('Completed training must not be restarted: '+prior)
        build=source.with_name(source.name+'-quota-retry1');build.mkdir(exist_ok=True)
        spec=read(source/'spec.json')
        spec.update(archive_root=spec['archive_root']+'-quota-retry1',
                    local_results_directory=arm['local_results_directory']+'-quota-retry1',
                    recovery_of=prior,recovery_reason='One retry after cluster GPU quota enforcement; unchanged recipe, restarting from base.')
        if (build/'source-manifest.json').exists():
            if read(build/'spec.json')!=spec:raise ValueError('Frozen recovery spec changed')
        else:
            for name,digest in read(source/'source-manifest.json').items():
                if hashlib.sha256((source/name).read_bytes()).hexdigest()!=digest:
                    raise ValueError('Original frozen launch changed: '+name)
                if name not in ('spec.json','workflow.yaml'):shutil.copyfile(source/name,build/name)
            write(build/'spec.json',spec)
            workflow=yaml.safe_load((source/'workflow.yaml').read_text())
            workflow['workflow']['name']+='-retry1'
            (build/'workflow.yaml').write_text(yaml.safe_dump(workflow,sort_keys=False));freeze_sources(build)
        dest=ROOT/spec['local_results_directory'];dest.mkdir(parents=True,exist_ok=True)
        if not (dest/'scheduler.json').exists():
            write(dest/'scheduler.json',{'status':'WAITING'})
            write(dest/'progress.json',{'phase':'waiting','detail':'One-node recovery queue; waiting for the preceding training attempt.'})
        arms.append({'id':arm['id'],'build':str(build),'local_results_directory':spec['local_results_directory'],
                     'original_build':str(source),'original_local_results_directory':arm['local_results_directory'],
                     'original_workflow':prior})
    plan={'created_utc':datetime.now(timezone.utc).isoformat(),'study_id':original['study_id'],'arms':arms,
          'max_concurrent_training_nodes':1,'attempts_per_arm':1,'priority':'HIGH','pool':'groot-h100-01',
          'reason':'GPU quota enforcement canceled the three original attempts. Retry serially with the original resources and recipe; stop on another failure.',
          'initialization':'Untouched base; original attempts ended before the first scheduled checkpoint.',
          'unchanged_conditions':['dataset','sampled order','seed','model','context','batch','learning rates','48 updates','evaluation tasks']}
    write(PLAN,plan);return plan


def advance(plan,arms=None,allow_submit=True):
    """Submit at most one training job and back off between capacity retries."""
    for arm in plan['arms'] if arms is None else arms:
        build=Path(arm['build']);dest=ROOT/arm['local_results_directory']
        receipt=read(build/'submission.json')
        if not receipt:
            if not allow_submit:return {'status':'queued','id':arm['id'],'detail':'Waiting for a training node reservation.'}
            receipt=submit(build);ensure_monitor(build)
            state=command(['osmo','workflow','query',receipt['name'],'--format-type','json']);write(dest/'scheduler.json',state)
            return {'status':'active','workflow':receipt['name'],'id':arm['id']}
        state=command(['osmo','workflow','query',receipt['name'],'--format-type','json']);write(dest/'scheduler.json',state)
        complete=read(dest/'complete.json',{})
        if complete.get('success'):
            ensure_monitor(build)
            if state['status']=='COMPLETED' or state['status'].startswith('FAILED'):continue
        if read(dest/'train/complete.json',{}).get('success') and not complete:
            ensure_monitor(build)
            return {'status':'recovering_receipt','workflow':receipt['name'],'id':arm['id'],
                    'detail':'Training completed; recover its final export before considering any retry.'}
        if state['status'].startswith('FAILED'):
            retryable=(capacity_interruption(state) or gpu_interconnect_interruption(state,dest)
                       or initialization_interruption(state,dest))
            if plan.get('retry_policy',{}).get('enabled') and retryable:
                capture_control_failure(state,dest)
                # A failure receipt supplies the cooldown clock, so restarting
                # the local controller cannot reset it or produce duplicate jobs.
                ended=datetime.fromisoformat(state['end_time'].replace('Z','+00:00'))
                if ended.tzinfo is None:ended=ended.replace(tzinfo=timezone.utc)
                due=ended+timedelta(seconds=retry_delay(plan,arm))
                if datetime.now(timezone.utc)<due:
                    return {'status':'retry_wait','workflow':receipt['name'],'id':arm['id'],
                            'next_retry_utc':due.isoformat(),'priority':plan.get('priority','LOW'),
                            'detail':'Capacity or infrastructure interruption; waiting before the next single-node attempt.'}
                if not allow_submit:
                    return {'status':'queued','id':arm['id'],'workflow':receipt['name'],
                            'detail':'Retry is due; waiting for a training node reservation.'}
                next_attempt(plan,arm,receipt['name'])
                return advance(plan,arms,allow_submit)
            # In parallel mode, quarantine only this arm. Other independent
            # arms must keep their reservations and authorized retries.
            for waiting in plan['arms'] if arms is None else []:
                if (Path(waiting['build'])/'submission.json').exists():continue
                pending=ROOT/waiting['local_results_directory']
                reason='Recovery queue held after '+receipt['name']+' ended; resolve GPU quota or the reported failure before further submission.'
                write(pending/'scheduler.json',{'status':'HELD','reason':reason})
                write(pending/'progress.json',{'phase':'recovery_held','detail':reason})
            return {'status':'needs_review','workflow':receipt['name'],'id':arm['id'],
                    'reason':'Failure requires diagnosis: '+state['status']}
        if state['status']=='COMPLETED' and complete.get('success'):continue
        if complete and not complete.get('success'):
            return {'status':'needs_review','workflow':receipt['name'],'id':arm['id'],
                    'reason':complete.get('error','Training did not complete')}
        ensure_monitor(build)
        return {'status':'active','workflow':receipt['name'],'id':arm['id']}
    return {'status':'training_complete','detail':'Selected fixed final checkpoints completed training; downstream evaluation follows the active execution policy.'}


def advance_queue(plan):
    """Keep independent LR arms moving within the configured training-node cap."""
    limit=plan.get('max_concurrent_training_nodes',1)
    if limit==1:return advance(plan)
    if limit!=2:raise ValueError('Unvalidated training concurrency')
    workflows=command(['osmo','workflow','list','--name','leact','--status',
                       'RUNNING','PENDING','WAITING','--format-type','json'])['workflows']
    names={row['name'] for row in workflows}
    active={read(Path(arm['build'])/'submission.json',{}).get('name')
            for arm in plan['arms']} & names
    available=max(0,limit-len(active));states=[]
    # Reconcile existing reservations first; a completed/preempted job may free
    # a slot for another arm while its retry is in backoff.
    ordered=sorted(plan['arms'],key=lambda arm:read(Path(arm['build'])/'submission.json',{}).get('name') not in active)
    for arm in ordered:
        previous=read(Path(arm['build'])/'submission.json',{}).get('name')
        reserved=previous in active
        state=advance(plan,[arm],allow_submit=reserved or available>0)
        states.append({'id':arm['id'],**state})
        occupied=state['status'] in {'active','recovering_receipt'}
        if reserved and not occupied:available+=1
        elif not reserved and occupied:available=max(0,available-1)
    if all(s['status']=='training_complete' for s in states):
        return {'status':'training_complete','arms':states}
    main=next(s for status in ('active','recovering_receipt','retry_wait','queued','needs_review')
              for s in states if s['status']==status)
    return {**main,'arms':states,'max_concurrent_training_nodes':limit,
            'review_required':[s for s in states if s['status']=='needs_review']}


def watch():
    lock=LOCK.open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        try:
            plan=read(PLAN)
            if plan.get('execution_paused'):
                write(STATE,{'status':'paused_for_continuation','detail':plan.get('pause_reason'),
                    'updated_utc':datetime.now(timezone.utc).isoformat()});publish();return
            state=advance_queue(plan);write(STATE,{**state,'updated_utc':datetime.now(timezone.utc).isoformat()});publish()
            if state['status'] in {'needs_review','training_complete'}:return
        except Exception as exc:
            write(STATE,{'status':'waiting','error':str(exc)[-1200:],'updated_utc':datetime.now(timezone.utc).isoformat()})
        time.sleep(30)


def main():
    global PLAN,STATE,LOCK
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['prepare','watch','start','start-retrying'])
    parser.add_argument('--plan',type=Path);parser.add_argument('--state',type=Path)
    parser.add_argument('--training-nodes',type=int,choices=[1,2]);args=parser.parse_args()
    if bool(args.plan)!=bool(args.state):parser.error('--plan and --state must be supplied together')
    if args.plan:
        PLAN=args.plan.resolve();STATE=args.state.resolve();LOCK=STATE.with_suffix('.lock')
        if not PLAN.exists():parser.error('A custom recovery plan must already exist')
    if args.action=='watch':watch();return
    if args.action=='start-retrying':enable_retries(args.training_nodes)
    else:prepare()
    if args.action in {'start','start-retrying'}:
        # Check for an existing watcher; the child holds this lock for its lifetime.
        lock=LOCK.open('a')
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            print('Existing recovery controller will load the updated policy.');return
        fcntl.flock(lock,fcntl.LOCK_UN)
        with (STATE.with_suffix('.log') if args.plan else STATE.parent/'schedule-recovery.log').open('a') as log:
            command_args=[sys.executable,'-u','-m','reproductions.leact.cluster.recover_efe_schedule','watch']
            if args.plan:command_args+=['--plan',str(PLAN),'--state',str(STATE)]
            proc=subprocess.Popen(command_args,
                 cwd=BASE.parents[1],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(STATE.with_name(STATE.stem+'-launch.json') if args.plan else STATE.parent/'schedule-recovery-launch.json',
              {'pid':proc.pid,'updated_utc':datetime.now(timezone.utc).isoformat()})
    publish();print(PLAN)


if __name__=='__main__':main()
