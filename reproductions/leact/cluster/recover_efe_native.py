"""Keep one native robot evaluation moving across verified OSMO interruptions."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone,timedelta
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import yaml

from .scale_efe import BASE,ROOT,BUILD_ROOT,read,write,submit,monitor,command
from .launch_efe_training import freeze_sources
from .recover_efe_schedule import capacity_interruption,capture_control_failure,initialization_interruption
from ..embodied_first_exam.report import publish

PLAN=ROOT/'training-pilot/native-step32-recovery.json'
STATE=PLAN.with_name('native-step32-recovery-status.json')


def retry_spec(previous,parent,build,serving_transport=None,local_results_directory=None):
    old_resume=previous.get('resume_completed_trials',{})
    fallback=[{k:v for k,v in old_resume.items() if k!='fallback_sources'},*old_resume.get('fallback_sources',[])] if old_resume else []
    spec={**previous,'archive_root':'/mnt/amlfs-03/home/rpunamiya/leact/studies/'+build.name,
          'local_results_directory':'training-pilot/'+build.name.replace('efe-warm-bb-1e4-p3-step32-dev-20261028-', 'warm-bb-1e4-p3-step32-evaluation-'),
          'recovery_of':parent,'recovery_reason':'Verified OSMO capacity/infrastructure interruption; retain every graded outcome and rerun only unfinished tasks.',
          'serving_diagnostics':'Capture server failure receipts and NCCL INFO logs; serving arguments, sampling, task inputs and grading unchanged.',
          'retry_of':{row['task']:parent for row in previous['tasks']},
          'resume_completed_trials':{'archive':previous['archive_root']+'/'+parent,'workflow':parent,
              'source_spec_sha256':hashlib.sha256(json.dumps(previous,sort_keys=True).encode()).hexdigest(),
              'fallback_sources':fallback}}
    if serving_transport:
        spec['serving_transport']=serving_transport
        spec['serving_diagnostics']='Reviewed NCCL communication workaround; checkpoint, TP=4, context, sampling, task inputs and grading retained. Transport and timing caveat recorded separately.'
    if local_results_directory:spec['local_results_directory']=local_results_directory
    return spec


def next_build(plan,parent):
    source=Path(plan['build']);attempt=plan['attempt']+1
    build=BUILD_ROOT/(plan.get('build_prefix','efe-warm-bb-1e4-p3-step32-dev-20261028-v2-retry')+str(attempt))
    local_results=(plan['local_results_prefix']+str(attempt)) if plan.get('local_results_prefix') else None
    build.mkdir(exist_ok=True);spec=retry_spec(read(source/'spec.json'),parent,build,plan.get('serving_transport'),local_results)
    if (build/'source-manifest.json').exists():
        if read(build/'spec.json')!=spec:raise ValueError('Frozen native retry changed')
    else:
        for name,digest in read(source/'source-manifest.json').items():
            if hashlib.sha256((source/name).read_bytes()).hexdigest()!=digest:
                raise ValueError('Source native launch checksum changed: '+name)
            if name not in ('spec.json','workflow.yaml','pilot.py','student.py'):shutil.copyfile(source/name,build/name)
        shutil.copyfile(BASE/'cluster/efe_astra_pilot.py',build/'pilot.py')
        shutil.copyfile(BASE/'cluster/efe_qwen_baseline.py',build/'student.py')
        write(build/'spec.json',spec)
        workflow=yaml.safe_load((source/'workflow.yaml').read_text())
        workflow['workflow']['name']=plan.get('workflow_prefix','leact-efe-warm-1e4-step32-dev-v2-r')+str(attempt)
        (build/'workflow.yaml').write_text(yaml.safe_dump(workflow,sort_keys=False));freeze_sources(build)
    plan.setdefault('previous_attempts',[]).append({'build':str(source),'workflow':parent,'attempt':plan['attempt']})
    plan.update(build=str(build),attempt=attempt);write(PLAN,plan)
    registry=read(ROOT/'training-pilot/intermediate-evaluations.json',{'runs':[]})
    row=next((r for r in registry['runs'] if r['build']==str(source)),None)
    new=None
    if row:
        new={**row,'id':f'qwen-warm-bb-1e4-p3-step32-native-v2-retry{attempt}',
             'name':f'Qwen LR 1e-4 · step 32 · resumed native evaluation · attempt {attempt}',
             'build':str(build),'evaluation_directory':spec['local_results_directory'],
             'serving_transport':spec.get('serving_transport',{'mode':'default'}),
             'recovery_of':parent,'resume_completed_trials':spec['resume_completed_trials']}
        if not any(r['id']==new['id'] for r in registry['runs']):registry['runs'].append(new)
        write(ROOT/'training-pilot/intermediate-evaluations.json',registry)
    if plan.get('evaluation_stage'):
        write(ROOT/('training-pilot/'+plan['evaluation_stage']+'-evaluation-active.json'),
              {'directory':spec['local_results_directory'],'build':str(build),'attempt':attempt})
    continuation=read(ROOT/'training-pilot/continuation-recovery.json',{})
    if continuation:
        for arm in continuation['arms']:
            if new:arm['baseline']=new['id']
            if arm['id']==plan.get('continuation_arm_id'):arm['evaluation_directory']=spec['local_results_directory']
        write(ROOT/'training-pilot/continuation-recovery.json',continuation)
    return build


def nccl_startup_interruption(state,dest):
    """A bounded retry for communicator initialization, before model serving."""
    if state['status']!='FAILED':return False
    failure=read(dest/'student-serving-failure.json',{})
    text=failure.get('log_tail','')
    required=('ncclCommInitRank','init_worker_distributed_environment',
              'RuntimeError: NCCL error: unhandled cuda error','WorkerProc failed to start')
    if not all(s in text for s in required):return False
    if any(s in text for s in ('CUDA out of memory','torch.OutOfMemoryError','Model loading took','Application startup complete')):
        return False
    transport=read(dest/'transport.json',{})
    if transport.get('completed',0):return False
    write(dest/'interruption.json',{'workflow':state['name'],
        'kind':'gpu_interconnect_failure' if 'P2P is disabled between NVLINK connected GPUs' in text else 'nccl_initialization_failure',
        'source':'student-serving-failure.json','source_log_sha256':failure['log_sha256'],
        'before_model_serving':True,'evidence':list(required),'updated_utc':datetime.now(timezone.utc).isoformat()})
    return True


def advance(plan):
    build=Path(plan['build']);receipt=read(build/'submission.json',{})
    if not receipt:
        receipt=submit(build,priority=plan.get('priority','LOW'));monitor(build)
        return {'status':'active','workflow':receipt['name'],'attempt':plan['attempt']}
    state=command(['osmo','workflow','query',receipt['name'],'--format-type','json'])
    dest=ROOT/read(build/'spec.json')['local_results_directory']
    dest.mkdir(parents=True,exist_ok=True);write(dest/'scheduler.json',state)
    if state['status'] in {'RUNNING','PENDING','WAITING'}:
        return {'status':'active','workflow':receipt['name'],'attempt':plan['attempt']}
    complete=read(dest/'complete.json',{});rows=read(dest/'results.json',{}).get('trials',[])
    if state['status']=='COMPLETED' and not complete:
        collected=read(build/'monitor-status.json',{})
        if collected.get('collection_needs_review'):
            return {'status':'needs_review','workflow':receipt['name'],
                    'detail':'Completed worker has no completion receipt after three shared-archive reads.'}
        pid=read(build/'monitor-launch.json',{}).get('pid')
        running=subprocess.run(['ps','-p',str(pid),'-o','command='],capture_output=True,text=True,timeout=5).stdout if pid else ''
        if 'reproductions.leact.cluster.monitor_efe_astra' not in running or str(build) not in running:monitor(build)
        return {'status':'collecting','workflow':receipt['name'],
                'detail':'Worker completed; recovering checksummed final receipts from shared storage before deciding on any retry.'}
    if len([r for r in rows if r['status']=='graded'])==7 and not complete.get('error'):
        return {'status':'complete','workflow':receipt['name'],'graded':7}
    startup_fault=nccl_startup_interruption(state,dest)
    known_interconnect=startup_fault and read(dest/'interruption.json',{}).get('kind')=='gpu_interconnect_failure'
    startup_retries=plan.setdefault('nccl_startup_retried_workflows',[])
    if startup_fault and not known_interconnect and receipt['name'] not in startup_retries and len(startup_retries)>=2:
        return {'status':'needs_review','workflow':receipt['name'],
                'detail':'Two communicator-startup retries already used; inspect the new NCCL diagnostics before further retries.'}
    retryable=capacity_interruption(state) or initialization_interruption(state,dest) or startup_fault
    if not retryable:
        return {'status':'needs_review','workflow':receipt['name'],'scheduler_status':state['status'],
                'detail':'No automatic policy retries; inspect incomplete grading or non-capacity failure.'}
    capture_control_failure(state,dest)
    if capacity_interruption(state) and plan.get('priority')=='HIGH':
        plan['priority']='LOW';write(PLAN,plan)
    ended=datetime.fromisoformat(state['end_time'].replace('Z','+00:00'))
    if ended.tzinfo is None:ended=ended.replace(tzinfo=timezone.utc)
    due=ended+timedelta(seconds=min(300*2**min(max(0,plan['attempt']-2),3),900))
    if datetime.now(timezone.utc)<due:
        return {'status':'retry_wait','workflow':receipt['name'],'next_retry_utc':due.isoformat()}
    if startup_fault and not known_interconnect and receipt['name'] not in startup_retries:
        startup_retries.append(receipt['name']);write(PLAN,plan)
    next_build(plan,receipt['name'])
    return advance(plan)


def watch():
    with STATE.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            try:
                plan=read(PLAN)
                if not plan.get('enabled',True):return
                result=advance(plan);write(STATE,{**result,'updated_utc':datetime.now(timezone.utc).isoformat()});publish()
                if result['status'] in {'complete','needs_review'}:return
            except Exception as exc:
                write(STATE,{'status':'waiting','error':str(exc)[-1600:],'updated_utc':datetime.now(timezone.utc).isoformat()})
            time.sleep(30)


def main():
    global PLAN,STATE
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['start','watch'])
    parser.add_argument('--plan',type=Path);parser.add_argument('--state',type=Path)
    args=parser.parse_args()
    if args.plan:
        PLAN=args.plan.resolve();STATE=args.state.resolve() if args.state else PLAN.with_name(PLAN.stem+'-status.json')
        if not PLAN.exists():raise FileNotFoundError('Explicit native recovery plan does not exist: '+str(PLAN))
    if args.action=='watch':watch();return
    if not PLAN.exists():write(PLAN,{'enabled':True,'attempt':2,
        'build':str(BUILD_ROOT/'efe-warm-bb-1e4-p3-step32-dev-20261028-v2-retry2'),
        'policy':'One native evaluation node, LOW priority, 5–15 minute backoff; preserve all graded outcomes. Only verified infrastructure interruptions retry.'})
    with STATE.with_suffix('.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:print('Native recovery already running');return
    with STATE.with_suffix('.log').open('a') as log:
        command_line=[sys.executable,'-u','-m','reproductions.leact.cluster.recover_efe_native','watch']
        if args.plan:command_line+=['--plan',str(PLAN),'--state',str(STATE)]
        child=subprocess.Popen(command_line,
            cwd=BASE.parents[1],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    launch=STATE.with_name(STATE.stem+'-launch.json') if args.plan else STATE.with_name('native-step32-recovery-launch.json')
    write(launch,{'pid':child.pid,'started_utc':datetime.now(timezone.utc).isoformat()})
    print(json.dumps({'pid':child.pid,'plan':str(PLAN),'capability_screen_enabled':False}))


if __name__=='__main__':main()
