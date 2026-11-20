"""Observe and recover only the separately owned Astra teacher collectors."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

import yaml

from .scale_efe import BASE, command, read, write
from .launch_efe_pear_scaled import DEST
from .launch_efe_pear_primary import worker_build
from .recover_efe_pear_graphics import frozen_files
from .efe_pear_allocation_jobs import running_tasks, submit
from .efe_pear_submission import infrastructure_failure, submit_reconciled
from .monitor_efe_preflight import remote
from .efe_pear_file_rpc import atomic_bytes,atomic_json
from .launch_astra_wide_collection import (
    ROOT, OWNER, TEACHER_ROOT, NEW_FILES, CREDENTIAL, freeze_worker,
    idle_verification, record_launch,
)


MAX_ALLOCATION_RETRIES = 64


def completed_taskset(spec, completed_cases):
    """Suppress recovery only for the exact assigned, natively finalized cases."""
    from .switch_astra_wide_profile import PROFILE_ROOT
    unavailable = {'complete': False, 'reason': 'Unproven exact-profile taskset completion'}
    if (not isinstance(spec, dict) or not isinstance(completed_cases, dict)
            or spec.get('owner_model_workflow') != OWNER or spec.get('teacher_root') != PROFILE_ROOT
            or spec.get('teacher_profile_id') != 'astra62-xhigh-v1'
            or spec.get('max_calls', 96) != 96 or spec.get('wall_seconds', 1500) != 1500):
        return unavailable
    plan = spec.get('teacher_plan', {})
    if not isinstance(plan, dict) or not isinstance(plan.get('cases'), list):
        return unavailable
    names = set(spec.get('teacher_task_names', []))
    cases = [case for case in plan['cases'] if isinstance(case, dict) and case.get('task') in names]
    if (not names or {case.get('task') for case in cases} != names or not cases
            or not plan.get('curriculum_sha256')
            or len({case.get('case_id') for case in cases}) != len(cases)):
        return unavailable
    matched = []
    identity = ['case_id', 'task', 'config_id', 'scene_sha256', 'partition', 'admitted']
    for case in cases:
        if (case.get('partition') != 'train' or case.get('admitted') is not True
                or any(key not in case for key in identity)):
            return unavailable
        candidates = completed_cases.get(case['case_id'], [])
        if not isinstance(candidates, list):
            candidates = []
        for row in candidates:
            if not isinstance(row, dict) or not isinstance(row.get('case'), dict):
                continue
            if (row.get('requested_model') != 'openai/openai/gpt-6.2-astra'
                    or row.get('requested_effort') != 'xhigh'
                    or row.get('teacher_profile_id') != 'astra62-xhigh-v1'
                    or row.get('curriculum_sha256') != plan['curriculum_sha256']
                    or any(row.get('case', {}).get(key) != case[key] for key in identity)
                    or row.get('status') not in {'graded_success', 'graded_failure'}
                    or row.get('finalized_native_outcome') is not True
                    or row.get('native_success') is not (row['status'] == 'graded_success')
                    or row.get('native_verdict_success') is not row.get('native_success')
                    or not re.fullmatch('[0-9a-f]{64}', str(row.get('native_verdict_sha256', '')))
                    or not re.fullmatch('[0-9a-f]{64}', str(row.get('result_sha256', '')))):
                continue
            matched.append({'case_id': case['case_id'], 'status': row['status'],
                            'result_sha256': row['result_sha256'],
                            'native_verdict_sha256': row['native_verdict_sha256']})
            break
    return {'complete': len(matched) == len(cases), 'planned': len(cases),
            'matched_finalized': len(matched), 'case_receipts': matched,
            'reason': 'All assigned cases have exact-profile finalized native outcomes'
                      if len(matched) == len(cases) else 'Assigned cases remain unfinished or unproven'}


def reconcile_lost_claims(root,transport,prefixes,*,now=None):
    """Retire only v2 RPCs claimed by independently proven lost native jobs."""
    root,transport=Path(root),Path(transport);now=time.time()if now is None else now;receipts=[]
    if any(not p.startswith(('leact-rp-jhh-pear-worker-','rp-jhh-astra-wide-'))for p in prefixes):
        raise ValueError('Lost-claim reconciliation requires owned teacher native prefixes')
    for path in root.glob('episodes/*/infra-*/result.json'):
        try:record=json.loads(path.read_text())
        except (OSError,ValueError):continue
        if(record.get('status')not in {'opening','running'}or record.get('requested_model')!='openai/openai/gpt-6.2-astra'
            or record.get('requested_effort')!='xhigh'):continue
        uid=record.get('episode_id')
        if not isinstance(uid,str)or not re.fullmatch('[0-9a-f]{32}',uid):continue
        episode=transport/'episodes'/uid
        try:claim=json.loads((episode/'claimed.json').read_text())
        except (OSError,ValueError):continue
        worker=claim.get('worker_id','')
        if not re.fullmatch('[A-Za-z0-9_.-]+',worker):continue
        if not any(worker==prefix or worker.startswith(prefix+'-')for prefix in prefixes):continue
        try:heartbeat=json.loads((transport/'workers'/(worker+'.json')).read_text())
        except (OSError,ValueError):heartbeat={}
        if now-heartbeat.get('updated_unix',0)<=120:continue
        error={'error':'Owned native job was independently verified lost; no policy reward assigned.',
            'lost_native_worker':worker,'reconciled_unix':now,'infrastructure_failure':True}
        for name in ['infrastructure-error.json','cancel.json']:
            if not(episode/name).exists():atomic_json(episode/name,error)
        orphan=False
        with(path.parent.parent/'writer.lock').open('a')as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:pass
            else:
                current=json.loads(path.read_text())
                if current.get('status')in {'opening','running'}and current.get('episode_id')==uid:
                    current.update(status='infrastructure_failure',error_type='LostOwnedNativeClaim',
                        error=error['error'],lost_native_worker=worker,updated_unix=now,
                        completed_utc=datetime.fromtimestamp(now,timezone.utc).isoformat())
                    atomic_json(path,current);orphan=True
        receipts.append({'episode_id':uid,'case_id':record['case']['case_id'],
            'lost_native_worker':worker,'orphan_record_finalized':orphan,'native_workers_canceled':False})
    return receipts


def reconcile_remote(host,prefixes):
    import inspect
    from .switch_astra_wide_profile import PROFILE_ROOT
    script=('from pathlib import Path\nfrom uuid import uuid4\nfrom datetime import datetime,timezone\n'
        'import os,json,re,fcntl,time\n'+inspect.getsource(atomic_bytes)+'\n'+inspect.getsource(atomic_json)+'\n'+
        inspect.getsource(reconcile_lost_claims)+'\n'+
        "print('EFE_CAPTURE '+json.dumps({'receipts':reconcile_lost_claims("+repr(PROFILE_ROOT)+','+
        repr(TEACHER_ROOT+'/transport')+','+repr(prefixes)+")}),flush=True)\n")
    return remote(host,script,task='native')


def refresh_build(state):
    """Freeze the same admitted task recipe with the latest collection recovery."""
    old=Path(state['build']);spec=read(old/'spec.json')
    from .switch_astra_wide_profile import PROFILE_ROOT,profile_spec
    if spec.get('owner_model_workflow')!=OWNER or spec.get('teacher_root')not in {TEACHER_ROOT,PROFILE_ROOT}:
        raise ValueError('Not an independently owned teacher collection')
    if (state.get('teacher_profile_id')=='astra62-xhigh-v1'or
        read(ROOT/'active-profile.json',{}).get('teacher_profile_id')=='astra62-xhigh-v1'):
        spec={**profile_spec(spec),'teacher_native_only':True}
    files=frozen_files(old)
    for name in NEW_FILES:files[name]=BASE/'cluster'/name
    spec['teacher_recovery_of']=state['workflow']+':'+str(state.get('job_id','initial'))
    workflow=yaml.safe_load((old/'workflow.yaml').read_text())['workflow']
    task=workflow['tasks'][0];resources=dict(workflow['resources']['default'])
    if task.get('privileged'):resources['privileged']=True
    entry='efe_astra_wide_reasoning_worker.py'if state['label']=='reasoning'else'efe_astra_wide_worker.py'
    build=worker_build(spec,files,entry,task['image'],task['command'][0],**resources)
    if (build/'submission.json').exists()or(build/'submission-intent.json').exists():return build
    value=yaml.safe_load((build/'workflow.yaml').read_text());wf=value['workflow'];native=wf['tasks'][0]
    wf['name']='rp-jhh-astra-wide-'+state['label']+'-'+build.name.rsplit('-',1)[-1]
    wf['labels']={'experiment':OWNER,'phase':'teacher_collection',
                  'teacher_model':'gpt-6.2-astra'if spec.get('teacher_profile_id')=='astra62-xhigh-v1'else'gpt-6-astra'}
    native['credentials']={CREDENTIAL:{'NVIDIA_INFERENCE_API_KEY':'api_key'}}
    if state['label']in {'efe','robodojo','robolab','yamlab'}:
        native['credentials']['rp-jhh-pear-swift-archive-20261118']={'RP_JHH_PRIVATE_ARCHIVE_CONFIGURATION':'configuration'}
    (build/'workflow.yaml').write_text(yaml.safe_dump(value,sort_keys=False))
    freeze_worker(build,[*files,'spec.json','bootstrap.py','workflow.yaml'])
    return build


def snapshot(host, states):
    from .switch_astra_wide_profile import PROFILE_ROOT
    controls={}
    for wf,row in states.items():
        if row.get('superseded_by'):continue
        spec=read(Path(row.get('allocation_build',row['build']))/'spec.json')
        controls[wf]=spec['allocation_control_base']+'/'+wf
    script='''from pathlib import Path
import collections,hashlib,json,time
root=Path(__ROOT__);controls=__CONTROLS__;nodes={};counts=collections.Counter();suites={};actions=invalid=tokens=admitted=0;errors=[]
def load(path):
 try:return json.loads(path.read_text())
 except (OSError,ValueError)as exc:
  errors.append({'path':str(path),'error_type':type(exc).__name__,'error':str(exc)[:180]});return None
for wf,control in controls.items():
 p=Path(control)
 node=load(p/'nodes/native.json');desired=load(p/'desired.json')
 if node and desired:nodes[wf]={'node':node,'desired_id':desired['id']}
for p in root.glob('episodes/*/infra-*/result.json'):
 r=load(p)
 if r is None:continue
 status=r['status'];suite=r['case']['suite'];counts[status]+=1
 suites.setdefault(suite,collections.Counter())[status]+=1
 actions+=r.get('valid_actions',0);invalid+=r.get('invalid_actions',0);tokens+=r.get('generated_tokens',0);admitted+=bool(r.get('sft_admitted'))
profile_root=Path(__PROFILE_ROOT__);profile_counts=collections.Counter();profile_suites={};workers={};completed_cases={}
for p in profile_root.glob('episodes/*/infra-*/result.json'):
 r=load(p)
 if r is None:continue
 if not isinstance(r,dict)or not isinstance(r.get('case'),dict)or not r.get('status')or not r['case'].get('suite'):
  errors.append({'path':str(p),'error_type':'MalformedEpisodeRecord','error':'Missing episode status or case identity'});continue
 profile_counts[r['status']]+=1
 profile_suites.setdefault(r['case']['suite'],collections.Counter())[r['status']]+=1
 if r.get('status')not in {'graded_success','graded_failure'}or r.get('finalized_native_outcome')is not True:continue
 native_path=p.with_name('native-verdict.json');verdict=load(native_path)
 if not isinstance(verdict,dict):continue
 verifier=verdict.get('verifier_result');rewards=verifier.get('rewards')if isinstance(verifier,dict)else None
 success=verdict.get('success',rewards.get('success')if isinstance(rewards,dict)else None)
 if type(success)not in(bool,int,float)or success not in(0,1):continue
 try:result_sha=hashlib.sha256(p.read_bytes()).hexdigest();native_sha=hashlib.sha256(native_path.read_bytes()).hexdigest()
 except OSError as exc:
  errors.append({'path':str(p),'error_type':type(exc).__name__,'error':str(exc)[:180]});continue
 completed_cases.setdefault(r['case']['case_id'],[]).append({
  'case':{k:r['case'].get(k)for k in ['case_id','task','config_id','scene_sha256','partition','admitted']},
  'requested_model':r.get('requested_model'),'requested_effort':r.get('requested_effort'),
  'teacher_profile_id':r.get('teacher_profile',{}).get('id'),'curriculum_sha256':r.get('curriculum_sha256'),
  'status':r['status'],'native_success':r.get('native_success'),'finalized_native_outcome':r.get('finalized_native_outcome'),
  'native_verdict_success':bool(success),'result_sha256':result_sha,'native_verdict_sha256':native_sha})
for p in profile_root.glob('profile-workers/*.json'):
 value=load(p)
 if value is not None:workers[p.stem]=value
print('EFE_CAPTURE '+json.dumps({'observed_unix':time.time(),'nodes':nodes,'attempt_status':dict(counts),
 'suites':{k:dict(v)for k,v in suites.items()},'valid_actions':actions,'invalid_actions':invalid,'provider_output_tokens':tokens,'sft_admitted':admitted,
 'read_errors':errors,'astra62_profile':{'root':str(profile_root),'attempt_status':dict(profile_counts),'suites':{k:dict(v)for k,v in profile_suites.items()},'workers':workers,'completed_cases':completed_cases}}),flush=True)
'''.replace('__ROOT__',repr(TEACHER_ROOT)).replace('__CONTROLS__',repr(controls)).replace('__PROFILE_ROOT__',repr(PROFILE_ROOT))
    return remote(host,script,task='native')


def never_started_queue_timeout(scheduler):
    """A terminal queue expiry can retry only with explicit no-allocation proof."""
    if (scheduler.get('status') != 'FAILED_QUEUE_TIMEOUT' or scheduler.get('cancelled_by')
            or 'start_time' not in scheduler or scheduler['start_time'] is not None
            or scheduler.get('duration') not in (None, 0)):
        return False
    groups = scheduler.get('groups', [])
    tasks = [task for group in groups for task in group.get('tasks', [])]
    tasks += scheduler.get('tasks', [])
    if not tasks or any('start_time' not in group or group['start_time'] is not None for group in groups):
        return False
    # OSMO can create a pod name while still queued. Node assignment, pod IP,
    # or any task start proves this was allocated and is outside this retry.
    return all(all(key in task and task[key] is None for key in
                   ['start_time', 'node_name', 'pod_ip']) for task in tasks)


def recover_allocation(state, scheduler):
    """A scheduler-ended allocation has no live nodes to cancel or repurpose."""
    queue_timeout = never_started_queue_timeout(scheduler)
    if scheduler['status']not in {'FAILED','FAILED_EVICTED','FAILED_PREEMPTED','FAILED_NODE_LOST'}and not queue_timeout:
        return None
    if scheduler.get('cancelled_by'):return None
    if not (queue_timeout or infrastructure_failure(scheduler)or any(code in str(scheduler)for code in ['2001','2143'])):
        return None
    if state.get('allocation_retries',0)>=MAX_ALLOCATION_RETRIES:return None
    build=refresh_build(state)
    pool=state.get('pool')or scheduler['pool']
    receipt=submit_reconciled(build,pool=pool,priority='LOW',nodes=1)
    replacement={**state,**receipt,'workflow':receipt['name'],'job_id':None,'build':str(build),
        'allocation_build':str(build),'pool':pool,'priority':'LOW','phase':'scheduler_recovery_submitted',
        'allocation_retries':state.get('allocation_retries',0)+1,'recovery_of':state['workflow']}
    replacement['lost_native_prefixes']=[*state.get('lost_native_prefixes',[]),state['workflow']]
    replacement.pop('allocation_control',None);replacement.pop('superseded_by',None)
    for key in ['profile_sidecar','profile_job_id','profile_build','profile_phase']:replacement.pop(key,None)
    record_launch(replacement)
    record_launch({**state,'superseded_by':replacement['workflow'],'phase':'scheduler_ended_recovered'})
    return replacement


def recover_child(state, node):
    if node['node']['phase']not in {'idle_failed','idle_peer_failed'}:return None
    if node['desired_id']!=state.get('job_id')or state.get('child_retries',0)>=3:return None
    # Both the durable desired job and native/GPU process proof must show idle.
    proof,control=idle_verification(state)
    build=refresh_build(state);number=state.get('child_retries',0)+1
    job='rp-jhh-astra-recovery-'+hashlib.sha256((state['workflow']+str(number)).encode()).hexdigest()[:20]
    receipt=submit(state,build,replace_running=False,job_id=job)
    replacement={**state,'build':str(build),'job_id':receipt['job_id'],'child_retries':number,
        'allocation_control':control,'phase':'child_recovery_staged','idle_proof':proof,
        'lost_native_prefixes':[*state.get('lost_native_prefixes',[]),state['workflow']+'-'+state['job_id']]}
    record_launch(replacement);return replacement


def tick():
    write(ROOT/'watch-loop-heartbeat.json',{'updated_unix':time.time(),'pid':os.getpid(),'phase':'querying_schedulers'})
    states=read(ROOT/'launches.json',{})
    active={k:v for k,v in states.items()if not v.get('superseded_by')}
    def query(wf):
        try:
            result=subprocess.run(['osmo','workflow','query',wf,'--format-type','json'],
                                  capture_output=True,text=True,timeout=20)
            if result.returncode:raise RuntimeError((result.stderr+result.stdout)[-600:])
            return wf,json.loads(result.stdout[result.stdout.index('{'):])
        except Exception as exc:return wf,{'status':'UNKNOWN','error':str(exc)[:300]}
    with ThreadPoolExecutor(4)as pool:schedulers=dict(pool.map(query,active))
    write(ROOT/'watch-schedulers.json',schedulers)
    status={'updated_utc':datetime.now(timezone.utc).isoformat(),'pid':os.getpid(),
        'scheduler_counts':dict(__import__('collections').Counter(q['status']for q in schedulers.values())),
        'hero_untouched':True,'recovery_actions':[],'phase':'capturing_teacher_state'}
    write(ROOT/'watch-status.json',status)
    try:primary=read(DEST/'primary-launch.json',{})
    except (OSError,ValueError)as exc:
        primary={};status['hero_read_error']=type(exc).__name__+':'+str(exc)[:240]
    if primary.get('workflow'):status['hero_scheduler']=query(primary['workflow'])[1].get('status')
    hosts=[]
    for wf,q in schedulers.items():
        try:running_tasks(q);hosts.append(wf)
        except (RuntimeError,KeyError):pass
    capture=None;complete={}
    if hosts:
        # Reading all shared status through one healthy owned worker avoids
        # exec during any other worker's scheduler startup barrier.
        for host in hosts[:3]:
            try:capture=snapshot(host,active);break
            except Exception as exc:status.setdefault('snapshot_errors',{})[host]=type(exc).__name__+':'+str(exc)[:240]
    if capture is not None:
        write(ROOT/'watch-census.json',capture)
        status['attempt_status']=capture['attempt_status'];status['sft_admitted']=capture['sft_admitted']
        status['snapshot_read_errors']=capture.get('read_errors',[])
        profile_enabled=read(ROOT/'active-profile.json',{}).get('teacher_profile_id')=='astra62-xhigh-v1'
        if profile_enabled:status['astra62_attempt_status']=capture['astra62_profile']['attempt_status']
        for wf,row in active.items():
            try:
                spec=read(Path(row.get('profile_build')or row['build'])/'spec.json')
                proof=completed_taskset(spec,capture.get('astra62_profile',{}).get('completed_cases',{}))
                if proof['complete']:complete[wf]=proof
            except (OSError,ValueError,KeyError,TypeError)as exc:
                status.setdefault('completion_read_errors',{})[wf]=type(exc).__name__+':'+str(exc)[:180]
        if complete:status['completed_tasksets_preserved']=complete
        prefixes=set(read(ROOT/'lost-native-workers.json',{}).get('prefixes',[]))
        for row in states.values():prefixes.update(row.get('lost_native_prefixes',[]))
        if prefixes:
            try:status['lost_claim_reconciliation']=reconcile_remote(hosts[0],sorted(prefixes))['receipts']
            except Exception as exc:status['lost_claim_reconciliation_pending']=str(exc)[:600]
        for wf,node in capture['nodes'].items():
            if wf not in hosts:continue
            if wf in complete:continue  # Preserve already accepted/live allocations; no new child/profile work.
            row=active[wf]
            if not row.get('job_id'):
                row={**row,'job_id':node['desired_id']};record_launch(row);active[wf]=row
            try:recovered=recover_child(row,node)
            except Exception as exc:
                status.setdefault('child_recovery_pending',{})[wf]=str(exc)[:600];continue
            if recovered:status['recovery_actions'].append({'workflow':wf,'child':recovered['job_id']})
            if profile_enabled and not recovered and node['node']['phase']=='running':
                from .switch_astra_wide_profile import deploy
                key=wf+'-'+str(row.get('profile_job_id',''))
                heartbeat=capture['astra62_profile']['workers'].get(key,{})
                if time.time()-heartbeat.get('updated_unix',0)<120:
                    status.setdefault('profile_collectors',{})[wf]=heartbeat['pid'];continue
                try:
                    result=deploy(row)
                    status.setdefault('profile_collectors',{})[wf]=result['profile_sidecar']['pid']
                except Exception as exc:
                    status.setdefault('profile_deployment_pending',{})[wf]=str(exc)[:600]
    for wf,q in schedulers.items():
        if wf in complete:continue  # A valid graded failure completes its planned collection attempt too.
        try:replacement=recover_allocation(active[wf],q)
        except Exception as exc:
            status.setdefault('allocation_recovery_pending',{})[wf]=str(exc)[:600];continue
        if replacement:status['recovery_actions'].append({'workflow':replacement['workflow'],'recovery_of':wf})
    status.update(updated_utc=datetime.now(timezone.utc).isoformat(),phase='observed')
    write(ROOT/'watch-status.json',status)
    write(ROOT/'watch-loop-heartbeat.json',{'updated_unix':time.time(),'pid':os.getpid(),'phase':'sleeping'})
    return status


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    with(ROOT/'watch.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write(ROOT/'watch-launch.json',{'pid':os.getpid(),'started_utc':datetime.now(timezone.utc).isoformat(),
            'module':__name__,'interval_seconds':120,'scope':'Independent Astra collectors only; preserve hero and all live allocations.'})
        while not read(ROOT/'control.json',{}).get('paused'):
            try:print(json.dumps(tick()),flush=True)
            except Exception as exc:
                write(ROOT/'watch-error.json',{'updated_unix':time.time(),'error':str(exc)[:1200]})
                write(ROOT/'watch-status.json',{'updated_utc':datetime.now(timezone.utc).isoformat(),
                    'pid':os.getpid(),'phase':'observer_error','hero_untouched':True,'error':str(exc)[:600]})
            time.sleep(120)


if __name__=='__main__':main()
