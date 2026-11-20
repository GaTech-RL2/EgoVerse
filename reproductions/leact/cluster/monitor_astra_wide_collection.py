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
import collections,json,time
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
profile_root=Path(__PROFILE_ROOT__);profile_counts=collections.Counter();profile_suites={};workers={}
for p in profile_root.glob('episodes/*/infra-*/result.json'):
 r=load(p)
 if r is None:continue
 profile_counts[r['status']]+=1
 profile_suites.setdefault(r['case']['suite'],collections.Counter())[r['status']]+=1
for p in profile_root.glob('profile-workers/*.json'):
 value=load(p)
 if value is not None:workers[p.stem]=value
print('EFE_CAPTURE '+json.dumps({'observed_unix':time.time(),'nodes':nodes,'attempt_status':dict(counts),
 'suites':{k:dict(v)for k,v in suites.items()},'valid_actions':actions,'invalid_actions':invalid,'provider_output_tokens':tokens,'sft_admitted':admitted,
 'read_errors':errors,'astra62_profile':{'root':str(profile_root),'attempt_status':dict(profile_counts),'suites':{k:dict(v)for k,v in profile_suites.items()},'workers':workers}}),flush=True)
'''.replace('__ROOT__',repr(TEACHER_ROOT)).replace('__CONTROLS__',repr(controls)).replace('__PROFILE_ROOT__',repr(PROFILE_ROOT))
    return remote(host,script,task='native')


def recover_allocation(state, scheduler):
    """A scheduler-ended allocation has no live nodes to cancel or repurpose."""
    if scheduler['status']not in {'FAILED','FAILED_EVICTED','FAILED_PREEMPTED','FAILED_NODE_LOST'}:
        return None
    if scheduler.get('cancelled_by'):return None
    if not (infrastructure_failure(scheduler)or any(code in str(scheduler)for code in ['2001','2143'])):
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
    capture=None
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
        prefixes=set(read(ROOT/'lost-native-workers.json',{}).get('prefixes',[]))
        for row in states.values():prefixes.update(row.get('lost_native_prefixes',[]))
        if prefixes:
            try:status['lost_claim_reconciliation']=reconcile_remote(hosts[0],sorted(prefixes))['receipts']
            except Exception as exc:status['lost_claim_reconciliation_pending']=str(exc)[:600]
        for wf,node in capture['nodes'].items():
            if wf not in hosts:continue
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
