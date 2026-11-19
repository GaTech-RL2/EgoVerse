"""Launch separate Astra-native collection without touching the active learner."""
import argparse
from collections import defaultdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

import yaml

from .scale_efe import BASE,read,write,command
from .launch_efe_pear_scaled import DEST,REMOTE
from .launch_efe_pear_primary import worker_build,bootstrap
from .recover_efe_pear_graphics import frozen_files
from .efe_pear_allocation_jobs import running_tasks,submit
from .efe_pear_submission import submit_reconciled
from .monitor_efe_preflight import remote
from .efe_astra_wide_collect import MODEL,collection_plan

ROOT=DEST/'astra-wide-warmstart-v1'
TEACHER_ROOT=REMOTE+'/teacher-collections/rp_jhh_astra_wide_warmstart_20261119_v1'
OWNER='rp_jhh_astra_wide_warmstart_20261119_v1'
CREDENTIAL='rp-jhh-astra-wide-inference-20261119'
NEW_FILES=['efe_astra_wide_collect.py','efe_astra_wide_worker.py','efe_astra_wide_reasoning_worker.py',
           'efe_astra_wide_profile_worker.py','efe_pear_file_rpc.py','efe_pear_remote_worker.py']


def freeze_worker(build,names):
    """Hash explicit source inputs, excluding manifests and submission receipts."""
    write(build/'source-manifest.json',{n:hashlib.sha256((build/n).read_bytes()).hexdigest()
                                      for n in sorted(set(names))})


def record_launch(state):
    ROOT.mkdir(parents=True,exist_ok=True)
    with(ROOT/'launches.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        ledger=read(ROOT/'launches.json',{});ledger[state['workflow']]=state
        write(ROOT/'launches.json',ledger)


def task_order(cases,suites):
    buckets={s:sorted({c['task']for c in cases if c['suite']==s})for s in sorted(suites)}
    return [buckets[s][i]for i in range(max(map(len,buckets.values()),default=0))
            for s in buckets if i<len(buckets[s])]


def prepared(item,*,reasoning=False):
    old=Path(item['build']);spec=read(old/'spec.json');files=frozen_files(old)
    plan=collection_plan(read(ROOT/'live-curriculum.json'));write(ROOT/'collection-plan.json',plan)
    for name in NEW_FILES:files[name]=BASE/'cluster'/name
    original_entry=spec['allocation_job_entry'];label='native-yam'
    if reasoning:suites=['procedural_reasoning'];names=task_order(plan['cases'],suites);label='reasoning'
    elif original_entry=='efe_pear_yam_server.py':names=[r['env_name']for r in spec['tasks']]
    elif original_entry=='efe_pear_primary_native.py':names=task_order(plan['cases'],spec['suites']);label='efe'
    elif original_entry=='efe_pear_primary_rtx.py':names=task_order(plan['cases'],[spec['suite']]);label=spec['suite']
    else:raise ValueError('Unsupported admitted simulator bootstrap')
    allowed={c['task']for c in plan['cases']}
    if not names or not set(names)<=allowed:raise ValueError('Teacher task is not in the admitted training mixture')
    spec.update(owner_model_workflow=OWNER,teacher_root=TEACHER_ROOT,
        output=TEACHER_ROOT+('/native-'+label),transport_root=TEACHER_ROOT+'/transport',
        finished=TEACHER_ROOT+'/finished.json',head_heartbeat=TEACHER_ROOT+'/head-heartbeat.json',
        teacher_native_entry='reasoning'if reasoning else original_entry,teacher_plan=plan,teacher_task_names=names,
        teacher_credential_file='/dev/shm/rp_jhh_astra_api_key',teacher_endpoint='https://inference-api.nvidia.com/v1',
        teacher_concurrency=2 if label=='native-yam'else 8 if label=='efe'else 1,
        source_model_job=read(DEST/'primary-launch.json')['job_id'])
    if 'episode_archive'in spec:spec['episode_archive']='s3://rp-jhh-pear-checkpoints/rpunamiya/rp_jhh_astra_wide_warmstart_20261119_v1/native-episodes'
    workflow=yaml.safe_load((old/'workflow.yaml').read_text())['workflow'];task=workflow['tasks'][0]
    resources=dict(workflow['resources']['default'])
    if label=='efe':resources['privileged']=True
    if reasoning:resources={'gpu':0,'cpu':8,'memory':'32Gi','storage':'20Gi','platform':'dgx-h100'}
    entry='efe_astra_wide_reasoning_worker.py'if reasoning else'efe_astra_wide_worker.py'
    build=worker_build(spec,files,entry,task['image'],task['command'][0],**resources)
    if (build/'submission.json').exists()or(build/'submission-intent.json').exists():
        return build,label
    value=yaml.safe_load((build/'workflow.yaml').read_text());wf=value['workflow'];t=wf['tasks'][0]
    wf['name']='rp-jhh-astra-wide-'+label+'-'+build.name.rsplit('-',1)[-1]
    wf['labels']={'experiment':OWNER,'phase':'teacher_collection','teacher_model':'gpt-6-astra'}
    t['credentials']={CREDENTIAL:{'NVIDIA_INFERENCE_API_KEY':'api_key'}}
    if label in {'efe','robodojo','robolab','yamlab'}:
        t['credentials']['rp-jhh-pear-swift-archive-20261118']={'RP_JHH_PRIVATE_ARCHIVE_CONFIGURATION':'configuration'}
    if reasoning:
        # CPU allocations use the same retained-child lifecycle as GPU workers.
        pinned=read(build/'spec.json');pinned.update(allocation_lifecycle='persistent_cluster',
            allocation_job_entry=entry,allocation_control_base=TEACHER_ROOT+'/allocations')
        write(build/'spec.json',pinned)
        (build/'bootstrap.py').write_text(bootstrap(build,[*files,'spec.json'],'efe_pear_allocation_keeper.py'))
        t['environment'].update(PEAR_ALLOCATION_BASE=pinned['allocation_control_base'],
            PEAR_KEEPER_ENTRY=entry,PEAR_ALLOCATION_ROLES='["native"]')
    (build/'workflow.yaml').write_text(yaml.safe_dump(value,sort_keys=False))
    freeze_worker(build,[*files,'spec.json','bootstrap.py','workflow.yaml'])
    return build,label


def register_credential():
    if read(ROOT/'credential-registration.json',{}).get('registered'):
        return
    from ..astra_expert.client import credential
    with tempfile.TemporaryDirectory(prefix='rp_jhh_astra_credential_')as temp:
        path=Path(temp)/'api_key';fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'w')as stream:stream.write(credential())
        result=subprocess.run(['osmo','credential','set',CREDENTIAL,'--type','GENERIC',
            '--payload-file','api_key='+str(path)],capture_output=True,text=True,timeout=60)
        if result.returncode:raise RuntimeError('Astra collection credential registration failed')
    write(ROOT/'credential-registration.json',{'name':CREDENTIAL,'registered':True,
        'source':'Existing authorized experiment Keychain item; no key persisted in artifacts.'})


def idle_verification(item):
    query=command(['osmo','workflow','query',item['workflow'],'--format-type','json']);running_tasks(query)
    allocation=read(Path(item.get('allocation_build')or item['build'])/'spec.json')
    control=allocation['allocation_control_base']+'/'+item['workflow']
    script='''from pathlib import Path
import json,subprocess,time
control=Path(__CONTROL__);node=json.loads((control/'nodes/native.json').read_text());native=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:cmd=(p/'cmdline').read_bytes().replace(b'\\0',b' ').decode(errors='replace')
 except OSError:continue
 if any(t in cmd for t in ['launch_experiment.py','efe_pear_yam_server.py','efe_astra_wide_worker.py']):native.append(int(p.name))
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()
print('EFE_CAPTURE '+json.dumps({'node':node,'native_processes':native,'gpu_processes':gpu,'observed_unix':time.time()}),flush=True)
'''.replace('__CONTROL__',repr(control))
    proof=remote(item['workflow'],script,task='native')
    if proof['node']['phase']not in {'idle_failed','idle_finished','waiting_for_job'}or proof['native_processes']or proof['gpu_processes']:
        raise RuntimeError('Retained allocation has an active owner/workload; preserved')
    return proof,control


def reuse(item,*,job_id,reasoning=False):
    primary=read(DEST/'primary-launch.json');queue=read(DEST/'primary-native-queue.json')
    if any(v['workflow']==item['workflow']and not v.get('superseded')
           and v.get('model_job_id')==primary.get('job_id')for v in queue.values()):
        raise RuntimeError('Active learner simulator allocation is reserved for RL')
    item={**item,'allocation_build':item.get('allocation_build')or item['build']}
    proof,control=idle_verification(item);write(ROOT/(item['workflow']+'-idle-proof.json'),proof)
    build,label=prepared(item,reasoning=reasoning)
    from ..astra_expert.client import credential
    with tempfile.TemporaryDirectory(prefix='rp_jhh_astra_node_')as temp:
        key=Path(temp)/'rp_jhh_astra_api_key';fd=os.open(key,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'w')as stream:stream.write(credential())
        result=subprocess.run(['osmo','workflow','rsync','upload',item['workflow'],'native',
            str(key)+':/dev/shm/','--no-progress'],capture_output=True,text=True,timeout=120)
        if result.returncode:raise RuntimeError('Private node-local teacher key upload failed')
    receipt=submit(item,build,replace_running=False,job_id=job_id)
    state={'workflow':item['workflow'],'build':str(build),'allocation_build':item['allocation_build'],
        'allocation_control':control,'job_id':receipt['job_id'],'label':label,
        'tasks':read(build/'spec.json')['teacher_task_names'],'teacher_root':TEACHER_ROOT,
        'phase':'submitted_in_retained_allocation','retained_gpus_reused':2,'new_gpus_allocated':0}
    record_launch(state)
    return state


def launch(item,*,pool,priority='LOW',reasoning=False):
    build,label=prepared(item,reasoning=reasoning)
    receipt=submit_reconciled(build,pool=pool,priority=priority,nodes=1)
    state={**receipt,'workflow':receipt['name'],'build':str(build),'allocation_build':str(build),
        'label':label,'pool':pool,'priority':priority,'phase':'scheduler_submitted',
        'teacher_root':TEACHER_ROOT,'tasks':read(build/'spec.json')['teacher_task_names']}
    record_launch(state)
    return state


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--launch-core',action='store_true')
    args=parser.parse_args()
    if not args.launch_core:return
    register_credential();primary=read(DEST/'primary-launch.json');queue=read(DEST/'primary-native-queue.json')
    wanted=['native-efe','rtx-robodojo-0','rtx-robolab-0','rtx-yamlab-0']
    for slot in wanted:
        label='efe'if slot=='native-efe'else slot.split('-')[1]
        existing=[v for v in read(ROOT/'launches.json',{}).values()
                  if v.get('label')==label and not v.get('superseded_by')]
        if existing:
            print(json.dumps({'label':label,'already_owned_workflows':[v['workflow']for v in existing]}),flush=True)
            continue
        item=next(v for v in queue.values()if not v.get('superseded')and v.get('model_job_id')==primary['job_id']
                  and read(Path(v['build'])/'spec.json').get('allocation_slot')==slot)
        state=launch(item,pool='groot-h100-01'if slot=='native-efe'else'groot-l40-03',priority='LOW')
        print(json.dumps({k:state.get(k)for k in ['workflow','label','phase','pool','priority']}),flush=True)


if __name__=='__main__':main()
