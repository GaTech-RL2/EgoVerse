"""Launch the authorized full-mixture warmstart arm after measured admission.

The dense arm is never enabled here. Model allocation profiles count the
additional eight-H100 native EFE node: 56+8=64, or 40+8=48. RTX workers are
separate because Isaac RGB requires them.
"""
import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import yaml

from .efe_pear_full_admission import compile_admission,model_qualification,PendingAdmission
from .launch_efe_pear_scaled import DEST,REMOTE,PEAR_REVISION,TEMPLATE,prepare_training_bundle,require_unpaused
from .launch_efe_training import freeze_sources
from .scale_efe import BASE,BUILD_ROOT,read,write,submit,command
from .efe_pear_submission import submit_reconciled
from .efe_pear_worker_bundle import FORMAT as BUNDLE_FORMAT,bootstrap

ARM='moe-warmstart496-pearplusplus'
PROFILES=[(6,'HIGH'),(4,'HIGH'),(6,'LOW'),(4,'LOW')]
NATIVE_QUEUE_TIMEOUT='45m'
EXTRA=['efe_pear_primary_worker.py','efe_pear_primary_native.py','efe_pear_primary_rtx.py',
    'efe_pear_native_entry.py','efe_pear_rtx_env.py','efe_pear_remote_worker.py','efe_pear_gpu_cdi.py',
    'efe_pear_yam_server.py','efe_pear_yam_session_policy.py','efe_pear_yam_api_policy.py',
    'efe_pear_yam_probe_entry.py','efe_pear_yam_contract.py','efe_pear_arena_compat.py','efe_pear_yam_reset_bank.py']


def validate_trainer_cpu_capacity(cpus,*,storage_units=32,ranks=8):
    # Each storage unit and its controller reserve a CPU; the driver reserves
    # one, and verl's colocated worker placement reserves three per GPU rank.
    required=storage_units+1+1+3*ranks
    if cpus<required:
        raise ValueError(f'Trainer Ray capacity {cpus} CPUs cannot fit queue, driver and rank reservations ({required})')
    return required


def allocation_profile(attempt):
    if type(attempt)is not int or attempt<0:
        raise ValueError('Allocation attempt must be a nonnegative integer')
    if attempt<len(PROFILES):return PROFILES[attempt]
    # Once both layouts and priorities have been tried, keep the smaller
    # qualified layout. Wrapping the initial sweep requested 56 model GPUs
    # again immediately after a 40-GPU allocation suffered a startup failure.
    return 4,'HIGH'if attempt%3==0 else'LOW'


def prepare(profile=0):
    require_unpaused();admission=compile_admission(DEST);qualification=model_qualification(DEST)
    package=prepare_training_bundle();common=Path(package['bundle'])
    sources={name:common/name for name in package['files']}
    sources.update({name:BASE/'cluster'/name for name in EXTRA})
    parent=next(m for m in read(DEST/'model-files-ready.json')['models']if m['kind']=='moe_warmstart496')
    rollout_nodes,priority=allocation_profile(profile)
    native_image=yaml.safe_load(TEMPLATE.read_text())['workflow']['tasks'][0]['image']
    data={'curriculum.json':admission['curriculum'],'native-membership.json':admission['native_membership'],
        'model-qualification.json':qualification,'reasoning-membership.json':read(DEST/'reasoning-remote-qualified/membership.json')}
    serialized={name:(json.dumps(value,sort_keys=True)+'\n').encode()for name,value in data.items()}
    identity=hashlib.sha256(serialized['curriculum.json']).hexdigest()[:12]
    run=REMOTE+'/runs/rp_jhh_pear_moe_warmstart496_full_'+identity
    spec={'run_id':ARM,'run':run,'model':parent['path'],'model_revision':parent['revision'],
        'runtime':REMOTE+'/runtime','pear_source':REMOTE+'/sources/pear-'+PEAR_REVISION,
        'rollout_nodes':rollout_nodes,'training_steps':32,'primary_training':True,
        'trainer_ray_cpus':64,'rollout_ray_cpus':56,
        'checkpoint_archive':'s3://rp-jhh-pear-checkpoints/rpunamiya/rp_jhh_pear_moe_warmstart496_full_'+identity,
        'adapter_digests':{name:hashlib.sha256(path.read_bytes()).hexdigest()for name,path in sources.items()},
        'data_digests':{name:hashlib.sha256(raw).hexdigest()for name,raw in serialized.items()},
        'admitted_counts':admission['counts'],'profile':profile,'priority':priority,
        'worker_bundle_format':BUNDLE_FORMAT}
    validate_trainer_cpu_capacity(spec['trainer_ray_cpus'])
    digest=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()[:12]
    build=BUILD_ROOT/('rp-jhh-pear-primary-'+digest)
    if not(build/'source-manifest.json').exists():
        build.mkdir()
        for name,path in sources.items():shutil.copyfile(path,build/name)
        for name,raw in serialized.items():(build/name).write_bytes(raw)
        write(build/'spec.json',spec)
        (build/'bootstrap.py').write_text(bootstrap(build,[*sources,*data,'spec.json'],'efe_pear_primary_worker.py'))
        resources={'trainer':{'gpu':8,'cpu':64,'memory':'1536Gi','storage':'300Gi','platform':'dgx-h100'},
            'rollout':{'gpu':8,'cpu':64,'memory':'512Gi','storage':'250Gi','platform':'dgx-h100'}}
        tasks=[]
        for role in ['trainer',*['rollout-'+str(i)for i in range(rollout_nodes)]]:
            task={'name':role,'resource':'trainer'if role=='trainer'else'rollout','image':native_image,
                'command':['python3'],'args':['-u','/tmp/rp-jhh-pear-primary/bootstrap.py'],
                'environment':{'EFE_ROLE':role,'EFE_WORKFLOW_ID':'{{workflow_id}}','EFE_HEAD_HOST':'{{host:trainer}}',
                    'PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1','NVIDIA_DRIVER_CAPABILITIES':'all'},
                'files':[{'localpath':'bootstrap.py','path':'/tmp/rp-jhh-pear-primary/bootstrap.py'}]}
            if role=='trainer':
                task['lead']=True
                task['credentials']={'rp-jhh-pear-swift-archive-20261118':{'RP_JHH_PRIVATE_ARCHIVE_CONFIGURATION':'configuration'}}
            tasks.append(task)
        workflow={'workflow':{'name':'leact-'+build.name,'resources':resources,
            'timeout':{'queue_timeout':'20m','exec_timeout':'48h'},
            'labels':{'pear_run_id':ARM,'pear_phase':'primary_training'},
            'groups':[{'name':'model-cluster','tasks':tasks}]}}
        (build/'workflow.yaml').write_text(yaml.safe_dump(workflow,sort_keys=False));freeze_sources(build)
    write(DEST/'primary-ready.json',{'build':str(build),'run':run,'counts':admission['counts'],
        'model_cluster_h100s':8*(1+rollout_nodes),'native_efe_h100s':8,'target_total_h100s':8*(2+rollout_nodes),
        'primary_rl_updates':0,'qualified':True})
    return build


def launch(profile=0,*,pool='groot-h100-02'):
    if pool not in {'groot-h100-01','groot-h100-02'}:
        raise ValueError('Primary model requires a qualified AMLFS07 H100 pool')
    with(DEST/'primary-submit.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        state=read(DEST/'primary-launch.json',{})
        if state.get('workflow')and not state.get('terminal_confirmed'):
            return state
        build=prepare(profile);spec=read(build/'spec.json')
        result=submit_reconciled(build,priority=spec['priority'],pool=pool,nodes=1+spec['rollout_nodes'])
        history=state.get('history',[])
        if state.get('workflow'):history.append({k:state[k]for k in ['workflow','profile','scheduler_status']if k in state})
        state={**result,'workflow':result['name'],'build':str(build),'profile':profile,'history':history,
            'pool':pool,'priority':spec['priority'],'run':spec['run'],'run_id':ARM,
            'gpu':8*(1+spec['rollout_nodes']),'primary_rl_updates':0,'native_workers_launched':False,
            'phase':'primary_submitted'}
        write(DEST/'primary-launch.json',state)
        write(DEST/'primary-status.json',state)
        return state


def native_builds(model_build,model_workflow):
    from .launch_efe_pear_warmstart import prepare_native_yam_check
    from .launch_efe_pear_rtx import FILES as RTX_FILES
    model=read(Path(model_build)/'spec.json');run=model['run'];membership=read(Path(model_build)/'native-membership.json')
    files={n:Path(model_build)/n for n in model['adapter_digests']}
    files['native-membership.json']=Path(model_build)/'native-membership.json'
    builds=[]
    native_setup=read(BUILD_ROOT/'efe-pear-native-qualification-20261114-v1/spec.json')
    rows=[v['row']for v in membership['training_tasks'].values()if v['row']['suite']in
        {'abc-sim','dexjoco','libero-max','metaworldplus','robocasa','vlabench'}]
    setup={k:native_setup[k]for k in ['repository','revision','podman','compose']}
    setup.update(tasks=rows,podman_storage_driver='overlay')
    spec={'kind':'efe','owner_model_workflow':model_workflow,'output':run+'/native-efe','native_setup':setup,
        'native_config':{'qualification_only':False,'membership_file':run+'/native-membership.json',
            'action_config':{**native_setup['fixed_action_config'],'model':'external-pear-policy',
                'allowed_action_types_by_robot':{r:['joint_position']for r in
                    ['franka-allegro-bimanual','franka-robotiq-2f85','i2rt-yam-bimanual']}}},
        'suites':sorted({r['suite']for r in rows}),'transport_root':run+'/transport',
        'finished':run+'/finished.json','head_heartbeat':run+'/head-heartbeat.json',
        'episode_archive':model['checkpoint_archive']+'/native-episodes'}
    base=yaml.safe_load(TEMPLATE.read_text())['workflow']['tasks'][0]['image']
    builds.append(worker_build(spec,files,'efe_pear_primary_native.py',base,'python3',
        gpu=8,cpu=64,memory='512Gi',storage='400Gi',platform='dgx-h100',privileged=True))
    images=read(DEST/'native-rtx-images.json')
    for suite,count in [('robodojo',4),('robolab',2),('yamlab',2)]:
        matching=next(v for v in read(DEST/'rtx-transport-queue.json').values()if v['suite']==suite)
        source=read(Path(matching['build'])/'spec.json')
        for index in range(count):
            spec={k:source[k]for k in ['payload','payload_sha256','files','suite','native_tool_relative']}
            spec['payload']=REMOTE+'/sources/'+Path(spec['payload']).name
            spec.update(kind='rtx',owner_model_workflow=model_workflow,replica=index,output=run+'/native-rtx/'+suite,transport_root=run+'/transport',
                finished=run+'/finished.json',episode_archive=model['checkpoint_archive']+'/native-episodes')
            builds.append(worker_build(spec,files,'efe_pear_primary_rtx.py',images[suite]['image'],
                '/opt/robolab/.venv/bin/python'if suite=='robolab'else'/opt/venv/bin/python',
                gpu=1,cpu=8,memory='64Gi',storage='150Gi',platform='ovx-l40'))
    names=sorted(read(DEST/'native-yam-generated-resets/generation-report.json')['tasks'])
    for offset in range(0,len(names),2):
        old=prepare_native_yam_check(names[offset:offset+2],generated_resets=True,native_transport_probe=True)
        spec=read(old/'spec.json');spec.update(owner_model_workflow=model_workflow,
            output=run+'/native-yam',transport_root=run+'/transport',finished=run+'/finished.json')
        # Production uses AMLFS07 alongside the H100 workers; validation on
        # AMLFS05 did not imply cross-filesystem IPC.
        spec['source_archive_path']=REMOTE+'/sources/'+Path(spec['source_archive_path']).name
        spec['generated_reset_archive']=REMOTE+'/sources/'+Path(spec['generated_reset_archive']).name
        spec['assets_release']['assets_root']='/mnt/amlfs-07/shared/datasets/gear_sim/assets'
        builds.append(worker_build(spec,files,'efe_pear_yam_server.py',spec['runtime_image'],
            '/opt/gr00t-sim/venv/bin/python',gpu=2,cpu=12,memory='128Gi',storage='300Gi',platform='ovx-l40'))
    return builds


def worker_build(spec,files,worker,image,python,**resources):
    privileged=resources.pop('privileged',False)
    key=hashlib.sha256(json.dumps([spec,worker,image,resources,privileged,BUNDLE_FORMAT,NATIVE_QUEUE_TIMEOUT,
        {n:hashlib.sha256(p.read_bytes()).hexdigest()for n,p in files.items()}],sort_keys=True).encode()).hexdigest()[:12]
    build=BUILD_ROOT/('rp-jhh-pear-worker-'+key)
    if(build/'source-manifest.json').exists():return build
    build.mkdir()
    for name,path in files.items():shutil.copyfile(path,build/name)
    write(build/'spec.json',spec)
    (build/'bootstrap.py').write_text(bootstrap(build,[*files,'spec.json'],worker))
    task={'name':'native','image':image,'command':[python],'args':['-u','/tmp/pear-native/bootstrap.py'],
        'environment':{'EFE_WORKFLOW_ID':'{{workflow_id}}','PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1',
            'NVIDIA_DRIVER_CAPABILITIES':'all','PEAR_PRIMARY_TRAINING':'1','OMP_NUM_THREADS':'2'},
        'files':[{'localpath':'bootstrap.py','path':'/tmp/pear-native/bootstrap.py'}]}
    if worker!='efe_pear_yam_server.py':
        task['credentials']={'rp-jhh-pear-swift-archive-20261118':{'RP_JHH_PRIVATE_ARCHIVE_CONFIGURATION':'configuration'}}
    if privileged:task['privileged']=True
    value={'workflow':{'name':'leact-'+build.name,'resources':{'default':resources},
        'timeout':{'queue_timeout':NATIVE_QUEUE_TIMEOUT,'exec_timeout':'48h'},
        'labels':{'pear_run_id':ARM,'pear_phase':'primary_native_workers'},'tasks':[task]}}
    (build/'workflow.yaml').write_text(yaml.safe_dump(value,sort_keys=False));freeze_sources(build)
    return build


def cancel_native_workers(model_workflow):
    queue=read(DEST/'primary-native-queue.json',{})
    for item in queue.values():
        if item.get('owner_model_workflow')!=model_workflow:continue
        if item.get('cancel_requested')or item.get('scheduler_status')=='COMPLETED'or str(item.get('scheduler_status','')).startswith('FAILED'):continue
        workflow=item['workflow']
        if not workflow.startswith('leact-rp-jhh-pear-worker-'):
            raise ValueError('Refusing to cancel a workflow outside this primary native-worker ledger')
        result=subprocess.run(['osmo','workflow','cancel',workflow,'--message',
            'Release simulator workers owned by the terminated primary model attempt'],capture_output=True,text=True,timeout=30)
        if result.returncode:raise RuntimeError('Native-worker cancellation needs reconciliation')
        item['cancel_requested']=True;write(DEST/'primary-native-queue.json',queue)


def tick():
    state=read(DEST/'primary-launch.json',{})
    if not state or(state.get('terminal_confirmed')and state.get('source_repair_ready')):
        try:return launch(state.get('profile',0),pool=state.get('pool','groot-h100-02'))
        except PendingAdmission as exc:
            value={'phase':'awaiting_primary_qualification','reason':str(exc),'primary_rl_updates':0}
            write(DEST/'primary-status.json',value);return value
    query=command(['osmo','workflow','query',state['workflow'],'--format-type','json'])
    state['scheduler_status']=query['status'];write(DEST/'primary-launch.json',state)
    write(DEST/'primary-status.json',state)
    if query.get('start_time')and not state.get('native_workers_launched'):
        worker_queue=read(DEST/'primary-native-queue.json',{})
        for build in native_builds(state['build'],state['workflow']):
            if build.name in worker_queue:continue
            spec=read(build/'spec.json')
            # Split the two renderer workloads across compatible shared-volume
            # pools; one pool rejected its single-GPU placements before startup.
            pool=('groot-h100-01'if spec.get('kind')=='efe'else
                'groot-l40-02'if spec.get('kind')=='rtx'else'groot-l40-03')
            receipt=submit_reconciled(build,priority=state['priority'],pool=pool,nodes=1)
            worker_queue[build.name]={**receipt,'workflow':receipt['name'],'build':str(build),
                'pool':pool,'priority':state['priority'],'owner_model_workflow':state['workflow']}
            write(DEST/'primary-native-queue.json',worker_queue)
        state['native_workers_launched']=True;write(DEST/'primary-launch.json',state)
    if state.get('native_workers_launched')and query['status']=='RUNNING':
        from .efe_pear_submission import infrastructure_failure as infra_retryable
        from .monitor_efe_pear_full_qualification import retry
        queue=read(DEST/'primary-native-queue.json',{})
        rows=command(['osmo','workflow','list','--name','rp-jhh-pear-worker','--count','100','--format-type','json'])['workflows']
        lookup={v['name']:v for v in rows}
        for item in queue.values():
            if item.get('owner_model_workflow')!=state['workflow']:continue
            worker=lookup.get(item['workflow'],{});status=worker.get('status')
            item['scheduler_status']=status
            if status and status.startswith('FAILED')and item.get('attempt',0)<8:
                detail=command(['osmo','workflow','query',item['workflow'],'--format-type','json'])
                if infra_retryable(detail):retry(item)
                else:item['source_failure']=True
            elif status=='COMPLETED':
                first=item.setdefault('unexpected_completion_checked_unix',time.time())
                if time.time()-first>180 and item.get('attempt',0)<8:retry(item)
        write(DEST/'primary-native-queue.json',queue)
    if query['status'].startswith('FAILED'):
        cancel_native_workers(state['workflow'])
        from .efe_pear_submission import infrastructure_failure as infra_retryable
        if infra_retryable(query):
            state['terminal_confirmed']=True;write(DEST/'primary-launch.json',state)
            return launch(state['profile']+1,pool=state.get('pool','groot-h100-02'))
        state['phase']='primary_source_failure_requires_review'
    write(DEST/'primary-status.json',state);return state


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--watch',action='store_true')
    parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    if args.prepare:print(prepare());return
    if not args.watch:print(json.dumps(tick(),indent=2));return
    with(DEST/'primary-watch.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not read(DEST/'control.json',{}).get('paused',True):
            try:
                value=tick()
                if value.get('scheduler_status')=='COMPLETED':return
            except Exception as exc:
                write(DEST/'primary-observer-error.json',{'error_type':type(exc).__name__,'error':str(exc)[-1200:]})
            time.sleep(90)


if __name__=='__main__':main()
