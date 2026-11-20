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
from .efe_pear_submission import submit_reconciled,scheduling_priority,simulator_pool
from .efe_pear_worker_bundle import FORMAT as BUNDLE_FORMAT,bootstrap

ARM='moe-warmstart496-pearplusplus'
LFS_ROOT='/mnt/lfs/shared/rpunamiya/rp_jhh_pear_scaled_20261116_v1'
PROFILES=[(6,'HIGH'),(4,'HIGH'),(6,'LOW'),(4,'LOW')]
NATIVE_QUEUE_TIMEOUT='45m'
EXTRA=['efe_pear_primary_worker.py','efe_pear_primary_native.py','efe_pear_primary_rtx.py',
    'efe_pear_native_entry.py','efe_pear_rtx_env.py','efe_pear_remote_worker.py','efe_pear_gpu_cdi.py',
    'efe_pear_yam_server.py','efe_pear_yam_session_policy.py','efe_pear_yam_api_policy.py',
    'efe_pear_yam_probe_entry.py','efe_pear_yam_contract.py','efe_pear_arena_compat.py','efe_pear_yam_reset_bank.py',
    'efe_pear_float32_json.py','efe_pear_allocation_keeper.py','efe_pear_job_cleanup.py']


def validate_trainer_cpu_capacity(cpus,*,storage_units=32,ranks=8):
    # Each storage unit and its controller reserve a CPU; the driver reserves
    # one, and verl's colocated worker placement reserves three per GPU rank.
    required=storage_units+1+1+3*ranks
    if cpus<required:
        raise ValueError(f'Trainer Ray capacity {cpus} CPUs cannot fit queue, driver and rank reservations ({required})')
    return required


def validate_rollout_cpu_capacity(nodes,cpus,*,inference_cpus_per_node=20):
    # These actors reserve a full CPU although the physics/rendering workers
    # run separately. Include both reasoning pools and conservative TP4
    # inference reservations; the four-node/56-CPU layout started only 152 of
    # 160 native proxies, leaving eight YAM proxies permanently unscheduled.
    required=32+128+6+2+nodes*inference_cpus_per_node
    if nodes*cpus<required:
        raise ValueError(f'Rollout Ray capacity {nodes*cpus} CPUs cannot fit inference and simulator proxies ({required})')
    return required


def allocation_profile(attempt):
    if type(attempt)is not int or attempt<0:
        raise ValueError('Allocation attempt must be a nonnegative integer')
    if attempt<len(PROFILES):return PROFILES[attempt]
    # Once both layouts and priorities have been tried, keep the smaller
    # qualified layout. Wrapping the initial sweep requested 56 model GPUs
    # again immediately after a 40-GPU allocation suffered a startup failure.
    return 4,'HIGH'if attempt%3==0 else'LOW'


def prepare(profile=0,*,standby=False,priority_override=None,control_interface=None,stage_runtime_only=False,
            rollout_nodes_override=None):
    if rollout_nodes_override is not None and(type(rollout_nodes_override)is not int or rollout_nodes_override not in {4,5,6,7}):
        raise ValueError('Explicit model layouts must have four to seven rollout nodes')
    require_unpaused();admission=compile_admission(DEST);qualification=model_qualification(DEST)
    control=read(DEST/'control.json',{})
    control_interface=control_interface or control.get('next_control_interface')
    eef_interface=None
    if control_interface:
        from .efe_pear_eef import enabled
        enabled(control_interface)
        from .efe_pear_eef_admission import qualify,admission as eef_admission,interface_metadata
        if not stage_runtime_only:qualify(DEST,BASE/'cluster')
        eef_interface=interface_metadata(BASE/'cluster')
        admission=eef_admission(admission,eef_interface)
    elif stage_runtime_only:raise ValueError('Deferred native qualification is only available to the EEF fork')
    from .efe_pear_native_finalization import VERSION as FINALIZATION_VERSION
    admission['curriculum']['native_reward_finalization']=FINALIZATION_VERSION
    if control.get('warmup_sampling'):
        admission['curriculum']['warmup_sampling']=control['warmup_sampling']
    if 'retained_groups_per_update'in control:
        admission['curriculum']['retained_groups_per_update']=control['retained_groups_per_update']
    policy=control.get('next_sampling_policy','strict-family-v1')
    if policy!='strict-family-v1':
        admission['curriculum']['sampling_policy']=policy
        from .efe_pear_curriculum_state import CurriculumState
        CurriculumState(admission['curriculum'])
        from .recover_efe_pear_numeric import qualified
        if not eef_interface and not qualified(read(DEST/'native-float32-proof.json',{})):
            raise PendingAdmission('The replacement recipe requires the current real native float32 action proof')
        if not eef_interface and not read(DEST/'recipe-transition.json',{}).get('validation_passed'):
            raise PendingAdmission('The replacement recipe awaits its known-training-case inference check')
    package=prepare_training_bundle();common=Path(package['bundle'])
    sources={name:common/name for name in package['files']}
    sources.update({name:BASE/'cluster'/name for name in EXTRA})
    parent=next(m for m in read(DEST/'model-files-ready.json')['models']if m['kind']=='moe_warmstart496')
    rollout_nodes,priority=allocation_profile(profile)
    if rollout_nodes_override is not None:rollout_nodes=rollout_nodes_override
    priority=scheduling_priority('groot-h100',priority,control)
    if priority_override is not None:
        if priority_override not in {'HIGH','LOW'}:raise ValueError('Invalid explicit allocation priority')
        priority=priority_override
    l40_pool=simulator_pool('groot-l40-02',control)
    if l40_pool=='groot-l40s-01':
        ready=read(DEST/'l40s01-storage-ready.json',{})
        if(not ready.get('passed')or not ready.get('transport_roundtrip_passed')
            or ready.get('transport_sha256')!=hashlib.sha256((BASE/'cluster/efe_pear_object_transport.py').read_bytes()).hexdigest()):
            raise PendingAdmission('L40S-01 pinned asset staging and cross-filesystem transport verification are pending')
    native_image=yaml.safe_load(TEMPLATE.read_text())['workflow']['tasks'][0]['image']
    data={'curriculum.json':admission['curriculum'],'native-membership.json':admission['native_membership'],
        'model-qualification.json':qualification,'reasoning-membership.json':read(DEST/'reasoning-remote-qualified/membership.json')}
    serialized={name:(json.dumps(value,sort_keys=True)+'\n').encode()for name,value in data.items()}
    identity=hashlib.sha256(serialized['curriculum.json']).hexdigest()[:12]
    prefix='rp_jhh_pear_moe_warmstart496_eef_'if eef_interface else'rp_jhh_pear_moe_warmstart496_full_'
    run=REMOTE+'/runs/'+prefix+identity
    spec={'run_id':ARM+'-eef-v1'if eef_interface else ARM,'run':run,'model':parent['path'],'model_revision':parent['revision'],
        'runtime':REMOTE+'/runtime','pear_source':REMOTE+'/sources/pear-'+PEAR_REVISION,
        'rollout_nodes':rollout_nodes,'training_steps':32,'primary_training':True,
        'trainer_ray_cpus':64,'rollout_ray_cpus':64,
        'checkpoint_archive':'s3://rp-jhh-pear-checkpoints/rpunamiya/rp_jhh_pear_moe_warmstart496_full_'+identity,
        'adapter_digests':{name:hashlib.sha256(path.read_bytes()).hexdigest()for name,path in sources.items()},
        'data_digests':{name:hashlib.sha256(raw).hexdigest()for name,raw in serialized.items()},
        'admitted_counts':admission['counts'],'profile':profile,'priority':priority,
        'model_queue_timeout':'12h',
        'worker_bundle_format':BUNDLE_FORMAT,'allocation_lifecycle':'persistent_cluster',
        'allocation_control_base':REMOTE+'/allocations','standby':bool(standby)}
    if eef_interface:
        spec.update(control_interface=control_interface,eef_release=run+'/native-eef-release.json',
            runtime_warmup_only_until_native_release=bool(stage_runtime_only))
    spec['l40_pool']=l40_pool
    if l40_pool=='groot-l40s-01':
        wire=hashlib.sha256(json.dumps([spec['adapter_digests'],profile,l40_pool],sort_keys=True).encode()).hexdigest()[:16]
        spec.update(object_transport_uri=spec['checkpoint_archive']+'/transport/'+wire,
            l40_native_root=LFS_ROOT+'/runs/'+wire,l40_assets=ready['assets_root'])
    validate_trainer_cpu_capacity(spec['trainer_ray_cpus'])
    validate_rollout_cpu_capacity(rollout_nodes,spec['rollout_ray_cpus'])
    digest=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()[:12]
    build=BUILD_ROOT/('rp-jhh-pear-primary-'+digest)
    if not(build/'source-manifest.json').exists():
        build.mkdir()
        for name,path in sources.items():shutil.copyfile(path,build/name)
        for name,raw in serialized.items():(build/name).write_bytes(raw)
        write(build/'spec.json',spec)
        (build/'bootstrap.py').write_text(bootstrap(build,[*sources,*data,'spec.json'],'efe_pear_allocation_keeper.py'))
        resources={'trainer':{'gpu':8,'cpu':64,'memory':'1536Gi','storage':'300Gi','platform':'dgx-h100'},
            'rollout':{'gpu':8,'cpu':64,'memory':'512Gi','storage':'250Gi','platform':'dgx-h100'}}
        tasks=[]
        for role in ['trainer',*['rollout-'+str(i)for i in range(rollout_nodes)]]:
            task={'name':role,'resource':'trainer'if role=='trainer'else'rollout','image':native_image,
                'command':['python3'],'args':['-u','/tmp/rp-jhh-pear-primary/bootstrap.py'],
                'environment':{'EFE_ROLE':role,'EFE_WORKFLOW_ID':'{{workflow_id}}','EFE_HEAD_HOST':'{{host:trainer}}',
                    'PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1','NVIDIA_DRIVER_CAPABILITIES':'all',
                    'PEAR_ALLOCATION_BASE':spec['allocation_control_base'],'PEAR_KEEPER_ENTRY':'efe_pear_primary_worker.py',
                    'PEAR_ALLOCATION_ROLES':json.dumps(['trainer',*['rollout-'+str(i)for i in range(rollout_nodes)]]),
                    'PEAR_KEEPER_AUTOSTART':'0'if standby else'1'},
                'files':[{'localpath':'bootstrap.py','path':'/tmp/rp-jhh-pear-primary/bootstrap.py'}]}
            if role=='trainer':
                task['lead']=True
                task['credentials']={'rp-jhh-pear-swift-archive-20261118':{'RP_JHH_PRIVATE_ARCHIVE_CONFIGURATION':'configuration'}}
            tasks.append(task)
        workflow={'workflow':{'name':'leact-'+build.name,'resources':resources,
            'timeout':{'queue_timeout':spec['model_queue_timeout'],'exec_timeout':'48h'},
            'labels':{'pear_run_id':ARM,'pear_phase':'primary_training'},
            'groups':[{'name':'model-cluster','tasks':tasks}]}}
        (build/'workflow.yaml').write_text(yaml.safe_dump(workflow,sort_keys=False));freeze_sources(build)
    write(DEST/('backup-ready.json'if standby else'primary-ready.json'),{'build':str(build),'run':run,'counts':admission['counts'],
        'model_cluster_h100s':8*(1+rollout_nodes),'native_efe_h100s':8,'target_total_h100s':8*(2+rollout_nodes),
        'primary_rl_updates':0,'qualified':not stage_runtime_only,
        'runtime_preparation_only':bool(stage_runtime_only)})
    return build


def launch(profile=0,*,pool='groot-h100-02'):
    if pool not in {'groot-h100-01','groot-h100-02'}:
        raise ValueError('Primary model requires a qualified AMLFS07 H100 pool')
    with(DEST/'primary-submit.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        state=read(DEST/'primary-launch.json',{})
        if state.get('workflow')and not state.get('terminal_confirmed'):
            return state
        # Allocations start idle. The controller selects one ready cluster
        # before submitting a job, so primary and backup never share a writer.
        target=read(DEST/'control.json',{}).get('primary_model_h100_target')
        layout={'rollout_nodes_override':7}if target==64 else{}
        build=prepare(profile,standby=True,**layout);spec=read(build/'spec.json')
        result=submit_reconciled(build,priority=spec['priority'],pool=pool,nodes=1+spec['rollout_nodes'])
        history=state.get('history',[])
        if state.get('workflow'):history.append({k:state[k]for k in ['workflow','profile','scheduler_status']if k in state})
        state={**result,'workflow':result['name'],'build':str(build),'profile':profile,'history':history,
            'pool':pool,'priority':spec['priority'],'run':spec['run'],'run_id':spec['run_id'],
            'gpu':8*(1+spec['rollout_nodes']),'primary_rl_updates':0,'native_workers_launched':False,
            'phase':'primary_submitted','standby':True,'allocation_lifecycle':'persistent_cluster'}
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
        'lanes_per_gpu':1,'startup_stagger_seconds':10,'retry_native_infrastructure':True,
        'native_config':{'qualification_only':False,'membership_file':run+'/native-membership.json',
            'action_config':{**native_setup['fixed_action_config'],'model':'external-pear-policy',
                'allowed_action_types_by_robot':{r:['joint_position']for r in
                    ['franka-allegro-bimanual','franka-robotiq-2f85','i2rt-yam-bimanual']}}},
        'suites':sorted({r['suite']for r in rows}),'transport_root':run+'/transport',
        'finished':run+'/finished.json','head_heartbeat':run+'/head-heartbeat.json',
        'episode_archive':model['checkpoint_archive']+'/native-episodes','allocation_slot':'native-efe'}
    if model.get('control_interface'):spec['control_interface']=model['control_interface']
    base=yaml.safe_load(TEMPLATE.read_text())['workflow']['tasks'][0]['image']
    builds.append(worker_build(spec,files,'efe_pear_primary_native.py',base,'python3',
        gpu=8,cpu=64,memory='512Gi',storage='400Gi',platform='dgx-h100',privileged=True))
    l40_pool=model.get('l40_pool','groot-l40-02');cross_pool=l40_pool=='groot-l40s-01'
    l40_root=model['l40_native_root']if cross_pool else run
    platform='ovx-l40s'if cross_pool else'ovx-l40'
    images=read(DEST/'native-rtx-images.json')
    for suite,count in [('robodojo',4),('robolab',2),('yamlab',2)]:
        matching=next(v for v in read(DEST/'rtx-transport-queue.json').values()if v['suite']==suite)
        source=read(Path(matching['build'])/'spec.json')
        for index in range(count):
            spec={k:source[k]for k in ['payload','payload_sha256','files','suite','native_tool_relative']}
            spec['payload']=(LFS_ROOT if cross_pool else REMOTE)+'/sources/'+Path(spec['payload']).name
            spec.update(kind='rtx',owner_model_workflow=model_workflow,replica=index,output=l40_root+'/native-rtx/'+suite,transport_root=l40_root+'/transport',
                finished=l40_root+'/finished.json',episode_archive=model['checkpoint_archive']+'/native-episodes',allocation_slot='rtx-'+suite+'-'+str(index))
            if model.get('control_interface'):spec['control_interface']=model['control_interface']
            builds.append(worker_build(spec,files,'efe_pear_primary_rtx.py',images[suite]['image'],
                '/opt/robolab/.venv/bin/python'if suite=='robolab'else'/opt/venv/bin/python',
                gpu=1,cpu=8,memory='64Gi',storage='150Gi',platform=platform))
    names=sorted(read(DEST/'native-yam-generated-resets/generation-report.json')['tasks'])
    for offset in range(0,len(names),2):
        old=prepare_native_yam_check(names[offset:offset+2],generated_resets=True,native_transport_probe=True)
        spec=read(old/'spec.json');spec.update(owner_model_workflow=model_workflow,
            output=l40_root+'/native-yam',transport_root=l40_root+'/transport',finished=l40_root+'/finished.json',allocation_slot='native-yam-'+str(offset//2))
        if model.get('control_interface'):
            from .efe_pear_float32_json import VERSION as NUMERIC_VERSION
            spec.update(control_interface=model['control_interface'],numeric_representation=NUMERIC_VERSION)
        from .recover_efe_pear_numeric import qualified
        if qualified(read(DEST/'native-float32-proof.json',{})):
            from .efe_pear_float32_json import VERSION
            spec['numeric_representation']=VERSION
        # Pool-local assets and inputs are byte-verified before admission;
        # L40S-01 uses the object bridge to reach the AMLFS07 model cluster.
        spec['source_archive_path']=(LFS_ROOT if cross_pool else REMOTE)+'/sources/'+Path(spec['source_archive_path']).name
        spec['generated_reset_archive']=(LFS_ROOT if cross_pool else REMOTE)+'/sources/'+Path(spec['generated_reset_archive']).name
        spec['assets_release']['assets_root']=model['l40_assets']if cross_pool else'/mnt/amlfs-07/shared/datasets/gear_sim/assets'
        if cross_pool:spec['asset_aliases']=['/mnt/amlfs-07/shared/datasets/gear_sim/assets']
        builds.append(worker_build(spec,files,'efe_pear_yam_server.py',spec['runtime_image'],
            '/opt/gr00t-sim/venv/bin/python',gpu=2,cpu=12,memory='128Gi',storage='300Gi',platform=platform))
    if cross_pool:
        bridge={'kind':'object_bridge','owner_model_workflow':model_workflow,'root':l40_root+'/transport',
            'uri':model['object_transport_uri']}
        builds.append(worker_build(bridge,files,'efe_pear_object_transport.py',base,'python3',
            gpu=0,cpu=4,memory='24Gi',storage='10Gi',platform='ovx-l40s'))
    return builds


def worker_build(spec,files,worker,image,python,**resources):
    spec=dict(spec);files=dict(files)
    persistent=bool(spec.get('owner_model_workflow')and resources.get('gpu',0)>0)
    if persistent:
        files['efe_pear_allocation_keeper.py']=BASE/'cluster/efe_pear_allocation_keeper.py'
        spec.update(allocation_lifecycle='persistent_cluster',allocation_job_entry=worker,
            allocation_control_base=str(Path(spec['transport_root']).parent/'allocations'))
    privileged=resources.pop('privileged',False)
    key=hashlib.sha256(json.dumps([spec,worker,image,resources,privileged,BUNDLE_FORMAT,NATIVE_QUEUE_TIMEOUT,
        {n:hashlib.sha256(p.read_bytes()).hexdigest()for n,p in files.items()}],sort_keys=True).encode()).hexdigest()[:12]
    build=BUILD_ROOT/('rp-jhh-pear-worker-'+key)
    if(build/'source-manifest.json').exists():return build
    build.mkdir()
    for name,path in files.items():shutil.copyfile(path,build/name)
    write(build/'spec.json',spec)
    (build/'bootstrap.py').write_text(bootstrap(build,[*files,'spec.json'],'efe_pear_allocation_keeper.py'if persistent else worker))
    task={'name':'native','image':image,'command':[python],'args':['-u','/tmp/pear-native/bootstrap.py'],
        'environment':{'EFE_WORKFLOW_ID':'{{workflow_id}}','PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1',
            'NVIDIA_DRIVER_CAPABILITIES':'all','PEAR_PRIMARY_TRAINING':'0'if spec.get('qualification_only')is True else'1','OMP_NUM_THREADS':'2'},
        'files':[{'localpath':'bootstrap.py','path':'/tmp/pear-native/bootstrap.py'}]}
    if persistent:
        task['environment'].update(PEAR_ALLOCATION_BASE=spec['allocation_control_base'],
            PEAR_KEEPER_ENTRY=worker,PEAR_ALLOCATION_ROLES='["native"]')
    if worker in {'efe_pear_primary_native.py','efe_pear_primary_rtx.py',
        'efe_pear_object_transport.py','efe_pear_object_probe.py','efe_pear_file_relay.py'}:
        task['credentials']={'rp-jhh-pear-swift-archive-20261118':{'RP_JHH_PRIVATE_ARCHIVE_CONFIGURATION':'configuration'}}
    if privileged:task['privileged']=True
    value={'workflow':{'name':'leact-'+build.name,'resources':{'default':resources},
        'timeout':{'queue_timeout':NATIVE_QUEUE_TIMEOUT,'exec_timeout':'48h'},
        'labels':{'pear_run_id':ARM,'pear_phase':'primary_native_workers'},'tasks':[task]}}
    (build/'workflow.yaml').write_text(yaml.safe_dump(value,sort_keys=False));freeze_sources(build)
    return build


def cancel_native_workers(model_workflow,*,explicit_user_release=False):
    if not explicit_user_release:
        write(DEST/'native-allocation-preservation.json',{'owner_model_workflow':model_workflow,
            'preserved':True,'reason':'Training failure or recipe change does not authorize releasing simulator allocations.'})
        return
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


def standby_allocation_paths():
    return[DEST/'backup-allocation.json',*sorted((DEST/'model-capacity-requests').glob('*.json'))]


def promote_allocated_backup(state,path=None):
    path=Path(path)if path is not None else DEST/'backup-allocation.json'
    backup=read(path,{})
    if backup.get('eligible_for_promotion')is False:return None
    if not backup.get('standby')or not backup.get('workflow')or backup['workflow']==state.get('workflow'):return None
    # Never start a second writer for the same experiment while the primary
    # allocation is queued or alive. A scheduler-ended request has no nodes
    # left to preserve; an already allocated backup can take over at once.
    if state.get('workflow'):
        current=command(['osmo','workflow','query',state['workflow'],'--format-type','json'])
        current_spec=read(Path(state['build'])/'spec.json')
        safely_idle=(state.get('standby')and current_spec.get('allocation_lifecycle')=='persistent_cluster')
        if not current['status'].startswith('FAILED')and not safely_idle:return None
    candidate=command(['osmo','workflow','query',backup['workflow'],'--format-type','json'])
    backup['scheduler_status']=candidate['status'];write(path,backup)
    if candidate['status']!='RUNNING':return None
    from .efe_pear_allocation_jobs import submit as submit_job,running_tasks,previous_model_writer
    try:running_tasks(candidate)
    except RuntimeError:return None
    allocated=read(Path(backup['build'])/'spec.json')
    layout={'rollout_nodes_override':allocated['rollout_nodes']}if allocated['rollout_nodes']in {5,7} else{}
    try:build=prepare(allocated['profile'],standby=True,priority_override=backup['priority'],**layout)
    except PendingAdmission as exc:
        backup['phase']='allocation_held_for_current_recipe_validation';backup['reason']=str(exc)
        write(path,backup);return None
    spec=read(build/'spec.json')
    if(spec['model_revision']!=allocated['model_revision']or spec['rollout_nodes']!=allocated['rollout_nodes']):
        raise ValueError('Backup promotion changed the model initialization or allocated roles')
    previous=previous_model_writer(backup['workflow'],spec['run'])
    receipt=submit_job(backup,build,previous_workflow=previous)
    history=state.get('history',[])
    if state.get('workflow'):history=history+[{k:state[k]for k in ['workflow','profile','scheduler_status']if k in state}]
    value={**backup,**receipt,'allocation_build':backup.get('allocation_build',backup['build']),
        'build':str(build),'run':spec['run'],'run_id':spec.get('run_id',ARM),'profile':spec['profile'],
        'standby':False,'native_workers_launched':False,'history':history,'phase':'job_submitted_in_existing_allocation',
        'primary_rl_updates':0,'scheduler_status':'RUNNING'}
    if state.get('standby')and not current['status'].startswith('FAILED'):
        write(path,{**state,'standby':True,'demoted_to_standby':True})
    else:
        backup.update(standby=False,promoted_to_primary=True,job_id=receipt['job_id']);write(path,backup)
    write(DEST/'primary-launch.json',value);return value


def reuse_native_allocation(build,queue,*,live_workflows=None,model_job_id=None):
    spec=read(Path(build)/'spec.json');slot=spec.get('allocation_slot')
    if not slot:return None
    from .efe_pear_allocation_jobs import submit as submit_job
    from .efe_pear_eef_admission import protected_teacher_allocations
    protected=protected_teacher_allocations(DEST)
    for item in queue.values():
        if item.get('workflow')in protected:continue
        if item.get('reuse_blocked_reason'):continue
        if live_workflows is not None and item.get('workflow')not in live_workflows:continue
        old=read(Path(item['build'])/'spec.json')
        if old.get('allocation_slot')!=slot or old.get('allocation_lifecycle')!='persistent_cluster':continue
        status=command(['osmo','workflow','query',item['workflow'],'--format-type','json'])
        if status['status']!='RUNNING':continue
        # A different run can replace the worker job on the same GPU node.
        # This terminates a child process, never its OSMO allocation.
        identity=json.dumps([spec['owner_model_workflow'],model_job_id,build.name],sort_keys=True)
        job='native-'+hashlib.sha256(identity.encode()).hexdigest()[:16]
        staging=({'workflow':spec['owner_model_workflow'],'task':'trainer'}
            if old.get('allocation_control_base','').startswith('/mnt/amlfs-07/shared/rpunamiya/')else None)
        receipt=submit_job(item,build,replace_running=True,job_id=job,staging=staging)
        item['superseded']=True
        return{**item,**receipt,'allocation_build':item.get('allocation_build',item['build']),
            'build':str(build),'owner_model_workflow':spec['owner_model_workflow'],'reused_allocation':True,'superseded':False}
    return None


def recover_job_inside_allocation(state,observed):
    nodes=observed.get('nodes',[])
    if not any(n.get('phase')=='idle_failed'for n in nodes):return state
    state['phase']='allocation_held_after_job_failure'
    state['allocation_job_status']=observed
    # Preserve progress for an explicit checkpoint handoff. Startup failures
    # before any optimizer update can retry the identical pinned job directly.
    if observed.get('completed_updates',0):
        state['phase']='allocation_held_for_checkpoint_recovery';return state
    if any(n.get('phase')=='running'for n in nodes)or state.get('in_allocation_retries',0)>=2:return state
    from .efe_pear_allocation_jobs import submit as submit_job
    receipt=submit_job(state,state['build'])
    state.update(job_id=receipt['job_id'],native_workers_launched=False,
        in_allocation_retries=state.get('in_allocation_retries',0)+1,phase='job_retried_in_existing_allocation')
    return state


def provision_native(build,state,queue,live_workflows):
    if build.name in queue:
        item=queue[build.name]
        if state.get('job_id')and item.get('model_job_id')!=state['job_id']:
            from .efe_pear_allocation_jobs import submit as submit_job
            query=command(['osmo','workflow','query',item['workflow'],'--format-type','json'])
            if query['status']=='RUNNING':
                job='native-rebind-'+hashlib.sha256((state['job_id']+build.name).encode()).hexdigest()[:16]
                allocated=read(Path(item.get('allocation_build',item['build']))/'spec.json')
                staging=({'workflow':state['workflow'],'task':'trainer'}
                    if allocated.get('allocation_control_base','').startswith('/mnt/amlfs-07/shared/rpunamiya/')else None)
                item.update(submit_job(item,build,replace_running=True,job_id=job,staging=staging),model_job_id=state['job_id'])
                write(DEST/'primary-native-queue.json',queue)
        return
    spec=read(build/'spec.json')
    reused=reuse_native_allocation(build,queue,live_workflows=live_workflows,model_job_id=state['job_id'])
    if reused is not None:
        reused['model_job_id']=state['job_id'];queue[build.name]=reused
    else:
        fallback='groot-l40-02'if spec.get('kind')=='rtx'else'groot-l40-03'
        pool='groot-h100-01'if spec.get('kind')=='efe'else read(Path(state['build'])/'spec.json').get('l40_pool',fallback)
        priority=scheduling_priority(pool,state['priority'],read(DEST/'control.json',{}))
        receipt=submit_reconciled(build,priority=priority,pool=pool,nodes=1)
        queue[build.name]={**receipt,'workflow':receipt['name'],'build':str(build),'pool':pool,'priority':priority,
            'owner_model_workflow':state['workflow'],'model_job_id':state['job_id']}
    write(DEST/'primary-native-queue.json',queue)


def provision_native_workers(builds,state,queue,live_workflows):
    import pexpect
    deferred={}
    for build in builds:
        try:provision_native(build,state,queue,live_workflows)
        except(pexpect.TIMEOUT,pexpect.EOF,subprocess.TimeoutExpired,RuntimeError)as exc:
            # A stalled exec into one retained allocation must not prevent
            # independent simulators from starting. Reuse the same request
            # identity on the next tick; uncertain acceptance never releases
            # that allocation or submits another copy of its worker.
            deferred[build.name]={'error_type':type(exc).__name__,'summary':str(exc).splitlines()[0][:240]}
    write(DEST/'native-provisioning-deferred.json',{'model_workflow':state['workflow'],
        'model_job_id':state['job_id'],'updated_unix':time.time(),'workers':deferred})
    state['native_workers_launched']=not deferred
    state['native_workers_partially_launched']=any(build.name in queue for build in builds)
    return state


def tick():
    state=read(DEST/'primary-launch.json',{})
    from .efe_pear_backup import maintain as maintain_backup
    standby_paths=standby_allocation_paths()
    for standby_path in standby_paths:maintain_backup(standby_path)
    recovery=read(DEST/'checkpoint-recovery-required.json',{})
    if recovery.get('required'):
        # Retain queued/allocated capacity, but never activate a fresh writer
        # against old metrics while the interrupted checkpoint is unresolved.
        if state.get('workflow'):
            query=command(['osmo','workflow','query',state['workflow'],'--format-type','json'])
            state['scheduler_status']=query['status']
        state.update(phase='awaiting_explicit_checkpoint_recovery',checkpoint_recovery=recovery)
        write(DEST/'primary-launch.json',state);write(DEST/'primary-status.json',state)
        return state
    promoted=promote_allocated_backup(state)
    if promoted is None:
        for standby_path in standby_paths[1:]:
            promoted=promote_allocated_backup(state,standby_path)
            if promoted is not None:break
    if promoted is not None:return promoted
    if state.get('workflow')and state.get('scheduler_status')=='RUNNING':
        from .efe_pear_batch_transition import tick as transition_batch
        replacement=transition_batch(state)
        if replacement is not None:return replacement
        from .efe_pear_native_reward_transition import tick as transition_native_reward
        replacement=transition_native_reward(state)
        if replacement is not None:return replacement
    if state.get('workflow')and not state.get('terminal_confirmed'):
        from .efe_pear_recipe_transition import tick as transition_recipe
        replacement=transition_recipe(state)
        if replacement is not None:return replacement
    if not state or(state.get('terminal_confirmed')and state.get('source_repair_ready')):
        try:return launch(state.get('next_profile',state.get('profile',0)),pool=state.get('pool','groot-h100-02'))
        except PendingAdmission as exc:
            value={'phase':'awaiting_primary_qualification','reason':str(exc),'primary_rl_updates':0}
            write(DEST/'primary-status.json',value);return value
    query=command(['osmo','workflow','query',state['workflow'],'--format-type','json'])
    state['scheduler_status']=query['status'];write(DEST/'primary-launch.json',state)
    write(DEST/'primary-status.json',state)
    if query['status']=='RUNNING'and state.get('standby'):
        from .efe_pear_allocation_jobs import activate_embedded,previous_model_writer
        previous=previous_model_writer(state['workflow'],state['run'])
        receipt=activate_embedded(state,previous_workflow=previous)
        state.update(**receipt,standby=False,phase='job_submitted_in_existing_allocation')
        write(DEST/'primary-launch.json',state)
    if query['status']=='RUNNING'and state.get('job_id'):
        from .efe_pear_allocation_jobs import status as job_status,running_tasks
        running_tasks(query)
        observed=job_status(state)
        if observed is not None:state=recover_job_inside_allocation(state,observed)
        write(DEST/'primary-launch.json',state)
        if state.get('control_interface'):
            from .release_efe_pear_eef import tick as release_eef
            release_eef(state)
    if (query['status']=='RUNNING'and state.get('job_id')and query.get('start_time')
        and not state.get('native_workers_launched')):
        worker_queue=read(DEST/'primary-native-queue.json',{})
        live_rows=command(['osmo','workflow','list','--name','leact-rp-jhh-pear-worker',
            '--count','1000','--format-type','json'])['workflows']
        live_workflows={v['name']for v in live_rows if v['status']=='RUNNING'}
        state=provision_native_workers(native_builds(state['build'],state['workflow']),state,worker_queue,live_workflows)
        write(DEST/'primary-launch.json',state)
    if (state.get('native_workers_launched')or state.get('native_workers_partially_launched'))and query['status']=='RUNNING':
        from .efe_pear_additional_capacity import tick as retarget_additional
        retarget_additional(state)
        from .efe_pear_native_child_recovery import tick as recover_native_children
        recover_native_children(state)
        from .recover_efe_pear_graphics import tick as recover_graphics
        recover_graphics(state)
        from .recover_efe_pear_numeric import tick as recover_numeric
        recover_numeric(state)
        from .efe_pear_submission import infrastructure_failure as infra_retryable
        from .monitor_efe_pear_full_qualification import retry
        queue=read(DEST/'primary-native-queue.json',{})
        rows=command(['osmo','workflow','list','--name','rp-jhh-pear-worker','--count','1000','--format-type','json'])['workflows']
        lookup={v['name']:v for v in rows}
        attempt_limit=read(DEST/'control.json',{}).get('primary_native_allocation_attempts',8)
        if type(attempt_limit)is not int or not 1<=attempt_limit<=64:
            raise ValueError('Native allocation retry budget must be between 1 and 64')
        for item in queue.values():
            if item.get('owner_model_workflow')!=state['workflow']or item.get('superseded'):continue
            worker=lookup.get(item['workflow'],{});status=worker.get('status')
            item['scheduler_status']=status
            if status and status.startswith('FAILED')and item.get('attempt',0)<attempt_limit:
                detail=command(['osmo','workflow','query',item['workflow'],'--format-type','json'])
                if infra_retryable(detail):retry(item)
                else:item['source_failure']=True
            elif status=='COMPLETED':
                first=item.setdefault('unexpected_completion_checked_unix',time.time())
                if time.time()-first>180 and item.get('attempt',0)<attempt_limit:retry(item)
        write(DEST/'primary-native-queue.json',queue)
    if query['status'].startswith('FAILED'):
        # A scheduler-ended model allocation cannot be revived in place, but
        # its independent simulator allocations must remain available.
        from .efe_pear_submission import infrastructure_failure as infra_retryable
        if infra_retryable(query):
            state.update(terminal_confirmed=True,source_repair_ready=True,next_profile=state['profile']+1)
            write(DEST/'primary-launch.json',state)
            return tick()
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
                import traceback
                write(DEST/'primary-observer-error.json',{'error_type':type(exc).__name__,'error':str(exc)[-1200:],
                    'frames':[{'file':Path(f.filename).name,'line':f.lineno,'function':f.name}
                        for f in traceback.extract_tb(exc.__traceback__)]})
            time.sleep(90)


if __name__=='__main__':main()
