"""Warmstart-only RL launch preparation, with discarded router qualification."""
import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

import yaml

from .scale_efe import BASE,BUILD_ROOT,read,submit,write
from .launch_efe_training import freeze_sources
from .launch_efe_pear_scaled import DEST,REMOTE,PEAR_REVISION,TEMPLATE

ARM='moe-warmstart496-pearplusplus'


def prepare_router_check():
    control=read(DEST/'control.json',{})
    if ARM not in control.get('enabled_training_arms',[]):raise RuntimeError('Warmstart RL is not enabled')
    build=BUILD_ROOT/'rp-jhh-pear-warmstart496-router-check-v2'
    if (build/'source-manifest.json').exists():return build
    model=next(m for m in read(DEST/'model-files-ready.json')['models']if m['kind']=='moe_warmstart496')
    build.mkdir(exist_ok=True)
    spec={'model':model['path'],'model_revision':model['revision'],'model_type':model['model_type'],
        'observation':read(DEST/'model-check-observation.json'),'remote_root':REMOTE,
        'pear_source':REMOTE+'/sources/pear-'+PEAR_REVISION,
        'output':REMOTE+'/model-checks/frozen-router-warmstart496-v2',
        'checkpoint_local_root':'/dev/shm/rp_jhh_pear_discarded_checkpoints',
        'minimum_checkpoint_free_bytes':500_000_000_000,
        'primary_rl_updates':0,'parameter_update':'full_finetuning','lora_rank':0,
        'router_policy':'frozen_expert_selection_router',
        'scope':'Discarded numerical update and local checkpoint reload, before on-policy RL qualification.'}
    write(build/'spec.json',spec)
    for source,target in [('efe_pear_model_check.py','worker.py'),('efe_pear_router.py','efe_pear_router.py'),
                          ('efe_pear_moe_kernels.py','efe_pear_moe_kernels.py'),
                          ('efe_pear_bridge.py','efe_pear_bridge.py'),('efe_pear_context.py','efe_pear_context.py')]:
        shutil.copyfile(BASE/'cluster'/source,build/target)
    workflow=yaml.safe_load(TEMPLATE.read_text())
    workflow['workflow']['name']='leact-'+build.name
    workflow['workflow']['timeout']={'queue_timeout':'20m','exec_timeout':'6h'}
    workflow['workflow']['labels']={'pear_run_id':ARM,'pear_phase':'router_optimizer_qualification'}
    workflow['workflow']['resources']['default'].update(gpu=8,cpu=48,memory='1536Gi',storage='200Gi')
    task=workflow['workflow']['tasks'][0];task.pop('credentials',None);task.pop('privileged',None)
    task['args']=['-u','/tmp/efe-pear-model-check/worker.py']
    task['files']=[{'localpath':p.name,'path':'/tmp/efe-pear-model-check/'+p.name}for p in build.iterdir()if p.is_file()]
    (build/'workflow.yaml').write_text(yaml.safe_dump(workflow,sort_keys=False));freeze_sources(build)
    return build


def prepare_control_admission():
    if ARM not in read(DEST/'control.json',{}).get('enabled_training_arms',[]):
        raise RuntimeError('Warmstart RL is not enabled')
    build=BUILD_ROOT/'rp-jhh-pear-warmstart-control-admission-v3'
    if (build/'source-manifest.json').exists():return build
    parent=BASE/'artifacts/embodied-first-exam'
    membership=read(parent/'pear-v1/training-membership.json')['training_tasks']
    splits=read(parent/'generalization-v2/splits.json')
    allowed=set(splits['train_candidates'])
    protected=set(splits['protected_core_all_modes'])|set(splits['development'])
    if set(membership)-allowed or set(membership)&protected:raise ValueError('Training reset overlaps a protected split')
    tasks=[row for row in read(parent/'generalization-v2/inventory.json')if row['task']in membership]
    if len(tasks)!=40:raise ValueError('The frozen warmstart control corpus must have 40 tasks')
    prior=BUILD_ROOT/'efe-pear-native-qualification-20261114-v1'
    source=read(prior/'spec.json')
    spec={key:source[key]for key in ['repository','revision','podman','compose','fixed_action_config']}
    spec.update(tasks=tasks,archive_root=REMOTE+'/control-admission-v1',tracking_run_id=ARM,
                expected_scenes={key:values[0]['scene_sha256']for key,values in membership.items()})
    build.mkdir(exist_ok=True);write(build/'spec.json',spec)
    for name in ['efe_pear_bridge.py','efe_pear_context.py','efe_action_loop.py','efe_action_agent.py','efe_native_pilot.py',
                 'efe_pear_catalog_probe.py','efe_pear_gpu_cdi.py','efe_pear_native_image.py','efe_pear_editable_restore.py',
                 'efe_pear_checkpoint_archive.py','efe_pear_observation_check.py']:
        shutil.copyfile(BASE/'cluster'/name,build/name)
    shutil.copyfile(BASE/'cluster/efe_pear_control_admission.py',build/'worker.py')
    shutil.copyfile(prior/'scene_identity.py',build/'scene_identity.py')
    workflow=yaml.safe_load(TEMPLATE.read_text());workflow['workflow']['name']='leact-'+build.name
    workflow['workflow']['timeout']={'queue_timeout':'20m','exec_timeout':'8h'}
    workflow['workflow']['labels']={'pear_run_id':ARM,'pear_phase':'training_reset_admission'}
    workflow['workflow']['resources']['default'].update(gpu=8,cpu=32,memory='128Gi',storage='300Gi')
    task=workflow['workflow']['tasks'][0];task.pop('credentials',None);task['privileged']=True
    task['args']=['-u','/tmp/efe-pear-model-check/worker.py']
    task['files']=[{'localpath':p.name,'path':'/tmp/efe-pear-model-check/'+p.name}for p in build.iterdir()if p.is_file()]
    (build/'workflow.yaml').write_text(yaml.safe_dump(workflow,sort_keys=False));freeze_sources(build)
    return build


def start_monitor(kind):
    path=DEST/('warmstart-'+kind+'-check.lock')
    with path.open('a')as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return None
    with (DEST/('warmstart-'+kind+'-observer.log')).open('ab',buffering=0)as log:
        child=subprocess.Popen([sys.executable,'-u','-m',__package__+'.launch_efe_pear_warmstart',
            '--kind',kind,'--monitor'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            start_new_session=True)
    write(DEST/('warmstart-'+kind+'-observer-launch.json'),{'pid':child.pid})
    return child.pid


def prepare_reasoning_check():
    if ARM not in read(DEST/'control.json',{}).get('enabled_training_arms',[]):
        raise RuntimeError('Warmstart RL is not enabled')
    files=['efe_pear_reasoning_scenes.py','efe_pear_reasoning_qualify.py','efe_pear_reasoning_env.py','efe_pear_reasoning_tool.py',
           'efe_pear_curriculum.py','efe_pear_reasoning_cluster_check.py','efe_pear_sim_process.py','efe_pear_child.py']
    digests={name:hashlib.sha256((BASE/'cluster'/name).read_bytes()).hexdigest()for name in files}
    key=hashlib.sha256(json.dumps(digests,sort_keys=True).encode()).hexdigest()[:12]
    build=BUILD_ROOT/('rp-jhh-pear-reasoning-check-'+key)
    if (build/'source-manifest.json').exists():return build
    build.mkdir(exist_ok=True)
    for name in files:shutil.copyfile(BASE/'cluster'/name,build/('worker.py'if name=='efe_pear_reasoning_cluster_check.py'else name))
    write(build/'spec.json',{'output':REMOTE+'/reasoning-cluster-checks','remote_root':REMOTE,
        'pear_source':REMOTE+'/sources/pear-'+PEAR_REVISION,'primary_rl_updates':0,
        'train_scenes_per_family':64,'development_scenes_per_family':8,'source_digests':digests})
    workflow=yaml.safe_load(TEMPLATE.read_text());workflow['workflow']['name']='leact-'+build.name
    workflow['workflow']['labels']={'pear_run_id':ARM,'pear_phase':'reasoning_scene_and_transport_qualification'}
    workflow['workflow']['timeout']={'queue_timeout':'20m','exec_timeout':'2h'}
    workflow['workflow']['resources']['default'].update(gpu=8,cpu=48,memory='256Gi',storage='200Gi')
    task=workflow['workflow']['tasks'][0];task.pop('credentials',None);task.pop('privileged',None)
    task['args']=['-u','/tmp/efe-pear-model-check/worker.py']
    task['files']=[{'localpath':p.name,'path':'/tmp/efe-pear-model-check/'+p.name}for p in build.iterdir()if p.is_file()]
    (build/'workflow.yaml').write_text(yaml.safe_dump(workflow,sort_keys=False));freeze_sources(build)
    return build


def prepare_native_yam_check(task_names=None,generated_resets=False,native_transport_probe=False,*,runtime_override=None):
    if ARM not in read(DEST/'control.json',{}).get('enabled_training_arms',[]):
        raise RuntimeError('Warmstart RL is not enabled')
    metadata=read(DEST/'sources/gr00t-sim-source.json');source=Path(metadata['source'])
    runtime={**read(DEST/'native-yam-runtime-config.json'),**(runtime_override or{})}
    remote_root=runtime.get('validation_remote_root',REMOTE)
    archive=DEST/'sources'/('rp_jhh_native_yam_runtime_'+metadata['revision']+'.tar.gz')
    if not archive.exists():
        def package(info):
            return None if '__pycache__'in info.name or info.name.endswith('.pyc')or info.issym()else info
        with tarfile.open(archive,'w:gz')as stream:
            for name in ['gr00t_sim','configs','pyproject.toml','uv.lock']:
                stream.add(source/name,arcname=name,filter=package)
    digests={name:hashlib.sha256((BASE/'cluster'/name).read_bytes()).hexdigest()for name in
        ['efe_pear_yam_api_policy.py','efe_pear_yam_api_check.py','efe_pear_yam_probe_entry.py',
         'efe_pear_yam_contract.py','efe_pear_arena_compat.py','efe_pear_yam_reset_bank.py']}
    extra_sources=['efe_pear_yam_session_policy.py','efe_pear_file_rpc.py','efe_action_loop.py',
        'efe_pear_bridge.py','efe_pear_context.py','efe_pear_public_observation.py',
        'efe_pear_observation_check.py']if native_transport_probe else[]
    digests.update({name:hashlib.sha256((BASE/'cluster'/name).read_bytes()).hexdigest()for name in extra_sources})
    tasks=[x for x in read(DEST/'native-yam-split-v1.json')['entries']if x['split']=='train']
    if task_names and set(task_names)-{row['env_name']for row in tasks}:
        raise ValueError('Native API diagnostics must remain inside the training task catalog')
    bank=read(DEST/'native-yam-generated-resets/generation-report.json',{})if generated_resets else{}
    if generated_resets:
        if not bank.get('tasks'):raise ValueError('Independent native reset generation has not completed')
        if task_names and set(task_names)-set(bank['tasks']):raise ValueError('Requested task has no independent reset bank')
        task_names=task_names or list(bank['tasks'])
    release=read(source/'configs/fidelity/asset_release.json')
    if runtime.get('validated_portable_assets_root'):release={**release,'assets_root':runtime['validated_portable_assets_root']}
    gpu_limit=runtime.get('validation_gpus',8)
    if type(gpu_limit)is not int or not 1<=gpu_limit<=8:raise ValueError('Native validation GPU count must be 1..8')
    gpu_count=min(gpu_limit,len(task_names or tasks))
    spec={'output':remote_root+'/native-yam-api-checks','source_revision':metadata['revision'],
        'required_gpu_class':'RTX-capable L40 or L40S','pool':runtime['pool'],'platform':runtime.get('platform','ovx-l40'),
        'runtime_image':runtime['image'],
        'source_archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'payload_format':'shared_gzip_tar',
        'source_archive_path':remote_root+'/sources/'+archive.name,
        'assets_release':release,
        'tasks':[x for x in tasks if not task_names or x['env_name']in task_names],
        'catalog_task_count':len(tasks),'gpu':gpu_count,
        'cpu':6*gpu_count,'memory_gib':64*gpu_count,
        'primary_rl_updates':0,'scope':'Existing native fixture API qualification; training reset admission remains separate.'}
    if generated_resets:
        bank_key=hashlib.sha256(json.dumps({k:v['manifest_sha256']for k,v in bank['tasks'].items()},sort_keys=True).encode()).hexdigest()[:16]
        bank_archive=DEST/'sources'/('rp_jhh_native_yam_independent_resets_'+bank_key+'.tar.gz')
        if not bank_archive.exists():
            with tarfile.open(bank_archive,'w:gz')as stream:
                for name,row in bank['tasks'].items():
                    path=Path(row['manifest'])
                    if hashlib.sha256(path.read_bytes()).hexdigest()!=row['manifest_sha256']:raise ValueError('Generated reset inputs changed')
                    stream.add(path,arcname=name+'.json')
        spec.update(generated_reset_archive=remote_root+'/sources/'+bank_archive.name,
            generated_reset_archive_sha256=hashlib.sha256(bank_archive.read_bytes()).hexdigest(),
            generated_reset_tasks={k:v for k,v in bank['tasks'].items()if k in task_names},
            scope='Independent generated training reset bank qualification; native reward/production transport admission remains separate.')
    if native_transport_probe:
        if not generated_resets:raise ValueError('Native transport checks require an independent reset bank')
        spec['native_transport_probe']=True
    key=hashlib.sha256(json.dumps([spec,digests],sort_keys=True).encode()).hexdigest()[:12]
    build=BUILD_ROOT/('rp-jhh-pear-native-yam-api-'+key)
    if (build/'source-manifest.json').exists():return build
    build.mkdir(exist_ok=True);write(build/'spec.json',spec)
    for src,dst in [('efe_pear_yam_api_policy.py','efe_pear_yam_api_policy.py'),('efe_pear_yam_api_check.py','worker.py'),
                    ('efe_pear_yam_probe_entry.py','efe_pear_yam_probe_entry.py'),('efe_pear_yam_contract.py','efe_pear_yam_contract.py'),
                    ('efe_pear_arena_compat.py','efe_pear_arena_compat.py'),('efe_pear_yam_reset_bank.py','efe_pear_yam_reset_bank.py')]:
        shutil.copyfile(BASE/'cluster'/src,build/dst)
    for name in extra_sources:shutil.copyfile(BASE/'cluster'/name,build/name)
    w=yaml.safe_load(TEMPLATE.read_text());w['workflow']['name']='leact-'+build.name
    w['workflow']['labels']={'pear_run_id':ARM,'pear_phase':'native_yam_runtime_api_qualification'}
    w['workflow']['timeout']={'queue_timeout':'20m','exec_timeout':'8h'}
    w['workflow']['resources']['default'].update(gpu=spec['gpu'],cpu=spec['cpu'],
        memory=str(spec['memory_gib'])+'Gi',storage='300Gi')
    w['workflow']['resources']['default']['platform']=spec['platform']
    task=w['workflow']['tasks'][0];task.pop('credentials',None);task.pop('privileged',None)
    task['image']=runtime['image'];task['command']=['/opt/gr00t-sim/venv/bin/python']
    task['args']=['-u','/tmp/efe-pear-model-check/worker.py']
    task['environment']['OMNI_KIT_ALLOW_ROOT']='1'
    task['files']=[{'localpath':p.name,'path':'/tmp/efe-pear-model-check/'+p.name}for p in build.iterdir()if p.is_file()]
    (build/'workflow.yaml').write_text(yaml.safe_dump(w,sort_keys=False));freeze_sources(build)
    return build


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--submit',action='store_true')
    parser.add_argument('--kind',choices=['router','admission','reasoning','yam'],default='router')
    parser.add_argument('--monitor',action='store_true')
    parser.add_argument('--task-names',nargs='+')
    parser.add_argument('--generated-resets',action='store_true')
    parser.add_argument('--priority',choices=['HIGH','LOW'],default='HIGH')
    args=parser.parse_args()
    if args.monitor:return monitor(args.kind)
    if args.task_names and args.kind!='yam':raise ValueError('Task filtering is only for native YAM diagnostics')
    if args.generated_resets and args.kind!='yam':raise ValueError('Independent reset checks require native YAM')
    build=(prepare_native_yam_check(args.task_names,args.generated_resets)if args.kind=='yam'else
           {'router':prepare_router_check,'admission':prepare_control_admission,'reasoning':prepare_reasoning_check}[args.kind]())
    result={'build':str(build),'arm':ARM,'phase':args.kind+'_qualification','primary_rl_updates':0}
    pool=(read(DEST/'native-yam-runtime-config.json')['pool']if args.kind=='yam'else
          'groot-h100-01'if args.kind=='admission'else'groot-h100-02')
    result['pool']=pool
    if args.submit:
        result.update(submit(build,priority=args.priority,pool=pool,nodes=1));result['priority']=args.priority
        write(DEST/('warmstart-'+args.kind+'-check-launch.json'),result)
    else:write(DEST/('warmstart-'+args.kind+'-check-prepared.json'),result)
    print(json.dumps(result))
    if args.submit:start_monitor(args.kind)


def monitor(kind):
    from .scale_efe import command
    from .launch_efe_coverage import infra_retryable
    from ..tracking.reasoning_live import command as raw_command
    marker={'router':'PEAR_MODEL_CHECK ','admission':'PEAR_ADMISSION_CHECK ','reasoning':'PEAR_REASONING_CHECK ','yam':'PEAR_NATIVE_YAM_CHECK '}[kind]
    local=DEST/('warmstart-'+kind+'-check')
    with local.with_suffix('.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        deadline=time.monotonic()+86400
        while time.monotonic()<deadline:
            try:
                if ARM not in read(DEST/'control.json',{}).get('enabled_training_arms',[]):return
                path=DEST/('warmstart-'+kind+'-check-launch.json');launch=read(path)
                state=command(['osmo','workflow','query',launch['name'],'--format-type','json'])
                value={'scheduler_status':state['status'],'phase':kind+'_qualification',
                       'workflow':launch['name'],'updated_utc':datetime.now(timezone.utc).isoformat(),
                       'primary_rl_updates':0}
                if state.get('start_time'):
                    for line in raw_command(['osmo','workflow','logs',launch['name'],'-n','45'],timeout=8).splitlines():
                        if marker in line:
                            try:value['worker']=json.JSONDecoder().raw_decode(line.split(marker,1)[1])[0]
                            except ValueError:pass
                if state['status']=='COMPLETED':
                    value['phase']=kind+'_qualification_completed';write(local.with_suffix('.json'),value);return
                if state['status'].startswith('FAILED'):
                    attempt=launch.get('attempt',0)
                    if not infra_retryable(state)or attempt>=8:
                        value['phase']='needs_source_review';write(local.with_suffix('.json'),value);return
                    source=Path(launch['build']);target=source.parent/(source.name.rsplit('-allocation-r',1)[0]+'-allocation-r'+str(attempt+1))
                    if not(target/'source-manifest.json').exists():
                        target.mkdir(exist_ok=True)
                        for name,digest in read(source/'source-manifest.json').items():
                            if hashlib.sha256((source/name).read_bytes()).hexdigest()!=digest:raise ValueError('Frozen qualification changed')
                            shutil.copyfile(source/name,target/name)
                        workflow=yaml.safe_load((target/'workflow.yaml').read_text())
                        workflow['workflow']['name']='leact-'+target.name
                        (target/'workflow.yaml').write_text(yaml.safe_dump(workflow,sort_keys=False));freeze_sources(target)
                    pool=launch.get('pool')or('groot-h100-01'if kind=='admission'else'groot-h100-02')
                    receipt=submit(target,priority='LOW',pool=pool,nodes=1)
                    write(path,{**launch,**receipt,'build':str(target),'attempt':attempt+1,'priority':'LOW'})
                    value.update(phase='infrastructure_retry_submitted',scheduler_status='PENDING',workflow=receipt['name'])
                write(local.with_suffix('.json'),value)
            except Exception as exc:
                write(local.with_suffix('.json'),{'phase':'observer_retry','error':str(exc)[-500:],
                      'updated_utc':datetime.now(timezone.utc).isoformat(),'primary_rl_updates':0})
            time.sleep(30)


if __name__=='__main__':main()
