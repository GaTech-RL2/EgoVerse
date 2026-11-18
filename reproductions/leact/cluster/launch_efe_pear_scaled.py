"""Prepare the two full-finetuning PEAR++ arms and their shared upstream runtime."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import yaml

from .scale_efe import BASE, BUILD_ROOT, read, write, submit, command
from .launch_efe_training import freeze_sources
from ..embodied_first_exam.report import ROOT
from .efe_pear_context import DEFAULT_PROMPT_BUDGET, context_policy

DEST=ROOT/'pear-scaled-v1'
REMOTE='/mnt/amlfs-07/shared/rpunamiya/leact/studies/efe-pear-scaled-20261116-v1'
PEAR_REVISION='d604856a1b16fdccd47e22a27d4dd935e3a62014'
EFE_REVISION='b88c450c46de8a96bba83bf9606b66396cccfff5'
RUNTIME_VERSION=6
MOE_PARENT='/mnt/amlfs-07/shared/rpunamiya/leact/studies/efe-coverage-20261105-v1/intermediate/warmstart-step496/model'
TEMPLATE=BUILD_ROOT/'efe-coverage-20261105-v1-warmstart-p7-allocation-retry-24/workflow.yaml'
SUITE_STAGES={
    **{s:'first' for s in ['metaworldplus','libero-max','robocasa','robocasa365','robocasa-gr1']},
    **{s:'broader' for s in ['roboarc','dexjoco','abc-sim','autobio','vlabench','molmospaces','mujoco-playground']},
    # BEHAVIOR is excluded by the study owner because of simulator reliability.
    **{s:'isaac' for s in ['robodojo','robolab','yamlab','labutopia','maniguard','ambench']},
}


def now():return datetime.now(timezone.utc).isoformat()


def update_interface_plan(plan):
    """Carry the requested conditioning into resumed plans without losing results."""
    plan['interface_context']=context_policy()
    plan.setdefault('recipe',{})['prompt_budget']=DEFAULT_PROMPT_BUDGET
    if plan.get('proposal'):
        plan['proposal'].setdefault('interface',{})['language_context']=context_policy()
        plan['proposal'].setdefault('optimizer',{})['prompt_budget_initial']=DEFAULT_PROMPT_BUDGET
    return plan


def require_unpaused():
    control=read(DEST/'control.json',{})
    if control.get('paused'):
        raise RuntimeError('PEAR++ is paused by the user: '+control.get('reason','recipe revision'))


def candidates(inventory,splits):
    allowed=set(splits['train_candidates'])
    protected=set(splits.get('protected_core_all_modes',[]))|set(splits.get('development',[]))
    held_suites=set(splits['held_out_suites']); held_runtimes=set(splits['held_out_runtimes'])
    result=[]
    for row in inventory:
        if row['task'] not in allowed or row['suite'] not in SUITE_STAGES:continue
        if row['suite']=='mujoco-playground' and 'panda' not in row['task']:continue
        if row['task'] in protected or row['suite'] in held_suites or row.get('simulator_runtime') in held_runtimes:
            raise ValueError('Candidate overlaps a protected holdout: '+row['task'])
        result.append({**row,'expansion_stage':SUITE_STAGES[row['suite']],
            'qualification':'pending_native_action_reward_and_scene_audit'})
    return result


def update_budget(unique_tasks):
    """Planning heuristic: preserve 225 LIBERO updates / 40 tasks, not resets."""
    if unique_tasks<=0:raise ValueError('At least one admitted task required')
    return (225*unique_tasks+39)//40


def prepare_plan():
    require_unpaused()
    DEST.mkdir(parents=True,exist_ok=True)
    existing=read(DEST/'plan.json',{})
    if existing.get('proposal',{}).get('id')=='efe-pearplusplus-multitask-v2':
        # Resuming infrastructure must not overwrite the user's revised curriculum.
        existing.update(updated_utc=now(),status='implementation_and_runtime_preparation')
        update_interface_plan(existing)
        write(DEST/'plan.json',existing)
        return existing
    inventory=read(ROOT/'generalization-v2/inventory.json')
    splits=read(ROOT/'generalization-v2/splits.json')
    rows=candidates(inventory,splits)
    membership={'source_revision':EFE_REVISION,'status':'candidates_only_not_training_admission',
        'rows':rows,'suite_counts':dict(sorted(Counter(r['suite'] for r in rows).items())),
        'task_variants':len(rows),'semantic_tasks':len({r['semantic_task'] for r in rows}),
        'held_out_suites':splits['held_out_suites'],'held_out_runtimes':splits['held_out_runtimes'],
        'gr00t':{'available_local_task':'YamSimPickRedCubeEnv','catalog_status':'requires_authenticated_gr00t_sim_checkout',
                 'admitted_tasks':0,'note':'EFE YAMLab is a separate suite; native GR00T task not yet admitted.'}}
    write(DEST/'candidate-membership.json',membership)
    base_metadata=read(DEST/'base-model-metadata.json',{})
    plan={'id':'efe-pearplusplus-two-arms-v1','updated_utc':now(),'status':'runtime_and_environment_preparation',
        'paper':'https://mit-han-lab.github.io/pear-blog/','pear_revision':PEAR_REVISION,'efe_revision':EFE_REVISION,
        'arms':[
            {'id':'moe-warmstart496-pearplusplus','model':'Qwen/Qwen3.6-35B-A3B',
             'initialization':MOE_PARENT,'revision':'efe-coverage-warmstart-intermediate-step496',
             'warmstart_step':496,'parameter_update':'full_finetuning','lora_rank':0,'rl_updates':0},
            {'id':'qwen38-27b-base-pearplusplus','model':'Qwen/Qwen3.8-27B',
             'initialization':'base_checkpoint','revision':base_metadata.get('sha'),
             'parameter_update':'full_finetuning','lora_rank':0,'rl_updates':0}],
        'comparison_note':'Different model architectures and initialization histories; this is not an isolated warmstart ablation.',
        'checkpoint_name_prefix':'rp_jhh_',
        'checkpoint_registry':'/mnt/amlfs-07/shared/rpunamiya/leact/checkpoints/rp_jhh_checkpoint_index.json',
        'recipe':{'retained_groups_per_update':128,'rollouts_per_group':8,'retained_episodes_per_update':1024,
            'optimizer':'AdamW','learning_rate':1e-6,'updates_per_batch':1,'loss_aggregation':'generated_token_mean',
            'score_centering':True,'sampler_top_logprobs':32,'group_filter':'discard_zero_reward_variance',
            'temperature':1.,'top_p':1.,'top_k':-1,'ppo_clipping':False,'kl_penalty':0.,'entropy_bonus':0.,
            'response_budget':160000,'prompt_budget':DEFAULT_PROMPT_BUDGET,'budget_increase_target':256000,
            'budget_increase_gate':'Memory qualification and measured truncation; apply the same stage to both arms.',
            'batch_scaling':'Keep 128x8 initially; scale total updates with admitted unique tasks.',
            'total_update_rule':'ceil(225 * admitted_unique_semantic_tasks / 40)',
            'budget_rule_status':'Planning heuristic based on PEAR LIBERO stage; not an empirically established scaling law.',
            'target_updates':None,'early_evaluation_updates':[1,8,32,128],
            'checkpoint_every_updates':1,'sampling':'Uniform over admitted tasks, uniform legal training resets within task.',
            'coverage_metrics':['attempted_groups_per_task','retained_groups_per_task','all_zero_groups',
                'all_success_groups','native_success','invalid_action_rate','truncation_rate','generated_tokens',
                'seconds_per_action','seconds_per_update']},
        'environments':{k:v for k,v in membership.items() if k!='rows'},
        'admitted_tasks':0,'admitted_suites':0,'primary_rl_updates':0,
        'qualification_gates':['native reset and action contract','native grader and infrastructure-failure distinction',
            'protected scene separation','full-model numerical replay','exact score-centering sampler data',
            'one discarded update and checkpoint reload'],
        'compute':{'max_study_nodes':8,'nodes_per_arm':2,'trainer_nodes_per_arm':1,'rollout_nodes_per_arm':1,
            'gpus_per_node':8,'trainer_fsdp':8,'trainer_sequence_parallel':8,'rollout_tensor_parallel':4,
            'rollout_sessions_per_replica_initial':16,'lmcache_gb_per_replica_initial':96,
            'resource_note':'Smaller host cache/concurrency than the paper to fit 512 GiB nodes; preserve optimizer batch.'},
        'remote_root':REMOTE}
    for arm in plan['arms']:
        arm['run_directory']=REMOTE+'/runs/'+arm['id']
        arm['checkpoint_name_prefix']='rp_jhh_'+('qwen36_moe_warmstart496_pearplusplus'
            if arm.get('warmstart_step') is not None else 'qwen38_27b_base_pearplusplus')
    update_interface_plan(plan)
    write(DEST/'plan.json',plan)
    return plan


def prepare_runtime(index=0):
    require_unpaused()
    build=BUILD_ROOT/(f'efe-pear-scaled-runtime-20261117-v{RUNTIME_VERSION}'+(f'-r{index}' if index else ''))
    if (build/'source-manifest.json').exists():return build
    build.mkdir(parents=True,exist_ok=True)
    source=read(DEST/'sources/pear-source.json')
    shutil.copyfile(BASE/'cluster/efe_pear_scaled_runtime.py',build/'worker.py')
    write(build/'spec.json',{'remote_root':REMOTE,'pear_revision':PEAR_REVISION,
        'source_sha256':source['sha256'],'source_url':source['url'],
        'role':'shared_runtime_preparation','primary_rl_updates':0})
    workflow=yaml.safe_load(TEMPLATE.read_text())
    workflow['workflow']['name']=f'leact-efe-pear-scaled-runtime-v{RUNTIME_VERSION}'+(f'-r{index}' if index else '')
    workflow['workflow']['timeout']={'queue_timeout':'20m','exec_timeout':'6h'}
    workflow['workflow']['resources']['default'].update(gpu=8,cpu=48,memory='512Gi',storage='200Gi')
    task=workflow['workflow']['tasks'][0];task.pop('credentials',None)
    task['args']=['-u','/tmp/efe-pear-scaled/worker.py']
    task['files']=[{'localpath':n,'path':'/tmp/efe-pear-scaled/'+n}
                   for n in ['worker.py','spec.json']]
    (build/'workflow.yaml').write_text(yaml.safe_dump(workflow,sort_keys=False))
    freeze_sources(build)
    return build


def prepare_bridge(index=0):
    require_unpaused()
    build=BUILD_ROOT/('efe-pear-expanded-bridge-20261116-v2'+(f'-r{index}' if index else ''))
    if (build/'source-manifest.json').exists():return build
    build.mkdir(parents=True,exist_ok=True)
    prior=BUILD_ROOT/'efe-pear-native-qualification-20261114-v1'
    template=read(prior/'spec.json')
    row=next(r for r in read(ROOT/'generalization-v2/inventory.json')
             if r['task']=='metaworldplus-door-lock-i00-standard')
    membership=read(ROOT/'pear-v1/training-membership.json')
    scene=membership['training_tasks'][row['task']][0]['scene_sha256']
    spec={k:template[k] for k in ['repository','revision','podman','compose','fixed_action_config']}
    spec.update(tasks=[dict(row,partition='train_candidates')],scene_sha256=scene,
                archive_root=REMOTE+'/bridge-qualification',primary_rl_updates=0)
    spec['fixed_action_config']={**spec['fixed_action_config'],'model':'external-pear-policy'}
    write(build/'spec.json',spec)
    for name in ['efe_pear_bridge.py','efe_pear_context.py','efe_action_loop.py','efe_action_agent.py','efe_native_pilot.py']:
        shutil.copyfile(BASE/'cluster'/name,build/name)
    shutil.copyfile(BASE/'cluster/efe_pear_bridge_qualification.py',build/'worker.py')
    shutil.copyfile(prior/'scene_identity.py',build/'scene_identity.py')
    workflow=yaml.safe_load(TEMPLATE.read_text())
    workflow['workflow']['name']='leact-efe-pear-expanded-bridge-v2'+(f'-r{index}' if index else '')
    # This H100 pool schedules complete eight-GPU nodes, including simulator checks.
    workflow['workflow']['resources']['default'].update(gpu=8,cpu=16,memory='128Gi',storage='300Gi')
    workflow['workflow']['timeout']={'queue_timeout':'20m','exec_timeout':'4h'}
    task=workflow['workflow']['tasks'][0];task.pop('credentials',None);task['privileged']=True
    task['args']=['-u','/tmp/efe-pear-bridge/worker.py']
    task['files']=[{'localpath':p.name,'path':'/tmp/efe-pear-bridge/'+p.name} for p in build.iterdir() if p.is_file()]
    (build/'workflow.yaml').write_text(yaml.safe_dump(workflow,sort_keys=False));freeze_sources(build)
    return build


def prepare_training_bundle():
    """Package the complete future trainer without enabling paused launches."""
    names=['efe_pear_scaled_train.sh','efe_pear_multitask_train.py','efe_pear_curriculum.py',
        'efe_pear_curriculum_state.py','efe_pear_telemetry.py','efe_pear_agent_metrics.py',
        'efe_pear_router.py','efe_pear_moe_kernels.py',
        'efe_pear_checkpoint_archive.py','efe_pear_object_archive.py','efe_pear_episode_archive.py',
        'efe_pear_public_observation.py','efe_pear_observation_check.py','efe_pear_launch_contract.py','efe_pear_action_json.py',
        'efe_pear_reasoning_env.py','efe_pear_reasoning_tool.py','efe_pear_reasoning_scenes.py','efe_pear_child.py','efe_pear_sim_process.py',
        'efe_pear_agent_metrics.yaml','efe_pear_native_env.py','efe_pear_native_episode.py','efe_pear_native_tool.py','efe_pear_bridge.py',
        'efe_pear_remote_env.py','efe_pear_file_rpc.py',
        'efe_pear_context.py','efe_action_loop.py','efe_action_agent.py','efe_native_pilot.py']
    sources={name:BASE/'cluster'/name for name in names}
    sources['scene_identity.py']=BASE/'cluster/efe_pear_scene_audit.py'
    digests={name:hashlib.sha256(path.read_bytes()).hexdigest()for name,path in sources.items()}
    digest=hashlib.sha256(json.dumps(digests,sort_keys=True).encode()).hexdigest()
    build=BUILD_ROOT/('efe-pear-training-adapters-'+digest[:16]);build.mkdir(exist_ok=True)
    if not (build/'source-manifest.json').exists():
        for name,path in sources.items():shutil.copyfile(path,build/name)
        freeze_sources(build)
    record={'bundle':str(build),'content_sha256':digest,'files':digests,'primary_launched':False,
        'qualification':'CPU packaging only; all model and scene admission gates remain required',
        'telemetry':{'directory':'$RUN/telemetry','heartbeat_seconds':10,'driver_metrics':'events.jsonl',
                     'rollout_counters':'rollout-workers/*.json','raw_upstream_metrics':'$RUN/metrics.jsonl',
                     'hosted_collector':'tracking/pear_collect.py','workflow_label':'pear_run_id=<arm id>',
                     'mac_required_for_run_tracking':False}}
    write(DEST/'training-adapters.json',record)
    return record


def launch(kind='runtime'):
    prepare_plan()
    with (DEST/(kind+'-launch.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        queue=read(DEST/(kind+'-queue.json'),{'attempts':[]})
        if not queue['attempts']:
            build=(prepare_runtime if kind=='runtime' else prepare_bridge)()
            require_unpaused()
            pool='groot-h100-02' if kind=='runtime' else 'groot-h100-01'
            receipt=submit(build,priority='HIGH',pool=pool)
            queue['attempts'].append({'workflow':receipt['name'],'build':str(build),'pool':pool,
                                     'priority':'HIGH','submitted_utc':now()})
            write(DEST/(kind+'-queue.json'),queue)
    return queue


def tick(kind='runtime'):
    control=read(DEST/'control.json',{})
    if control.get('paused'):
        state=read(DEST/('status.json' if kind=='runtime' else kind+'-status.json'),{})
        return True,{**state,'phase':'paused_by_user','paused':True,'reason':control.get('reason'),
                     'primary_rl_updates':0}
    from .launch_efe_coverage import infra_retryable
    from .efe_coverage_retry import retry_due
    queue=read(DEST/(kind+'-queue.json'));attempt=queue['attempts'][-1]
    scheduler=command(['osmo','workflow','query',attempt['workflow'],'--format-type','json'])
    write(Path(attempt['build'])/'scheduler.json',scheduler)
    state={'updated_utc':now(),'workflow':attempt['workflow'],'scheduler_status':scheduler['status'],
        'phase':kind+'_'+scheduler['status'].lower(),'primary_rl_updates':0,
        'purpose':'Shared PEAR runtime preparation only' if kind=='runtime' else 'Native bridge check only; no policy inference'}
    terminal=scheduler['status']=='COMPLETED' or scheduler['status'].startswith('FAILED')
    if terminal or scheduler['status']=='RUNNING':
        try:
            result=subprocess.run(['osmo','workflow','logs',attempt['workflow'],'-n','40'],
                                  capture_output=True,text=True,timeout=12)
            logs=result.stdout
        except subprocess.TimeoutExpired as exc:
            logs=exc.stdout or b''
            if isinstance(logs,bytes):logs=logs.decode(errors='replace')
        marker='PEAR_SCALED_RUNTIME ' if kind=='runtime' else 'PEAR_BRIDGE_QUALIFICATION '
        for line in logs.splitlines():
            if marker in line:
                try:state['worker']=json.loads(line.split(marker,1)[1])
                except json.JSONDecodeError:pass
        if state.get('worker',{}).get('success') is True:
            state['phase']=kind+'_ready';terminal=True
        elif scheduler['status']=='COMPLETED':
            state['phase']=kind+'_receipt_missing';terminal=True
    known_probe_repair=(kind=='runtime' and scheduler['status'].startswith('FAILED')
                        and 'efe-pear-scaled-runtime-20261117-v5' in str(attempt['build']))
    if known_probe_repair:
        # v5 imports a class attribute as a module attribute after installation.
        # Keep its frozen payload intact and retry once from corrected v6 source.
        build=prepare_runtime(len(queue['attempts']));require_unpaused()
        pool=attempt.get('pool',scheduler.get('pool','groot-h100-02'))
        receipt=submit(build,priority='LOW',pool=pool)
        queue['attempts'].append({'workflow':receipt['name'],'build':str(build),'pool':pool,
            'priority':'LOW','submitted_utc':now(),'source_repair':'Read MODEL_TYPES from Qwen3_5TrainingKernels'})
        state.update(phase='runtime_source_repair_queued',workflow=receipt['name'],scheduler_status='PENDING')
        terminal=False
    elif scheduler['status'].startswith('FAILED') and infra_retryable(scheduler):
        terminal=False
        due=retry_due(queue,attempt['workflow'])
        state.update(phase=kind+'_retry_backoff',retry_not_before=queue['retry_not_before'])
        if due:
            build=(prepare_runtime if kind=='runtime' else prepare_bridge)(len(queue['attempts']))
            require_unpaused()
            pool=attempt.get('pool','groot-h100-01')
            receipt=submit(build,priority='LOW',pool=pool)
            queue['attempts'].append({'workflow':receipt['name'],'build':str(build),'priority':'LOW','pool':pool,'submitted_utc':now()})
            state.update(phase=kind+'_retry_queued',workflow=receipt['name'],scheduler_status='PENDING')
    write(DEST/(kind+'-queue.json'),queue);write(DEST/('status.json' if kind=='runtime' else kind+'-status.json'),state)
    from ..embodied_first_exam.report import publish
    publish()
    return terminal,state


def monitor(kind='runtime'):
    with (DEST/(kind+'-monitor.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            try:
                done,state=tick(kind); print(json.dumps(state),flush=True)
                if done:return
            except Exception as exc:
                print(json.dumps({'utc':now(),'controller_error':repr(exc)}),flush=True)
            time.sleep(45)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','launch','monitor','status','package'])
    parser.add_argument('--kind',choices=['runtime','bridge'],default='runtime')
    args=parser.parse_args()
    if args.action=='package':print(json.dumps(prepare_training_bundle(),indent=2))
    elif args.action=='prepare':print(json.dumps(prepare_plan(),indent=2))
    elif args.action=='launch':print(json.dumps(launch(args.kind),indent=2))
    elif args.action=='monitor':monitor(args.kind)
    else:print(json.dumps(tick(args.kind)[1],indent=2))
