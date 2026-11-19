"""Compile the full training mixture from retained native validation evidence."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

try:
    from .efe_pear_curriculum_state import CurriculumState
except ImportError:
    from efe_pear_curriculum_state import CurriculumState

CONTROL_SUITES={'abc-sim','dexjoco','libero-max','metaworldplus','robocasa','vlabench'}
RTX_SUITES={'robolab','robodojo','yamlab'}


class PendingAdmission(RuntimeError):pass


def read(path,default=None):
    try:return json.loads(Path(path).read_text())
    except FileNotFoundError:return default


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_receipts(root,queue_file):
    queue=read(root/queue_file,{})
    if not queue or any(not v.get('receipt',{}).get('passed')for v in queue.values()):
        raise PendingAdmission('Native production transport checks are incomplete: '+queue_file)
    values={};origins={}
    for item in queue.values():
        receipt=item['receipt'];path=Path(receipt['path'])
        if sha(path)!=receipt['sha256']:raise ValueError('A native transport receipt changed after download')
        for task,value in read(path).items():
            if task in values:raise ValueError('Duplicate native transport task ownership')
            values[task]=value;origins[task]=receipt['sha256']
    return values,origins


def compile_admission(root):
    root=Path(root);audit=read(root/'source-scene-audit.json')
    if not audit or audit.get('protected_state_errors'):
        raise PendingAdmission('Protected scene audit is incomplete')
    volume=read(root/'native-yam-07-transport-status.json',{})
    latest=read(root/'native-yam-07-transport-launch.json',{})
    worker=volume.get('worker',{})
    if(volume.get('workflow')!=latest.get('name')or volume.get('scheduler_status')!='COMPLETED'or
        worker.get('checked')!=1 or worker.get('passed')!=1):
        raise PendingAdmission('Latest native YAM adapter has not passed on the production shared volume')
    rows=read(root/'proposed-candidate-membership-v2.json')['rows']
    rgb=read(root/'control-observation-audit.json',{}).get('tasks',{})
    membership={};excluded=[];configs=[]
    checks={};origins={}
    for path in(root/'validation-receipts').glob('*/checks.json'):
        for task,value in read(path).items():checks[task]=value;origins[task]=sha(path)
    rtx,rtx_origins=checked_receipts(root,'rtx-transport-queue.json')
    for row in rows:
        task=row['task'];suite=row['suite'];source=audit['tasks'][task]
        if not source['source_scene_check_passed']:
            excluded.append({'task':task,'reason':'reserved_source_scene_overlap'});continue
        if suite in CONTROL_SUITES and task in rgb and not rgb[task]['passed']:
            excluded.append({'task':task,'reason':'blank_or_invalid_native_RGB'});continue
        if suite in CONTROL_SUITES:
            value=checks.get(task,{})
            if (not value.get('success')or value.get('qualification')!='passed'or
                not rgb.get(task,{}).get('passed')or value.get('oracle_success')!=1 or
                value.get('nop_success')!=0):
                raise PendingAdmission('Native control qualification is incomplete: '+task)
            receipt_sha=origins[task]
        elif suite in RTX_SUITES:
            value=rtx.get(task,{})
            if not value.get('native_rtx_api_passed')or not value.get('native_file_transport_passed'):
                raise PendingAdmission('Native RTX qualification is incomplete: '+task)
            verdict=value.get('native_final',{})
            if verdict.get('finalized')is not True or verdict.get('success')is not False:
                raise ValueError('The discarded RTX control is missing its native finalized failure verdict')
            if row.get('oracle')=='full'and value.get('positive_solver_control')is not True:
                raise PendingAdmission('An advertised native oracle has not passed: '+task)
            receipt_sha=rtx_origins[task]
        else:raise ValueError('Unapproved simulator in the candidate mixture: '+suite)
        scene=source['scene']['scene_sha256']
        if value.get('scene_sha256')!=scene:
            raise ValueError('Qualified native task and audited source scene differ')
        if value.get('row',{}).get('task_toml_sha256')!=row['task_toml_sha256']:
            raise ValueError('Qualified native task metadata differs from the frozen inventory')
        family=('broader_bimanual'if suite in {'dexjoco','robodojo','yamlab'}else
                'language_conditioned_control'if suite in {'libero-max','robolab'}else'general_manipulation')
        membership[task]={'qualification':'passed','row':row,'scene_sha256':scene,
            'admission_receipt_sha256':receipt_sha,'action_contract':value['action_contract'],
            'positive_solver_control':value.get('positive_solver_control',value.get('oracle_success')==1),
            'benchmark_resource_enforced':suite in CONTROL_SUITES}
        configs.append({'config_id':task,'family':family,'suite':suite,'task':task,
            'env_class':'efe_pear_remote_env.RemoteNativeEnv','env_args':{'task':task,'seed':row['seeds']['default']},
            'partition':'train','stratum':'full','admitted':True,'already_successful':False,
            'observation_complete':True,'scene_sha256':scene,'admission_receipt_sha256':receipt_sha})
    if {v['row']['suite']for v in membership.values()}!=CONTROL_SUITES|RTX_SUITES:
        raise PendingAdmission('At least one authorized control simulator has no validated training task')
    bank=read(root/'native-yam-bank-readback.json',{})
    if not bank.get('passed')or not bank.get('all_expected_reset_assignments_verified'):
        raise PendingAdmission('Independent native YAM reset bank has no verified readback')
    for receipt in bank.get('receipts',{}).values():
        if sha(receipt['path'])!=receipt['sha256']:raise ValueError('A native YAM bank receipt changed')
    yam,yam_origins=checked_receipts(root,'yam-production-transport-queue.json')
    generated=read(root/'native-yam-generated-resets/generation-report.json')
    if set(yam)!=set(generated['tasks']):raise PendingAdmission('Native YAM production transport coverage is incomplete')
    if generated.get('saved_scene_poses_used_to_generate_training')is not False:
        raise ValueError('Training YAM resets must be independent of saved evaluation scene poses')
    for task,row in generated['tasks'].items():
        check=yam[task]
        if not check.get('success')or not check.get('native_file_transport_passed'):
            raise PendingAdmission('Native YAM transport did not pass: '+task)
        if row['protected_pose_overlap']!=0:raise ValueError('Native YAM reset overlaps a reserved pose')
        path=Path(row['manifest'])
        if sha(path)!=row['manifest_sha256']:raise ValueError('Independent YAM reset bank changed')
        entries=read(path)['entries']
        if len(entries)!=row['generated_resets']:raise ValueError('Independent YAM reset count changed')
        for entry,scene in zip(entries,row['pose_fingerprints'],strict=True):
            configs.append({'config_id':task+'/'+entry['id'],'family':'native_gr00t_yam','suite':'native_gr00t_yam',
                'task':task,'env_class':'efe_pear_remote_env.NativeYamEnv','env_args':{'task':task,'reset_id':entry['id']},
                'partition':'train','stratum':'full','admitted':True,'already_successful':False,
                'observation_complete':True,'scene_sha256':scene,'admission_receipt_sha256':yam_origins[task]})
    path=root/'reasoning-remote-qualified/membership.json';reasoning=read(path)
    if not reasoning or reasoning['qualification']!='passed'or reasoning['train_development_scene_overlap']!=0:
        raise PendingAdmission('Reasoning scene/reward/split qualification is incomplete')
    reasoning_sha=sha(path)
    for key,value in reasoning['scenes'].items():
        if value['partition']!='train':continue
        if (value['qualification']!='passed'or value['positive_control']!=1 or
            value['negative_control']!=0 or not value['stateful_finalization_passed']):
            raise ValueError('A procedural reasoning scene lacks native positive/negative controls')
        family=value['family']
        configs.append({'config_id':key,'family':family,'suite':'procedural_reasoning','task':family,
            'env_class':'efe_pear_reasoning_env.ReasoningEnv',
            'env_args':{k:value[k]for k in ['family','seed','partition']},
            'partition':'train','stratum':'full','admitted':True,'already_successful':False,
            'observation_complete':True,'scene_sha256':value['scene_sha256'],'admission_receipt_sha256':reasoning_sha})
    curriculum={'partition':'train','qualification':'passed','curriculum_stage':'expanded_full','configs':configs,
        'source_disjoint_control_exclusions':excluded,
        'native_yam_scope':'16 independent generated task templates; other native catalog templates remain deferred.',
        'native_yam_deferred':generated.get('not_generated',{}),
        'reasoning_scope':'Eight independently generated primitive-scene templates; no held-out pointing datasets.'}
    state=CurriculumState(curriculum)
    return {'curriculum':curriculum,'native_membership':{'training_tasks':membership,'full_curriculum_qualified':True},
        'counts':{'control_tasks':len(membership),'native_yam_templates':len(yam),
            'native_yam_resets':sum(v['family']=='native_gr00t_yam'for v in configs),
            'reasoning_scenes':sum(v['suite']=='procedural_reasoning'for v in configs),
            'configs':len(state.configs),'excluded_source_or_RGB':len(excluded),
            'family_counts':dict(Counter(v['family']for v in configs))}}


def model_qualification(root):
    root=Path(root)
    driver=read(root/'primary-driver-check-status.json',{})
    command=driver.get('worker',{})
    if(driver.get('scheduler_status')!='COMPLETED'or command.get('passed')is not True
        or command.get('source_check_passed')is not True or command.get('rollout_nodes')!=[6,4]
        or command.get('script_sha256')!=sha(Path(__file__).with_name('efe_pear_scaled_train.sh'))
        or command.get('custom_trainer_config_passed')is not True
        or command.get('driver_entry_sha256')!=sha(Path(__file__).with_name('efe_pear_multitask_train.py'))
        or command.get('multitask_config_sha256')!=sha(Path(__file__).with_name('efe_pear_multitask_config.py'))):
        raise PendingAdmission('Exact primary training command has not passed source and CPU configuration checks')
    next_policy=read(root/'control.json',{}).get('next_sampling_policy','strict-family-v1')
    if next_policy=='task-signal-window-v2'and(
        next_policy not in command.get('sampling_policies',[])
        or command.get('curriculum_state_sha256')!=sha(Path(__file__).with_name('efe_pear_curriculum_state.py'))
        or command.get('signal_curriculum_sha256')!=sha(Path(__file__).with_name('efe_pear_signal_curriculum.py'))):
        raise PendingAdmission('Task curriculum and rolling family balancing have not passed the exact driver check')
    if read(root/'control.json',{}).get('warmup_sampling')and(
        command.get('focus_curriculum_sha256')!=sha(Path(__file__).with_name('efe_pear_curriculum_focus.py'))
        or not command.get('warmup_sampling_checked')):
        raise PendingAdmission('Focused training-task sampling has not passed the exact driver check')
    queue=read(root/'queue-checkpoint-status.json',{})
    queue_result=queue.get('worker',{})
    if(queue.get('scheduler_status')!='COMPLETED'or queue_result.get('passed')is not True
        or queue_result.get('ray_nodes')!=2 or queue_result.get('storage_units')!=2
        or queue_result.get('fresh_process_restore')is not True or queue_result.get('tensors_equal')is not True
        or queue_result.get('head_local_path_invisible_on_other_node')is not True):
        raise PendingAdmission('Two-node rollout-queue save and fresh-process restore are not yet verified')
    numerical=read(root/'warmstart-router-check.json',{}).get('worker',{}).get('result',{})
    update=read(root/'onpolicy-update-evidence.json',{})
    storage=read(root/'checkpoint-storage-status.json',{}).get('worker',{})
    if not numerical.get('engine_check_passed')or not numerical.get('checkpoint_reload_passed'):
        raise PendingAdmission('Full-model numerical and fresh-process reload checks are incomplete')
    if (numerical.get('parameter_update')!='full_finetuning'or numerical.get('lora_rank')!=0 or
        numerical.get('router_policy')!='frozen_expert_selection_router'or
        numerical.get('router_unchanged_after_update_and_reload')is not True):
        raise ValueError('Model qualification does not match the frozen-router full-update recipe')
    if (not update.get('passed')or not update.get('on_policy_score_centering_validated')or
        not update.get('router_unchanged_after_on_policy_update')or not update.get('checkpoint_api_reload_returned')):
        raise PendingAdmission('Actual frozen-router score-centered policy update/reload check is incomplete')
    ranks=update.get('ranks',[])
    if (len(ranks)!=8 or {v['rank']for v in ranks}!=set(range(8))or
        any(not v['routers_unchanged']or not v['trainable_probe_changed']for v in ranks)or
        not math.isfinite(update.get('gradient_norm',float('nan')))or update['gradient_norm']<=0):
        raise ValueError('The real policy update lacks eight valid frozen-router optimizer receipts')
    if not storage.get('passed')or not storage.get('model_readback_verified'):
        raise PendingAdmission('Model-sized private checkpoint backup/readback is incomplete')
    if (not storage.get('multipart_and_restore_passed')or
        storage.get('model_bytes',0)<.9*2*numerical['checked_parameters']):
        raise ValueError('Storage qualification is smaller than the loaded model or lacks a verified restore')
    if storage.get('model_revision')!=numerical['model_revision']:
        raise ValueError('Storage and model qualifications refer to different parent revisions')
    return {**numerical,'primary_training_qualified':True,'on_policy_score_centering_validated':True,
        'on_policy_router_checks':update['ranks'],'discarded_policy_gradient_norm':update['gradient_norm'],
        'durable_on_policy_checkpoint_archived':False,
        'durable_storage_qualification':storage,
        'distributed_queue_checkpoint_qualification':queue_result,
        'driver_command_qualification':command,
        'scope':'Composed independent evidence: full-weight/fresh-process numerical restart, real discarded on-policy optimizer update and API reload, multipart/restore plus model-sized private archive readback, and two-node fresh-process rollout-queue restore. The discarded RL state was not retained.'}
