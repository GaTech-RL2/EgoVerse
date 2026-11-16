"""Dependency-gated launches for the expanded corpus, warmstart, and LeAct."""
import json
from pathlib import Path
import shutil

import yaml

from .scale_efe import BASE,ROOT,BUILD_ROOT,read,write,submit,command
from .compute_budget import reserved_nodes,node_limit
from .launch_efe_training import freeze_sources
from .efe_coverage_retry import retry_due,submission_priority,cuda_unavailable_before_training

LANE='/tmp/efe-coverage-training'
PAYLOAD_VERSION=7


def retry_build(source,number):
    """An allocation retry preserves all model/data/optimizer code and inputs."""
    source=Path(source)
    spec=read(source/'spec.json')
    run_id=spec.get('run_id',spec['stage'])
    folder=source.parent/('efe-coverage-20261105-v1-'+run_id+f'-p{PAYLOAD_VERSION}-allocation-retry-{number}')
    if (folder/'source-manifest.json').exists():return folder
    folder.mkdir(exist_ok=True)
    for name in read(source/'source-manifest.json'):
        shutil.copyfile(source/name,folder/name)
    workflow=yaml.safe_load((folder/'workflow.yaml').read_text())
    # A short deterministic name avoids growing the retry ancestry in OSMO IDs.
    workflow['workflow']['name']='leact-efe-coverage-'+run_id+f'-v1-p{PAYLOAD_VERSION}-r{number}'
    (folder/'workflow.yaml').write_text(yaml.safe_dump(workflow,sort_keys=False))
    freeze_sources(folder)
    return folder


def build(stage,plan,parent=None,retry=0,*,arm=None):
    from .launch_efe_coverage import DEST,DURABLE,TEMPLATE
    suffix=f'-r{retry}' if retry else ''
    run_id=arm['id'] if arm else stage
    folder=BUILD_ROOT/('efe-coverage-20261105-v1-'+run_id+f'-p{PAYLOAD_VERSION}'+suffix)
    if (folder/'source-manifest.json').exists():return folder
    folder.mkdir(exist_ok=True)
    mapping={
        'efe_coverage_train.py':'pipeline.py','efe_coverage_admission.py':'efe_coverage_admission.py',
        'efe_qwen36_leact.py':'leact.py','efe_train_qwen36_lora.py':'trainer.py',
        'efe_qwen36_restart.py':'efe_qwen36_restart.py','efe_qwen36_restart_probe.py':'probe.py',
        'efe_qwen36_export_recovery.py':'verify.py','efe_merge_qwen36_warm.py':'merge.py',
        'efe_qwen36_adapter_io.py':'efe_qwen36_adapter_io.py','efe_training_state.py':'efe_training_state.py',
        'efe_training_data.py':'efe_training_data.py','efe_budget_gateway.py':'efe_budget_gateway.py',
        'efe_training_pilot.py':'efe_training_pilot.py','efe_leact_vl.py':'efe_leact_vl.py',
        'efe_action_loop.py':'efe_action_loop.py','efe_visual_tensor_store.py':'efe_visual_tensor_store.py',
        'efe_model_cache.py':'efe_model_cache.py','efe_qwen36_health.py':'efe_qwen36_health.py'}
    for source,dest in mapping.items():shutil.copyfile(BASE/'cluster'/source,folder/dest)
    shutil.copyfile(DEST/'plan.json',folder/'coverage-plan.json')
    spec={'stage':stage,'coverage_root':DURABLE,'archive_root':DURABLE+'/'+stage,
        'model':'Qwen/Qwen3.6-35B-A3B','base_model':DURABLE+'/base-model',
        'base_revision':plan['warmstart']['base_revision'],'context':65536,
        'effective_batch':32,'lr':1e-4,'seed':20261105,'lora_rank':32,'lora_alpha':64,
        'collective_timeout_seconds':1800,'tensor_storage':'shared-visual-tensors-v1',
        'node_local_model_cache':True,
        'require_empty_shard_restart_probe':True,
        'require_full_model_replay':True,'validate_each_restart_probe':True,
        'restart_keep_recent':3,'loss_jump_guard':True,
        'nccl_collective_workaround':{'NCCL_ALGO':'Ring','NCCL_PROTO':'Simple'},
        'balanced_passes':3 if stage=='warmstart' else 1,
        'sampling_policy':'coverage-first-balanced-v1' if stage=='warmstart' else 'balanced-replacement-v1',
        'checkpoint_fractions':[.25,.5,1.],'ready_name':'adapter-ready.json',
        'export_ready_name':'warmstart-ready.json' if stage=='warmstart' else 'leact-ready.json',
        'checkpoint_revision_prefix':'efe-qwen36-coverage-v1-'+stage,
        'teacher_model':'azure/openai/gpt-6-astra','rationale_max_tokens':8192,
        'prefetch_explanations_before_corpus_ready':stage=='warmstart',
        'local_results_directory':'fixed-action-v1/coverage-scale-v1/'+stage,
        'scope':'Expanded verified training corpus. Training budget is determined by admitted states, with exact FP32 adapter/Adam/RNG restart. No held-out data enters training.'}
    if stage=='warmstart' and (DEST/'warmstart-replay-reference.json').exists():
        import hashlib
        spec['reference_training_metrics']={'path':DURABLE+'/warmstart/reference-training-metrics.json',
            'sha256':hashlib.sha256((DEST/'warmstart-replay-reference.json').read_bytes()).hexdigest()}
    if stage=='leact-r1':
        ready=parent['ready'];data=parent['dataset'];admission=parent['admission']
        spec.update(base_model=ready['checkpoint'],base_revision=ready['revision'],
            parent_checkpoint=ready['checkpoint'],parent_manifest=ready['manifest'],parent_revision=ready['revision'],
            parent_corpus=DURABLE+'/warmstart',parent_dataset_sha256=data['sha256'],
            decisions_sha256=admission['decisions_sha256'],expected_teacher_states=admission['decisions'],
            teacher_episodes=admission['successful_episodes'],max_balanced_states=2048,
            candidate_tokens=4096,candidates_per_state=8,top_positive=2,replay_fraction=.5)
    if arm:
        if stage!='leact-r1':raise ValueError('A warmstart branch must use the LeAct pipeline')
        spec.update(run_id=run_id,archive_root=arm['archive_root'],
            local_results_directory=arm['local_results_directory'],
            checkpoint_revision_prefix='efe-qwen36-coverage-v1-'+run_id,
            reuse_parent_visual_blobs=True,parent_step=arm['parent_step'],scope=arm['scope'])
    write(folder/'spec.json',spec)
    bootstrap='''from pathlib import Path
import json,os,re,subprocess,sys
lane=Path('/tmp/efe-coverage-training');spec=json.loads((lane/'spec.json').read_text())
version=re.search(r'([0-9]{3}[.][0-9]+(?:[.][0-9]+)?)',Path('/proc/driver/nvidia/version').read_text()).group(1)
libs=[Path('/usr/lib/x86_64-linux-gnu')/(name+'.so.'+version) for name in ['libcuda','libnvidia-ptxjitcompiler','libnvidia-nvvm']]
if not all(p.is_file() for p in libs):raise RuntimeError('Matching host CUDA libraries required')
env={**os.environ,'LD_PRELOAD':':'.join(map(str,libs)),'LD_LIBRARY_PATH':'/usr/lib/x86_64-linux-gnu:'+os.environ.get('LD_LIBRARY_PATH',''),'NCCL_P2P_DISABLE':'1','OMP_NUM_THREADS':'2'}
if spec['stage']=='restart-probe':
 environment=lane/'probe-env'
 subprocess.run([sys.executable,'-m','venv','--system-site-packages',str(environment)],check=True)
 python=str(environment/'bin/python')
 subprocess.run([python,'-m','pip','install','--disable-pip-version-check','transformers==5.11.0','peft==0.21.0','accelerate==1.11.0'],env=env,check=True)
 root=Path(spec['archive_root'])/os.environ['EFE_WORKFLOW_ID'];root.mkdir(parents=True,exist_ok=True)
 result=subprocess.run([python,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',str(lane/'probe.py'),'--root',str(root)],env=env)
 if (root/'result.json').exists():
  data=json.loads((root/'result.json').read_text());Path(spec['archive_root']+'/result.json').write_text(json.dumps(data)+'\\n')
 sys.exit(result.returncode)
os.execve(sys.executable,[sys.executable,'-u',str(lane/'pipeline.py'),'--spec',str(lane/'spec.json')],env)
'''
    (folder/'bootstrap.py').write_text(bootstrap)
    workflow=yaml.safe_load((TEMPLATE/'workflow.yaml').read_text())
    workflow['workflow']['name']='leact-efe-coverage-'+run_id+f'-v1-p{PAYLOAD_VERSION}'+suffix
    workflow['workflow']['resources']['default'].update(cpu=48,memory='512Gi',storage='800Gi')
    workflow['workflow']['timeout']={'queue_timeout':'20m','exec_timeout':'30m' if stage=='restart-probe' else '72h'}
    task=workflow['workflow']['tasks'][0]
    task.update(command=['python3'],args=['-u',LANE+'/bootstrap.py'],privileged=False)
    if stage!='warmstart':task.pop('credentials',None)
    task['files']=[{'localpath':name,'path':LANE+'/'+name} for name in ['spec.json','bootstrap.py','coverage-plan.json',*mapping.values()]]
    (folder/'workflow.yaml').write_text(yaml.safe_dump(workflow,sort_keys=False))
    for path in folder.glob('*.py'):compile(path.read_text(),str(path),'exec')
    freeze_sources(folder)
    if stage!='restart-probe':
        path=ROOT/'training-pilot/additional-runs.json';registry=read(path,{'runs':[]})
        ident='efe-qwen36-coverage-'+run_id
        if not any(r['id']==ident for r in registry['runs']):
            registry['runs'].append({'id':ident,'name':arm['label'] if arm else 'Qwen3.6 expanded coverage · '+stage,
                'category':'efe-warmstart' if stage=='warmstart' else 'efe-leact',
                'local_results_directory':spec['local_results_directory'],
                'evaluation_partition':'coverage_panel','baseline':arm['baseline'] if arm else 'efe-qwen36-coverage-base' if stage=='warmstart' else 'efe-qwen36-coverage-warmstart',
                'phase':'256-episode coverage expansion','build':str(folder),
                'description':spec['scope']})
            write(path,registry)
        else:
            arm=next(r for r in registry['runs'] if r['id']==ident)
            if arm.get('build')!=str(folder):
                arm.setdefault('prior_prepared_builds',[]).append(arm['build'])
                arm['build']=str(folder)
                write(path,registry)
    return folder


def tick(plan,evidence,collection_gate,active,preparation_ready=False):
    from .launch_efe_coverage import DEST,now,infra_retryable
    path=DEST/'training-queue.json'
    queue=read(path,{'stages':{}})
    stages=queue['stages']
    available=node_limit()-reserved_nodes(active)
    remote=evidence.get('training',{})
    complete_warm=remote.get('warmstart',{}).get('complete',{}).get('success')
    branches={r['id']:r for r in read(DEST/'branch-runs.json',{}).get('runs',[])}
    jobs=[]
    for stage in ['restart-probe','warmstart','leact-r1',*branches]:
        data=remote.get(stage,{})
        stage_done = bool(data.get('success')) if stage=='restart-probe' else bool(data.get('complete',{}).get('success'))
        record=stages.setdefault(stage,{'attempts':[]})
        quality_hold=read(DEST/stage/'quality-hold.json',{})
        if quality_hold.get('active'):
            record.update(status='QUALITY_HOLD',retry=False)
            jobs.append({'stage':stage,'status':'QUALITY_HOLD',
                'workflow':record['attempts'][-1]['workflow'] if record['attempts'] else None,
                'progress':data.get('progress',{}),'complete':{},'quality_hold':quality_hold})
            continue
        if stage_done:
            record['status']='COMPLETED'
        elif record['attempts']:
            attempt=record['attempts'][-1]
            state=command(['osmo','workflow','query',attempt['workflow'],'--format-type','json'])
            write(Path(attempt['build'])/'scheduler.json',state)
            if stage!='restart-probe':write(DEST/stage/'scheduler.json',state)
            record['status']=state['status']
            if state['status'].startswith('FAILED'):
                upgrade=branches.get(stage,{})
                reviewed_upgrade=(state['status']=='FAILED_CANCELED'
                    and upgrade.get('execution_upgrade_from_workflow')==attempt['workflow']
                    and upgrade.get('selection_execution_build'))
                cuda_startup=cuda_unavailable_before_training(state,data)
                if cuda_startup:
                    review={'workflow':attempt['workflow'],'utc':now(),
                        'classification':'CUDA device unavailable at initial set_device before restart-probe computation or training',
                        'committed_restart_step':data.get('committed_restart_step'),
                        'recipe_changed':False,'error':data['error']['error']}
                    write(Path(attempt['build'])/'cuda-startup-review.json',review)
                    record['infrastructure_review']=review
                if reviewed_upgrade:
                    record.update(retry=True,status='EXECUTION_UPGRADE')
                elif infra_retryable(state) or cuda_startup:
                    record['retry']=retry_due(record,attempt['workflow'])
                    if not record['retry']:record['status']='RETRY_BACKOFF'
                else:record.update(retry=False,status='NEEDS_REVIEW')
            elif stage=='warmstart' and state['status']=='COMPLETED' and not data.get('complete',{}).get('success'):
                record['status']='WAITING_FOR_CORPUS' if data.get('waiting_for_corpus') else 'WAITING_FOR_RECEIPT'
                record['retry']=bool(data.get('waiting_for_corpus') and collection_gate)
        if not stage_done and (not record['attempts'] or record.get('retry')):
            dependency = stage=='restart-probe' or (stage=='warmstart' and (collection_gate or preparation_ready) and remote.get('restart-probe',{}).get('success')) or (stage=='leact-r1' and complete_warm) or (stage in branches and branches[stage].get('authorized') is True)
            if dependency and available>0:
                parent=None
                if stage=='leact-r1':
                    parent={**remote['warmstart'],'admission':evidence['admission']}
                upgrade=branches.get(stage,{}).get('selection_execution_build')
                applied=any(a['build']==upgrade for a in record['attempts']) if upgrade else False
                if upgrade and not applied:folder=Path(upgrade)
                elif record['attempts']:folder=retry_build(record['attempts'][-1]['build'],len(record['attempts']))
                elif stage in branches:folder=Path(branches[stage]['build'])
                else:folder=build(stage,plan,parent)
                priority=submission_priority(record)
                if upgrade and not applied:priority=branches[stage].get('execution_upgrade_priority',priority)
                receipt=submit(folder,priority=priority)
                record['attempts'].append({'workflow':receipt['name'],'build':str(folder),'submitted_utc':now(),'priority':priority})
                record.update(status='PENDING',retry=False)
                available-=1
        if not record['attempts']:
            record['status']='WAITING_FOR_CORPUS' if stage=='warmstart' else 'WAITING_FOR_WARMSTART' if stage=='leact-r1' else 'WAITING_FOR_NODE'
        jobs.append({'stage':stage,'status':record['status'],'workflow':record['attempts'][-1]['workflow'] if record['attempts'] else None,
                     'progress':data.get('progress',{}),'complete':data.get('complete',{})})
    write(path,queue)
    summary='Public explanation preparation can overlap collection after 128 candidate successes and the diversity audit. Optimizer training still requires the full verified corpus: three-pass-equivalent warmstart → one LeAct round → matched native evaluations. GPU restart validation gates training.'
    status={'updated_utc':now(),'jobs':jobs,'summary':summary}
    write(DEST/'training-status.json',status)
    return status
