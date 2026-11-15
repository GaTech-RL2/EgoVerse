"""Advance the curated warmstart to its predeclared first LeAct round."""
import argparse
import fcntl
from pathlib import Path
import shutil
import subprocess
import sys
import time
import yaml

from .scale_efe import BASE, ROOT, BUILD_ROOT, read, write, submit, command
from .compute_budget import reserved_nodes, node_limit
from .launch_efe_training import freeze_sources
from .launch_efe_leact_warm96 import collect, download
from .launch_efe_native_followups import LANE, collect_arm
from .launch_efe_direct_warmstart import infrastructure_retry, now
from .monitor_efe_preflight import remote

DEST=ROOT/'direct-control-v1/curated-leact-v1'
BUILD=BUILD_ROOT/'efe-curated-leact-20261101-v1'
REMOTE='/mnt/amlfs-03/home/rpunamiya/leact/studies/efe-curated-leact-20261101-v1/curated-leact'
EVAL='direct-control-v1/students/curated-leact'


def prepare():
    DEST.mkdir(parents=True,exist_ok=True)
    (ROOT/EVAL).mkdir(parents=True,exist_ok=True)
    plan={'enabled':True,'stage':'waiting_for_verified_curated_warmstart',
        'parent':'qwen-curated-direct64-v2','states':128,'candidates_per_state':8,'top_positive':2,
        'scoring_context':65536,'generation_context':98304,'candidate_tokens':16384,
        'lr':1e-6,'balanced_passes':.5,'effective_batch':32,'replay_fraction':.5,
        'maximum_optimizer_steps':18,'checkpoint_selection':'Final predeclared step; no outcome-based selection.',
        'interpretation':'First conservative LeAct round retains the previous pilot LR/exposure policy. Curated corpus, parent checkpoint and context replace the legacy inputs; larger LeAct budgets remain a separate comparison.',
        'evaluation':'Same six native training tasks and verified image-processing settings as curated warmstart and base-aligned.',
        'priority':'HIGH','capacity_fallback':'LOW','updated_utc':now()}
    write(DEST/'plan.json',plan)
    registry_path=ROOT/'training-pilot/additional-runs.json';registry=read(registry_path,{'runs':[]})
    if not any(r['id']=='qwen-curated-direct64-leact-v1' for r in registry['runs']):
        registry['runs'].append({'id':'qwen-curated-direct64-leact-v1','name':'Curated warmstart + LeAct R1',
            'baseline':'qwen-curated-direct64-v2','category':'efe-curated','phase':'LeAct from curated warmstart',
            'build':str(BUILD),'local_results_directory':'direct-control-v1/curated-leact-v1',
            'evaluation_directory':EVAL,'evaluation_partition':'training_evaluation',
            'description':'Waiting for the verified curated warmstart. First LeAct round: 128 states, eight candidates, top two positive gains, 50% replay; LR 1e-6 and at most 18 updates. No measurements yet.'})
        write(registry_path,registry)
    if (BUILD/'source-manifest.json').exists():return BUILD
    control=ROOT/'direct-control-v1/curated-warmstart-v1/curated-warm64'
    ready=read(control/'durable-training-ready.json',{})
    if not ready.get('all_files_verified'):return None
    source=Path(read(control/'active-build.json')['build'])
    config=read(source/'spec.json');data=read(control/'production/dataset-manifest.json')
    if not data or not ready.get('durable_from_start'):raise ValueError('Expected the verified curated production corpus')
    BUILD.mkdir(exist_ok=True)
    # Mount the exact same native evaluator as the curated warmstart.
    workflow=yaml.safe_load((source/'workflow.yaml').read_text());task=workflow['workflow']['tasks'][0]
    files=[]
    for entry in task['files']:
        if entry['localpath'].startswith('native-'):
            shutil.copyfile(source/entry['localpath'],BUILD/entry['localpath']);files.append(entry)
    evaluation=read(source/'evaluation-template.json')
    evaluation.update(model_id='curated-leact',agent_label='Curated warmstart + LeAct R1',
        agent_slug='qwen-curated-leact',export_role='training_curated_leact',local_results_directory=EVAL,
        scope='First curated LeAct round on the same six training-task diagnostic scenes; compare to curated warmstart and image-aligned base. Not a held-out score.')
    write(BUILD/'evaluation-template.json',evaluation)
    files.append({'localpath':'evaluation-template.json','path':LANE+'/evaluation-template.json'})
    previous=read(BUILD_ROOT/'efe-leact-warm96-20261029-v1/spec.json')
    spec={k:previous[k] for k in ['lr','balanced_passes','effective_batch','leact_decisions','candidate_tokens']}
    spec.update(base_model=ready['checkpoint'],base_manifest=ready['manifest'],base_revision=ready['revision'],
        parent_corpus=ready['corpus'],tokenizer_base=config['training']['base_model'],
        archive_root=str(Path(REMOTE).parent),run_id='curated-leact',context=65536,
        generation_context=98304,generation_max_sequences=1,verify_generation_tokens=True,
        seed=20261101,stage='curated-leact-r1',ready_name='leact-ready.json',checkpoint_fractions=[1.],
        tensor_storage='shared-visual-tensors-v1',checkpoint_durability={'restart_interval':4},
        resume_completed_candidates=True,collective_timeout_seconds=900,training_timeout_seconds=24*3600,
        collection_wait_seconds=0,register_capability=False,scene_audit_sha256=data['scene_audit_sha256'],
        local_results_directory='direct-control-v1/curated-leact-v1',local_evaluation_directory=EVAL)
    write(BUILD/'spec.json',spec);files.append({'localpath':'spec.json','path':'/tmp/efe-training-spec.json'})
    names=['efe_curated_leact_pipeline.py','efe_leact_vl.py','efe_training_pilot.py','efe_training_data.py',
        'efe_train_vl.py','efe_training_state.py','efe_durable_training.py','efe_visual_tensor_store.py',
        'efe_direct_warmstart.py','efe_leact_pipeline.py','efe_budget_gateway.py']
    for name in names:
        shutil.copyfile(BASE/'cluster'/name,BUILD/name);files.append({'localpath':name,'path':'/tmp/'+name})
    shutil.copyfile(BASE/'cluster/efe_h100_runtime.py',BUILD/'efe_cuda_helpers.py')
    files.append({'localpath':'efe_cuda_helpers.py','path':'/tmp/efe_cuda_helpers.py'})
    workflow['workflow']['name']='leact-efe-curated-leact-r1'
    workflow['workflow']['timeout']={'queue_timeout':'20m','exec_timeout':'36h'}
    task.update(command=['python3'],args=['-u','/tmp/efe_curated_leact_pipeline.py'],files=files)
    task.pop('credentials',None)
    (BUILD/'workflow.yaml').write_text(yaml.safe_dump(workflow,sort_keys=False))
    for path in BUILD.glob('*.py'):compile(path.read_text(),str(path),'exec')
    freeze_sources(BUILD)
    registry_path=ROOT/'training-pilot/additional-runs.json';registry=read(registry_path,{'runs':[]})
    entry={'id':'qwen-curated-direct64-leact-v1','name':'Curated warmstart + LeAct R1',
        'baseline':'qwen-curated-direct64-v2','category':'efe-curated','phase':'LeAct from curated warmstart',
        'build':str(BUILD),'local_results_directory':'direct-control-v1/curated-leact-v1',
        'evaluation_directory':EVAL,'evaluation_partition':'training_evaluation',
        'description':'128 fixed curated training states; eight student backward candidates, top two positive full-action likelihood gains; 50% replay. First conservative round: LR 1e-6, half-pass exposure budget, at most 18 updates. Same six native training-task evaluations.'}
    registry['runs']=[r for r in registry['runs'] if r['id']!=entry['id']]+[entry];write(registry_path,registry)
    return BUILD


def monitor():
    prepare();lock=(DEST/'monitor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    from ..embodied_first_exam.report import publish
    while True:
        try:
            build=prepare()
            if build is None:
                write(DEST/'queue-status.json',{'phase':'waiting_for_verified_curated_warmstart','updated_utc':now()})
                time.sleep(30);continue
            active_build=read(DEST/'active-build.json',{})
            build=Path(active_build.get('build',str(build)))
            remote_root=active_build.get('remote_root',REMOTE)
            receipt=read(build/'submission.json')
            if not receipt:
                active=command(['osmo','workflow','list','--name','leact','--status','RUNNING','PENDING','WAITING','--format-type','json'])['workflows']
                if reserved_nodes(active)>=node_limit():time.sleep(30);continue
                receipt=submit(build,priority='HIGH')
            workflow=receipt['name'];state=command(['osmo','workflow','query',workflow,'--format-type','json']);write(DEST/'scheduler.json',state)
            if state['status']=='RUNNING':
                from .launch_efe_native_mirror import ensure as ensure_native_mirror
                try:
                    ensure_native_mirror(workflow,remote_root,DEST)
                except Exception as error:
                    write(DEST/'mirror-start-error.json',{'updated_utc':now(),'workflow':workflow,
                        'error':str(error)[-1800:],'local_collection_continues':True})
                collect(workflow,remote_root,DEST,['pipeline-progress.json','pipeline-complete.json','progress.json','complete.json',
                    'selection.json','membership.json','dataset-manifest.json','restart-ready.json','leact-ready.json',
                    'portable-ready.json','durable-training-ready.json','train/metrics.jsonl','train/complete.json',
                    'train/configuration.json','train/schedule.json','train/checkpoint-timings.jsonl','smoke/complete.json',
                    'train.log','leact-pipeline.log','training-lineage.json','optimizer-resume.json'])
                collect_arm({'directory':EVAL},workflow)
                if read(DEST/'selection.json') and not (DEST/'candidates-collected.json').exists():
                    download(workflow,remote_root+'/candidates',DEST,timeout=600)
                    write(DEST/'candidates-collected.json',{'workflow':workflow,'updated_utc':now()})
                complete=read(DEST/'pipeline-complete.json',{})
                if complete.get('success'):
                    remote(workflow,"from pathlib import Path\nPath("+repr(remote_root+'/pipeline-collected')+").touch()\nprint('EFE_CAPTURE {\"ack\":true}',flush=True)")
                    write(DEST/'queue-status.json',{'workflow':workflow,'status':'COMPLETED',
                        'experiment_complete':True,'scheduler_status_at_collection':state['status'],'updated_utc':now()})
                    publish();return
            elif infrastructure_retry(state) and read(DEST/'pipeline-progress.json',{}).get('phase')!='native_evaluation':
                number=read(DEST/'active-build.json',{}).get('retry_number',0)+1
                retry=Path(str(build)+'-low'+str(number));retry.mkdir(exist_ok=True)
                if not (retry/'source-manifest.json').exists():
                    for name in read(build/'source-manifest.json'):shutil.copyfile(build/name,retry/name)
                    w=yaml.safe_load((retry/'workflow.yaml').read_text());w['workflow']['name']+='-low'+str(number);w['workflow']['timeout']['queue_timeout']='12h'
                    (retry/'workflow.yaml').write_text(yaml.safe_dump(w,sort_keys=False));freeze_sources(retry)
                submit(retry,priority='LOW');write(DEST/'active-build.json',{'build':str(retry),'retry_number':number,'remote_root':remote_root})
            write(DEST/'queue-status.json',{'workflow':workflow,'status':state['status'],'updated_utc':now()});publish()
        except Exception as error:write(DEST/'monitor-error.json',{'error':str(error)[-1800:],'updated_utc':now()})
        time.sleep(30)


def start():
    prepare()
    with (DEST/'monitor.log').open('a') as log:
        p=subprocess.Popen([sys.executable,'-u','-m','reproductions.leact.cluster.launch_efe_curated_leact','monitor'],
            cwd=BASE.parents[1],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(DEST/'monitor-launch.json',{'pid':p.pid,'updated_utc':now()})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','start','monitor']);args=parser.parse_args()
    if args.action=='monitor':monitor()
    elif args.action=='start':start()
    else:prepare()
