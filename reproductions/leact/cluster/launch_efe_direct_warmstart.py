"""Queue the first curated warmstart and its image-aligned base control."""
import argparse
import base64
from datetime import datetime, timezone
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

from .scale_efe import BASE, ROOT, BUILD_ROOT, read, write, submit, command
from .compute_budget import reserved_nodes, node_limit
from .launch_efe_training import freeze_sources
from .launch_efe_leact_warm96 import collect, download, preserve_model
from .launch_efe_native_followups import LANE, collect_arm, verified_native_completion
from .monitor_efe_preflight import remote

DEST = ROOT / 'direct-control-v1/curated-warmstart-v1'
BUILD = BUILD_ROOT / 'efe-direct-warm64-20261101-v4'
BASE_BUILD = BUILD_ROOT / 'efe-direct-base-aligned-20261030-v1'
SNAPSHOT = '63ddcce37704c6fc'
REMOTE = '/mnt/amlfs-03/home/rpunamiya/leact/studies/efe-direct-warm64-20261101-v4'
ARMS = [
    {'id':'curated-warm64','label':'Curated warmstart · 64K training','directory':'direct-control-v1/students/curated-warm64','build':str(BUILD)},
    {'id':'base-aligned','label':'Base Qwen · verified image preprocessing','directory':'direct-control-v1/students/base-aligned','build':str(BASE_BUILD)},
]


def now():return datetime.now(timezone.utc).isoformat()
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def infrastructure_retry(state):
    if state['status'] in {'FAILED_QUEUE_TIMEOUT','FAILED_START_ERROR','FAILED_START_TIMEOUT','FAILED_PREEMPTED'}:
        return True
    reason=' '.join(str(g.get('failure_message','')) for g in state.get('groups',[]))
    return state['status'].startswith('FAILED') and (
        'OSMO Control failure' in reason or
        state.get('cancelled_by')=='groot-idle-job-shutdown' and 'GPU quota enforcement' in reason)


def prepare():
    DEST.mkdir(parents=True, exist_ok=True)
    profile_dir = ROOT / 'direct-control-v1/preparation-profile' / SNAPSHOT
    profile = read(profile_dir / 'result.json')
    profile_source = Path(read(profile_dir / 'launch.json')['build'])
    if profile['decisions'] != 441 or profile['coverage']['65536']['complete_decisions'] != 441:
        raise ValueError('The frozen first curated corpus must fit complete 64K histories')
    original = BUILD_ROOT / 'efe-native-followups-20261030-v1-base-direct'
    for arm in ARMS:
        build = Path(arm['build'])
        if (build / 'source-manifest.json').exists():continue
        build.mkdir(exist_ok=True)
        spec = read(original / 'spec.json')
        spec.update(model_id=arm['id'],agent_label=arm['label'],agent_slug='qwen-'+arm['id'],
            export_role='training_'+arm['id'].replace('-','_'),local_results_directory=arm['directory'],
            image_processor_size=read(profile_dir / 'processor.json')['size'],
            protocol_version='direct-control-v1-qwen-verified-vision',
            protocol_label='Direct-control instruction suffix, verified Qwen3-VL image size, native sampling and grading. Six training-task diagnostic.',
            scope='Six fixed training tasks. Base-aligned isolates the image-processing correction; curated-warm64 adds the frozen cleaned warmstart.',
            comparison_to_development='New curated study; compare to base-aligned. Legacy direct-prompt controls retain different image preprocessing.')
        files = []
        for name in ['efe_native_pilot.py','student.py','pilot.py','efe_budget_gateway.py','export_efe_viewer.py','worker.py','bootstrap.py','method.txt']:
            value = (original / name).read_text()
            if name == 'student.py':
                old="json.dumps({'max_pixels':1048576})"
                if value.count(old)!=1:raise ValueError('Native image kwargs hook changed')
                value=value.replace(old,"json.dumps({'size':spec['image_processor_size']})")
                old="'Legacy max_pixels=1048576 request retained identically across the native base/warmstart/LeAct comparison. A later audit found this override does not enforce the HF training image size in this vLLM version; do not interpret it as a verified pixel cap. Raw observations retained.'"
                value=value.replace(old,"'Explicit Qwen3-VL image size matches the curated HF processor. Compare this condition to base-aligned; legacy controls used different image preprocessing.'")
            # Distinct mounted content avoids OSMO duplicate-volume names for
            # the native and training gateway modules.
            if name.endswith('.py'):value='# Curated native evaluation copy.\n'+value
            local='native-'+name
            (build / local).write_text(value)
            files.append({'localpath':local,'path':LANE+'/'+name})
        workflow = yaml.safe_load((original / 'workflow.yaml').read_text())
        workflow['workflow'].update(name='leact-efe-direct-warm64-v4' if arm['id']=='curated-warm64' else 'leact-efe-direct-base-aligned-v1')
        workflow['workflow']['timeout']={'queue_timeout':'20m','exec_timeout':'36h'}
        workflow['workflow']['resources']['default']['storage']='800Gi'
        task=workflow['workflow']['tasks'][0]
        if arm['id']=='curated-warm64':
            write(build/'evaluation-template.json',spec)
            files.append({'localpath':'evaluation-template.json','path':LANE+'/evaluation-template.json'})
            manifest=read(profile_source/'source-manifest.json')
            bundle=build/'corpus.tar.gz'
            with tarfile.open(bundle,'w:gz') as archive:
                for name,digest in manifest.items():
                    if sha(profile_source/name)!=digest:raise ValueError('Frozen corpus source changed')
                    archive.add(profile_source/name,arcname=name)
            (build/'corpus.b64').write_text(base64.b64encode(bundle.read_bytes()).decode()+'\n')
            files.append({'localpath':'corpus.b64','path':'/tmp/efe-direct-corpus.b64'})
            base=read(BUILD_ROOT/'efe-training-pilot-20261027-v1/spec.json')
            training={k:base[k] for k in ['base_model','base_manifest','base_revision','teacher_model','teacher_concurrency','rationale_max_tokens']}
            training.update(context=65536,lr=1e-4,effective_batch=32,balanced_passes=3.,seed=20261030,
                sampling_policy='coverage-first-balanced-v1',tensor_storage='shared-visual-tensors-v1',
                checkpoint_fractions=[1.],collective_timeout_seconds=900,minimum_suites=2,
                stage='curated-direct64-v2',ready_name='warmstart-ready.json',register_capability=False,
                run_id='curated-production',checkpoint_durability={'restart_interval':8},training_timeout_seconds=24*3600)
            config={'archive_root':REMOTE,'local_results_directory':'direct-control-v1/curated-warmstart-v1',
                'durable_root':REMOTE,'durable_from_start':True,'run_id':'curated-run',
                'final_collection_wait_seconds':1800,
                'recovery_of':'leact-efe-direct-warm64-v3-low1-1',
                'recovery_reason':'Previous node-local attempt ended with OSMO Control failure before a checkpoint. Same frozen 441 decisions and recipe; public backward explanations are regenerated as a separately labeled attempt and then retained on shared storage.',
                'corpus_snapshot':SNAPSHOT,'expected_decisions':441,'expected_teacher_episodes':16,
                'corpus_bundle_sha256':sha(bundle),'corpus_source_manifest':manifest,'training':training,
                'scope':'First frozen cleaned-corpus warmstart, while 72-attempt collection continues. 16 teacher episodes, 15 tasks, 441 eligible decisions before backward-explanation validation. New recipe; not a held-out benchmark score.',
                'gates':['Lossless real-tensor round trip','Exact HF/vLLM multimodal tokens',
                         'Complete-action likelihood within 0.05 nats/token','64K optimizer memory with 10% headroom'],
                'next':'Run the six matched native training-task rollouts; preserve the checkpoint and curated corpus for LeAct.'}
            write(build/'spec.json',config)
            files.append({'localpath':'spec.json','path':'/tmp/efe-direct-warmstart-config.json'})
            names=['efe_direct_warmstart.py','efe_direct_corpus_profile.py','efe_visual_tensor_store.py',
                'efe_training_pilot.py','efe_training_data.py','efe_train_vl.py','efe_leact_vl.py',
                'efe_leact_pipeline.py','efe_budget_gateway.py','efe_training_state.py','efe_durable_training.py']
            for name in names:
                shutil.copyfile(BASE/'cluster'/name,build/name)
                files.append({'localpath':name,'path':'/tmp/'+name})
            shutil.copyfile(BASE/'cluster/efe_h100_runtime.py',build/'efe_cuda_helpers.py')
            files.append({'localpath':'efe_cuda_helpers.py','path':'/tmp/efe_cuda_helpers.py'})
            task.update(command=['python3'],args=['-u','/tmp/efe_direct_warmstart.py'],
                credentials={'leact-efe-inference-20261026':{'NVIDIA_INFERENCE_API_KEY':'api_key'}})
            task['environment']['NCCL_P2P_DISABLE']='1'
        else:
            write(build/'spec.json',spec)
            files.append({'localpath':'spec.json','path':LANE+'/spec.json'})
            task.update(command=['python3'],args=['-u',LANE+'/bootstrap.py'])
        task['files']=files
        if len({sha(build/f['localpath']) for f in files})!=len(files):raise ValueError('Duplicate mounted content')
        (build/'workflow.yaml').write_text(yaml.safe_dump(workflow,sort_keys=False))
        for file in build.glob('*.py'):compile(file.read_text(),str(file),'exec')
        freeze_sources(build)
    write(DEST/'plan.json',{'created_utc':now(),'arms':ARMS,'corpus_snapshot':SNAPSHOT,
        'decisions':441,'teacher_episodes':16,'tasks':15,'suites':2,'training_context':65536,
        'training_steps_if_all_pairs_admitted':83,'node_cap':node_limit(),
        'automatic_next':'64K gates → teacher backward explanations → warmstart → native rollouts. Full collection and later LeAct remain separate stages.'})
    registry_path=ROOT/'training-pilot/additional-runs.json'
    registry=read(registry_path,{'runs':[]})
    entry={'id':'qwen-curated-direct64-v2','name':'Curated direct-control warmstart · durable restart',
        'baseline':'qwen-base-aligned','category':'efe-curated','phase':'Curated Astra backward warmstart',
        'build':str(BUILD),'local_results_directory':'direct-control-v1/curated-warmstart-v1/curated-warm64/production',
        'evaluation_directory':'direct-control-v1/students/curated-warm64','evaluation_partition':'training_evaluation',
        'description':'Same frozen 16-episode, 15-task, 441-decision recipe after infrastructure loss. Public backward explanations regenerated; this is a new attempt from untouched Qwen. LR 1e-4, 64K, batch 32, three exposure budgets with full first coverage. Shared corpus and periodic full-precision Adam/RNG snapshots preserve future progress. Compare to image-aligned base Qwen.',
        'optimizer_steps':83}
    registry['runs']=[r for r in registry['runs'] if r['id']!=entry['id']]+[entry]
    write(registry_path,registry)
    return ARMS


def monitor():
    arms=prepare()
    lock=(DEST/'monitor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    from ..embodied_first_exam.report import publish
    while True:
        try:
            active=command(['osmo','workflow','list','--name','leact','--status','RUNNING','PENDING','WAITING','--format-type','json'])['workflows']
            summaries=[];done=0
            for arm in arms:
                control=DEST/arm['id'];control.mkdir(exist_ok=True)
                (control/'production').mkdir(exist_ok=True)
                build=Path(read(control/'active-build.json',{}).get('build',arm['build']))
                receipt=read(build/'submission.json')
                if not receipt:
                    if reserved_nodes(active)>=node_limit():
                        summaries.append({**arm,'status':'waiting_for_compute_slot','graded':0,'successes':0,'planned':6,'control_steps':0,
                            'progress':{'detail':'Queued for the next available H100 node within the six-node study limit.'}});continue
                    priority=read(control/'active-build.json',{}).get('priority','HIGH')
                    receipt=submit(build,priority=priority);active.append({'name':receipt['name'],'status':'PENDING'})
                workflow=receipt['name']
                state=command(['osmo','workflow','query',workflow,'--format-type','json']);write(control/'scheduler.json',state)
                if arm['id']=='curated-warm64':write(control/'production/scheduler.json',state)
                native_done={}
                if state['status']=='RUNNING':
                    if arm['id']=='curated-warm64':
                        config=read(build/'spec.json')
                        archive=config['archive_root']+'/'+config.get('run_id',workflow)
                        from .launch_efe_native_mirror import ensure as ensure_native_mirror
                        ensure_native_mirror(workflow,archive,control)
                        collect(workflow,archive,control,['pipeline-progress.json','pipeline-complete.json','storage-check.json',
                            'token-parity.json','preflight-passed.json','training-complete.json','portable-ready.json',
                            'durable-training-ready.json','durable-copy-progress.json',
                            'preflight/smoke/metrics.jsonl','preflight/smoke/complete.json','preflight/smoke/scoring-parity.json'])
                        production=archive+'/production/'+config['training'].get('run_id',workflow)
                        collect(workflow,production,control/'production',['progress.json','complete.json','warmstart-ready.json',
                            'corpus-manifest.json','dataset-manifest.json','restart-ready.json','training-lineage.json',
                            'optimizer-resume.json',
                            'train/metrics.jsonl','train/checkpoint-timings.jsonl','train/complete.json','train/configuration.json','train/schedule.json'])
                        if read(control/'portable-ready.json'):
                            preserve_model(workflow,archive,control)
                    native_done=collect_arm(arm,workflow)
                pipeline=read(control/'pipeline-progress.json',{}) if arm['id']=='curated-warm64' else {}
                result=read(control/'pipeline-complete.json',{}) if arm['id']=='curated-warm64' else read(ROOT/arm['directory']/'worker-complete.json',{})
                if arm['id']=='base-aligned' and not result:result=verified_native_completion(arm)
                if result:
                    retained=arm['id']=='base-aligned' or read(control/'durable-training-ready.json',{}).get('all_files_verified')
                    if result.get('success') and retained:
                        done+=1
                        if arm['id']=='curated-warm64' and state['status']=='RUNNING':
                            remote(workflow,"from pathlib import Path\nPath("+repr(archive+'/pipeline-collected')+").touch()\nprint('EFE_CAPTURE {\"ack\":true}',flush=True)")
                    if not result.get('success'):write(control/'experiment-error.json',result)
                elif infrastructure_retry(state) and (
                    arm['id']=='curated-warm64' and (not pipeline or read(build/'spec.json').get('durable_from_start'))
                        and not read(ROOT/arm['directory']/'behavior.json',{}).get('graded') or
                    arm['id']=='base-aligned' and not read(ROOT/arm['directory']/'behavior.json',{}).get('graded')):
                    attempt=read(control/'active-build.json',{}).get('retry_number',0)+1
                    template=Path(read(control/'active-build.json',{}).get('retry_template',arm['build']))
                    retry=Path(str(template)+'-low'+str(attempt));retry.mkdir(exist_ok=True)
                    if not (retry/'source-manifest.json').exists():
                        for name in read(template/'source-manifest.json'):shutil.copyfile(template/name,retry/name)
                        w=yaml.safe_load((retry/'workflow.yaml').read_text());w['workflow']['name']+='-low'+str(attempt);w['workflow']['timeout']['queue_timeout']='12h'
                        (retry/'workflow.yaml').write_text(yaml.safe_dump(w,sort_keys=False));freeze_sources(retry)
                    submit(retry,priority='LOW');write(control/'active-build.json',{'build':str(retry),'retry_number':attempt,'retry_template':str(template)})
                elif state['status'].startswith('FAILED') or state['status']=='COMPLETED':
                    interruption={'scheduler':state['status'],'detail':'Allocation ended without a verified completion receipt. Preserve partial outputs before any scientific retry.'}
                    write(control/'experiment-error.json',interruption)
                    pipeline={'phase':'needs_review','detail':state['status']+': '+interruption['detail']}
                behavior=read(ROOT/arm['directory']/'behavior.json',{})
                progress=read(control/'production/progress.json',{})
                summaries.append({**arm,'workflow':workflow,'status':state['status'],'progress':pipeline,
                    'training_progress':progress,'graded':behavior.get('graded',0),'successes':behavior.get('successes',0),'planned':6,
                    'control_steps':sum(t['control_steps'] for t in behavior.get('trials',[]))})
            write(DEST/'status.json',{'updated_utc':now(),'arms':summaries,'complete':done==len(arms)})
            from ..embodied_first_exam.direct_control_corpus import refresh
            from ..embodied_first_exam.direct_control_viewer import build as viewer
            refresh();viewer();publish()
            if done==len(arms):return
        except Exception as error:write(DEST/'monitor-error.json',{'error':str(error)[-1800:],'updated_utc':now()})
        time.sleep(30)


def start():
    with (DEST/'monitor.log').open('a') as log:
        child=subprocess.Popen([sys.executable,'-u','-m','reproductions.leact.cluster.launch_efe_direct_warmstart','monitor'],
            cwd=BASE.parents[1],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(DEST/'monitor-launch.json',{'pid':child.pid,'updated_utc':now()})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','start','monitor']);args=parser.parse_args()
    if args.action=='monitor':monitor()
    else:
        prepare()
        if args.action=='start':start()
        print(json.dumps(read(DEST/'plan.json')))
