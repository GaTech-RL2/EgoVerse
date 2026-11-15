"""Advance the verified constrained Qwen warmstart through LeAct and robot eval."""
import argparse
from datetime import datetime,timezone
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import yaml

from .scale_efe import BASE,ROOT,BUILD_ROOT,read,write,submit,command
from .launch_efe_training import freeze_sources,recovery_reader
from .launch_efe_leact_warm96 import collect
from .launch_efe_native_followups import collect_arm,verified_native_completion
from .launch_efe_student_qualification import retry_build
from .launch_efe_qwen36_warm import publish
from .monitor_efe_preflight import remote

DEST=ROOT/'fixed-action-v1/leact-lora-v1'
BUILD=BUILD_ROOT/'efe-qwen36-constrained-leact-20261105-v1'
ARCHIVE='/mnt/amlfs-01/home/rpunamiya/leact/studies/efe-qwen36-constrained-leact-20261105-v1'
LANE='/tmp/efe-qwen36-leact'
EVAL='fixed-action-v1/leact-lora-v1/evaluation'


def now():return datetime.now(timezone.utc).isoformat()


def prepare():
    if (BUILD/'source-manifest.json').exists():return BUILD
    parent=ROOT/'fixed-action-v1/warmstart-lora-v2'
    if not read(parent/'complete.json',{}).get('success') or not read(parent/'evaluation/verified-completion.json',{}).get('success'):
        raise ValueError('Finish and verify the warmstart comparison before advancing')
    ready=read(parent/'warmstart-ready.json');old=read(BUILD_ROOT/'efe-qwen36-constrained-warm-20261105-v6/spec.json')
    DEST.mkdir(parents=True,exist_ok=True);BUILD.mkdir(exist_ok=True)
    spec={'archive_root':ARCHIVE,'local_results_directory':'fixed-action-v1/leact-lora-v1',
        'model':old['model'],'base_model':'/tmp/efe-qwen36-leact-base/model','base_revision':ready['revision'],
        'parent_checkpoint':ready['checkpoint'],'parent_manifest':ready['manifest'],'parent_revision':ready['revision'],
        'parent_corpus':old['archive_root'],'parent_dataset_sha256':old['prepared_dataset']['manifest_sha256'],
        'decisions_sha256':old['decisions_sha256'],'context':65536,'candidate_tokens':4096,
        'candidates_per_state':8,'top_positive':2,'seed':20261105,'lr':1e-4,'effective_batch':32,
        'balanced_passes':1.0,'fixed_optimizer_steps':32,'expected_optimizer_steps':32,
        'sampling_policy':'balanced-replacement-v1','replay_fraction':.5,'checkpoint_fractions':[.25,.5,1.],
        'lora_rank':32,'lora_alpha':64,'collective_timeout_seconds':900,'ready_name':'adapter-ready.json',
        'export_ready_name':'leact-ready.json','checkpoint_revision_prefix':'efe-qwen36-constrained-leact-lora-v1',
        'scope':'One LeAct pilot on the same 43 verified teacher states, initialized from the exported step-32 warmstart. Eight student backward explanations; retain at most two with positive complete-command likelihood gain. Free-text summary cannot determine admission. Train 32 further LoRA updates, 50% teacher warmstart replay, then the same four native tasks. No held-out score or extra-optimization control is claimed.'}
    write(BUILD/'spec.json',spec)
    mapping={'efe_qwen36_leact.py':'pipeline.py','efe_train_qwen36_lora.py':'trainer.py',
        'efe_qwen36_export_recovery.py':'verify.py','efe_merge_qwen36_warm.py':'merge.py',
        'efe_qwen36_adapter_io.py':'efe_qwen36_adapter_io.py','efe_training_data.py':'efe_training_data.py',
        'efe_budget_gateway.py':'efe_budget_gateway.py','efe_leact_vl.py':'efe_leact_vl.py'}
    for source,destination in mapping.items():shutil.copyfile(BASE/'cluster'/source,BUILD/destination)
    bootstrap='''from pathlib import Path
import json,os,re,sys
lane=Path('/tmp/efe-qwen36-leact');spec=json.loads((lane/'spec.json').read_text());root=Path(spec['archive_root']);root.mkdir(parents=True,exist_ok=True)
try:
 version=re.search(r'([0-9]{3}[.][0-9]+(?:[.][0-9]+)?)',Path('/proc/driver/nvidia/version').read_text()).group(1)
 libs=[Path('/usr/lib/x86_64-linux-gnu')/(name+'.so.'+version) for name in ['libcuda','libnvidia-ptxjitcompiler','libnvidia-nvvm']]
 assert all(p.is_file() for p in libs)
 env={**os.environ,'LD_PRELOAD':':'.join(map(str,libs)),'LD_LIBRARY_PATH':'/usr/lib/x86_64-linux-gnu:'+os.environ.get('LD_LIBRARY_PATH','')}
 os.execve(sys.executable,[sys.executable,'-u',str(lane/'pipeline.py'),'--spec',str(lane/'spec.json')],env)
except Exception as error:
 (root/'bootstrap-error.json').write_text(json.dumps({'error':str(error)}));raise
'''
    (BUILD/'bootstrap.py').write_text(bootstrap)
    workflow=yaml.safe_load((BUILD_ROOT/'efe-qwen36-constrained-warm-20261105-v6/workflow.yaml').read_text())
    workflow['workflow']['name']='leact-efe-qwen36-constrained-leact-v1'
    workflow['workflow']['timeout']={'queue_timeout':'20m','exec_timeout':'10h'}
    task=workflow['workflow']['tasks'][0];task.update(command=['python3'],args=['-u',LANE+'/bootstrap.py'])
    task.pop('credentials',None)
    task['files']=[{'localpath':n,'path':LANE+'/'+n} for n in ['bootstrap.py','spec.json',*mapping.values()]]
    (BUILD/'workflow.yaml').write_text(yaml.safe_dump(workflow,sort_keys=False))
    for path in BUILD.glob('*.py'):compile(path.read_text(),str(path),'exec')
    freeze_sources(BUILD)
    write(DEST/'plan.json',{'updated_utc':now(),'build':str(BUILD),'spec':spec,'status':'prepared','priority':'HIGH',
        'capacity_fallback':'LOW','maximum_concurrent_nodes':2,'parent_result':{'successes':1,'graded':4},
        'sequence':['Verify warmstart weights','43 states × 8 student explanations','Positive executable-command gain, top 2','HF/vLLM scoring parity and smoke save/reload','32 LeAct updates with 50% replay','Verify final reload and merge','Four matched native task evaluations'],
        'interpretation':'Tests the planned next LeAct step. Extra optimization and selected reasoning both change relative to warmstart; a same-budget continued-SFT control would be needed to isolate the admission rule.'})
    registry_path=ROOT/'training-pilot/additional-runs.json';registry=read(registry_path,{'runs':[]})
    ident='efe-qwen36-constrained-leact-lora-v1'
    registry['runs']=[r for r in registry['runs'] if r['id']!=ident]+[{'id':ident,'name':'Qwen3.6 warmstart + LeAct R1',
        'category':'efe-leact','phase':'43 states · 8 candidates · top 2 positive command gains · 50% replay',
        'optimizer_steps':32,'baseline':'efe-qwen36-constrained-warm-lora-v2','build':str(BUILD),
        'local_results_directory':'fixed-action-v1/leact-lora-v1','evaluation_directory':EVAL,
        'description':'First LeAct round from the verified Qwen3.6 warmstart (1/4 native tasks). Same 43 teacher states; eight student explanations, top two positive command-likelihood gains, 50% replay, and 32 further LoRA updates. Same four native task evaluations follow.',
        'evaluation_partition':'training_evaluation'}]
    write(registry_path,registry);return BUILD


def evaluation(ready):
    build=BUILD_ROOT/'efe-qwen36-constrained-leact-eval-20261105-v1'
    if (build/'source-manifest.json').exists():
        spec=read(build/'spec.json')
        if any(spec[k]!=ready[v] for k,v in [('student_checkpoint','checkpoint'),('student_manifest','manifest'),('student_revision','revision')]):
            raise ValueError('Frozen eval checkpoint differs')
        return build
    build.mkdir(exist_ok=True);source=BUILD_ROOT/'efe-qwen36-constrained-warm-eval-20261105-v2'
    for name in read(source/'source-manifest.json'):shutil.copyfile(source/name,build/name)
    spec=read(build/'spec.json');spec.update(student_checkpoint=ready['checkpoint'],student_manifest=ready['manifest'],student_revision=ready['revision'],
        model_id='qwen36-leact-lora-v1',agent_slug='qwen36-leact-lora',agent_label='Qwen3.6 warmstart + LeAct R1',
        export_role='constrained_leact_lora_v1',local_results_directory=EVAL,action_durable_root=ARCHIVE+'/evaluation',
        training_parent=read(DEST/'status.json',{}).get('workflow'),
        scope='Fixed 32-update LeAct round after the 32-update warmstart. Same three training-task scenes plus one untrained RoboCasa task, same v4 native harness. Not an official held-out EFE score.')
    write(build/'spec.json',spec)
    workflow=yaml.safe_load((build/'workflow.yaml').read_text());workflow['workflow']['name']='leact-efe-qwen36-constrained-leact-eval-v1'
    (build/'workflow.yaml').write_text(yaml.safe_dump(workflow,sort_keys=False));freeze_sources(build);return build


def monitor():
    prepare();lock=(DEST/'monitor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    capacity_failures={'FAILED_QUEUE_TIMEOUT','FAILED_START_ERROR','FAILED_START_TIMEOUT','FAILED_PREEMPTED'}
    while True:
        try:
            active=read(DEST/'active-build.json',{'build':str(BUILD),'priority':'HIGH','retry_number':0});build=Path(active['build'])
            receipt=read(build/'submission.json',{}) or submit(build,priority=active['priority'])
            state=command(['osmo','workflow','query',receipt['name'],'--format-type','json']);write(DEST/'scheduler.json',state)
            reader=receipt['name'] if state['status']=='RUNNING' else None
            if state['status']=='COMPLETED' and not read(DEST/'complete.json',{}):reader=recovery_reader(build)
            if reader:
                collect(reader,ARCHIVE,DEST,['worker-start.json','progress.json','complete.json','bootstrap-error.json','parent-verification.json',
                    'selection-runtime.json','membership.json','selection.json','candidate-report.json','selection-complete.json','selection.log',
                    'training-environment.log','modality-parity.json','modality-parity.log','dataset-manifest.json','adapter-configuration.json',
                    'adapter-ready.json','leact-ready.json','smoke/complete.json','smoke/metrics.jsonl','smoke/scoring-parity.json','smoke.log',
                    'smoke/checkpoint-2/serialization-audit.json','smoke-loss-replay.json','smoke-reload.log',
                    'train/complete.json','train/configuration.json','train/metrics.jsonl','train/scoring-parity.json','train.log',
                    'train/checkpoint-8/serialization-audit.json','train/checkpoint-16/serialization-audit.json','train/checkpoint-32/serialization-audit.json',
                    'train-loss-replay.json','train-reload.log','merge.log','export-adapter-verification.json','export-key-mismatch.json'])
            complete=read(DEST/'complete.json',{});ready=read(DEST/'leact-ready.json',{})
            status={'updated_utc':now(),'workflow':receipt['name'],'scheduler_status':state['status'],
                'progress':read(DEST/'progress.json',{}),'selection':read(DEST/'selection.json',{}),'complete':complete}
            if complete and reader:
                remote(reader,"from pathlib import Path;Path("+repr(ARCHIVE+'/collected')+").touch();print('EFE_CAPTURE {\"ack\":true}',flush=True)")
            if complete.get('success') and ready.get('final'):
                ea=read(DEST/'evaluation-active-build.json',{});b=Path(ea['build']) if ea else evaluation(ready)
                er=read(b/'submission.json',{}) or submit(b,priority=ea.get('priority','HIGH'))
                arm={'directory':EVAL,'require_retained':True,'retained_storage':'durable'};(ROOT/EVAL).mkdir(exist_ok=True)
                es=command(['osmo','workflow','query',er['name'],'--format-type','json'])
                if es['status']=='RUNNING':collect_arm(arm,er['name'])
                elif es['status']=='COMPLETED' and not verified_native_completion(arm):
                    rr=reader or recovery_reader(b)
                    if rr:collect_arm({**arm,'host_workflow':rr,'remote_archive':ARCHIVE+'/evaluation/'+er['name']},er['name'])
                elif es['status'] in capacity_failures and not (ROOT/EVAL/'worker-start.json').exists():
                    number=ea.get('retry_number',0)+1;retry=retry_build(b,number)
                    write(DEST/'evaluation-active-build.json',{'build':str(retry),'retry_number':number,'priority':'LOW'})
                done=verified_native_completion(arm)
                status['evaluation']={'workflow':er['name'],'status':es['status'],'behavior':read(ROOT/EVAL/'behavior.json',{}),'complete':done}
                write(DEST/'evaluation-submission.json',{'workflow':er['name'],'build':str(b)})
                if status['evaluation']['behavior'].get('graded'):
                    from ..embodied_first_exam.direct_control_viewer import build as viewer
                    viewer()
            elif state['status'] in capacity_failures and not (DEST/'worker-start.json').exists():
                number=active['retry_number']+1;retry=retry_build(build,number)
                write(DEST/'active-build.json',{'build':str(retry),'priority':'LOW','retry_number':number})
            write(DEST/'status.json',status);publish()
            if complete and (not complete.get('success') or complete.get('no_admissions')):return
            if status.get('evaluation',{}).get('complete'):return
        except Exception as error:write(DEST/'monitor-error.json',{'updated_utc':now(),'error':str(error)[-2000:]})
        time.sleep(30)


def start():
    prepare()
    with (DEST/'monitor.log').open('a') as stream:
        process=subprocess.Popen([sys.executable,'-u','-m',__spec__.name,'monitor'],cwd=BASE.parents[1],
            stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    write(DEST/'monitor-launch.json',{'pid':process.pid,'updated_utc':now()});return process.pid


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','start','monitor']);a=p.parse_args()
    if a.action=='monitor':monitor()
    elif a.action=='start':print(json.dumps({'monitor_pid':start()}))
    else:print(prepare())
