"""Launch a bounded Qwen3.6 constrained-action warmstart and matched robot eval."""
from datetime import datetime,timezone
import argparse
import fcntl
import importlib
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
from .monitor_efe_preflight import remote

DEST=ROOT/'fixed-action-v1/warmstart-lora-v2'
BUILD=BUILD_ROOT/'efe-qwen36-constrained-warm-20261105-v6'
ARCHIVE='/mnt/amlfs-01/home/rpunamiya/leact/studies/efe-qwen36-constrained-warm-20261105-v6'
LANE='/tmp/efe-qwen36-warmstart'
NATIVE='/tmp/efe-training-evaluation-v1/warm96'


def now():return datetime.now(timezone.utc).isoformat()


def prepare():
    if (BUILD/'source-manifest.json').exists():return BUILD
    BUILD.mkdir(exist_ok=True)
    DEST.mkdir(exist_ok=True)
    for name in ['admission.json','decisions.jsonl.gz']:
        if not (DEST/name).exists():shutil.copyfile(ROOT/'fixed-action-v1/warmstart-lora-v1'/name,DEST/name)
    admission=read(DEST/'admission.json')
    spec={'archive_root':ARCHIVE,'local_results_directory':'fixed-action-v1/warmstart-lora-v2',
        'model':'Qwen/Qwen3.6-35B-A3B','base_model':'/tmp/efe-qwen36-warm-base/model',
        'base_revision':'995ad96eacd98c81ed38be0c5b274b04031597b0','context':65536,
        'effective_batch':32,'lr':1e-4,'seed':20261104,'balanced_passes':1024/86,
        'training_transformers':'5.11.0','collective_timeout_seconds':900,
        'prepared_dataset':{'root':'/mnt/amlfs-01/home/rpunamiya/leact/studies/efe-qwen36-constrained-warm-20261104-v3',
            'manifest_sha256':'5ef9b7302b38748cb290ac6f55df8b70702d6ce1668200de7b90ad082bcf748a'},
        'expected_optimizer_steps':32,'sampling_policy':'coverage-first-balanced-v1',
        'checkpoint_fractions':[0.25,0.5,1.0],'lora_rank':32,'lora_alpha':64,
        'checkpoint_revision_prefix':'efe-qwen36-constrained-warm-lora-v2',
        'prior_checkpoint_audit':'/mnt/amlfs-01/home/rpunamiya/leact/studies/efe-qwen36-constrained-warm-20261104-v5',
        'ready_name':'adapter-ready.json','teacher_model':'azure/openai/gpt-6-astra',
        'rationale_max_tokens':8192,'decisions_sha256':admission['decisions_sha256'],
        'scope':'Three successful training-task episodes, 43 commands, 86 forward/backward examples. Language LoRA pilot; distinct from earlier full-parameter 8B SFT. Robot evaluation follows the fixed final update, including the additional RoboCasa task.'}
    write(BUILD/'spec.json',spec)
    mapping={'efe_qwen36_warmstart.py':'runner.py','efe_merge_qwen36_warm.py':'merge.py',
        'efe_train_qwen36_lora.py':'trainer.py','efe_training_data.py':'efe_training_data.py',
        'efe_qwen36_adapter_io.py':'efe_qwen36_adapter_io.py','efe_qwen36_export_recovery.py':'verify.py',
        'efe_training_pilot.py':'efe_training_pilot.py','efe_budget_gateway.py':'efe_budget_gateway.py',
        'text_capability_eval.py':'efe_cuda_helpers.py'}
    for source,dest in mapping.items():shutil.copyfile(BASE/'cluster'/source,BUILD/dest)
    shutil.copyfile(DEST/'admission.json',BUILD/'admission.json')
    shutil.copyfile(DEST/'decisions.jsonl.gz',BUILD/'decisions.jsonl.gz')
    bootstrap='''from pathlib import Path
import hashlib,json,os,re,subprocess,sys,time
lane=Path('/tmp/efe-qwen36-warmstart');spec=json.loads((lane/'spec.json').read_text());root=Path(spec['archive_root']);root.mkdir(parents=True,exist_ok=True)
try:
 (root/'progress.json').write_text(json.dumps({'phase':'waiting_for_corpus','detail':'Waiting for checksum-verified binary corpus transfer.','training_started':False}))
 for _ in range(720):
  source=root/'decisions.jsonl.gz'
  if source.exists() and hashlib.sha256(source.read_bytes()).hexdigest()==spec['decisions_sha256']:break
  time.sleep(5)
 else:raise TimeoutError('Corpus transfer did not complete')
 (lane/'decisions.jsonl.gz').write_bytes(source.read_bytes())
 version=re.search(r'([0-9]{3}[.][0-9]+(?:[.][0-9]+)?)',Path('/proc/driver/nvidia/version').read_text()).group(1)
 libs=[Path('/usr/lib/x86_64-linux-gnu')/(name+'.so.'+version) for name in ['libcuda','libnvidia-ptxjitcompiler','libnvidia-nvvm']]
 assert all(p.is_file() for p in libs)
 env={**os.environ,'LD_PRELOAD':':'.join(map(str,libs)),'LD_LIBRARY_PATH':'/usr/lib/x86_64-linux-gnu:'+os.environ.get('LD_LIBRARY_PATH','')}
 environment=lane/'training-env'
 subprocess.run([sys.executable,'-m','venv','--system-site-packages',str(environment)],check=True)
 python=str(environment/'bin/python')
 subprocess.run([python,'-m','pip','install','--disable-pip-version-check','transformers==5.11.0','peft==0.21.0','accelerate==1.11.0'],env=env,check=True)
 env.update(VIRTUAL_ENV=str(environment),PATH=str(environment/'bin')+':'+env['PATH'])
 subprocess.run([python,'-c',"from transformers import Qwen3_5MoeForConditionalGeneration; import peft; print('Training architecture import passed',flush=True)"],env=env,check=True)
 os.execve(python,[python,'-u',str(lane/'runner.py')],env)
except Exception as e:
 (root/'bootstrap-error.json').write_text(json.dumps({'error':str(e)}));raise
'''
    (BUILD/'bootstrap.py').write_text(bootstrap)
    template=yaml.safe_load((BUILD_ROOT/'efe-fixed-action-20261104-v4-qwen36-movement/workflow.yaml').read_text())
    task=template['workflow']['tasks'][0]
    task.update(command=['python3'],args=['-u',LANE+'/bootstrap.py'])
    task['files']=[{'localpath':name,'path':LANE+'/'+name} for name in
        ['spec.json','bootstrap.py','admission.json',*mapping.values()]]
    template['workflow']['name']='leact-efe-qwen36-warm-lora-v6'
    template['workflow']['resources']['default'].update(cpu=48,memory='512Gi',storage='800Gi')
    template['workflow']['timeout']={'queue_timeout':'20m','exec_timeout':'10h'}
    (BUILD/'workflow.yaml').write_text(yaml.safe_dump(template,sort_keys=False))
    for p in BUILD.glob('*.py'):compile(p.read_text(),str(p),'exec')
    freeze_sources(BUILD)
    write(DEST/'plan.json',{'updated_utc':now(),'build':str(BUILD),'spec':spec,'status':'prepared',
        'sequence':['Audit all prior checkpoint headers','Distributed serialization reproduction and fix check','Reuse the exact 86 prepared examples','Two discarded optimizer smoke updates and fresh-model reload','32 warmstart optimizer updates','Replay final loss from saved weights and verify native merge','Four matched native task evaluations'],
        'previous_attempt':'fixed-action-v1/warmstart-lora-v1','rerun_reason':'Original training finished 32 updates, but PEFT filtered out every FSDP adapter tensor during saving. This attempt uses explicit 700-tensor save validation, a backup, and fresh-model loss replay before production training and export.',
        'priority':'HIGH','capacity_fallback':'LOW','maximum_concurrent_nodes':2})
    write(ROOT/'fixed-action-v1/active-warmstart.json',{'directory':'fixed-action-v1/warmstart-lora-v2',
        'history_id':'efe-qwen36-constrained-warm-lora-v2','updated_utc':now()})
    return BUILD


def evaluation(ready):
    build=BUILD_ROOT/'efe-qwen36-constrained-warm-eval-20261105-v2'
    if (build/'source-manifest.json').exists():
        frozen=read(build/'spec.json')
        for destination,source in [('student_checkpoint','checkpoint'),('student_manifest','manifest'),('student_revision','revision')]:
            if frozen[destination]!=ready[source]:raise ValueError('Verified export differs from the frozen evaluation input')
        return build
    build.mkdir(exist_ok=True);source=BUILD_ROOT/'efe-fixed-action-20261104-v4-qwen36-task'
    for filename in read(source/'source-manifest.json'):shutil.copyfile(source/filename,build/filename)
    spec=read(build/'spec.json')
    spec.update(student_checkpoint=ready['checkpoint'],student_manifest=ready['manifest'],student_revision=ready['revision'],
        model_id='qwen36-warm-lora-v2',agent_slug='qwen36-warm-lora',agent_label='Qwen3.6 · constrained warmstart LoRA',
        export_role='constrained_warm_lora_v2',local_results_directory='fixed-action-v1/warmstart-lora-v2/evaluation',
        action_durable_root=ARCHIVE+'/evaluation',scope='Fixed 32-update LoRA warmstart: three exact training tasks plus one untrained RoboCasa task. Same v4 constrained harness as base Qwen; not an official held-out EFE score.')
    write(build/'spec.json',spec)
    p=build/'native-bootstrap.py';text=p.read_text()
    line="subprocess.run([python, '-u', str(lane / 'efe_prepare_qualification_student.py')], env=env, check=True)\n"
    if text.count(line)!=1:raise ValueError('Native bootstrap hook changed')
    p.write_text(text.replace(line,''))
    workflow=yaml.safe_load((build/'workflow.yaml').read_text());workflow['workflow']['name']='leact-efe-qwen36-warm-lora-eval-v2'
    (build/'workflow.yaml').write_text(yaml.safe_dump(workflow,sort_keys=False));freeze_sources(build)
    return build


def publish():
    from ..embodied_first_exam import report
    try:importlib.reload(report).publish()
    except Exception as error:
        try:print('Reporting error: '+str(error)[-300:],flush=True)
        except OSError:pass


def transfer_corpus(workflow,build):
    if read(DEST/'corpus-upload.json',{}).get('workflow')==workflow:return
    expected=read(build/'spec.json')['decisions_sha256']
    script="from pathlib import Path\nimport hashlib,json\np=Path("+repr(ARCHIVE+'/decisions.jsonl.gz')+");p.parent.mkdir(parents=True,exist_ok=True)\nprint('EFE_CAPTURE '+json.dumps({'sha256':hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None}),flush=True)"
    if remote(workflow,script)['sha256']!=expected:
        result=subprocess.run(['osmo','workflow','rsync','upload',workflow,'preflight',
            str(build/'decisions.jsonl.gz')+':'+ARCHIVE+'/'],capture_output=True,text=True,timeout=120)
        if remote(workflow,script)['sha256']!=expected:
            raise RuntimeError('Corpus transfer incomplete: '+(result.stdout+result.stderr)[-800:])
    write(DEST/'corpus-upload.json',{'workflow':workflow,'sha256':expected,'verified_utc':now()})


def refresh_baseline_status():
    path=ROOT/'fixed-action-v1/status.json';status=read(path,{})
    for arm in status.get('arms',[]):
        local=ROOT/arm['directory'];behavior=read(local/'behavior.json',{})
        complete=read(local/'verified-completion.json',{}) or read(local/'worker-complete.json',{})
        receipt=read(Path(arm['build'])/'submission.json',{})
        arm.update(workflow=receipt.get('name'),graded=behavior.get('graded',0),successes=behavior.get('successes',0),
            control_steps=sum(r['control_steps'] for r in behavior.get('trials',[])),
            movement_passes=behavior.get('movement_passes',0),complete=complete)
        if complete.get('success'):
            arm.update(status='COMPLETED',detail='Verified native grades and retained episode evidence.');arm.pop('needs_review',None)
        elif arm.get('status')=='COMPLETED':
            arm.update(detail='Workflow ended; recovering durable native grades. No task score assigned before receipt verification.')
    status.update(updated_utc=now(),complete=all(a.get('complete',{}).get('success') for a in status.get('arms',[])))
    write(path,status)


def recover_baseline(reader):
    arm={'directory':'fixed-action-v1/runs/v4/qwen36-task','require_retained':True,'retained_storage':'durable',
        'host_workflow':reader,'remote_archive':'/mnt/amlfs-01/home/rpunamiya/leact/studies/efe-constrained-action-20261104-v4/leact-efe-fixed-qwen36-task-v4-1'}
    if read(ROOT/arm['directory']/'verified-completion.json',{}):return
    try:
        collect_arm(arm,'leact-efe-fixed-qwen36-task-v4-1')
        done=verified_native_completion(arm)
        if not done:return
        summary=(f"With the JSON-wrapper adapter, base Qwen3.6 passed 4/9 movement checks with 34/34 valid commands, "
            f"and solved {done['successes']}/{done['graded']} native tasks. Astra passed 9/9 movement checks and solved 3/4 tasks. "
            'These are fixed training-scene diagnostics. The Qwen3.6 LoRA warmstart uses 43 verified teacher actions from three successful episodes; its native task evaluation is a separate upcoming result.')
        write(ROOT/'fixed-action-v1/runs/v4/completed-comparison.json',{'updated_utc':now(),'summary':summary,'qwen36_task':done})
        plan=read(ROOT/'fixed-action-v1/plan.json');plan['results_summary']=summary;write(ROOT/'fixed-action-v1/plan.json',plan)
        refresh_baseline_status()
        from ..embodied_first_exam.direct_control_viewer import build
        build()
    except Exception as error:
        write(DEST/'baseline-recovery-error.json',{'updated_utc':now(),'error':str(error)[-1800:]})


def monitor():
    prepare();lock=(DEST/'monitor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        try:
            active=read(DEST/'active-build.json',{'build':str(BUILD),'retry_number':0,'priority':'HIGH'})
            build=Path(active['build']);receipt=read(build/'submission.json',{}) or submit(build,priority=active['priority'])
            state=command(['osmo','workflow','query',receipt['name'],'--format-type','json'])
            write(DEST/'scheduler.json',state)
            reader=receipt['name'] if state['status']=='RUNNING' else None
            if state['status']=='RUNNING':transfer_corpus(reader,build)
            if not reader and state['status']=='COMPLETED' and not read(DEST/'complete.json',{}):reader=recovery_reader(build)
            if reader:
                collect(reader,ARCHIVE,DEST,['worker-start.json','progress.json','complete.json','bootstrap-error.json','versions.json',
                    'prior-checkpoint-audit.json','serialization-probe.json','serialization-probe.log',
                    'smoke-loss-replay.json','train-loss-replay.json','smoke-reload.log','train-reload.log',
                    'smoke/checkpoint-2/serialization-audit.json','train/checkpoint-8/serialization-audit.json',
                    'train/checkpoint-16/serialization-audit.json','train/checkpoint-32/serialization-audit.json',
                    'export-adapter-verification.json','export-key-mismatch.json',
                    'admission.json','dataset-manifest.json','architecture-preflight.json','prepared-dataset-receipt.json','modality-parity.json','adapter-configuration.json','processor.json','warmstart-ready.json','adapter-ready.json',
                    'smoke/metrics.jsonl','smoke/complete.json','smoke.log','train/metrics.jsonl','train/complete.json','train/configuration.json','train.log','merge.log'])
                recover_baseline(reader)
            complete=read(DEST/'complete.json',{});ready=read(DEST/'warmstart-ready.json',{})
            status={'updated_utc':now(),'workflow':receipt['name'],'scheduler_status':state['status'],
                'progress':read(DEST/'progress.json',{}),'complete':complete,'training_started':(DEST/'train/metrics.jsonl').exists()}
            if complete and reader:
                remote(reader,"from pathlib import Path;Path("+repr(ARCHIVE+'/collected')+").touch();print('EFE_CAPTURE {\"ack\":true}',flush=True)")
            if complete.get('success') and ready.get('final'):
                eval_active=read(DEST/'evaluation-active-build.json',{})
                b=Path(eval_active['build']) if eval_active else evaluation(ready)
                r=read(b/'submission.json',{}) or submit(b,priority=eval_active.get('priority','HIGH'))
                arm={'directory':'fixed-action-v1/warmstart-lora-v2/evaluation','require_retained':True,'retained_storage':'durable'}
                (ROOT/arm['directory']).mkdir(exist_ok=True)
                ev=command(['osmo','workflow','query',r['name'],'--format-type','json'])
                if ev['status']=='RUNNING':collect_arm(arm,r['name'])
                elif ev['status']=='COMPLETED' and not verified_native_completion(arm):
                    eval_reader=reader or recovery_reader(b)
                    if eval_reader:
                        collect_arm({**arm,'host_workflow':eval_reader,'remote_archive':ARCHIVE+'/evaluation/'+r['name']},r['name'])
                        if eval_reader!=reader and verified_native_completion(arm):
                            remote(eval_reader,"from pathlib import Path;Path('/tmp/efe-reader-collected').touch();print('EFE_CAPTURE {\"ack\":true}',flush=True)")
                elif ev['status'] in {'FAILED_QUEUE_TIMEOUT','FAILED_START_ERROR','FAILED_START_TIMEOUT','FAILED_PREEMPTED'} and not (ROOT/arm['directory']/'worker-start.json').exists():
                    retry_number=eval_active.get('retry_number',0)+1;retry=retry_build(b,retry_number)
                    write(DEST/'evaluation-active-build.json',{'build':str(retry),'retry_number':retry_number,'priority':'LOW'})
                status['evaluation']={'workflow':r['name'],'status':ev['status'],'behavior':read(ROOT/arm['directory']/'behavior.json',{})}
                done=verified_native_completion(arm)
                write(DEST/'evaluation-submission.json',{'workflow':r['name'],'build':str(b)})
                if done:status['evaluation']['complete']=done
                if status['evaluation'].get('behavior',{}).get('graded',0):
                    from ..embodied_first_exam.direct_control_viewer import build as build_viewer
                    build_viewer()
            elif state['status'] in {'FAILED_QUEUE_TIMEOUT','FAILED_START_ERROR','FAILED_START_TIMEOUT','FAILED_PREEMPTED'} and not (DEST/'worker-start.json').exists():
                retry=retry_build(build,active['retry_number']+1)
                write(DEST/'active-build.json',{'build':str(retry),'retry_number':active['retry_number']+1,'priority':'LOW'})
                status['retry']='LOW-priority allocation retry prepared.'
            write(DEST/'status.json',status);publish()
            if complete and not complete.get('success'):return
            if status.get('evaluation',{}).get('complete'):return
        except Exception as error:
            try:write(DEST/'monitor-error.json',{'updated_utc':now(),'error':str(error)[-1800:]})
            except OSError:pass
        time.sleep(30)


def start():
    prepare()
    with (DEST/'monitor.log').open('a') as log:
        p=subprocess.Popen([sys.executable,'-u','-m',__spec__.name,'monitor'],cwd=BASE.parents[1],
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(DEST/'monitor-launch.json',{'pid':p.pid,'updated_utc':now()});return p.pid


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','start','monitor']);a=p.parse_args()
    if a.action=='monitor':monitor()
    elif a.action=='start':print(json.dumps({'pid':start()}))
    else:print(prepare())
