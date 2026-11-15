"""Export-only recovery of the completed warmstart, followed by native evaluation."""
import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
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
from . import launch_efe_qwen36_warm as warm

PARENT=warm.DEST
DEST=PARENT/'export-recovery-v1'
BUILD=BUILD_ROOT/'efe-qwen36-warm-export-20261105-v1'
ARCHIVE='/mnt/amlfs-01/home/rpunamiya/leact/studies/efe-qwen36-warm-export-20261105-v1'
LANE='/tmp/efe-qwen36-export'


def now():return datetime.now(timezone.utc).isoformat()
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    DEST.mkdir(parents=True,exist_ok=True)
    if (BUILD/'source-manifest.json').exists():return BUILD
    done=read(PARENT/'train/complete.json');ready=read(PARENT/'adapter-ready.json')
    if not done['success'] or done['optimizer_steps']!=32 or not ready['final']:raise ValueError('Completed fixed-budget training required')
    BUILD.mkdir(exist_ok=True)
    spec={'archive_root':ARCHIVE,'training_root':warm.ARCHIVE,'checkpoint':ready['checkpoint'],
        'model':'Qwen/Qwen3.6-35B-A3B','base_revision':'995ad96eacd98c81ed38be0c5b274b04031597b0',
        'base_model':'/tmp/efe-qwen36-export-base/model','expected_post_update_ce':done['post_update_full_target_ce'],
        'training_complete_sha256':sha(PARENT/'train/complete.json'),'dataset_sha256':sha(PARENT/'dataset-manifest.json'),
        'additional_optimizer_updates':0,'scope':'Verify and export the existing step-32 adapter; no additional training or data changes.'}
    write(BUILD/'spec.json',spec)
    shutil.copyfile(BASE/'cluster/efe_qwen36_export_recovery.py',BUILD/'recovery.py')
    source=warm.BUILD
    for name in ['trainer.py','efe_training_data.py','efe_budget_gateway.py']:
        shutil.copyfile(source/name,BUILD/name)
    bootstrap=(source/'bootstrap.py').read_text().replace(warm.LANE,LANE)
    a=bootstrap.index(" (root/'progress.json').write_text");b=bootstrap.index(' version=re.search',a)
    bootstrap=bootstrap[:a]+bootstrap[b:]
    bootstrap=bootstrap.replace("env.update(VIRTUAL_ENV=", "env.update(NCCL_P2P_DISABLE='1',CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',TOKENIZERS_PARALLELISM='false',VIRTUAL_ENV=")
    bootstrap=bootstrap.replace("[python,'-u',str(lane/'runner.py')]", "[python,'-u',str(lane/'recovery.py'),'--spec',str(lane/'spec.json')]")
    (BUILD/'bootstrap.py').write_text(bootstrap)
    workflow=yaml.safe_load((source/'workflow.yaml').read_text())
    workflow['workflow']['name']='leact-efe-qwen36-warm-export-v1'
    workflow['workflow']['timeout']={'queue_timeout':'20m','exec_timeout':'3h'}
    task=workflow['workflow']['tasks'][0];task.pop('credentials',None)
    task.update(command=['python3'],args=['-u',LANE+'/bootstrap.py'],files=[{'localpath':p.name,'path':LANE+'/'+p.name} for p in BUILD.iterdir() if p.is_file()])
    (BUILD/'workflow.yaml').write_text(yaml.safe_dump(workflow,sort_keys=False))
    for p in BUILD.glob('*.py'):compile(p.read_text(),str(p),'exec')
    freeze_sources(BUILD)
    write(DEST/'plan.json',{'updated_utc':now(),'build':str(BUILD),'spec':spec,'priority':'HIGH','capacity_fallback':'LOW',
        'sequence':['Verify saved adapter keys/shapes/checksums','Exact tensor reload','Replay original final loss without optimizer updates','Merge native-serving model','Matched four-task robot evaluation']})
    return BUILD


def publish():
    from ..embodied_first_exam import report
    try:importlib.reload(report).publish()
    except Exception as error:print('Reporting error: '+str(error)[-300:],flush=True)


def monitor():
    prepare();lock=(DEST/'monitor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        try:
            active=read(DEST/'active-build.json',{'build':str(BUILD),'priority':'HIGH','retry_number':0})
            build=Path(active['build']);receipt=read(build/'submission.json',{}) or submit(build,priority=active['priority'])
            state=command(['osmo','workflow','query',receipt['name'],'--format-type','json']);write(DEST/'scheduler.json',state)
            reader=receipt['name'] if state['status']=='RUNNING' else None
            if reader:
                collect(reader,ARCHIVE,DEST,['worker-start.json','progress.json','complete.json','bootstrap-error.json',
                    'checkpoint-inspection.json','adapter-load-verification.json','loss-replay.json','versions.json',
                    'verify.log','export.log','export-adapter-verification.json','export-key-mismatch.json','warmstart-ready.json'])
            complete=read(DEST/'complete.json',{});ready=read(DEST/'warmstart-ready.json',{})
            progress=read(DEST/'progress.json',{'phase':'export_recovery_queued','detail':'All 32 training updates completed. Export-only recovery submitted; robot evaluation follows verification.','training_started':True})
            status={'updated_utc':now(),'workflow':receipt['name'],'scheduler_status':state['status'],'progress':progress,
                'complete':complete,'training_started':True,'training_completed_steps':32,'additional_optimizer_updates':0,
                'training_workflow':'leact-efe-qwen36-warm-lora-v5-1'}
            if complete and reader:
                remote(reader,"from pathlib import Path;Path("+repr(ARCHIVE+'/collected')+").touch();print('EFE_CAPTURE {\"ack\":true}',flush=True)")
            if complete.get('success') and ready.get('final'):
                write(PARENT/'warmstart-ready.json',ready)
                eval_active=read(DEST/'evaluation-active-build.json',{})
                b=Path(eval_active['build']) if eval_active else warm.evaluation(ready)
                r=read(b/'submission.json',{}) or submit(b,priority=eval_active.get('priority','HIGH'))
                arm={'directory':'fixed-action-v1/warmstart-lora-v1/evaluation','require_retained':True,'retained_storage':'durable'}
                (ROOT/arm['directory']).mkdir(parents=True,exist_ok=True)
                ev=command(['osmo','workflow','query',r['name'],'--format-type','json'])
                if ev['status']=='RUNNING':collect_arm(arm,r['name'])
                elif ev['status']=='COMPLETED' and not verified_native_completion(arm):
                    host=reader or recovery_reader(b)
                    if host:collect_arm({**arm,'host_workflow':host,'remote_archive':warm.ARCHIVE+'/evaluation/'+r['name']},r['name'])
                elif ev['status'] in {'FAILED_QUEUE_TIMEOUT','FAILED_START_ERROR','FAILED_START_TIMEOUT','FAILED_PREEMPTED'} and not (ROOT/arm['directory']/'worker-start.json').exists():
                    number=eval_active.get('retry_number',0)+1;retry=retry_build(b,number)
                    write(DEST/'evaluation-active-build.json',{'build':str(retry),'retry_number':number,'priority':'LOW'})
                status['evaluation']={'workflow':r['name'],'status':ev['status'],'behavior':read(ROOT/arm['directory']/'behavior.json',{}),
                    'progress':read(ROOT/arm['directory']/'progress.json',{})}
                graded=status['evaluation']['behavior'].get('graded',0)
                if graded>read(DEST/'viewer-published.json',{}).get('graded',0):
                    from ..embodied_first_exam.direct_control_viewer import build as viewer
                    viewer();write(DEST/'viewer-published.json',{'graded':graded,'updated_utc':now()})
                write(PARENT/'evaluation-submission.json',{'workflow':r['name'],'build':str(b)})
                done=verified_native_completion(arm)
                if done:status['evaluation']['complete']=done
                write(PARENT/'complete.json',complete)
            elif state['status'] in {'FAILED_QUEUE_TIMEOUT','FAILED_START_ERROR','FAILED_START_TIMEOUT','FAILED_PREEMPTED'} and not (DEST/'worker-start.json').exists():
                number=active['retry_number']+1;retry=retry_build(build,number)
                write(DEST/'active-build.json',{'build':str(retry),'priority':'LOW','retry_number':number})
                status['retry']='LOW-priority allocation retry prepared.'
            write(DEST/'status.json',status);write(PARENT/'status.json',status);write(PARENT/'progress.json',progress);publish()
            if complete and not complete.get('success'):return
            if status.get('evaluation',{}).get('complete'):
                from ..embodied_first_exam.direct_control_viewer import build as viewer
                viewer();publish();return
        except Exception as error:
            write(DEST/'monitor-error.json',{'updated_utc':now(),'error':str(error)[-2000:]})
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
