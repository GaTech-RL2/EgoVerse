"""Launch fixed-input LeAct recovery and automatically evaluate the rebuilt model."""
from pathlib import Path
import argparse
import base64
import fcntl
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import time
import yaml

from .scale_efe import BASE, ROOT, BUILD_ROOT, read, write, submit, command
from .launch_efe_training import freeze_sources
from .launch_efe_leact_warm96 import collect, download, preserve_model
from .launch_efe_training_eval import DEST, publish_status, publish_viewer, now
from .monitor_efe_preflight import remote

BUILD=BUILD_ROOT/'efe-leact-warm96-rebuild-20261030-v5'
LOCAL=ROOT/'training-pilot/leact-warm96-rebuild-v2'
EVAL=DEST/'leact-rebuilt'
ARCHIVE='/tmp/efe-leact-warm96-rebuild-v2'
LANE='/tmp/efe-training-evaluation-v1/warm96'


def prepare():
    BUILD.mkdir(exist_ok=True);LOCAL.mkdir(exist_ok=True);EVAL.mkdir(exist_ok=True)
    if (BUILD/'source-manifest.json').exists():return BUILD
    original=BUILD_ROOT/'efe-leact-warm96-20261029-v1'
    retained=ROOT/'training-pilot/leact-warm96-r1-v1'
    bundle=BUILD/'retained.tar.gz'
    with tarfile.open(bundle,'w:gz') as archive:
        for name in ['dataset-manifest.json','membership.json','selection.json','train/schedule.json']:
            archive.add(retained/name,arcname=name)
        archive.add(retained/'collected/portable-model/checkpoint-manifest.json',arcname='checkpoint-manifest.json')
        for path in sorted((retained/'candidates').glob('*.json')):archive.add(path,arcname='candidates/'+path.name)
    (BUILD/'retained.b64').write_text(base64.b64encode(bundle.read_bytes()).decode())
    spec=read(original/'spec.json');schedule=read(retained/'train/schedule.json')
    spec.update(archive_root=ARCHIVE,stage='leact-warm96-rebuild2',
        retained_bundle_sha256=hashlib.sha256(bundle.read_bytes()).hexdigest(),
        expected_optimizer_steps=12,expected_schedule_sha256=schedule['schedule_sha256'],
        scope='Reconstruct exactly retained tensor hashes and replay the original 12-step schedule after node eviction; no new candidates, selection, or expert requests.',
        local_results_directory='training-pilot/leact-warm96-rebuild-v2',
        local_evaluation_directory='training-evaluation-v1/leact-rebuilt')
    write(BUILD/'spec.json',spec)
    files=[{'localpath':'spec.json','path':'/tmp/efe-rebuild-spec.json'}, {'localpath':'retained.b64','path':'/tmp/efe-retained-leact.b64'}]
    frozen=read(original/'source-manifest.json')
    for name in ['efe_training_data.py','efe_training_pilot.py','efe_train_vl.py','efe_leact_pipeline.py','efe_budget_gateway.py']:
        if hashlib.sha256((original/name).read_bytes()).hexdigest()!=frozen[name]:raise ValueError('Original trainer source changed')
        shutil.copyfile(original/name,BUILD/name);files.append({'localpath':name,'path':'/tmp/'+name})
        if name=='efe_train_vl.py':
            text=(BUILD/name).read_text()
            before="                if error>.05:raise ValueError('Training/selection complete-target likelihood differs by over 0.05 nats/token')"
            after="                if rank==0:\n                    write(directory/'scoring-parity.json',checks)\n                    print(json.dumps({'scoring_parity':checks[-1]}),flush=True)\n"+before
            if text.count(before)!=1:raise ValueError('Original parity guard changed')
            (BUILD/name).write_text(text.replace(before,after))
    shutil.copyfile(BASE/'cluster/efe_rebuild_saved_leact.py',BUILD/'rebuild.py');files.append({'localpath':'rebuild.py','path':'/tmp/efe_rebuild_saved_leact.py'})
    native=BUILD_ROOT/'efe-training-evaluation-20261029-v1/warm96'
    (BUILD/'native').mkdir(exist_ok=True)
    for name in ['efe_native_pilot.py','student.py','pilot.py','efe_budget_gateway.py','export_efe_viewer.py','worker.py']:
        text=(native/name).read_text()
        if name=='pilot.py':
            current=(BASE/'cluster/efe_astra_pilot.py').read_text();first=current.index('            # A model that uses up its task budget');last=current.index("\n            valid &= rewards.get('success')",first)
            text=text.replace("            valid = bool(trial) and not trial.get('exception_info') and receipt['returncode'] == 0",current[first:last])
        if name=='worker.py':text=text.replace("if name == 'system_packages':", "if name == 'system_packages' and spec.get('reuse_system_packages', True):")
        (BUILD/'native'/name).write_text(text)
        # OSMO rejects two mounts with identical content-derived volume names.
        # The already mounted training gateway is copied to this path at runtime.
        if name!='efe_budget_gateway.py':files.append({'localpath':'native/'+name,'path':LANE+'/'+name})
        elif text!=(BUILD/'efe_budget_gateway.py').read_text():raise ValueError('Training/native gateway sources differ')
    evaluation=read(native/'spec.json');evaluation.update(model_id='leact-rebuilt',export_role='training_leact_rebuilt',
        agent_slug='qwen-training-leact-rebuilt',agent_label='Warmstart 96 + LeAct rebuilt',reuse_system_packages=False,
        local_results_directory='training-evaluation-v1/leact-rebuilt',
        purpose='Separately labeled 21-task training-set evaluation of a checkpoint rebuilt from the exact retained LeAct data/schedule after eviction.')
    write(BUILD/'native/spec.json',evaluation);files.append({'localpath':'native/spec.json','path':LANE+'/spec.json'})
    shutil.copyfile(BASE/'cluster/efe_eval_fresh_bootstrap.py',BUILD/'bootstrap.py');files.append({'localpath':'bootstrap.py','path':'/tmp/efe_eval_bootstrap.py'})
    digests=[hashlib.sha256((BUILD/f['localpath']).read_bytes()).hexdigest() for f in files]
    if len(digests)!=len(set(digests)):raise ValueError('Duplicate OSMO mounted file content')
    w=yaml.safe_load((original/'workflow.yaml').read_text());w['workflow']['name']='leact-efe-warm96-rebuild-v4'
    w['workflow']['resources']['default']['storage']='800Gi';w['workflow']['timeout']={'queue_timeout':'20m','exec_timeout':'30h'}
    w['workflow']['tasks'][0].update(command=['python3'],args=['-u','/tmp/efe_rebuild_saved_leact.py'],files=files)
    (BUILD/'workflow.yaml').write_text(yaml.safe_dump(w,sort_keys=False));freeze_sources(BUILD)
    plan=read(DEST/'plan.json');plan.update(models=['warm96','leact','leact-rebuilt'],
        model_labels={'leact':'Warmstart 96 + LeAct (original; interrupted)','leact-rebuilt':'Warmstart 96 + LeAct (rebuilt)'},
        recovery_note='The original node was evicted for storage. Warmstart continues only ungraded tasks with the same weights; ordinary agent timeouts remain failed outcomes. The original LeAct result remains separate; the rebuilt checkpoint replays the saved data/schedule and gets its own 21-task evaluation.')
    write(DEST/'plan.json',plan);write(EVAL/'progress.json',{'phase':'queued_rebuild','detail':'Rebuilding original 12 updates from retained candidates before separately labeled evaluation.','updated_utc':now()})
    publish_status();publish_viewer()
    return BUILD


def monitor():
    lock=(LOCAL/'monitor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    build=Path(read(LOCAL/'active-build.json',{}).get('build',str(BUILD)))
    retry_number=int(read(LOCAL/'active-build.json',{}).get('retry_number',0))
    while True:
        try:
            workflow=read(build/'submission.json')['name'];state=command(['osmo','workflow','query',workflow,'--format-type','json']);write(LOCAL/'scheduler.json',state)
            if state['status']=='RUNNING':
                root=ARCHIVE+'/'+workflow
                collect(workflow,root,LOCAL,['progress.json','complete.json','reconstruction.json','weight-comparison.json','portable-ready.json','leact-ready.json',
                    'train/metrics.jsonl','train/schedule.json','train/configuration.json','train/complete.json','train/scoring-parity.json','smoke/scoring-parity.json','smoke/complete.json','smoke.log','train.log','native.log',
                    'probes/base-0-tokens.json','probes/base-1-tokens.json'])
                ready=read(LOCAL/'portable-ready.json')
                if ready:
                    preserve_model(workflow,root,LOCAL)
                    native=LANE+'/results/'+workflow
                    collect(workflow,native,EVAL,['worker-start.json','worker-complete.json','progress.json','results.json','complete.json','behavior.json',
                        'student-serving.json','student-preflight.json','student-serving-failure.json','setup-error.json','public-episodes/manifest.json'])
                    for row in read(EVAL/'public-episodes/manifest.json',{}).get('episodes',[]):
                        def valid():
                            return all((EVAL/'public-episodes'/row['id']/name).is_file() and hashlib.sha256((EVAL/'public-episodes'/row['id']/name).read_bytes()).hexdigest()==row[key]
                                for name,key in [('trace.json','trace_sha256'),('replay.mp4','video_sha256')])
                        if not valid():download(workflow,native+'/public-episodes/'+row['id'],EVAL/'public-episodes',timeout=120)
                        if not valid():raise ValueError('Rebuilt evaluation recording checksum mismatch')
                    if read(EVAL/'complete.json'):
                        remote(workflow,"from pathlib import Path\nPath("+repr(LANE+'/runtime/collected')+").touch()\nprint('EFE_CAPTURE {\"ack\":true}',flush=True)")
                else:
                    progress=read(LOCAL/'progress.json',{});write(EVAL/'progress.json',progress)
                publish_status();publish_viewer()
                if read(EVAL/'worker-complete.json') and read(LOCAL/'portable-collected.json'):
                    remote(workflow,"from pathlib import Path\nPath("+repr(root+'/collected')+").touch()\nprint('EFE_CAPTURE {\"ack\":true}',flush=True)")
                    write(LOCAL/'monitor-complete.json',{'workflow':workflow,'updated_utc':now()});return
                if read(LOCAL/'complete.json',{}).get('success') is False:
                    write(LOCAL/'monitor-error.json',read(LOCAL/'complete.json'));return
            elif state['status'].startswith('FAILED'):
                if not read(LOCAL/'progress.json') and state['status'] in ['FAILED_QUEUE_TIMEOUT','FAILED_PREEMPTED','FAILED_START_ERROR','FAILED_START_TIMEOUT']:
                    retry_number+=1
                    retry=BUILD.with_name(BUILD.name+'-low'+str(retry_number));retry.mkdir(exist_ok=True)
                    if not (retry/'source-manifest.json').exists():
                        for name in read(BUILD/'source-manifest.json'):
                            dst=retry/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(BUILD/name,dst)
                        w=yaml.safe_load((retry/'workflow.yaml').read_text());w['workflow']['name']+='-low'+str(retry_number);w['workflow']['timeout']['queue_timeout']='12h';(retry/'workflow.yaml').write_text(yaml.safe_dump(w,sort_keys=False));freeze_sources(retry)
                    submit(retry,priority='LOW');build=retry
                    write(LOCAL/'active-build.json',{'build':str(build),'retry_number':retry_number,'updated_utc':now()})
                else:write(LOCAL/'monitor-error.json',{'scheduler_status':state['status'],'updated_utc':now()});return
            write(LOCAL/'monitor-status.json',{'workflow':workflow,'status':state['status'],'updated_utc':now()})
        except Exception as error:
            write(LOCAL/'monitor-error.json',{'error':str(error)[-1800:],'updated_utc':now()})
        time.sleep(30)


def start_monitor():
    with (LOCAL/'monitor.log').open('a') as log:
        p=subprocess.Popen([sys.executable,'-u','-m','reproductions.leact.cluster.rebuild_efe_leact_checkpoint','monitor'],cwd=BASE.parents[1],
             stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(LOCAL/'monitor-launch.json',{'pid':p.pid,'updated_utc':now()})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','submit','monitor']);args=parser.parse_args()
    if args.action=='monitor':monitor()
    else:
        prepare()
        if args.action=='submit':print(json.dumps(submit(BUILD,priority='HIGH')));start_monitor()
