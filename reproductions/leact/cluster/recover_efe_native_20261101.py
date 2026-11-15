"""Finish the two interrupted native evaluations without repeating graded tasks."""
import argparse
import fcntl
import hashlib
import shutil
import subprocess
import sys
import time
from pathlib import Path
import yaml

from .scale_efe import BASE, ROOT, BUILD_ROOT, read, write, submit, command
from .compute_budget import node_limit, reserved_nodes
from .launch_efe_training import freeze_sources
from .launch_efe_native_followups import collect_arm
from .launch_efe_training_eval import publish_status, publish_viewer
from .launch_efe_direct_warmstart import infrastructure_retry, now

DEST=ROOT/'recovery-20261101/native'
DURABLE='/mnt/amlfs-03/home/rpunamiya/leact/studies/efe-native-recovery-20261101-v1'
ARMS=[
    {'id':'warm96','source':'efe-training-warm96-resume-20261030-v4','remaining':2,'suffix':'v5'},
    {'id':'leact-rebuilt','source':'efe-native-followups-20261030-v1-leact-rebuilt','remaining':17,'suffix':'v2'},
]


def prepare():
    DEST.mkdir(parents=True,exist_ok=True)
    arms=[]
    for original in ARMS:
        ident=original['id'];canonical='training-evaluation-v1/'+ident
        directory=canonical+original.get('directory_suffix','-recovery-20261101')
        build=BUILD_ROOT/original.get('build_name','efe-native-recovery-20261101-'+ident)
        arm={**original,'canonical':canonical,'directory':directory,'build':str(build)}
        (ROOT/directory).mkdir(exist_ok=True);(DEST/ident).mkdir(exist_ok=True)
        if not (build/'source-manifest.json').exists():
            source=BUILD_ROOT/original['source'];build.mkdir(exist_ok=True)
            for name in read(source/'source-manifest.json'):shutil.copyfile(source/name,build/name)
            spec=read(build/'spec.json');graded={t['task'] for t in read(ROOT/canonical/'behavior.json')['trials']}
            spec['tasks']=[t for t in spec['tasks'] if t['task'] not in graded]
            if len(spec['tasks'])!=original['remaining']:raise ValueError('Unexpected remaining evaluation membership')
            spec.update(local_results_directory=directory,durable_results_root=DURABLE,
                recovery_of=read(source/'submission.json')['name'],
                recovery_reason='Cluster quota reclaim; resume only tasks without a native grade. Original checkpoint, sampling, prompt, tools, simulator seed and all existing outcomes are preserved.')
            write(build/'spec.json',spec)
            worker=(build/'worker.py').read_text()
            worker=worker.replace('def write(path, value):','DURABLE_ARCHIVE = None\n\ndef write(path, value):',1)
            anchor='    temporary.replace(path)'
            addition='''
    if DURABLE_ARCHIVE is not None:
        source, destination = DURABLE_ARCHIVE
        if path.is_relative_to(source):
            relative = path.relative_to(source)
            target = destination / relative
            if relative.as_posix() == 'public-episodes/manifest.json':
                for row in value.get('episodes', []):
                    shutil.copytree(source/'public-episodes'/row['id'], destination/'public-episodes'/row['id'], dirs_exist_ok=True)
            if relative.as_posix() == 'student-prompt-manifest.json':
                shutil.copytree(source/'student-prompt-requests', destination/'student-prompt-requests', dirs_exist_ok=True)
            target.parent.mkdir(parents=True, exist_ok=True)
            staged=target.with_suffix(target.suffix+'.tmp');shutil.copyfile(path,staged);staged.replace(target)
'''
            if worker.count(anchor)!=1:raise ValueError('Worker write hook changed')
            worker=worker.replace(anchor,anchor+addition,1)
            worker=worker.replace('    public = archive / \'public-episodes\'',
                "    global DURABLE_ARCHIVE\n    durable=Path(spec['durable_results_root'])/runtime.WORKFLOW\n    durable.mkdir(parents=True,exist_ok=True)\n    DURABLE_ARCHIVE=(archive,durable)\n    public = archive / 'public-episodes'",1)
            anchor='        original_write(name, value)'
            if worker.count(anchor)!=1:raise ValueError('Native receipt hook changed')
            worker=worker.replace(anchor,anchor+"\n        if name in {'results.json','progress.json','complete.json','setup-error.json','student-serving.json','student-preflight.json','student-serving-failure.json'}:\n            write(durable/name,value)",1)
            (build/'worker.py').write_text(worker)
            workflow=yaml.safe_load((build/'workflow.yaml').read_text())
            workflow['workflow']['name']='leact-efe-native-'+ident+'-recovery-'+original['suffix']
            (build/'workflow.yaml').write_text(yaml.safe_dump(workflow,sort_keys=False))
            for p in build.glob('*.py'):compile(p.read_text(),str(p),'exec')
            freeze_sources(build)
        arms.append(arm)
    write(DEST/'plan.json',{'arms':arms,'updated_utc':now(),'node_cap':node_limit()})
    return arms


def merge(arm):
    source=ROOT/arm['directory'];target=ROOT/arm['canonical']
    current=read(target/'behavior.json');trials={t['task']:t for t in current['trials']}
    incoming=read(source/'behavior.json',{})
    allowed={t['task'] for t in read(Path(arm['build'])/'spec.json')['tasks']}
    for trial in incoming.get('trials',[]):
        if trial['task'] not in allowed:raise ValueError('Unexpected resumed evaluation task')
        previous=trials.get(trial['task'])
        if previous and previous!=trial:raise ValueError('Refusing to replace a graded outcome')
        trials[trial['task']]=trial
    manifest=read(target/'public-episodes/manifest.json',{'episodes':[]})
    media={r['task']:r for r in manifest['episodes']}
    for row in read(source/'public-episodes/manifest.json',{'episodes':[]})['episodes']:
        if row['task'] in media:
            if row['trace_sha256']!=media[row['task']]['trace_sha256']:raise ValueError('Previously graded trace changed')
            continue
        shutil.copytree(source/'public-episodes'/row['id'],target/'public-episodes'/row['id'],dirs_exist_ok=True)
        media[row['task']]=row
    manifest.update(episodes=list(media.values()),new_rollouts=len(media),created_utc=now())
    write(target/'public-episodes/manifest.json',manifest)
    current.update(trials=list(trials.values()),graded=len(trials),successes=sum(t['score'] for t in trials.values()),
        success_rate=sum(t['score'] for t in trials.values())/len(trials),
        matched_initial_states=sum(bool(t.get('initial_state_matches_teacher')) for t in trials.values()),updated_utc=now())
    write(target/'behavior.json',current)
    if read(source/'progress.json'):
        progress=read(source/'progress.json');progress.update(completed_trials=len(trials),planned_trials=21);write(target/'progress.json',progress)
    if len(trials)==21:
        write(target/'worker-complete.json',{'success':True,'graded':21,'successes':current['successes'],
            'provenance':'Disjoint native trials of the same frozen checkpoint; previously graded failures retained.',
            'completed_utc':now()})


def prepare_partial_retry(arm, build, workflow, control):
    """Preserve recovered native grades and freeze a disjoint remaining subset."""
    canonical=read(ROOT/arm['canonical']/'behavior.json')
    graded={t['task'] for t in canonical['trials']}
    raw=read(ROOT/arm['directory']/'results.json',{'trials':[]})['trials']
    if any(t.get('status')!='graded' for t in raw):raise ValueError('Review non-graded native outcomes before retrying')
    if not {t['task'] for t in raw}<=graded:raise ValueError('A native grade lacks its merged behavior record; preserve it before retrying')
    spec=read(build/'spec.json');remaining=[t for t in spec['tasks'] if t['task'] not in graded]
    if not remaining:return None
    previous=read(control/'active-build.json',{})
    number=previous.get('retry_number',0)+1
    root_build=Path(previous.get('root_build',arm['build']))
    target=Path(str(root_build)+'-resume'+str(number))
    directory=arm['canonical']+'-retry-'+hashlib.sha256(workflow.encode()).hexdigest()[:10]
    if not (target/'source-manifest.json').exists():
        target.mkdir(exist_ok=True)
        for name in read(build/'source-manifest.json'):shutil.copyfile(build/name,target/name)
        spec.update(tasks=remaining,local_results_directory=directory,recovery_of=workflow,
            recovery_reason='Automatic quota-reclaim recovery. Shared native grades merged first; only the remaining ungraded tasks are evaluated.')
        write(target/'spec.json',spec)
        w=yaml.safe_load((target/'workflow.yaml').read_text())
        origin=yaml.safe_load((root_build/'workflow.yaml').read_text())
        w['workflow']['name']=origin['workflow']['name']+'-r'+str(number)
        w['workflow']['timeout']['queue_timeout']='12h'
        (target/'workflow.yaml').write_text(yaml.safe_dump(w,sort_keys=False));freeze_sources(target)
    if {t['task'] for t in read(target/'spec.json')['tasks']}!={t['task'] for t in remaining}:
        raise ValueError('Existing native retry has different task membership')
    (ROOT/directory).mkdir(exist_ok=True)
    record={'build':str(target),'root_build':str(root_build),'directory':directory,
        'retry_number':number,'priority':'LOW','previous_workflow':workflow,'updated_utc':now()}
    write(control/'active-build.json',record)
    write(control/'automatic-recovery.json',{**record,'preserved_graded':len(graded),
        'remaining_tasks':[t['task'] for t in remaining]})
    return record


def monitor():
    arms=prepare();lock=(DEST/'monitor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        try:
            active=command(['osmo','workflow','list','--name','leact','--status','RUNNING','PENDING','WAITING','--format-type','json'])['workflows']
            done=0;rows=[]
            for declared in arms:
                control=DEST/declared['id'];active_build=read(control/'active-build.json',{})
                arm={**declared,**{k:active_build[k] for k in ['build','directory'] if k in active_build}}
                build=Path(arm['build'])
                receipt=read(build/'submission.json')
                if not receipt:
                    if reserved_nodes(active)>=node_limit():continue
                    receipt=submit(build,priority=active_build.get('priority','HIGH'));active.append({'name':receipt['name'],'status':'PENDING'})
                workflow=receipt['name'];state=command(['osmo','workflow','query',workflow,'--format-type','json'])
                write(control/'scheduler.json',state)
                if state['status']=='RUNNING':
                    collect_arm(arm,workflow);merge(arm)
                elif state['status'].startswith('FAILED'):
                    graded=read(ROOT/arm['directory']/'behavior.json',{}).get('graded',0)
                    if state.get('cancelled_by')=='groot-idle-job-shutdown' and not read(ROOT/arm['canonical']/'worker-complete.json',{}).get('success'):
                        host=next((r['name'] for r in active if r['status']=='RUNNING'),None)
                        if host:
                            collect_arm({**arm,'host_workflow':host,'remote_archive':DURABLE+'/'+workflow},workflow)
                            merge(arm)
                            prepare_partial_retry(arm,build,workflow,control)
                        else:
                            write(control/'recovery-wait.json',{'workflow':workflow,'phase':'waiting_for_artifact_reader',
                                'detail':'Awaiting a running allocation to recover shared grades before assigning remaining tasks.','updated_utc':now()})
                    elif infrastructure_retry(state) and not graded and not read(ROOT/arm['directory']/'worker-start.json'):
                        number=read(control/'active-build.json',{}).get('retry_number',0)+1
                        retry=Path(arm['build']+'-low'+str(number));retry.mkdir(exist_ok=True)
                        if not (retry/'source-manifest.json').exists():
                            for name in read(Path(arm['build'])/'source-manifest.json'):shutil.copyfile(Path(arm['build'])/name,retry/name)
                            w=yaml.safe_load((retry/'workflow.yaml').read_text());w['workflow']['name']+='-low'+str(number);w['workflow']['timeout']['queue_timeout']='12h'
                            (retry/'workflow.yaml').write_text(yaml.safe_dump(w,sort_keys=False));freeze_sources(retry)
                        submit(retry,priority='LOW');write(control/'active-build.json',{**active_build,'build':str(retry),'retry_number':number,'priority':'LOW'})
                    else:write(control/'interruption.json',{'workflow':workflow,'status':state['status'],'graded_collected':graded,
                        'durable_results':DURABLE+'/'+workflow,'next':'Recover durable results before reassigning any ungraded task.','updated_utc':now()})
                complete=read(ROOT/arm['canonical']/'worker-complete.json',{}).get('success',False)
                done+=int(complete);behavior=read(ROOT/arm['canonical']/'behavior.json')
                rows.append({'id':arm['id'],'workflow':workflow,'status':state['status'],
                    'graded':behavior['graded'],'successes':behavior['successes'],'planned':21})
            write(DEST/'status.json',{'arms':rows,'updated_utc':now(),'complete':done==len(arms)})
            publish_status();publish_viewer()
            if done==len(arms):return
        except Exception as error:write(DEST/'monitor-error.json',{'error':str(error)[-1800:],'updated_utc':now()})
        time.sleep(30)


def start():
    prepare()
    with (DEST/'monitor.log').open('a') as log:
        child=subprocess.Popen([sys.executable,'-u','-m','reproductions.leact.cluster.recover_efe_native_20261101','monitor'],
            cwd=BASE.parents[1],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(DEST/'monitor-launch.json',{'pid':child.pid,'updated_utc':now()})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','start','monitor']);args=parser.parse_args()
    if args.action=='monitor':monitor()
    elif args.action=='start':start()
    else:prepare()
