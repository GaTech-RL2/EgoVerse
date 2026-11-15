"""Resume ungraded training tasks after eviction; retain completed failures."""
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import yaml

from .scale_efe import BASE, ROOT, BUILD_ROOT, read, write, submit, command
from .launch_efe_training import freeze_sources
from .launch_efe_leact_warm96 import collect, download
from .monitor_efe_preflight import remote
from .launch_efe_training_eval import DEST, publish_status, publish_viewer, now

BUILD = BUILD_ROOT / 'efe-training-warm96-resume-20261030-v4'
LANE = '/tmp/efe-training-evaluation-v1/warm96'
LOCAL = DEST / 'warm96-resume-v4'


def prepare():
    BUILD.mkdir(exist_ok=True)
    if (BUILD / 'source-manifest.json').exists():
        return BUILD
    original = BUILD_ROOT / 'efe-training-evaluation-20261029-v1'
    manifest = read(original / 'source-manifest.json')
    for name, digest in manifest.items():
        if hashlib.sha256((original / name).read_bytes()).hexdigest() != digest:
            raise ValueError('Frozen evaluation source changed: ' + name)
    spec = read(original / 'warm96/spec.json')
    graded = {r['task'] for r in read(DEST / 'warm96/behavior.json')['trials']}
    spec['tasks'] = [t for t in spec['tasks'] if t['task'] not in graded]
    assert len(spec['tasks']) == 21-len(graded) and 0 < len(graded) < 21
    spec.update(local_results_directory='training-evaluation-v1/warm96-resume-v4',
                reuse_system_packages=False, recovery_of='leact-efe-warm96-leact-r1-1',
                recovery_reason='Continue only ungraded tasks after recording the ordinary agent timeout as a valid failed outcome. Preserve every existing result; one fresh 800 GiB simulator store.')
    write(BUILD / 'spec.json', spec)
    files = []
    for name in ['efe_native_pilot.py','student.py','pilot.py','efe_budget_gateway.py','export_efe_viewer.py','worker.py']:
        text = (original / 'warm96' / name).read_text()
        if name == 'pilot.py':
            before="valid = bool(trial) and not trial.get('exception_info') and receipt['returncode'] == 0"
            current=(BASE/'cluster/efe_astra_pilot.py').read_text()
            first=current.index('            # A model that uses up its task budget')
            last=current.index("\n            valid &= rewards.get('success')",first)
            text=text.replace('            '+before,current[first:last])
        if name == 'worker.py':
            text = text.replace("if name == 'system_packages':", "if name == 'system_packages' and spec.get('reuse_system_packages', True):")
        (BUILD / name).write_text(text)
        files.append({'localpath':name, 'path':LANE+'/'+name})
    shutil.copyfile(BASE / 'cluster/efe_eval_fresh_bootstrap.py', BUILD / 'bootstrap.py')
    files += [{'localpath':'spec.json','path':LANE+'/spec.json'}, {'localpath':'bootstrap.py','path':'/tmp/efe_eval_bootstrap.py'}]
    workflow = yaml.safe_load((BUILD_ROOT / 'efe-leact-warm96-20261029-v1/workflow.yaml').read_text())
    workflow['workflow']['name'] = 'leact-efe-training-warm96-resume-v4'
    workflow['workflow']['resources']['default']['storage'] = '800Gi'
    workflow['workflow']['timeout'] = {'queue_timeout':'20m','exec_timeout':'24h'}
    task = workflow['workflow']['tasks'][0]
    task.update(command=['python3'], args=['-u','/tmp/efe_eval_bootstrap.py'], files=files)
    (BUILD / 'workflow.yaml').write_text(yaml.safe_dump(workflow, sort_keys=False))
    freeze_sources(BUILD)
    return BUILD


def merge():
    canonical = DEST / 'warm96'
    manifest = read(LOCAL / 'public-episodes/manifest.json', {})
    existing = read(canonical / 'public-episodes/manifest.json', {'episodes':[]})
    rows = {r['task']:r for r in existing['episodes']}
    for row in manifest.get('episodes', []):
        if row['task'] in rows:
            if rows[row['task']]['trace_sha256'] != row['trace_sha256']:
                raise ValueError('Refusing to replace a previously graded task')
            continue
        src = LOCAL / 'public-episodes' / row['id']
        dst = canonical / 'public-episodes' / row['id']
        shutil.copytree(src, dst, dirs_exist_ok=True)
        rows[row['task']] = row
    existing.update(episodes=list(rows.values()), new_rollouts=len(rows), created_utc=now())
    write(canonical / 'public-episodes/manifest.json', existing)
    behavior = read(canonical / 'behavior.json')
    trials = {t['task']:t for t in behavior['trials']}
    for t in read(LOCAL / 'behavior.json', {}).get('trials', []):
        trials.setdefault(t['task'], t)
    behavior.update(trials=list(trials.values()), graded=len(trials), successes=sum(t['score'] for t in trials.values()),
                    success_rate=sum(t['score'] for t in trials.values())/len(trials),
                    matched_initial_states=sum(bool(t.get('initial_state_matches_teacher')) for t in trials.values()), updated_utc=now())
    write(canonical / 'behavior.json', behavior)
    progress = read(LOCAL / 'progress.json', {})
    if progress:
        progress.update(completed_trials=len(trials), planned_trials=21)
        write(canonical / 'progress.json', progress)
    if len(trials) == 21:
        write(canonical / 'worker-complete.json', {'success':True,'graded':21,'successes':behavior['successes'],
              'provenance':'Previously graded tasks plus disjoint resumed tasks; identical checkpoint manifest. Unavailable initial-state recordings remain unknown.', 'completed_utc':now()})


def monitor():
    LOCAL.mkdir(exist_ok=True)
    lock = (LOCAL / 'monitor.lock').open('a');fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    build = Path(read(LOCAL/'active-build.json', {}).get('build', str(BUILD)))
    retry_number = int(read(LOCAL/'active-build.json', {}).get('retry_number', 0))
    while True:
        try:
            workflow = read(build / 'submission.json')['name']
            state = command(['osmo','workflow','query',workflow,'--format-type','json'])
            write(LOCAL / 'scheduler.json', state)
            plan = read(DEST / 'plan.json');plan.update(status='recovery_in_progress', recovery_workflow=workflow, updated_utc=now());write(DEST/'plan.json',plan)
            if state['status'] == 'RUNNING':
                root = LANE + '/results/' + workflow
                collect(workflow, root, LOCAL, ['worker-start.json','worker-complete.json','progress.json','results.json',
                    'complete.json','behavior.json','student-serving.json','student-preflight.json','student-serving-failure.json',
                    'setup-error.json','public-episodes/manifest.json'])
                manifest = read(LOCAL/'public-episodes/manifest.json', {})
                for row in manifest.get('episodes', []):
                    def valid():
                        return all((LOCAL/'public-episodes'/row['id']/name).exists() and
                            hashlib.sha256((LOCAL/'public-episodes'/row['id']/name).read_bytes()).hexdigest()==row[key]
                            for name,key in [('trace.json','trace_sha256'),('replay.mp4','video_sha256')])
                    if not valid():download(workflow,root+'/public-episodes/'+row['id'],LOCAL/'public-episodes',timeout=120)
                    if not valid():raise ValueError('Resumed media checksum mismatch')
                merge()
                if read(LOCAL/'complete.json'):
                    remote(workflow,"from pathlib import Path\nPath("+repr(LANE+'/runtime/collected')+").touch()\nprint('EFE_CAPTURE {\"ack\":true}',flush=True)")
                publish_status();publish_viewer()
                if read(LOCAL/'worker-complete.json'):
                    write(LOCAL/'monitor-status.json', {'complete':True,'worker':read(LOCAL/'worker-complete.json'),'updated_utc':now()})
                    return
            elif state['status'] == 'COMPLETED':
                complete = read(LOCAL/'worker-complete.json', {})
                write(LOCAL/'monitor-status.json', {'workflow':workflow,'status':state['status'],
                    'complete': bool(complete.get('success')),
                    'message':'Scheduler ended; experiment completion requires a collected successful worker receipt.',
                    'updated_utc':now()})
                return
            elif state['status'].startswith('FAILED'):
                if not read(LOCAL/'worker-start.json') and state['status'] in ['FAILED_QUEUE_TIMEOUT','FAILED_PREEMPTED','FAILED_START_ERROR','FAILED_START_TIMEOUT']:
                    retry_number += 1
                    retry = BUILD.with_name(BUILD.name+'-low'+str(retry_number));retry.mkdir(exist_ok=True)
                    if not (retry/'source-manifest.json').exists():
                        for name in read(BUILD/'source-manifest.json'):shutil.copyfile(BUILD/name,retry/name)
                        w=yaml.safe_load((retry/'workflow.yaml').read_text());w['workflow']['name']+='-low'+str(retry_number);w['workflow']['timeout']['queue_timeout']='12h';(retry/'workflow.yaml').write_text(yaml.safe_dump(w,sort_keys=False));freeze_sources(retry)
                    submit(retry,priority='LOW');build=retry
                    write(LOCAL/'active-build.json', {'build':str(build),'retry_number':retry_number,'updated_utc':now()})
                else:
                    write(LOCAL/'monitor-error.json',{'scheduler_status':state['status'],'updated_utc':now()});return
            write(LOCAL/'monitor-status.json',{'workflow':workflow,'status':state['status'],'updated_utc':now()})
        except Exception as e:
            write(LOCAL/'monitor-error.json',{'error':str(e)[-1800:],'updated_utc':now()})
        time.sleep(30)


def start_monitor():
    LOCAL.mkdir(exist_ok=True)
    with (LOCAL/'monitor.log').open('a') as log:
        p=subprocess.Popen([sys.executable,'-u','-m','reproductions.leact.cluster.recover_efe_training_evals','monitor'],
             cwd=BASE.parents[1],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(LOCAL/'monitor-launch.json',{'pid':p.pid,'updated_utc':now()})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','submit','monitor']);args=parser.parse_args()
    if args.action=='monitor':monitor()
    else:
        prepare()
        if args.action=='submit':print(json.dumps(submit(BUILD,priority='HIGH')));start_monitor()
