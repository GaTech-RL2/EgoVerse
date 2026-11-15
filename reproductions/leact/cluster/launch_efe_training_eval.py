"""Launch and collect two native training-task evaluations on the retained node."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import time

from .scale_efe import BASE, ROOT, BUILD_ROOT, read, write
from .monitor_efe_preflight import remote
from .launch_efe_leact_warm96 import collect, download

WORKFLOW = 'leact-efe-warm96-leact-r1-1'
REMOTE = '/tmp/efe-training-evaluation-v1'
BUILD = BUILD_ROOT / 'efe-training-evaluation-20261029-v1'
DEST = ROOT / 'training-evaluation-v1'
MODELS = ['warm96', 'leact']


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    BUILD.mkdir(exist_ok=True)
    if (BUILD / 'source-manifest.json').exists():
        return
    original = BUILD_ROOT / 'efe-leact-warm96-20261029-v1'
    frozen = read(original / 'source-manifest.json')
    for name in ['efe_native_pilot.py', 'efe_qwen_baseline.py', 'efe_astra_pilot.py', 'efe_budget_gateway.py']:
        if sha(original / name) != frozen[name]:
            raise ValueError('Frozen development source changed: ' + name)
    template = read(original / 'evaluation-template.json')
    inventory = read(ROOT / 'generalization-v2/inventory.json')
    if sha(ROOT / 'generalization-v2/inventory.json') != template['inventory_sha256']:
        raise ValueError('Frozen task inventory changed')
    teacher_root = ROOT / 'rollout-viewer-v3/collected/training-viewer-v3-export'
    teachers = read(teacher_root / 'manifest.json')
    tasks = []
    for row in teachers['episodes']:
        trace_path = teacher_root / row['id'] / 'trace.json'
        if sha(trace_path) != row['trace_sha256']:
            raise ValueError('Training teacher trace changed')
        trace = read(trace_path)
        task = next(t for t in inventory if t['task'] == row['task'])
        if task['seeds']['allowed'] != [0] or task['seeds']['default'] != 0:
            raise ValueError('Training task needs explicit seed handling')
        initial = next(e['state_sha'] for e in trace['simulator_events'] if e.get('ev') == 'start')
        tasks.append({**task, 'partition':'training_evaluation', 'teacher_initial_state_sha256':initial,
            'source_teacher_episode':trace['source_episode'], 'teacher_trace_sha256':row['trace_sha256'],
            'retained_training_decisions':row['retained_decisions'],
            'teacher_elapsed_s':trace['resource_metrics'].get('elapsed_s', 1e9)})
    if len(tasks) != 21 or len({t['task'] for t in tasks}) != 21:
        raise ValueError('Expected the exact 21 admitted training task instances')
    # Short teacher episodes first within each suite; rotate suites for early coverage.
    groups = [[t for t in tasks if t['suite'] == suite] for suite in ['metaworldplus','robocasa','embodied-safety-exam']]
    for group in groups:
        group.sort(key=lambda t:(t['teacher_elapsed_s'], t['task']))
    tasks = [group[i] for i in range(max(map(len,groups))) for group in groups if i < len(group)]
    warm = read(ROOT / 'training-pilot/warm-continue32-96-v3/warmstart-ready.json')
    leact = read(ROOT / 'training-pilot/leact-warm96-r1-v1/portable-ready.json')
    for index, model in enumerate(MODELS):
        folder = BUILD / model
        folder.mkdir(exist_ok=True)
        lane = REMOTE + '/' + model
        runtime_root = lane + '/runtime'
        server_port, probe_port, gateway_port = (19880,19862,19861) if model == 'warm96' else (20880,20862,20861)
        checkpoint = warm if model == 'warm96' else leact
        spec = {**template, 'tasks':tasks, 'model_id':model,
            'student_checkpoint':checkpoint['checkpoint'], 'student_manifest':checkpoint['manifest'],
            'student_revision':checkpoint['revision'], 'archive_root':lane + '/results',
            'local_results_directory':'training-evaluation-v1/' + model,
            'upstream':f'http://127.0.0.1:{server_port}/v1', 'gateway_script':lane + '/efe_budget_gateway.py',
            'agent_label':'Qwen warmstart 96' if model == 'warm96' else 'Qwen warmstart 96 + LeAct R1',
            'agent_slug':'qwen-training-' + model, 'export_role':'training_' + model,
            'purpose':'Closed-loop success and behavior on the exact 21 task instances that supplied the warmstart corpus; no new optimization.',
            'training_admission':'Evaluation only. These new student rollouts do not enter any training dataset.',
            'comparison_to_development':'Same native prompts, limits, image path, TP=4, sampling, checkpoint and grading as the seven development tasks. Only task membership and process isolation change.',
            'scope':'In-distribution training-task diagnostic, one attempt per checkpoint/task, seed 0. Successful Astra source episodes were selected for the corpus; their 21/21 is not an unbiased teacher benchmark rate.',
            'execution':{'gpu_ids':'0,1,2,3' if model == 'warm96' else '4,5,6,7',
                         'server_port':server_port, 'gateway_port':gateway_port, 'probe_port':probe_port},
            'retry_of':{}}
        for key in ['comparison_to_step32','comparison_to_v4','resume_completed_trials']:
            spec.pop(key, None)
        write(folder / 'spec.json', spec)
        runtime = (original / 'efe_native_pilot.py').read_text()
        runtime = runtime.replace("Path('/tmp/efe-native-spec.json')", 'Path(' + repr(lane + '/spec.json') + ')')
        runtime = runtime.replace('/tmp/efe-native', runtime_root)
        (folder / 'efe_native_pilot.py').write_text(runtime)
        student = (original / 'efe_qwen_baseline.py').read_text()
        student = student.replace('/tmp/efe-astra-pilot.py', lane + '/pilot.py').replace('/tmp/efe-budget-gateway.py',lane + '/efe_budget_gateway.py')
        student = student.replace('18880',str(server_port)).replace('18862',str(probe_port)).replace('29880',str(29880 + index * 1000))
        student = student.replace("'CUDA_VISIBLE_DEVICES':'0,1,2,3'", "'CUDA_VISIBLE_DEVICES':" + repr(spec['execution']['gpu_ids']))
        (folder / 'student.py').write_text(student)
        pilot = (original / 'efe_astra_pilot.py').read_text().replace('port = 18861', 'port = ' + str(gateway_port))
        pilot = pilot.replace('/tmp/efe-budget-gateway.py',lane + '/efe_budget_gateway.py')
        (folder / 'pilot.py').write_text(pilot)
        (folder / 'efe_budget_gateway.py').write_bytes((original / 'efe_budget_gateway.py').read_bytes())
        (folder / 'export_efe_viewer.py').write_bytes((BASE / 'cluster/export_efe_viewer.py').read_bytes())
        (folder / 'worker.py').write_bytes((BASE / 'cluster/efe_training_eval_worker.py').read_bytes())
        for path in folder.glob('*.py'):
            compile(path.read_text(), str(path), 'exec')
    write(BUILD / 'source-manifest.json', {str(p.relative_to(BUILD)):sha(p) for p in BUILD.rglob('*') if p.is_file()})
    plan = {**read(DEST / 'plan.json', {}), 'created_utc':now(), 'workflow':WORKFLOW, 'retain_allocation':True,
        'status':'prepared', 'build':str(BUILD), 'remote_root':REMOTE, 'nodes':1, 'gpus_per_model':4,
        'task_count':21, 'total_new_episodes':42, 'models':MODELS, 'repetitions_per_task':1,
        'training_teacher_collection_sha256':sha(teacher_root / 'manifest.json'),
        'partition':'training_evaluation', 'new_training':False, 'capability_screen_enabled':False,
        'tasks':[{'task':t['task'], 'suite':t['suite'], 'source_teacher_episode':t['source_teacher_episode'],
                  'initial_state_sha256':t['teacher_initial_state_sha256']} for t in tasks]}
    write(DEST / 'plan.json', plan)


def launch():
    prepare()
    if (DEST / 'launch.json').exists():
        return read(DEST / 'launch.json')
    manifest = read(BUILD / 'source-manifest.json')
    for name, digest in manifest.items():
        if sha(BUILD / name) != digest:
            raise ValueError('Frozen launch member changed: ' + name)
    bundle = DEST / 'efe-training-evaluation-v1.tar.gz'
    with tarfile.open(bundle, 'w:gz') as archive:
        for name in manifest:
            archive.add(BUILD / name, arcname=name)
    with (DEST / 'upload.log').open('a') as log:
        subprocess.run(['osmo','workflow','rsync','upload',WORKFLOW,'preflight',str(bundle.resolve())+':/tmp',
            '--no-progress','--timeout','20'],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
    script = ('import json,pathlib,subprocess,sys,os,tarfile,hashlib,re\nroot=pathlib.Path(' + repr(REMOTE) +
              ')\nmanifest=' + repr(manifest) + '\nworkflow=' + repr(WORKFLOW) + '\nbundle_sha=' + repr(sha(bundle)) + '\n') + '''
if (root/'launch.json').exists():
 print('EFE_CAPTURE '+(root/'launch.json').read_text(),flush=True)
else:
 bundle=pathlib.Path('/tmp/efe-training-evaluation-v1.tar.gz')
 if hashlib.sha256(bundle.read_bytes()).hexdigest()!=bundle_sha:raise ValueError('Uploaded bundle checksum mismatch')
 root.mkdir(parents=True,exist_ok=True)
 with tarfile.open(bundle) as archive:archive.extractall(root,filter='data')
 for name,digest in manifest.items():
  if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise ValueError('Uploaded launch member mismatch')
 version=re.search(r'([0-9]{3}[.][0-9]+(?:[.][0-9]+)?)',pathlib.Path('/proc/driver/nvidia/version').read_text()).group(1)
 libraries=[pathlib.Path('/usr/lib/x86_64-linux-gnu')/(name+'.'+version) for name in ['libcuda.so','libnvidia-ptxjitcompiler.so','libnvidia-nvvm.so']]
 if not all(p.is_file() for p in libraries):raise RuntimeError('Matching CUDA driver/compiler libraries missing')
 cuda_env={'LD_PRELOAD':':'.join(map(str,libraries)),'LD_LIBRARY_PATH':'/usr/lib/x86_64-linux-gnu:'+os.environ.get('LD_LIBRARY_PATH','')}
 (root/'launch-cuda-environment.json').write_text(json.dumps(cuda_env))
 rows=[]
 for model in ['warm96','leact']:
  lane=root/model
  with (lane/'worker.log').open('a') as log:
   process=subprocess.Popen([sys.executable,'-u',str(lane/'worker.py')],env={**os.environ,**cuda_env,'EFE_WORKFLOW_ID':workflow},stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  rows.append({'model':model,'pid':process.pid,'root':str(lane)})
 result={'workflow':workflow,'workers':rows}
 (root/'launch.json').write_text(json.dumps(result))
 print('EFE_CAPTURE '+json.dumps(result),flush=True)
'''
    receipt = remote(WORKFLOW, script)
    write(DEST / 'launch.json', {**receipt, 'launched_utc':now()})
    plan = read(DEST / 'plan.json');plan.update(status='running', updated_utc=now());write(DEST / 'plan.json',plan)
    return receipt


def publish_status():
    plan = read(DEST / 'plan.json')
    rows = []
    for model in plan.get('models', MODELS):
        path = DEST / model
        behavior = read(path / 'behavior.json', {})
        progress = read(path / 'progress.json', {})
        complete = read(path / 'worker-complete.json', {})
        progress = {**progress, 'detail':progress.get('detail','').replace('native development evaluation','native training-task evaluation')}
        trials = behavior.get('trials', [])
        suites = []
        for suite in ['metaworldplus','robocasa','embodied-safety-exam']:
            subset = [t for t in trials if t['suite'] == suite]
            suites.append({'suite':suite, 'planned':sum(t['suite'] == suite for t in plan['tasks']),
                'graded':len(subset), 'successes':sum(t['score'] for t in subset),
                'episodes_with_control':sum(t['control_steps'] > 0 for t in subset),
                'control_steps':sum(t['control_steps'] for t in subset)})
        moving = sum(t['control_steps'] > 0 for t in trials)
        observing = sum(t['control_steps'] == 0 and t['simulator_calls'] > 0 for t in trials)
        behavior_note = (f'Of {len(trials)} graded episodes, {moving} commanded robot-control steps, '
                         f'{observing} called the simulator without commanding control, and '
                         f'{len(trials)-moving-observing} made no simulator calls.') if trials else 'No completed training-task rollouts yet; success rate and behavior are not measured.'
        label = plan.get('model_labels', {}).get(model, 'Warmstart 96' if model == 'warm96' else 'Warmstart 96 + LeAct')
        rows.append({'id':model, 'label':label,
            'planned':21, 'graded':behavior.get('graded',0), 'successes':behavior.get('successes',0),
            'success_rate':behavior.get('success_rate'), 'matched_initial_states':behavior.get('matched_initial_states',0),
            'progress':progress, 'complete':complete, 'trials':trials, 'suites':suites, 'behavior_note':behavior_note})
    status = {'updated_utc':now(), 'workflow':plan.get('recovery_workflow', WORKFLOW), 'models':rows, 'planned_per_model':21,
        'scope':'Qwen acting freely on the exact tasks and initial states used to collect the 21 admitted Astra training episodes. One rollout per checkpoint/task. Kept separate from held-out development results. ' + plan.get('recovery_note',''),
        'serving_caveat':'The native model settings are unchanged from development. Known training/serving image and history differences still apply. The first two episodes shared a node; recovery uses separate nodes. This is not a matched latency experiment.',
        'status':plan['status']}
    write(DEST / 'status.json', status)
    from ..embodied_first_exam.report import publish
    publish()


def collect_episodes(model):
    dest = DEST / model
    manifest = read(dest / 'public-episodes/manifest.json', {})
    for row in manifest.get('episodes', []):
        def valid():
            return all((dest / 'public-episodes' / row['id'] / member).is_file() and
                sha(dest / 'public-episodes' / row['id'] / member) == row[key]
                for member,key in [('trace.json','trace_sha256'),('replay.mp4','video_sha256')])
        if not valid():
            download(WORKFLOW, REMOTE+'/'+model+'/results/'+WORKFLOW+'/public-episodes/'+row['id'],
                     dest / 'public-episodes', timeout=120)
        if not valid():
            raise ValueError('Training evaluation media checksum mismatch')
    return len(manifest.get('episodes', []))


def publish_viewer():
    manifests = [read(DEST / model / 'public-episodes/manifest.json', {}) for model in read(DEST/'plan.json').get('models', MODELS)]
    signature = hashlib.sha256(json.dumps({'episodes':[m.get('episodes',[]) for m in manifests],
        'renderer':sha(BASE / 'embodied_first_exam/rollout_viewer.py'),
        'template':sha(BASE / 'embodied_first_exam/rollout_viewer.html')},sort_keys=True).encode()).hexdigest()
    if read(DEST / 'viewer-published.json', {}).get('signature') == signature:
        return
    from ..embodied_first_exam.rollout_viewer import build
    leact = ROOT / 'training-pilot/leact-warm96-r1-v1/public-episodes'
    if not (leact / 'manifest.json').exists():
        leact = ROOT / 'rollout-viewer-v2/collected/viewer-v2-export'
    build(ROOT / 'rollout-viewer-v1/collected/efe-viewer-export', ROOT / 'efe-rollouts.html', leact,
          ROOT / 'rollout-viewer-v2/viewer-training-context.json',
          ROOT / 'training-pilot/leact-warm96-r1-v1/candidates')
    write(DEST / 'viewer-published.json', {'signature':signature, 'updated_utc':now(),
          'student_training_videos':sum(len(m.get('episodes',[])) for m in manifests)})


def watch():
    lock = (DEST / 'monitor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        try:
            complete = True
            for model in MODELS:
                dest = DEST / model;dest.mkdir(exist_ok=True)
                root = REMOTE+'/'+model+'/results/'+WORKFLOW
                collect(WORKFLOW,root,dest,['worker-start.json','worker-complete.json','progress.json','results.json',
                    'complete.json','behavior.json','student-serving.json','student-preflight.json',
                    'student-serving-failure.json','public-episodes/manifest.json'])
                done = read(dest / 'worker-complete.json', {})
                complete &= bool(done)
                if read(dest / 'complete.json'):
                    remote(WORKFLOW, 'from pathlib import Path\nPath('+repr(REMOTE+'/'+model+'/runtime/collected')+').touch()\nprint(\'EFE_CAPTURE {"ack":true}\',flush=True)')
                count = collect_episodes(model)
                if done:
                    if count != read(dest / 'behavior.json',{}).get('graded',0):
                        raise ValueError('Some completed training evaluation recordings have not been collected')
            if complete:
                succeeded = all(read(DEST/m/'worker-complete.json')['success'] for m in MODELS)
                plan=read(DEST / 'plan.json');plan.update(status='complete' if succeeded else 'needs_review', retain_allocation=not succeeded, completed_utc=now());write(DEST / 'plan.json',plan)
            publish_status()
            publish_viewer()
            write(DEST / 'monitor-status.json', {'updated_utc':now(), 'complete':complete})
            if complete:
                return
        except Exception as error:
            write(DEST / 'monitor-error.json', {'updated_utc':now(), 'error':str(error)[-1800:]})
        time.sleep(30)


def start_monitor():
    with (DEST / 'monitor.log').open('a') as log:
        process=subprocess.Popen([sys.executable,'-u','-m','reproductions.leact.cluster.launch_efe_training_eval','monitor'],
            cwd=BASE.parents[1],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(DEST / 'monitor-launch.json', {'pid':process.pid, 'updated_utc':now()})


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['prepare','launch','monitor'])
    args=parser.parse_args();DEST.mkdir(exist_ok=True)
    if args.action == 'monitor':watch()
    elif args.action == 'prepare':prepare();print(BUILD)
    else:print(json.dumps(launch()));publish_status();start_monitor()
