"""Submit and collect the fixed LeAct round from the 96-update warmstart."""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zlib

import yaml

from .scale_efe import BASE, ROOT, BUILD_ROOT, read, write, submit, command
from .launch_efe_training import freeze_sources
from .monitor_efe_preflight import remote
from .recover_efe_schedule import capacity_interruption, initialization_interruption

PLAN = ROOT / 'training-pilot/leact-warm96-plan.json'
BUILD = BUILD_ROOT / 'efe-leact-warm96-20261029-v1'
FILES = [
    ('efe_leact_pipeline.py', '/tmp/efe_leact_pipeline.py'),
    ('efe_leact_vl.py', '/tmp/efe_leact_vl.py'),
    ('efe_training_pilot.py', '/tmp/efe_training_pilot.py'),
    ('efe_training_data.py', '/tmp/efe_training_data.py'),
    ('efe_train_vl.py', '/tmp/efe_train_vl.py'),
    ('efe_budget_gateway.py', '/tmp/efe_budget_gateway.py'),
    ('text_capability_eval.py', '/tmp/efe_cuda_helpers.py'),
    ('efe_native_pilot.py', '/tmp/efe_native_pilot.py'),
    ('efe_astra_pilot.py', '/tmp/efe-astra-pilot.py'),
    ('efe_qwen_baseline.py', '/tmp/efe-qwen-baseline.py'),
    ('efe_budget_gateway.py', '/tmp/efe-budget-gateway.py'),
    ('export_efe_viewer.py', '/tmp/export_efe_viewer.py'),
]


def now():
    return datetime.now(timezone.utc).isoformat()


def register(build, spec):
    path = ROOT / 'training-pilot/additional-runs.json'
    registry = read(path, {'runs': []})
    run = {'id': 'qwen-warm96-leact-r1', 'name': 'Warmstart step 96 + LeAct R1',
           'baseline': 'qwen-warm-continue32-96', 'category': 'efe-leact',
           'phase': 'LeAct from step-96 warmstart · 50% replay', 'build': str(build),
           'local_results_directory': spec['local_results_directory'],
           'evaluation_directory': spec['local_evaluation_directory'],
           'description': 'One fixed LeAct round: 128 frozen training states, N=8 backward candidates, top 2 positive complete-action likelihood gains, 50% warmstart replay. LR 1e-6, half a balanced pass; 9–12 additional optimizer steps if states are admitted. Same seven-task native evaluation; capability screen paused.'}
    registry['runs'] = [r for r in registry['runs'] if r['id'] != run['id']] + [run]
    write(path, registry)
    for field in ['local_results_directory', 'local_evaluation_directory']:
        dest = ROOT / spec[field]
        dest.mkdir(parents=True, exist_ok=True)
        if not (dest / 'progress.json').exists():
            write(dest / 'progress.json', {'phase': 'waiting', 'detail': run['description']})
    return run


def prepare():
    BUILD.mkdir(exist_ok=True)
    if (BUILD / 'source-manifest.json').exists():
        return BUILD
    original = BUILD_ROOT / 'efe-leact-pilot-20261027-v2'
    spec = read(original / 'spec.json')
    ready = read(ROOT / 'training-pilot/warm-continue32-96-v3/warmstart-ready.json')
    if not ready['final'] or ready['step'] != 96:
        raise ValueError('The completed step-96 checkpoint is required')
    spec = {k: v for k, v in spec.items() if k not in {
        'recovery_of', 'recovery_reason', 'prepared_corpus', 'prepared_dataset',
        'scene_audit', 'teacher_concurrency', 'student_prompt_requests'}}
    spec.update(stage='leact-warm96', base_model=ready['checkpoint'], base_manifest=ready['manifest'],
        base_revision=ready['revision'], archive_root='/tmp/efe-leact-warm96-v1',
        evaluation_archive_root='/tmp/efe-leact-warm96-native-v1',
        local_results_directory='training-pilot/leact-warm96-r1-v1',
        local_evaluation_directory='training-pilot/leact-warm96-r1-evaluation-v1',
        checkpoint_fractions=[1.0], collection_wait_seconds=0, final_collection_wait_seconds=3600,
        node_local_outputs=True, register_capability=False,
        scope='Same frozen EFE LeAct pilot recipe, now initialized from the completed 96-update warmstart. No newly collected task data, no core or capability evaluation.',
        storage_policy='Read frozen inputs from shared storage. Generate/train/evaluate on node-local disk. Retain the complete final portable model on the tracking host before releasing the worker; Adam is node-local. No existing artifact deletion.',
        next='Native evaluation on the same allocation, with the step-96 serving recipe and seven frozen tasks.')
    write(BUILD / 'spec.json', spec)
    native_build = BUILD_ROOT / 'efe-warm-continue32-96-pilot-dev-20261027-v1'
    native = read(native_build / 'spec.json')
    write(BUILD / 'evaluation-template.json', native)
    shutil.copyfile(native_build / 'bootstrap.sh', BUILD / 'native-bootstrap.sh')
    for source, _ in FILES:
        shutil.copyfile(BASE / 'cluster' / source, BUILD / source)
    workflow = yaml.safe_load((original / 'workflow.yaml').read_text())
    workflow['workflow']['name'] = 'leact-efe-warm96-leact-r1'
    workflow['workflow']['timeout'] = {'queue_timeout': '2h', 'exec_timeout': '30h'}
    task = workflow['workflow']['tasks'][0]
    task['command'], task['args'] = ['python3'], ['-u', '/tmp/efe_leact_pipeline.py']
    task['files'] = [{'localpath': name, 'path': path} for name, path in FILES] + [
        {'localpath': 'spec.json', 'path': '/tmp/efe-training-spec.json'},
        {'localpath': 'evaluation-template.json', 'path': '/tmp/efe-evaluation-template.json'},
        {'localpath': 'native-bootstrap.sh', 'path': '/tmp/efe-native-bootstrap.sh'}]
    task.pop('credentials', None)
    (BUILD / 'workflow.yaml').write_text(yaml.safe_dump(workflow, sort_keys=False))
    freeze_sources(BUILD)
    register(BUILD, spec)
    write(PLAN, {'created_utc': now(), 'build': str(BUILD), 'attempt': 1, 'priority': 'HIGH',
        'fallback_priority': 'LOW', 'max_nodes': 1, 'enabled': True,
        'base_revision': ready['revision'], 'capability_screen_enabled': False,
        'selection': {'states': 128, 'candidates_per_state': 8, 'top_positive': 2,
                      'context': 32768, 'candidate_tokens': 16384},
        'optimizer': {'lr': 1e-6, 'balanced_passes': .5, 'effective_batch': 32, 'replay_fraction': .5},
        'storage_policy': spec['storage_policy'],
        'comparison_note': 'The warmstart96 and new LeAct native runs share the same legacy serving recipe. Train/serve image processing and failed continuation semantics remain separately recorded limitations.'})
    return BUILD


def collect(workflow, root, dest, names):
    script = '''from pathlib import Path
import base64,hashlib,json,zlib,time
root=Path(ROOT);files={}
for name in NAMES:
 p=root/name
 if not p.is_file():continue
 data=p.read_bytes()
 if name.endswith('.log'):data=data[-24000:]
 files[name]={'sha256':hashlib.sha256(data).hexdigest(),'data':base64.b64encode(zlib.compress(data)).decode()}
s='EFE_CAPTURE '+json.dumps({'files':files})+'\\n'
for i in range(0,len(s),1000):print(s[i:i+1000],end='',flush=True);time.sleep(.002)
'''.replace('ROOT', repr(root)).replace('NAMES', repr(names))
    result = remote(workflow, script)
    for name, record in result['files'].items():
        data = zlib.decompress(base64.b64decode(record['data']))
        if hashlib.sha256(data).hexdigest() != record['sha256']:
            raise ValueError('Receipt transfer checksum mismatch')
        path = dest / name
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + '.tmp')
        temp.write_bytes(data)
        temp.replace(path)


def download(workflow, remote_path, dest, timeout=1800):
    dest.mkdir(parents=True, exist_ok=True)
    with (dest / 'transfer.log').open('a') as log:
        subprocess.run(['osmo', 'workflow', 'rsync', 'download', workflow, 'preflight',
                        remote_path + ':' + str(dest), '--no-progress', '--timeout', '20'],
                       stdout=log, stderr=subprocess.STDOUT, timeout=timeout, check=True)


def preserve_model(workflow, root, dest):
    ready = read(dest / 'portable-ready.json')
    if not ready or (dest / 'portable-collected.json').exists():
        return
    # Copy in a separate process so evaluation reporting continues during a
    # slow transfer. The collector retains small files across dropped tunnels.
    with (dest / 'model-copy.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
    with (dest / 'model-copy.log').open('a') as log:
        process = subprocess.Popen([sys.executable, '-u', '-m',
            'reproductions.leact.cluster.collect_efe_model_chunks', workflow, str(dest)],
            cwd=BASE.parents[1], stdin=subprocess.DEVNULL, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True)
    write(dest / 'model-copy-launch.json', {'pid': process.pid, 'updated_utc': now()})


def hold_for_collection(workflow, root):
    """Keep the finished worker alive if artifact transfer exceeds its grace period.

    Only the completed pipeline supervisor is paused; this cannot run until its
    final receipt exists, after all training, evaluation and trace export.
    """
    script = '''from pathlib import Path
import json,os,signal,time
root=Path(ROOT)
if not (root/'pipeline-complete.json').exists():raise RuntimeError('Scientific work is not complete')
held=root/'collection-hold.json'
if held.exists():record=json.loads(held.read_text())
else:
 matches=[]
 for entry in Path('/proc').iterdir():
  if not entry.name.isdigit():continue
  try:args=(entry/'cmdline').read_bytes().split(bytes([0]))
  except OSError:continue
  if args and Path(args[0].decode()).name.startswith('python') and b'/tmp/efe_leact_pipeline.py' in args:matches.append(int(entry.name))
 if len(matches)!=1:raise RuntimeError('Cannot identify completed pipeline supervisor')
 pid=matches[0];os.kill(pid,signal.SIGSTOP)
 time.sleep(.1)
 state=(Path('/proc')/str(pid)/'status').read_text()
 if not any(line.startswith('State:') and 'T' in line for line in state.splitlines()):
  raise RuntimeError('Supervisor did not enter collection hold')
 record={'pid':pid,'reason':'Preserve final checkpoint until verified collection'}
 held.write_text(json.dumps(record))
print('EFE_CAPTURE '+json.dumps(record),flush=True)
'''.replace('ROOT', repr(root))
    return remote(workflow, script)


def collect_public_viewer(workflow, root, dest):
    """Publish final videos promptly, independently of the large model copy."""
    receipt = dest / 'public-episodes-collected.json'
    exported = dest / 'public-episodes'
    if not receipt.exists():
        download(workflow, root + '/public-episodes', dest, timeout=300)
        manifest = read(exported / 'manifest.json')
        for episode in manifest['episodes']:
            for member, key in [('trace.json', 'trace_sha256'), ('replay.mp4', 'video_sha256')]:
                path = exported / episode['id'] / member
                if hashlib.sha256(path.read_bytes()).hexdigest() != episode[key]:
                    raise ValueError('Public episode export checksum mismatch: ' + episode['id'])
        write(receipt, {'episodes': len(manifest['episodes']), 'updated_utc': now(),
                        'all_files_verified': True})
    context = ROOT / 'rollout-viewer-v2/viewer-training-context.json'
    original = ROOT / 'rollout-viewer-v1/collected/efe-viewer-export'
    if context.exists() and original.exists() and not (dest / 'viewer-published.json').exists():
        from ..embodied_first_exam.rollout_viewer import build
        output = ROOT / 'efe-rollouts.html'
        build(original, output, exported, context, dest / 'candidates')
        write(dest / 'viewer-published.json', {'updated_utc': now(), 'output': str(output),
            'manifest': str(output.with_suffix('.manifest.json'))})


def watch():
    lock = (PLAN.parent / 'leact-warm96-monitor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = read(PLAN)
    last_bulk = 0
    for _ in range(4000):
        build = Path(plan['build']);spec = read(build / 'spec.json')
        dest = ROOT / spec['local_results_directory']
        evaluation = ROOT / spec['local_evaluation_directory']
        workflow = read(build / 'submission.json')['name']
        root = spec['archive_root'] + '/' + workflow
        native = spec['evaluation_archive_root'] + '/' + workflow
        try:
            state = command(['osmo', 'workflow', 'query', workflow, '--format-type', 'json'])
            write(dest / 'scheduler.json', state);write(evaluation / 'scheduler.json', state)
            if state['status'] == 'RUNNING':
                collect(workflow, root, dest, ['progress.json', 'pipeline-progress.json', 'pipeline-complete.json',
                    'complete.json', 'membership.json', 'selection.json', 'leact-ready.json', 'portable-ready.json',
                    'dataset-manifest.json', 'smoke/complete.json', 'smoke/scoring-parity.json',
                    'train/complete.json', 'train/metrics.jsonl', 'train/schedule.json', 'train/configuration.json',
                    'train/scoring-parity.json', 'smoke.log', 'train.log', 'leact-pipeline.log', 'native-pipeline.log',
                    *[f'worker-{i}.log' for i in range(8)]])
                collect(workflow, native, evaluation, ['progress.json', 'results.json', 'complete.json',
                    'transport.json', 'student-serving.json', 'student-preflight.json', 'student-serving-failure.json'])
                from ..embodied_first_exam.report import publish
                publish()
                preserve_model(workflow, root, dest)
                membership = read(dest / 'membership.json')
                if membership and time.monotonic() - last_bulk > 300:
                    download(workflow, root + '/candidates', dest, timeout=300)
                    last_bulk = time.monotonic()
                pipeline = read(dest / 'pipeline-complete.json')
                if pipeline:
                    awaiting_model = (dest / 'portable-ready.json').exists() and not (dest / 'portable-collected.json').exists()
                    followup = read(ROOT / 'training-evaluation-v1/plan.json', {})
                    awaiting_followup = followup.get('workflow') == workflow and followup.get('retain_allocation', False)
                    if awaiting_model or awaiting_followup:
                        hold = hold_for_collection(workflow, root)
                        write(dest / 'collection-hold.json', {**hold, 'updated_utc': now()})
                    if pipeline.get('graded'):
                        collect_public_viewer(workflow, root, dest)
                    if awaiting_model or awaiting_followup:
                        write(build / 'monitor-status.json', {'workflow': workflow,
                            'status': 'TRAINING_TASK_EVALUATION_ACTIVE' if awaiting_followup else 'WAITING_FOR_MODEL_COLLECTION', 'updated_utc': now()})
                        time.sleep(30)
                        continue
                    if membership:
                        download(workflow, root + '/candidates', dest, timeout=300)
                    remote(workflow, 'from pathlib import Path\nimport json,os,signal\nroot=Path(' + repr(root) + ')\n'
                        '(root/"pipeline-collected").touch()\nheld=root/"collection-hold.json"\n'
                        'if held.exists():os.kill(json.loads(held.read_text())["pid"],signal.SIGCONT)\n'
                        'print(\'EFE_CAPTURE {"collected":true}\',flush=True)')
                    write(dest / 'collection-complete.json', {'workflow': workflow, 'updated_utc': now(),
                        'pipeline': pipeline, 'model_verified': (dest / 'portable-collected.json').exists()})
            from ..embodied_first_exam.report import publish
            publish()
            write(build / 'monitor-status.json', {'workflow': workflow, 'status': state['status'], 'updated_utc': now()})
            if (dest / 'collection-complete.json').exists():
                return
            if state['status'].startswith('FAILED'):
                # Retry allocation failures without regenerating already started
                # scientific work. Completed work requires explicit recovery.
                retryable = capacity_interruption(state) or initialization_interruption(state, dest)
                if retryable and not (dest / 'membership.json').exists():
                    attempt = plan['attempt'] + 1
                    nxt = BUILD.with_name(BUILD.name + '-retry' + str(attempt));nxt.mkdir(exist_ok=True)
                    for name, digest in read(BUILD / 'source-manifest.json').items():
                        if hashlib.sha256((BUILD / name).read_bytes()).hexdigest() != digest:
                            raise ValueError('Frozen original source changed')
                        shutil.copyfile(BUILD / name, nxt / name)
                    updated = read(nxt / 'spec.json')
                    updated['local_results_directory'] += '-retry' + str(attempt)
                    updated['local_evaluation_directory'] += '-retry' + str(attempt)
                    write(nxt / 'spec.json', updated)
                    w = yaml.safe_load((nxt / 'workflow.yaml').read_text())
                    w['workflow']['name'] += '-r' + str(attempt)
                    (nxt / 'workflow.yaml').write_text(yaml.safe_dump(w, sort_keys=False))
                    freeze_sources(nxt)
                    submit(nxt, priority='LOW');register(nxt, updated)
                    plan.update(build=str(nxt), attempt=attempt, priority='LOW');write(PLAN, plan)
                else:
                    write(build / 'monitor-error.json', {'error': 'Worker failed after scientific work began; preserve collected artifacts and diagnose recovery.', 'updated_utc': now()})
                    return
            elif state['status'] == 'COMPLETED':
                write(build / 'monitor-error.json', {'error': 'Allocation completed before collection receipt; inspect retained logs and local artifacts.', 'updated_utc': now()})
                return
        except Exception as error:
            write(build / 'monitor-error.json', {'error': str(error)[-1800:], 'updated_utc': now()})
        time.sleep(30)


def start_monitor():
    with (BUILD / 'monitor.log').open('a') as log:
        process = subprocess.Popen([sys.executable, '-u', '-m', 'reproductions.leact.cluster.launch_efe_leact_warm96', 'monitor'],
            cwd=BASE.parents[1], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write(BUILD / 'monitor-launch.json', {'pid': process.pid, 'updated_utc': now()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'submit', 'monitor'])
    args = parser.parse_args()
    if args.action == 'monitor':
        watch()
    else:
        build = prepare()
        if args.action == 'submit':
            receipt = submit(build, priority='HIGH');start_monitor()
            print(json.dumps({'workflow': receipt['name'], 'build': str(build)}))
        else:
            print(build)
