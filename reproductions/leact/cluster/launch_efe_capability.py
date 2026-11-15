"""Launch/collect the optional capability screen without changing the EFE task queue."""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tarfile
import time

import yaml
import pexpect

from .monitor_efe_preflight import remote
from .scale_efe import BUILD_ROOT, command, submit
from .compute_budget import require_capacity
from ..embodied_first_exam import report
from ..embodied_first_exam.report import BASE, read, write
from ..embodied_first_exam.capability_retention import DEST, prepare, sha
from ..embodied_first_exam.capability_score import paired_comparison

SHARED = '/mnt/amlfs-03/home/rpunamiya/leact/studies/efe-capability-retention-v1'


def execution_enabled():
    return read(DEST/'execution-policy.json',{}).get('enabled',True)


def reference_ids(stage, registry):
    family = registry.get('families', {}).get(stage.get('family'), {})
    return {stage.get('parent'), family.get('base_checkpoint', 'qwen-base')} - {None, stage['id']}


def build_run(stage_id):
    plan = prepare()
    registry = read(DEST / 'checkpoints.json')
    stage = next(r for r in registry['checkpoints'] if r['id'] == stage_id)
    family = registry.get('families', {}).get(stage.get('family'), {})
    if family:
        family_plan = read(DEST / family['suite'])
        if family_plan['file_sha256'] != plan['file_sha256']:
            raise ValueError('Capability families must use the same frozen evaluation inputs')
        plan = family_plan
    if stage['status'] not in {'ready', 'submitted'}:
        raise ValueError('Checkpoint does not exist yet: ' + stage_id)
    build = BUILD_ROOT / ('efe-capability-' + stage_id + '-' + stage.get('build_version', 'v1'))
    build.mkdir(exist_ok=True)
    if (build / 'source-manifest.json').exists():
        return build
    bundle = DEST / family.get('bundle', 'frozen-inputs.tar.gz')
    if not bundle.exists():
        with tarfile.open(bundle, 'w:gz') as tar:
            tar.add(DEST / family.get('suite', 'suite.json'), arcname='suite.json')
            for name in ['inputs', 'vendor']:
                tar.add(DEST / name, arcname=name)
    for source, target in [(family.get('runner', 'cluster/efe_capability_eval.py'), 'runner.py'),
                            ('embodied_first_exam/capability_score.py', 'score.py'),
                            ('cluster/efe_capability_scorer.Dockerfile', 'Dockerfile')]:
        shutil.copyfile(BASE / source, build / target)
    protocol = {'suite_sha256': sha(DEST / family.get('suite', 'suite.json')),
                'sources': {name: sha(build / name) for name in ['runner.py', 'score.py', 'Dockerfile']},
                'serving_image': 'docker.io/vllm/vllm-openai@sha256:4801151759655c57606c844662e5213403c032a62d149c7ce61d615759a821ef'}
    protocol_sha = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    bundle_sha = sha(bundle)
    base_id = family.get('base_checkpoint', 'qwen-base')
    base = next(r for r in registry['checkpoints'] if r['id'] == base_id)
    original = read(BUILD_ROOT / 'efe-qwen-base-20261026-v4/spec.json')
    spec = {'checkpoint': stage, 'base_path': base['path'], 'base_manifest': base['manifest'],
            'shared_root': SHARED, 'bundle_path': SHARED + '/inputs/' + bundle_sha + '.tar.gz',
            'bundle_sha256': bundle_sha, 'protocol': protocol, 'protocol_sha256': protocol_sha,
            'podman': original['podman'], 'local_results_directory': 'capability-retention/runs/' + stage_id}
    if family:
        spec.update(comparison_base=base_id, family=stage['family'], evaluation_model=plan['model'],
                    local_bundle=str(bundle), text_only=family.get('text_only', False))
    write(build / 'spec.json', spec)
    shutil.copyfile(BASE / 'cluster/efe_capability_storage.py', build / 'storage.py')
    spec['storage_adapter_sha256'] = sha(build / 'storage.py')
    spec['storage_note'] = 'Node-local grader output, archived by the workflow UID. Frozen runner, scorer and comparison protocol unchanged.'
    write(build / 'spec.json', spec)
    (build / 'bootstrap.sh').write_text('set -euo pipefail\npython3 /tmp/efe-capability-storage.py\n')
    workflow = {'workflow': {'name': 'leact-efe-capability-' + stage_id,
                'resources': {'default': {'gpu': 8, 'cpu': 32, 'memory': '256Gi', 'storage': '256Gi', 'platform': 'dgx-h100'}},
                'timeout': {'queue_timeout': '2h', 'exec_timeout': '3h'},
                'tasks': [{'name': 'preflight', 'image': protocol['serving_image'], 'privileged': True,
                           'command': ['bash'], 'args': ['/tmp/efe-capability-bootstrap.sh'],
                           'environment': {'EFE_WORKFLOW_ID': '{{workflow_id}}', 'PYTHONUNBUFFERED': '1',
                                           'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '2',
                                           'NVIDIA_DRIVER_CAPABILITIES': 'all'},
                           'files': [{'localpath': a, 'path': b} for a, b in [
                               ('bootstrap.sh', '/tmp/efe-capability-bootstrap.sh'),
                               ('runner.py', '/tmp/efe-capability-runner.py'), ('score.py', '/tmp/efe-capability-score.py'),
                               ('storage.py', '/tmp/efe-capability-storage.py'),
                               ('Dockerfile', '/tmp/efe-capability-Dockerfile'), ('spec.json', '/tmp/efe-capability-spec.json')]]}]}}
    (build / 'workflow.yaml').write_text(yaml.safe_dump(workflow, sort_keys=False))
    write(build / 'source-manifest.json', {p.name: sha(p) for p in build.iterdir() if p.is_file()})
    return build


def transfer(build, via_workflow):
    spec = read(build / 'spec.json')
    local_bundle = Path(spec.get('local_bundle', str(DEST / 'frozen-inputs.tar.gz')))
    capture = remote(via_workflow, 'from pathlib import Path;import json;'
                     + 'p=Path(' + repr(spec['bundle_path']) + ');p.parent.mkdir(parents=True,exist_ok=True);'
                     + 'print("EFE_CAPTURE "+json.dumps({"exists":p.exists()}),flush=True)')
    if not capture['exists']:
        # This deployment denied the rsync endpoint during the original base
        # screen; its destination is also a directory rather than a filename.
        # Exec upload is already verified and checks the complete byte stream.
        upload_exec(via_workflow, local_bundle, spec['bundle_path'], spec['bundle_sha256'])
    captured = remote(via_workflow, 'from pathlib import Path;import json,hashlib;'
                      + 'p=Path(' + repr(spec['bundle_path']) + ');'
                      + 'print("EFE_CAPTURE "+json.dumps({"sha256":hashlib.sha256(p.read_bytes()).hexdigest()}),flush=True)')
    if captured['sha256'] != spec['bundle_sha256']:
        raise ValueError('Uploaded frozen input checksum mismatch')
    write(build / 'input-transfer.json', {'via_workflow': via_workflow, **captured})


def upload_exec(workflow, source, destination, digest):
    script = '''import sys,termios,base64,hashlib,json
from pathlib import Path
attrs=termios.tcgetattr(0);attrs[3]&=~(termios.ECHO|termios.ICANON);termios.tcsetattr(0,termios.TCSANOW,attrs)
p=Path(DESTINATION);tmp=p.with_suffix('.upload');h=hashlib.sha256()
print('CAPABILITY_UPLOAD_READY',flush=True)
with tmp.open('wb') as out:
 while True:
  line=sys.stdin.readline().strip()
  if line=='CAPABILITY_UPLOAD_END':break
  if not line:raise RuntimeError('Upload stream ended before marker')
  chunk=base64.b64decode(line,validate=True);out.write(chunk);h.update(chunk)
if h.hexdigest()!=EXPECTED:raise ValueError('Upload digest mismatch')
tmp.replace(p)
print('CAPABILITY_UPLOAD_DONE '+h.hexdigest(),flush=True)
'''.replace('DESTINATION', repr(destination)).replace('EXPECTED', repr(digest))
    child = pexpect.spawn('osmo', ['workflow', 'exec', workflow, 'preflight', '--entry',
                                 'python3 -c ' + shlex.quote(script)], encoding='utf-8', timeout=45, echo=False)
    child.delaybeforesend = None
    try:
        child.expect('CAPABILITY_UPLOAD_READY')
        with source.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1500), b''):
                child.sendline(base64.b64encode(chunk).decode())
                time.sleep(.003)
        child.sendline('CAPABILITY_UPLOAD_END')
        child.expect('CAPABILITY_UPLOAD_DONE ' + digest)
        child.expect(pexpect.EOF, timeout=15)
    finally:
        child.close(force=True)


def launch_monitor(build):
    with (build / 'monitor.log').open('a') as log:
        proc = subprocess.Popen([sys.executable, '-u', '-m', 'reproductions.leact.cluster.launch_efe_capability', 'monitor', '--build', str(build)],
                                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True, cwd=BASE.parents[1])
    write(build / 'monitor-launch.json', {'pid': proc.pid, 'updated_utc': datetime.now(timezone.utc).isoformat()})


def monitor(build):
    lock = (build / 'monitor.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    spec = read(build / 'spec.json')
    workflow = read(build / 'submission.json')['name']
    dest = DEST / 'runs' / spec['checkpoint']['id']; dest.mkdir(parents=True, exist_ok=True)
    names = ['progress.json', 'results.json', 'complete.json', 'checkpoint.json', 'generation-preflight.json', 'serving.json', 'cuda-preflight.json']
    for _ in range(720):
        try:
            query = subprocess.run(['osmo', 'workflow', 'query', workflow, '--format-type', 'json'],
                                   text=True, capture_output=True, check=True, timeout=35)
            state = json.loads(query.stdout[query.stdout.index('{'):])
            write(dest / 'scheduler.json', {'status': state['status'], 'workflow': workflow,
                                            'updated_utc': datetime.now(timezone.utc).isoformat()})
            via=workflow if state['status']=='RUNNING' else None
            terminal=state['status']=='COMPLETED' or state['status'].startswith('FAILED')
            if via is None and terminal and not (dest/'complete.json').exists():
                active=command(['osmo','workflow','list','--name','leact','--status','RUNNING','--format-type','json'])['workflows']
                via=next((row['name'] for row in active if row['user']=='rpunamiya@nvidia.com'),None)
                if via is None:
                    from .launch_efe_training import recovery_reader
                    via=recovery_reader(build)
            if via:
                script = '''from pathlib import Path
import base64,hashlib,json,time
root=Path(ARCHIVE);files={}
for name in NAMES:
 p=root/name
 if p.exists():
  d=p.read_bytes();files[name]={'sha256':hashlib.sha256(d).hexdigest(),'data':base64.b64encode(d).decode()}
s='EFE_CAPTURE '+json.dumps({'files':files})+'\\n'
for i in range(0,len(s),1000):print(s[i:i+1000],end='',flush=True);time.sleep(.02)
'''.replace('NAMES', repr(names)).replace('ARCHIVE',repr(spec['shared_root']+'/runs/'+workflow))
                capture = remote(via, script)
                for name, record in capture['files'].items():
                    if name not in names: raise ValueError('Unexpected result file')
                    data = base64.b64decode(record['data'])
                    if hashlib.sha256(data).hexdigest() != record['sha256']:
                        raise ValueError('Result checksum mismatch')
                    write(dest / name, json.loads(data))
                if (dest / 'results.json').exists():
                    result = read(dest / 'results.json')
                    comparisons = {}
                    for parent in dict.fromkeys([spec.get('comparison_base', 'qwen-base'), spec['checkpoint'].get('parent')]):
                        if not parent or parent == spec['checkpoint']['id']: continue
                        prior = read(DEST / 'runs' / parent / 'results.json')
                        if prior:
                            comparisons[parent] = paired_comparison(prior, result)
                    write(dest / 'comparisons.json', comparisons)
                if (dest / 'complete.json').exists() and via==workflow:
                    remote(workflow, 'from pathlib import Path;Path("/tmp/efe-capability/collected").touch();print(\'EFE_CAPTURE {"ack":true}\',flush=True)')
                elif (dest/'complete.json').exists() and via==read(build/'completion-reader/submission.json',{}).get('name'):
                    remote(via,'from pathlib import Path;Path("/tmp/efe-reader-collected").touch();print(\'EFE_CAPTURE {"ack":true}\',flush=True)')
            importlib.reload(report).publish()
            write(build / 'monitor-status.json', {'workflow': workflow, 'status': state['status'],
                                                   'updated_utc': datetime.now(timezone.utc).isoformat()})
            if terminal and ((dest/'complete.json').exists() or via is not None):
                return
        except Exception as exc:
            write(build / 'monitor-error.json', {'error': str(exc)[-1200:], 'updated_utc': datetime.now(timezone.utc).isoformat()})
        time.sleep(30)


def watch():
    """Wait locally for completed exports; never allocate a GPU for a missing stage."""
    lock = (DEST / 'controller.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    deadline = time.monotonic() + 48 * 3600
    while time.monotonic() < deadline:
        try:
            if not execution_enabled():
                write(DEST/'controller-status.json',{'status':'paused_by_user',
                    'detail':'Capability evaluations paused; prioritize native robot task evaluations.',
                    'updated_utc':datetime.now(timezone.utc).isoformat()})
                return
            waiting = []
            registry = read(DEST / 'checkpoints.json')
            for stage in registry['checkpoints']:
                if stage['status'] == 'waiting_for_checkpoint':
                    waiting.append(stage['id']); continue
                references = reference_ids(stage, registry)
                if any(not read(DEST / 'runs' / parent / 'results.json', {}).get('complete') for parent in references):
                    waiting.append(stage['id'] + ': awaiting complete reference evaluation'); continue
                build = build_run(stage['id'])
                if (build / 'submission.json').exists(): continue
                if (build / 'submission-intent.json').exists():
                    waiting.append(stage['id'] + ': reconcile submission intent'); continue
                active = command(['osmo', 'workflow', 'list', '--name', 'leact', '--status',
                                  'RUNNING', 'PENDING', 'WAITING', '--format-type', 'json'])['workflows']
                if any('capability' in r['name'] for r in active):
                    waiting.append(stage['id'] + ': another capability screen is active'); continue
                # Future screens leave two reservations free for the next main
                # EFE training job. The initial base screen is explicitly launched.
                try:
                    require_capacity(active, additional_nodes=3)
                except RuntimeError:
                    waiting.append(stage['id'] + ': reserving two node slots for main EFE work'); continue
                spec = read(build / 'spec.json')
                prior = read(build / 'input-transfer.json') or read(BUILD_ROOT / 'efe-capability-qwen-base-v1/input-transfer.json')
                if not prior or prior['sha256'] != spec['bundle_sha256']:
                    raise ValueError('Frozen shared inputs lack a verified transfer receipt')
                write(build / 'input-transfer.json', prior)
                receipt = submit(build); launch_monitor(build)
                waiting.append(stage['id'] + ': submitted ' + receipt['name'])
            write(DEST / 'controller-status.json', {'pid': os.getpid(), 'status': 'watching_checkpoint_registry',
                  'waiting': waiting, 'updated_utc': datetime.now(timezone.utc).isoformat(),
                  'remaining_seconds': round(deadline - time.monotonic())})
        except Exception as exc:
            write(DEST / 'controller-status.json', {'pid': os.getpid(), 'status': 'waiting',
                  'detail': str(exc)[-1200:], 'updated_utc': datetime.now(timezone.utc).isoformat()})
        time.sleep(45)
    write(DEST / 'controller-status.json', {'status': 'watch_window_expired',
          'updated_utc': datetime.now(timezone.utc).isoformat()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['build', 'submit', 'monitor', 'register', 'watch'])
    parser.add_argument('--checkpoint', default='qwen-base')
    parser.add_argument('--build', type=Path)
    parser.add_argument('--via-workflow')
    parser.add_argument('--path')
    parser.add_argument('--manifest')
    args = parser.parse_args()
    if args.action == 'watch':
        watch(); return
    if args.action == 'register':
        if not args.path or not args.manifest:
            parser.error('register needs --path and --manifest for a completed checkpoint')
        registry = read(DEST / 'checkpoints.json')
        stage = next(r for r in registry['checkpoints'] if r['id'] == args.checkpoint)
        if stage['status'] != 'waiting_for_checkpoint':
            raise ValueError('Existing checkpoint registration is immutable; use a new stage ID')
        stage.update(path=args.path, manifest=args.manifest, status='ready')
        write(DEST / 'checkpoints.json', registry)
        print(json.dumps({'registered': stage['id']}))
        return
    if args.action == 'monitor':
        monitor(args.build.resolve()); return
    build = build_run(args.checkpoint)
    if args.action == 'submit':
        if not execution_enabled():raise RuntimeError('Capability evaluations are paused by the user')
        if not args.via_workflow: parser.error('--via-workflow is required to verify shared inputs')
        transfer(build, args.via_workflow)
        receipt = submit(build)
        launch_monitor(build)
        print(json.dumps({'workflow': receipt['name'], 'checkpoint': args.checkpoint, 'build': str(build)}))
    else:
        print(build)


if __name__ == '__main__':
    main()
