"""Finish the reviewed quota relocation and resume the unchanged EFE campaign."""
import argparse
import base64
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import zlib

import pexpect

from . import launch_efe_coverage as coverage
from .launch_efe_coverage_train import retry_build
from .scale_efe import command, read, write, submit

DEST = coverage.DEST
REMOTE = '/tmp/efe-coverage-storage-relocation-20261107'


def now():
    return datetime.now(timezone.utc).isoformat()


def remote(host, task, script):
    payload = base64.b64encode(zlib.compress(script.encode())).decode()
    entry = 'python3 -c ' + shlex.quote('import base64,zlib;exec(zlib.decompress(base64.b64decode(' + repr(payload) + ')))')
    child = pexpect.spawn('osmo', ['workflow', 'exec', host, task, '--entry', entry],
                          encoding='utf-8', timeout=35, maxread=65536, echo=False)
    try:
        child.expect(r'EFE_CAPTURE (\{[^\r\n]+\})\r?\n')
        # A background preparation process can keep the exec session open.
        # Its explicit receipt, not a later terminal EOF, acknowledges launch.
        return json.loads(child.match.group(1))
    except (pexpect.TIMEOUT, pexpect.EOF) as error:
        raise RuntimeError('Storage recovery remote receipt unavailable: ' + type(error).__name__) from None
    finally:
        child.close(force=True)


def status(phase, summary, **extra):
    value = {'utc': now(), 'pid': os.getpid(), 'phase': phase, 'summary': summary, **extra}
    write(DEST / 'storage-recovery-status.json', value)
    print(json.dumps(value), flush=True)
    try:
        from ..embodied_first_exam import report
        report.publish()
    except Exception as error:
        write(DEST / 'storage-recovery-publish-error.json', {'utc': now(), 'error': str(error)[-500:]})


def active():
    return command(['osmo', 'workflow', 'list', '--name', 'leact', '--status',
                    'RUNNING', 'PENDING', 'WAITING', '--format-type', 'json'])['workflows']


def inspect_remote(host, task):
    return remote(host, task, """from pathlib import Path
import json
p=Path(ROOT_LITERAL)
r={n:json.loads((p/n).read_text()) for n in ['progress.json','complete.json','error.json'] if (p/n).exists()}
print('EFE_CAPTURE '+json.dumps(r),flush=True)
""".replace('ROOT_LITERAL', repr(REMOTE)))


def launch_remote(host, task, action, quiescent=None):
    script = """from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,hashlib
p=Path(ROOT_LITERAL)
expected=EXPECTED_LITERAL
assert hashlib.sha256((p/'relocate.py').read_bytes()).hexdigest()==expected
action=ACTION_LITERAL
receipt=QUIET_LITERAL
if receipt:(p/'quiescent.json').write_text(json.dumps(receipt)+'\\n')
if (p/'error.json').exists():(p/'error.json').rename(p/('error-before-'+action+'.json'))
argv=['python3','-u',str(p/'relocate.py'),action]
if action=='activate':argv+=['--quiescent-receipt',str(p/'quiescent.json')]
with (p/(action+'.log')).open('ab') as log:
 child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
r={'pid':child.pid,'utc':datetime.now(timezone.utc).isoformat(),'action':action}
(p/'stage-process.json').write_text(json.dumps(r)+'\\n')
print('EFE_CAPTURE '+json.dumps(r),flush=True)
""".replace('ROOT_LITERAL', repr(REMOTE)).replace('EXPECTED_LITERAL', repr(
        hashlib.sha256(Path(__file__).with_name('efe_relocate_coverage_storage.py').read_bytes()).hexdigest()))
    script = script.replace('ACTION_LITERAL', repr(action)).replace('QUIET_LITERAL', repr(quiescent))
    return remote(host, task, script)


def wait_remote(host, task, launch, wanted, timeout=900):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = inspect_remote(host, task)
        write(DEST / 'storage-recovery-progress.json', record)
        error = record.get('error.json', {})
        if error.get('utc', '') >= launch['utc']:
            raise RuntimeError(error['error'])
        progress = record.get('progress.json', {})
        if progress.get('utc', '') >= launch['utc'] and progress.get('phase') == wanted:
            return record
        if progress.get('utc', '') >= launch['utc'] and progress.get('phase') == 'source_changed_wait_for_workers':
            raise RuntimeError('Campaign changed after quiescence; activation refused')
        time.sleep(20)
    raise TimeoutError('Storage recovery stage exceeded its bounded runtime')


def prepare_reviewed_retries(evidence):
    queue = read(DEST / 'queue.json')
    graded = coverage.outcomes(queue, evidence)
    reviewed = read(DEST / 'reviewed-infrastructure-retries.json', {})
    for batch in queue['batches']:
        if not batch.get('attempts'):
            continue
        attempt = batch['attempts'][-1]
        # Only explicit infrastructure interruptions or our unstarted pause.
        if batch['id'] not in {'b033', 'b070', 'b081', 'b082', 'b083', 'b084'}:
            continue
        state = command(['osmo', 'workflow', 'query', attempt['workflow'], '--format-type', 'json'])
        write(Path(attempt['build']) / 'scheduler.json', state)
        native = evidence.get('runs', {}).get(attempt['workflow'], {})
        failures = [r for r in native.get('results', {}).get('trials', []) if r['status'] == 'infrastructure_failure']
        transient = bool(failures) and all(any(x in r.get('action_metrics', {}).get('error', '') for x in
            ['RemoteDisconnected', 'ConnectionResetError', 'ConnectionRefusedError', 'TimeoutError']) for r in failures)
        paused = (batch['id'] == 'b084' and attempt['workflow'] == 'leact-efe-coverage-v1-b084-r1-1'
                  and state['status'] == 'FAILED_CANCELED' and not state.get('start_time'))
        if not (coverage.infra_retryable(state) or transient or paused) or len(batch['attempts']) >= 12:
            continue
        if state['status'] in {'RUNNING', 'PENDING', 'WAITING'}:
            raise ValueError('Worker unexpectedly became active during recovery')
        unseen = [r for r in batch['tasks'] if (r['task'], r['collection_attempt']) not in graded
                  and r['task'] not in batch.get('unrecoverable_tasks', [])]
        if unseen:
            batch['retry_pending'] = unseen
            reviewed[batch['id']] = {'source_workflow': attempt['workflow'], 'reviewed_utc': now(),
                'reason': 'Fresh durable outcomes reconciled after storage recovery. Retry only confirmed infrastructure-interrupted or unstarted attempts; all graded outcomes retained.',
                'storage_relocation_receipt': 'storage-relocation.json'}
    write(DEST / 'queue.json', queue)
    write(DEST / 'reviewed-infrastructure-retries.json', reviewed)


def resume(evidence, relocation):
    prepare_reviewed_retries(evidence)
    path = DEST / 'training-queue.json'
    queue = read(path)
    record = queue['stages']['warmstart']
    previous = record['attempts'][-1]
    if previous['workflow'] != 'leact-efe-coverage-warmstart-v1-p3-1':
        raise ValueError('Warmstart history changed; reconcile before submitting')
    if evidence['training']['warmstart']['error'].get('error') != '[Errno 122] Disk quota exceeded':
        raise ValueError('Expected the reviewed pre-training quota failure')
    build = retry_build(previous['build'], len(record['attempts']))
    receipt = submit(build, priority='LOW')
    record['attempts'].append({'workflow': receipt['name'], 'build': str(build), 'submitted_utc': now(),
                               'recovery_of': previous['workflow'], 'reason': 'Verified storage relocation; frozen payload unchanged.'})
    record.update(status='PENDING', retry=False, storage_relocation=relocation['destination'])
    write(path, queue)
    # Publish and submit reviewed collection continuations before detaching.
    result = coverage.tick()
    log = (DEST / 'monitor.log').open('ab', buffering=0)
    environment = {**os.environ, 'PYTHONPATH': str(Path('emimic/lib/python3.11/site-packages').resolve())}
    process = subprocess.Popen([sys.executable, '-S', '-u', '-m',
        'reproductions.leact.cluster.launch_efe_coverage', 'monitor'],
        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
        env=environment, start_new_session=True)
    launch = read(DEST / 'monitor-launch.json', {})
    launch.update(pid=process.pid, utc=now(), storage_relocation='storage-relocation.json',
                  controller_sha256=hashlib.sha256(Path(coverage.__file__).read_bytes()).hexdigest())
    write(DEST / 'monitor-launch.json', launch)
    return {'warmstart_workflow': receipt['name'], 'monitor_pid': process.pid,
            'candidate_successful_episodes': result['candidate_successful_episodes']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', required=True)
    parser.add_argument('--task', required=True)
    args = parser.parse_args()
    lock = (DEST / 'storage-recovery-controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    old = read(DEST / 'monitor-launch.json')
    check = subprocess.run(['ps', '-p', str(old['pid']), '-o', 'command='], capture_output=True, text=True)
    if 'launch_efe_coverage monitor' in check.stdout:
        raise ValueError('Stop the campaign controller before storage relocation')
    deadline = time.monotonic() + 7200
    while True:
        jobs = active()
        if not jobs:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError('Workers did not finish within the bounded relocation wait')
        status('waiting_for_workers', 'Storage quota recovery is waiting for the current collector episodes to finish; new launches are paused. No expanded optimizer updates have run.',
               active_campaign_workflows=[{'name': r['name'], 'status': r['status']} for r in jobs])
        time.sleep(30)
    # An earlier live copy is only a prefill; rebuild after every writer ended.
    initial = inspect_remote(args.host, args.task)
    if initial.get('progress.json', {}).get('phase') == 'copying':
        raise RuntimeError('Initial storage preparation still running; inspect before another copy')
    status('copying', 'Collectors have ended. Verifying every copied file on the replacement shared filesystem before resuming the same recipe.')
    launch = launch_remote(args.host, args.task, 'prepare')
    wait_remote(args.host, args.task, launch, 'prepared')
    if active():
        raise ValueError('New campaign workers appeared before activation')
    quiet = {'utc': now(), 'controller_stopped': True, 'active_campaign_workflows': []}
    launch = launch_remote(args.host, args.task, 'activate', quiet)
    finished = wait_remote(args.host, args.task, launch, 'storage_relocated', timeout=300)
    relocation = finished['complete.json']
    if not relocation.get('success') or relocation.get('original_tree_deleted'):
        raise ValueError('Missing preservation proof')
    write(DEST / 'storage-relocation.json', relocation)
    original_remote = coverage.remote
    try:
        coverage.remote = lambda host, script: remote(args.host, args.task, script)
        evidence = coverage.capture(args.host)
    finally:
        coverage.remote = original_remote
    result = resume(evidence, relocation)
    status('resumed', 'Storage relocation is complete and all original evidence is retained. Warmstart preparation and collection have resumed at LOW priority; optimizer training still requires 256 admitted successes.',
           relocation=relocation, **result)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        status('needs_review', 'Storage recovery stopped for review before further launches: ' + str(error), error=repr(error))
        raise
