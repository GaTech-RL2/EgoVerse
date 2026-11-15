"""Observe native PEAR qualification and retry only unstarted allocations."""
from datetime import datetime, timezone
from pathlib import Path
import fcntl
import json
import shutil
import subprocess
import time
import yaml

from .launch_efe_pear import DEST
from .scale_efe import read, write, submit, command
from .launch_efe_training import freeze_sources
from .launch_efe_coverage import infra_retryable
from .efe_coverage_retry import retry_due
from .recover_efe_coverage_storage import remote


def tick(destination=None):
    dest = destination or DEST
    queue = read(dest/'native-qualification-queue.json')
    attempt = queue['attempts'][-1]
    build, name = Path(attempt['build']), attempt['workflow']
    scheduler = command(['osmo','workflow','query',name,'--format-type','json'])
    write(build/'scheduler.json', scheduler)
    status = {'workflow':name, 'scheduler_status':scheduler['status'], 'optimizer_updates':0,
              'updated_utc':datetime.now(timezone.utc).isoformat(), 'native_rollouts':0,
              'phase':'native_qualification_'+scheduler['status'].lower()}
    complete = scheduler['status']=='COMPLETED' or scheduler['status'].startswith('FAILED')
    archive = read(build/'spec.json')['archive_root']+'/'+name
    capture = read(dest/'native-qualification-evidence.json', {})
    if capture.get('workflow') != name:
        capture = {}
    host = name if scheduler['status']=='RUNNING' else None
    if complete:
        active = command(['osmo','workflow','list','--name','leact-efe-coverage-warmstart',
                          '--status','RUNNING','--format-type','json'])['workflows']
        host = active[0]['name'] if active else None
    if host:
        try:
            value = remote(host, 'preflight', """from pathlib import Path
import json
p=Path(ARCHIVE)
r={n:json.loads((p/n).read_text()) for n in ['progress.json','worker-start.json',
 'sampler/progress.json','sampler/ready.json','pear-native-result.json','pear-native-error.json'] if (p/n).exists()}
if (p/'pear-episodes.json').exists():
 data=json.loads((p/'pear-episodes.json').read_text())
 r['episodes']=[{k:v for k,v in e.items() if k!='captures'} for e in data['episodes']]
if (p/'logs/pear-sampler.log').exists():r['sampler_log_tail']=(p/'logs/pear-sampler.log').read_text(errors='replace')[-3500:]
print('EFE_CAPTURE '+json.dumps(r),flush=True)
""".replace('ARCHIVE', repr(archive)))
            capture = {'workflow':name, 'captured_utc':datetime.now(timezone.utc).isoformat(), **value}
            write(dest/'native-qualification-evidence.json', capture)
        except Exception as error:
            status['capture_error'] = repr(error)
    rows = capture.get('episodes', [])
    status.update(native_rollouts=len(rows), successes=sum(r['reward'] for r in rows),
        generated_tokens=sum(r['generated_tokens'] for r in rows), model_seconds=sum(r['model_seconds'] for r in rows))
    result = capture.get('pear-native-result.json') or capture.get('pear-native-error.json')
    if complete and not result:
        logs = subprocess.run(['osmo','workflow','logs',name], capture_output=True, text=True, timeout=50)
        (dest/'native-qualification.log').write_text(logs.stdout+logs.stderr)
        for line in logs.stdout.splitlines():
            if 'PEAR_NATIVE_RESULT ' in line:
                try:result=json.loads(line.split('PEAR_NATIVE_RESULT ',1)[1])
                except json.JSONDecodeError:pass
    if result:
        write(dest/'native-qualification-result.json', result)
        status.update(phase='native_qualification_passed' if result.get('success') else 'native_qualification_needs_review',
                      native_rollouts=result.get('native_rollouts',len(rows)))
    status['summary'] = (f"Native PEAR pipeline qualification: {status['native_rollouts']}/8 training-only episodes verified "
        f"on the base student; scheduler {scheduler['status'].lower()}. Exact sampler tokens, probabilities, grades and timings are retained. "
        'Primary RL training still waits for the common final warmstart and a verified real optimizer update.')
    if result and result.get('success'):
        group = result['group']
        status['summary'] = ('Native PEAR group qualification passed: '+str(group['successes'])+'/8 successes on one training scene. '
            +('The group has reward variance.' if group['admitted'] else 'The constant-reward group is filtered from RL while its cost remains recorded.')
            +' This is a base-policy runtime check, with zero optimizer updates; it is not a trained PEAR result.')
    elif result:
        status['summary'] += ' Review required: '+result.get('error','unknown native error')
    # Any started native attempt may have graded outcomes, even if its observer
    # lost contact. Do not silently replace those outcomes with another draw.
    if complete and not scheduler.get('start_time') and infra_retryable(scheduler) and len(queue['attempts'])<8:
        complete = False
        status['phase'] = 'native_qualification_retry_backoff'
        if retry_due(queue, name):
            number = len(queue['attempts'])
            original = Path(queue['attempts'][0]['build'])
            retry = original.parent/(original.name+f'-allocation-retry-{number}')
            if not (retry/'source-manifest.json').exists():
                retry.mkdir(exist_ok=True)
                for filename in read(original/'source-manifest.json'):
                    shutil.copyfile(original/filename, retry/filename)
                workflow = yaml.safe_load((retry/'workflow.yaml').read_text())
                workflow['workflow']['name'] += f'-r{number}'
                (retry/'workflow.yaml').write_text(yaml.safe_dump(workflow, sort_keys=False))
                freeze_sources(retry)
            receipt = submit(retry, priority='LOW')
            queue['attempts'].append({'workflow':receipt['name'], 'build':str(retry), 'priority':'LOW',
                                     'submitted_utc':datetime.now(timezone.utc).isoformat()})
            status.update(workflow=receipt['name'], scheduler_status='PENDING')
        write(dest/'native-qualification-queue.json', queue)
    write(dest/'status.json', status)
    if destination is not None:
        from .launch_efe_pear_update_qualification import publish_progress
        publish_progress()
    from ..embodied_first_exam import report
    report.publish()
    return complete


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--group',choices=['door-lock','coffee-button'])
    args=parser.parse_args()
    destination=DEST/'optimizer-qualification'/args.group if args.group else DEST
    with (destination/'native-monitor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        deadline = time.monotonic()+16*3600
        while time.monotonic()<deadline:
            try:
                if tick(destination if args.group else None):break
            except Exception as error:
                write(destination/'native-monitor-error.json', {'utc':datetime.now(timezone.utc).isoformat(), 'error':repr(error)})
            time.sleep(45)
