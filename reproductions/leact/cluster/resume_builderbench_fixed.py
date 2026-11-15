"""Resume the repaired timing job after the BuilderBench evaluation finishes."""
from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

BASE = Path(__file__).resolve().parents[1]
BUILD = BASE / 'cluster/.build/builderbench-gpt6-student-fixed-20261024-v1'
REPAIR = BUILD / 'repair-20261026'
EVALUATION = BASE / 'cluster/.build/builderbench-gpt6-student-evaluate-20261024-v1'


def write(name, value):
    value['updated_utc'] = datetime.now(timezone.utc).isoformat()
    path = REPAIR / name
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    tmp.replace(path)


def query(args):
    p = subprocess.run(['osmo', *args, '--format-type', 'json'], capture_output=True,
                       text=True, check=True, timeout=35)
    return json.loads(p.stdout[p.stdout.index('{'):])


def main():
    from ..builderbench.queue import stopped_by_user
    if stopped_by_user():
        write('waiter-status.json',{'status':'stopped_by_user','pid':os.getpid()})
        return
    lock = (REPAIR / 'waiter.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for _ in range(5760):
        if stopped_by_user():
            write('waiter-status.json',{'status':'stopped_by_user','pid':os.getpid()})
            return
        try:
            dependency = json.loads((EVALUATION / 'submission.json').read_text())['name']
            state = query(['workflow', 'query', dependency])
            if state['status'] != 'COMPLETED':
                write('waiter-status.json', {'pid': os.getpid(), 'status': 'waiting_for_evaluation',
                                             'dependency': dependency, 'dependency_status': state['status']})
                time.sleep(60)
                continue
            active = query(['workflow', 'list', '--name', 'leact', '--status', 'RUNNING', 'PENDING', 'WAITING'])['workflows']
            from .compute_budget import node_limit, reserved_nodes
            if reserved_nodes(active) >= node_limit():
                write('waiter-status.json', {'pid': os.getpid(), 'status': 'waiting_for_node_slot'})
                time.sleep(60)
                continue
            # The queue rechecks the shared submission lock and node cap.
            with (BUILD / 'controller.lock').open('a') as controller_lock:
                fcntl.flock(controller_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                if not (REPAIR / 'installed.json').exists():
                    old = json.loads((BUILD / 'submission.json').read_text())['name']
                    archive = BUILD / ('prior-' + old)
                    archive.mkdir(exist_ok=True)
                    names = ['submission.json', 'submission-intent.json', 'launch-record.json',
                             'status-current.json', 'upload-complete.json', 'payload.tar.gz',
                             'cpu-validation.json', 'payload-sha256.json']
                    for name in names:
                        source = BUILD / name
                        if source.exists():
                            shutil.copyfile(source, archive / name)
                    shutil.copyfile(REPAIR / 'payload.tar.gz', BUILD / 'payload.tar.gz')
                    shutil.copyfile(REPAIR / 'cpu-validation.json', BUILD / 'cpu-validation.json')
                    for name in ['submission.json', 'submission-intent.json', 'upload-complete.json']:
                        (BUILD / name).unlink(missing_ok=True)
                    write('installed.json', {'prior_workflow': old, 'archive': str(archive)})
            write('waiter-status.json', {'pid': os.getpid(), 'status': 'starting_repaired_queue'})
            result = subprocess.run([sys.executable, '-u', '-m', 'reproductions.leact.builderbench.queue',
                                     '--build', str(BUILD), '--role', 'fixed'])
            if result.returncode == 0:
                write('waiter-status.json', {'pid': os.getpid(), 'status': 'queue_finished'})
                return
            if (BUILD / 'submission.json').exists():
                write('waiter-status.json', {'pid': os.getpid(), 'status': 'queue_needs_review',
                                             'returncode': result.returncode})
                return
            # Another authorized job may have reserved the slot first.
            write('waiter-status.json', {'pid': os.getpid(), 'status': 'retrying_slot_check'})
        except Exception as error:
            write('waiter-error.json', {'error': repr(error)})
        time.sleep(60)
    write('waiter-status.json', {'pid': os.getpid(), 'status': 'wait_deadline_reached'})


if __name__ == '__main__':
    main()
