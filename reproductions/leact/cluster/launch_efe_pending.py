"""Reconcile transient OSMO failures and launch the bounded EFE queue once."""
from datetime import datetime, timezone
import fcntl
import importlib
import json
import os
from pathlib import Path
import time

from .scale_efe import BUILD_ROOT, command, monitor, submit
from ..embodied_first_exam.report import ROOT, read, write, publish


def main():
    lock=(BUILD_ROOT/'efe-launch-queue.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for _ in range(720):
        queue=read(ROOT/'launch-queue.json',{'builds':[]})
        unfinished=False
        for name in queue['builds']:
            build=Path(name)
            try:
                if (build/'submission.json').exists():
                    if not (build/'monitor-launch.json').exists():monitor(build)
                    continue
                unfinished=True
                if (build/'launch-needs-review.json').exists():continue
                intent=read(build/'submission-intent.json')
                if intent:
                    import yaml
                    prefix=yaml.safe_load((build/'workflow.yaml').read_text())['workflow']['name']
                    matches=command(['osmo','workflow','list','--name',prefix,'--format-type','json'])['workflows']
                    matches=[w for w in matches if w['name'].startswith(prefix+'-')]
                    if len(matches)>1:raise RuntimeError('Multiple submission candidates; manual reconciliation required')
                    if matches:
                        write(build/'submission.json',matches[0]);monitor(build);continue
                    # A successful listing after the failed submit proves no
                    # workflow exists under this queue's unique name.
                    (build/'submission-intent.json').rename(build/f'unaccepted-intent-{int(time.time())}.json')
                receipt=submit(build);monitor(build)
                write(build/'launcher-status.json',{'status':'submitted','workflow':receipt['name'],
                      'updated_utc':datetime.now(timezone.utc).isoformat()})
                print(json.dumps({'workflow':receipt['name'],'build':str(build)}),flush=True)
            except Exception as exc:
                transient=any(x in str(exc).lower() for x in ('503','502','504','timed out','timeout','already reserved','connection'))
                record={'pid':os.getpid(),'error':str(exc), 'updated_utc':datetime.now(timezone.utc).isoformat(),
                        'status':'waiting_or_reconciling' if transient else 'needs_review'}
                write(build/'launcher-status.json',record)
                if not transient:write(build/'launch-needs-review.json',record)
        write(ROOT/'launcher-status.json',{'pid':os.getpid(),'unfinished':unfinished,
              'updated_utc':datetime.now(timezone.utc).isoformat()})
        try:
            from ..embodied_first_exam import report
            importlib.reload(report).publish()
        except Exception as exc:
            write(ROOT/'launcher-publish-error.json',{'error':str(exc),'updated_utc':datetime.now(timezone.utc).isoformat()})
        # Stay alive for the authorized student build to join this queue.
        time.sleep(30)


if __name__=='__main__':main()
