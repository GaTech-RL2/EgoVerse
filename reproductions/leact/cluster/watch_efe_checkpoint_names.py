"""Publish checkpoint aliases through an already allocated study worker."""
import fcntl
import json
from pathlib import Path
import time

from .scale_efe import command,write
from .recover_efe_coverage_storage import remote
from .launch_efe_pear_scaled import DEST,now


def tick():
    rows=command(['osmo','workflow','list','--name','leact-efe','--status','RUNNING',
                  '--format-type','json'])['workflows']
    rows=[r for r in rows if r.get('pool')=='groot-h100-01']
    # Long-lived training/selection/preparation jobs are less likely to vanish
    # during the read than the short-lived receipt recovery tasks.
    rows.sort(key=lambda r:('receipt' in r['name'],r['name']))
    if not rows:return {'phase':'waiting_for_allocated_worker','utc':now()}
    host=rows[0]['name']
    scheduler=command(['osmo','workflow','query',host,'--format-type','json'])
    tasks=[t for g in scheduler.get('groups',[]) for t in g.get('tasks',[]) if t['status']=='RUNNING']
    if not tasks:return {'phase':'allocation_changed','utc':now()}
    source=Path(__file__).with_name('efe_checkpoint_names.py').read_text().split("if __name__=='__main__':")[0]
    result=remote(host,tasks[0]['name'],source+"\nprint('EFE_CAPTURE '+json.dumps(sync()),flush=True)\n")
    result.update(phase='aliases_published',workflow=host)
    write(DEST/'checkpoint-names.json',result)
    return {k:v for k,v in result.items() if k!='checkpoints'}


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    with (DEST/'checkpoint-names-watch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            try:value=tick()
            except Exception as exc:value={'phase':'retrying','error':str(exc),'utc':now()}
            write(DEST/'checkpoint-names-watch-status.json',value)
            print(json.dumps(value),flush=True)
            time.sleep(60)


if __name__=='__main__':main()
