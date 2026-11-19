"""Retire canceled or expired unclaimed requests without changing model workers."""
import json
from pathlib import Path
import time

from efe_pear_file_rpc import atomic_json,retire_abandoned_pending


def main():
    spec=json.loads(Path(__file__).with_name('spec.json').read_text())
    root=Path(spec['run']);owner=spec['owner_model_workflow'];deadline=time.monotonic()+47*3600
    output=Path(spec['output']);output.mkdir(mode=0o700,parents=True,exist_ok=True)
    while time.monotonic()<deadline:
        if(root/'finished.json').exists():return
        head=json.loads((root/'head-heartbeat.json').read_text())
        if head['workflow']!=owner or time.time()-head['updated_unix']>300:return
        rows=retire_abandoned_pending(root/'transport')
        if rows:
            receipt={'workflow':owner,'updated_unix':time.time(),'retired_count':len(rows),'rows':rows,
                'model_processes_changed':False,'policy_rewards_created':False}
            atomic_json(output/('retired-'+str(time.time_ns())+'.json'),receipt)
            print('PEAR_PENDING_RETIREMENT '+json.dumps(receipt),flush=True)
        time.sleep(20)


if __name__=='__main__':main()
