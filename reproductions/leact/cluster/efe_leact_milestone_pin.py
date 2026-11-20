"""CPU-only watcher that pins one declared R1 update before retention removes it."""
import hashlib
import json
import os
from pathlib import Path
import time

from efe_intermediate_checkpoint import pin


def main():
    spec=json.loads((Path(__file__).parent/'spec.json').read_text())
    root=Path(spec['training_root']);destination=Path(spec['destination'])
    deadline=time.monotonic()+spec.get('timeout_seconds',24*3600)
    target=root/'restart-states'/f"checkpoint-{spec['step']}"/'training-state.json'
    while not target.exists() and not (destination/'checkpoint-pin.json').exists():
        if time.monotonic()>deadline:raise TimeoutError('Declared R1 checkpoint not committed before pin deadline')
        existing=[int(p.parent.name.split('-')[-1]) for p in (root/'restart-states').glob('checkpoint-*/training-state.json')]
        if existing and min(existing)>spec['step']:
            raise RuntimeError('Declared R1 checkpoint was already removed; do not select a later checkpoint silently')
        print('EFE_MILESTONE_WAIT '+json.dumps({'step':spec['step'],'latest_committed_step':max(existing,default=0),
            'workflow':os.environ.get('EFE_WORKFLOW_ID'),'optimizer_updates_by_watcher':0}),flush=True)
        time.sleep(30)
    value=pin(root,destination,spec['step'],selection=spec['selection'])
    raw=(destination/'checkpoint-pin.json').read_text()
    print('EFE_MILESTONE_PIN '+json.dumps({'success':True,'step':value['step'],'pin_json':raw,
        'pin_sha256':hashlib.sha256(raw.encode()).hexdigest(),'optimizer_updates_by_watcher':0}),flush=True)


if __name__=='__main__':main()
