"""Verify task/scene separation using frozen metadata only, without oracle access."""
from pathlib import Path
import hashlib
import json
import subprocess
import traceback

from scene_identity import scene_identity

ROOT=Path('/tmp/efe-direct-scene-audit-v1')


def main():
    spec=json.loads((ROOT/'spec.json').read_text())
    reference=Path(spec['reference'])
    actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=reference,text=True).strip()
    if actual!=spec['revision']:raise ValueError('Native reference revision differs')
    staging=ROOT/'metadata';receipts=[]
    for row in spec['rows']:
        for name in ['task.toml','DIGEST','environment/sim/instance.npz']:
            relative=row['path']+'/'+name;path=staging/relative;path.parent.mkdir(parents=True,exist_ok=True)
            data=subprocess.check_output(['git','show',spec['revision']+':'+relative],cwd=reference,timeout=120)
            path.write_bytes(data)
        receipt=scene_identity(staging,row)
        expected=spec['protected_expected'].get(row['task'])
        if expected and receipt['scene_sha256']!=expected:raise ValueError('Protected scene audit does not reproduce: '+row['task'])
        receipts.append({**receipt,'partition':'protected' if expected else 'train_candidates'})
        (ROOT/'progress.json').write_text(json.dumps({'completed':len(receipts),'total':len(spec['rows'])}))
    protected={(r['suite'],r['scene_sha256']):r['task'] for r in receipts if r['partition']=='protected'}
    collisions=[{'task':r['task'],'protected_task':protected[(r['suite'],r['scene_sha256'])]}
                for r in receipts if r['partition']=='train_candidates' and (r['suite'],r['scene_sha256']) in protected]
    result={'upstream_revision':actual,'rows':receipts,'collisions':collisions,
        'protected_scenes_reproduced':len(protected),'candidate_scenes':sum(r['partition']=='train_candidates' for r in receipts),
        'scope':'Only immutable task.toml, DIGEST and physical instance.npz metadata read. No oracle controller or demonstration inspected.'}
    (ROOT/'result.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')


if __name__=='__main__':
    try:main()
    except Exception as error:
        (ROOT/'error.json').write_text(json.dumps({'error':str(error),'traceback':traceback.format_exc()}));raise
