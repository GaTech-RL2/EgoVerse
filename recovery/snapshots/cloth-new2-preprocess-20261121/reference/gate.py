"""Independent saved-checkpoint audit; no optimizer updates."""
import json,os,sys
from pathlib import Path
import torch
from campaign.data import sha
from campaign.train import code_hash
R=Path(os.environ['CAMPAIGN_ROOT']);rid=sys.argv[1]
cfg=json.loads((R/'configs'/(rid+'.json')).read_text());run=R/'runs'/(rid+'-smoke')
p=json.loads((run/'proof.json').read_text());a=json.loads((run/'transfer-audit.json').read_text())
assert json.loads((R/'reports/photometric-audit.json').read_text())['passed']
assert p['passed'] and p['step']==200 and p['resumed_from']==100 and p['changed_weights'] and p['full_validation']
assert p['code_sha256']==code_hash(R) and p['config_sha256']==sha(R/'configs'/(rid+'.json'))
assert a['passed'] and a['parent_sha256']==cfg['parent_checkpoint_sha256'] and a['optimizer_fresh']
ck=torch.load(run/'step_000200.pt',map_location='cpu',weights_only=False)
assert ck['step']==200 and ck['scheduler']['last_epoch']==200 and ck['scheduler']['T_max']==60000
assert all(torch.isfinite(v).all() for v in ck['model'].values())
assert ck['optimizer']['state']
for state in ck['optimizer']['state'].values():
    for v in state.values():
        if torch.is_tensor(v):assert torch.isfinite(v).all()
val=[json.loads(x) for x in (run/'full_validation_components.jsonl').read_text().splitlines()][-1]
assert val['rows']==3027 and val['step']==200 and val['full']
metrics=[json.loads(x) for x in (run/'metrics.jsonl').read_text().splitlines()]
assert len(metrics)==200 and [m['step'] for m in metrics]==list(range(1,201))
if cfg['photometric']['consistency_weight']:
    assert sum(m['consistency'] for m in metrics)>0
report=dict(passed=True,run=rid,checkpoint_sha256=sha(run/'step_000200.pt'),step=200,
    identity=ck['identity'],scheduler_T_max=60000,full_validation=val,transfer=a,
    mean_consistency=sum(m['consistency'] for m in metrics)/len(metrics))
(R/'reports'/(rid+'-independent-smoke-audit.json')).write_text(json.dumps(report,indent=2));print(json.dumps(report))
