from pathlib import Path
import ast,shutil
R=Path('/coc/flash7/scratch/cxin34/cart-drawer-nola-fix-bs-20261125');W=R/'training/cart/lora'
shutil.copy2(R/'scripts/dino_lora.py',W/'campaign/dino_lora.py')
p=W/'campaign/train.py';backup=p.with_name('train.before_lora.py')
if not backup.exists():shutil.copy2(p,backup)
s=backup.read_text();s=s.replace('from .policy import CampaignPolicy','from .policy import CampaignPolicy\nfrom . import dino_lora')
s=s.replace('    optimizer=torch.optim.AdamW(model.parameters(),',"    lora_modules=dino_lora.install(model,cfg['dino_head_lora'])\n    optimizer=torch.optim.AdamW(model.parameters(),")
start=s.index('        fresh_dino={');end=s.index('        del ck',start)
s=s[:start]+'''        transferred=dino_lora.transfer(model,ck['model'])
        assert not optimizer.state and scheduler.last_epoch==0 and step==0
        (run/'transfer-audit.json').write_text(json.dumps(dict(passed=True,parent=str(source),parent_step=ck['step'],parent_sha256=sha(source),transferred_keys=transferred,optimizer_fresh=True,scheduler_step=0,target_step=0,inherited_dino_head=True,dino_base_frozen=True,lora_modules=lora_modules,other_policy_parameters_trainable=True),indent=2))
'''+s[end:]
s=s.replace("    checkpoint(last);checkpoint(run/f'step_{step:06d}.pt')", "    if args.smoke:\n        (run/'lora-smoke-audit.json').write_text(json.dumps(dino_lora.smoke_checks(model,initial),indent=2))\n    checkpoint(last);checkpoint(run/f'step_{step:06d}.pt')")
ast.parse(s);p.write_text(s)
p=W/'scripts/smoke_gate.py';backup=p.with_name('smoke_gate.before_lora.py')
if not backup.exists():shutil.copy2(p,backup)
s=backup.read_text();needle="assert ck['optimizer']['state']"
s=s.replace(needle,needle+'''
from campaign.dino_lora import PREFIX,adapter_key
la=json.loads((run/'lora-smoke-audit.json').read_text());assert all(la[k] for k in ['inherited_head_base_unchanged','lora_weights_changed','rest_policy_changed'])
assert transfer['inherited_dino_head'] and transfer['dino_base_frozen']
for k,v in ck['initial'].items():
    if k.startswith(PREFIX) and not adapter_key(k):assert torch.equal(v,ck['model'][k]),k
assert any(not torch.equal(v,ck['model'][k])for k,v in ck['initial'].items()if adapter_key(k))
assert any(not torch.equal(v,ck['model'][k])for k,v in ck['initial'].items()if not k.startswith(PREFIX))
''').replace('fresh DINO head','inherited frozen DINO head + rank8 LoRA; rest full FT')
ast.parse(s);p.write_text(s)
print('LORA_HEAD_ONLY_PATCHED')
