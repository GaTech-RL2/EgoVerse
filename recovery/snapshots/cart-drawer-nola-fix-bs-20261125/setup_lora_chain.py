from pathlib import Path
R=Path('/coc/flash7/scratch/cxin34/cart-drawer-nola-fix-bs-20261125');W=R/'training/cart/lora'
gate=W/'scripts/lora_unit_gate.py'
gate.write_text('''import json,os
from pathlib import Path
from campaign.data import sha
w=Path(os.environ['CAMPAIGN_ROOT'])
a=json.load(open(w/'reports/lora-unit.json'))
assert a['passed'] and a['lora_code_sha256']==sha(w/'campaign/dino_lora.py')
''')
for name,body in {'cart-lora-unit.sbatch':'source '+str(W/'scripts/env.sh')+'\nexec "$CAMPAIGN_PYTHON" '+str(R/'scripts/lora_unit.py'),'cart-lora-chain.sbatch':'export CAMPAIGN_ROOT='+str(W)+'\nexport RUN_ID=cart-nola-fix-bs-finetune-dinoLoRA-MSCstrong-s42\nexec bash "$CAMPAIGN_ROOT/scripts/chain.sbatch"'}.items():
 (R/'scripts'/name).write_text('#!/bin/bash\nset -euo pipefail\n'+body+'\n')
p=W/'scripts/chain.sbatch';s=p.read_text()
if 'lora_unit_gate.py' not in s:
 s=s.replace('source "$CAMPAIGN_ROOT/scripts/env.sh"','source "$CAMPAIGN_ROOT/scripts/env.sh"\n"$CAMPAIGN_PYTHON" "$CAMPAIGN_ROOT/scripts/lora_unit_gate.py"');p.write_text(s)
for i,l in enumerate((W/'campaign/train.py').read_text().splitlines()):
 if any(k in l for k in ['lora','initial=','checkpoint(last)','optimizer=torch']):print(i+1,l)
