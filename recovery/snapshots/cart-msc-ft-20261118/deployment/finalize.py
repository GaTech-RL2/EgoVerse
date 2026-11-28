import hashlib,json,subprocess
from pathlib import Path
R=Path('/coc/flash7/cxin34/cart-B180-MSCaug-FT10k-robotpc-20261118')
status=subprocess.check_output(['/opt/slurm/Ubuntu-24.04/current/bin/sacct','-X','-n','-j','4794957','-o','State,ExitCode'],text=True)
assert 'COMPLETED' in status and '0:0' in status,status
test=json.loads((R/'reports/skynet-server-test.json').read_text());assert test['passed']
plan=json.loads((R/'reports/plan.json').read_text());plan.update(status='CPU_OFFLINE_VERIFIED_ROBOTPC_GPU_SELFTEST_REQUIRED',test_job='4794957')
(R/'reports/plan.json').write_text(json.dumps(plan,indent=2))
(R/'README.md').write_text('''# MSC augmentation-only B180 → FT10k robotpc

Seed42, original30train/10validation. Source checkpoint unchanged, MSC augmentation
only (no consistency loss), augmentation disabled at inference. Virtual camera ON,
TGZ10HD2, RTC8/10/13,16inference steps,64x49actions, port8172.

Offline CPU check4794957 passed: original model tensors identical; three heldout
plain/RTC WebSocket predictions match reference; no-seed payload finite.
Actual robot hardware was not tested or commanded. Run the robotpc GPU selftest.

On robotpc, terminal1:
```bash
mkdir -p /home/aloha/RB_Y1_workspace/policies/cart_B180_MSCaug_FT10k
rsync -rt --links --info=progress2 -e "ssh -o IdentitiesOnly=yes" cxin34@sky1.cc.gatech.edu:/coc/flash7/cxin34/cart-B180-MSCaug-FT10k-robotpc-20261118/ /home/aloha/RB_Y1_workspace/policies/cart_B180_MSCaug_FT10k/
cd /home/aloha/RB_Y1_workspace/policies/cart_B180_MSCaug_FT10k
sha256sum -c SHA256SUMS
bash scripts/serve.sh
```

Terminal2:
```bash
cd /home/aloha/RB_Y1_workspace/policies/cart_B180_MSCaug_FT10k
bash scripts/check.sh
bash scripts/client.sh dry-run
```

After the existing installed driver/reset/stop workflow is ready:
```bash
bash scripts/client.sh run
```
''')
lines=[]
for p in sorted(R.rglob('*')):
 if not p.is_file() or p.name=='SHA256SUMS' or '__pycache__' in p.parts:continue
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 lines.append(h.hexdigest()+'  '+str(p.relative_to(R)))
(R/'SHA256SUMS').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(passed=True,files=len(lines),test=test,plan=plan),indent=2))
