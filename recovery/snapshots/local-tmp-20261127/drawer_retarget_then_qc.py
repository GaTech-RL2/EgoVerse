import os,sys,subprocess,json
from pathlib import Path
R=Path('/coc/flash7/scratch/cxin34/cart-drawer-nola-fix-bs-20261125');i=int(sys.argv[1]);e=json.load(open(R/'drawer/segment-manifest.json'))[i];p=Path(e['segment']);py='/coc/flash7/czhang883/miniconda3/envs/rby1/bin/python'
r=subprocess.run([py,str(R/'scripts/drawer_retarget.py'),str(i)])
assert (p/'bs-gate-rgb-merge.json').exists(),'Processing failure before quality gate requires audit'
r=subprocess.run([py,str(R/'scripts/drawer_qc_v2.py'),str(i)],check=True)
q=json.load(open(p/'retarget-proof-v2.json'));assert q['passed']or q['quality_quarantined']
print('TERMINAL_ACCEPTED'if q['passed']else'TERMINAL_QUALITY_QUARANTINED',i,flush=True)
