from pathlib import Path
import json,hashlib,ast
R=Path('/coc/flash7/scratch/cxin34/drawer-ik-feasible-20261125')
R.mkdir(exist_ok=True)
for d in ['logs','reports','out','tmp']: (R/d).mkdir(exist_ok=True)
C=Path('/coc/flash7/czhang883/warp-data/pipelines/drawer_base_check_1125')
s=(C/'base_variants2.py').read_text()
s=s.replace('import os, sys','import os, sys, json\nimport feasible_arm')
s=s.replace('CUR = [None]; TG = {}; TE = {}','CUR = [None]; TG = {}; TE = {}\nfeasible_arm.install(lambda: CUR[0])')
s=s.replace('rec = []; k = [0]; TG.clear(); TE.clear()', 'rec = []; k = [0]; TG.clear(); TE.clear(); feasible_arm.EVENTS.clear()')
s=s.replace('(("bs_head", yh), ("bs_body", None), ("bs_hip", yhip))', '(("bs_head", yh),)')
needle='def save(name, S, **extra):'
s=s.replace(needle,needle+'\n    from pathlib import Path\n    ev=list(feasible_arm.EVENTS)\n    assert ev, "adapter did not intercept IK"\n    report={"events":ev,"invalid_calls":sum(not e["valid"] for e in ev),"training_ready":False,\n            "reason":"Offline diagnostic only; final FK and release QC required"}\n    (Path(OUT).parent/"reports"/(name+"_"+TAG+".json")).write_text(json.dumps(report,indent=2))\n')
ast.parse(s)
(R/'pilot.py').write_text(s)
files=[C/'base_variants2.py',C/'segments_all.tsv',Path('/coc/flash7/czhang883/Documents/SEW-aria-pipeline/projects/rby1_teleop/geometric_kinematics_rby1.py')]
(R/'reports/provenance.json').write_text(json.dumps({str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},indent=2))
lines=(C/'segments_all.tsv').read_text().splitlines()
(R/'inputs.tsv').write_text('\n'.join(lines[i] for i in [7,14])+'\n')
print('STAGED',R)
