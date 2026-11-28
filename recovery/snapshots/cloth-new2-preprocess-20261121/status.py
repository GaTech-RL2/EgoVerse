from pathlib import Path
import json,collections,subprocess
R=Path('/coc/flash7/cxin34/cloth-new2-scratch-20261121');rows=json.loads((R/'plan.json').read_text());out=[]
for r in rows:
 e=Path(r['episode']);st={p.stem:json.loads(p.read_text()).get('status') for p in e.glob('*-stage.json')}
 if st or (e/'quality-quarantine.json').exists():
  out.append({'take':r['take'],'stages':st,'result':json.loads((e/'worker-result.json').read_text()) if (e/'worker-result.json').exists() else None,'quarantine':json.loads((e/'quality-quarantine.json').read_text()) if (e/'quality-quarantine.json').exists() else None})
print(json.dumps(out,indent=2))
print(subprocess.check_output(['/opt/slurm/Ubuntu-24.04/current/bin/squeue','-u','cxin34','-h','-o','%i %j %T %R'],text=True).split('cxin_new2')[-1] if False else '\n'.join(l for l in subprocess.check_output(['/opt/slurm/Ubuntu-24.04/current/bin/squeue','-u','cxin34','-h','-o','%i %j %T %R'],text=True).splitlines() if 'new2' in l))
