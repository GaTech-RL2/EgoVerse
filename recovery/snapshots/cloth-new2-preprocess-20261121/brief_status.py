from pathlib import Path
import json,collections,subprocess,shutil
R=Path('/coc/flash7/cxin34/cloth-new2-scratch-20261121');plan=json.loads((R/'plan.json').read_text());states=collections.Counter();active=[];errors=[];quarantine=[]
for r in plan:
 e=Path(r['episode']);q=e/'quality-quarantine.json';result=e/'worker-result.json'
 if q.exists():quarantine.append({'take':r['take'],**json.loads(q.read_text())});states['QUARANTINED']+=1;continue
 if result.exists():
  x=json.loads(result.read_text());states[x['status']]+=1
  if x['status']=='FAILED':errors.append({'take':r['take'],**x})
  continue
 stages={p.stem:json.loads(p.read_text()) for p in e.glob('*-stage.json')};running=[k for k,v in stages.items() if v['status']=='RUNNING'];failed=[k for k,v in stages.items() if v['status']=='FAILED']
 if failed:errors.append({'take':r['take'],'failed':failed})
 if running:active.append({'take':r['take'],'stages':running});states['ACTIVE']+=1
 else:states['UNSTARTED']+=1
out={'states':dict(states),'active':active,'errors':errors,'quarantine':quarantine,'accepted':len(list((R/'reports/accepted').glob('*.json'))),'free_GiB':round(shutil.disk_usage(R).free/1024**3,1)}
queue=subprocess.check_output(['/opt/slurm/Ubuntu-24.04/current/bin/squeue','-u','cxin34','-h','-o','%i %j %T %R'],text=True)
out['queue']=[l for l in queue.splitlines() if 'new2' in l]
(R/'reports/live-status.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
