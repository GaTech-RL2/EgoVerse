from pathlib import Path
import json,time,collections
R=Path('/coc/flash7/cxin34/cloth-new2-scratch-20261121');plan=json.loads((R/'plan.json').read_text());previous=None
while True:
 states=collections.Counter();active=[];errors=[]
 for r in plan:
  e=Path(r['episode']);q=e/'quality-quarantine.json';result=e/'worker-result.json'
  if q.exists():states['QUARANTINED']+=1;continue
  if result.exists():
   x=json.loads(result.read_text());states[x['status']]+=1
   if x['status']=='FAILED':errors.append({'take':r['take'],**x})
   continue
  stages={p.stem:json.loads(p.read_text()) for p in e.glob('*-stage.json')};running=[k for k,v in stages.items() if v['status']=='RUNNING']
  if running:active.append((r['take'],running));states['ACTIVE']+=1
  else:states['UNSTARTED']+=1
 accepted=len(list((R/'reports/accepted').glob('*.json')));out={'states':dict(states),'accepted':accepted,'active':active,'errors':errors}
 key=json.dumps(out,sort_keys=True)
 if key!=previous:print(json.dumps(out,indent=2),flush=True);previous=key
 if errors or states['STAGED']+states['QUARANTINED']==60:break
 time.sleep(30)
