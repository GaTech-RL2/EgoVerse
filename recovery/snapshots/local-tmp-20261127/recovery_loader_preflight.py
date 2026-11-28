from pathlib import Path
import json,sys,numpy as np,torch
R=Path('/coc/flash7/cxin34/cloth-valid-recovery-fix1124-20261124');O=Path('/coc/flash7/cxin34/egoa-target-retarget-fix-20261124/ft/cloth50');sys.path.insert(0,str(O))
from campaign.data import EpisodeDataset
meta=json.load(open(O/'releases/cloth/target.json'));meta['episodes']=[];norm=json.load(open(O/'releases/cloth/normalizer.json'))
for x in json.load(open(R/'plan.json')):
 if not(R/'reports/accepted'/(x['uuid']+'.json')).exists():continue
 td=R/'tensors/episodes'/x['uuid'];ep=json.load(open(td/'episode.json'));ep['split']='train' # fixture only, no published split mutation
 for v in ep['arrays'].values():v['file']=str(td/v['file'])
 meta['episodes'].append(ep)
p=R/'reports/loader-preflight-fixture.json';p.write_text(json.dumps(meta,indent=2));checks=[]
for arm in ['B','C','D']:
 ds=EpisodeDataset(p,'train',arm,norm)
 for ei,ep in enumerate(ds.episodes):
  n=len(ep['action']);offset=sum(len(z['action'])for z in ds.episodes[:ei])
  for t in [0,n-1]:
   b=ds[offset+t];assert all(torch.isfinite(v).all()for v in b.values());assert int(b['action_valid'].sum())==int(ep['action_valid'][t:min(n,t+64)].sum())
   if arm=='B':assert bool(b['dino_valid'])==(t+10<n)
  checks.append(dict(arm=arm,uuid=meta['episodes'][ei]['uuid'],rows=n,first_action_valid=bool(ep['action_valid'][0]),boundary_masks_passed=True))
(R/'reports/loader-preflight.json').write_text(json.dumps(dict(passed=True,fixture_only=True,published_splits_unchanged=True,checks=checks),indent=2));print('PASS',len(meta['episodes']),'recovered segments x B/C/D boundary loader')
