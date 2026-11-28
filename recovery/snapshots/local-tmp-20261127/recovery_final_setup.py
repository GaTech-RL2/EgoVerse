from pathlib import Path
import ast
R=Path('/coc/flash7/cxin34/cloth-valid-recovery-fix1124-20261124')
s='''import json,sys,os,hashlib,time
from pathlib import Path
import numpy as np,torch,h5py
R=Path(__file__).resolve().parents[1];OLD=Path('/coc/flash7/cxin34/egoa-target-retarget-fix-20261124');W=OLD/'ft/cloth50';sys.path.insert(0,str(W))
from campaign.data import EpisodeDataset,sha
oldmeta=json.load(open(W/'releases/cloth/target.json'));meta=dict(oldmeta);eps=list(oldmeta['episodes']);added=[];excluded=[];unresolved=[]
for x in json.load(open(R/'plan.json')):
 e=Path(x['episode']);result=e/'recovery-result.json'
 if not result.exists():unresolved.append(x['uuid']);continue
 status=json.load(open(result))
 if status['status']=='QUALITY_QUARANTINED_RETARGET':excluded.append(dict(x,status=status));continue
 if status['status']!='NUMERIC_PASS_VISUAL_LOADER_PENDING':unresolved.append(dict(uuid=x['uuid'],status=status));continue
 ap=R/'reports/accepted'/(x['uuid']+'.json');a=json.load(open(ap));assert a['passed'] and a['height_check']['full_fk_equal']
 td=R/'tensors/episodes'/x['uuid'];ep=json.load(open(td/'episode.json'));assert sha(td/'episode.json')==a['episode_metadata_sha256'];arrays={}
 for k,v in ep['arrays'].items():
  p=td/v['file'];arr=np.load(p,mmap_mode='r');assert sha(p)==v['sha256'] and np.isfinite(arr).all();arrays[k]=dict(v,file=str(p))
 with h5py.File(e/'retarget/selected.hdf5')as f,np.load(e/'retarget/work'/(x['retarget_tag']+'_rgb224.npz'))as rgb:
  ts=f['data/demo_0/obs/aria_ts'][:]/1000;lo,hi=x['recovery_grid_bounds'];np.testing.assert_allclose(ts,rgb['grid_us'][lo:hi],atol=2,rtol=0)
  for k in range(0,len(ts),32):np.testing.assert_array_equal(f['data/demo_0/obs/aria_image'][k:k+32],rgb['frames'][rgb['sel'][lo+k:min(lo+k+32,hi)]])
  assert ts[-1]-ts[0]>1e6
  with np.load(e/'retarget/work'/(x['retarget_tag']+'_aria60.npz'))as aa:assert len(aa['ts'])>hi-1
 old=next(z for z in oldmeta['episodes']if z['uuid']==x['recording_id']);assert old['split']==x['split'] and old['raw_vrs_sha256']==x['raw_vrs_sha256']
 added.append(dict(uuid=x['uuid'],recording_id=x['recording_id'],take=x['take'],task='cloth',split=x['split'],rows=ep['rows'],arrays=arrays,raw_vrs_sha256=x['raw_vrs_sha256'],audit_sha256=sha(ap),recovery_grid_bounds=x['recovery_grid_bounds'],parent_episode=x['parent_episode'],episode=x['episode']))
report=dict(added=len(added),excluded=excluded,unresolved=unresolved,old_episodes=len(eps),old_rows=sum(e['rows']for e in eps))
(R/'reports/final-disposition.json').write_text(json.dumps(report,indent=2));assert not unresolved,unresolved
# Verify old files are unchanged; never mutate or republish their release.
for f,d in json.load(open(R/'reports/original-integrity-before.json')).items():assert sha(f)==d,f
for e in eps:
 for k,v in e['arrays'].items():assert sha(v['file'])==v['sha256'],v['file']
eps+=added;meta.update(episodes=eps,recovery_excluded=excluded,training_release_ready=False,quality_passed=False,recipe=oldmeta['recipe']+'; extra recovered strict-valid segments, no stitching',parent_manifest_sha256=sha(W/'releases/cloth/target.json'))
by={}
for e in eps:
 by.setdefault(e['recording_id'],set()).add(e['split'])
assert all(len(v)==1 for v in by.values())
release=R/'release';release.mkdir(exist_ok=True);target=release/'target.json';target.write_text(json.dumps(meta,indent=2))
normalizer=W/'releases/cloth/normalizer.json';norm=json.load(open(normalizer));(release/'normalizer.json').write_bytes(normalizer.read_bytes())
loaders=[]
for split in ['train','valid']:
 ds=EpisodeDataset(target,split,'B',norm);assert len(ds)==sum(e['rows']for e in eps if e['split']==split)
 for j in range(len(ds)):
  b=ds[j];assert all(torch.isfinite(v).all()for v in b.values());ei,t=ds.index[j];ep=ds.episodes[ei];n=len(ep['action'])
  assert b['action'].shape==(64,49) and int(b['action_valid'].sum())==min(64,n-t) and bool(b['dino_valid'])==(t+10<n)
  if t in [0,n-1]:np.testing.assert_array_equal(b['dino'].numpy(),ep['dino'][min(t+10,n-1)])
  if j%5000==0:print('LOADER',split,j,len(ds),flush=True)
 loaders.append(dict(split=split,rows=len(ds),passed=True))
metrics={}
for k in ['action','state_robot0_joint_pos','state_eef_pose_glass','state_hand_left_qpos','state_hand_right_qpos']:
 c=np.asarray(norm[k]['center']);scale=np.asarray(norm[k]['scale']);assert np.isfinite(scale).all()and(scale>0).all();metrics[k]={}
 for split in ['train','valid']:
  v=np.concatenate([np.load(e['arrays'][k]['file'])for e in eps if e['split']==split]);z=(v-c)/scale;assert np.isfinite(z).all();np.testing.assert_allclose(z*scale+c,v,atol=2e-6,rtol=2e-6)
  metrics[k][split]=dict(max_abs=float(abs(z).max()),fraction_abs_gt10=float((abs(z)>10).mean()))
report.update(passed=True,episodes=len(eps),unique_recordings=len(by),rows=sum(e['rows']for e in eps),added_rows=sum(e['rows']for e in added),loader=loaders,normalization=metrics,normalizer_provenance='unchanged frozen source-train FT normalizer; no target/validation fitting',normalizer_sha256=sha(release/'normalizer.json'),manifest_sha256=sha(target),original_integrity_passed=True,visual_review_pending=True,training_started=False)
(R/'reports/final-numeric-audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
'''
(R/'scripts/final_audit.py').write_text(s);ast.parse(s)
(R/'scripts/final.sbatch').write_text('#!/bin/bash\nset -euo pipefail\nexport PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1\nexec /coc/flash7/cxin34/mobilemani-pretraining/env-train/bin/python '+str(R/'scripts/final_audit.py')+'\n')
print('Final audit AST OK')
