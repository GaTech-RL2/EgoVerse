"""Freeze normalization and run every real loader row separately for each run."""
import json,sys,os,fcntl,copy,collections
from pathlib import Path
import numpy as np,torch
R=Path('/coc/flash7/scratch/cxin34/cart-drawer-nola-fix-bs-20261125');stage=sys.argv[1];W=R/'training/cart'/stage
sys.path.insert(0,str(W));sys.path.insert(0,str(W/'source'));sys.path.insert(0,str(W/'scripts'))
from campaign.data import EpisodeDataset,sha
from campaign.contracts import check_normalizer
lock=(W/'reports/release.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
rel=W/'releases/cart';assert not(rel/'READY.json').exists()
audit=json.load(open(R/'cart/reports/export-audit.json'));assert audit['passed'] and audit['source_isolation']
assert sha(R/'cart/target.json')==audit['manifest_sha256']
visual=json.load(open(R/'cart/reports/visual-review.json'));assert visual['passed'] and visual['manifest_sha256']==audit['manifest_sha256']
for p,h in visual['images'].items():assert sha(p)==h
m=json.load(open(R/'cart/target.json'));target=rel/'target.json';target.write_text(json.dumps(m,indent=2))
cfgpath=next(p for p in(W/'configs').glob('*.json'));cfg=json.load(open(cfgpath));torch.set_num_threads(4)
keys=['action','state_robot0_joint_pos','state_eef_pose_glass','state_hand_left_qpos','state_hand_right_qpos']
counts=collections.Counter(e['split']for e in m['episodes']);assert counts=={'train':97,'valid':10}
for e in m['episodes']:
 for k,a in e['arrays'].items():
  assert sha(a['file'])==a['sha256'];x=np.load(a['file'],mmap_mode='r');assert list(x.shape)==a['shape']and str(x.dtype)==a['dtype']and np.isfinite(x).all()
if stage=='scratch':
 from scratch_norm import fit_and_audit
 norm,stats=fit_and_audit(W,cfg,m)
else:
 source=Path('/coc/flash7/cxin34/egoversea-multitask-release-20261121/formal-workspace/releases/r2/normalizer.json');assert sha(source)==cfg['source_identity']['normalizer_sha256']
 norm=json.load(open(source));before=copy.deepcopy(norm);changes={'action':[3,8],'state_robot0_joint_pos':[0,5]}
 for k,inds in changes.items():
  x=np.concatenate([np.load(e['arrays'][k]['file'])for e in m['episodes']if e['split']=='train'])
  for j in inds:
   lo,hi=np.quantile(x[:,j],[.01,.99]);mn,mx=x[:,j].min(),x[:,j].max();sparse=hi-lo<.05*(mx-mn)
   if sparse:lo,hi=mn,mx
   scale=(hi-lo+1e-6)/2
   for field,v in dict(center=lo+scale,scale=scale,quantile_1=lo,quantile_99=hi,min=mn,max=mx,sparse_fallback=bool(sparse)).items():norm[k][field][j]=bool(v)if field=='sparse_fallback'else float(v)
 for k in keys:
  for field,values in norm[k].items():
   for j,v in enumerate(values):
    if j not in changes.get(k,[]):assert v==before[k][field][j]
 norm['adaptation']=dict(kind='target_train_only_newly_active_source_constant_dimensions',source_normalizer_sha256=sha(source),train_uuids=[e['uuid']for e in m['episodes']if e['split']=='train'],valid_used=False,dimensions=changes)
 (rel/'normalizer.json').write_text(json.dumps(norm,indent=2))
 cfg['normalizer_adaptation']=dict(passed=True,changes=changes,source_sha256=sha(source),adapted_sha256=sha(rel/'normalizer.json'),train_recordings=97,valid_used=False);cfgpath.write_text(json.dumps(cfg,indent=2))
 parent=W/'runs'/cfg['parent']/'step_180000.pt';assert sha(parent)==cfg['parent_checkpoint_sha256']
check_normalizer(cfg,norm,m,sha(target));stats={}
for split in ['train','valid']:
 stats[split]={}
 for k in keys:
  x=np.concatenate([np.load(e['arrays'][k]['file'])for e in m['episodes']if e['split']==split]);z=(x-np.asarray(norm[k]['center']))/np.asarray(norm[k]['scale']);assert np.isfinite(z).all()
  v=dict(max_abs=float(abs(z).max()),q999_abs=float(np.quantile(abs(z),.999)),fraction_abs_gt10=float((abs(z)>10).mean()));assert v['max_abs']<100 and v['q999_abs']<10 and v['fraction_abs_gt10']<.001,(split,k,v)
  stats[split][k]=v
loader=[]
for split in ['train','valid']:
 ds=EpisodeDataset(target,split,'B',norm)
 for i in range(len(ds)):
  b=ds[i];ei,row=ds.index[i];a=ds.episodes[ei];n=len(a['action']);assert all(torch.isfinite(v).all()for v in b.values());assert b['action'].shape==(64,49)and b['dino'].shape==(16,768)
  assert int(b['action_valid'].sum())==min(64,n-row)and bool(b['dino_valid'])==(row+10<n)
  if row in [0,n-1]:np.testing.assert_array_equal(b['dino'].numpy(),a['dino'][min(row+10,n-1)])
  if i%5000==0:print('LOADER',stage,split,i,len(ds),flush=True)
 loader.append(dict(split=split,rows=len(ds),passed=True))
report=dict(passed=True,run=cfg['id'],counts=dict(counts),loader=loader,normalization=stats,manifest_sha256=sha(target),normalizer_sha256=sha(rel/'normalizer.json'),visual_sha256=sha(R/'cart/reports/visual-review.json'),export_audit_sha256=sha(R/'cart/reports/export-audit.json'),config_sha256=sha(cfgpath),source_isolation=True)
(W/'reports/data-independent-audit.json').write_text(json.dumps(report,indent=2))
ready=dict(passed=True,normalizer_frozen=True,retarget_verified=True,recording_hz=60,horizon_rows=64,dino_offset_rows=10,manifests={'target':sha(target)},normalizer_sha256=sha(rel/'normalizer.json'),independent_audit_sha256=sha(W/'reports/data-independent-audit.json'),look_at=False,virtual_camera_filter=True,smoke_required_before_formal=True)
with(rel/'READY.json').open('x')as f:json.dump(ready,f,indent=2)
print('READY',stage,flush=True)
