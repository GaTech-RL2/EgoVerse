"""Independent non-look-at fixed+BS tensor adapter; no upstream mutations."""
import os,json,fcntl,collections
from pathlib import Path
import numpy as np,h5py,torch
from campaign.data import sha,EpisodeDataset
from campaign.contracts import check_normalizer
from retarget_tensor_state import StateConverter

R=Path(os.environ['CAMPAIGN_ROOT']);rel=R/'releases/cart'
lock=open(R/'reports/build.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (rel/'READY.json').exists()
p=json.load(open(R/'reports/input-manifest.json'));pre=json.load(open(R/'reports/input-audit.json'))
assert pre['passed'] and len(pre['records'])==107 and not p['look_at']
original=json.load(open(p['source_manifest']));old={e['uuid']:e for e in original['episodes']}
source=json.load(open('/coc/flash7/cxin34/egoa-cart-bs-ft-20261124/cart/releases/cart/source.json'))
source_ids={e.get('recording_id',e.get('original_uuid',e['uuid'])) for e in source['episodes']}
source_sha={e.get('raw_vrs_sha256',e.get('raw_sha256')) for e in source['episodes']}
lineage={e['recording_id']:e for e in [json.loads(l) for l in (Path(next(iter(old.values()))['release'])/'source_map.jsonl').read_text().splitlines()]}
assert set(old)==set(lineage)
target_sha={lineage[u]['raw_sha256'] for u in old}
assert None not in source_sha and len(target_sha)==107 and not source_sha&target_sha
assert not source_ids&set(old)
cfgpath=next((R/'configs').glob('egoa180*.json'));cfg=json.load(open(cfgpath));norm=json.load(open(rel/'normalizer.json'))
conv=StateConverter();records=[];diagnostics=[];samples=[];stats=collections.defaultdict(list)
lookup={e['uuid']:e for e in pre['records']}
obs=Path(p['observation_root']);bs=Path(p['retarget_root'])
with h5py.File(obs/'retargeted/chuye_cart_1102.hdf5')as h,h5py.File(bs/'retargeted/chuye_cart_1102.hdf5')as g:
 for ix,e in enumerate(p['records']):
  uid=e['uuid'];n=e['rows'];a=h['data'][e['demo']];b=g['data'][e['demo']]
  np.testing.assert_array_equal(a['obs/aria_ts'][:],b['obs/aria_ts'][:])
  np.testing.assert_array_equal(a['obs/robot0_joint_pos'][:n,24:26],b['obs/robot0_joint_pos'][:n,24:26])
  # As in the original BS recipe, preserve original camera-frame observations.
  base=conv.convert(a,n);q=b['obs/robot0_joint_pos'][:n].astype(np.float32);raw=b['actions/joint'][:n]
  data={k:base[k] for k in ['state_eef_pose_glass','timestamp']}
  data.update(action=np.concatenate((raw[:,22:25],raw[:,14:22],raw[:,:14],raw[:,25:]),axis=-1).astype(np.float32),state_robot0_joint_pos=q[:,4:26],state_hand_left_qpos=b['obs/hand_left_qpos'][:n].astype(np.float32),state_hand_right_qpos=b['obs/hand_right_qpos'][:n].astype(np.float32),action_valid=np.ones(n,bool),dino_valid=np.ones(n,bool))
  for key,src,shape in [('front_pcd_1','global',(n,2048,6)),('front_pcd_2','right',(n,1024,6)),('front_pcd_3','left',(n,1024,6)),('dino','dino',(n,16,768)),('gaze_point_glass','gaze',(n,3))]:
   info=lookup[uid]['files'][src];assert sha(info['path'])==info['sha256'];data[key]=np.asarray(np.load(info['path'],mmap_mode='r')).reshape(shape).astype(np.float32)
  # BS does not change hands/actions ordering: compare to accepted BS target.
  for key in ['action','state_robot0_joint_pos','state_hand_left_qpos','state_hand_right_qpos']:
   prev=np.load(old[uid]['arrays'][key]['file'])
   if key=='action':dims=[j for j in range(49)if j not in [9,10]]
   elif key=='state_robot0_joint_pos':dims=list(range(20))
   else:dims=list(range(12))
   np.testing.assert_allclose(data[key][:,dims],prev[:,dims],atol=2e-6,rtol=0)
  np.testing.assert_allclose(data['timestamp'],np.load(old[uid]['arrays']['timestamp']['file']),atol=1e-6,rtol=0)
  # Quantify BS camera shift without pretending observations were reprojected.
  shifts=[]
  for j in range(n):
   th,_=conv.fk(q[j]);ta,_=conv.fk(a['obs/robot0_joint_pos'][j]);shifts.append(np.linalg.norm((th@conv.mount)[:3,3]-(ta@conv.mount)[:3,3]))
  dest=R/'tensors'/uid;dest.mkdir(parents=True,exist_ok=True);arrays={}
  for k,v in data.items():
   assert len(v)==n and np.isfinite(v).all(),(uid,k)
   if k.startswith('front_pcd_'):assert v[...,3:].min()>=0 and v[...,3:].max()<=1
   out=dest/(k+'.npy')
   if out.exists():np.testing.assert_array_equal(np.load(out),v)
   else:np.save(out,v)
   np.testing.assert_array_equal(np.load(out,mmap_mode='r'),v)
   arrays[k]=dict(file=str(out),sha256=sha(out),shape=list(v.shape),dtype=str(v.dtype))
   if k in norm and isinstance(norm[k],dict) and 'center'in norm[k]:
    z=(v-np.asarray(norm[k]['center']))/np.asarray(norm[k]['scale']);assert np.isfinite(z).all()
    np.testing.assert_allclose(z*np.asarray(norm[k]['scale'])+np.asarray(norm[k]['center']),v,atol=2e-6,rtol=2e-6)
    stats[e['split']+'_'+k].append(z)
  rec=dict(old[uid]);rec.update(arrays=arrays,raw_sha256=lineage[uid]['raw_sha256'],retarget_hdf5=str(bs/'retargeted/chuye_cart_1102.hdf5'),retarget_demo=e['demo'],retarget_run='fix1124_bs',run='fix1124_bs',observation_root=str(obs),look_at=False)
  for k in ['parquet','parquet_sha256','release']:rec.pop(k,None)
  records.append(rec);(dest/'episode.json').write_text(json.dumps(rec,indent=2))
  diagnostics.append(dict(uuid=uid,rows=n,split=e['split'],bs_camera_shift_max_m=max(shifts),head_unchanged=True))
  if not any(s['split']==e['split']for s in samples):
   inds=[0,n//2,n-1];samples.append(dict(split=e['split'],uuid=uid,rgb=b['obs/aria_image'][inds],cloud=data['front_pcd_1'][inds],right=data['front_pcd_2'][inds],left=data['front_pcd_3'][inds]))
  print('EXPORTED',ix,uid,n,flush=True)
manifest=dict(task='cart',episodes=records,recording_hz=60,horizon_rows=64,dino_offset_rows=10,quality_passed=True,split_frozen=True,look_at=False,retarget_fix=True,base_smoothing=True,recipe='non-LA fix1124 observations with BS actions/proprio; retained original camera-frame observations',training_release_ready=False)
target=rel/'target.json';target.write_text(json.dumps(manifest,indent=2));check_normalizer(cfg,norm,manifest,sha(target))
normstats={k:dict(max_abs=float(abs(np.concatenate(v)).max()),q999_abs=float(np.quantile(abs(np.concatenate(v)),.999)),fraction_abs_gt10=float((abs(np.concatenate(v))>10).mean()))for k,v in stats.items()}
for k,v in normstats.items():assert v['max_abs']<100 and v['q999_abs']<10 and v['fraction_abs_gt10']<.001,(k,v)
loaders=[]
for split in ['train','valid']:
 ds=EpisodeDataset(target,split,'B',norm)
 for i in range(len(ds)):
  b=ds[i];assert all(torch.isfinite(v).all()for v in b.values());ei,row=ds.index[i];n=len(ds.episodes[ei]['action'])
  assert b['action'].shape==(64,49) and int(b['action_valid'].sum())==min(64,n-row)
  assert bool(b['dino_valid'])==(row+10<n)
 loaders.append(dict(split=split,rows=len(ds),all_rows_finite=True,h64_mask=True,dino10_mask=True))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
for s in samples:
 fig,axs=plt.subplots(4,3,figsize=(12,12))
 for j in range(3):
  axs[0,j].imshow(s['rgb'][j]);axs[0,j].axis('off')
  for row,key in enumerate(['cloud','right','left'],1):
   c=s[key][j];axs[row,j].scatter(c[:,0],c[:,2],c=c[:,3:],s=2);axs[row,j].set_aspect('equal');axs[row,j].set_title(key)
 fig.suptitle(s['split']+' '+s['uuid']);fig.tight_layout();fig.savefig(R/'reports'/('contact-'+s['split']+'.jpg'),dpi=90);plt.close(fig)
report=dict(passed=True,episodes=107,rows=sum(e['rows']for e in records),look_at=False,source_overlap_uuid=0,source_overlap_rawsha=0,loader=loaders,normalization=normstats,diagnostics=diagnostics,visual_review_pending=True,manifest_sha256=sha(target),normalizer_sha256=sha(rel/'normalizer.json'))
(R/'reports/data-independent-audit.json').write_text(json.dumps(report,indent=2));print('DATA_AUDIT_PASS',flush=True)
