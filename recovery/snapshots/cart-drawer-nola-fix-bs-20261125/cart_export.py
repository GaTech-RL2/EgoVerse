"""Non-LA upstream observations + independently verified BS labels; no READY here."""
import os,sys,json,fcntl,hashlib,collections
from pathlib import Path
import numpy as np,h5py
R=Path('/coc/flash7/scratch/cxin34/cart-drawer-nola-fix-bs-20261125')
sys.path.insert(0,'/coc/flash7/cxin34/egoa-target-retarget-fix-20261124/scripts')
from retarget_tensor_state import StateConverter
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb')as f:
  for c in iter(lambda:f.read(8*1024*1024),b''):h.update(c)
 return h.hexdigest()
def save(p,d):p.write_text(json.dumps(d,indent=2))
lock=(R/'cart/reports/export.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert json.load(open(R/'cart/reports/input-audit.json'))['passed']
plan=json.load(open(R/'cart/input-manifest.json'));old=json.load(open(plan['source_manifest']));oldmap={e['uuid']:e for e in old['episodes']}
S=Path('/coc/flash7/cxin34/egoversea-multitask-release-20261121/formal-workspace/releases/r2/source.json')
source=json.load(open(S));srcids={e.get('recording_id',e.get('original_uuid',e['uuid']))for e in source['episodes']};srchashes={e.get('raw_sha256',e.get('raw_vrs_sha256'))for e in source['episodes']};assert None not in srchashes
a=Path(plan['observation_root']);b=Path(plan['retarget_root']);conv=StateConverter();episodes=[];audit=[];maps={};pictures=[]
with h5py.File(a/'retargeted/chuye_cart_1102.hdf5','r')as f,h5py.File(b/'retargeted/chuye_cart_1102.hdf5','r')as g:
 for idx,e in enumerate(plan['records']):
  uid=e['uuid'];old=oldmap[uid];release=Path(old['release'])
  if str(release)not in maps:maps[str(release)]={x['episode_index']:x for x in map(json.loads,(release/'source_map.jsonl').read_text().splitlines())}
  raw=maps[str(release)][e['episode_index']];assert raw['recording_id']==uid and raw['raw_sha256']not in srchashes and uid not in srcids
  n=e['rows'];i=e['episode_index'];x=f['data'][e['demo']];y=g['data'][e['demo']]
  q=y['obs/robot0_joint_pos'][:n];qold=x['obs/robot0_joint_pos'][:n];act=y['actions/joint'][:n]
  data=dict(action=np.concatenate([act[:,22:25],act[:,14:22],act[:,:14],act[:,25:]],-1).astype('float32'),state_robot0_joint_pos=q[:,4:26].astype('float32'),state_hand_left_qpos=y['obs/hand_left_qpos'][:n].astype('float32'),state_hand_right_qpos=y['obs/hand_right_qpos'][:n].astype('float32'),timestamp=y['obs/aria_ts'][:n]/1e9,action_valid=np.ones(n,bool),dino_valid=np.ones(n,bool))
  paths={}
  for key,sub,nm,shape in [('front_pcd_1','clouds_hd2','g_hd2',(n,2048,6)),('front_pcd_2','clouds_hd2','l_hd2',(n,1024,6)),('front_pcd_3','clouds_hd2','l3_hd2',(n,1024,6)),('state_eef_pose_glass','clouds_hd2','p',(n,9)),('gaze_point_glass','clouds_hd2','gp',(n,3)),('dino','dino_wm_r2','dino',(n,16,768))]:
   p=a/sub/(nm+f'_{i:06d}.npy');data[key]=np.load(p).reshape(shape);paths[str(p)]=sha(p)
  # The upstream cloud uses recorded world EEF proprio, not fresh joint FK.
  # Independently reconstruct its base placement and glass transform exactly.
  fkold=conv.convert(x,n)['state_eef_pose_glass']
  # BS modifies the base trajectory; its stored world EEF is not required
  # to share the pre-BS constant-placement convention. Compare joint FK locally.
  old_local=[];new_local=[]
  for row in range(n):
   for qs,dst in [(qold,old_local),(q,new_local)]:
    head,ee=conv.fk(qs[row]);T=np.linalg.inv(head@conv.mount)@ee;dst.append(T[:3,3])
  np.testing.assert_allclose(data['state_eef_pose_glass'],fkold,atol=2e-5,rtol=0,err_msg=uid+' upstream P9 geometry')
  delta=np.linalg.norm(np.asarray(new_local)-np.asarray(old_local),axis=1)
  assert delta.max()<.05,'BS upstream-observation displacement exceeds5cm'
  np.testing.assert_array_equal(q[:,24:26],qold[:,24:26])
  np.testing.assert_array_equal(data['timestamp'],np.load(old['arrays']['timestamp']['file']))
  assert np.allclose(np.diff(data['timestamp']),1/60,rtol=.05,atol=.001)
  out=R/'cart/tensors'/uid;out.mkdir(parents=True,exist_ok=True);arrays={}
  for key,v in data.items():
   assert len(v)==n and np.isfinite(v).all(),(uid,key)
   p=out/(key+'.npy')
   if p.exists():np.testing.assert_array_equal(np.load(p,mmap_mode='r'),v)
   else:np.save(p,v)
   arrays[key]=dict(file=str(p),sha256=sha(p),shape=list(v.shape),dtype=str(v.dtype))
  rec=dict(uuid=uid,recording_id=uid,raw_sha256=raw['raw_sha256'],task='cart',split=e['split'],rows=n,take=e['take'],arrays=arrays,retarget_hdf5=str(b/'retargeted/chuye_cart_1102.hdf5'),retarget_demo=e['demo'],look_at=False,upstream_files=paths)
  save(out/'episode.json',rec);episodes.append(rec);audit.append(dict(uuid=uid,rows=n,old_fk_max_abs=float(abs(data['state_eef_pose_glass']-fkold).max()),bs_camera_frame_eef_delta_max_m=float(delta.max())))
  if idx in [0,20,40,60,80,106]:pictures.append(dict(uid=uid,rgb=x['obs/aria_image'][[0,n//2,n-1]],cloud=data['front_pcd_1'][[0,n//2,n-1]]))
  print('EXPORT_PASS',uid,n,flush=True)
assert len(episodes)==107 and sum(e['rows']for e in episodes)==32798
for k in ['recording_id','raw_sha256']:assert not {e[k]for e in episodes if e['split']=='train'}&{e[k]for e in episodes if e['split']=='valid'}
m=dict(task='cart',episodes=episodes,recording_hz=60,horizon_rows=64,dino_offset_rows=10,quality_passed=True,split_frozen=True,split_seed=42,split_unit='original_recording_uuid',validation_scope='same collection day',look_at=False,recipe='fix1124 non-LA hd2 observations + fix1124_bs labels',bs_observations='upstream camera-frame observations retained as original BS recipe; not exactly reprojected')
save(R/'cart/target.json',m)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
for idx,p in enumerate(pictures):
 fig,ax=plt.subplots(2,3,figsize=(12,7))
 for j in range(3):
  ax[0,j].imshow(p['rgb'][j]);ax[0,j].axis('off');c=p['cloud'][j];ax[1,j].scatter(c[:,0],c[:,2],c=np.clip(c[:,3:],0,1),s=2);ax[1,j].set_aspect('equal');ax[1,j].set_xlabel('X');ax[1,j].set_ylabel('Z')
 fig.suptitle(p['uid']);fig.tight_layout();fig.savefig(R/'cart/reports'/f'contact-{idx}.jpg');plt.close(fig)
save(R/'cart/reports/export-audit.json',dict(passed=True,training_ready=False,source_isolation=True,source_sha256=sha(S),manifest_sha256=sha(R/'cart/target.json'),episodes=audit,visual_review_pending=True))
print('EXPORT_COMPLETE',flush=True)
