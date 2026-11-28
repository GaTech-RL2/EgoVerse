"""Replace only three-cloud observations in the frozen non-LA BS target."""
import json,copy,hashlib,fcntl
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/coc/flash7/scratch/cxin34/cart-drawer-nola-fix-bs-20261125');O=R/'cart-novfilter';(O/'reports').mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
lock=(O/'reports/export.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
m=json.load(open(R/'cart/target.json'));plan=json.load(open(R/'cart/input-manifest.json'));records={e['uuid']:e for e in plan['records']};a=json.load(open(R/'cart/reports/export-audit.json'));assert a['passed'] and a['source_isolation'] and a['manifest_sha256']==sha(R/'cart/target.json')
recipe=json.load(open(O/'recipe.json'));assert recipe['virtual_camera_filter'] is False
audit=[]
for ix,e in enumerate(m['episodes']):
 i=records[e['uuid']]['episode_index'];n=e['rows'];proof=O/'clouds_hd2'/f'proof_{i:06d}.json';p=json.load(open(proof));assert p['passed'] and p['virtual_camera_filter'] is False and p['rows']==n
 gz=json.load(open(O/'clouds_hd2'/f'gz_{i:06d}.json'));assert gz['vis_frac_median'] is None
 for key,nm in [('state_eef_pose_glass','p')]:
  new=O/'clouds_hd2'/f'{nm}_{i:06d}.npy';np.testing.assert_array_equal(np.load(new),np.load(e['arrays'][key]['file']))
 changes={};out=O/'tensors'/e['uuid'];out.mkdir(parents=True,exist_ok=True)
 for key,nm,count in [('front_pcd_1','g_hd2',2048),('front_pcd_2','l_hd2',1024),('front_pcd_3','l3_hd2',1024)]:
  src=O/'clouds_hd2'/f'{nm}_{i:06d}.npy';assert sha(src)==p['files'][nm]
  v=np.load(src).reshape(n,count,6);assert np.isfinite(v).all() and v[:,:,3:].min()>=0 and v[:,:,3:].max()<=1
  old=np.load(e['arrays'][key]['file']);changes[key]=dict(different=not np.array_equal(v,old),mean_radius_m=float(np.linalg.norm(v[:,:,:3],axis=-1).mean()))
  dst=out/(key+'.npy')
  if dst.exists():np.testing.assert_array_equal(np.load(dst),v)
  else:np.save(dst,v)
  e['arrays'][key]=dict(file=str(dst),sha256=sha(dst),shape=list(v.shape),dtype=str(v.dtype))
 # Visibility filtering also changes the gaze-ray point intersection; retain the
 # recomputed point so it describes this observation rather than the ON cloud.
 src=O/'clouds_hd2'/f'gp_{i:06d}.npy';assert sha(src)==p['files']['gp'];v=np.load(src);dst=out/'gaze_point_glass.npy';np.save(dst,v)
 e['arrays']['gaze_point_glass']=dict(file=str(dst),sha256=sha(dst),shape=list(v.shape),dtype=str(v.dtype))
 e['virtual_camera_filter']=False;e['cloud_proof_sha256']=sha(proof)
 audit.append(dict(uuid=e['uuid'],rows=n,changes=changes,visibility_report= gz['vis_frac_median']))
 if ix in [0,20,40,60,80,106]:
  fig,ax=plt.subplots(2,3,figsize=(12,7))
  for j,key in enumerate(['front_pcd_1','front_pcd_2','front_pcd_3']):
   new=np.load(e['arrays'][key]['file']);old_ep=next(x for x in json.load(open(R/'cart/target.json'))['episodes'] if x['uuid']==e['uuid']);old=np.load(old_ep['arrays'][key]['file'])
   for row,v,label in [(0,old,'ON'),(1,new,'OFF')]:
    c=v[n//2];ax[row,j].scatter(c[:,0],c[:,2],c=np.clip(c[:,3:],0,1),s=2);ax[row,j].set_aspect('equal');ax[row,j].set_title(label+' '+key)
  fig.suptitle(e['uuid']);fig.tight_layout();fig.savefig(O/'reports'/f'compare-{ix:03d}.jpg');plt.close(fig)
 print('NOVFILTER_EXPORT',i,n,flush=True)
assert len(audit)==107 and any(x['changes']['front_pcd_1']['different']for x in audit)
m['virtual_camera_filter']=False;m['recipe']='fix1124 non-LA hd2 with TOY_VIS=0 + identical BS labels';(O/'target.json').write_text(json.dumps(m,indent=2))
(O/'reports/export-audit.json').write_text(json.dumps(dict(passed=True,training_ready=False,source_isolation=True,manifest_sha256=sha(O/'target.json'),matched_on_manifest_sha256=sha(R/'cart/target.json'),unchanged='split, BS action/state, DINO, timestamps, source recordings',changed='three clouds and observation-dependent gaze intersection',episodes=audit,visual_review_pending=True),indent=2))
