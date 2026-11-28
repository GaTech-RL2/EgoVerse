import json,hashlib,sys
from pathlib import Path
import numpy as np,h5py
R=Path('/coc/flash7/scratch/cxin34/cart-drawer-nola-fix-bs-20261125');m=json.load(open(R/'drawer/segment-manifest.json'))
def fit(x,y):
 u,s,vt=np.linalg.svd((x-x.mean(0)).T@(y-y.mean(0)));r=vt.T@u.T
 if np.linalg.det(r)<0:vt[-1]*=-1;r=vt.T@u.T
 return r,y.mean(0)-r@x.mean(0)
for arg in sys.argv[1:]:
 i=int(arg);e=m[i];p=Path(e['segment']);a=np.load(p/'retarget/work'/(e['retarget_tag']+'_aria60.npz'))
 with h5py.File(p/'retarget/fix.hdf5')as f:
  g=f['data/demo_0'];ts=g['obs/aria_ts'][:].astype(float);fi=np.clip(np.searchsorted(a['ts']*1000,ts-500),0,len(a['ts'])-1);assert np.max(abs(a['ts'][fi]*1000-ts))<1000
  xs=[];ys=[]
  for j in range(0,len(ts),2):
   for side,off in [('right',[.0265,.0065,-.1077,1]),('left',[-.0265,.0065,-.1077,1])]:
    xs.append(a[side+'_landmarks'][fi[j],8]);ys.append((g['obs/eef_'+side+'_proprio'][j].reshape(4,4)@np.array(off))[:3])
 x,y=np.array(xs),np.array(ys);r,t=fit(x,y);res=np.linalg.norm(x@r.T+t-y,axis=1);inl=res<.01
 if inl.sum()>=12:r,t=fit(x[inl],y[inl]);res=np.linalg.norm(x@r.T+t-y,axis=1);inl=res<.01
 stats=dict(index=i,uuid=e['uuid'],pairs=len(x),inlier_pairs=int(inl.sum()),median_cm=float(np.median(res)*100),p90_cm=float(np.quantile(res,.9)*100),inlier_median_cm=float(np.median(res[inl])*100)if inl.any()else None,each_hand={side:dict(median_cm=float(np.median(res[k::2])*100),max_cm=float(res[k::2].max()*100))for k,side in enumerate(['right','left'])},passed=bool(inl.sum()>=max(12,.5*len(x))and np.median(res[inl])<.005),method='same fixed 1cm/50%-pairs,5mm inlier median gate; read-only independent numpy implementation',training_ready=False)
 print(json.dumps(stats),flush=True)
 out=R/'drawer/reports'/f'registration-{i:03d}.json'
 if not out.exists():out.write_text(json.dumps(stats,indent=2))
