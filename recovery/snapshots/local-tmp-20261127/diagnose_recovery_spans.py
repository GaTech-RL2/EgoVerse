from pathlib import Path
import json,numpy as np,h5py,sys
R=Path('/coc/flash7/cxin34/cloth-valid-recovery-fix1124-20261124');sys.path.insert(0,str(R/'scripts'))
from retarget_tensor_state import StateConverter
c=StateConverter();out=[]
def runs(v):
 z=np.diff(np.r_[False,v,False].astype(int));return [(int(a),int(b))for a,b in zip(np.flatnonzero(z==1),np.flatnonzero(z==-1))]
for x in json.load(open(R/'plan.json')):
 e=Path(x['episode'])
 with h5py.File(e/'retarget/selected.hdf5')as h:
  g=h['data/demo_0'];q=g['obs/robot0_joint_pos'][:];base=g['obs/robot_base_pose'][:];eef=g['obs/eef_right_proprio'][:].reshape(-1,4,4);n=len(q);ds=[]
  for i in range(n):
   ang=base[i,2];T=np.eye(4);T[:2,:2]=[[np.cos(ang),-np.sin(ang)],[np.sin(ang),np.cos(ang)]];T[:2,3]=base[i,:2];ds.append(eef[i,:3,3]-(T@c.fk(q[i])[1])[:3,3])
  ds=np.array(ds);res=np.linalg.norm(ds-np.median(ds,axis=0),axis=1);good=res<=.005
  badstep=(np.abs(np.diff(q[:,10:24],axis=0)).max(1)>.15)|(np.linalg.norm(np.diff(base[:,:2],axis=0),axis=1)>.03)
  ids=np.flatnonzero(badstep);good[ids]=False;good[ids+1]=False
  rr=[(a,b)for a,b in runs(good)if b-a>=64]
  out.append(dict(index=x['index'],take=x['take'],uuid=x['uuid'],median_fk_m=float(np.median(res)),p95_fk_m=float(np.quantile(res,.95)),bad_step_count=int(badstep.sum()),strict_quality_subspans=rr,subspan_rows=sum(b-a for a,b in rr)))
(R/'reports/quality-subspan-diagnosis.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
