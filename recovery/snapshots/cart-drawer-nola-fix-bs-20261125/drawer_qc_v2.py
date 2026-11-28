"""Independent frame-correct BS FK audit; never changes data or threshold."""
import sys,json,hashlib
from pathlib import Path
import h5py,numpy as np
R=Path('/coc/flash7/scratch/cxin34/cart-drawer-nola-fix-bs-20261125')
sys.path.insert(0,str(R/'drawer/source/bs'));from base_headings import zp,wrap
sys.path.insert(0,'/coc/flash7/cxin34/egoa-target-retarget-fix-20261124/scripts');from retarget_tensor_state import StateConverter

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb')as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def audit(index):
 e=json.load(open(R/'drawer/segment-manifest.json'))[index];p=Path(e['segment']);W=p/'retarget/work';tag=e['retarget_tag'];out=p/'retarget-proof-v2.json'
 if out.exists():return json.load(open(out))
 gatep=p/'bs-gate-rgb-merge.json';gate=json.load(open(gatep));fix=p/'retarget/fix.hdf5';bs=p/'retarget/fix_bs.hdf5';errs=list(gate['fails']);stats={};conv=StateConverter()
 with h5py.File(fix)as f,h5py.File(bs)as b:
  x=f['data/demo_0'];g=b['data/demo_0'];q=g['obs/robot0_joint_pos'][:];bp=g['obs/robot_base_pose'][:];n=len(q)
  np.testing.assert_array_equal(x['obs/aria_image'][:],g['obs/aria_image'][:]);np.testing.assert_array_equal(x['obs/aria_ts'][:],g['obs/aria_ts'][:]);ts=g['obs/aria_ts'][:]/1e9
  assert ts[-1]-ts[0]>1 and np.allclose(np.diff(ts),1/60,rtol=.05,atol=.001)
  yaw=x['obs/robot_base_pose'][:n,2].astype(float);P=np.load(W/(tag+'_smpl.npz'))['positions'][:n].astype(float);a=np.load(W/(tag+'_aria60.npz'))['head_R'][:n]
  yv=P[:,16]-P[:,17];yv/=np.linalg.norm(yv,axis=1,keepdims=True);tv=(P[:,16]+P[:,17])/2-P[:,6];xv=np.cross(-tv,yv);xv/=np.linalg.norm(xv,axis=1,keepdims=True);rb=np.stack([xv,yv,np.cross(xv,yv)],axis=2);u,_,vt=np.linalg.svd(rb);rb=u@vt;ys=np.arctan2(rb[:,1,0],rb[:,0,0]);yh=np.arctan2(a[:,1,0],a[:,0,0]);d=wrap(yaw-ys);off=np.arctan2(np.sin(d).mean(),np.cos(d).mean());assert np.degrees(np.sqrt(np.mean(wrap(d-off)**2)))<.5
  yn=zp(np.unwrap(yaw+wrap(yh+off-yaw)),.5);heading_error=float(abs(wrap(bp[:,2]-(yn-yn[0]))).max());assert heading_error<1e-5,heading_error
  co,si=np.cos(yn[0]),np.sin(yn[0]);initial_R=np.array([[co,-si,0],[si,co,0],[0,0,1]])
  local={s:[]for s in ['right','left']}
  for z,qq in zip(bp,q):
   conv.fk(qq);co,si=np.cos(z[2]),np.sin(z[2]);rot=np.array([[co,-si,0],[si,co,0],[0,0,1]])
   for side in local:local[side].append(initial_R@(rot@conv.d.body(side+'_eef').xpos+np.r_[z[:2],0]))
  for side in local:
   er=g['obs/eef_'+side+'_proprio'][:].reshape(-1,4,4)[:,:3,3];dd=er-np.array(local[side]);res=np.linalg.norm(dd-np.median(dd,0),axis=1);bad=int((res>.02).sum());stats[side]=dict(frames_gt20mm=bad,p99_mm=float(np.quantile(res,.99)*1000),max_mm=float(res.max()*1000))
   if bad>10:errs.append(side+' FK residual exceeds original 20mm/10-frame gate')
  basesteps=int((np.linalg.norm(np.diff(bp[:,:2],axis=0),axis=1)>.03).sum())
  if basesteps>10:errs.append('base >30mm steps exceeds 10-frame gate')
  for k in ['obs/robot0_joint_pos','actions/joint','obs/eef_right_proprio','obs/eef_left_proprio']:assert np.isfinite(g[k][:]).all()
 result=dict(passed=not errs,quality_quarantined=bool(errs),fails=errs,index=index,uuid=e['uuid'],recording_id=e['recording_id'],rows=n,head_height_m=1.57,look_at=False,mps_requested=False,fix_sha256=sha(fix),bs_sha256=sha(bs),bs_gate_sha256=sha(gatep),audit_source_sha256=sha(__file__),heading_reconstruction_max_rad=heading_error,initial_heading_deg=float(np.degrees(yn[0])),fk=stats,base_large_steps=basesteps,old_thresholds_preserved=True,left_hand_additionally_checked=True,old_attempts_preserved=True,training_ready=False)
 with out.open('x')as f:json.dump(result,f,indent=2)
 return result
if __name__=='__main__':
 for arg in sys.argv[1:]:
  r=audit(int(arg));print(json.dumps(r),flush=True)
