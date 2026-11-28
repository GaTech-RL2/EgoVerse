"""BS builder deliberately omits RGB; copy the identical source frames, then QC."""
import sys,json,os,shutil,subprocess,hashlib
from pathlib import Path
import numpy as np,h5py,pandas as pd
R=Path('/coc/flash7/scratch/cxin34/cart-drawer-nola-fix-bs-20261125');e=json.load(open(R/'drawer/segment-manifest.json'))[int(sys.argv[1])];p=Path(e['segment']);W=p/'retarget/work';tag=e['retarget_tag']
def sha(f):
 h=hashlib.sha256()
 with open(f,'rb')as b:
  for c in iter(lambda:b.read(8388608),b''):h.update(c)
 return h.hexdigest()
fix=p/'retarget/fix.hdf5';bs=p/'retarget/fix_bs.hdf5';backup=bs.with_name('fix_bs.before_rgb_merge.hdf5')
if not backup.exists():shutil.copy2(bs,backup)
for f in [p/'bs-gate.json',p/'logs/bs-gate.log']:
 if f.exists() and not f.with_name(f.name+'.before_rgb_merge').exists():shutil.copy2(f,f.with_name(f.name+'.before_rgb_merge'))
with h5py.File(fix)as f,h5py.File(bs,'r+')as b:
 x=f['data/demo_0'];y=b['data/demo_0'];np.testing.assert_array_equal(x['obs/aria_ts'][:],y['obs/aria_ts'][:]);assert x['obs/aria_image'].shape==y['obs/aria_image'].shape
 rgb=x['obs/aria_image'][:];assert rgb.std()>3 and (rgb>0).mean()>.1
 y['obs/aria_image'][:]=rgb
with h5py.File(backup)as before,h5py.File(bs)as after:
 def check(name,v):
  if isinstance(v,h5py.Dataset) and not name.endswith('obs/aria_image'):np.testing.assert_array_equal(v[:],after[name][:],err_msg=name)
 before.visititems(check)
env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1';env['MUJOCO_GL']='egl';py='/coc/flash7/czhang883/miniconda3/envs/rby1/bin/python'
with (p/'logs/bs-gate-rgb-merge.log').open('x')as log:subprocess.run([py,'-I',str(R/'drawer/source/bs/gate_basefix.py'),str(fix),str(bs),str(p/'bs-gate-rgb-merge.json')],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
gate=json.load(open(p/'bs-gate-rgb-merge.json'));assert not gate['fails'],gate['fails']
with h5py.File(fix)as f,h5py.File(bs)as b:
 x=f['data/demo_0'];y=b['data/demo_0'];ts=y['obs/aria_ts'][:]/1e9;assert ts[-1]-ts[0]>1 and np.allclose(np.diff(ts),1/60,rtol=.05,atol=.001)
 np.testing.assert_array_equal(x['obs/aria_image'][:],y['obs/aria_image'][:])
 for key in ['obs/robot0_joint_pos','actions/joint','obs/eef_right_proprio','obs/eef_left_proprio']:assert np.isfinite(y[key][:]).all()
 rows=len(ts)
s=np.load(W/(tag+'_smpl.npz'));assert abs(s['head_p_in'][0,2]-1.57)<1e-6;csv=pd.read_csv(W/(tag+'_human.csv'));errs={}
for side,bid in [('left',19),('right',45)]:
 actual=csv.loc[csv.id==bid,['pos_x','pos_y','pos_z']].to_numpy();expected=s[side+'_landmarks'][:,5];errs[side]=float(np.linalg.norm(actual-expected,axis=1).max());assert errs[side]<1e-6
with(p/'logs/retarget-qc.log').open('x')as log:subprocess.run([py,'/coc/flash7/cxin34/cloth-new50-h168-20261123/source/glitch_scan.py',str(bs)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
passed='<--'not in(p/'logs/retarget-qc.log').read_text()
result=dict(passed=passed,quality_quarantined=not passed,segment=e['uuid'],recording_id=e['recording_id'],rows=rows,head_height_m=1.57,measured_wrist_max_error=errs,look_at=False,mps_requested=False,bs_gate_sha256=sha(p/'bs-gate-rgb-merge.json'),fix_sha256=sha(fix),bs_sha256=sha(bs),qc_log_sha256=sha(p/'logs/retarget-qc.log'),rgb_merge=dict(source=str(fix),previous_hdf5_sha256=sha(backup),only_RGB_changed=True),training_ready=False,downstream='depth/hd2/DINO/tensor QC still required')
(p/'retarget-proof.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));assert passed,'Quality quarantine; no downstream for this segment'
