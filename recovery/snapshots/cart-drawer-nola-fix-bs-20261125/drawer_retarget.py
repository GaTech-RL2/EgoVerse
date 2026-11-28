"""Owned on-device CPF/measured-wrist h1.57 retarget, then BS; no clouds/MPS."""
import sys,os,json,subprocess,hashlib,traceback,fcntl
from pathlib import Path
import numpy as np,h5py,pandas as pd
R=Path('/coc/flash7/scratch/cxin34/cart-drawer-nola-fix-bs-20261125');e=json.load(open(R/'drawer/segment-manifest.json'))[int(sys.argv[1])];p=Path(e['segment']);W=p/'retarget/work';tag=e['retarget_tag'];A=R/'drawer/source/aria_egoposer'
lock=(p/'retarget.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
if(p/'retarget-proof.json').exists():assert json.load(open(p/'retarget-proof.json'))['passed'];raise SystemExit(0)
assert not(p/'retarget.intent.json').exists(),'Unresolved attempt; audit required'
(p/'retarget.intent.json').write_text(json.dumps(dict(job=os.getenv('SLURM_JOB_ID'),segment=e['uuid'],head_height_m=1.57,mps=False,look_at=False)))
env=os.environ.copy();env.pop('PYTHONPATH',None);env.pop('VIRTUAL_ENV',None)
EG='/coc/flash7/czhang883/Documents/mobile_generalist/data/aria_egoposer_warp_handoff/aria_egoposer_warp_handoff/02_egoposer'
env.update(ARIA_ARTIFACTS=str(W),EGOPOSER_ROOT=EG,SEW_REPO='/coc/flash7/czhang883/Documents/SEW-aria-pipeline',PYTHONDONTWRITEBYTECODE='1',MUJOCO_GL='egl',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',FIX_HEAD_HEIGHT='1.57',ARIA_HANDS='ondevice',ARIA_WRIST='measured',BASEFIX_LOG=str(p/'logs/basefix-checks.jsonl'))
py='/coc/flash7/czhang883/miniconda3/envs/rby1/bin/python';aria='/coc/flash7/czhang883/miniconda3/envs/aria2/bin/python';ego='/coc/flash7/czhang883/miniconda3/envs/egoposer/bin/python'
def run(name,args,cwd=None):
 with(p/'logs'/(name+'.log')).open('x')as f:subprocess.run(args,env=env,cwd=cwd,stdout=f,stderr=subprocess.STDOUT,check=True)
def sha(f):
 h=hashlib.sha256()
 with open(f,'rb')as b:
  for c in iter(lambda:b.read(8388608),b''):h.update(c)
 return h.hexdigest()
try:
 a=np.load(W/(tag+'_aria60.npz'));assert a['left_valid'].all()and a['right_valid'].all();assert a['ts'][-1]-a['ts'][0]>1e6
 run('rgb',[aria,str(A/'extract_aria_rgb.py'),'--vrs',e['vrs'],'--tag',tag,'--res','224'])
 run('egoposer',[ego,str(A/'run_aria_egoposer.py'),'--aria_npz',str(W/(tag+'_aria60.npz')),'--out',str(W/(tag+'_smpl.npz')),'--head_height','1.57'],EG)
 run('bone',[py,str(A/'write_bone_csv.py'),'--smpl_npz',str(W/(tag+'_smpl.npz')),'--out',str(W/(tag+'_human.csv')),'--wrist','measured'])
 s=np.load(W/(tag+'_smpl.npz'));assert abs(s['head_p_in'][0,2]-1.57)<1e-6
 csv=pd.read_csv(W/(tag+'_human.csv'));errs={}
 for side,bid in [('left',19),('right',45)]:
  actual=csv.loc[csv.id==bid,['pos_x','pos_y','pos_z']].to_numpy();expected=s[side+'_landmarks'][:,5];errs[side]=float(np.linalg.norm(actual-expected,axis=1).max());assert errs[side]<1e-6
 fix=p/'retarget/fix.hdf5';bs=p/'retarget/fix_bs.hdf5'
 run('fix',[py,str(A/'build_training_hdf5.py'),'--no_firm','--res','224','--tags',tag,'--out',str(fix)])
 run('bs',[py,str(R/'drawer/source/bs/build_hdf5_basefix.py'),'--no_firm','--res','224','--tags',tag,'--out',str(bs)])
 run('bs-gate',[py,'-I',str(R/'drawer/source/bs/gate_basefix.py'),str(fix),str(bs),str(p/'bs-gate.json')])
 gate=json.load(open(p/'bs-gate.json'));assert not gate['fails']
 with h5py.File(fix)as f,h5py.File(bs)as b:
  x=f['data/demo_0'];y=b['data/demo_0'];ts=y['obs/aria_ts'][:]/1e9;assert ts[-1]-ts[0]>1 and np.allclose(np.diff(ts),1/60,rtol=.05,atol=.001)
  np.testing.assert_array_equal(x['obs/aria_ts'][:],y['obs/aria_ts'][:]);np.testing.assert_array_equal(x['obs/aria_image'][:],y['obs/aria_image'][:]);assert y['obs/aria_image'][:].max()>0
  for key in ['obs/robot0_joint_pos','actions/joint','obs/eef_right_proprio','obs/eef_left_proprio']:assert np.isfinite(y[key][:]).all()
  rows=len(ts)
 run('retarget-qc',[py,'/coc/flash7/cxin34/cloth-new50-h168-20261123/source/glitch_scan.py',str(bs)])
 qc=(p/'logs/retarget-qc.log').read_text();passed='<--'not in qc
 result=dict(passed=passed,quality_quarantined=not passed,segment=e['uuid'],recording_id=e['recording_id'],rows=rows,head_height_m=1.57,measured_wrist_max_error=errs,look_at=False,mps_requested=False,bs_gate_sha256=sha(p/'bs-gate.json'),fix_sha256=sha(fix),bs_sha256=sha(bs),qc_log_sha256=sha(p/'logs/retarget-qc.log'),training_ready=False,downstream='depth/hd2/DINO/tensor QC still required')
 (p/'retarget-proof.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));assert passed,'Original retarget quality gate flagged segment'
except Exception:
 (p/'processing-error.json').write_text(json.dumps(dict(job=os.getenv('SLURM_JOB_ID'),segment=e['uuid'],error=traceback.format_exc(),training_ready=False),indent=2));raise
