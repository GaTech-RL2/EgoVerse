"""Accepted fixed+BS segment -> shared-per-recording FFS depth -> hd2/DINO/tensors."""
import os,sys,json,subprocess,hashlib,time,fcntl,shutil,traceback
from pathlib import Path
import numpy as np,h5py
R=Path('/coc/flash7/scratch/cxin34/cart-drawer-nola-fix-bs-20261125');D=R/'drawer/source/downstream';e=json.load(open(R/'drawer/segment-manifest.json'))[int(sys.argv[1])];p=Path(e['segment']);out=p/'downstream';out.mkdir(exist_ok=True)
PY='/coc/flash7/cxin34/mobilemani-pretraining/env-train/bin/python';DP='/coc/flash7/scratch/czhang883/envs/aria_fs/bin/python'
sys.path.insert(0,'/coc/flash7/cxin34/egoa-target-retarget-fix-20261124/scripts');from retarget_tensor_state import StateConverter
def sha(f):
 h=hashlib.sha256()
 with open(f,'rb')as b:
  for v in iter(lambda:b.read(8388608),b''):h.update(v)
 return h.hexdigest()
def save(f,x):f.write_text(json.dumps(x,indent=2))
lock=(out/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
if(out/'proof.json').exists():assert json.load(open(out/'proof.json'))['passed'];raise SystemExit(0)
rp=json.load(open(p/'retarget-proof.json'));assert rp['passed'] and not rp['quality_quarantined']
fix=p/'retarget/fix.hdf5';bs=p/'retarget/fix_bs.hdf5';assert sha(fix)==rp['fix_sha256'] and sha(bs)==rp['bs_sha256']
assert not(out/'intent.json').exists(),'Unresolved downstream attempt requires audit'
save(out/'intent.json',dict(job=os.getenv('SLURM_JOB_ID'),segment=e['uuid'],retarget_proof_sha256=sha(p/'retarget-proof.json')))
for n in ['logs','input','rgbd','clouds','dino','tensors','cache']:(out/n).mkdir(exist_ok=True)
env=os.environ.copy();env.pop('VIRTUAL_ENV',None);env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',MUJOCO_GL='egl',PYTHONPATH='/coc/flash7/cxin34/human-pretraining-hd2-20261107/source',HF_HUB_OFFLINE='1',HF_HOME='/coc/flash7/czhang883/.cache/huggingface',MPLCONFIGDIR=str(out/'cache/mpl'),NUMBA_CACHE_DIR=str(out/'cache/numba'),XDG_CACHE_HOME=str(out/'cache'),TMPDIR=str(out/'cache'))
def run(name,cmd,env=env,cwd=None):
 status=out/(name+'-stage.json')
 if status.exists():assert json.load(open(status))['returncode']==0;return
 d=dict(command=cmd,job=os.getenv('SLURM_JOB_ID'),started=time.time());save(status,d)
 with(out/'logs'/(name+'.log')).open('x')as f:r=subprocess.run(cmd,env=env,cwd=cwd,stdout=f,stderr=subprocess.STDOUT)
 d.update(returncode=r.returncode,finished=time.time());save(status,d);assert r.returncode==0,name
try:
 assert shutil.disk_usage(R).free>150*2**30,'scratch free space floor'
 record=R/'drawer/episodes'/e['recording_id'];depthroot=R/'drawer/depth'/e['recording_id'];depthroot.mkdir(exist_ok=True);depth=depthroot/Path(e['vrs']).stem
 dl=(depthroot/'depth.lock').open('a');fcntl.flock(dl,fcntl.LOCK_EX)
 if not(depthroot/'proof.json').exists():
  assert not(depthroot/'intent.json').exists(),'Existing failed depth attempt needs audit'
  cmd=[DP,'src/run_fast_stereo_depth_to_npy.py','--vrs',e['vrs'],'--output-dir',str(depth),'--frame-step','4','--fs-home','/coc/flash7/yzhu827/Fast-FoundationStereo','--ckpt','/coc/flash7/yzhu827/Fast-FoundationStereo/weights/23-36-37/model_best_bp2_serialize.pth']
  save(depthroot/'intent.json',dict(command=cmd,job=os.getenv('SLURM_JOB_ID'),recording_id=e['recording_id']))
  with(depthroot/'depth.log').open('x')as f:r=subprocess.run(cmd,cwd='/coc/flash7/czhang883/Documents/aria_gen2_scripts',env=env,stdout=f,stderr=subprocess.STDOUT)
  assert r.returncode==0,'depth failed'
  files=list(depth.glob('*.npy'));assert len(files)>10
  for f in files:
   a=np.load(f);assert a.shape==(512,512)and np.isfinite(a).all()
  save(depthroot/'proof.json',dict(passed=True,raw_sha256=e['raw_sha256'],frames=len(files),log_sha256=sha(depthroot/'depth.log')))
 assert json.load(open(depthroot/'proof.json'))['raw_sha256']==e['raw_sha256'];fcntl.flock(dl,fcntl.LOCK_UN)
 link=out/'input'/Path(e['vrs']).name
 if not link.exists():link.symlink_to(e['vrs'])
 (out/'manifest.tsv').write_text(e['retarget_tag']+'\tdrawer\t'+Path(e['vrs']).name+'\tok\n')
 with h5py.File(fix)as f:
  g=f['data/demo_0'];n=len(g['obs/aria_ts']);assert n==rp['rows'];ts=g['obs/aria_ts'][:];assert (ts[-1]-ts[0])/1e9>1
 save(out/'cuts.json',dict(cuts={'0':n}))
 cal=json.load(open(record/'stream-audit.json'))['T_device_cpf'];save(p/'retarget/work'/(e['retarget_tag']+'_extr.json'),dict(T_device_cpf=cal))
 env.update(TOY_SRC=str(fix),TOY_H5=str(fix),TOY_MAN=str(out/'manifest.tsv'),TOY_VRSD=str(out/'input'),DEPD=str(depthroot),OUTD=str(out/'rgbd'),SHARD='0',NSHARD='1',TOY_MRG=str(out/'rgbd/shard_0.hdf5'),TOY_SHARDS=str(out/'rgbd'),TOY_OUTD=str(out/'clouds'),MPS_ROOT=str(out/'unused-no-mps'),GAZE_CSV=str(record/'gaze/ondevice_eye_gaze-v2.csv'),TOY_CUTS=str(out/'cuts.json'),TOY_VIS='1',TOY_LAT='1.0',TOY_GZR='1.0',TOY_NG='2048',TOY_ZMAX='2.5',TOY_NEUT='.30',TOY_HR='.30',TOY_HC='tcp',TOY_LR='1.0',TOY_REG='exact',TOY_WORK=str(p/'retarget/work'),TOY_LEROBOT=str(out),TOY_DST=str(out/'unused'),EP_STRIDE='1',EP_OFF='0')
 run('rgbd',[DP,str(D/'rgbd.py')]);
 with h5py.File(out/'rgbd/shard_0.hdf5')as f:
  g=f['data/demo_0'];assert len(g['obs/aria_depth'])==n and g.attrs['depth_staleness_ms_max']<200;dep=g['obs/aria_depth'][:];assert np.isfinite(dep).all()and(dep>.05).mean()>.05
 run('cloud',[PY,str(D/'clouds_ondevice_exact.py')]);run('dino',[PY,str(D/'dino_hdf5.py'),str(fix),str(out/'dino')])
 with h5py.File(fix)as f,h5py.File(bs)as b:
  x=f['data/demo_0'];y=b['data/demo_0'];conv=StateConverter();base=conv.convert(x,n);np.testing.assert_allclose(np.load(out/'clouds/p_000000.npy'),base['state_eef_pose_glass'],atol=2e-5,rtol=0)
  q=y['obs/robot0_joint_pos'][:];raw=y['actions/joint'][:];data=dict(action=np.concatenate([raw[:,22:25],raw[:,14:22],raw[:,:14],raw[:,25:]],-1).astype('float32'),state_robot0_joint_pos=q[:,4:26].astype('float32'),state_hand_left_qpos=y['obs/hand_left_qpos'][:].astype('float32'),state_hand_right_qpos=y['obs/hand_right_qpos'][:].astype('float32'),timestamp=y['obs/aria_ts'][:]/1e9,action_valid=np.ones(n,bool),dino_valid=np.ones(n,bool))
  np.testing.assert_array_equal(data['timestamp'],base['timestamp'])
 for key,fn,shape in [('front_pcd_1','clouds/g_hd2_000000.npy',(n,2048,6)),('front_pcd_2','clouds/l_hd2_000000.npy',(n,1024,6)),('front_pcd_3','clouds/l3_hd2_000000.npy',(n,1024,6)),('dino','dino/dino_000000.npy',(n,16,768)),('state_eef_pose_glass','clouds/p_000000.npy',(n,9)),('gaze_point_glass','clouds/gp_000000.npy',(n,3))]:data[key]=np.load(out/fn).reshape(shape)
 arrays={}
 for k,a in data.items():
  assert len(a)==n and np.isfinite(a).all(),k
  if k.startswith('front_pcd'):assert a[:,:,3:].min()>=0 and a[:,:,3:].max()<=1 and a[:,:,:3].std()>.001
  f=out/'tensors'/(k+'.npy');np.save(f,a);arrays[k]=dict(file=str(f),shape=list(a.shape),dtype=str(a.dtype),sha256=sha(f))
 assert abs(data['dino'].mean(-1)).max()<.01 and .9<data['dino'].std(-1).mean()<1.1
 ep=dict(e,task='drawer',rows=n,arrays=arrays,retarget_hdf5=str(bs),retarget_demo='demo_0',bs_observations='fixed pre-BS camera observations retained, matching Cart BS recipe',virtual_camera_filter=True)
 save(out/'episode.json',ep);save(out/'proof.json',dict(passed=True,rows=n,uuid=e['uuid'],episode_sha256=sha(out/'episode.json'),retarget_proof_sha256=sha(p/'retarget-proof.json'),cloud_source_sha256=sha(D/'clouds_ondevice_exact.py'),mps_requested=False,training_ready=False));print('DRAWER_DOWNSTREAM_PASS',e['uuid'],n,flush=True)
except Exception:
 save(out/'failure.json',dict(job=os.getenv('SLURM_JOB_ID'),traceback=traceback.format_exc(),training_ready=False));raise
