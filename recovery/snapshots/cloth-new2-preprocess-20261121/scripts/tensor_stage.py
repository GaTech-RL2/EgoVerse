import os,sys,json,hashlib,pathlib,traceback
import numpy as np,h5py
from retarget_tensor_state import StateConverter
from gaze_preprocessing import lift_sensor,uniform_clean_cloud
R=pathlib.Path('/coc/flash7/cxin34/cloth-new2-scratch-20261121/tensors')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def save(p,v):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2));t.replace(p)
e=json.loads((R/'plan.json').read_text())[int(sys.argv[1])]
e.update(json.loads((pathlib.Path(e['episode'])/'input.json').read_text()))
u=e['uuid'];out=R/'episodes'/u;out.mkdir(parents=True,exist_ok=True)
assert not (out/'intent.json').exists(),'Existing attempt requires audit'
save(out/'intent.json',{'uuid':u,'job_id':os.getenv('SLURM_JOB_ID'),'status':'RUNNING'})
try:
 src=pathlib.Path(e['episode']);proof=json.loads((src/'preprocess-proof.json').read_text())
 assert proof['passed'] and proof['uuid']==u
 h5=src/'retarget/episode.hdf5';rgbd=src/'rgbd/shards/shard_0.hdf5'
 for p in [h5,rgbd]:assert sha(p)==proof['artifacts'][str(p)]
 with h5py.File(h5) as f:
  g=f['data/demo_0'];data=StateConverter().convert(g)
  raw=g['actions/joint'][:];q=g['obs/robot0_joint_pos'][:]
  np.testing.assert_array_equal(data['action'][:,:3],raw[:,22:25].astype('float32'))
  np.testing.assert_array_equal(data['current_q'][:,:6],q[:,4:10].astype('float32'))
 n=len(data['action']);assert n==proof['timestamps']['rows']
 data.update(action_valid=np.ones(n,bool),dino_valid=np.ones(n,bool))
 arrays={}
 def write(k,a):
  assert np.isfinite(a).all(),k
  p=out/(k+'.npy');np.save(p,a);arrays[k]={'file':p.name,'sha256':sha(p),'shape':list(a.shape),'dtype':str(a.dtype)}
 for k,a in data.items():write(k,a)
 for k,fn,shape in [('front_pcd_1','clouds/g_hd2_000000.npy',(n,2048,6)),('front_pcd_2','clouds/l_hd2_000000.npy',(n,1024,6)),('front_pcd_3','clouds/l3_hd2_000000.npy',(n,1024,6)),('dino','dino/dino_000000.npy',(n,16,768))]:
  p=src/fn;assert sha(p)==proof['artifacts'][str(p)]
  a=np.load(p,mmap_mode='r');assert a.size==np.prod(shape) and np.isfinite(a).all()
  # Shape-only view copied to immutable training staging; source untouched.
  write(k,a.reshape(shape))
 clean=np.lib.format.open_memmap(out/'clean_cloud.npy',mode='w+',dtype='float32',shape=(n,2048,6))
 with h5py.File(rgbd) as f:
  g=f['data/demo_0'];assert len(g['obs/aria_depth'])==n
  for i in range(n):
   dep=np.asarray(g['obs/aria_depth'][i],dtype='float32').reshape(224,224)
   rgb=np.asarray(g['obs/aria_image'][i])
   K=np.asarray(g.attrs['source_rect_K224']);sensor=lift_sensor(dep,rgb,(K[0,0],K[1,1],K[0,2],K[1,2]))
   clean[i]=uniform_clean_cloud(sensor,np.random.default_rng(i))
 assert np.isfinite(clean).all()
 clean.flush();del clean
 p=out/'clean_cloud.npy';arrays['clean_cloud']={'file':p.name,'sha256':sha(p),'shape':[n,2048,6],'dtype':'float32'}
 save(out/'episode.json',dict(e,rows=n,arrays=arrays,preprocess_proof_sha256=sha(src/'preprocess-proof.json'),clean_cloud_provenance='sensor_uniform_no_pose_no_gaze',recording_hz=60,horizon_rows=64,dino_offset_rows=10,training_release_ready=False,pending=['task-tail visual QC','grouped split freeze','train-only normalization','independent release audit']))
 save(out/'result.json',{'uuid':u,'status':'STAGED','rows':n,'array_count':len(arrays),'job_id':os.getenv('SLURM_JOB_ID')})
 print(u,'STAGED',n,flush=True)
except Exception:
 save(out/'result.json',{'uuid':u,'status':'ISOLATED_BUILD_FAILURE','traceback':traceback.format_exc(),'job_id':os.getenv('SLURM_JOB_ID')});raise
