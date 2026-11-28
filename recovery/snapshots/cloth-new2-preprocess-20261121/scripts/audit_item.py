import json,hashlib,sys,os,fcntl
from pathlib import Path
import numpy as np,h5py
R=Path('/coc/flash7/cxin34/cloth-new2-scratch-20261121')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def audit(item):
 e=Path(item['episode']);td=R/'tensors/episodes'/item['uuid']
 p=json.loads((e/'preprocess-proof.json').read_text());ep=json.loads((td/'episode.json').read_text())
 assert p['passed'] and p['uuid']==item['uuid'] and ep['preprocess_proof_sha256']==sha(e/'preprocess-proof.json')
 for f,d in p['artifacts'].items():assert sha(f)==d,f
 data={}
 for k,a in ep['arrays'].items():
  f=td/a['file'];assert sha(f)==a['sha256'];v=np.load(f,mmap_mode='r');assert list(v.shape)==a['shape'] and np.isfinite(v).all();data[k]=v
 n=ep['rows'];ts=data['timestamp'];assert n>=64 and ts[-1]-ts[0]>1 and np.allclose(np.diff(ts),1/60,atol=.001,rtol=.05)
 window=json.loads((e/'valid-window.json').read_text());assert abs(ts[0]*1e9-window['start_ns'])<2000 and abs(ts[-1]*1e9-window['end_ns'])<2000
 for k,N in [('front_pcd_1',2048),('front_pcd_2',1024),('front_pcd_3',1024),('clean_cloud',2048)]:
  v=data[k];assert v.shape==(n,N,6) and np.linalg.norm(v[:,:,:3].std(1),axis=1).min()>.001
  assert v[:,:,3:].min()>=0 and v[:,:,3:].max()<=1
 assert data['dino'].shape==(n,16,768)
 with h5py.File(e/'retarget/episode.hdf5') as h,h5py.File(e/'rgbd/shards/shard_0.hdf5') as hr:
  g=h['data/demo_0'];raw=g['actions/joint'][:];q=g['obs/robot0_joint_pos'][:]
  np.testing.assert_array_equal(data['action'],np.concatenate((raw[:,22:25],raw[:,14:22],raw[:,:14],raw[:,25:]),-1).astype('float32'))
  np.testing.assert_array_equal(data['current_q'],np.concatenate((q[:,4:10],q[:,24:26],q[:,17:24],q[:,10:17],g['obs/hand_left_qpos'][:],g['obs/hand_right_qpos'][:]),-1).astype('float32'))
  np.testing.assert_array_equal(ts,g['obs/aria_ts'][:]/1e9)
  gr=hr['data/demo_0'];K=np.asarray(gr.attrs['source_rect_K224']);assert float(gr.attrs['depth_staleness_ms_max'])<200
  from gaze_preprocessing import lift_sensor,uniform_clean_cloud
  for t in [0,n//2,n-1]:
   sensor=lift_sensor(gr['obs/aria_depth'][t].reshape(224,224),gr['obs/aria_image'][t],(K[0,0],K[1,1],K[0,2],K[1,2]))
   np.testing.assert_array_equal(data['clean_cloud'][t],uniform_clean_cloud(sensor,np.random.default_rng(t)))
 out=dict(uuid=item['uuid'],task=item['task'],recording_id=item['recording_id'],rows=n,seconds=float(ts[-1]-ts[0]),passed=True,episode_metadata_sha256=sha(td/'episode.json'),job_id=os.getenv('SLURM_JOB_ID'),training_release_ready=False,visual_review_pending=True)
 dest=R/'reports/accepted'/ (item['uuid']+'.json');dest.parent.mkdir(parents=True,exist_ok=True);tmp=dest.with_suffix('.tmp');tmp.write_text(json.dumps(out,indent=2));tmp.replace(dest)
 print(json.dumps(out),flush=True)
if __name__=='__main__':audit(json.loads((R/'plan.json').read_text())[int(sys.argv[1])])
