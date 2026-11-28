import json,os,hashlib,importlib.util,shutil,tarfile
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from prepare import sha,save
R=Path(os.environ['GAZE_ROOT']);OUT=R/'assets'
UTIL=Path('/coc/flash7/czhang883/Documents/SEW-Geometric-Teleop-vv/projects/rby1_teleop/utils')
RDR=np.array([[.934902,-.141367,.325537],[.149511,.988760,0],[-.321878,.048671,.945529]])
def module(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def fit():
 if (OUT/'geometry-audit.json').exists():return
 data=json.loads((R/'reports/data-ready.json').read_text());assert data['passed']
 entries=json.loads((R/'reports/geometry-inputs.json').read_text())
 tr=[];va=[];initial={k:[] for k in ['robot0_joint_pos','hand_left_qpos','hand_right_qpos']}
 for e in entries:
  p=Path(e['file']);assert sha(p)==e['sha256']
  with np.load(p) as z:
   q=z['q'][:,4:26];T=z['tg'];assert q.shape==(e['rows'],22) and T.shape==(e['rows'],4,4)
   assert np.isfinite(q).all() and np.isfinite(T).all()
   y=np.concatenate(((RDR.T@T[:,:3,3].T).T,Rot.from_matrix(RDR.T@T[:,:3,:3]).as_rotvec()),1)
   (tr if e['split']=='train' else va).append((q,y,T,e['uuid']))
   if e['split']=='train':
    for k,source in [('robot0_joint_pos','q'),('hand_left_qpos','left'),('hand_right_qpos','right')]:initial[k].append(z[source][0].astype('float64'))
 assert len(tr)==55 and len(va)==5
 qt=np.concatenate([a[0] for a in tr]);yt=np.concatenate([a[1] for a in tr])
 qv=np.concatenate([a[0] for a in va]);tv=np.concatenate([a[2] for a in va])
 assert len(qt)==25077 and len(qv)==2267
 mu=qt.mean(0);sd=np.maximum(qt.std(0),.05);sd[qt.std(0)<1e-3]=np.inf
 def feats(q):
  z=np.clip((q-mu)/sd,-4,4);return np.concatenate((z,z*z,np.ones((len(q),1))),1)
 x=feats(qt);W=np.linalg.solve(x.T@x+np.eye(45),x.T@yt)
 lo,hi=np.percentile(yt[:,:3],[.5,99.5],axis=0);rmax=float(np.percentile(np.linalg.norm(yt[:,3:],axis=1),99.5))
 cam=OUT/'virtual_cam_cloth_new2.npz';assert not cam.exists()
 np.savez(cam,mu=mu,sd=sd,W=W,R_DR=RDR,zclip=4.,t_lo=lo,t_hi=hi,r_max=rmax,KH=308.052437*224/512,about='Cloth new2 h168; 55 train / 5 validation recording UUIDs; q22 ridge lambda1; train-only fit and clamps. See geometry-audit.json.')
 vv=module('cloth_virtual_view',UTIL/'virtual_view.py');model=vv.load_model(cam)
 pred=np.stack([vv.predict_virtual_T(q,model) for q in qv]);assert np.isfinite(pred).all()
 z=feats(qv)@W;z[:,:3]=np.clip(z[:,:3],lo,hi);norm=np.linalg.norm(z[:,3:],axis=1);z[:,3:]*=np.minimum(1,rmax/np.maximum(norm,1e-12))[:,None]
 np.testing.assert_allclose(pred[:,:3,3],(RDR@z[:,:3].T).T,atol=1e-10)
 np.testing.assert_allclose(pred[:,:3,:3],RDR@Rot.from_rotvec(z[:,3:]).as_matrix(),atol=1e-10)
 def metric(p,t):
  et=np.linalg.norm(p[:,:3,3]-t[:,:3,3],axis=1)*100
  er=np.degrees((Rot.from_matrix(p[:,:3,:3]).inv()*Rot.from_matrix(t[:,:3,:3])).magnitude())
  return dict(translation_cm_median=float(np.median(et)),translation_cm_p90=float(np.percentile(et,90)),rotation_deg_median=float(np.median(er)),rotation_deg_p90=float(np.percentile(er,90)))
 results={'cloth_new2_native_clamped':metric(pred,tv)}
 oldfile=Path('/coc/flash7/czhang883/warp-data/pipelines/cloth_basket_train_1107/deploy/virtual_cam_cloth_basket1106.npz')
 if oldfile.is_file():
  old=vv.load_model(oldfile);results['old_cloth_same_valid']=metric(np.stack([vv.predict_virtual_T(q,old) for q in qv]),tv)
 per_record=[];offset=0
 for q,y,t,u in va:
  per_record.append(dict(uuid=u,rows=len(q),metrics=metric(pred[offset:offset+len(q)],t)));offset+=len(q)
 reset=dict(version=1,num_episodes=55,initial_step=0,skipped=np.array([],dtype='U1'),source=json.dumps(dict(source_manifest=data['source_manifest'],manifest_sha256=data['source_manifest_sha256'],training_uuids=[e['uuid'] for e in entries if e['split']=='train'],method='arithmetic mean of row0; population std ddof0; validation excluded')))
 for k,shape in [('robot0_joint_pos',(26,)),('hand_left_qpos',(12,)),('hand_right_qpos',(12,))]:
  a=np.stack(initial[k]);assert a.shape==(55,)+shape and np.isfinite(a).all();reset[k]=a.mean(0);reset[k+'_std']=a.std(0)
 resetfile=OUT/'reset_pose_cloth_new2.npz';assert not resetfile.exists();np.savez(resetfile,**reset)
 with np.load(resetfile,allow_pickle=False) as z:
  assert int(z['num_episodes'])==55 and z['skipped'].size==0
  for k in initial:
   np.testing.assert_array_equal(z[k],np.stack(initial[k]).mean(0));np.testing.assert_array_equal(z[k+'_std'],np.stack(initial[k]).std(0))
 with np.load(cam,allow_pickle=False) as z:
  assert z['W'].shape==(45,6) and set(z.files)=={'mu','sd','W','R_DR','zclip','t_lo','t_hi','r_max','KH','about'}
  assert np.isfinite(z['W']).all() and np.isfinite(z['mu']).all()
 save(OUT/'geometry-audit.json',dict(passed=True,train_recordings=55,valid_recordings=5,train_rows=len(qt),valid_rows=len(qv),results=results,per_validation_recording=per_record,source_manifest_sha256=data['source_manifest_sha256'],split_sha256=data['split_sha256'],provenance=entries,reset_exact_train_row0_mean_verified=True,reset_skipped_key_present=True,native_camera_parity_passed=True,constant_sd_infinity='intentional native-client constant-channel disabling',client_sha256=sha(UTIL/'virtual_view.py'),file_sha256={p.name:sha(p) for p in [cam,resetfile]},hardware_executed=False))
 print(json.dumps(results),flush=True)
def package():
 import torch
 from common import Heat7,load_rows
 torch.set_num_threads(4)
 report=json.loads((R/'reports/final-independent-audit.json').read_text());assert report['passed'] and len(report['members'])==6
 geom=json.loads((OUT/'geometry-audit.json').read_text());assert geom['passed']
 files=[]
 for i,rec in enumerate(report['members']):
  src=R/f'deploy/gaze_heat7_cloth_new2_m{i}.pt';assert sha(src)==rec['sha256']
  dst=OUT/'gaze'/src.name
  if dst.exists():assert sha(dst)==sha(src)
  else:shutil.copyfile(src,dst)
  files.append(dst)
 gaze=module('cloth_gaze_native',UTIL/'gaze_crop.py').HeatGaze(','.join(str(p) for p in files),device='cpu')
 split=json.loads((R/'splits/split_final.json').read_text());imgs,uv,_,_=load_rows(split['selection'],stride=1,inimg=False)
 idx=np.linspace(0,len(imgs)-1,20).astype(int);x=torch.from_numpy(imgs[idx].copy()).permute(0,3,1,2).float()/255
 pp=[]
 with torch.inference_mode():
  for p in files:
   net=Heat7(pretrained=False).eval();net.load_state_dict(torch.load(p,map_location='cpu',weights_only=True),strict=True);pp.append(net(x).numpy())
 actual=np.array([gaze.predict_uv(imgs[i].astype('float32')/255) for i in idx]);err=float(abs(actual-np.mean(pp,0)).max());assert err<.05
 shutil.copyfile(R/'reports/final-independent-audit.json',OUT/'gaze-validation.json')
 save(OUT/'client-compatibility.json',dict(passed=True,gaze_exact6=True,native_cpu_strict_load=True,native_uv_max_error_pixels=err,test_frames=len(idx),gaze_loader_sha256=sha(UTIL/'gaze_crop.py'),virtual_camera_native_parity=True,reset_shape_q26_hands12_12=True,reset_skipped_key=True,training_recordings_only=55,hardware_executed=False))
 (OUT/'README.md').write_text('''# Cloth new2 1.68m rollout assets
Uses the exact policy 55 train / 5 validation recording split. Gaze: six independent ImageNet-initialized Heat7 members (seeds42–47,30epochs), same standard RGB/255 pipeline. Gen2 RGB is calibrated to the deployed camera KH before training; actual eye-gaze hits only. The five validation recordings select checkpoints and are not an independent test set.
Virtual camera: q22 ridge fit and clamps from training data only.
Reset: mean first row of the55training episodes, q26 and hands12+12; includes native skipped metadata.
No robot motion or rollout has been executed.

Set these alongside the existing cloth policy/depth/crop client setup:
export GAZE_ENS=1
export GAZE_ENS_NAME=cloth_new2
export GAZE_ENS_DIR="<absolute assets path>/gaze"
export VV_MODEL="<absolute assets path>/virtual_cam_cloth_new2.npz"
export RESET_CACHE="<absolute assets path>/reset_pose_cloth_new2.npz"

Run sha256sum -c SHA256SUMS from this folder.
Human validation is not robot-camera validation; inspect geometry-audit.json and gaze-validation.json for measured errors. Do not substitute these files into the 1.60m recipe without evaluating the mismatch.
''')
 entries=[p for p in OUT.rglob('*') if p.is_file() and p.name!='SHA256SUMS']
 (OUT/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.relative_to(OUT)}\n' for p in sorted(entries)))
 archive=R/'cloth-new2-h168-rollout-assets-20261123.tar.gz';assert not archive.exists()
 with tarfile.open(archive,'w:gz') as t:t.add(OUT,arcname='cloth-new2-h168-rollout-assets')
 save(R/'reports/deploy-assets-package.json',dict(passed=True,path=str(OUT),archive=str(archive),archive_sha256=sha(archive),files=len(entries),gaze_validation=report['validation'],camera_validation=geom['results'],hardware_executed=False))
 print((R/'reports/deploy-assets-package.json').read_text())
if __name__=='__main__':
 import sys
 fit() if sys.argv[1]=='fit' else package()
