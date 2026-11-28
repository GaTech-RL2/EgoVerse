import os,json,hashlib,shutil
from pathlib import Path
import cv2,h5py,numpy as np
from common import Heat7,project_labels,in_image,KH,C0
R=Path(os.environ['GAZE_ROOT'])
SOURCE=Path('/coc/flash7/cxin34/cloth-new2-h168-scratch-20261121')
MAN=SOURCE/'releases/cloth-new2-scratch-split55-5/target.json'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,d):
 p=Path(p);t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2));t.replace(p)
def main():
 if (R/'reports/data-ready.json').exists():return
 assert shutil.disk_usage(R).free>30*2**30
 manifest=json.loads(MAN.read_text());entries=manifest['episodes']
 assert len(entries)==60 and sum(e['split']=='train' for e in entries)==55
 assert manifest['head_height_assumption_m']==1.68
 ready=json.loads((MAN.parent/'READY.json').read_text())
 split=dict(name='cloth_new2_h168_55train_5valid',train=[],selection=[],test=[],test_is_selection_alias=True,scope='Five held-out original recordings used for checkpoint selection; no independent test set or robot validation.')
 reports=[];tiles=[];geometry=[]
 Kout=np.array([[KH,0,C0],[0,KH,C0],[0,0,1.]])
 yy,xx=np.mgrid[:224,:224];pixels=np.stack([xx,yy,np.ones_like(xx)],-1)
 for ep,e in enumerate(entries):
  u=e['uuid'];n=e['rows'];d=SOURCE/'episodes'/u
  paths=[d/'clouds'/f'{name}_000000.npy' for name in ['gaze_center','camera_transform','gaze_hit']]
  gp,tg,hit=[np.load(p) for p in paths]
  assert gp.shape==(n,3) and tg.shape==(n,4,4) and hit.shape==(n,)
  assert np.isfinite(gp).all() and np.isfinite(tg).all()
  with h5py.File(d/'retarget/episode.hdf5') as h:
   g=h['data/demo_0/obs'];ts=g['aria_ts'][:];q=g['robot0_joint_pos'][:]
   lh=g['hand_left_qpos'][:];rh=g['hand_right_qpos'][:]
  with h5py.File(d/'rgbd/shards/shard_0.hdf5') as h:
   g=h['data/demo_0'];K=np.array(g.attrs['source_rect_K224']);ims=g['obs/aria_image'][:]
   np.testing.assert_array_equal(g['obs/robot0_joint_pos'][:],q)
  assert ims.shape==(n,224,224,3) and ims.dtype==np.uint8
  assert len(ts)==n and np.all(np.diff(ts)>0) and abs(np.median(np.diff(ts))-1e9/60)<2000
  for key,arr in [('state_robot0_joint_pos',q[:,4:26]),('state_hand_left_qpos',lh),('state_hand_right_qpos',rh)]:
   p=Path(e['arrays'][key]['file']);assert sha(p)==e['arrays'][key]['sha256'];np.testing.assert_allclose(np.load(p),arr,atol=1e-6)
  source_rgb_sha=hashlib.sha256(ims.tobytes()).hexdigest()
  # Same rectified optical axes; match deployed KH by calibrated central resampling (no invented borders).
  mapping=pixels@(K@np.linalg.inv(Kout)).T
  mx=mapping[:,:,0].astype('float32');my=mapping[:,:,1].astype('float32')
  assert mx.min()>=0 and my.min()>=0 and mx.max()<=223 and my.max()<=223,(K,mx.min(),mx.max())
  ims=np.stack([cv2.remap(im,mx,my,cv2.INTER_LINEAR) for im in ims])
  uv,z=project_labels(gp,tg)
  xyz=np.linalg.solve(tg,np.concatenate([gp,np.ones((n,1))],1)[...,None])[:,:3,0]
  np.testing.assert_allclose(uv,xyz[:,:2]/xyz[:,2,None]*KH+C0,atol=1e-6)
  keep=(hit==1)&np.isfinite(uv).all(1)&np.isfinite(z)&(z>=.1)
  assert keep.mean()>.95,('label coverage',u,float(keep.mean()))
  ix=np.nonzero(keep)[0];inside=in_image(uv[ix]);assert inside.mean()>.95,(u,inside.mean())
  key=f'cloth_new2:{ep:06d}';out=R/'data/cloth_new2'/f'ep_{ep:06d}.npz'
  if out.exists():
   with np.load(out) as old:
    np.testing.assert_array_equal(old['imgs'],ims[ix]);np.testing.assert_array_equal(old['uv'],uv[ix].astype('float32'))
  else:np.savez(out,imgs=ims[ix],uv=uv[ix].astype('float32'),rows=ix.astype('int32'),z=z[ix].astype('float32'),inimg=inside,timestamp_ns=ts[ix])
  split['train' if e['split']=='train' else 'selection'].append(key)
  reports.append(dict(uuid=u,take=e['take'],key=key,split=e['split'],rows=n,cached_rows=len(ix),in_image=int(inside.sum()),gaze_hit=int((hit==1).sum()),source_K=K.tolist(),output_K=Kout.tolist(),raw_sha256=e['raw_vrs_sha256'],label_sha={str(p):sha(p) for p in paths},cache_sha256=sha(out),source_rgb_sha256=source_rgb_sha,rgb_source=str(d/'rgbd/shards/shard_0.hdf5'),retarget_source=str(d/'retarget/episode.hdf5')))
  geom=R/'geometry'/f'{u}.npz'
  if not geom.exists():np.savez(geom,q=q,tg=tg,left=lh,right=rh,timestamp_ns=ts)
  geometry.append(dict(uuid=u,take=e['take'],split=e['split'],rows=n,file=str(geom),sha256=sha(geom)))
  i=n//2;tile=cv2.cvtColor(ims[i],cv2.COLOR_RGB2BGR);a,b=uv[i]
  if np.isfinite([a,b]).all():cv2.circle(tile,(round(a),round(b)),5,(0,255,0),2)
  cv2.putText(tile,f'take {e["take"]} {e["split"]}',(3,16),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,255),1);tiles.append(tile)
  print(json.dumps(reports[-1],default=str),flush=True)
 split['test']=list(split['selection'])
 assert len(split['train'])==55 and len(split['selection'])==5 and not set(split['train'])&set(split['selection'])
 assert len({x['uuid'] for x in reports})==60
 save(R/'splits/split_final.json',split)
 save(R/'reports/geometry-inputs.json',geometry)
 for j in range(3):
  chunk=tiles[j*20:(j+1)*20];cv2.imwrite(str(R/f'reports/labels-{j}.jpg'),np.concatenate([np.concatenate(chunk[i:i+5],1) for i in range(0,20,5)],0))
 net=Heat7(pretrained=True);assert sum(p.numel() for p in net.parameters())==11177025
 save(R/'reports/data-ready.json',dict(passed=True,episodes=reports,rows=sum(x['rows'] for x in reports),split=split,source_manifest=str(MAN),source_manifest_sha256=sha(MAN),split_sha256=sha(R/'splits/split_final.json'),camera_note='Gen2 source K calibrated resampling to deployed KH, same rectified optical axes; actual gaze hits only; no fallback labels.'))
if __name__=='__main__':main()
