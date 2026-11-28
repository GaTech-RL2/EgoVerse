import os,sys,json,hashlib,fcntl,subprocess,shutil,time
from pathlib import Path
os.umask(0o022)
W=Path('/coc/flash7/czhang883/warp-data')
P=W/'registry/cxin34-drawer-diagnostic-20261125'
R=Path('/coc/flash7/scratch/cxin34/cart-drawer-nola-fix-bs-20261125/drawer')
BASE=W/'datasets/cxin34_drawer_1125_qc/releases/r2_fix_bs_all145'
N='cxin34_drawer_1125_diagnostic'
PY='/coc/flash7/cxin34/mobilemani-pretraining/env-train/bin/python'
DP='/coc/flash7/scratch/czhang883/envs/aria_fs/bin/python'
segments=json.loads((R/'segment-manifest.json').read_text())
maps=[json.loads(l) for l in (BASE/'source_map.jsonl').read_text().splitlines()]
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2));t.replace(p)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def lock(p):
 p.parent.mkdir(parents=True,exist_ok=True);f=p.open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);return f

def depth_for(e):
 old=R/'depth'/e['recording_id'];proof=old/'proof.json'
 if proof.exists() and json.loads(proof.read_text()).get('passed') and len(list((old/Path(e['vrs']).stem).glob('*.npy')))>10:return old
 new=P/'depth'/e['recording_id']
 assert (new/'proof.json').exists(),'depth not ready'
 assert json.loads((new/'proof.json').read_text())['passed']
 return new

def make_depth(rid):
 import numpy as np
 e=next(e for e in segments if e['recording_id']==rid)
 try:return str(depth_for(e))
 except AssertionError:pass
 root=P/'depth'/rid;lk=lock(root/'lock')
 try:return str(depth_for(e))
 except AssertionError:pass
 assert not (root/'intent.json').exists(),'existing depth attempt needs audit'
 vrs=W/'collections/cxin34_aria_20261125_0556_114/raw'/Path(e['vrs']).name
 out=root/vrs.stem;out.mkdir()
 cmd=[DP,'src/run_fast_stereo_depth_to_npy.py','--vrs',str(vrs),'--output-dir',str(out),'--frame-step','4','--fs-home','/coc/flash7/yzhu827/Fast-FoundationStereo','--ckpt','/coc/flash7/yzhu827/Fast-FoundationStereo/weights/23-36-37/model_best_bp2_serialize.pth']
 write(root/'intent.json',dict(cmd=cmd,raw_sha256=e['raw_sha256'],time=time.time()))
 with (root/'depth.log').open('w') as f:subprocess.run(cmd,cwd='/coc/flash7/czhang883/Documents/aria_gen2_scripts',stdout=f,stderr=subprocess.STDOUT,check=True)
 files=sorted(out.glob('*.npy'));assert len(files)>10
 for p in files:
  a=np.load(p);assert a.shape==(512,512) and np.isfinite(a).all()
 write(root/'proof.json',dict(passed=True,frames=len(files),raw_sha256=e['raw_sha256'],training_authorized=False))
 return str(root)

def build(i):
 import numpy as np,h5py,pyarrow as pa,pyarrow.parquet as pq,mujoco
 sys.path.insert(0,'/coc/flash7/cxin34/egoa-target-retarget-fix-20261124/scripts')
 from retarget_tensor_state import StateConverter
 e=segments[i];o=P/'segments'/e['uuid'];lk=lock(o/'lock')
 if (o/'done.json').exists():return
 assert not (o/'intent.json').exists(),'existing segment attempt needs audit'
 dep=depth_for(e);write(o/'intent.json',dict(index=i,time=time.time(),depth=str(dep)))
 src=Path(e['segment']);fix=src/'retarget/fix.hdf5';bs=src/'retarget/fix_bs.hdf5';hashes=[sha(fix),sha(bs)]
 (o/'input').mkdir();vrs=W/'collections/cxin34_aria_20261125_0556_114/raw'/Path(e['vrs']).name
 (o/'input'/vrs.name).symlink_to(vrs);(o/'rgbd').mkdir()
 (o/'manifest.tsv').write_text('\t'.join([e['retarget_tag'],N,vrs.name,'ok'])+'\n')
 code=(R/'source/downstream/rgbd.py').read_text()
 needle='    Rl_m = np.asarray(Rl.to_matrix())'
 code=code.replace(needle,needle+"\n    import json\n    tdr=lc.get_transform_device_camera().to_matrix().copy(); rr=np.eye(4);rr[:3,:3]=Rl_m;tdr=tdr@rr\n    json.dump(dict(K=K.tolist(),T_device_rect=tdr.tolist()),open(OUTD+'/camera.json','w'))")
 (o/'rgbd.py').write_text(code)
 env=dict(os.environ,TOY_SRC=str(fix),TOY_MAN=str(o/'manifest.tsv'),TOY_VRSD=str(o/'input'),DEPD=str(dep),OUTD=str(o/'rgbd'),SHARD='0',NSHARD='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2')
 with (o/'rgbd.log').open('w') as f:subprocess.run([DP,str(o/'rgbd.py')],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 c=StateConverter()
 def fk(q):
  H,E=c.fk(q);b=c.d.body('left_eef');L=np.eye(4);L[:3,:3]=b.xmat.reshape(3,3);L[:3,3]=b.xpos;return H,E,L
 def se2(b):
  x,y,a=b;T=np.eye(4);T[:2,:2]=[[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]];T[:2,3]=[x,y];return T
 def kab(X,Y):
  xc=X.mean(0);yc=Y.mean(0);u,s,v=np.linalg.svd((X-xc).T@(Y-yc));rot=v.T@u.T
  if np.linalg.det(rot)<0:v[-1]*=-1;rot=v.T@u.T
  T=np.eye(4);T[:3,:3]=rot;T[:3,3]=yc-rot@xc;return T
 def xform(T,x):return x@T[:3,:3].T+T[:3,3]
 with h5py.File(fix) as ff,h5py.File(bs) as bf,h5py.File(o/'rgbd/shard_0.hdf5') as df:
  a=ff['data/demo_0'];b=bf['data/demo_0'];d=df['data/demo_0'];q=b['obs/robot0_joint_pos'][:];bp=b['obs/robot_base_pose'][:];ts=b['obs/aria_ts'][:];n=len(q)
  assert np.isfinite(q).all() and np.isfinite(bp).all() and np.all(np.diff(ts)>0)
  stale=float(d.attrs['depth_staleness_ms_max']);assert stale<200
  oldq=a['obs/robot0_joint_pos'][:];oldbp=a['obs/robot_base_pose'][:];oldR=a['obs/eef_right_proprio'][:].reshape(n,4,4);oldL=a['obs/eef_left_proprio'][:].reshape(n,4,4)
  shifts=np.array([oldR[k,:3,3]-(se2(oldbp[k])@fk(oldq[k])[1])[:3,3] for k in range(n)])
  anchor=np.eye(4);anchor[:3,3]=np.median(shifts,axis=0)
  z=np.load(src/'retarget/work'/(e['retarget_tag']+'_aria60.npz'));ix=np.abs(z['ts'][:,None]-ts[None,:]/1000).argmin(axis=0);assert np.max(abs(z['ts'][ix]-ts/1000))<1
  X=[];Y=[]
  for k in range(0,n,2):
   for side,ee,off in [('right',oldR[k],[.0265,.0065,-.1077,1]),('left',oldL[k],[-.0265,.0065,-.1077,1])]:
    if z[side+'_valid'][ix[k]]:X.append(z[side+'_landmarks'][ix[k],8]);Y.append((ee@off)[:3])
  X=np.array(X);Y=np.array(Y);assert len(X)>=12
  align=kab(X,Y);res=np.linalg.norm(xform(align,X)-Y,axis=1);inl=res<.01
  if inl.sum()>=12:align=kab(X[inl],Y[inl]);res=np.linalg.norm(xform(align,X)-Y,axis=1);inl=res<.01
  regok=bool(inl.sum()>=max(12,.5*len(X)) and np.median(res[inl])<.005)
  FLU=np.array([[0,1,0],[0,0,1],[1,0,0]])
  Rdh=z['T_device_cpf'][:3,:3]@FLU;Rwd=z['head_R'][ix]@Rdh.T
  cam=json.loads((o/'rgbd/camera.json').read_text());K=np.array(cam['K'])*224/512;K[2,2]=1;tdr=np.array(cam['T_device_rect'])
  u,v=np.meshgrid(np.arange(224),np.arange(224));rays=np.stack([(u-K[0,2])/K[0,0],(v-K[1,2])/K[1,1],np.ones_like(u)],-1).reshape(-1,3)
  clouds=[np.empty((n,budget,6),np.float32) for budget in [2048,1024,1024]];p9=[];worldR=[];worldL=[];fallback=[0,0];poolmin=10**9
  for k in range(n):
   depth=d['obs/aria_depth'][k].reshape(-1);rgb=d['obs/aria_image'][k].reshape(-1,3).astype('float32')/255
   valid=np.isfinite(depth)&(depth>.25)&(depth<5);assert valid.sum()>=64,(i,k,valid.sum());poolmin=min(poolmin,int(valid.sum()))
   dev=np.eye(4);dev[:3,:3]=Rwd[k];dev[:3,3]=z['dev_p'][ix[k]];points=xform(align@dev@tdr,rays[valid]*depth[valid,None]);colors=rgb[valid];colors[colors.sum(1)<.02]=.5
   H,E,L=fk(q[k]);B=anchor@se2(bp[k]);G=B@H@c.mount;E=B@E;L=B@L;worldR.append(E);worldL.append(L)
   T=np.linalg.inv(G)@E;p9.append(np.r_[T[:3,3],T[:3,0],T[:3,1]])
   for j,(frame,budget) in enumerate([(G,2048),(E,1024),(L,1024)]):
    local=xform(np.linalg.inv(frame),points);mask=np.ones(len(local),bool) if j==0 else np.linalg.norm(local,axis=1)<1
    if mask.sum()<16:mask[:]=True;fallback[j-1]+=1
    pts=local[mask];col=colors[mask];sel=np.linspace(0,len(pts)-1,budget).astype(int);clouds[j][k]=np.concatenate([pts[sel],col[sel]],axis=1)
   if k%250==0:print('PROJECT',i,k,n,flush=True)
  # Display FK is rebuilt from unchanged saved q/base, not substituted into training artifacts.
  with h5py.File(o/'display.hdf5','x') as out:
   g=out.create_group('data/demo_0')
   for key in b.attrs:g.attrs[key]=b.attrs[key]
   for key in b:g[key]=h5py.ExternalLink(str(bs),'/data/demo_0/'+key)
   del g['obs'];obs=g.create_group('obs')
   for key in b['obs']:
    if key not in ['eef_right_proprio','eef_left_proprio']:obs[key]=h5py.ExternalLink(str(bs),'/data/demo_0/obs/'+key)
   for key,val in [('eef_right_proprio',worldR),('eef_left_proprio',worldL)]:obs.create_dataset(key,data=np.array(val).reshape(b['obs/'+key].shape))
  tab=pq.read_table(BASE/'data/chunk-000'/f'episode_{i:06d}.parquet')
  vals=dict(zip(['obs.aria_pcdc','obs.aria_pcdc_local','obs.aria_pcdc_local_left'],clouds));vals['obs.eef_pose_glass']=np.asarray(p9,np.float32)
  for key,val in vals.items():
   val=val.reshape(n,-1);assert np.isfinite(val).all();tab=tab.append_column(key,pa.FixedSizeListArray.from_arrays(pa.array(val.ravel()),val.shape[1]))
  hf=json.loads(tab.schema.metadata[b'huggingface']);features=json.loads((BASE/'meta/info.json').read_text())['features']
  for key,val in vals.items():
   size=int(np.prod(val.shape[1:]));hf['info']['features'][key]={'feature':{'dtype':'float32','_type':'Value'},'length':size,'_type':'Sequence'};features[key]={'dtype':'float32','shape':[size],'names':None}
  tab=tab.replace_schema_metadata({b'huggingface':json.dumps(hf).encode()});pq.write_table(tab,o/'episode.parquet',compression='zstd',row_group_size=128);back=pq.read_table(o/'episode.parquet');assert back.equals(tab,check_metadata=False)
  original=pq.read_table(BASE/'data/chunk-000'/f'episode_{i:06d}.parquet')
  for key in original.column_names:assert original[key].equals(back[key]),key
  assert hashes==[sha(fix),sha(bs)]
  write(o/'done.json',dict(index=i,uuid=e['uuid'],rows=n,features=features,qc_status=maps[i]['qc_status'],qc_reasons=maps[i]['qc_reasons'],training_authorized=False,source_sha256=hashes,parquet_sha256=sha(o/'episode.parquet'),registration_pass=regok,registration_pairs=len(X),registration_inliers=int(inl.sum()),registration_median_cm=float(np.median(res)*100),camera_to_robot=align.tolist(),preBS_FK_anchor_residual_median_cm=float(np.median(np.linalg.norm(shifts-anchor[:3,3],axis=1))*100),local_fallback_frames=fallback,depth_staleness_ms_max=stale,min_depth_pool=poolmin))
 print('DONE',i,n,regok,flush=True)

def publish(version,indices):
 import numpy as np,h5py,pyarrow as pa,pyarrow.parquet as pq
 D=W/'datasets'/N;C=W/'collections'/N;dest=D/'releases'/version;st=D/('.staging-'+version);lk=lock(P/(version+'.lock'))
 if dest.exists():return
 assert not st.exists(),'existing publish attempt'
 st.mkdir(parents=True);ret=C/'processed'/version/'retargeted';(ret/'work').mkdir(parents=True);(C/'raw').mkdir(exist_ok=True)
 sm=[];proof=[];demos=[];tags=[];offset=0
 annotations=json.loads((P/'user_annotations.json').read_text()) if (P/'user_annotations.json').exists() else {}
 with h5py.File(ret/(N+'.hdf5'),'x') as bundle:
  for j,i in enumerate(indices):
   e=segments[i];o=P/'segments'/e['uuid'];a=json.loads((o/'done.json').read_text());n=a['rows'];tab=pq.read_table(o/'episode.parquet')
   for key,val in [('episode_index',np.full(n,j,dtype='int64')),('index',np.arange(offset,offset+n,dtype='int64'))]:tab=tab.set_column(tab.schema.get_field_index(key),key,pa.array(val))
   path=st/'data/chunk-000'/f'episode_{j:06d}.parquet';path.parent.mkdir(parents=True,exist_ok=True);pq.write_table(tab,path,compression='zstd',row_group_size=128)
   m=dict(maps[i],episode_index=j,collection=N,run=version,demo='demo_'+str(j),source_hdf5=str(o/'display.hdf5'),diagnostic_registration_pass=a['registration_pass'],diagnostic_registration_median_cm=a['registration_median_cm']);m.update(annotations.get(e['uuid'],{}));sm.append(m)
   bundle['data/demo_'+str(j)]=h5py.ExternalLink(str(o/'display.hdf5'),'/data/demo_0')
   tag=e['retarget_tag'];oldwork=W/'collections/cxin34_drawer_1125_qc/processed/r2_fix_bs_all145/retargeted/work'
   for suffix in ['_aria60.npz','_rgb224.npz']:(ret/'work'/(tag+suffix)).symlink_to(oldwork/(tag+suffix))
   raw=Path(e['vrs']).name
   if not (C/'raw'/raw).exists():(C/'raw'/raw).symlink_to(W/'collections/cxin34_aria_20261125_0556_114/raw'/raw)
   tags.append('\t'.join([tag,N,raw,'ok']));demos.append(dict(demo=m['demo'],source_tag=tag,vrs=raw,frames=n,recording_id=e['recording_id']));proof.append(dict(episode_index=j,segment_index=i,rows=n,sha256=sha(path),qc_status=m['qc_status'],registration_pass=a['registration_pass'],registration_median_cm=a['registration_median_cm']));offset+=n
 info=json.loads((BASE/'meta/info.json').read_text());info.update(total_episodes=len(indices),total_frames=offset,splits={'visualization':'0:'+str(len(indices))},features=a['features']);write(st/'meta/info.json',info)
 (st/'meta/episodes.jsonl').write_text(''.join(json.dumps(dict(episode_index=p['episode_index'],length=p['rows'],tasks=['DIAGNOSTIC ONLY '+p['qc_status']+'; registration '+str(p['registration_pass'])]))+'\n' for p in proof));(st/'meta/tasks.jsonl').write_text(json.dumps(dict(task_index=0,task='Drawer diagnostic: failed QC retained; not training data'))+'\n')
 (st/'source_map.jsonl').write_text(''.join(json.dumps(m)+'\n' for m in sm));(ret/'tag_manifest.tsv').write_text('\n'.join(tags)+'\n')
 write(ret/'metadata.json',dict(dataset=N,source_vrs_root=str(C/'raw'),num_hdf5=1,pipeline=dict(fps=60,aria_ts='absolute device time ns'),hdf5_files={N+'.hdf5':dict(num_recordings_in_group=len(indices),num_demos=len(indices),total_frames=offset,file_bytes=(ret/(N+'.hdf5')).stat().st_size,demos=demos)}))
 write(st/'qc.json',sm);write(st/'recipe.json',dict(visualization_only=True,training_authorized=False,head_height_m=1.57,look_at=False,mps=False,clouds='Real FFS SLAM stereo depth, 0.25-5m, uniform diagnostic samples global2048/right1024/left1024. NOT the HD2 training crop or visibility filter. Sparse hand pools use full real cloud and are counted in per-segment audit.',registration='Pre-BS knuckle rigid fit with original threshold result retained. Failed registration is approximate; consult qc.json. Post-BS q/base unchanged; display EEF recomputed from q for consistent rendering.',source=str(BASE)))
 write(st/'manifest.json',dict(dataset=N,release=version,episodes=len(indices),frames=offset,checksums={f"data/chunk-000/episode_{p['episode_index']:06d}.parquet":p['sha256'] for p in proof}));write(st/'READY.json',dict(passed=True,purpose='visualization integrity only, NOT training readiness'))
 write(P/(version+'-audit.json'),dict(episodes=proof,frames=offset,training_authorized=False));dest.parent.mkdir(parents=True,exist_ok=True);st.rename(dest);print('PUBLISHED',dest,flush=True)

if __name__=='__main__':
 if sys.argv[1]=='segment':build(int(sys.argv[2]))
 elif sys.argv[1]=='recording':
  rid=sys.argv[2];make_depth(rid)
  for i,e in enumerate(segments):
   if e['recording_id']==rid:
    try:build(i)
    except Exception as ex:write(P/'segments'/e['uuid']/'error.json',dict(error=repr(ex)));print('FAILED_SEGMENT',i,repr(ex),flush=True)
 elif sys.argv[1]=='publish':publish(sys.argv[2],[int(x) for x in sys.argv[3].split(',')])
