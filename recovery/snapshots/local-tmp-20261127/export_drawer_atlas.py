import os,json,hashlib,io,fcntl,sys
from pathlib import Path
import numpy as np,h5py,pyarrow as pa,pyarrow.parquet as pq
from PIL import Image
os.umask(0o022)
R=Path('/coc/flash7/scratch/cxin34/cart-drawer-nola-fix-bs-20261125')
W=Path('/coc/flash7/scratch/cxin34/warp-data'); N='cxin34_drawer_1125_qc'
P=W/'pipelines/drawer_atlas_20261125'; C=W/'collections'/N; D=W/'datasets'/N
V=sys.argv[1]; assert V in ['r1_fix_all145','r2_fix_bs_all145','r3_fix_bs_hd2_pass23']
cloud=V.startswith('r3'); bs=not V.startswith('r1')
ST=D/('.staging-'+V); DEST=D/'releases'/V; RET=C/'processed'/V/'retargeted'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def write(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2))
def arr(v):
 v=np.asarray(v);return pa.array(v) if v.ndim==1 else pa.FixedSizeListArray.from_arrays(pa.array(v.ravel()),int(np.prod(v.shape[1:])))
P.mkdir(parents=True,exist_ok=True); lock=(P/(V+'.lock')).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not DEST.exists() and not ST.exists() and not RET.exists(),'existing attempt: audit before recovery'
segments=json.loads((R/'drawer/segment-manifest.json').read_text()); ds=json.loads((R/'drawer/reports/disposition.json').read_text())['segments']
pairs=[(e,d) for e,d in zip(segments,ds) if not cloud or d['status']=='ACCEPTED']; assert len(pairs)==(23 if cloud else 145)
(C/'raw').mkdir(parents=True,exist_ok=True);RET.mkdir(parents=True);(RET/'work').mkdir();ST.mkdir(parents=True)
proof=[];maps=[];demos=[];tags=[];offset=0
bundle=RET/(N+'.hdf5')
with h5py.File(bundle,'x') as out:
 for idx,(e,d) in enumerate(pairs):
  src=Path(e['segment']);tag=e['uuid'];h=src/('retarget/fix_bs.hdf5' if bs else 'retarget/fix.hdf5');before=sha(h)
  with h5py.File(h) as f:
   g=f['data/demo_0'];ts=g['obs/aria_ts'][:];n=len(ts);q=g['obs/robot0_joint_pos'][:].astype('float32');raw=g['actions/joint'][:].astype('float32');rgb=g['obs/aria_image'][:]
   assert n>60 and np.all(np.diff(ts)>0) and len(rgb)==n
   # Read-only links preserve the exact source robot sequence, including failures.
   out['data/demo_'+str(idx)]=h5py.ExternalLink(str(h),'/data/demo_0')
   canonical=np.concatenate((raw[:,22:25],raw[:,14:22],raw[:,:14],raw[:,25:]),axis=1)
   data={'obs.robot0_joint_pos':q,'obs.robot0_joint_pos_no_wheel':q[:,4:],'obs.right_arm':q[:,10:17],'obs.left_arm':q[:,17:24],
    'obs.hand_left_qpos':g['obs/hand_left_qpos'][:].astype('float32'),'obs.hand_right_qpos':g['obs/hand_right_qpos'][:].astype('float32'),
    'obs.robot_base_pose':g['obs/robot_base_pose'][:].astype('float32'),'actions.joint':raw,'actions.joint_base_torso_head_arm_hand':canonical,
    'obs.aria_ts':ts.astype('float64'),'timestamp':((ts-ts[0])*1e-9).astype('float32'),'frame_index':np.arange(n,dtype='int64'),
    'episode_index':np.full(n,idx,dtype='int64'),'index':np.arange(offset,offset+n,dtype='int64'),'task_index':np.zeros(n,dtype='int64')}
   if cloud:
    ep=json.loads((src/'downstream/episode.json').read_text());arrays={}
    for key in ['action','timestamp','front_pcd_1','front_pcd_2','front_pcd_3','state_eef_pose_glass']:
     a=ep['arrays'][key];assert sha(a['file'])==a['sha256'];arrays[key]=np.load(a['file']);assert np.isfinite(arrays[key]).all()
    np.testing.assert_array_equal(arrays['action'],canonical);np.testing.assert_array_equal(arrays['timestamp'],ts/1e9)
    for key,col in [('front_pcd_1','obs.aria_pcdc'),('front_pcd_2','obs.aria_pcdc_local'),('front_pcd_3','obs.aria_pcdc_local_left')]:data[col]=arrays[key].reshape(n,-1)
    data['obs.eef_pose_glass']=arrays['state_eef_pose_glass']
   for v in data.values():assert np.isfinite(v).all()
   cache={};images=[]
   for im in rgb:
    key=hashlib.sha256(im.tobytes()).digest()
    if key not in cache:
     b=io.BytesIO();Image.fromarray(im).save(b,format='PNG',compress_level=1);cache[key]=b.getvalue()
    images.append({'bytes':cache[key],'path':None})
   tab=pa.table({k:arr(v) for k,v in data.items()}).append_column('obs.aria_image',pa.array(images,type=pa.struct([('bytes',pa.binary()),('path',pa.string())])))
   features={k:({'dtype':str(v.dtype),'_type':'Value'} if v.ndim==1 else {'feature':{'dtype':str(v.dtype),'_type':'Value'},'length':v.shape[1],'_type':'Sequence'}) for k,v in data.items()};features['obs.aria_image']={'_type':'Image'}
   tab=tab.replace_schema_metadata({b'huggingface':json.dumps({'info':{'features':features}}).encode()})
   dest=ST/'data/chunk-000'/f'episode_{idx:06d}.parquet';dest.parent.mkdir(parents=True,exist_ok=True)
   prior=D/('.staging-'+V+'.attempt1')/'data/chunk-000'/dest.name
   if prior.exists():
    assert pq.read_table(prior).equals(tab,check_metadata=False);os.link(prior,dest)
   else:pq.write_table(tab,dest,compression='zstd',row_group_size=128)
   back=pq.read_table(dest);assert back.equals(tab,check_metadata=False) and back.schema.metadata==tab.schema.metadata
   for t in [0,n//2,n-1]:np.testing.assert_array_equal(np.asarray(Image.open(io.BytesIO(back['obs.aria_image'][t].as_py()['bytes']))),rgb[t])
   meta_features={k:{'dtype':str(v.dtype),'shape':list(v.shape[1:]) or [1],'names':None} for k,v in data.items()};meta_features['obs.aria_image']={'dtype':'image','shape':[224,224,3],'names':['height','width','channel']}
  work=src/'retarget/work'
  with np.load(work/(e['retarget_tag']+'_aria60.npz')) as z:
   grid=z['ts'];ix=np.searchsorted(grid,ts/1000).clip(0,len(grid)-1);prev=np.maximum(ix-1,0);ix=np.where(abs(grid[prev]-ts/1000)<abs(grid[ix]-ts/1000),prev,ix);assert max(abs(grid[ix]-ts/1000))<.002
   vals={k:(z[k][ix] if z[k].ndim and len(z[k])==len(grid) else z[k]) for k in z.files};vals['n']=np.array(n);np.savez(RET/'work'/(tag+'_aria60.npz'),**vals)
  # Reuse exact original compact RGB cache only after frame-for-frame verification.
  rc=work/(e['retarget_tag']+'_rgb224.npz')
  with np.load(rc) as z:
   grid=z['grid_us'];ri=np.searchsorted(grid,ts/1000).clip(0,len(grid)-1);prev=np.maximum(ri-1,0);ri=np.where(abs(grid[prev]-ts/1000)<abs(grid[ri]-ts/1000),prev,ri)
   assert max(abs(grid[ri]-ts/1000))<1 and np.all(np.diff(ri)>0)
   np.testing.assert_array_equal(z['frames'][z['sel'][ri]],rgb)
   np.savez(RET/'work'/(tag+'_rgb224.npz'),frames=z['frames'],sel=z['sel'][ri],grid_us=grid[ri],res=z['res'])
  rawname=Path(e['vrs']).name;link=C/'raw'/rawname
  if not link.exists():
   try:link.symlink_to(e['vrs'])
   except FileExistsError:pass
  assert link.resolve()==Path(e['vrs']).resolve() and sha(h)==before
  gate=json.loads((src/'bs-gate-rgb-merge.json').read_text())
  maps.append(dict(episode_index=idx,collection=N,run=V,demo='demo_'+str(idx),source_tag=tag,raw_vrs=rawname,recording_id=e['recording_id'],start=0,length=n,source_hdf5=str(h),raw_sha256=e['raw_sha256'],gaze_source='ondevice_combined',split=e['split'],segment_index=d['index'],qc_status=d['status'],qc_reasons=d.get('reasons',[]),bs_hands_cm=gate['hands_cm'],visualization_only=True))
  demos.append(dict(demo='demo_'+str(idx),source_tag=tag,vrs=rawname,frames=n,recording_id=e['recording_id']));tags.append('\t'.join([tag,N,rawname,'ok']))
  proof.append(dict(episode_index=idx,uuid=tag,segment_index=d['index'],rows=n,qc_status=d['status'],parquet=str(dest.relative_to(ST)),sha256=sha(dest),all_values_readback_exact=True,hdf5=str(h),hdf5_sha256=before))
  offset+=n;print('EXPORTED',V,d['index'],n,flush=True)
(ST/'source_map.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in maps));(RET/'tag_manifest.tsv').write_text('\n'.join(tags)+'\n')
write(RET/'metadata.json',dict(dataset=N,created='2026-11-25',source_vrs_root=str(C/'raw'),num_hdf5=1,pipeline=dict(fps=60,aria_ts='absolute device time ns'),hdf5_files={bundle.name:dict(num_recordings_in_group=len(pairs),num_demos=len(pairs),total_frames=offset,file_bytes=bundle.stat().st_size,demos=demos)}))
write(ST/'meta/info.json',dict(codebase_version='v2.0',robot_type='rby1',total_episodes=len(pairs),total_frames=offset,total_tasks=1,total_videos=0,total_chunks=1,chunks_size=1000,fps=60,splits={'visualization':'0:'+str(len(pairs))},data_path='data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet',features=meta_features,viewer_compatibility=dict(revision=3,source_grid='selected microseconds',rgb_cache_grid_us=True)))
(ST/'meta/episodes.jsonl').write_text(''.join(json.dumps(dict(episode_index=x['episode_index'],length=x['rows'],tasks=['Drawer QC '+V+' '+x['qc_status']]))+'\n' for x in proof));(ST/'meta/tasks.jsonl').write_text(json.dumps(dict(task_index=0,task='Drawer QC visualization only'))+'\n')
write(ST/'qc.json',maps);write(ST/'recipe.json',dict(visualization_only=True,training_authorized=False,head_height_m=1.57,look_at=False,mps=False,variant=V,source=str(R),clouds='exact existing hd2 training clouds, pre-BS camera observations retained' if cloud else 'no clouds; retarget comparison only',quality='QC status refers to final combined pipeline; r1 is not separately accepted'))
write(ST/'manifest.json',dict(dataset=N,release=V,source_collection=N,source_run=V,episodes=len(pairs),frames=offset,fps=60,checksums={x['parquet']:x['sha256'] for x in proof}))
write(P/(V+'-audit.json'),dict(passed=True,episodes=proof,frames=offset,training_authorized=False,viewer_playback_verified=False));write(ST/'READY.json',dict(passed=True,purpose='visualization export integrity ONLY; not training acceptance',audit=str(P/(V+'-audit.json'))))
DEST.parent.mkdir(parents=True,exist_ok=True);ST.rename(DEST);print('PUBLISHED',DEST,flush=True)
