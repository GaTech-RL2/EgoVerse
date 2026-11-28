from pathlib import Path
import os,json,hashlib,sys
import numpy as np
R=Path('/coc/flash7/cxin34/cloth-new2-scratch-20261121');sys.path.insert(0,str(R))
from campaign.data import sha,EpisodeDataset
from campaign.contracts import check_normalizer
plan=json.loads((R/'plan.json').read_text());entries=[];excluded=[];report=[]
for item in plan:
 u=item['uuid'];e=Path(item['episode']);proof=R/'reports/accepted'/f'{u}.json'
 if (e/'quality-quarantine.json').exists():excluded.append(dict(item,reason=json.loads((e/'quality-quarantine.json').read_text())));continue
 assert proof.exists(),('Unresolved',item['take'])
 ap=json.loads(proof.read_text());td=R/'tensors/episodes'/u;ep=json.loads((td/'episode.json').read_text());assert ap['passed'] and ap['episode_metadata_sha256']==sha(td/'episode.json')
 arrays={}
 for k,a in ep['arrays'].items():
  f=td/a['file'];assert sha(f)==a['sha256'];v=np.load(f,mmap_mode='r');assert np.isfinite(v).all() and list(v.shape)==a['shape'];arrays[k]=dict(a,file=str(f))
 entries.append(dict(uuid=u,recording_id=u,take=item['take'],split=item['split'],rows=ep['rows'],arrays=arrays,raw_vrs_sha256=item['raw_vrs_sha256'],audit_sha256=sha(proof)))
 report.append(dict(json.loads((e/'valid-window.json').read_text()),split=item['split']))
train=[e for e in entries if e['split']=='train'];valid=[e for e in entries if e['split']=='valid']
assert len(entries)+len(excluded)==60 and len(train)>=40 and len(valid)>=8,(len(train),len(valid),len(excluded))
assert not {e['recording_id'] for e in train}&{e['recording_id'] for e in valid}
release=R/'releases/cloth-new2-scratch';release.mkdir(parents=True,exist_ok=True)
assert not (release/'READY.json').exists(),'Already published'
meta=dict(episodes=entries,excluded=excluded,recording_hz=60,horizon_rows=64,dino_offset_rows=10,quality_passed=True,split_frozen=True,split_seed=42,split_unit='recording_uuid',clean_cloud_provenance='sensor_uniform_no_pose_no_gaze',source='Gen2 cloud MPS SLAM/HAND_TRACKING/EYE_GAZE',gaze_calibrated=False,recipe='hd2 Gen2, longest bilateral-valid window after existing tail trim; MSC is train-time only')
manifest=release/'target.json';manifest.write_text(json.dumps(meta,indent=2))
conf=R/'configs/cloth-new2-B-msc-scratch-s42.json';cfg=json.loads(conf.read_text());cfg.update(train_recordings=len(train),valid_recordings=len(valid));conf.write_text(json.dumps(cfg,indent=2))
assert cfg['stage']=='scratch' and cfg['steps']==cfg['scheduler_steps']==180000 and not any(k in cfg for k in ['parent','parent_step','parent_checkpoint_sha256'])
stats=dict(fit_partition='target_train',manifest_sha256=sha(manifest),train_uuids=[e['uuid'] for e in train],recipe='hd2_quantile_q01_q99_sparse_minmax_eps1e-6',epsilon=1e-6,source_data_used=False)
metrics={}
for k in ['action','state_robot0_joint_pos','state_eef_pose_glass','state_hand_left_qpos','state_hand_right_qpos']:
 x=np.concatenate([np.load(e['arrays'][k]['file']) for e in train]);y=np.concatenate([np.load(e['arrays'][k]['file']) for e in valid]);lo=x.min(0);hi=x.max(0);low,high=np.quantile(x,[.01,.99],axis=0);sparse=high-low<.05*(hi-lo)
 low=np.where(sparse,lo,low).astype('float32');high=np.where(sparse,hi,high).astype('float32');width=high-low+np.float32(1e-6);scale=width/2;center=low+scale
 assert np.isfinite(scale).all() and (scale>0).all();z=(x-center)/scale
 np.testing.assert_allclose(z,2*(x-low)/width-1,atol=2e-5,rtol=2e-5);np.testing.assert_allclose(z*scale+center,x,atol=2e-6,rtol=2e-6)
 stats[k]=dict(center=center.tolist(),scale=scale.tolist(),quantile_1=low.tolist(),quantile_99=high.tolist(),sparse_fallback=sparse.tolist(),min=lo.tolist(),max=hi.tolist())
 metrics[k]=dict(train_rows=len(x),valid_rows=len(y),valid_max_abs_normalized=float(abs((y-center)/scale).max()))
check_normalizer(cfg,stats,meta,sha(manifest));norm=release/'normalizer.json';norm.write_text(json.dumps(stats,indent=2))
tr=EpisodeDataset(manifest,'train','B',stats);va=EpisodeDataset(manifest,'valid','B',stats)
for ds in [tr,va]:
 for j in range(0,len(ds),max(1,len(ds)//30)):
  b=ds[j];assert b['action'].shape==(64,49) and b['dino'].shape==(16,768)
proof=dict(passed=True,train_recordings=len(train),valid_recordings=len(valid),excluded=excluded,train_rows=len(tr),valid_rows=len(va),windows=report,normalizer_fit_partition='target_train',validation_used_for_fit=False,metrics=metrics,manifest_sha256=sha(manifest),normalizer_sha256=sha(norm),config_sha256=sha(conf),visual_review_pending=True)
(R/'reports/release-audit.json').write_text(json.dumps(proof,indent=2));print(json.dumps({k:v for k,v in proof.items() if k not in ['windows','metrics','excluded']},indent=2))
