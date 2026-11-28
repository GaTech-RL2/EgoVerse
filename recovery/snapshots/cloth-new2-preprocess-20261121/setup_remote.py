from pathlib import Path
import json,random,shutil,csv,hashlib
R=Path('/coc/flash7/cxin34/cloth-new2-scratch-20261121');C=Path('/coc/flash7/cxin34/warp-data/collections/cxin34_cloth_new2_60');old=Path('/coc/flash7/cxin34/human-ondevice-pilot-20261110')
for d in ['scripts','source','episodes','reports','logs','tensors/episodes','cache','configs','releases','runs']: (R/d).mkdir(parents=True,exist_ok=True)
assert not (R/'plan.json').exists(),'Do not overwrite existing plan'
msc=Path('/coc/flash7/cxin34/cart-msc-ft-20261118')
shutil.copytree(msc/'campaign',R/'campaign',ignore=shutil.ignore_patterns('__pycache__'))
shutil.copytree(msc/'source',R/'source',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
for n in ['dino_hdf5.py','make_dino_wm_cart.py','glitch_scan.py','hd_sampling.py','virtual_view.py']:shutil.copy2(old/'source'/n,R/'source'/n)
for n in ['rgbd.py','real_dp3c_tri_gz_vis_hd2.py','mps_adapter.py']:shutil.copy2(R/'scripts'/n,R/'source'/n)
shutil.copy2('/coc/flash7/cxin34/egoversea-multitask-20261119/scripts/gaze_preprocessing.py',R/'scripts/gaze_preprocessing.py')
rows=json.loads((C/'download-manifest.json').read_text());assert len(rows)==60
valid=set(random.Random(42).sample(range(1,61),10));plan=[]
for r in rows:
 u=r['uuid'];e=R/'episodes'/u
 for d in ['logs','retarget','input/episode','gaze','depth','rgbd/shards','clouds','dino','tmp']:(e/d).mkdir(parents=True,exist_ok=True)
 vrs=C/'raw'/r['file'];mps=C/'raw'/('mps_'+vrs.stem+'_vrs');assert vrs.stat().st_size==r['bytes']
 (e/'input/episode'/vrs.name).symlink_to(vrs);(e/'mps').symlink_to(C/'raw')
 item=dict(uuid=u,recording_id=u,take=r['take'],task='cloth_new2',split='valid' if r['take'] in valid else 'train',vrs=str(vrs),mps=str(mps),episode=str(e),raw_vrs_sha256=r['sha256'])
 (e/'input.json').write_text(json.dumps(item,indent=2));plan.append(item)
 with (mps/'eye_gaze/general_eye_gaze.csv').open() as f: gaze=list(csv.DictReader(f))
 fields=['tracking_timestamp_us','yaw_rads_cpf','pitch_rads_cpf','origin_x_cpf','origin_y_cpf','origin_z_cpf']
 with (e/'gaze/mps-general.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for g in gaze:
   z={k:g[k] for k in fields[:3]};z.update({f'origin_{a}_cpf':(float(g[f't{a}_left_eye_cpf'])+float(g[f't{a}_right_eye_cpf']))/2 for a in 'xyz'});w.writerow(z)
for f in [R/'plan.json',R/'tensors/plan.json'] :f.write_text(json.dumps(plan,indent=2))
(R/'reports/split-intent.json').write_text(json.dumps({'seed':42,'unit':'whole recording UUID','train_takes':[r['take'] for r in plan if r['split']=='train'],'valid_takes':sorted(valid),'frozen_before_quality_results':True},indent=2))
print('PREPARED',len(plan),'valid',sorted(valid))
