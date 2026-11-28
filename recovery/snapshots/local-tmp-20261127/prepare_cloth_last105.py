import json,hashlib,random,os
from pathlib import Path
from collections import Counter
root=Path('/coc/flash7/czhang883/warp-data')
collection=root/'collections/cxin34_cloth_new2_165'
src=collection/'manifests/all165.json'
d=json.loads(src.read_text());rows=d['recordings'][60:]
assert len(rows)==105 and len({x['uuid'] for x in rows})==105
oldpath=Path('/coc/flash7/scratch/cxin34/cloth50-recovered-bs-20261125/ft/releases/cloth/target.json')
old=json.loads(oldpath.read_text())
valold={x.get('recording_id',x['uuid']) for x in old['episodes'] if x['split'] in ('valid','val')}
assert len(valold)==5
assert valold <= {x['uuid'] for x in rows if x['batch']=='new50'}
new=sorted([x for x in rows if x['batch']!='new50'],key=lambda x:x['take'])
valnew={x['uuid'] for x in random.Random(42).sample(new,5)}
vals=valold|valnew
rec=[dict(x,split='valid' if x['uuid'] in vals else 'train') for x in rows]
assert Counter(x['split'] for x in rec)=={'train':95,'valid':10}
for x in rec:
 p=Path(x['raw_path']);assert p.is_file() and p.stat().st_size==x['bytes']
 for rel in ['slam/closed_loop_trajectory.csv','hand_tracking/hand_tracking_results.csv','eye_gaze/general_eye_gaze.csv']:
  assert (Path(x['mps_path'])/rel).is_file()
manifest=dict(count=105,recordings=rec,parent_manifest=str(src),parent_manifest_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),selection='recordings[60:] in the supplied all165 manifest; take61..110 plus available55 of111..170',split_frozen=True,split_seed=42,split_unit='original_recording_uuid',split_counts_before_qc={'train':95,'valid':10},old_validation_source=str(oldpath),old_validation_source_sha256=hashlib.sha256(oldpath.read_bytes()).hexdigest(),new_validation_method='random.Random(42).sample(take-sorted new55, 5)',head_height_assumption_m=1.68,training_ready=False)
pipeline=collection/'provenance/last105_training_handoff_20261125';pipeline.mkdir(exist_ok=True)
assert pipeline.stat().st_uid==os.getuid() and (collection/'manifests').stat().st_uid==os.getuid()
def put(p,s):
 if p.exists():assert p.read_text()==s, f'Existing different artifact: {p}'
 else:
  with p.open('x') as f:f.write(s)
put(collection/'manifests/last105.json',json.dumps(manifest,indent=2)+'\n')
plan=dict(status='AWAITING_CHUYE_FIXED_BS_RELEASE',input_manifest=str(collection/'manifests/last105.json'),task='cloth_new2',head_height_assumption_m=1.68,preprocessing=['EgoPoser retarget fix','base smoothing','matched hd2 observations and training tensors','final FK/joint-limit/continuity QC'],split_counts_before_qc={'train':95,'valid':10},same_split_for_both=True,seed=42,augmentation='MSC strong',checkpoint_every_steps=10000,runs=[dict(name='EgoVerse180k FT',initialization='existing accepted EgoVerse180k checkpoint; resolve exact SHA before smoke',steps=60000),dict(name='Scratch',initialization='random',steps=180000)],launch_gate='verified processed release, training-only normalization, independent real-loader and smoke acceptance',submitted_jobs=[],notes='Preserve all prior data and runs. No submissions made at input handoff. Failed QC remains excluded; actual accepted counts may be smaller.')
put(pipeline/'plan.json',json.dumps(plan,indent=2)+'\n')
validtakes=sorted(x['take'] for x in rec if x['split']=='valid')
handoff=f"""# Cloth new2: 105-recording retarget + BS handoff

Please process the latest 105 recordings of Chendong's Cloth new2 task using the EgoPoser retarget fix followed by base smoothing, and prepare the matched hd2 training release. Use the same 1.68 m head-height setting as Chendong's previous Cloth new2 data (not 1.60 m).

Exact input and frozen split: `{collection}/manifests/last105.json`.
Parent: `{src}`; selection is its last105 entries, excluding the original60.

- Nov23 new50: takes61–110; Aria Gen2 serial1M0YDC5H8M0282.
  Raw: `{root}/collections/cxin34_cloth_new2_50_20261123/raw/`
- Nov25 new55: takes111–170 excluding123,132,152,157,160; Aria Gen2 serial1M0YDD5H7P0119.
  Raw: `{root}/collections/cxin34_cloth_new2_55_111_170_20261125/raw/`
- MPS outputs are under each batch's `processed/mps_v1/`; use each manifest entry's exact `mps_path` and `raw_path`. Existing SLAM/head trajectory, hand tracking and gaze files are present for all105; no new MPS request is needed. Five omitted takes are MPS-incomplete, not a claim of bad motion quality.
- Raw size:20,198,458,862 bytes (~20.2 GB). Existing per-file SHA256 values are in the manifest. This handoff rechecked presence and sizes plus MPS component presence; it did not rehash20GB or certify final motion quality.

Freeze95 train/10 validation by original recording before QC. Preserve the five original new50 validation recordings; five new55 were sampled using seed42. Validation takes: {validtakes}. Keep every segment of one recording in its original split; do not replace failed validation takes with training takes. Actual accepted counts may decrease after QC.

Please preserve original inputs/older releases, report rejected segments and reasons, and verify hand/TCP preservation, joint limits and continuity using final saved joint trajectories after smoothing. Some previous Drawer BS re-solves selected infeasible arm branches then clipped joint angles; do not relax gates to admit these. Keep point-cloud coordinate frames consistent with the released retargeted state.

Publish processed data under `collections/<collection-id>/processed/<run-id>/` and the new versioned training/Atlas release under `{root}/datasets/<dataset-id>/releases/<release-id>/`, including source UUID mappings, timestamps, provenance and quality reports. Please return the release path and acceptance report.

After data/loader checks and smoke pass, the two authorized training runs use the same split and MSC strong, seed42:
1. EgoVerse180k initialization → FT60k.
2. Train from scratch →180k.
Save checkpoints every10k. No new training has been submitted at this handoff.
"""
put(pipeline/'HANDOFF.md',handoff)
print(json.dumps({'manifest':str(collection/'manifests/last105.json'),'handoff':str(pipeline/'HANDOFF.md'),'validation_takes':validtakes,'old_validation_takes':sorted(x['take'] for x in rows if x['uuid'] in valold),'counts':dict(Counter(x['split'] for x in rec))},indent=2))
