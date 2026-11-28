"""Freeze all terminal Drawer dispositions and accepted tensors; no READY here."""
import json,sys,hashlib,fcntl,collections
from pathlib import Path
import numpy as np
R=Path('/coc/flash7/scratch/cxin34/cart-drawer-nola-fix-bs-20261125');D=R/'drawer';lock=(D/'reports/assemble.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb')as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
source=Path('/coc/flash7/cxin34/egoversea-multitask-release-20261121/formal-workspace/releases/r2/source.json');sm=json.load(open(source));srcids={e.get('recording_id',e.get('original_uuid',e['uuid']))for e in sm['episodes']};srcsha={e.get('raw_sha256',e.get('raw_vrs_sha256'))for e in sm['episodes']};assert None not in srcsha
records=json.load(open(D/'input-manifest.json'))['records'];byid={e['uuid']:e for e in records};segments=json.load(open(D/'segment-manifest.json'));assert len(records)==114 and len(segments)==145
assert all(e['uuid']not in srcids and e['sha256']not in srcsha for e in records)
episodes=[];disposition=[]
for i,e in enumerate(segments):
 p=Path(e['segment']);rp=p/'retarget-proof-v2.json';assert rp.exists(),(i,'retarget audit incomplete');q=json.load(open(rp));assert q['index']==i and q['uuid']==e['uuid'] and q['old_thresholds_preserved'];assert sha(p/'retarget/fix.hdf5')==q['fix_sha256']and sha(p/'retarget/fix_bs.hdf5')==q['bs_sha256']
 d=dict(index=i,uuid=e['uuid'],recording_id=e['recording_id'],split=e['split'],qc_sha256=sha(rp))
 if not q['passed']:
  assert q['quality_quarantined']and q['fails'];d.update(status='QUALITY_QUARANTINED',reasons=q['fails']);disposition.append(d);continue
 gq=p/'downstream-geometry-quarantine.json'
 if gq.exists():
  z=json.load(open(gq));assert z['status']=='GEOMETRY_QUARANTINED' and not z['passed'] and z['uuid']==e['uuid'] and z['index']==i
  assert z['retarget_proof_sha256']==sha(rp) and z['fix_sha256']==sha(p/'retarget/fix.hdf5')
  rr=Path(z['registration_report']);assert sha(rr)==z['registration_report_sha256'] and not json.load(open(rr))['passed']
  if 'failure_sha256' in z:assert sha(p/'downstream/failure.json')==z['failure_sha256'] and sha(p/'downstream/logs/cloud.log')==z['log_sha256']
  d.update(status='GEOMETRY_QUARANTINED',reasons=[z['reason']],quarantine_sha256=sha(gq));disposition.append(d);continue
 dp=p/'downstream/proof.json';assert dp.exists(),(i,'downstream incomplete');proof=json.load(open(dp));assert proof['passed'] and proof['retarget_proof_sha256']==sha(rp)
 ep=p/'downstream/episode.json';assert sha(ep)==proof['episode_sha256'];x=json.load(open(ep));parent=byid[x['recording_id']];assert x['uuid']==e['uuid']and x['split']==parent['split']and x['raw_sha256']==parent['sha256']
 for k,a in x['arrays'].items():
  assert sha(a['file'])==a['sha256'];v=np.load(a['file'],mmap_mode='r');assert list(v.shape)==a['shape'] and str(v.dtype)==a['dtype']and np.isfinite(v).all()
 t=np.load(x['arrays']['timestamp']['file']);assert t[-1]-t[0]>1 and np.allclose(np.diff(t),1/60,rtol=.05,atol=.001)
 episodes.append(x);d.update(status='ACCEPTED',rows=x['rows'],downstream_sha256=sha(dp));disposition.append(d);print('DRAWER_ASSEMBLE',i,d['status'],flush=True)
assert episodes
for k in ['recording_id','raw_sha256']:assert not {e[k]for e in episodes if e['split']=='train'} & {e[k]for e in episodes if e['split']=='valid'}
counts={s:len({e['recording_id']for e in episodes if e['split']==s})for s in ['train','valid']};assert min(counts.values())>0
m=dict(task='drawer',episodes=episodes,recording_hz=60,horizon_rows=64,dino_offset_rows=10,quality_passed=True,split_frozen=True,split_seed=42,split_unit='original_recording_uuid',validation_scope='same_collection_day',head_height_m=1.57,look_at=False,tracking_source='ondevice_combined',mps_requests=False,recipe='CPF/measured-wrist fix + BS; calibrated hd2 three clouds + DINO',bs_observations='fixed pre-BS camera observations retained, matching Cart BS recipe')
(D/'target.json').write_text(json.dumps(m,indent=2));(D/'reports/disposition.json').write_text(json.dumps(dict(input_recordings=114,candidate_segments=145,accepted_recordings=counts,segments=disposition),indent=2));(D/'reports/export-audit.json').write_text(json.dumps(dict(passed=True,training_ready=False,source_isolation=True,source_sha256=sha(source),manifest_sha256=sha(D/'target.json'),counts=counts,accepted_segments=len(episodes),rows=sum(e['rows']for e in episodes),disposition_sha256=sha(D/'reports/disposition.json'),visual_review_pending=True),indent=2));print('DRAWER_ASSEMBLE_PASS',counts,len(episodes),flush=True)
