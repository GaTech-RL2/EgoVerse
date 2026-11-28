from pathlib import Path
import ast
R=Path('/coc/flash7/cxin34/cloth-valid-recovery-fix1124-20261124')
s='''import pathlib,json,hashlib,subprocess,time,fcntl,shutil
P=pathlib.Path(__file__).resolve().parents[1];W=P/'ft';run=json.load(open(P/'reports/ft-plan.json'))['run']
lock=(P/'reports/publish-train.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb')as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
a=json.load(open(P/'reports/final-numeric-audit.json'));v=json.load(open(P/'reports/visual-review.json'));assert a['passed']and v['passed']and not a['unresolved']
assert sha(P/'release/target.json')==a['manifest_sha256'];assert set(v['reviewed_accepted_uuids'])=={x['uuid']for x in json.load(open(P/'plan.json'))if(P/'reports/accepted'/(x['uuid']+'.json')).exists()}
for f,h in v['files'].items():assert sha(f)==h
for key,parts in a['normalization'].items():
 for split,st in parts.items():assert st['max_abs']<100 and st['q999_abs']<10 and st['fraction_abs_gt10']<.001,(key,split,st)
rel=W/'releases/cloth';target=rel/'target.json'
if target.exists():assert sha(target)==a['manifest_sha256']
else:shutil.copyfile(P/'release/target.json',target)
cfgp=W/'configs'/(run+'.json');cfg=json.load(open(cfgp));meta=json.load(open(target));norm=json.load(open(rel/'normalizer.json'))
tr={e['recording_id']for e in meta['episodes']if e['split']=='train'};va={e['recording_id']for e in meta['episodes']if e['split']=='valid'};assert len(tr)==38 and len(va)==5 and not tr&va
source=json.load(open(rel/'source.json'));assert not {e['recording_id']for e in source['episodes']}&(tr|va)
assert not {e.get('raw_vrs_sha256',e.get('raw_sha256'))for e in source['episodes']}&{e['raw_vrs_sha256']for e in meta['episodes']}
assert sha(rel/'normalizer.json')==cfg['source_identity']['normalizer_sha256']==a['normalizer_sha256']
parent=W/'runs'/cfg['parent']/('step_%06d.pt'%cfg['parent_step']);assert parent.is_file() and sha(parent)==cfg['parent_checkpoint_sha256'];proof=json.load(open(parent.parent/'proof.json'));assert proof['passed']and proof['step']==180000 and proof['full_validation']
assert shutil.disk_usage(W).free>15*1024**3,'Output storage below15GiB; no submission'
ready=dict(passed=True,normalizer_frozen=True,retarget_verified=True,clean_view_audited=True,fk_reference_verified=True,recording_hz=60,horizon_rows=64,dino_offset_rows=10,manifests=dict(target=sha(target),source=sha(rel/'source.json')),normalizer_sha256=sha(rel/'normalizer.json'),normalizer_fit_partition='source_train',visual_review_passed=True,visual_review_sha256=sha(P/'reports/visual-review.json'),data_audit_sha256=sha(P/'reports/final-numeric-audit.json'),train_recordings=len(tr),valid_recordings=len(va),train_rows=a['loader'][0]['rows'],valid_rows=a['loader'][1]['rows'],exclusions=a['excluded'],retarget_fix='CPF/measured-wrist plus recovered valid intervals',cloud_registration='B_knuckle_to_tcp',visual_scope='Every newly accepted segment RGB andthreeclouds first/middle/last; all rows numeric/stricttracking/FK/hash/realH64loader checked')
for path in [rel/'READY.json',P/'release/READY.json']:
 if path.exists():assert json.load(open(path))==ready
 else:
  with path.open('x')as f:json.dump(ready,f,indent=2)
receipt=W/'reports/submit.json';intent=W/'reports/submit.intent.json'
if receipt.exists():
 z=json.load(open(receipt));assert z['returncode']==0;print('ALREADY_SUBMITTED',z['stdout']);raise SystemExit
assert not intent.exists(),'Unresolved submission intent'
O='/coc/flash7/cxin34/egoa-target-retarget-fix-20261124/ft/cloth50';cmd=json.load(open(O+'/reports/submit.json'))['command'];cmd=[c.replace(O,str(W)).replace('egoa180-cloth50-fix1124-ft60k-MSCstrong-s42',run)for c in cmd]
with intent.open('x')as f:json.dump(dict(command=cmd,time=time.time()),f,indent=2)
z=subprocess.run(cmd,capture_output=True,text=True);d=dict(command=cmd,returncode=z.returncode,stdout=z.stdout,stderr=z.stderr,time=time.time());receipt.write_text(json.dumps(d,indent=2));assert z.returncode==0 and z.stdout.strip().split(';')[0].isdigit(),d
print('NEW_SMOKE_GATED_FT60K_SUBMITTED',z.stdout)
'''
(R/'scripts/publish_train.py').write_text(s);ast.parse(s);print('Publication and unique60k submission guard prepared; NOT executed')
