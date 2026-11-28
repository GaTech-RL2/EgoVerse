from pathlib import Path
import json,hashlib
R=Path('/coc/flash7/cxin34/cloth-new2-scratch-20261121');release=R/'releases/cloth-new2-scratch'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
review=json.loads((R/'reports/final-visual-review.json').read_text());assert review['passed']
audit=json.loads((R/'reports/release-audit.json').read_text());assert audit['passed']
assert review['manifest_sha256']==sha(release/'target.json')==audit['manifest_sha256']
for e in review['files']:assert sha(e['file'])==e['sha256']
for f,h in json.loads((R/'reports/preprocessing-source-hashes.json').read_text()).items():assert sha(f)==h,('Preprocessing source changed',f)
assert audit['normalizer_sha256']==sha(release/'normalizer.json')
audit['visual_review_pending']=False;audit['visual_review_sha256']=sha(R/'reports/final-visual-review.json');(R/'reports/release-audit.json').write_text(json.dumps(audit,indent=2))
gate=dict(passed=True,normalizer_frozen=True,retarget_verified=True,manifests={'target':audit['manifest_sha256']},normalizer_sha256=audit['normalizer_sha256'],normalizer_fit_partition='target_train',recording_hz=60,horizon_rows=64,dino_offset_rows=10,visual_review_passed=True,visual_review_sha256=sha(R/'reports/final-visual-review.json'),release_audit_sha256=sha(R/'reports/release-audit.json'),train_recordings=audit['train_recordings'],valid_recordings=audit['valid_recordings'],train_rows=audit['train_rows'],valid_rows=audit['valid_rows'])
assert not (release/'READY.json').exists();(release/'READY.json').write_text(json.dumps(gate,indent=2))
C=Path('/coc/flash7/cxin34/warp-data/collections/cxin34_cloth_new2_60');d=C/'processed';d.mkdir(exist_ok=True)
for name,target in [('msc-scratch-20261121',R/'episodes'),('training-release',release)]:
 dest=d/name
 if not dest.exists():dest.symlink_to(target,target_is_directory=True)
print(json.dumps(gate,indent=2))
