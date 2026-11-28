import concurrent.futures as cf
import hashlib
import json
import os
from pathlib import Path
import shutil

OLD = Path('/coc/flash7/cxin34/warp-data/collections')
ROOT = Path('/coc/flash7/czhang883/warp-data')
OUT = ROOT / 'collections'
CONTROL = OUT / 'cxin34_cloth_new2_165/provenance/migration_20261125'
CONTROL.mkdir(parents=True, exist_ok=True)
ROWS = json.loads((OLD / 'cxin34_cloth_new2_170/manifests/all170.json').read_text())['recordings']
MISSING = {123, 132, 152, 157, 160}
NAMES = ['cxin34_cloth_new2_60', 'cxin34_cloth_new2_50_20261123', 'cxin34_cloth_new2_55_111_170_20261125']

def write(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2) + '\n')

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def copy_verified(src, dst, expected=None):
    digest = sha(src)
    if expected:
        assert digest == expected, ('source hash mismatch', str(src))
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        tmp = dst.with_name(dst.name + '.migration-partial')
        shutil.copyfile(src, tmp)
        assert sha(tmp) == digest, str(tmp)
        os.replace(tmp, dst)
    else:
        assert sha(dst) == digest, ('destination conflict', str(dst))
    return dict(source=str(src), destination=str(dst), bytes=src.stat().st_size, sha256=digest)

def process(row):
    r = dict(row)
    take = int(r['take'])
    name = NAMES[0 if take <= 60 else 1 if take <= 110 else 2]
    src = Path(r['raw_path'])
    dst = OUT / name / 'raw' / src.name
    proof = [copy_verified(src, dst, r['sha256'])]
    msrc = Path(r['mps_path'])
    mdst = OUT / name / 'processed/mps_v1' / msrc.name
    for feature in ['slam', 'hand_tracking', 'eye_gaze']:
        assert (msrc / feature / 'summary.json').is_file(), (take, feature)
    assert (msrc / 'hand_tracking/hand_tracking_results.csv').stat().st_size > 0
    for f in sorted(msrc.rglob('*')):
        if f.is_file():
            proof.append(copy_verified(f, mdst / f.relative_to(msrc)))
    compat = dst.parent / msrc.name
    if not compat.exists():
        compat.symlink_to(mdst)
    r.update(raw_path=str(dst), mps_path=str(mdst), mps_status='COMPLETE', canonical_collection=name)
    print('verified take', take, flush=True)
    return r, proof

selected = [r for r in ROWS if int(r['take']) not in MISSING]
assert len(selected) == 165 and len({r['uuid'] for r in selected}) == 165
results = []
proofs = []
with cf.ThreadPoolExecutor(max_workers=4) as pool:
    for row, proof in pool.map(process, selected):
        results.append(row)
        proofs.extend(proof)
results.sort(key=lambda r: int(r['take']))
union = OUT / 'cxin34_cloth_new2_165'
for name, count in zip(NAMES, [60, 50, 55]):
    rows = [r for r in results if r['canonical_collection'] == name]
    assert len(rows) == count
    write(OUT / name / 'manifests/recordings.json', dict(count=count, recordings=rows, training_ready=False))
    write(OUT / name / 'collection.json', dict(count=count, raw_mps_verified=True, training_ready=False))
    for r in rows:
        for key in ['raw_path', 'mps_path']:
            target = Path(r[key])
            link = union / 'raw' / target.name
            link.parent.mkdir(parents=True, exist_ok=True)
            if not link.exists():
                link.symlink_to(target)
            assert link.resolve() == target.resolve()
    write(union / ('manifests/' + {60:'old60',50:'new50',55:'new55'}[count] + '.json'), dict(count=count, recordings=rows))
write(union / 'manifests/all165.json', dict(count=165, recordings=results, training_ready=False))
write(union / 'manifests/excluded5.json', dict(reason='MPS incomplete; excluded from canonical input, originals retained', takes=sorted(MISSING), recordings=[r for r in ROWS if int(r['take']) in MISSING]))
write(union / 'collection.json', dict(count=165, batches=dict(old60=60,new50=50,new55=55), raw_mps_verified=True, training_ready=False, excluded_takes=sorted(MISSING)))
report = dict(passed=True, recordings=165, files=len(proofs), bytes=sum(p['bytes'] for p in proofs), excluded_takes=sorted(MISSING), old_sources_retained=True, sha256_verified=True, files_manifest='file-integrity.json')
write(CONTROL / 'file-integrity.json', proofs)
write(CONTROL / 'verification.json', report)
shutil.copyfile(__file__, CONTROL / 'migrate.py')
(union / 'README.md').write_text('# Cloth new2 canonical raw + MPS input\n\n165 recordings: old60 + new50 + new55. Use manifests/all165.json.\nMPS folders are in each batch processed/mps_v1; raw/ has compatibility links.\nExcluded takes: 123,132,152,157,160. Original files retained at old source.\nRaw/MPS completeness is verified; retarget/features/splits are NOT a new training release.\nExisting downstream training campaigns remain unchanged.\nProof: ' + str(CONTROL / 'verification.json') + '\n')
print(json.dumps(report), flush=True)
