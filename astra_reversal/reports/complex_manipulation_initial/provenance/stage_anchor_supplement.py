"""Add only the two missing replay-anchor object trees to an owned CPU stage."""
import hashlib
import json
import os
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, '/tmp')
from stage_bench2dex import download
from stage_robocasa import archive_client, sha256

workflow=os.environ['ASTRA_RUN_ID']
if workflow != 'astra-complex-20261006-bench-stage-2':
    raise SystemExit('Unexpected task identity')
repository=json.loads(Path('/tmp/bench_anchor_asset_supplement.json').read_text())
assert repository['revision']=='bf65215f844d1fca30750e4cc70d7266650abf0a'
assert all(f['path'].startswith(('Objects/004_sugar_box/','Objects/025_mug/')) for f in repository['files'])
root=Path('/opt/astra-bench-anchor-supplement')
root.mkdir(exist_ok=False)
records=download(repository,root/'dex2bench_dataset')
archive=Path('/opt/astra-bench-anchor-supplement.tar.gz')
with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
    tar.add(root/'dex2bench_dataset',arcname='dex2bench_dataset',filter=lambda x: None if '.cache' in Path(x.name).parts else x)
client=archive_client()
prefix=f'experiments/astra-complex-20261006/{workflow}/anchor_supplement'
if client.list_objects_v2(Bucket='rldb',Prefix=prefix+'/',MaxKeys=1).get('KeyCount'):
    raise SystemExit('Supplement already exists')
client.upload_file(str(archive),'rldb',prefix+'/assets.tar.gz')
audit_files=[]
for name in ('astra-bench-anchor-inspection.json','astra-bench-usd-inspection.json'):
    p=Path('/tmp')/name
    client.upload_file(str(p),'rldb',prefix+'/'+name)
    audit_files.append({'path':name,'sha256':sha256(p),'bytes':p.stat().st_size})
receipt={'workflow':workflow,'status':'missing_anchor_objects_staged_runtime_validation_pending','gpu_count':0,'policy_rollouts':0,'repository':repository['repo_id'],'revision':repository['revision'],'files':records,'manifest_sha256':sha256(Path('/tmp/bench_anchor_asset_supplement.json')),'script_sha256':sha256(Path(__file__)),'archive':{'key':prefix+'/assets.tar.gz','bytes':archive.stat().st_size,'sha256':sha256(archive)},'audits':audit_files,'note':'CPU USD audit resolves task objects except engine MDL modules; two missing distractor roots supplied here. Full Isaac runtime and robot asset closure remain unvalidated.'}
client.put_object(Bucket='rldb',Key=prefix+'/receipt.json',Body=json.dumps(receipt,indent=2).encode(),ContentType='application/json')
print(json.dumps({'status':receipt['status'],'files':len(records),'bytes':archive.stat().st_size}),flush=True)
