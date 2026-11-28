from pathlib import Path
import json,hashlib,sys
W=Path('/coc/flash7/scratch/cxin34/warp-data');N='cxin34_drawer_1125_qc';P=W/'pipelines/drawer_atlas_20261125'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
v=sys.argv[1];d=W/'datasets'/N/'releases'/v;a=json.loads((P/(v+'-audit.json')).read_text());assert a['passed'] and not a['training_authorized']
for e in a['episodes']:
 assert sha(d/e['parquet'])==e['sha256'];assert sha(e['hdf5'])==e['hdf5_sha256']
links={}
for kind in ['collections','datasets']:
 src=W/kind/N;dst=Path('/coc/flash7/czhang883/warp-data')/kind/N
 if dst.is_symlink():assert dst.resolve()==src.resolve()
 else:
  assert not dst.exists();dst.symlink_to(src,target_is_directory=True)
 links[str(dst)]=str(src)
(P/(v+'-publication.json')).write_text(json.dumps(dict(passed=True,episodes=len(a['episodes']),frames=a['frames'],links=links,visualization_only=True,training_authorized=False),indent=2))
print(v,len(a['episodes']),a['frames'])
