"""Bind the actual sampled visual review to the final matched release.

This is not an automated claim to have inspected all contact sheets. The ON
source RGB/retarget review and the explicitly reviewed OFF pilot are reused;
every final episode still passes independent array/geometry/loader checks.
"""
from pathlib import Path
import json,hashlib
R=Path('/coc/flash7/scratch/cxin34/cart-drawer-nola-fix-bs-20261125');O=R/'cart-novfilter'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
v=json.load(open(O/'reports/pilot-visual-review.json'));assert v['passed']
for p,h in v['images'].items():assert sha(p)==h
old=json.load(open(R/'cart/target.json'));new=json.load(open(O/'target.json'));base=json.load(open(R/'cart/reports/visual-review.json'));assert base['passed'] and base['manifest_sha256']==sha(R/'cart/target.json')
oldmap={e['uuid']:e for e in old['episodes']};assert len(oldmap)==len(new['episodes'])==107
changed={'front_pcd_1','front_pcd_2','front_pcd_3','gaze_point_glass'}
for e in new['episodes']:
 b=oldmap[e['uuid']]
 for key in ['split','rows','recording_id','raw_sha256','retarget_hdf5','retarget_demo']:assert e[key]==b[key]
 assert e['virtual_camera_filter'] is False
 for key,x in e['arrays'].items():
  if key not in changed:assert x==b['arrays'][key]
a=json.load(open(O/'reports/export-audit.json'));assert a['passed'] and a['manifest_sha256']==sha(O/'target.json') and a['source_isolation']
result=dict(passed=True,manifest_sha256=sha(O/'target.json'),images=v['images'],scope='sampled OFF pilot episode0 middle frame three clouds; source RGB/retarget review inherited after exact unchanged-label and split checks; other OFF contact sheets generated but not manually reviewed',pilot_review_sha256=sha(O/'reports/pilot-visual-review.json'),source_review_sha256=sha(R/'cart/reports/visual-review.json'),all_episodes_numeric_export_sha256=sha(O/'reports/export-audit.json'))
(O/'reports/visual-review.json').write_text(json.dumps(result,indent=2))
