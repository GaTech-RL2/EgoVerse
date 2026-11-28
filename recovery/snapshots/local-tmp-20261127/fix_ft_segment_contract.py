from pathlib import Path
import ast,shutil,json,hashlib,copy
R=Path('/coc/flash7/cxin34/cloth-valid-recovery-fix1124-20261124');W=R/'ft';O=Path('/coc/flash7/cxin34/egoa-target-retarget-fix-20261124/ft/cloth50')
shutil.copyfile(O/'configs/hd2-reference.yaml',W/'configs/hd2-reference.yaml')
for name in ['contracts.py','train.py']:
 p=W/'campaign'/name;s=p.read_text();(W/'reports'/(name+'.before-segment-contract')).write_text(s)
 if name=='contracts.py':s=s.replace("e['uuid'] for e in manifest['episodes']","e.get('recording_id',e['uuid']) for e in manifest['episodes']")
 else:s=s.replace("x['uuid'] for x in train.meta['episodes']","x.get('recording_id',x['uuid']) for x in train.meta['episodes']")
 ast.parse(s);p.write_text(s)
ns={};exec((W/'campaign/contracts.py').read_text(),ns);cfg=json.load(open(next((W/'configs').glob('*.json'))));norm=json.load(open(W/'releases/cloth/normalizer.json'));m=json.load(open(O/'releases/cloth/target.json'))
ns['check_normalizer'](cfg,norm,m)
q=copy.deepcopy(m);extra=copy.deepcopy(next(e for e in q['episodes']if e['split']=='train'));extra['uuid']='fixture-extra-segment';q['episodes'].append(extra);ns['check_normalizer'](cfg,norm,q)
extra['split']='valid'
try:ns['check_normalizer'](cfg,norm,q)
except AssertionError:blocked=True
else:blocked=False
assert blocked
(W/'reports/segment-contract-audit.json').write_text(json.dumps(dict(passed=True,original_split_passed=True,same_recording_extra_train_segment_passed=True,same_recording_cross_split_rejected=True,changes='Only new workspace accounting: recording_id group counts instead of episode UUID counts; policy/loss unchanged',reference_yaml_sha256=hashlib.sha256((W/'configs/hd2-reference.yaml').read_bytes()).hexdigest()),indent=2));print('Segment grouping contract passed; cross-split leakage rejected')
