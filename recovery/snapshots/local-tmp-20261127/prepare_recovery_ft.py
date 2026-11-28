from pathlib import Path
import json,shutil,hashlib,ast
R=Path('/coc/flash7/cxin34/cloth-valid-recovery-fix1124-20261124');O=Path('/coc/flash7/cxin34/egoa-target-retarget-fix-20261124/ft/cloth50');W=R/'ft';assert not W.exists();W.mkdir()
oldid='egoa180-cloth50-fix1124-ft60k-MSCstrong-s42';run='egoa180-cloth50-recovered-fix1124-ft60k-MSCstrong-s42'
for d in ['campaign','scripts','assets']:shutil.copytree(O/d,W/d,ignore=shutil.ignore_patterns('__pycache__'))
for d in ['configs','reports','logs','runs','cache','releases/cloth']:(W/d).mkdir(parents=True,exist_ok=True)
(W/'source').symlink_to(O/'source',target_is_directory=True)
c=json.load(open(O/'configs'/(oldid+'.json')));c.update(id=run,data_repair='CPF/measured-wrist fixed data plus quality-accepted recovered valid segments from15 recordings; no cross-gap stitching; original38/5 recording split retained')
(W/'configs'/(run+'.json')).write_text(json.dumps(c,indent=2));(W/'runs'/c['parent']).symlink_to((O/'runs'/c['parent']).resolve(),target_is_directory=True)
for f in ['source.json','normalizer.json']:shutil.copyfile(O/'releases/cloth'/f,W/'releases/cloth'/f)
p=R/'scripts/final_audit.py';s=p.read_text().replace("fraction_abs_gt10=float((abs(z)>10).mean())","q999_abs=float(np.quantile(abs(z),.999)),fraction_abs_gt10=float((abs(z)>10).mean())")
ast.parse(s);p.write_text(s)
(R/'reports/ft-plan.json').write_text(json.dumps(dict(authorized=True,root=str(W),run=run,parent=c['parent'],parent_step=180000,parent_sha256=c['parent_checkpoint_sha256'],steps=60000,seed=42,recipe='B + strongMSC',original_recording_split=[38,5],formal_gate='new release numeric + visual READY, then100-to200 resume/fullval smoke and independent gate',submission_pending=True,existing_runs_unchanged=True),indent=2));print(run)
