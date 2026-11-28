import json,uuid,shutil,subprocess,ast,time
from pathlib import Path
R=Path('/coc/flash7/cxin34/cloth-valid-recovery-fix1124-20261124');plan=json.load(open(R/'plan.json'));assert len(plan)==18
D=json.load(open(R/'reports/quality-subspan-diagnosis.json'));added=[]
for idx in [0,1,3,14]:
 old=plan[idx];oe=Path(old['episode']);d=D[idx]
 assert (oe/'reviewed-disposition.json').exists() or not json.load(open(oe/'retarget-independent.json'))['passed']
 for aa,bb in d['strict_quality_subspans']:
  lo=old['recovery_grid_bounds'][0]+aa;hi=old['recovery_grid_bounds'][0]+bb;uid=str(uuid.uuid5(uuid.UUID(old['recording_id']),f'fix1124-recovery-quality-{lo}-{hi}'));e=R/'episodes'/uid
  for dd in ['retarget','logs','tmp/mpl','rgbd/shards','clouds','dino','gaze','input/episode']:(e/dd).mkdir(parents=True)
  for name in ['depth','mps','retarget/work']:(e/name).symlink_to((oe/name).resolve(),target_is_directory=True)
  for dd in ['gaze','input/episode']:
   for p in (oe/dd).iterdir():(e/dd/p.name).symlink_to(p.resolve())
  (e/'retarget/episode.hdf5').symlink_to((oe/'retarget/episode.hdf5').resolve());shutil.copyfile(oe/'retarget/tag_manifest.tsv',e/'retarget/tag_manifest.tsv')
  x=dict(old,uuid=uid,episode=str(e),recovery_grid_bounds=[lo,hi],index=len(plan),refines_uuid=old['uuid'],quality_refinement='Rows within5mm of parent median FK offset and no adjacent bilateral arm step>0.15rad/base step>30mm; >1s contiguous; no interpolation')
  plan.append(x);added.append(x);(e/'input.json').write_text(json.dumps(x,indent=2));proof=json.load(open(oe/'retarget-result.json'));proof['uuid']=uid;(e/'retarget-result.json').write_text(json.dumps(proof,indent=2))
(R/'reports/plan-original18.json').write_bytes((R/'plan.json').read_bytes())
for p in [R/'plan.json',R/'tensors/plan.json']:p.write_text(json.dumps(plan,indent=2))
(R/'reports/quality-refinement-plan.json').write_text(json.dumps(dict(added=added,reason='Further recover reliable continuous subspans only from quality-failed candidates; original18 attempts retained; accepted segments unchanged'),indent=2))
script=R/'scripts/refine-audit.sbatch';script.write_text('#!/bin/bash\nset -euo pipefail\nexport PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1\nexec /coc/flash7/cxin34/mobilemani-pretraining/env-train/bin/python '+str(R/'scripts/retarget_audit.py')+' "'+','.join(str(x['index'])for x in added)+'"\n')
cmd=['/opt/slurm/Ubuntu-24.04/current/bin/sbatch','--parsable','--partition=interactive-cpu','--account=rl2-lab','--qos=debug','--exclude=hal,bmo','--cpus-per-task=2','--mem=8G','--time=00:30:00','--job-name=cloth-recovery-refine-audit','--output='+str(R/'logs/refine-audit-%j.log'),str(script)]
intent=R/'reports/refine-audit.intent';assert not intent.exists();intent.write_text(json.dumps(dict(command=cmd,time=time.time())));z=subprocess.run(cmd,text=True,capture_output=True);(R/'reports/refine-audit-submit.json').write_text(json.dumps(dict(command=cmd,returncode=z.returncode,stdout=z.stdout,stderr=z.stderr),indent=2));assert z.returncode==0,z.stderr;job=z.stdout.strip().split(';')[0]
base=json.load(open(R/'reports/segments.json'))['command'];cmd=[]
for c in base:
 if c.startswith('--array='):c='--array=18-21%1'
 elif c.startswith('--dependency='):c='--dependency=afterok:'+job+',afterany:4855089'
 c=c.replace('--job-name=cloth-recovery-segments','--job-name=cloth-recovery-refined').replace('/logs/segments-','/logs/refined-');cmd.append(c)
intent=R/'reports/refined.intent';assert not intent.exists();intent.write_text(json.dumps(dict(command=cmd,time=time.time())));z=subprocess.run(cmd,text=True,capture_output=True);(R/'reports/refined-submit.json').write_text(json.dumps(dict(command=cmd,returncode=z.returncode,stdout=z.stdout,stderr=z.stderr),indent=2));assert z.returncode==0,z.stderr;ref=z.stdout.strip().split(';')[0]
for dest in ['4855137','4855324']:
 z=subprocess.run(['/opt/slurm/Ubuntu-24.04/current/bin/scontrol','update','JobId='+dest,'Dependency=afterany:4855089:'+ref],text=True,capture_output=True);assert z.returncode==0,z.stderr
print(json.dumps(dict(audit=job,refined=ref,new_indices=[x['index']for x in added])))
