from pathlib import Path
import json,hashlib,subprocess,math
R=Path('/coc/flash7/cxin34/cloth-new2-scratch-20261121');jobs=json.loads((R/'reports/jobs.json').read_text());rid='cloth-new2-B-msc-scratch-s42';run=R/'runs'/rid
cfg=json.loads((R/'configs'/f'{rid}.json').read_text());assert cfg['stage']=='scratch' and cfg['steps']==cfg['scheduler_steps']==180000 and not any(k in cfg for k in ['parent','parent_checkpoint_sha256'])
smoke=json.loads((R/'reports'/f'{rid}-independent-smoke-audit.json').read_text());assert smoke['passed']
queue=subprocess.check_output(['/opt/slurm/Ubuntu-24.04/current/bin/squeue','-j',jobs['formal'],'-h','-o','%i|%T|%N|%b'],text=True).strip();assert '|RUNNING|' in queue,queue
metrics=[json.loads(l) for l in (run/'metrics.jsonl').read_text().splitlines()];m=metrics[-1];assert m['step']>=100 and all(math.isfinite(v) for v in m.values() if isinstance(v,(int,float)))
assert not (run/'transfer-audit.json').exists()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
report={'passed':True,'formal_job':jobs['formal'],'queue':queue,'metrics':m,'configuration_sha256':sha(R/'configs'/f'{rid}.json'),'data_ready_sha256':sha(R/'releases/cloth-new2-scratch/READY.json'),'smoke_audit_sha256':sha(R/'reports'/f'{rid}-independent-smoke-audit.json'),'initialization':'scratch; no parent checkpoint','steps':180000,'checkpoint_every_steps':10000,'model':'three PCD + DINO B objective with MSC augmentation only','run_dir':str(run),'training_complete':False}
(R/'reports/startup-verified.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
